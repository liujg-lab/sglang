"""Keep accepted OPT draft KV on CUDA page size 1.

A forwarded node was scored at position ``prefix + depth`` while attending
only to its ancestors. That KV matches a later causal extend of the same
token. The next step rewrites ``req_to_token`` for the accepted nodes and
frees every other tree slot. Page size 128 is not handled here: those slots
share pages, so NPU still drops the tree.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

_APPEND = ("append_one", "append_n")


@dataclass
class OptBonusGuess:
    """One guessed bonus token written after the tree, outside the lease.

    ``parent == -1`` is the committed tail. Any other parent is an index into
    the verify window. ``logits`` is the distribution after this token.
    """

    parent: int
    token: int
    slot: int
    logits: torch.Tensor


@dataclass
class OptTreeKVLease:
    """Uncommitted tree slots owned outside ``kv_allocated_len``.

    ``layer_slots[row][col]`` is the pool index written when that layer was
    forwarded. ``depths`` are 0-based rows. A parent of -1 is the committed
    tail. ``released`` makes a second free a no-op.
    """

    keep: int
    layer_slots: Tuple[Tuple[int, ...], ...]
    tokens: Tuple[int, ...]
    parents: Tuple[int, ...]
    cols: Tuple[int, ...]
    depths: Tuple[int, ...]
    released: bool = False

    def all_slot_ids(self) -> List[int]:
        ids: List[int] = []
        for layer in self.layer_slots:
            ids.extend(int(slot) for slot in layer)
        return ids


def opt_page_size(scheduler) -> int:
    tree = getattr(scheduler, "tree_cache", None)
    if tree is None:
        return 1
    return int(getattr(tree, "page_size", 1) or 1)


def opt_kv_release_allowed(scheduler) -> bool:
    """False when the device context must not return pages to the allocator."""
    if getattr(scheduler, "_sr_device_poisoned", False):
        return False
    try:
        from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
            has_unresolved_resources,
        )
    except Exception:
        return False
    try:
        unresolved = has_unresolved_resources(
            scheduler, include_quarantine=False
        )
    except Exception:
        logger.warning("[OPT] lease release check failed", exc_info=True)
        return False
    return not unresolved


def opt_finish_req(scheduler, rid, state):
    """Request object for a FINISH/ABORT that has not looked the state up yet."""
    if state is None:
        manager = getattr(scheduler, "sr_state", None)
        getter = getattr(manager, "get", None)
        if callable(getter):
            state = getter(rid)
    if state is None:
        return None
    return getattr(state, "req_object", None)


def detach_opt_tree_lease(req) -> Optional[OptTreeKVLease]:
    """Pop the lease without freeing its slots."""
    lease = getattr(req, "opt_tree_lease", None)
    if lease is None:
        return None
    req.opt_tree_lease = None
    if not isinstance(lease, OptTreeKVLease):
        return None
    return lease


def node_depths(parents: Sequence[int]) -> Optional[List[int]]:
    """Hop count from the committed tail. ``None`` when a parent is illegal."""
    depths: List[int] = []
    for index, parent in enumerate(parents):
        parent = int(parent)
        if parent == -1:
            depths.append(0)
            continue
        if parent < 0 or parent >= index:
            return None
        depths.append(depths[parent] + 1)
    return depths


def release_opt_tree_lease(scheduler, req) -> None:
    """Free every slot still owned by the lease. A second call does nothing."""
    lease = detach_opt_tree_lease(req)
    if lease is None or lease.released:
        return
    lease.released = True
    if opt_page_size(scheduler) != 1:
        logger.error(
            "[OPT] draft %s lease is not freed because page_size is not 1",
            getattr(req, "rid", None),
        )
        return
    _free_slot_ids(scheduler, lease.all_slot_ids())


def consume_opt_tree_lease(scheduler, req) -> None:
    """Install accepted slots, or free the lease when this step cannot reuse it.

    Reuse requires page size 1, an ``append_*`` alignment, and the same
    prefix the tree was built on. Any other alignment frees the whole lease.
    """
    lease = getattr(req, "opt_tree_lease", None)
    if not isinstance(lease, OptTreeKVLease) or lease.released:
        return
    if opt_page_size(scheduler) != 1 or not _alignment_matches(req, lease):
        release_opt_tree_lease(scheduler, req)
        release_opt_bonus_guesses(scheduler, req)
        return
    kept = _accepted_slots(req, lease)
    if kept is None:
        logger.warning(
            "[OPT] draft %s dropped tree lease at prefix %s",
            getattr(req, "rid", None),
            lease.keep,
        )
        release_opt_tree_lease(scheduler, req)
        release_opt_bonus_guesses(scheduler, req)
        return
    if not _install_accepted(scheduler, req, lease, kept):
        release_opt_tree_lease(scheduler, req)
        release_opt_bonus_guesses(scheduler, req)
        return
    accept_opt_bonus_guess(scheduler, req, lease, kept)


def publish_opt_tree_lease(scheduler, req, runner, keep: int, window) -> bool:
    """Move this round's tree slots onto ``req.opt_tree_lease``.

    Returns False when the caller should roll the tree back instead. Page
    size other than 1 always returns False.
    """
    if opt_page_size(scheduler) != 1:
        return False
    if getattr(scheduler, "_sr_device_poisoned", False):
        return False
    slots = getattr(runner, "layer_slots", None) if runner is not None else None
    if not slots:
        return False
    parsed = _window_fields(window)
    if parsed is None:
        return False
    tokens, parents, cols = parsed
    depths = node_depths(parents)
    if depths is None or len(depths) != len(tokens):
        return False
    layers = tuple(_layer_ids(layer) for layer in slots)
    if not layers or any(len(layer) == 0 for layer in layers):
        return False
    span = sum(len(layer) for layer in layers)
    keep = int(keep)
    if int(getattr(req, "kv_committed_len", 0) or 0) != keep:
        return False
    if int(getattr(req, "kv_allocated_len", 0) or 0) != keep + span:
        return False
    row = getattr(req, "req_pool_idx", None)
    pool = getattr(getattr(scheduler, "req_to_token_pool", None), "req_to_token", None)
    if row is None or not torch.is_tensor(pool):
        return False
    if int(row) < 0 or int(row) >= int(pool.shape[0]):
        return False
    end = keep + span
    if end > int(pool.shape[1]):
        return False
    lease = OptTreeKVLease(
        keep=keep,
        layer_slots=layers,
        tokens=tuple(int(token) for token in tokens),
        parents=tuple(int(parent) for parent in parents),
        cols=tuple(int(col) for col in cols),
        depths=tuple(depths),
    )
    saved_allocated = req.kv_allocated_len
    saved_row = pool[int(row), keep:end].clone()
    try:
        pool[int(row), keep:end] = -1
        req.kv_allocated_len = keep
        req.opt_tree_lease = lease
    except Exception:
        pool[int(row), keep:end] = saved_row
        req.kv_allocated_len = saved_allocated
        req.opt_tree_lease = None
        raise
    return True


def _window_fields(window):
    if window is None or len(window) != 3:
        return None
    tokens, parents, cols = window
    if tokens is None or parents is None or cols is None:
        return None
    tokens = list(tokens)
    parents = list(parents)
    cols = list(cols)
    if not tokens or not (len(tokens) == len(parents) == len(cols)):
        return None
    return tokens, parents, cols


def _layer_ids(layer) -> Tuple[int, ...]:
    if torch.is_tensor(layer):
        values = layer.detach().to(device="cpu").reshape(-1).tolist()
    else:
        values = list(layer)
    return tuple(int(value) for value in values)


def _alignment_matches(req, lease: OptTreeKVLease) -> bool:
    align = getattr(req, "sr_align_result", None)
    kind = getattr(align, "kind", None)
    old_len = getattr(align, "old_local_len", None)
    if kind not in _APPEND or old_len is None:
        return False
    return int(old_len) == int(lease.keep)


def _logical_len(req) -> int:
    origin = getattr(req, "origin_input_ids", None) or []
    output = getattr(req, "output_ids", None) or []
    return len(origin) + len(output)


def _logical_token(req, index: int) -> Optional[int]:
    origin = getattr(req, "origin_input_ids", None) or []
    output = getattr(req, "output_ids", None) or []
    if index < 0:
        return None
    if index < len(origin):
        return int(origin[index])
    output_index = index - len(origin)
    if output_index >= len(output):
        return None
    return int(output[output_index])


def _accepted_slots(req, lease: OptTreeKVLease) -> Optional[List[int]]:
    """Slots along the accepted path. ``None`` abandons the whole lease."""
    if _logical_len(req) < int(lease.keep):
        return None
    if not (
        len(lease.tokens)
        == len(lease.parents)
        == len(lease.cols)
        == len(lease.depths)
    ):
        return None
    kept: List[int] = []
    parent = -1
    index = int(lease.keep)
    while index < _logical_len(req):
        token = _logical_token(req, index)
        if token is None:
            return None
        child = _child_index(lease, parent, token)
        if child is None:
            break
        depth = int(lease.depths[child])
        if depth != len(kept):
            return None
        if depth >= len(lease.layer_slots):
            break
        col = int(lease.cols[child])
        layer = lease.layer_slots[depth]
        if col < 0 or col >= len(layer):
            return None
        kept.append(int(layer[col]))
        parent = child
        index += 1
    return kept


def _child_index(lease: OptTreeKVLease, parent: int, token: int) -> Optional[int]:
    for index, (node_parent, node_token) in enumerate(
        zip(lease.parents, lease.tokens)
    ):
        if int(node_parent) == int(parent) and int(node_token) == int(token):
            return index
    return None


def _install_accepted(scheduler, req, lease: OptTreeKVLease, kept: Sequence[int]) -> bool:
    row = getattr(req, "req_pool_idx", None)
    pool = getattr(getattr(scheduler, "req_to_token_pool", None), "req_to_token", None)
    keep = int(lease.keep)
    if row is None or not torch.is_tensor(pool):
        return False
    end = keep + len(kept)
    if int(row) < 0 or int(row) >= int(pool.shape[0]) or end > int(pool.shape[1]):
        return False
    kept_set = {int(slot) for slot in kept}
    complement = [slot for slot in lease.all_slot_ids() if slot not in kept_set]
    if complement and getattr(scheduler, "token_to_kv_pool_allocator", None) is None:
        return False
    req.opt_tree_lease = None
    lease.released = True
    for offset, slot in enumerate(kept):
        pool[int(row), keep + offset] = int(slot)
    req.kv_committed_len = end
    req.kv_allocated_len = end
    _free_slot_ids(scheduler, complement)
    return True


def _free_slot_ids(scheduler, slot_ids: Sequence[int]) -> None:
    unique: List[int] = []
    seen = set()
    for slot in slot_ids:
        slot = int(slot)
        if slot < 0 or slot in seen:
            continue
        seen.add(slot)
        unique.append(slot)
    if not unique:
        return
    allocator = getattr(scheduler, "token_to_kv_pool_allocator", None)
    if allocator is None or not hasattr(allocator, "free"):
        raise RuntimeError("OPT tree lease has slots but no KV allocator")
    device = _free_device(scheduler)
    allocator.free(torch.tensor(unique, dtype=torch.int64, device=device))


def _free_device(scheduler):
    pool = getattr(scheduler, "req_to_token_pool", None)
    mapping = getattr(pool, "req_to_token", None)
    if torch.is_tensor(mapping):
        return mapping.device
    allocator = getattr(scheduler, "token_to_kv_pool_allocator", None)
    pages = getattr(allocator, "free_pages", None)
    if torch.is_tensor(pages):
        return pages.device
    return "cpu"


def release_opt_bonus_guesses(scheduler, req, guesses=None) -> None:
    """Free every guessed bonus slot. A second call does nothing."""
    if guesses is None:
        guesses = getattr(req, "opt_bonus_guesses", None)
        req.opt_bonus_guesses = None
    if (
        not guesses
        or opt_page_size(scheduler) != 1
        or getattr(scheduler, "_sr_device_poisoned", False)
    ):
        return
    _free_slot_ids(scheduler, [int(guess.slot) for guess in guesses])


def _accepted_end(req, lease: OptTreeKVLease, n_kept: int) -> int:
    """Verify index of the last installed node. ``-1`` is the committed tail."""
    parent = -1
    index = int(lease.keep)
    for _ in range(int(n_kept)):
        token = _logical_token(req, index)
        child = None if token is None else _child_index(lease, parent, token)
        if child is None:
            return -1
        parent = child
        index += 1
    return parent


def _install_bonus_slot(scheduler, req, index: int, slot: int) -> bool:
    row = getattr(req, "req_pool_idx", None)
    pool = getattr(getattr(scheduler, "req_to_token_pool", None), "req_to_token", None)
    if row is None or not torch.is_tensor(pool):
        return False
    index = int(index)
    if int(row) < 0 or int(row) >= int(pool.shape[0]):
        return False
    if index < 0 or index >= int(pool.shape[1]):
        return False
    pool[int(row), index] = int(slot)
    end = index + 1
    req.kv_committed_len = end
    req.kv_allocated_len = max(int(getattr(req, "kv_allocated_len", 0) or 0), end)
    return True


def accept_opt_bonus_guess(scheduler, req, lease: OptTreeKVLease, kept: Sequence[int]) -> None:
    """Install the guess at the rejected node when it is the bonus token.

    Every other guess is freed. The stored row becomes the next seed only
    when this bonus is the new committed tail.
    """
    guesses = list(getattr(req, "opt_bonus_guesses", None) or [])
    req.opt_bonus_guesses = None
    if not guesses:
        return
    bonus_at = int(lease.keep) + len(kept)
    bonus = _logical_token(req, bonus_at)
    v_end = _accepted_end(req, lease, len(kept))
    hit = None
    if bonus is not None:
        for guess in guesses:
            if int(guess.parent) == int(v_end) and int(guess.token) == int(bonus):
                hit = guess
                break
    installed = False
    if hit is not None:
        installed = _install_bonus_slot(scheduler, req, bonus_at, int(hit.slot))
    if installed and bonus_at + 1 == _logical_len(req):
        req.opt_prefill_tail_logits = hit.logits.detach()
    release_opt_bonus_guesses(
        scheduler,
        req,
        [guess for guess in guesses if not installed or guess is not hit],
    )
