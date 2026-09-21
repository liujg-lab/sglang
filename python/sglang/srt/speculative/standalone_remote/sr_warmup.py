"""Shared STANDALONE_REMOTE extra-graph warmup helpers.

CPU-importable. Plans batch sizes, prefix remainders, and allocator ownership.
Does not import Triton or torch_npu.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Optional, Sequence

import torch

from sglang.srt.speculative.standalone_remote.drafter.sr_tree_paged_layout import (
    pages_per_branch,
)
from sglang.srt.speculative.standalone_remote.sr_align import SRWarmupFatalError


def sr_warmup_raw_batch_sizes(
    configured_capture_bs,
    req_to_token_rows,
    max_running_requests=None,
    standalone_remote_max_batch_size=None,
) -> tuple[tuple[int, ...], Optional[str]]:
    """Return ``(1..upper)`` or ``((), skip_reason)``.

    ``cuda_graph_max_bs`` is not an upper bound by itself.
    Optional capacity arguments that are ``None`` are ignored.
    """
    configured = [int(b) for b in (configured_capture_bs or []) if int(b) > 0]
    if not configured:
        return (), "no configured capture batch sizes"
    rows = int(req_to_token_rows or 0)
    if rows <= 0:
        return (), "req_to_token has no rows"
    caps = [max(configured), rows]
    for optional in (max_running_requests, standalone_remote_max_batch_size):
        if optional is not None:
            caps.append(int(optional))
    upper = min(caps)
    if upper <= 0:
        return (), "warmup upper batch size is not positive"
    return tuple(range(1, upper + 1)), None


def warmup_prefix_remainders(page_size: int, steps: int) -> list[int]:
    """Representative remainders. ``steps > page`` includes jump sides."""
    page = max(int(page_size), 1)
    steps = max(int(steps), 1)
    if steps <= page:
        return sorted({0, 1, max(page - steps + 1, 1) % page, page - 1})
    m = steps % page
    remainders = {0, 1, page - 1}
    if m == 0:
        remainders.update({0, 1})
    else:
        before = page - m
        remainders.add(before)
        if before + 1 < page:
            remainders.add(before + 1)
    return sorted(remainders)


def warmup_nnp_jump_reachable(page_size: int, steps: int) -> bool:
    """False when every legal remainder has the same branch page count."""
    nnps = {
        pages_per_branch(rem, steps, page_size)
        for rem in warmup_prefix_remainders(page_size, steps)
    }
    return len(nnps) > 1


def warmup_mapping_end(seq_len: int, page_size: int, topk: int, num_steps: int) -> int:
    page = max(int(page_size), 1)
    k = max(int(topk), 1)
    steps = max(int(num_steps), 0)
    length = max(int(seq_len), 0)
    rem = length % page
    branch_pages = (rem + steps + page - 1) // page
    return length - rem + k * branch_pages * page


def warmup_mapping_fits(seq_lens, page_size: int, topk: int, num_steps: int, pool_len: int) -> bool:
    pool = int(pool_len)
    return all(
        warmup_mapping_end(int(length), page_size, topk, num_steps) <= pool
        for length in seq_lens
    )


def warmup_prefix_candidates(page_size: int, steps: int, pool_cols: int, topk: int) -> list[int]:
    page = max(int(page_size), 1)
    prefixes = []
    for rem in warmup_prefix_remainders(page, steps):
        prefix = page + rem
        if prefix >= int(pool_cols):
            prefix = rem
        if prefix <= 0 or prefix >= int(pool_cols):
            continue
        if warmup_mapping_fits([prefix], page, topk, steps, pool_cols):
            prefixes.append(int(prefix))
    return prefixes


def warmup_prefixes_for_bs(candidates: Sequence[int], raw_bs: int) -> list[list[int]]:
    raw_bs = int(raw_bs)
    cands = [int(x) for x in candidates]
    if raw_bs <= 0 or not cands:
        return []
    if raw_bs == 1:
        return [[p] for p in cands]
    if raw_bs == 2 and len(cands) >= 2:
        return [cands[:2], cands[-2:]]
    return [[cands[i % len(cands)] for i in range(raw_bs)]]


def slots_to_pages(slots, page_size: int) -> set[int]:
    page = max(int(page_size), 1)
    if slots is None:
        return set()
    if torch.is_tensor(slots):
        if int(slots.numel()) == 0:
            return set()
        pages = torch.unique(slots.detach().reshape(-1).to(dtype=torch.int64) // page)
        return {int(x) for x in pages.tolist()}
    values = list(slots)
    if not values:
        return set()
    return {int(v) // page for v in values}


def clone_page_lists(allocator) -> tuple:
    free_pages = getattr(allocator, "free_pages", None)
    release_pages = getattr(allocator, "release_pages", None)
    if torch.is_tensor(free_pages):
        free_pages = free_pages.detach().clone()
    if torch.is_tensor(release_pages):
        release_pages = release_pages.detach().clone()
    return free_pages, release_pages


def restore_page_lists(allocator, snapshot) -> None:
    free_pages, release_pages = snapshot
    if torch.is_tensor(free_pages):
        allocator.free_pages = free_pages.detach().clone()
    else:
        allocator.free_pages = free_pages
    if torch.is_tensor(release_pages):
        allocator.release_pages = release_pages.detach().clone()
    else:
        allocator.release_pages = release_pages


def allocator_in_free_group(allocator) -> bool:
    if not bool(getattr(allocator, "is_not_in_free_group", True)):
        return True
    group = getattr(allocator, "free_group", None)
    return bool(group)


def page_set(allocator) -> set[int]:
    free_pages = getattr(allocator, "free_pages", None)
    release_pages = getattr(allocator, "release_pages", None)
    out: set[int] = set()
    for tensor in (free_pages, release_pages):
        if tensor is None:
            continue
        if torch.is_tensor(tensor):
            out.update(int(x) for x in tensor.detach().reshape(-1).tolist())
        else:
            out.update(int(x) for x in tensor)
    return out


class SRWarmupTrackingAllocator:
    """Forwards to the production allocator and records page ownership."""

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "owned_pages", set())
        object.__setattr__(self, "_known", True)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @property
    def page_size(self) -> int:
        return int(getattr(self._inner, "page_size", 1) or 1)

    def _record_new_slots(self, slots) -> None:
        pages = slots_to_pages(slots, self.page_size)
        self.owned_pages.update(pages - self.owned_pages)

    def alloc(self, need_size: int):
        out = self._inner.alloc(need_size)
        self._record_new_slots(out)
        return out

    def alloc_extend(self, *args, **kwargs):
        out = self._inner.alloc_extend(*args, **kwargs)
        self._record_new_slots(out)
        return out

    def free(self, free_index):
        pages = slots_to_pages(free_index, self.page_size)
        if pages and not pages <= self.owned_pages:
            object.__setattr__(self, "_known", False)
            raise SRWarmupFatalError(
                f"warmup free of unowned pages {sorted(pages - self.owned_pages)}"
            )
        self.owned_pages.difference_update(pages)
        return self._inner.free(free_index)

    def backup_state(self):
        return (self._inner.backup_state(), frozenset(self.owned_pages))

    def restore_state(self, state):
        if (
            isinstance(state, tuple)
            and len(state) == 2
            and isinstance(state[1], (set, frozenset))
        ):
            inner_state, owned = state
            self._inner.restore_state(inner_state)
            self.owned_pages = set(owned)
            return
        object.__setattr__(self, "_known", False)
        raise SRWarmupFatalError("allocator restore without ownership ledger")

    def still_owns(self, slots) -> bool:
        if not self._known:
            raise SRWarmupFatalError("warmup ownership ledger is unknown")
        pages = slots_to_pages(slots, self.page_size)
        return bool(pages) and pages <= self.owned_pages


class SRWarmupCacheAdapter:
    """References the tracking allocator and never evicts the production cache."""

    def __init__(self, allocator):
        self.token_to_kv_pool_allocator = allocator

    def evict(self, *args, **kwargs):
        return None

    def is_chunk_cache(self):
        return False

    def pretty_print(self):
        return None

    def supports_mamba(self):
        return False

    @property
    def page_size(self):
        return getattr(self.token_to_kv_pool_allocator, "page_size", 1)


def warmup_should_run_finished_filter(kind: str) -> bool:
    """Filter only when some requests finished and others continue."""
    return kind == "partial"


def make_scratch_req_pool(rows: int, cols: int, device, dtype=torch.int64):
    mapping = torch.full((int(rows), int(cols)), -1, dtype=dtype, device=device)
    return SimpleNamespace(req_to_token=mapping)


def fill_prefix_slots(mapping, row: int, prefix: int, slots, page_size: int) -> None:
    prefix = int(prefix)
    if prefix <= 0:
        return
    if torch.is_tensor(slots):
        values = slots.reshape(-1)[:prefix]
    else:
        values = torch.as_tensor(list(slots)[:prefix], dtype=mapping.dtype, device=mapping.device)
    if int(values.numel()) < prefix:
        raise SRWarmupFatalError("prefix slots shorter than prefix length")
    mapping[int(row), :prefix] = values.to(dtype=mapping.dtype, device=mapping.device)


def alloc_prefix_pages(adapter: SRWarmupTrackingAllocator, prefixes: Sequence[int]):
    """Allocate exclusive prefix pages and return per-row token slots."""
    page = adapter.page_size
    device = getattr(adapter, "device", "cpu")
    rows = []
    for prefix in prefixes:
        n_pages = max(int(math.ceil(int(prefix) / page)), 1)
        need = n_pages * page
        if int(adapter.available_size()) < need:
            return None
        slots = adapter.alloc(need)
        if slots is None:
            return None
        rows.append(slots)
        _ = device
    return rows


def warmup_synchronize(device) -> None:
    text = str(getattr(device, "type", device)).lower()
    try:
        if text.startswith("npu"):
            torch.npu.synchronize()
        elif text.startswith("cuda"):
            torch.cuda.synchronize()
    except Exception as exc:
        raise SRWarmupFatalError(f"warmup device synchronize failed: {exc}") from exc


def target_keep_len(seq_len: int, accept_length: int, draft_token_num: int, page_size: int) -> int:
    page = max(int(page_size), 1)
    extended = int(seq_len) + int(draft_token_num)
    keep = ((int(seq_len) + int(accept_length) + 1 + page - 1) // page) * page
    return min(keep, extended)
