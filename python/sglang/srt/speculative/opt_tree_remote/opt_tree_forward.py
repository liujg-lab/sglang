"""One OPT-Tree forward per layer.

The committed tail is scored once. Each non-final ``update`` then runs N
tokens at one shared position under ``ancestor_mask``. Speculative KV is
appended past the committed prefix and dropped by the caller.

Several requests that are still growing share one forward at the same depth.
A single request, and any batch that fails, still grow through
``_opt_expand_one``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from sglang.srt.speculative.opt_tree_remote.opt_tree_config import OptTreeConfig
from sglang.srt.speculative.opt_tree_remote.opt_tree_grow import grow_opt_tree
from sglang.srt.speculative.opt_tree_remote.opt_tree_ops import OptTreeState, update

logger = logging.getLogger(__name__)


def layer_positions(prefix_len: int, depth: int, width: int) -> torch.Tensor:
    """Absolute token index shared by one OPT layer.

    ``depth`` is 1-based. The first frontier sits at ``prefix_len``, which is
    the next index after the committed tail.
    """
    if int(depth) < 1 or int(width) < 1:
        raise ValueError(f"invalid OPT layer depth={depth} width={width}")
    return torch.full(
        (int(width),),
        int(prefix_len) + int(depth) - 1,
        dtype=torch.long,
    )


def layer_attention_mask(prefix_len: int, ancestor: torch.Tensor) -> torch.Tensor:
    """Flattened TARGET_VERIFY mask: committed prefix, then the ancestor mask.

    ``ancestor`` is ``[N, depth * N]`` with 1 for a visible tree column.
    Every query sees the whole committed prefix. Tree columns stay
    ancestor-only (``leak=False``).
    """
    if ancestor.dim() != 2:
        raise ValueError(
            f"ancestor mask must be [N, cols], got {tuple(ancestor.shape)}"
        )
    n = int(ancestor.shape[0])
    prefix = torch.ones(
        (n, int(prefix_len)),
        dtype=torch.bool,
        device=ancestor.device,
    )
    tree = ancestor.to(dtype=torch.bool)
    return torch.cat([prefix, tree], dim=1).reshape(-1)


def pack_verify_attention(
    prefix_lens: Sequence[int],
    ancestors: Sequence[torch.Tensor],
    depth: int,
    width: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenate per-request TARGET_VERIFY masks and positions.

    Mask segment ``b`` is ``[N, prefix_lens[b] + depth * N]`` flattened, which
    is ``[N, cached_len_b + N]`` once ``cached_len`` counts layers already
    forwarded. Positions for one request are identical.
    """
    if len(prefix_lens) != len(ancestors):
        raise ValueError(
            f"OPT verify pack has {len(prefix_lens)} prefixes and "
            f"{len(ancestors)} masks"
        )
    masks = [
        layer_attention_mask(prefix, ancestor)
        for prefix, ancestor in zip(prefix_lens, ancestors)
    ]
    positions = [layer_positions(prefix, depth, width) for prefix in prefix_lens]
    if not masks:
        return torch.empty(0, dtype=torch.bool), torch.empty(0, dtype=torch.long)
    return torch.cat(masks, dim=0), torch.cat(positions, dim=0)


def release_uncommitted(req, rollback) -> int:
    """Free slots past ``kv_committed_len`` before extend reuses that span.

    ``kv_allocated_len`` can sit at or below the logical committed length.
    Rolling back only past that length leaves ``[kv_committed_len,
    kv_allocated_len)`` allocated. The next extend writes new slots over the
    same columns and drops the old indices.
    """
    cached = int(getattr(req, "kv_committed_len", 0) or 0)
    if int(getattr(req, "kv_allocated_len", 0) or 0) > cached:
        rollback(req, cached)
    return cached


def opt_extend_chunk_limit(chunked_prefill_size) -> Optional[int]:
    """Largest single OPT extend. ``None`` or a non-positive size means no cap."""
    if chunked_prefill_size is None:
        return None
    limit = int(chunked_prefill_size)
    if limit <= 0:
        return None
    return limit


def plan_opt_tail(req, chunked_prefill_size) -> str:
    """How OPT writes the committed tail.

    ``reprefill`` when the KV prefix is missing or does not cover the prompt.
    ``noop`` when that prefix already covers every committed token.
    ``extend`` when the missing output fits in one chunk. ``extend_chunked``
    when it does not. This does not use the per-token decode threshold.
    """
    origin = getattr(req, "origin_input_ids", None) or []
    output = getattr(req, "output_ids", None) or []
    kv_len = int(getattr(req, "kv_committed_len", 0) or 0)
    if getattr(req, "req_pool_idx", None) is None or kv_len < len(origin):
        return "reprefill"
    missing = len(origin) + len(output) - kv_len
    if missing <= 0:
        return "noop"
    limit = opt_extend_chunk_limit(chunked_prefill_size)
    if limit is None or missing <= limit:
        return "extend"
    return "extend_chunked"


def extend_chunk_bounds(start: int, end: int, chunk: int) -> List[Tuple[int, int]]:
    """Half-open spans of at most ``chunk`` tokens covering ``[start, end)``."""
    start = int(start)
    end = int(end)
    chunk = int(chunk)
    if chunk <= 0:
        raise ValueError(f"OPT extend chunk must be positive, got {chunk}")
    if start >= end:
        return []
    bounds = []
    cursor = start
    while cursor < end:
        nxt = min(end, cursor + chunk)
        bounds.append((cursor, nxt))
        cursor = nxt
    return bounds


def stash_finished_prefill_logits(scheduler, batch, logits) -> None:
    """Keep the last row for requests whose prompt KV just finished."""
    reqs = list(getattr(batch, "reqs", None) or [])
    if not reqs:
        return
    try:
        rows = _extend_request_rows(logits, len(reqs))
    except RuntimeError as exc:
        logger.warning("[OPT] skip prefill tail logits: %s", exc)
        return
    for req, row in zip(reqs, rows):
        if scheduler._sr_prefix_pending(req):
            continue
        req.opt_prefill_tail_logits = row


def take_cached_tail_logits(req, logical_len: int):
    """Return a stashed prefill row when KV already covers ``logical_len``."""
    row = getattr(req, "opt_prefill_tail_logits", None)
    if row is None:
        return None
    req.opt_prefill_tail_logits = None
    cached = int(getattr(req, "kv_committed_len", 0) or 0)
    if cached != int(logical_len):
        return None
    return row


def clear_prefill_tail_logits(req) -> None:
    if getattr(req, "opt_prefill_tail_logits", None) is not None:
        req.opt_prefill_tail_logits = None


def reprefill_tail_logits(scheduler, reqs) -> List[torch.Tensor]:
    """Rebuild an invalid prefix and keep the finishing chunk's last row."""

    def _keep(batch, logits) -> None:
        stash_finished_prefill_logits(scheduler, batch, logits)

    scheduler._sr_reprefill_committed(reqs, on_extend_logits=_keep)
    rows = []
    missing = []
    for req in reqs:
        row = take_cached_tail_logits(req, _committed_len(req))
        if row is None:
            missing.append(getattr(req, "rid", None))
            continue
        rows.append(row)
    if missing:
        raise RuntimeError(
            f"OPT draft reprefill returned no tail logits for {missing}"
        )
    return rows


def materialize_missing_prefix(scheduler, reqs, capture_tail_logits: bool = False) -> None:
    """Build draft KV before the seed forward.

    ``_sr_kv_len`` treats a request with no pool slot as already cached, so
    OPT checks ``req_pool_idx`` itself and does not change that helper.
    STEP keeps the last row of the chunk that finishes the prompt so expand
    does not decode that tail again. PREFILL does not need those logits.
    """
    missing = [req for req in reqs if req.req_pool_idx is None]
    if not missing:
        return

    def _keep(batch, logits) -> None:
        stash_finished_prefill_logits(scheduler, batch, logits)

    if capture_tail_logits:
        scheduler._sr_materialize_prefix_batch(missing, on_extend_logits=_keep)
    else:
        scheduler._sr_materialize_prefix_batch(missing)
    for req in missing:
        if req.req_pool_idx is None:
            raise RuntimeError(f"OPT draft {req.rid} prefix KV was not materialized")


def produce_opt_windows(scheduler, action, pairs) -> list:
    """PREFILL only materializes KV. STEP materializes, then grows a tree.

    A short committed tail is written by the seed extend. Grammar replay
    still runs first: that forward clears the grammar, and the replay keeps
    constraints aligned with the committed tokens.
    """
    from sglang.srt.speculative.standalone_remote.sr_protocol import SRAction

    if not pairs:
        return []
    reqs = [req for _, req in pairs]
    if action == SRAction.PREFILL:
        materialize_missing_prefix(scheduler, reqs)
        return [([], None, None)] * len(pairs)
    if action != SRAction.STEP:
        return [([], None, None)] * len(pairs)
    materialize_missing_prefix(scheduler, reqs, capture_tail_logits=True)
    scheduler._sr_replay_grammars(reqs)
    if len(reqs) <= 1:
        return [scheduler._opt_expand_one(req) for req in reqs]
    try:
        return grow_batch(scheduler, reqs)
    except Exception as exc:
        logger.warning("[OPT] batch tree expand failed, falling back: %s", exc)
        return [scheduler._opt_expand_one(req) for req in reqs]


def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    """Temperature-1 full-vocabulary probabilities. No top-N renormalization."""
    return torch.softmax(logits.detach().float(), dim=-1)


def grow_with_forwards(
    config: OptTreeConfig,
    tail_logits: torch.Tensor,
    forward_layer: Callable,
) -> Tuple[List[int], List[int], List[int]]:
    """Seed from the committed tail, then forward every layer except the last.

    ``forward_layer(step)`` receives a non-final step and returns logits for
    its N tokens. The stopping ``update`` verifies that layer and is not
    forwarded.
    """

    def next_probs(step, state):
        if step is None or state is None:
            return softmax_probs(tail_logits)
        return softmax_probs(forward_layer(step))

    return grow_opt_tree(config, next_probs)


def _verify_window(step) -> Tuple[List[int], List[int], List[int]]:
    verify = step.verify
    if verify is None:
        raise RuntimeError("OPT tree finished without a verify set")
    return (
        [int(x) for x in verify.tokens.tolist()],
        [int(x) for x in verify.parents.tolist()],
        [int(x) for x in verify.cols.tolist()],
    )


def _shared_depth(steps) -> int:
    depths = [int(step.positions.reshape(-1)[0].item()) for step in steps]
    if any(depth != depths[0] for depth in depths):
        raise RuntimeError(f"OPT batch layers are at different depths: {depths}")
    return depths[0]


def _as_layer_rows(logits, bs: int, width: int) -> List[torch.Tensor]:
    """Split one layer's logits into ``bs`` tensors of shape ``[N, vocab]``."""
    if isinstance(logits, (list, tuple)):
        rows = list(logits)
        if len(rows) != bs:
            raise RuntimeError(
                f"OPT layer returned {len(rows)} rows for {bs} requests"
            )
        return rows
    t = logits.detach()
    if t.dim() == 3 and int(t.shape[0]) == bs and int(t.shape[1]) == width:
        return [t[i] for i in range(bs)]
    if t.dim() == 3 and int(t.shape[0]) == 1:
        t = t[0]
    if t.dim() == 1:
        t = t.unsqueeze(0)
    need = bs * width
    if t.dim() != 2 or int(t.shape[0]) != need:
        raise RuntimeError(
            f"OPT layer logits shape {tuple(logits.shape)} != [{bs}, {width}, vocab]"
        )
    return [t[i * width : (i + 1) * width] for i in range(bs)]


def grow_active_layers(config: OptTreeConfig, tail_logits, forward_active: Callable):
    """Grow one tree per tail. Requests still at one depth share a forward.

    ``forward_active(steps, indices)`` sees only non-final steps. ``indices``
    are positions into ``tail_logits``. The return value is a list of
    ``[N, vocab]`` logits, or a tensor of shape ``[bs, N, vocab]`` or
    ``[bs * N, vocab]``. A request that stops is absent from the next call.
    """
    tails = list(tail_logits)
    states = [OptTreeState.create(config) for _ in tails]
    steps = [None] * len(tails)
    results = [None] * len(tails)
    active: List[int] = []
    for i, logits in enumerate(tails):
        step = update(states[i], softmax_probs(logits))
        steps[i] = step
        if step.is_final:
            results[i] = _verify_window(step)
        else:
            active.append(i)
    guard = 0
    while active:
        guard += 1
        if guard > config.max_depth:
            raise RuntimeError(
                f"OPT-Tree did not finish within max_depth {config.max_depth}"
            )
        chosen = [steps[i] for i in active]
        _shared_depth(chosen)
        rows = _as_layer_rows(
            forward_active(chosen, list(active)), len(active), config.width
        )
        nxt: List[int] = []
        for i, row in zip(active, rows):
            step = update(states[i], softmax_probs(row))
            steps[i] = step
            if step.is_final:
                results[i] = _verify_window(step)
            else:
                nxt.append(i)
        active = nxt
    if any(item is None for item in results):
        raise RuntimeError("OPT batch finished with a missing verify set")
    return results


@dataclass
class OptLayerSpec:
    """Fields TARGET_VERIFY attention already reads. Not an EAGLE verify input.

    ``seq_lens`` is the prefix. A DECODE graph counts the token being written,
    so this spec must not replay one even when the layer width is 1.
    """

    forbids_decode_graph = True

    custom_mask: torch.Tensor
    positions: torch.Tensor
    draft_token_num: int
    num_tokens_per_req: int
    capture_hidden_mode: object = None

    def is_draft_input(self) -> bool:
        return False

    def get_spec_adjusted_global_num_tokens(self, batch):
        return batch.global_num_tokens, batch.global_num_tokens_for_logprob


class OptLayerForward:
    """Tail logits plus one N-token extend per OPT layer. Batch size is 1."""

    def __init__(self, scheduler, req, prefix_len: int):
        self.scheduler = scheduler
        self.req = req
        self.prefix_len = int(prefix_len)
        self.cached_len = int(prefix_len)

    def logits_at_tail(self) -> torch.Tensor:
        """Recompute the committed tail in its existing KV slot."""
        if self.prefix_len < 1:
            raise RuntimeError(f"OPT draft {self.req.rid} has an empty prefix")
        if self.req.req_pool_idx is None:
            raise RuntimeError(f"OPT draft {self.req.rid} has no KV pool slot")
        pool = self.scheduler.req_to_token_pool.req_to_token
        slot = pool[self.req.req_pool_idx, self.prefix_len - 1]
        if int(slot.item()) < 0:
            raise RuntimeError(f"OPT draft {self.req.rid} tail KV slot is empty")
        batch = self.scheduler._sr_make_decode_batch([self.req])
        batch.input_ids = batch.output_ids
        batch.out_cache_loc = slot.reshape(1)
        batch.output_ids = None
        _keep_raw_logits(batch)
        return _row_logits(self._run(batch), 1, tail=True)

    def extend_uncommitted(self) -> torch.Tensor:
        """Write ``[kv_committed_len, committed)`` in one EXTEND.

        The last row is the next-token distribution after the committed tail.
        ``prefix_lens`` is the real KV length. This is not the zero-prefix
        tail-extend graph.
        """
        start = int(getattr(self.req, "kv_committed_len", 0) or 0)
        return self._extend_range(start, self.prefix_len)

    def extend_uncommitted_chunked(self, chunk: int) -> torch.Tensor:
        """Write the same span in chunks. Only the last chunk's row is kept."""
        start = int(getattr(self.req, "kv_committed_len", 0) or 0)
        logits = None
        for begin, end in extend_chunk_bounds(start, self.prefix_len, chunk):
            logits = self._extend_range(begin, end)
        if logits is None:
            raise RuntimeError(
                f"OPT draft {self.req.rid} chunked extend "
                f"[{start}, {self.prefix_len}) is empty"
            )
        return logits

    def _extend_range(self, start: int, end: int) -> torch.Tensor:
        """Write ``[start, end)`` and publish that end as committed KV."""
        start = int(start)
        end = int(end)
        if self.req.req_pool_idx is None:
            raise RuntimeError(f"OPT draft {self.req.rid} has no KV pool slot")
        if start < 0 or start >= end:
            raise RuntimeError(
                f"OPT draft {self.req.rid} extend span [{start}, {end}) is empty"
            )
        n = end - start
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
            read_token_span,
        )

        tokens = read_token_span(
            self.req.origin_input_ids, self.req.output_ids, start, n
        )
        if len(tokens) != n:
            raise RuntimeError(
                f"OPT draft {self.req.rid} is missing tokens at [{start}, {end})"
            )
        out_cache_loc = self._alloc_layer_slots(n, start=start)
        self.req.kv_allocated_len = end
        try:
            batch = self._extend_batch(tokens, start, out_cache_loc)
            logits = _row_logits(self._run(batch), n, tail=True)
        except Exception:
            self.scheduler._opt_rollback(self.req, start)
            raise
        self.req.kv_committed_len = end
        self.req.kv_allocated_len = end
        self.cached_len = end
        return logits

    def forward_layer(self, step) -> torch.Tensor:
        """Forward the N tokens ``update`` just kept. KV stays uncommitted."""
        tokens = step.tokens
        if not torch.is_tensor(tokens):
            tokens = torch.tensor(list(tokens), dtype=torch.long)
        n = int(tokens.shape[0])
        width = int(self.scheduler.server_args.opt_tree_width)
        if n != width:
            raise RuntimeError(
                f"OPT layer has {n} tokens but attention was built for {width}"
            )
        depth = int(step.positions.reshape(-1)[0].item())
        ancestor = step.ancestor_mask
        if int(ancestor.shape[0]) != n or int(ancestor.shape[1]) != depth * n:
            raise RuntimeError(
                f"OPT ancestor mask {tuple(ancestor.shape)} != [{n}, {depth * n}]"
            )
        if self.cached_len + n != self.prefix_len + depth * n:
            raise RuntimeError(
                f"OPT cache length {self.cached_len} is not layer {depth} "
                f"after prefix {self.prefix_len}"
            )
        device = self.scheduler.req_to_token_pool.device
        positions = layer_positions(self.prefix_len, depth, n).to(device)
        mask = layer_attention_mask(self.prefix_len, ancestor).to(device)
        out_cache_loc = self._alloc_layer_slots(n)
        self.req.kv_allocated_len = self.cached_len + n
        batch = self._verify_batch(tokens.to(device), positions, mask, out_cache_loc)
        logits = _row_logits(self._run(batch), n, tail=False)
        self.cached_len += n
        return logits

    def _alloc_layer_slots(self, n: int, start: Optional[int] = None) -> torch.Tensor:
        from sglang.srt.mem_cache.common import (
            alloc_paged_token_slots_extend,
            alloc_token_slots,
            get_last_loc,
        )
        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        scheduler = self.scheduler
        tree_cache = scheduler.tree_cache
        page = int(getattr(tree_cache, "page_size", 1) or 1)
        device = scheduler.req_to_token_pool.device
        base = self.cached_len if start is None else int(start)
        seq = torch.tensor([base], dtype=torch.int64, device=device)
        pool_index = torch.tensor(
            [int(self.req.req_pool_idx)], dtype=torch.int64, device=device
        )
        if page == 1:
            out_cache_loc = alloc_token_slots(tree_cache, n)
        else:
            seq_cpu = torch.tensor([base], dtype=torch.int64)
            end = seq + n
            end_cpu = seq_cpu + n
            last = get_last_loc(
                scheduler.req_to_token_pool.req_to_token,
                pool_index,
                seq,
            )
            out_cache_loc = alloc_paged_token_slots_extend(
                tree_cache,
                seq,
                seq_cpu,
                end,
                end_cpu,
                last,
                n,
            )
        if int(out_cache_loc.numel()) != n:
            raise RuntimeError(
                f"OPT layer allocated {int(out_cache_loc.numel())} slots for {n} tokens"
            )
        assign_req_to_token_pool_func(
            pool_index,
            scheduler.req_to_token_pool.req_to_token,
            seq,
            seq + n,
            out_cache_loc,
            1,
        )
        return out_cache_loc

    def _verify_batch(self, tokens, positions, mask, out_cache_loc):
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardMode,
        )

        n = int(tokens.shape[0])
        batch = self.scheduler._sr_make_decode_batch([self.req])
        device = batch.seq_lens.device
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.global_forward_mode = ForwardMode.TARGET_VERIFY
        batch.input_ids = tokens.to(dtype=torch.int64, device=device).reshape(-1)
        batch.output_ids = None
        batch.seq_lens = torch.tensor([self.cached_len], dtype=torch.int64, device=device)
        batch.seq_lens_cpu = torch.tensor([self.cached_len], dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(
            [self.cached_len], dtype=torch.int32, device=device
        )
        batch.seq_lens_sum = int(self.cached_len)
        batch.out_cache_loc = out_cache_loc
        batch.extend_lens = [n]
        batch.prefix_lens = [self.cached_len]
        batch.extend_num_tokens = n
        batch.extend_logprob_start_lens = [self.cached_len]
        batch.spec_info = OptLayerSpec(
            custom_mask=mask.reshape(-1),
            positions=positions.to(dtype=torch.int64, device=device).reshape(-1),
            draft_token_num=n,
            num_tokens_per_req=n,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        _keep_raw_logits(batch)
        return batch

    def _extend_batch(self, tokens, prefix: int, out_cache_loc):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        n = len(tokens)
        end = int(prefix) + n
        batch = self.scheduler._sr_make_decode_batch([self.req])
        device = batch.seq_lens.device
        batch.forward_mode = ForwardMode.EXTEND
        batch.global_forward_mode = ForwardMode.EXTEND
        batch.input_ids = torch.tensor(tokens, dtype=torch.int64, device=device)
        batch.output_ids = None
        batch.seq_lens = torch.tensor([end], dtype=torch.int64, device=device)
        batch.seq_lens_cpu = torch.tensor([end], dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor([end], dtype=torch.int32, device=device)
        batch.seq_lens_sum = end
        batch.out_cache_loc = out_cache_loc
        batch.extend_lens = [n]
        batch.prefix_lens = [int(prefix)]
        batch.extend_num_tokens = n
        batch.extend_logprob_start_lens = [0]
        _keep_raw_logits(batch)
        return batch

    def _run(self, batch) -> torch.Tensor:
        return _run_batch(self.scheduler, batch)


def _keep_raw_logits(batch) -> None:
    """Leave next-token logits unscaled so OPT can softmax at temperature 1.

    The sampler turns non-greedy logits into probabilities in place. OPT's
    path score is the full-vocabulary temperature-1 softmax, so this batch
    only argmaxes and discards that token. Penalty and logit bias are also
    cleared: ``apply_logits_bias`` would add them onto these logits. This
    sampling info belongs to the batch, not the request.
    """
    info = batch.sampling_info
    info.is_all_greedy = True
    info.need_top_p_sampling = False
    info.need_top_k_sampling = False
    info.need_min_p_sampling = False
    info.temperatures = torch.ones_like(info.temperatures)
    info.acc_additive_penalties = None
    info.acc_scaling_penalties = None
    info.penalizer_orchestrator = None
    info.logit_bias = None
    info.vocab_mask = None
    batch.has_grammar = False
    batch.return_logprob = False
    batch.return_hidden_states = False


def _row_logits(logits: torch.Tensor, n: int, *, tail: bool) -> torch.Tensor:
    t = logits.detach()
    if t.dim() == 3 and int(t.shape[0]) == 1:
        t = t[0]
    if tail:
        if t.dim() == 2:
            return t[-1]
        if t.dim() != 1:
            raise RuntimeError(f"OPT tail logits rank {t.dim()} is not 1 or 2")
        return t
    if t.dim() != 2 or int(t.shape[0]) != n:
        raise RuntimeError(f"OPT layer logits shape {tuple(t.shape)} != [{n}, vocab]")
    return t


def _run_batch(scheduler, batch) -> torch.Tensor:
    scheduler.cur_batch = batch
    result = scheduler.run_batch(batch)
    logits = getattr(getattr(result, "logits_output", None), "next_token_logits", None)
    if logits is None:
        raise RuntimeError("OPT draft forward returned no logits")
    return logits


def _flatten_logits(logits: torch.Tensor) -> torch.Tensor:
    t = logits.detach()
    if t.dim() == 3 and int(t.shape[0]) == 1:
        t = t[0]
    if t.dim() == 1:
        return t.unsqueeze(0)
    if t.dim() != 2:
        raise RuntimeError(f"OPT logits rank {t.dim()} is not 1 or 2")
    return t


def _extend_request_rows(logits: torch.Tensor, num_reqs: int) -> List[torch.Tensor]:
    """One tail row per request.

    EXTEND without ``return_logprob`` already keeps only the last hidden
    state of each request, so ``next_token_logits`` is ``[num_reqs, vocab]``.
    """
    flat = _flatten_logits(logits)
    if int(flat.shape[0]) != int(num_reqs):
        raise RuntimeError(
            f"OPT extend logits rows {int(flat.shape[0])} != {int(num_reqs)} requests"
        )
    return [flat[i] for i in range(int(num_reqs))]


def _extend_batch_many(scheduler, runners, token_spans, prefix_lens, out_cache_loc):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    reqs = [runner.req for runner in runners]
    batch = scheduler._sr_make_decode_batch(reqs)
    device = batch.seq_lens.device
    extend_lens = [len(span) for span in token_spans]
    ends = [int(prefix) + n for prefix, n in zip(prefix_lens, extend_lens)]
    flat = [int(tok) for span in token_spans for tok in span]
    batch.forward_mode = ForwardMode.EXTEND
    batch.global_forward_mode = ForwardMode.EXTEND
    batch.input_ids = torch.tensor(flat, dtype=torch.int64, device=device)
    batch.output_ids = None
    batch.seq_lens = torch.tensor(ends, dtype=torch.int64, device=device)
    batch.seq_lens_cpu = torch.tensor(ends, dtype=torch.int64)
    batch.orig_seq_lens = torch.tensor(ends, dtype=torch.int32, device=device)
    batch.seq_lens_sum = sum(ends)
    batch.out_cache_loc = out_cache_loc
    batch.extend_lens = extend_lens
    batch.prefix_lens = [int(prefix) for prefix in prefix_lens]
    batch.extend_num_tokens = sum(extend_lens)
    batch.extend_logprob_start_lens = [0] * len(runners)
    _keep_raw_logits(batch)
    return batch


def _verify_batch_many(
    scheduler, runners, tokens, positions, custom_mask, out_cache_loc, cached
):
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )

    reqs = [runner.req for runner in runners]
    batch = scheduler._sr_make_decode_batch(reqs)
    device = batch.seq_lens.device
    width = int(tokens[0].shape[0]) if tokens else 0
    cached = [int(n) for n in cached]
    batch.forward_mode = ForwardMode.TARGET_VERIFY
    batch.global_forward_mode = ForwardMode.TARGET_VERIFY
    batch.input_ids = torch.cat([row.reshape(-1) for row in tokens]).to(
        dtype=torch.int64, device=device
    )
    batch.output_ids = None
    batch.seq_lens = torch.tensor(cached, dtype=torch.int64, device=device)
    batch.seq_lens_cpu = torch.tensor(cached, dtype=torch.int64)
    batch.orig_seq_lens = torch.tensor(cached, dtype=torch.int32, device=device)
    batch.seq_lens_sum = sum(cached)
    batch.out_cache_loc = out_cache_loc
    batch.extend_lens = [width] * len(runners)
    batch.prefix_lens = list(cached)
    batch.extend_num_tokens = width * len(runners)
    batch.extend_logprob_start_lens = list(cached)
    batch.spec_info = OptLayerSpec(
        custom_mask=custom_mask.reshape(-1),
        positions=positions.reshape(-1).to(dtype=torch.int64, device=device),
        draft_token_num=width,
        num_tokens_per_req=width,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )
    _keep_raw_logits(batch)
    return batch


def _logits_at_tail_batch(scheduler, runners) -> List[torch.Tensor]:
    """One DECODE row per request, reusing the committed tail slot."""
    if not runners:
        return []
    pool = scheduler.req_to_token_pool.req_to_token
    slots = []
    for runner in runners:
        if runner.prefix_len < 1:
            raise RuntimeError(f"OPT draft {runner.req.rid} has an empty prefix")
        if runner.req.req_pool_idx is None:
            raise RuntimeError(f"OPT draft {runner.req.rid} has no KV pool slot")
        slot = pool[runner.req.req_pool_idx, runner.prefix_len - 1]
        if int(slot.item()) < 0:
            raise RuntimeError(f"OPT draft {runner.req.rid} tail KV slot is empty")
        slots.append(slot.reshape(-1))
    batch = scheduler._sr_make_decode_batch([runner.req for runner in runners])
    batch.input_ids = batch.output_ids
    batch.out_cache_loc = torch.cat(slots)
    batch.output_ids = None
    _keep_raw_logits(batch)
    logits = _flatten_logits(_run_batch(scheduler, batch))
    if int(logits.shape[0]) != len(runners):
        raise RuntimeError(
            f"OPT tail logits rows {int(logits.shape[0])} != {len(runners)} requests"
        )
    return [logits[i] for i in range(len(runners))]


def _extend_span_batch(scheduler, runners, ends) -> List[torch.Tensor]:
    """EXTEND each runner from its committed KV to ``ends[i]``.

    A failed runner rolls back only to the start of this span. Earlier chunks
    stay committed.
    """
    from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
        read_token_span,
    )

    if len(runners) != len(ends):
        raise RuntimeError(
            f"OPT extend has {len(runners)} runners and {len(ends)} ends"
        )
    prepared = []
    try:
        spans = []
        starts = []
        locs = []
        for runner, end in zip(runners, ends):
            start = int(getattr(runner.req, "kv_committed_len", 0) or 0)
            end = int(end)
            if runner.req.req_pool_idx is None:
                raise RuntimeError(f"OPT draft {runner.req.rid} has no KV pool slot")
            if start < 0 or start >= end:
                raise RuntimeError(
                    f"OPT draft {runner.req.rid} extend span [{start}, {end}) is empty"
                )
            n = end - start
            tokens = read_token_span(
                runner.req.origin_input_ids, runner.req.output_ids, start, n
            )
            if len(tokens) != n:
                raise RuntimeError(
                    f"OPT draft {runner.req.rid} is missing tokens at [{start}, {end})"
                )
            loc = runner._alloc_layer_slots(n, start=start)
            runner.req.kv_allocated_len = end
            prepared.append((runner, start, end))
            spans.append(tokens)
            starts.append(start)
            locs.append(loc.reshape(-1))
        batch = _extend_batch_many(
            scheduler,
            runners,
            spans,
            starts,
            torch.cat(locs),
        )
        rows = _extend_request_rows(_run_batch(scheduler, batch), len(runners))
    except Exception:
        for runner, start, _end in prepared:
            scheduler._opt_rollback(runner.req, start)
        raise
    for runner, _start, end in prepared:
        runner.req.kv_committed_len = end
        runner.req.kv_allocated_len = end
        runner.cached_len = end
    return rows


def _extend_uncommitted_batch(scheduler, runners) -> List[torch.Tensor]:
    """Write every short tail in one EXTEND and keep each request's last row."""
    return _extend_span_batch(
        scheduler, runners, [runner.prefix_len for runner in runners]
    )


def _extend_uncommitted_chunked_batch(
    scheduler, runners, chunk: int
) -> List[torch.Tensor]:
    """Extend each long tail by ``chunk`` tokens. Keep the row that finishes it."""
    pending = list(runners)
    saved = {}
    while pending:
        before = [
            int(getattr(runner.req, "kv_committed_len", 0) or 0) for runner in pending
        ]
        ends = []
        for runner, start in zip(pending, before):
            bounds = extend_chunk_bounds(start, runner.prefix_len, chunk)
            if not bounds:
                raise RuntimeError(
                    f"OPT draft {runner.req.rid} chunked extend "
                    f"[{start}, {runner.prefix_len}) is empty"
                )
            ends.append(bounds[0][1])
        rows = _extend_span_batch(scheduler, pending, ends)
        nxt = []
        for runner, row, end in zip(pending, rows, ends):
            if int(end) >= int(runner.prefix_len):
                saved[id(runner)] = row
            else:
                nxt.append(runner)
        after = [
            int(getattr(runner.req, "kv_committed_len", 0) or 0) for runner in pending
        ]
        if after == before:
            raise RuntimeError("OPT chunked extend made no progress")
        pending = nxt
    return [saved[id(runner)] for runner in runners]


def _forward_verify_batch(scheduler, runners, steps) -> torch.Tensor:
    """One TARGET_VERIFY for requests that still share a depth."""
    if not runners:
        raise RuntimeError("OPT verify batch is empty")
    width = int(scheduler.server_args.opt_tree_width)
    device = scheduler.req_to_token_pool.device
    depth = _shared_depth(steps)
    token_rows = []
    ancestors = []
    for runner, step in zip(runners, steps):
        tokens = step.tokens
        if not torch.is_tensor(tokens):
            tokens = torch.tensor(list(tokens), dtype=torch.long)
        n = int(tokens.shape[0])
        if n != width:
            raise RuntimeError(
                f"OPT layer has {n} tokens but attention was built for {width}"
            )
        ancestor = step.ancestor_mask
        anc_shape = None if ancestor is None else tuple(ancestor.shape)
        if anc_shape != (n, depth * n):
            raise RuntimeError(
                f"OPT ancestor mask {anc_shape} != [{n}, {depth * n}]"
            )
        if runner.cached_len + n != runner.prefix_len + depth * n:
            raise RuntimeError(
                f"OPT cache length {runner.cached_len} is not layer {depth} "
                f"after prefix {runner.prefix_len}"
            )
        token_rows.append(tokens.to(dtype=torch.long, device=device))
        ancestors.append(ancestor)
    prefix_lens = [runner.prefix_len for runner in runners]
    flat_mask, flat_positions = pack_verify_attention(
        prefix_lens, ancestors, depth, width
    )
    locs = []
    cached = []
    for runner in runners:
        loc = runner._alloc_layer_slots(width)
        runner.req.kv_allocated_len = runner.cached_len + width
        locs.append(loc.reshape(-1))
        cached.append(runner.cached_len)
    batch = _verify_batch_many(
        scheduler,
        runners,
        token_rows,
        flat_positions.to(device),
        flat_mask.to(device),
        torch.cat(locs),
        cached,
    )
    logits = _run_batch(scheduler, batch)
    for runner in runners:
        runner.cached_len += width
    return logits


def ingest_tail_logits(scheduler, runners) -> List[torch.Tensor]:
    """Logits after each committed tail, without SR's per-token decode threshold."""
    limit = getattr(scheduler, "chunked_prefill_size", None)
    logits = {}
    extend = []
    chunked = []
    noop = []
    reprefill = []
    for runner in runners:
        cached_row = take_cached_tail_logits(runner.req, runner.prefix_len)
        if cached_row is not None:
            logits[id(runner)] = cached_row
            continue
        mode = plan_opt_tail(runner.req, limit)
        if mode == "reprefill":
            reprefill.append(runner)
        elif mode == "extend":
            extend.append(runner)
        elif mode == "extend_chunked":
            chunked.append(runner)
        elif mode == "noop":
            noop.append(runner)
        else:
            raise RuntimeError(
                f"OPT draft {runner.req.rid} ingest mode {mode} is unknown"
            )
    if reprefill:
        rows = reprefill_tail_logits(scheduler, [runner.req for runner in reprefill])
        for runner, row in zip(reprefill, rows):
            logits[id(runner)] = row
    if extend:
        for runner, row in zip(extend, _extend_uncommitted_batch(scheduler, extend)):
            logits[id(runner)] = row
    if chunked:
        rows = _extend_uncommitted_chunked_batch(scheduler, chunked, int(limit))
        for runner, row in zip(chunked, rows):
            logits[id(runner)] = row
    if noop:
        for runner, row in zip(noop, _logits_at_tail_batch(scheduler, noop)):
            logits[id(runner)] = row
    return [logits[id(runner)] for runner in runners]


def _committed_len(req) -> int:
    return len(req.origin_input_ids or []) + len(req.output_ids or [])


def grow_batch(scheduler, reqs):
    """Grow every request together. Tree KV is dropped before returning.

    A missing output that fits in one chunk shares one EXTEND. A longer output
    is extended in ``chunked_prefill_size`` pieces, and only the last piece's
    row is kept. Requests that still have the same depth share one
    TARGET_VERIFY. ``finally`` rolls each request back to the committed
    length, matching ``_opt_expand_one``.
    """
    if not reqs:
        return []
    committed = [_committed_len(req) for req in reqs]
    try:
        for req in reqs:
            scheduler._sr_resume_req(req)
        scheduler._sr_isolate_need({req.rid for req in reqs})
        for req in reqs:
            scheduler._opt_release_uncommitted(req)
        config = OptTreeConfig(
            width=int(scheduler.server_args.opt_tree_width),
            tau=float(scheduler.server_args.opt_tree_tau),
            max_depth=int(scheduler.server_args.opt_tree_max_depth),
        )
        runners = [
            OptLayerForward(scheduler, req, keep) for req, keep in zip(reqs, committed)
        ]
        tail_logits = ingest_tail_logits(scheduler, runners)
        for runner, keep in zip(runners, committed):
            written = int(getattr(runner.req, "kv_committed_len", 0) or 0)
            if written != keep:
                raise RuntimeError(
                    f"OPT draft {runner.req.rid} committed KV {written} != {keep} tokens"
                )

        def forward_active(steps, indices):
            return _forward_verify_batch(
                scheduler, [runners[i] for i in indices], steps
            )

        return grow_active_layers(config, tail_logits, forward_active)
    finally:
        for req, keep in zip(reqs, committed):
            clear_prefill_tail_logits(req)
            scheduler._opt_rollback(req, keep)
            if not scheduler._sr_is_finished(req):
                scheduler._sr_pause_req(req)
        scheduler.last_batch = None
        scheduler.cur_batch = None
