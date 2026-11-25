"""One OPT-Tree forward per layer.

The committed tail is scored once. Each non-final ``update`` then runs N
tokens at one shared position under ``ancestor_mask``. Speculative KV is
appended past the committed prefix. On CUDA page size 1 a successful expand
keeps those slots in ``req.clear_tree_lease``. Every other path drops them.

Several requests that are still growing share one forward at the same depth.
A single request, and any batch that fails for a reason other than a device
context error, still grow through ``_clear_expand_one``. A device context error
poisons the draft and propagates. On page size 1 the last layer's KV can be
forwarded after the reply; the verify window is already fixed by then.
"""

from __future__ import annotations

import bisect
import copy
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.clear_tree_remote.clear_tree_config import (
    ClearTreeConfig,
    bonus_prediction_enabled,
    opt_draft_lookahead_enabled,
    opt_probe_layer_wait_enabled,
)
from sglang.srt.speculative.clear_tree_remote.clear_tree_grow import grow_clear_tree
from sglang.srt.speculative.clear_tree_remote.clear_tree_ops import (
    ClearTreeState,
    ClearTreeStep,
    _layer_positions,
    expand_from_topk,
    expand_on_device,
    expand_on_device_many,
    host_topk_rows,
    mirror_layer,
    seed_from_topk,
    seed_on_device,
    seed_on_device_many,
    step_after_layer,
)

logger = logging.getLogger(__name__)

OPT_INGEST_PHASES = (
    "opt_ingest_plan",
    "opt_ingest_alloc",
    "opt_ingest_batch",
    "opt_ingest_forward",
)


class OptPhaseClock:
    """Sequential host timer. Each lap covers the time since the previous one."""

    def __init__(self):
        self._mark = time.perf_counter()
        self.parts = {}

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.parts[name] = self.parts.get(name, 0.0) + (now - self._mark)
        self._mark = now


def opt_other_seconds(elapsed, ingest, grow) -> float:
    """Host time in one draft step that is neither ingest nor tree growth."""
    return max(0.0, float(elapsed) - float(ingest) - float(grow))


def flush_opt_ingest_phases(metrics, parts, wall: float) -> None:
    """Publish plan, alloc, batch, and forward so they sum to ``wall``.

    Unaccounted host time is added to plan. Time past ``wall`` is clipped
    from forward first. ``opt_ingest`` is that sum, including the synchronize
    the caller already added to forward.
    """
    if metrics is None or not getattr(metrics, "active", False):
        return
    wall = max(0.0, float(wall))
    amounts = {
        name: max(0.0, float(parts.get(name, 0.0) or 0.0)) for name in OPT_INGEST_PHASES
    }
    gap = wall - sum(amounts.values())
    if gap >= 0.0:
        amounts["opt_ingest_plan"] += gap
    else:
        rest = -gap
        for name in OPT_INGEST_PHASES[::-1]:
            take = min(amounts[name], rest)
            amounts[name] -= take
            rest -= take
            if rest <= 1e-12:
                break
    for name in OPT_INGEST_PHASES:
        if amounts[name] > 0.0:
            metrics.add_host(name, amounts[name])
    metrics.add_host("opt_ingest", sum(amounts[name] for name in OPT_INGEST_PHASES))


def _phase_clock(phases: Optional["OptPhaseClock"]) -> "OptPhaseClock":
    return phases if phases is not None else OptPhaseClock()


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


def layer_attention_mask(
    prefix_len: int, ancestor: torch.Tensor, device
) -> torch.Tensor:
    """Flattened TARGET_VERIFY mask: committed prefix, then the ancestor mask.

    ``ancestor`` is ``[N, depth * N]`` with 1 for a visible tree column.
    Every query sees the whole committed prefix. The prefix of ones is
    allocated on ``device``. Only the tree columns are copied.
    """
    if ancestor.dim() != 2:
        raise ValueError(
            f"ancestor mask must be [N, cols], got {tuple(ancestor.shape)}"
        )
    n, cols = ancestor.shape
    prefix_len = int(prefix_len)
    out = torch.ones((n, prefix_len + cols), dtype=torch.bool, device=device)
    out[:, prefix_len:].copy_(ancestor.to(dtype=torch.bool), non_blocking=True)
    return out.reshape(-1)


@dataclass(frozen=True)
class CascadeRows:
    """One forward's cascade inputs. Nothing here is reused by the next layer."""

    prefix_lens: list
    width: int
    tree_mask: torch.Tensor
    prefix_rows: Optional[torch.Tensor] = None


def _model_runner_of(scheduler):
    worker = getattr(scheduler, "model_worker", None)
    return getattr(worker, "model_runner", None)


def _cascade_on(runner) -> bool:
    from sglang.srt.speculative.clear_tree_remote.clear_tree_attention import (
        cascade_enabled,
    )

    scheduler = getattr(runner, "scheduler", None)
    return cascade_enabled(_model_runner_of(scheduler))


def layer_verify_mask(runner, ancestor, device) -> torch.Tensor:
    """Mask stored on the verify spec.

    Cascade attention reads ancestor columns from the static buffers, so the
    prefix of ones is not allocated. The ordinary kernel still needs it.
    """
    if _cascade_on(runner):
        return ancestor.reshape(-1).to(device=device, dtype=torch.bool)
    return layer_attention_mask(runner.prefix_len, ancestor, device)


def bind_cascade(
    spec,
    runners,
    ancestors,
    width: int,
    prefix_rows=None,
    cached_lens=None,
) -> None:
    """Attach this forward's rows. Slots are already in ``req_to_token``."""
    if not runners:
        return
    from sglang.srt.speculative.clear_tree_remote.clear_tree_attention import (
        ClearTreeAttention,
        cascade_enabled,
    )

    scheduler = runners[0].scheduler
    if not cascade_enabled(_model_runner_of(scheduler)):
        return
    width = int(width)
    if cached_lens is None:
        cached_lens = [int(runner.cached_len) for runner in runners]
    if len(ancestors) != len(runners) or len(cached_lens) != len(runners):
        raise RuntimeError(
            f"cascade bind has {len(runners)} runners, {len(ancestors)} masks, "
            f"and {len(cached_lens)} lengths"
        )
    device = scheduler.req_to_token_pool.device
    pieces = []
    for runner, ancestor, cached in zip(runners, ancestors, cached_lens):
        if ancestor is None or ancestor.dim() != 2 or int(ancestor.shape[0]) != width:
            shape = None if ancestor is None else tuple(ancestor.shape)
            raise RuntimeError(f"cascade tree mask {shape} != [{width}, cols]")
        expect = int(cached) - int(runner.prefix_len) + width
        if int(ancestor.shape[1]) != expect:
            raise RuntimeError(
                f"cascade tree columns {int(ancestor.shape[1])} != "
                f"{int(cached)} - {int(runner.prefix_len)} + {width}"
            )
        pieces.append(ancestor.to(device=device, dtype=torch.uint8))
    max_cols = max(int(piece.shape[1]) for piece in pieces)
    tree = torch.zeros((len(runners) * width, max_cols), dtype=torch.uint8, device=device)
    for index, piece in enumerate(pieces):
        tree[index * width : (index + 1) * width, : int(piece.shape[1])].copy_(piece)
    rows = None
    if prefix_rows is not None:
        rows = prefix_rows.to(device=device, dtype=torch.int32).reshape(-1)
        if int(rows.shape[0]) != len(runners) * width:
            raise RuntimeError(
                f"cascade prefix rows {int(rows.shape[0])} != {len(runners) * width}"
            )
    spec.target_verify_attention = ClearTreeAttention(
        CascadeRows(
            [int(runner.prefix_len) for runner in runners],
            width,
            tree,
            rows,
        )
    )


def pack_verify_attention(
    prefix_lens: Sequence[int],
    ancestors: Sequence[torch.Tensor],
    depth: int,
    width: int,
    device,
    model_runner=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenate per-request TARGET_VERIFY masks and positions on ``device``.

    With cascade on, each mask is the ancestor columns only. Otherwise mask
    segment ``b`` is ``[N, prefix_lens[b] + depth * N]`` flattened. Positions
    for one request are identical.
    """
    from sglang.srt.speculative.clear_tree_remote.clear_tree_attention import (
        cascade_enabled,
    )

    if len(prefix_lens) != len(ancestors):
        raise ValueError(
            f"OPT verify pack has {len(prefix_lens)} prefixes and "
            f"{len(ancestors)} masks"
        )
    if cascade_enabled(model_runner):
        masks = [
            ancestor.reshape(-1).to(device=device, dtype=torch.bool)
            for ancestor in ancestors
        ]
    else:
        masks = [
            layer_attention_mask(prefix, ancestor, device)
            for prefix, ancestor in zip(prefix_lens, ancestors)
        ]
    positions = [
        layer_positions(prefix, depth, width).to(device=device)
        for prefix in prefix_lens
    ]
    if not masks:
        return (
            torch.empty(0, dtype=torch.bool, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )
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


def plan_opt_tail(req, chunked_prefill_size, page_size: int = 1) -> str:
    """How OPT writes the committed tail.

    ``reprefill`` when the KV prefix is missing or does not cover the prompt.
    ``noop`` when that prefix already covers every committed token.
    ``decode`` when page size is 1 and one or two tokens are missing, so the
    step can replay the captured ntpb=1 DECODE graph. ``extend`` when the
    missing output fits in one chunk. ``extend_chunked`` when it does not.
    Page size other than 1 keeps the extend path for a short tail.
    """
    origin = getattr(req, "origin_input_ids", None) or []
    output = getattr(req, "output_ids", None) or []
    kv_len = int(getattr(req, "kv_committed_len", 0) or 0)
    if getattr(req, "req_pool_idx", None) is None or kv_len < len(origin):
        return "reprefill"
    missing = len(origin) + len(output) - kv_len
    if missing <= 0:
        return "noop"
    if int(page_size) == 1 and missing <= 2:
        return "decode"
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


def _rollback_failed_ingest(scheduler, req, start: int) -> None:
    """Undo an ingest that raised. A poisoned device keeps its slots.

    ``_clear_rollback`` refuses on a poisoned device; calling it here would
    replace the ingest error the caller is about to re-raise.
    """
    if getattr(scheduler, "_sr_device_poisoned", False):
        return
    scheduler._clear_rollback(req, start)


def clear_prefill_tail_logits(req) -> None:
    if getattr(req, "opt_prefill_tail_logits", None) is not None:
        req.opt_prefill_tail_logits = None


def reprefill_tail_logits(scheduler, reqs) -> List[torch.Tensor]:
    """Rebuild an invalid prefix and keep the finishing chunk's last row.

    The prefill callback is skipped when sampling would rewrite logits.
    Those requests fall back to ``_logits_at_tail_batch``, which scores the
    committed tail with raw logits.
    """

    def _keep(batch, logits) -> None:
        stash_finished_prefill_logits(scheduler, batch, logits)

    scheduler._sr_reprefill_committed(
        reqs,
        on_extend_logits=_keep,
        logits_survive=raw_logits_survive_sampling,
    )
    rows: List[Optional[torch.Tensor]] = [None] * len(reqs)
    missing = []
    for i, req in enumerate(reqs):
        row = take_cached_tail_logits(req, _committed_len(req))
        if row is None:
            missing.append(i)
            continue
        rows[i] = row
    if missing:

        class _Tail:
            def __init__(self, req, prefix_len):
                self.req = req
                self.prefix_len = prefix_len

        runners = [_Tail(reqs[i], _committed_len(reqs[i])) for i in missing]
        for i, row in zip(missing, _logits_at_tail_batch(scheduler, runners)):
            rows[i] = row
    if any(row is None for row in rows):
        missing_rids = [
            getattr(reqs[i], "rid", None) for i, row in enumerate(rows) if row is None
        ]
        raise RuntimeError(
            f"OPT draft reprefill returned no tail logits for {missing_rids}"
        )
    return rows


def materialize_missing_prefix(
    scheduler, reqs, capture_tail_logits: bool = False
) -> None:
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
        scheduler._sr_materialize_prefix_batch(
            missing,
            on_extend_logits=_keep,
            logits_survive=raw_logits_survive_sampling,
        )
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

    A request whose committed length is past its ``req_to_token`` row gets
    an empty draft and is not materialized: its prefix map would spill into
    the next row. The committed length only grows, so it stays empty.
    """
    if not pairs:
        return []
    row = _req_row_width(scheduler)
    short = set()
    if row is not None:
        short = {
            index
            for index, (_dreq, req) in enumerate(pairs)
            if _committed_len(req) > row
        }
    if not short:
        return _produce_fitting_windows(scheduler, action, pairs)
    from sglang.srt.speculative.standalone_remote.sr_protocol import SRAction

    _opt_draft_count(scheduler, "opt_row_short_empty", len(short))
    _drop_row_short(
        scheduler,
        [pairs[index][1] for index in sorted(short)],
        row,
        step=action == SRAction.STEP,
    )
    fitting = [pair for index, pair in enumerate(pairs) if index not in short]
    produced = list(_produce_fitting_windows(scheduler, action, fitting))
    if len(produced) < len(fitting):
        # Without row-short requests the caller's zip would have left these
        # replies empty; keep that instead of failing the whole packet.
        logger.warning(
            "[OPT] %d of %d fitting draft requests got no window; sending empty drafts",
            len(fitting) - len(produced),
            len(fitting),
        )
        produced.extend([([], None, None)] * (len(fitting) - len(produced)))
    produced_iter = iter(produced)
    return [
        ([], None, None) if index in short else next(produced_iter)
        for index in range(len(pairs))
    ]


# Rids already warned about; a long request stays short for every later STEP.
_ROW_SHORT_WARNED_MAX = 4096


def _drop_row_short(scheduler, reqs, row: int, step: bool) -> None:
    """Free what a row-short request still holds; on STEP, fix the ledger round.

    Such a request never reaches ``consume_clear_tree_lease`` again, so the
    tree lease and bonus guesses of its last grown round are freed here, the
    same way a misaligned lease is. Its committed prefix KV stays until the
    request finishes. Its cached lookahead decode batch is dropped too; that
    is host state, not KV, so it does not wait for a clean device. On STEP
    the open ledger round is this packet's round,
    so these rids leave it (``mark_round_row_short``) and the round is
    counted in ``opt_row_short_rounds``. A PREFILL only adds a gap to the
    previous round and does not touch it.
    """
    from sglang.srt.speculative.clear_tree_remote.clear_tree_lease import (
        opt_kv_release_allowed,
        release_clear_tree_lease,
        release_opt_bonus_guesses,
    )
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        mark_round_row_short,
    )

    release = opt_kv_release_allowed(scheduler)
    warned = getattr(scheduler, "_clear_row_short_warned", None)
    if not isinstance(warned, set) or len(warned) > _ROW_SHORT_WARNED_MAX:
        warned = set()
        scheduler._clear_row_short_warned = warned
    for req in reqs:
        clear_prefill_tail_logits(req)
        if release:
            release_clear_tree_lease(scheduler, req)
            release_opt_bonus_guesses(scheduler, req)
        rid = getattr(req, "rid", None)
        drop_lookahead_request(scheduler, rid)
        if rid in warned:
            continue
        warned.add(rid)
        logger.warning(
            "[OPT] draft %s committed length %d exceeds req_to_token width %d; "
            "it gets no Draft KV and no drafts",
            rid,
            _committed_len(req),
            row,
        )
    if step:
        _opt_draft_count(scheduler, "opt_row_short_rounds")
        mark_round_row_short(
            scheduler, [str(getattr(req, "rid", "")) for req in reqs]
        )


def _produce_fitting_windows(scheduler, action, pairs) -> list:
    """``produce_opt_windows`` for requests whose committed length fits a row."""
    from sglang.srt.speculative.standalone_remote.sr_protocol import SRAction

    if not pairs:
        return []
    reqs = [req for _, req in pairs]
    if action == SRAction.PREFILL:
        materialize_missing_prefix(scheduler, reqs)
        return [([], None, None)] * len(pairs)
    if action != SRAction.STEP:
        return [([], None, None)] * len(pairs)
    prefill_started = time.perf_counter()
    materialize_missing_prefix(scheduler, reqs, capture_tail_logits=True)
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        add_draft_prefill,
    )

    add_draft_prefill(scheduler, time.perf_counter() - prefill_started)
    scheduler._sr_replay_grammars(reqs)
    if len(reqs) <= 1:
        return [scheduler._clear_expand_one(req) for req in reqs]
    from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
        get_sr_round_metrics,
    )

    if getattr(scheduler, "_sr_device_poisoned", False):
        # Same guard as ``_clear_expand_one``: no lease consume, rollback or
        # tree growth on a poisoned device.
        for req in reqs:
            clear_prefill_tail_logits(req)
        get_sr_round_metrics(scheduler, "Draft").counts[
            "opt_expand_skipped_poisoned"
        ] += len(reqs)
        return [([], None, None)] * len(reqs)

    windows = [None] * len(reqs)
    batch_failed = False
    try:
        windows = list(grow_batch(scheduler, reqs))
    except Exception as exc:
        _reraise_device_context(scheduler, exc)
        if isinstance(exc, LookaheadUnsettled) or getattr(
            scheduler, "_sr_device_poisoned", False
        ):
            # Quarantined slots are still mapped on these requests. A
            # per-request rebuild would roll them back and grow on a
            # poisoned device; send empty drafts instead.
            logger.warning("[OPT] batch tree expand failed, no rebuild: %s", exc)
            get_sr_round_metrics(scheduler, "Draft").counts["opt_batch_unsettled"] += 1
            return [([], None, None)] * len(reqs)
        logger.warning("[OPT] batch tree expand failed, falling back: %s", exc)
        batch_failed = True
    if batch_failed or any(window is None for window in windows):
        get_sr_round_metrics(scheduler, "Draft").counts["opt_batch_fallbacks"] += 1
        from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import (
            install_fallback_plan,
        )

        install_fallback_plan(scheduler, reqs)
    for i, window in enumerate(windows):
        if window is None:
            windows[i] = scheduler._clear_expand_one(reqs[i])
    return windows


def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    """Temperature-1 full-vocabulary probabilities. No top-N renormalization."""
    return torch.softmax(logits.detach().float(), dim=-1)


def grow_with_forwards(
    config: ClearTreeConfig,
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

    return grow_clear_tree(config, next_probs)


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
            raise RuntimeError(f"OPT layer returned {len(rows)} rows for {bs} requests")
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


class _LayerTiming:
    """One metered verify forward. Absent when round metrics are off.

    Per ``opt_layer_forwards``, read the counters as:

    - ``opt_layer_host_us`` is preparation before ``_run``.
    - ``opt_layer_launch_us`` is the wall clock from ``_mark_layer_launch``
      until the layer forward returns. The lookahead template path calls
      ``model_runner.forward`` and does not enter ``ForwardBatch.init_new``.
      The ``run_batch`` fallback still includes the wait inside ``init_new``.
      Replaying the same ``r{N}_1`` graph returns once that wait is gone.
    - ``opt_lookahead_launch_wait_us`` stays zero unless
      ``SGLANG_CLEAR_PROBE_LAYER_WAIT`` is set. The probe synchronizes the
      stream before ``_run`` and records that stall here, outside ``host_us``
      and ``launch_us``. That schedule is not the online path: the host waits
      for the previous layer to finish before starting ``_run``.
    - ``opt_layer_gpu_us`` is ``ev0.elapsed_time(ev1)``. ``ev0`` is recorded at
      the start of the ``_run`` window, so the interval includes host
      preparation while the stream is idle. It is not GPU-busy time. Confirm
      busy and idle with nsys; do not subtract ``launch_us``.
    - ``opt_layer_forwards`` counts every queued ``_forward_verify_batch``
      and every lookahead ``launch_layer``, including the last one lookahead
      submits early. Requests that share a depth and fit in one forward count
      once. A KV-budget split counts once per sub-batch.
      ``opt_layer_forwards_critical`` is the lookahead launch count minus
      one, matching the old path's pre-reply forwards.
      ``opt_layer_forwards_deferred`` counts each finished
      ``_forward_verify_batch`` that ``forward_deferred_final_layers`` runs
      after the reply. A depth group that fits in one forward still counts
      once. A group split by the KV budget counts once per sub-batch. A
      lookahead layer already in ``opt_layer_forwards`` is not counted again.
      A failed replay is not counted.
    - ``opt_ingest_gpu_us`` is each ingest ``_run_batch``. The start event is
      recorded immediately before that call, so host planning and allocation
      are outside the interval. Stream idle inside ``_run_batch`` can still
      be included. Chunked extend and a two-token decode gap each add one
      pair. The pairs are read in the same drain as the layer events. An
      unpaired start is dropped.
    - ``opt_layer_select_gpu_us`` is ``ev0.elapsed_time(ev1)`` around
      ``_vocab_topk``, ``expand_on_device``, and ``copy_expand`` on each
      non-final lookahead expansion. The seed top-k, the stopping bonus
      top-k, ``clone_cpu``, and ``launch_layer`` sit outside. A fallback
      drops pairs already queued. Divide by ``opt_layer_forwards_critical``.
      A zero denominator means the window did not finish a lookahead expand.

    Per-layer ``opt_layer_host_us``, ``opt_layer_launch_us``, and
    ``opt_layer_gpu_us`` use
    ``opt_layer_forwards + opt_layer_forwards_deferred`` as the denominator.
    ``opt_seed_d2h_wait_us`` is once per grow and is not divided by that sum.
    ``opt_layer_d2h_wait_us`` is the later ``.cpu()`` of each layer. Logits
    stay on device until that copy. Warmup samples include the first replay
    and are reference only.
    """

    __slots__ = (
        "metrics",
        "t0",
        "t_launch",
        "ev0",
        "ev1",
        "opened",
        "host_closed",
        "host_us",
        "ledger_meta",
        "ledger_only",
    )

    def __init__(self, metrics, t0, ev0, ev1, ledger_only=False):
        self.metrics = metrics
        self.t0 = t0
        self.t_launch = None
        self.ev0 = ev0
        self.ev1 = ev1
        self.opened = False
        self.host_closed = False
        self.host_us = None
        self.ledger_meta = None
        self.ledger_only = bool(ledger_only)


def _draft_metrics(scheduler):
    """Active Draft round metrics, or None on the unmetered path."""
    if scheduler is None:
        return None
    from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
        get_sr_round_metrics,
    )

    metrics = get_sr_round_metrics(scheduler, "Draft")
    if not getattr(metrics, "active", False):
        return None
    return metrics


class _SilentLayerCounts:
    """Drop host-counter writes when the ledger times a layer outside a metrics window."""

    def __init__(self) -> None:
        self.counts = defaultdict(int)


def _begin_layer_timing(scheduler):
    """Start the host clock. Event creation failure drops this sample only.

    An active round-metrics window keeps the existing counters. With the
    ledger on and that window off, events stay on a side list so the counters
    do not move.
    """
    metrics = _draft_metrics(scheduler)
    ledger_only = False
    if metrics is None:
        from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
            ledger_enabled,
        )

        if not ledger_enabled():
            return None
        metrics = _SilentLayerCounts()
        ledger_only = True
    try:
        factory = torch.get_device_module(scheduler.device).Event
        ev0 = factory(enable_timing=True)
        ev1 = factory(enable_timing=True)
    except Exception:
        return None
    return _LayerTiming(metrics, time.perf_counter(), ev0, ev1, ledger_only=ledger_only)


def _close_layer_host(timing) -> None:
    """Stop the preparation clock before ``_run``."""
    if timing is None or timing.host_closed:
        return
    now = time.perf_counter()
    timing.host_us = int((now - timing.t0) * 1e6)
    timing.metrics.counts["opt_layer_host_us"] += timing.host_us
    timing.host_closed = True


def _mark_layer_launch(timing) -> None:
    """Start the ``run_batch`` clock. Preparation is closed here if still open.

    A failed ``ev0`` leaves ``opened`` false, so that sample records host time
    and skips launch time.
    """
    if timing is None:
        return
    now = time.perf_counter()
    if not timing.host_closed:
        timing.host_us = int((now - timing.t0) * 1e6)
        timing.metrics.counts["opt_layer_host_us"] += timing.host_us
        timing.host_closed = True
    timing.t_launch = now
    try:
        timing.ev0.record()
    except Exception:
        return
    timing.opened = True


def _finish_layer_timing(scheduler, timing) -> None:
    """Queue the GPU interval. A failed replay never reaches this.

    ``opt_layer_launch_us`` is the wall clock of ``_run`` and stops before
    ``ev1``. An explicit previous-layer wait is recorded only when
    ``SGLANG_CLEAR_PROBE_LAYER_WAIT`` is set, and it sits outside this window.
    ``ev0``..``ev1`` includes host time while the stream is idle; it is not
    GPU-busy time.
    """
    if timing is None or not timing.opened:
        return
    timing.metrics.counts["opt_layer_launch_us"] += int(
        (time.perf_counter() - timing.t_launch) * 1e6
    )
    try:
        timing.ev1.record()
    except Exception:
        return
    if not timing.ledger_only:
        events = getattr(scheduler, "_opt_layer_events", None)
        if events is None:
            events = []
            scheduler._opt_layer_events = events
        events.append((timing.ev0, timing.ev1))
    if timing.ledger_meta is not None:
        pairs = getattr(scheduler, "_clear_layer_pairs", None)
        if not isinstance(pairs, list):
            pairs = []
            scheduler._clear_layer_pairs = pairs
        pairs.append((timing.ev0, timing.ev1, timing.ledger_meta))


def _queue_ledger_layer(scheduler, timing, runners, layer: int, width: int) -> None:
    """One layer sub-batch for the shadow ledger. No extra device sync."""
    if timing is None:
        return
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        queue_layer_sample,
    )

    timing.ledger_meta = queue_layer_sample(
        scheduler,
        layer=int(layer),
        runners=runners,
        host_us=getattr(timing, "host_us", None),
        width=width,
        t0=getattr(timing, "t0", None),
    )


def _round_width(scheduler) -> int:
    """Layer width for this round. The flat table wins once a plan exists."""
    plan = getattr(scheduler, "_clear_round_plan", None)
    if plan is not None:
        return int(plan.width)
    return int(scheduler.server_args.clear_tree_width)


def _require_layer_width(n: int) -> None:
    """Reject a layer whose token count has no captured graph."""
    from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
        LAYER_WIDTHS,
        layer_width_supported,
    )

    if layer_width_supported(n):
        return
    raise RuntimeError(
        f"OPT layer has {n} tokens but the captured layer graphs are "
        f"{list(LAYER_WIDTHS)}"
    )


def critical_layer_forwards(launched: int) -> int:
    """Layers on the pre-reply path. The early final launch is excluded."""
    return max(int(launched) - 1, 0)


def _begin_ingest_gpu(scheduler) -> None:
    """Record the start of one ingest ``_run_batch``. No host synchronize.

    Host planning and batch setup sit before this call. A previous open
    start is dropped so a missed finish cannot pair with the next forward.
    """
    scheduler._opt_ingest_open = None
    if _draft_metrics(scheduler) is None:
        return
    try:
        event = torch.get_device_module(scheduler.device).Event(enable_timing=True)
        event.record()
    except Exception:
        return
    scheduler._opt_ingest_open = event


def _finish_ingest_gpu(scheduler) -> None:
    """Close the open ingest interval and keep the pair. No host synchronize."""
    start = getattr(scheduler, "_opt_ingest_open", None)
    scheduler._opt_ingest_open = None
    if start is None:
        return
    try:
        end = torch.get_device_module(scheduler.device).Event(enable_timing=True)
        end.record()
    except Exception:
        return
    pairs = getattr(scheduler, "_opt_ingest_events", None)
    if not isinstance(pairs, list):
        pairs = []
        scheduler._opt_ingest_events = pairs
    pairs.append((start, end))


def _note_lookahead_launch_wait(runner) -> None:
    """Optionally time an explicit wait for the previous layer.

    Default does not synchronize. The template path does not call
    ``init_new``. A missing template falls back to ``run_batch``, and that
    path still waits inside ``ForwardBatch.init_new``.
    ``SGLANG_CLEAR_PROBE_LAYER_WAIT`` inserts ``current_stream().synchronize()``
    before ``_run`` and records that stall in
    ``opt_lookahead_launch_wait_us``. The probe runs only when round metrics
    are active and a previous layer is already queued. That path is not the
    online schedule. The first layer waits for ingest, not for another layer.
    """
    if not getattr(runner, "_lookahead_queued", False):
        return
    if not opt_probe_layer_wait_enabled():
        return
    scheduler = runner.scheduler
    metrics = _draft_metrics(scheduler)
    if metrics is None:
        return
    started = time.perf_counter()
    try:
        torch.get_device_module(scheduler.device).current_stream().synchronize()
    except Exception as exc:
        _reraise_device_context(scheduler, exc)
        return
    metrics.counts["opt_lookahead_launch_wait_us"] += int(
        (time.perf_counter() - started) * 1e6
    )


def _begin_select_timing(scheduler):
    """Start one expand top-k interval. Absent when round metrics are off."""
    if _draft_metrics(scheduler) is None:
        return None
    try:
        factory = torch.get_device_module(scheduler.device).Event
        pair = (factory(enable_timing=True), factory(enable_timing=True))
        pair[0].record()
    except Exception:
        return None
    return pair


def _finish_select_timing(scheduler, pair) -> None:
    """Queue the expand interval. A raise before this leaves it uncounted."""
    if pair is None:
        return
    try:
        pair[1].record()
    except Exception:
        return
    events = getattr(scheduler, "_opt_select_events", None)
    if not isinstance(events, list):
        events = []
        scheduler._opt_select_events = events
    events.append(pair)


def drain_opt_layer_gpu_us(scheduler, metrics) -> None:
    """Read queued verify, ingest, and select events once.

    An empty queue does not synchronize. A poisoned device drops the events.
    ``elapsed_time`` is milliseconds. Other timing errors increment
    ``device_timing_dropped_opt_layer`` and do not replace a business result
    unless the accelerator context is already dead.
    """
    events = list(getattr(scheduler, "_opt_layer_events", None) or [])
    ingest = getattr(scheduler, "_opt_ingest_events", None)
    select = getattr(scheduler, "_opt_select_events", None)
    ledger_pairs = list(getattr(scheduler, "_clear_layer_pairs", None) or [])
    scheduler._opt_layer_events = []
    scheduler._opt_ingest_events = []
    scheduler._opt_ingest_open = None
    scheduler._opt_select_events = []
    scheduler._clear_layer_pairs = []
    pairs = [
        pair
        for pair in (ingest if isinstance(ingest, list) else [])
        if isinstance(pair, tuple) and len(pair) == 2
    ]
    select_pairs = [
        pair
        for pair in (select if isinstance(select, list) else [])
        if isinstance(pair, tuple) and len(pair) == 2
    ]
    if not events and not pairs and not select_pairs and not ledger_pairs:
        return
    if metrics is None or getattr(scheduler, "_sr_device_poisoned", False):
        return
    from sglang.srt.speculative.standalone_remote.sr_align import (
        is_device_context_error,
    )

    try:
        torch.get_device_module(scheduler.device).synchronize()
    except Exception as exc:
        if is_device_context_error(exc):
            raise
        metrics.counts["device_timing_dropped_opt_layer"] += 1
        return
    for ev0, ev1 in events:
        try:
            metrics.counts["opt_layer_gpu_us"] += int(ev0.elapsed_time(ev1) * 1000)
        except Exception as exc:
            if is_device_context_error(exc):
                raise
            metrics.counts["device_timing_dropped_opt_layer"] += 1
    if ledger_pairs:
        from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
            remember_layer_gpu,
        )

        for ev0, ev1, sample in ledger_pairs:
            gpu_us = None
            try:
                gpu_us = int(ev0.elapsed_time(ev1) * 1000)
            except Exception as exc:
                if is_device_context_error(exc):
                    raise
                metrics.counts["device_timing_dropped_opt_layer"] += 1
            remember_layer_gpu(sample, gpu_us)
    for start, end in pairs:
        try:
            metrics.counts["opt_ingest_gpu_us"] += int(start.elapsed_time(end) * 1000)
        except Exception as exc:
            if is_device_context_error(exc):
                raise
            metrics.counts["device_timing_dropped_opt_layer"] += 1
    for ev0, ev1 in select_pairs:
        try:
            metrics.counts["opt_layer_select_gpu_us"] += int(
                ev0.elapsed_time(ev1) * 1000
            )
        except Exception as exc:
            if is_device_context_error(exc):
                raise
            metrics.counts["device_timing_dropped_opt_layer"] += 1


def _commit_prob_rows(
    states, indices, logits_rows, *, seed: bool, topk_out=None, metrics=None
):
    """Softmax each row, copy one stacked top-k, then update those trees.

    The copy keeps ``N + 1`` ids. The tree still uses the first ``N``. The
    extra id is the bonus once every verify child of that node is excluded.
    ``topk_out[index]`` receives the full id tensor: ``[N + 1]`` for the root
    and ``[N, N + 1]`` for a layer.

    ``metrics``, when active, times ``host_topk_rows``. That wall clock
    includes the top-k and the ``.cpu()`` wait for earlier GPU work. The seed
    copy is ``opt_seed_d2h_wait_us`` (once per grow). Each layer copy is
    ``opt_layer_d2h_wait_us``.
    """
    width = int(states[indices[0]].config.width)
    probs = [softmax_probs(row) for row in logits_rows]
    timed = metrics is not None and getattr(metrics, "active", False)
    t0 = time.perf_counter() if timed else None
    scores, ids = host_topk_rows(probs, width + 1, seed=seed)
    if t0 is not None:
        key = "opt_seed_d2h_wait_us" if seed else "opt_layer_d2h_wait_us"
        metrics.counts[key] += int((time.perf_counter() - t0) * 1e6)
    written = []
    touched = []
    for row, index in enumerate(indices):
        state = states[index]
        if state.finished:
            raise RuntimeError("update called after finish")
        if seed:
            seed_from_topk(state, scores[row][:width], ids[row][:width])
        else:
            expand_from_topk(state, scores[row][:, :width], ids[row][:, :width])
        if topk_out is not None:
            topk_out[index].append(ids[row])
        touched.append(state)
    decider = getattr(touched[0], "decider", None) if touched else None
    if decider is not None:
        decider.begin_layer(touched)
    for state in touched:
        written.append(step_after_layer(state))
    return written


def _reraise_device_context(scheduler, exc) -> None:
    """Poison and re-raise when the accelerator context is already dead.

    ``scheduler`` may be missing in a CPU unit test. The error still
    propagates so a later caller cannot turn it into an empty draft.
    """
    from sglang.srt.speculative.standalone_remote.sr_align import (
        is_device_context_error,
    )

    if not is_device_context_error(exc):
        return
    if scheduler is not None:
        scheduler._sr_device_poisoned = True
        scheduler._sr_clear_stamps_after_poison()
        logger.error("[OPT] device context error: %s", exc)
    raise


def _cut_deferred(plan, states, pending, results) -> None:
    """Cut every stopped tree to the one K chosen after growth has stopped."""
    if not pending:
        return
    if plan is None or not hasattr(plan, "freeze_k"):
        raise RuntimeError("deferred finish without a round plan")
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ops import finish

    keep = int(plan.freeze_k(states))
    for index in list(pending):
        verify = finish(states[index], keep)
        plan.note(states[index], keep)
        results[index] = (
            [int(token) for token in verify.tokens.tolist()],
            [int(parent) for parent in verify.parents.tolist()],
            [int(col) for col in verify.cols.tolist()],
        )
        pending.remove(index)


class RowOverflow(RuntimeError):
    """A layer would write past its request's ``req_to_token`` row."""


def _req_row_width(scheduler) -> Optional[int]:
    """Columns in one ``req_to_token`` row, or None when there is no pool."""
    pool = getattr(getattr(scheduler, "req_to_token_pool", None), "req_to_token", None)
    shape = getattr(pool, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    return int(shape[1])


def row_layer_cap(scheduler, cached_len: int, width: int) -> Optional[int]:
    """Whole ``width``-token layers that still fit after ``cached_len``.

    The map write (``assign_req_to_token_pool``) has no column bound: a column
    past the row lands in the next request's row. None when there is no pool.
    """
    row = _req_row_width(scheduler)
    if row is None:
        return None
    return max((row - int(cached_len)) // max(int(width), 1), 0)


def _require_row_room(scheduler, runners, width: int, starts=None) -> None:
    """Raise ``RowOverflow`` before any slot is allocated for a short row."""
    row = _req_row_width(scheduler)
    if row is None:
        return
    if starts is None:
        starts = [int(runner.cached_len) for runner in runners]
    for runner, start in zip(runners, starts):
        if int(start) + int(width) > row:
            _opt_draft_count(scheduler, "opt_row_overflow_blocked")
            raise RowOverflow(
                f"OPT draft {getattr(runner.req, 'rid', None)} columns "
                f"[{int(start)}, {int(start) + int(width)}) exceed req_to_token "
                f"width {row}"
            )


def row_max_depths(scheduler, runners, config: ClearTreeConfig) -> Optional[list]:
    """Per tree ``max_depth`` so every layer this round writes fits its row.

    A tree with final depth D writes at most D layers: lookahead launches
    layer d before judging d, and the ordinary path forwards the last layer
    after the reply when KV reuse defers it. ``max_depth`` M stops growth at
    depth M - 1, so M = cap + 1 keeps D <= cap. M is at least 2 (a seed-only
    tree); with cap 0 that tree writes nothing before the reply, admission
    keeps it off lookahead and the deferred forward skips it. None entries
    leave the round's config. Returns None when no tree is capped. Nothing
    is counted here: a lookahead fallback asks again for the same round, so
    ``count_row_capped`` counts the trees the round finally kept.
    """
    if _req_row_width(scheduler) is None:
        return None
    caps = []
    for runner in runners:
        cap = row_layer_cap(scheduler, int(runner.prefix_len), int(config.width))
        if cap is None or cap + 1 >= int(config.max_depth):
            caps.append(None)
            continue
        caps.append(max(cap + 1, 2))
    if all(cap is None for cap in caps):
        return None
    return caps


def _apply_depth_cap(state, config: ClearTreeConfig, cap) -> None:
    """Give one tree a lower ``max_depth``; its tensors keep the round's size.

    ``row_capped`` marks the tree for ``count_row_capped`` and for the width
    replay, which must not compare widths on a tree the row cut short.
    """
    if cap is None or int(cap) >= int(config.max_depth):
        return
    state.config = replace(config, max_depth=int(cap))
    state.row_capped = True


def count_row_capped(scheduler, states, windows=None) -> None:
    """Count this round's row-capped trees once, after growth is final.

    With ``windows``, only trees whose window survived count: a request whose
    window is None is rebuilt by ``_clear_expand_one``, which counts its own
    tree.
    """
    if windows is not None:
        states = [
            state
            for index, state in enumerate(states)
            if index < len(windows) and windows[index] is not None
        ]
    capped = sum(1 for state in states if getattr(state, "row_capped", False))
    if capped:
        _opt_draft_count(scheduler, "opt_row_depth_capped", capped)


def _round_states(scheduler, config: ClearTreeConfig, count: int, depth_caps=None):
    """One tree per request, bound to this round's plan, rid and prefix.

    ``depth_caps[i]``, when not None, is tree ``i``'s ``max_depth``
    (``row_max_depths``).
    """
    states = [ClearTreeState.create(config) for _ in range(int(count))]
    for index, state in enumerate(states):
        if depth_caps is not None and index < len(depth_caps):
            _apply_depth_cap(state, config, depth_caps[index])
    plan = (
        getattr(scheduler, "_clear_round_plan", None) if scheduler is not None else None
    )
    rids = (
        list(getattr(scheduler, "_clear_round_rids", None) or [])
        if scheduler is not None
        else []
    )
    prefixes = (
        list(getattr(scheduler, "_clear_round_prefixes", None) or [])
        if scheduler is not None
        else []
    )
    for index, state in enumerate(states):
        state.decider = plan
        if index < len(rids):
            state.rid = rids[index]
        if index < len(prefixes):
            state.prefix = prefixes[index]
    if plan is not None and hasattr(plan, "bind_states"):
        plan.bind_states(states)
    return states, plan


def grow_active_layers(
    config: ClearTreeConfig,
    tail_logits,
    forward_active: Callable,
    scheduler=None,
    states_out=None,
    topk_out=None,
    depth_caps=None,
):
    """Grow one tree per tail. Requests still at one depth share a forward.

    ``forward_active(steps, indices)`` sees only non-final steps. ``indices``
    are positions into ``tail_logits``. The return value is a list of
    ``[N, vocab]`` logits, or a tensor of shape ``[bs, N, vocab]`` or
    ``[bs * N, vocab]``. A request that stops is absent from the next call.
    Each request is softmaxed on its own. One layer's top-k is copied back
    once, then each tree is updated on CPU.

    ``states_out``, when given, receives each tree in tail order before any
    forward. The last layer is present and is not forwarded.
    ``topk_out[i]`` collects that tail's host id tensors, root first.
    ``depth_caps`` lowers single trees' ``max_depth`` (``row_max_depths``).
    """
    tails = list(tail_logits)
    states, plan = _round_states(scheduler, config, len(tails), depth_caps)
    if states_out is not None:
        states_out.extend(states)
    steps = [None] * len(tails)
    results = [None] * len(tails)
    pending: List[int] = []
    active: List[int] = []
    metrics = _draft_metrics(scheduler)
    if tails:
        seeded = _commit_prob_rows(
            states,
            list(range(len(tails))),
            tails,
            seed=True,
            topk_out=topk_out,
            metrics=metrics,
        )
        for i, step in enumerate(seeded):
            steps[i] = step
            if step.is_final:
                if step.verify is None:
                    pending.append(i)
                else:
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
        try:
            logits = forward_active(chosen, list(active))
        except Exception as exc:
            _reraise_device_context(scheduler, exc)
            logger.warning(
                "[OPT] layer forward failed, keeping finished trees: %s", exc
            )
            try:
                _cut_deferred(plan, states, pending, results)
            except Exception as cut_exc:
                logger.warning("[OPT] deferred window cut failed: %s", cut_exc)
            return results
        rows = _as_layer_rows(logits, len(active), config.width)
        nxt: List[int] = []
        for i, step in zip(
            active,
            _commit_prob_rows(
                states, active, rows, seed=False, topk_out=topk_out, metrics=metrics
            ),
        ):
            steps[i] = step
            if step.is_final:
                if step.verify is None:
                    pending.append(i)
                else:
                    results[i] = _verify_window(step)
            else:
                nxt.append(i)
        active = nxt
    if scheduler is not None:
        from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
            mark_stop,
        )

        mark_stop(scheduler)
    _cut_deferred(plan, states, pending, results)
    if any(item is None for item in results):
        raise RuntimeError("OPT batch finished with a missing verify set")
    return results


@dataclass
class ClearLayerSpec:
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
    # Earlier layers sit in the KV prefix, so every TARGET_VERIFY reads that
    # part of the mask. ``is_bonus`` only distinguishes a bonus forward.
    apply_prefix_custom_mask: bool = True
    is_bonus: bool = False
    # ClearTree cascade only. Other algorithms leave this unset.
    target_verify_attention: object = None
    # A locally rejected graph combination must not be readmitted downstream.
    bonus_eager: bool = False

    def is_draft_input(self) -> bool:
        return False

    def get_spec_adjusted_global_num_tokens(self, batch):
        return batch.global_num_tokens, batch.global_num_tokens_for_logprob


def opt_bonus_graph_extra(spec_info):
    """Bonus and layer forwards share unsuffixed ``r{G}_{bs}`` graphs."""
    del spec_info
    return None


def opt_draft_bonus_ntpbs(server_args) -> list:
    """Opt-in Draft bonus graphs, each one wider than a verify K tier."""
    if (
        getattr(server_args, "clear_tree_role", None) != "draft"
        or not bonus_prediction_enabled()
    ):
        return []
    from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
        bonus_ntpbs,
    )

    return list(bonus_ntpbs())


def opt_draft_bonus_ntpb(server_args):
    """Widest bonus graph, or None when this role has none.

    Each bonus graph is one wider than a verify K tier.
    """
    widths = opt_draft_bonus_ntpbs(server_args)
    return max(widths) if widths else None


def opt_draft_capture_extra_keys(server_args, ntpb) -> list:
    """One graph per (bs, width); a width shared by layer and bonus is captured once."""
    del server_args, ntpb
    return [None]


def opt_layer_verify_graph_matches(spec_info, actual_ntpb: int) -> bool:
    """True when this spec may replay a TARGET_VERIFY graph of this width.

    The captured ntpb must equal this spec's query width. The ntpb=1 dual
    graph is DECODE, so width 1 stays eager. A bonus spec uses its group width.
    """
    if not isinstance(spec_info, ClearLayerSpec) or getattr(
        spec_info, "bonus_eager", False
    ):
        return False
    width = int(getattr(spec_info, "num_tokens_per_req", 0) or 0)
    ntpb = int(actual_ntpb)
    return ntpb != 1 and width == ntpb


class OptLayerForward:
    """Tail logits plus one N-token extend per OPT layer. Batch size is 1."""

    def __init__(self, scheduler, req, prefix_len: int):
        self.scheduler = scheduler
        self.req = req
        self.prefix_len = int(prefix_len)
        self.cached_len = int(prefix_len)
        self.layer_slots: List[torch.Tensor] = []
        self.bonus_ids: List[torch.Tensor] = []

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
            _rollback_failed_ingest(self.scheduler, self.req, start)
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
        _require_layer_width(n)
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
        timing = _begin_layer_timing(self.scheduler)
        positions = layer_positions(self.prefix_len, depth, n).to(device)
        mask = layer_verify_mask(self, ancestor, device)
        out_cache_loc = self._alloc_layer_slots(n)
        self.req.kv_allocated_len = self.cached_len + n
        batch = self._verify_batch(tokens.to(device), positions, mask, out_cache_loc)
        bind_cascade(batch.spec_info, [self], [ancestor], n)
        _mark_layer_launch(timing)
        logits = self._run(batch)
        _queue_ledger_layer(self.scheduler, timing, [self], depth, n)
        _finish_layer_timing(self.scheduler, timing)
        logits = _row_logits(logits, n, tail=False)
        self.cached_len += n
        return logits

    def launch_layer(self, tokens_dev, anc_dev, depth: int, batch) -> torch.Tensor:
        """Queue one layer from tensors already on device.

        Device lengths and the pool index are ``fill_`` into buffers allocated
        while the GPU was idle. ``seq_lens_cpu`` is a new pageable CPU tensor
        each layer, so a later fill cannot overwrite an in-flight copy.
        This does not build a new ``SamplingBatchInfo``.
        """
        tokens = tokens_dev
        if not torch.is_tensor(tokens):
            raise RuntimeError("lookahead tokens must already be a tensor")
        n = int(tokens.shape[0])
        _require_layer_width(n)
        depth = int(depth)
        ancestor = anc_dev
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
        timing = _begin_layer_timing(self.scheduler)
        self._la_positions.fill_(self.prefix_len + depth - 1)
        mask = layer_verify_mask(self, ancestor, device)
        out_cache_loc = self._alloc_lookahead_slots(n)
        self.req.kv_allocated_len = self.cached_len + n
        self._la_seq.fill_(self.cached_len)
        self._la_orig.fill_(self.cached_len)
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.global_forward_mode = ForwardMode.TARGET_VERIFY
        batch.input_ids = tokens.to(dtype=torch.int64).reshape(-1)
        batch.output_ids = None
        batch.seq_lens = self._la_seq
        batch.seq_lens_cpu = torch.tensor([self.cached_len], dtype=torch.int64)
        batch.orig_seq_lens = self._la_orig
        batch.seq_lens_sum = int(self.cached_len)
        batch.out_cache_loc = out_cache_loc
        batch.extend_lens = [n]
        batch.prefix_lens = [self.cached_len]
        batch.extend_num_tokens = n
        batch.extend_logprob_start_lens = [self.cached_len]
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        batch.spec_info = ClearLayerSpec(
            custom_mask=mask.reshape(-1),
            positions=self._la_positions.reshape(-1),
            draft_token_num=n,
            num_tokens_per_req=n,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        bind_cascade(batch.spec_info, [self], [ancestor], n)
        _close_layer_host(timing)
        _note_lookahead_launch_wait(self)
        _mark_layer_launch(timing)
        self._lookahead_queued = True
        logits = self._run_lookahead(batch)
        if self._lookahead_launch_event is not None:
            self._lookahead_launch_event.record()
        _queue_ledger_layer(self.scheduler, timing, [self], depth, n)
        _finish_layer_timing(self.scheduler, timing)
        logits = _row_logits(logits, n, tail=False)
        self._last_logits = logits
        self.cached_len += n
        _opt_draft_count(self.scheduler, "opt_lookahead_layers")
        return logits

    def _alloc_lookahead_slots(self, n: int) -> torch.Tensor:
        """Page-1 slot alloc using the prepared device index buffers."""
        from sglang.srt.mem_cache.common import alloc_token_slots
        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        scheduler = self.scheduler
        _require_row_room(scheduler, [self], n)
        self._la_seq.fill_(self.cached_len)
        out_cache_loc = alloc_token_slots(scheduler.tree_cache, n)
        try:
            if int(out_cache_loc.numel()) != n:
                raise RuntimeError(
                    f"OPT layer allocated {int(out_cache_loc.numel())} slots for {n} tokens"
                )
            assign_req_to_token_pool_func(
                self._la_pool,
                scheduler.req_to_token_pool.req_to_token,
                self._la_seq,
                self._la_seq + n,
                out_cache_loc,
                1,
            )
        except BaseException as exc:
            _free_unrecorded_slots(scheduler, out_cache_loc, exc)
            raise
        # The map is written: cover it by rollback before anything else can fail.
        self.req.kv_allocated_len = int(self.cached_len) + n
        self.layer_slots.append(out_cache_loc.detach().reshape(-1).clone())
        return out_cache_loc

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
        _require_row_room(scheduler, [self], n, starts=[base])
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
        try:
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
        except BaseException as exc:
            if page == 1:
                _free_unrecorded_slots(scheduler, out_cache_loc, exc)
            raise
        # The map is written: cover it by rollback before anything else can fail.
        self.req.kv_allocated_len = base + n
        if start is None:
            # Ingest passes start= and must not become part of the tree lease.
            # Stay on the allocator device; publish copies every layer once.
            self.layer_slots.append(out_cache_loc.detach().reshape(-1).clone())
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
        batch.seq_lens = torch.tensor(
            [self.cached_len], dtype=torch.int64, device=device
        )
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
        batch.spec_info = ClearLayerSpec(
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

    def _run_lookahead(self, batch) -> torch.Tensor:
        """Replay from the admission template. A missing template uses ``run_batch``.

        ``model_runner.forward`` returns logits and does not sample. It also
        decides again whether this layer can replay a CUDA graph. A false
        ``can_run_graph`` means the layer already ran eager; count it and
        still return the logits. A forward that raises is not retried: the
        slots for this layer are already allocated, and a second launch
        would queue the same work again.
        """
        template = getattr(self, "_la_forward_template", None)
        delta = getattr(self, "_la_mrope_delta", None)
        if template is None or (
            getattr(template, "mrope_positions", None) is not None and delta is None
        ):
            _opt_draft_count(self.scheduler, "opt_lookahead_template_fallbacks")
            return self._run(batch)
        forward_batch = _refresh_lookahead_forward(template, batch, delta)
        out = self.scheduler.model_worker.model_runner.forward(forward_batch)
        from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
            remember_layer_path,
        )

        remember_layer_path(self.scheduler, out)
        if not bool(getattr(out, "can_run_graph", True)):
            _opt_draft_count(self.scheduler, "opt_lookahead_eager_layers")
        logits = getattr(getattr(out, "logits_output", None), "next_token_logits", None)
        if logits is None:
            raise RuntimeError("OPT lookahead forward returned no logits")
        return logits


def raw_logits_survive_sampling(batch) -> bool:
    """True when the sampler leaves next_token_logits untouched."""
    info = batch.sampling_info
    orch = getattr(info, "penalizer_orchestrator", None)
    return (
        bool(info.is_all_greedy)
        and not getattr(batch, "has_grammar", False)
        and getattr(info, "logit_bias", None) is None
        and not getattr(info, "has_custom_logit_processor", False)
        and not (orch is not None and orch.is_required)
    )


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
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        remember_layer_path,
    )

    remember_layer_path(scheduler, result)
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
    scheduler,
    runners,
    tokens,
    positions,
    custom_mask,
    out_cache_loc,
    cached,
    is_bonus: bool = False,
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
    batch.spec_info = ClearLayerSpec(
        custom_mask=custom_mask.reshape(-1),
        positions=positions.reshape(-1).to(dtype=torch.int64, device=device),
        draft_token_num=width,
        num_tokens_per_req=width,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        apply_prefix_custom_mask=True,
        is_bonus=bool(is_bonus),
    )
    _keep_raw_logits(batch)
    return batch


def _logits_at_tail_batch(scheduler, runners, phases=None) -> List[torch.Tensor]:
    """One DECODE row per request, reusing the committed tail slot.

    Each row is copied. A later DECODE replay in the same ingest writes the
    same graph output buffer and would otherwise replace these views.
    """
    phases = _phase_clock(phases)
    if not runners:
        phases.lap("opt_ingest_batch")
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
    phases.lap("opt_ingest_batch")
    _begin_ingest_gpu(scheduler)
    try:
        logits = _flatten_logits(_run_batch(scheduler, batch))
    finally:
        _finish_ingest_gpu(scheduler)
    phases.lap("opt_ingest_forward")
    if int(logits.shape[0]) != len(runners):
        raise RuntimeError(
            f"OPT tail logits rows {int(logits.shape[0])} != {len(runners)} requests"
        )
    return [logits[i].clone() for i in range(len(runners))]


def _extend_span_batch(scheduler, runners, ends, phases=None) -> List[torch.Tensor]:
    """EXTEND each runner from its committed KV to ``ends[i]``.

    A failed runner rolls back only to the start of this span. Earlier chunks
    stay committed.
    """
    from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
        read_token_span,
    )

    phases = _phase_clock(phases)
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
        phases.lap("opt_ingest_alloc")
        batch = _extend_batch_many(
            scheduler,
            runners,
            spans,
            starts,
            torch.cat(locs),
        )
        phases.lap("opt_ingest_batch")
        _begin_ingest_gpu(scheduler)
        try:
            rows = _extend_request_rows(_run_batch(scheduler, batch), len(runners))
        finally:
            _finish_ingest_gpu(scheduler)
        phases.lap("opt_ingest_forward")
    except Exception:
        for runner, start, _end in prepared:
            _rollback_failed_ingest(scheduler, runner.req, start)
        raise
    for runner, _start, end in prepared:
        runner.req.kv_committed_len = end
        runner.req.kv_allocated_len = end
        runner.cached_len = end
    return rows


def _extend_uncommitted_batch(scheduler, runners, phases=None) -> List[torch.Tensor]:
    """Write every short tail in one EXTEND and keep each request's last row."""
    return _extend_span_batch(
        scheduler,
        runners,
        [runner.prefix_len for runner in runners],
        phases=phases,
    )


def _extend_uncommitted_chunked_batch(
    scheduler, runners, chunk: int, phases=None
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
        rows = _extend_span_batch(scheduler, pending, ends, phases=phases)
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


def _page_aligned_layer_tokens(cached_len: int, n: int, page: int) -> int:
    """Tokens a new layer takes from ``available_size``.

    ``page <= 1`` charges ``n`` tokens. A larger page charges every newly
    opened page, matching the allocator's page delta.
    """
    n = int(n)
    page = int(page or 1)
    if page <= 1:
        return n
    cached = int(cached_len)
    end_pages = (cached + n + page - 1) // page
    start_pages = (cached + page - 1) // page
    return (end_pages - start_pages) * page


def _take_verify_group(runners, available: int, page: int, width: int):
    """Front of ``runners`` whose page-aligned layer fits in ``available``.

    A request that does not fit by itself is still returned alone so the
    allocator can evict. Later requests wait for the next read of
    ``available_size``.
    """
    group = []
    used = 0
    budget = int(available)
    for runner in runners:
        need = _page_aligned_layer_tokens(runner.cached_len, width, page)
        if group and used + need > budget:
            break
        group.append(runner)
        used += need
    return group


def _verify_layer_budget(scheduler) -> int:
    """Free KV tokens plus radix tokens ``alloc_token_slots`` can evict.

    ``ChunkCache.evictable_size`` is 0. SWA and Mamba caches raise
    ``NotImplementedError`` from ``evictable_size``; those keep the free-slot
    budget.
    """
    budget = int(scheduler.token_to_kv_pool_allocator.available_size())
    evictable = getattr(scheduler.tree_cache, "evictable_size", None)
    if not callable(evictable):
        return budget
    try:
        extra = evictable()
    except NotImplementedError:
        return budget
    return budget + int(extra)


def _forward_verify_fitting(scheduler, runners, steps, consume=None):
    """Run one layer in sub-batches that fit free plus evictable KV tokens.

    ``consume(group_runners, rows)`` runs before the next replay, while a
    graph output view still holds this sub-batch. A group that is not last
    is copied: the next replay overwrites that output buffer before
    ``_commit_prob_rows`` reads the returned rows. The last group stays a
    view because the caller reads it before the next layer.

    The second return is the number of ``_forward_verify_batch`` calls that
    finished. A later sub-batch that raises leaves those earlier calls
    uncounted: the exception propagates and this function does not return.
    """
    if not runners:
        raise RuntimeError("OPT verify batch is empty")
    page = int(getattr(scheduler.tree_cache, "page_size", 1) or 1)
    width = _round_width(scheduler)
    pending = list(zip(runners, steps))
    rows = []
    forwards = 0
    while pending:
        group_pairs = _take_verify_group(
            [runner for runner, _step in pending],
            _verify_layer_budget(scheduler),
            page,
            width,
        )
        group_runners = group_pairs
        count = len(group_runners)
        group_steps = [step for _runner, step in pending[:count]]
        logits = _forward_verify_batch(scheduler, group_runners, group_steps)
        forwards += 1
        group_rows = _as_layer_rows(logits, count, width)
        if consume is not None:
            consume(group_runners, group_rows)
        if count < len(pending):
            group_rows = [row.clone() for row in group_rows]
        rows.extend(group_rows)
        pending = pending[count:]
    return rows, forwards


def _forward_verify_batch(scheduler, runners, steps) -> torch.Tensor:
    """One TARGET_VERIFY for requests that still share a depth."""
    if not runners:
        raise RuntimeError("OPT verify batch is empty")
    timing = _begin_layer_timing(scheduler)
    width = _round_width(scheduler)
    device = scheduler.req_to_token_pool.device
    depth = _shared_depth(steps)
    token_rows = []
    ancestors = []
    for runner, step in zip(runners, steps):
        tokens = step.tokens
        if not torch.is_tensor(tokens):
            tokens = torch.tensor(list(tokens), dtype=torch.long)
        n = int(tokens.shape[0])
        from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
            LAYER_WIDTHS,
            layer_width_supported,
        )

        if n != width or not layer_width_supported(n):
            raise RuntimeError(
                f"OPT layer has {n} tokens but this round's graph width is {width} "
                f"and the captured layer graphs are {list(LAYER_WIDTHS)}"
            )
        ancestor = step.ancestor_mask
        anc_shape = None if ancestor is None else tuple(ancestor.shape)
        if anc_shape != (n, depth * n):
            raise RuntimeError(f"OPT ancestor mask {anc_shape} != [{n}, {depth * n}]")
        if runner.cached_len + n != runner.prefix_len + depth * n:
            raise RuntimeError(
                f"OPT cache length {runner.cached_len} is not layer {depth} "
                f"after prefix {runner.prefix_len}"
            )
        token_rows.append(tokens.to(dtype=torch.long, device=device))
        ancestors.append(ancestor)
    prefix_lens = [runner.prefix_len for runner in runners]
    flat_mask, flat_positions = pack_verify_attention(
        prefix_lens,
        ancestors,
        depth,
        width,
        device,
        model_runner=_model_runner_of(scheduler),
    )
    # Every row first: a later short row must not leave earlier ones allocated.
    _require_row_room(scheduler, runners, width)
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
        flat_positions,
        flat_mask,
        torch.cat(locs),
        cached,
    )
    bind_cascade(
        batch.spec_info, runners, ancestors, width, cached_lens=cached
    )
    _mark_layer_launch(timing)
    logits = _run_batch(scheduler, batch)
    _queue_ledger_layer(scheduler, timing, runners, depth, width)
    _finish_layer_timing(scheduler, timing)
    for runner in runners:
        runner.cached_len += width
    return logits


def _opt_page_size(scheduler) -> int:
    tree = getattr(scheduler, "tree_cache", None)
    if tree is None:
        return 1
    return int(getattr(tree, "page_size", 1) or 1)


def _decode_append_batch(scheduler, runners, phases=None) -> List[torch.Tensor]:
    """Write one missing token per request with a DECODE batch.

    ``spec_info`` stays unset so ntpb=1 can replay the captured DECODE graph.
    ``seq_lens`` counts the token being written. ``sr_decode_seq_len`` does
    not: that length is the already committed tail.
    """
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
        read_token_span,
    )

    phases = _phase_clock(phases)
    if not runners:
        phases.lap("opt_ingest_alloc")
        return []
    prepared = []
    try:
        tokens = []
        starts = []
        locs = []
        for runner in runners:
            start = int(getattr(runner.req, "kv_committed_len", 0) or 0)
            end = start + 1
            if runner.req.req_pool_idx is None:
                raise RuntimeError(f"OPT draft {runner.req.rid} has no KV pool slot")
            if start < 0 or end > int(runner.prefix_len):
                raise RuntimeError(
                    f"OPT draft {runner.req.rid} decode append [{start}, {end}) "
                    f"is outside prefix {runner.prefix_len}"
                )
            span = read_token_span(
                runner.req.origin_input_ids, runner.req.output_ids, start, 1
            )
            if len(span) != 1:
                raise RuntimeError(
                    f"OPT draft {runner.req.rid} is missing a token at {start}"
                )
            loc = runner._alloc_layer_slots(1, start=start)
            runner.req.kv_allocated_len = end
            prepared.append((runner, start, end))
            tokens.append(int(span[0]))
            starts.append(start)
            locs.append(loc.reshape(-1))
        phases.lap("opt_ingest_alloc")
        batch = scheduler._sr_make_decode_batch([runner.req for runner in runners])
        device = batch.seq_lens.device
        seq = [start + 1 for start in starts]
        token_ids = torch.tensor(tokens, dtype=torch.int64, device=device)
        batch.forward_mode = ForwardMode.DECODE
        batch.global_forward_mode = ForwardMode.DECODE
        batch.spec_info = None
        batch.input_ids = token_ids
        batch.output_ids = token_ids
        batch.seq_lens = torch.tensor(seq, dtype=torch.int64, device=device)
        batch.seq_lens_cpu = torch.tensor(seq, dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(seq, dtype=torch.int32, device=device)
        batch.seq_lens_sum = sum(seq)
        batch.out_cache_loc = torch.cat(locs)
        _keep_raw_logits(batch)
        phases.lap("opt_ingest_batch")
        _begin_ingest_gpu(scheduler)
        try:
            logits = _flatten_logits(_run_batch(scheduler, batch))
        finally:
            _finish_ingest_gpu(scheduler)
        phases.lap("opt_ingest_forward")
        if int(logits.shape[0]) != len(runners):
            raise RuntimeError(
                f"OPT decode append rows {int(logits.shape[0])} != {len(runners)}"
            )
        rows = [logits[i].clone() for i in range(len(runners))]
    except Exception:
        for runner, start, _end in prepared:
            _rollback_failed_ingest(scheduler, runner.req, start)
        raise
    for runner, _start, end in prepared:
        runner.req.kv_committed_len = end
        runner.req.kv_allocated_len = end
        runner.cached_len = end
    return rows


def _decode_uncommitted_batch(scheduler, runners, phases=None) -> List[torch.Tensor]:
    """Append a gap of one or two tokens, one DECODE step at a time."""
    pending = list(runners)
    saved = {}
    for _step in range(2):
        if not pending:
            break
        before = [
            int(getattr(runner.req, "kv_committed_len", 0) or 0) for runner in pending
        ]
        rows = _decode_append_batch(scheduler, pending, phases=phases)
        nxt = []
        for runner, row in zip(pending, rows):
            if int(getattr(runner.req, "kv_committed_len", 0) or 0) >= int(
                runner.prefix_len
            ):
                saved[id(runner)] = row
            else:
                nxt.append(runner)
        after = [
            int(getattr(runner.req, "kv_committed_len", 0) or 0) for runner in pending
        ]
        if after == before:
            raise RuntimeError("OPT decode append made no progress")
        pending = nxt
    if pending:
        raise RuntimeError(
            "OPT decode append left " + ", ".join(runner.req.rid for runner in pending)
        )
    return [saved[id(runner)] for runner in runners]


def _opt_draft_count(scheduler, name: str, n: int = 1) -> None:
    from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
        get_sr_round_metrics,
    )

    get_sr_round_metrics(scheduler, "Draft").counts[name] += int(n)


def ingest_tail_logits(scheduler, runners, phases=None) -> List[torch.Tensor]:
    """Logits after each committed tail, without SR's per-token decode threshold."""
    phases = _phase_clock(phases)
    limit = getattr(scheduler, "chunked_prefill_size", None)
    page = _opt_page_size(scheduler)
    logits = {}
    extend = []
    chunked = []
    noop = []
    decode = []
    reprefill = []
    for runner in runners:
        cached_row = take_cached_tail_logits(runner.req, runner.prefix_len)
        if cached_row is not None:
            logits[id(runner)] = cached_row
            _opt_draft_count(scheduler, "opt_ingest_zero_fwd")
            from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
                note_bonus_hit,
            )

            note_bonus_hit(scheduler, runner.req)
            continue
        mode = plan_opt_tail(runner.req, limit, page)
        if mode == "reprefill":
            reprefill.append(runner)
        elif mode == "extend":
            extend.append(runner)
        elif mode == "extend_chunked":
            chunked.append(runner)
        elif mode == "noop":
            noop.append(runner)
        elif mode == "decode":
            decode.append(runner)
        else:
            raise RuntimeError(
                f"OPT draft {runner.req.rid} ingest mode {mode} is unknown"
            )
    phases.lap("opt_ingest_plan")
    if reprefill:
        rows = reprefill_tail_logits(scheduler, [runner.req for runner in reprefill])
        phases.lap("opt_ingest_forward")
        for runner, row in zip(reprefill, rows):
            logits[id(runner)] = row
    if extend:
        for runner, row in zip(
            extend, _extend_uncommitted_batch(scheduler, extend, phases=phases)
        ):
            logits[id(runner)] = row
    if chunked:
        rows = _extend_uncommitted_chunked_batch(
            scheduler, chunked, int(limit), phases=phases
        )
        for runner, row in zip(chunked, rows):
            logits[id(runner)] = row
    if noop:
        for runner, row in zip(
            noop, _logits_at_tail_batch(scheduler, noop, phases=phases)
        ):
            logits[id(runner)] = row
    if decode:
        for runner, row in zip(
            decode, _decode_uncommitted_batch(scheduler, decode, phases=phases)
        ):
            logits[id(runner)] = row
    return [logits[id(runner)] for runner in runners]


def note_forwarded_layers(states, runners, width: int) -> None:
    """Stamp each tree with the layers forwarded for it before the reply.

    The width replay prices each layer row with the trees that were really
    in that forward. Lookahead launches layer d before judging d, so a tree
    that stops at d was in layer d's forward; the ordinary path forwards
    only trees that keep growing. ``cached_len`` counts both: every forward
    advanced it by ``width``. A deferred post-reply layer is not counted yet.
    """
    step = max(int(width), 1)
    for state, runner in zip(states, runners):
        if state is None or runner is None:
            continue
        written = int(runner.cached_len) - int(runner.prefix_len)
        state.forwarded_layers = max(written // step, 0)


def _committed_len(req) -> int:
    return len(req.origin_input_ids or []) + len(req.output_ids or [])


@dataclass
class OptDeferredTree:
    """One request whose last layer is forwarded after the reply is sent."""

    req: object
    runner: "OptLayerForward"
    keep: int
    window: tuple
    state: ClearTreeState


def _request_finished(scheduler, req) -> bool:
    checker = getattr(scheduler, "_sr_is_finished", None)
    if not callable(checker):
        return True
    try:
        return bool(checker(req))
    except Exception:
        return True


def _can_defer(scheduler, req, expanded, window) -> bool:
    """Page-1 OPT drafts can publish the tree after the last layer exists."""
    deferred = getattr(scheduler, "_opt_deferred", None)
    if not isinstance(deferred, list):
        return False
    return (
        expanded
        and window is not None
        and _opt_page_size(scheduler) == 1
        and not getattr(scheduler, "_sr_device_poisoned", False)
        and not _request_finished(scheduler, req)
        and envs.SGLANG_CLEAR_TREE_KV_REUSE.get()
    )


def _deferrable_state(state) -> bool:
    return (
        state is not None
        and bool(getattr(state, "finished", False))
        and int(getattr(state, "depth", 0) or 0) >= 1
        and getattr(state, "anc_mask", None) is not None
    )


@dataclass
class _BonusCandidate:
    parent: int
    token: int
    position: int
    chain: Tuple[int, ...]
    keep: bool = True
    end_rank: int = 0


def _first_unbanned(ids, banned) -> Optional[int]:
    """First ranked id that is not one of ``banned``.

    The ids are already the node's top ``N + 1``. A verify child is one of
    the first ``N``, so the next id is the bonus.
    """
    if ids is None or not torch.is_tensor(ids) or int(ids.numel()) == 0:
        return None
    ban = {int(token) for token in banned}
    for token in ids.detach().reshape(-1).tolist():
        token = int(token)
        if token not in ban:
            return token
    return None


def _host_bonus_ids(row: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k token ids of one layer, on CPU before the next replay."""
    ids = torch.topk(row.detach().float(), k=int(k), dim=-1).indices
    return ids.detach().to(device="cpu", dtype=torch.long)


def _append_layer_bonus_ids(runners, rows) -> None:
    """Store bonus ids only when speculative bonus precomputation is enabled."""
    if not bonus_prediction_enabled():
        return
    for runner, row in zip(runners, rows):
        if runner is None or not torch.is_tensor(row):
            continue
        plan = getattr(runner.scheduler, "_clear_round_plan", None)
        width = (
            int(plan.width)
            if plan is not None
            else int(runner.scheduler.server_args.clear_tree_width)
        )
        ids = getattr(runner, "bonus_ids", None)
        if ids is None:
            runner.bonus_ids = []
            ids = runner.bonus_ids
        ids.append(_host_bonus_ids(row, width + 1))


def ancestor_columns(state, prefix: int, depth: int, col: int) -> Tuple[int, ...]:
    """Absolute cache columns of ``(depth, col)`` and its ancestors.

    Layer 0's stored parent is not the committed tail. The walk stops there.
    """
    width = int(state.config.width)
    layer = int(depth)
    column = int(col)
    found = []
    while layer >= 0:
        found.append(int(prefix) + layer * width + column)
        if layer == 0:
            break
        column = int(state.parents[layer, column].item())
        layer -= 1
    return tuple(found)


def bonus_verify_chunks(plans, width: int) -> List[list]:
    """Split bonus candidates into TARGET_VERIFY groups of exactly ``width``.

    The attention backend sizes every verify query by
    ``speculative_num_draft_tokens``. A short tail repeats the group's last
    real candidate so every query has a position inside the committed
    sequence. Those copies are not kept.
    """
    width = int(width)
    if width < 1:
        raise ValueError(f"OPT bonus verify width must be positive, got {width}")
    items = list(plans)
    if not items:
        return []
    chunks = []
    for start in range(0, len(items), width):
        chunk = items[start : start + width]
        if len(chunk) < width:
            source = chunk[-1]
            pad = _BonusCandidate(
                int(source.parent),
                int(source.token),
                int(source.position),
                tuple(source.chain),
                keep=False,
            )
            chunk.extend([pad] * (width - len(chunk)))
        chunks.append(chunk)
    return chunks


def _host_device_is_cpu(device) -> bool:
    if device is None:
        return True
    return torch.device(device).type == "cpu"


def _copy_bonus_host(host: torch.Tensor, device) -> torch.Tensor:
    """Copy one CPU tensor to ``device``.

    CUDA destinations use pinned memory and one ``non_blocking`` copy. CPU
    destinations keep ``host``. Other devices use a blocking copy.
    """
    if _host_device_is_cpu(device):
        return host
    if torch.device(device).type == "cuda":
        if not host.is_pinned():
            host = host.pin_memory()
        out = torch.empty(host.shape, dtype=host.dtype, device=device)
        out.copy_(host, non_blocking=True)
        return out
    return host.to(device=device)


def _bonus_long_tensor(values, device) -> torch.Tensor:
    """Build one long vector on the CPU, then copy it once."""
    pinned = not _host_device_is_cpu(device) and torch.device(device).type == "cuda"
    host = torch.tensor(list(values), dtype=torch.long, pin_memory=pinned)
    return _copy_bonus_host(host, device)


def bonus_attention_mask(
    prefix_len: int,
    cached_len: int,
    chains: Sequence[Sequence[int]],
    device=None,
    query_width: Optional[int] = None,
) -> torch.Tensor:
    """Rows for one TARGET_VERIFY group laid after the forwarded tree.

    Real rows see the committed prefix, that node's ancestor chain, and their
    own new column. ``query_width`` pads with dummy rows that see only
    themselves. A real token does not see a dummy column. The self column is
    the index inside this group, and every row is ``cached_len + query_width``
    wide.

    The mask is written on the CPU. The real rows' prefix is one slice. A
    CUDA ``device`` then receives one pinned copy.
    """
    real = len(chains)
    queries = real if query_width is None else int(query_width)
    if queries < real:
        raise ValueError(
            f"OPT bonus query width {queries} is smaller than {real} real tokens"
        )
    row_width = int(cached_len) + queries
    pinned = not _host_device_is_cpu(device) and torch.device(device).type == "cuda"
    mask = torch.zeros((queries, row_width), dtype=torch.bool, pin_memory=pinned)
    prefix = int(prefix_len)
    tree_end = int(cached_len)
    if prefix > 0 and real > 0:
        mask[:real, :prefix] = True
    for index, chain in enumerate(chains):
        for col in chain:
            col = int(col)
            if 0 <= col < tree_end:
                mask[index, col] = True
        mask[index, tree_end + index] = True
    for index in range(real, queries):
        mask[index, tree_end + index] = True
    return _copy_bonus_host(mask, device)


def _drop_stored_logits(runner) -> None:
    if runner is None:
        return
    if getattr(runner, "bonus_ids", None):
        runner.bonus_ids = []


def select_bonus_candidates(state, window, runner) -> List[_BonusCandidate]:
    """First top-(N+1) id at the root and every verify node that is not a child.

    ``end_rank`` orders those candidates by residual path mass. This list stays
    in insertion order.
    """
    if state is None or window is None or len(window) != 3 or runner is None:
        return []
    from sglang.srt.speculative.clear_tree_remote.clear_tree_lease import (
        node_depths,
        strip_verify_padding,
    )

    parsed = strip_verify_padding(*window)
    if parsed is None:
        return []
    tokens, parents, cols = parsed
    depths = node_depths(parents)
    if depths is None:
        return []
    width = int(state.config.width)
    prefix = int(getattr(runner, "prefix_len", 0) or 0)
    bonus_ids = list(getattr(runner, "bonus_ids", None) or [])
    plans: List[_BonusCandidate] = []

    def add(parent, ids, banned, position, chain):
        token = _first_unbanned(ids, banned)
        if token is None:
            return
        plans.append(
            _BonusCandidate(int(parent), int(token), int(position), tuple(chain))
        )

    if bonus_ids:
        banned = [
            int(tokens[i]) for i, parent in enumerate(parents) if int(parent) == -1
        ]
        add(-1, bonus_ids[0], banned, prefix, ())
    for index, depth in enumerate(depths):
        depth = int(depth)
        col = int(cols[index])
        slot = depth + 1
        if slot < 0 or slot >= len(bonus_ids) or not (0 <= col < width):
            continue
        row = bonus_ids[slot]
        if not torch.is_tensor(row) or row.dim() < 1 or col >= int(row.shape[0]):
            continue
        banned = [
            int(tokens[j]) for j, parent in enumerate(parents) if int(parent) == index
        ]
        add(
            index,
            row[col],
            banned,
            prefix + depth + 1,
            ancestor_columns(state, prefix, depth, col),
        )
    if plans:
        end_score = {-1: 1.0}
        for i, depth in enumerate(depths):
            end_score[i] = float(state.weight[int(depth), int(cols[i])])
        for i, parent in enumerate(parents):
            end_score[int(parent)] -= float(state.weight[int(depths[i]), int(cols[i])])
        ranked = sorted(
            plans, key=lambda plan: end_score[int(plan.parent)], reverse=True
        )
        for rank, plan in enumerate(ranked):
            plan.end_rank = rank
    return plans


@dataclass
class _BonusAdmit:
    """Whether this round already tried to admit the bonus template.

    A successful template and its M-RoPE delta stay on
    ``_opt_la_batches[rid]`` until the KV pool index changes or the request
    is dropped. This object only stops a second ``init_new`` inside one round.
    """

    tried: bool = False
    template: object = None
    delta: object = None


def _copy_ids_to_host(flat: torch.Tensor) -> List[int]:
    """Read slot ids. CUDA uses one pinned ``non_blocking`` copy and one sync."""
    flat = flat.detach().reshape(-1).to(dtype=torch.long)
    if flat.device.type == "cpu":
        return [int(value) for value in flat.tolist()]
    if flat.device.type == "cuda":
        host = torch.empty(int(flat.numel()), dtype=torch.long, pin_memory=True)
        host.copy_(flat, non_blocking=True)
        torch.get_device_module(flat.device).current_stream().synchronize()
        return [int(value) for value in host.tolist()]
    return [int(value) for value in flat.cpu().tolist()]


def _read_bonus_slot_ids(locs: Sequence[torch.Tensor]) -> List[List[int]]:
    """Read all request rows of one group with a single host copy."""
    pieces = []
    for loc in locs:
        if loc is None:
            raise RuntimeError(
                "OPT bonus slot ids were read after the buffer was cleared"
            )
        pieces.append(loc.detach().reshape(-1))
    if not pieces:
        return []
    widths = [int(piece.numel()) for piece in pieces]
    values = _copy_ids_to_host(torch.cat(pieces))
    rows = []
    start = 0
    for width in widths:
        rows.append(values[start : start + width])
        start += width
    return rows


def _bonus_model_runner(scheduler):
    worker = getattr(scheduler, "model_worker", None)
    return getattr(worker, "model_runner", None)


def _bind_bonus_cascade(batch, runners, mask, ends, width: int) -> None:
    """Attach this bonus group's rows. Slots already sit at ``[end, end + width)``."""
    from sglang.srt.speculative.clear_tree_remote.clear_tree_attention import (
        cascade_enabled,
    )

    if not isinstance(runners, (list, tuple)):
        runners = [runners]
        ends = [ends]
    if not runners or not cascade_enabled(
        _model_runner_of(getattr(runners[0], "scheduler", None))
    ):
        return
    width = int(width)
    flat = mask.reshape(-1)
    ancestors = []
    prefix_chunks = []
    saw_prefix = False
    offset = 0
    for runner, end in zip(runners, ends):
        end = int(end)
        row_width = end + width
        block = flat[offset : offset + width * row_width].reshape(width, row_width)
        offset += width * row_width
        prefix = int(getattr(runner, "prefix_len", 0) or 0)
        ancestors.append(block[:, prefix:])
        if prefix > 0:
            saw_prefix = True
            prefix_chunks.append(block[:, 0].to(dtype=torch.int32))
        else:
            prefix_chunks.append(
                torch.ones(width, dtype=torch.int32, device=block.device)
            )
    if offset != int(flat.shape[0]):
        raise RuntimeError(
            f"bonus mask length {int(flat.shape[0])} does not match tree ends {list(ends)}"
        )
    prefix_rows = torch.cat(prefix_chunks) if saw_prefix else None
    bind_cascade(
        batch.spec_info,
        list(runners),
        ancestors,
        width,
        prefix_rows=prefix_rows,
        cached_lens=[int(end) for end in ends],
    )


def _fill_bonus_schedule_batch(
    scheduler, runner, tokens, positions, mask, loc, tree_end: int, width: int
):
    """Install this group on the cached decode batch.

    Device ``seq_lens`` and ``orig_seq_lens`` are filled in place. ``seq_lens_cpu``
    is a new pageable tensor so a later fill cannot overwrite an in-flight copy.
    """
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )

    batch = _lookahead_batch(scheduler, runner.req)
    batch.forward_mode = ForwardMode.TARGET_VERIFY
    batch.global_forward_mode = ForwardMode.TARGET_VERIFY
    batch.input_ids = tokens.reshape(-1).to(dtype=torch.int64)
    batch.output_ids = None
    batch.seq_lens.fill_(int(tree_end))
    batch.orig_seq_lens.fill_(int(tree_end))
    batch.seq_lens_cpu = torch.tensor([int(tree_end)], dtype=torch.int64)
    batch.seq_lens_sum = int(tree_end)
    batch.out_cache_loc = loc.reshape(-1)
    batch.extend_lens = [int(width)]
    batch.prefix_lens = [int(tree_end)]
    batch.extend_num_tokens = int(width)
    batch.extend_logprob_start_lens = [int(tree_end)]
    batch.spec_info = ClearLayerSpec(
        custom_mask=mask.reshape(-1),
        positions=positions.reshape(-1).to(dtype=torch.int64),
        draft_token_num=int(width),
        num_tokens_per_req=int(width),
        capture_hidden_mode=CaptureHiddenMode.NULL,
        apply_prefix_custom_mask=True,
        is_bonus=True,
    )
    _bind_bonus_cascade(batch, [runner], mask, [tree_end], width)
    return batch


def _bonus_template_forward(scheduler, batch, template, delta, width: int):
    """Forward once with the width-specific template; false graph means eager.

    Do not call ``run_batch`` after that forward: the slots were just written.
    """
    forward_batch = _refresh_lookahead_forward(template, batch, delta)
    out = scheduler.model_worker.model_runner.forward(forward_batch)
    return _bonus_result(out, 1, width)


def _bonus_batch_hit(scheduler, req):
    """The lookahead batch entry for this request and its current pool."""
    batches = getattr(scheduler, "_opt_la_batches", None)
    if not batches:
        return None
    hit = batches.get(getattr(req, "rid", None))
    if hit is None or hit.get("pool_idx") != getattr(req, "req_pool_idx", None):
        return None
    return hit


def _load_bonus_admit(scheduler, req, width: int):
    """Templates are specific to both the request pool and query width."""
    hit = _bonus_batch_hit(scheduler, req)
    if hit is None:
        return None
    return hit.get("bonus_templates", {}).get(int(width))


def _store_bonus_admit(scheduler, req, width: int, template, delta) -> None:
    """Keep a successful width-specific admission. A miss is not stored."""
    if template is None:
        return
    hit = _bonus_batch_hit(scheduler, req)
    if hit is not None:
        hit.setdefault("bonus_templates", {})[int(width)] = (template, delta)


def _load_layer_admit(scheduler, req):
    """Return a cached layer template, or None when this pool has none."""
    hit = _bonus_batch_hit(scheduler, req)
    if hit is None:
        return None
    template = hit.get("layer_template")
    if template is None:
        return None
    return template, hit.get("layer_delta")


def _store_layer_admit(scheduler, req, template, delta) -> None:
    """Keep a successful layer admission. A miss is not stored."""
    if template is None:
        return
    hit = _bonus_batch_hit(scheduler, req)
    if hit is None:
        return
    hit["layer_template"] = template
    hit["layer_delta"] = delta


def _bonus_result(result, count: int, width: int):
    """Normalize worker and scheduler results without guessing graph success."""
    flag = getattr(result, "can_run_cuda_graph", None)
    if flag is None:
        flag = getattr(result, "can_run_graph", None)
    path = "unknown" if flag is None else ("graph" if flag else "eager")
    logits = getattr(getattr(result, "logits_output", None), "next_token_logits", None)
    if logits is None:
        raise RuntimeError("ClearTree bonus forward returned no logits")
    return _as_layer_rows(logits, count, width), path


class _BonusBatch:
    """Ordinary request-major verify on the shared unsuffixed graph."""

    def __call__(self, scheduler, entries, width, tokens, positions, mask, loc, ends):
        batch = _verify_batch_many(
            scheduler,
            [job.runner for job, _chunk in entries],
            list(tokens),
            positions,
            mask,
            loc,
            ends,
            is_bonus=True,
        )
        _bind_bonus_cascade(
            batch, [job.runner for job, _chunk in entries], mask, ends, width
        )
        _selected, graph_expected = _bonus_width(scheduler, len(entries), width)
        batch.spec_info.bonus_eager = not graph_expected
        scheduler.cur_batch = batch
        _opt_draft_count(scheduler, "opt_bonus_forwards")
        _opt_draft_count(scheduler, f"opt_bonus_width_{width}")
        return _bonus_result(scheduler.run_batch(batch), len(entries), width)


class _BonusSingle:
    """Cascade singleton with admission isolated by request and query width."""

    def __init__(self):
        self.admissions = {}

    def __call__(self, scheduler, entries, width, tokens, positions, mask, loc, ends):
        runner = entries[0][0].runner
        batch = _fill_bonus_schedule_batch(
            scheduler, runner, tokens, positions, mask, loc, ends[0], width
        )
        batch.spec_info.bonus_eager = not _bonus_width(scheduler, 1, width)[1]
        key = (runner.req.rid, int(runner.req.req_pool_idx), int(width))
        admit = self.admissions.setdefault(key, _BonusAdmit())
        if not admit.tried:
            admit.tried = True
            cached = _load_bonus_admit(scheduler, runner.req, width)
            if cached is not None:
                admit.template, admit.delta = cached
            else:
                admit.template = _lookahead_forward_template(scheduler, batch)
                admit.delta = (
                    _mrope_delta_from_template(admit.template)
                    if admit.template is not None
                    else None
                )
                _store_bonus_admit(
                    scheduler, runner.req, width, admit.template, admit.delta
                )
        if admit.template is None:
            # Admission has not submitted a forward. This is the only fallback.
            return _BonusBatch()(
                scheduler, entries, width, tokens, positions, mask, loc, ends
            )
        scheduler.cur_batch = batch
        _opt_draft_count(scheduler, "opt_bonus_forwards")
        _opt_draft_count(scheduler, f"opt_bonus_width_{width}")
        return _bonus_template_forward(
            scheduler, batch, admit.template, admit.delta, width
        )


def _bonus_executor(entries, single):
    if len(entries) == 1:
        from sglang.srt.speculative.clear_tree_remote.clear_tree_attention import (
            cascade_enabled,
        )

        runner = entries[0][0].runner
        if cascade_enabled(_model_runner_of(getattr(runner, "scheduler", None))):
            return single
    return _BonusBatch()


def _bonus_job_ready(scheduler, job) -> bool:
    runner = job.runner
    return (
        runner is not None
        and _opt_page_size(scheduler) == 1
        and not getattr(scheduler, "_sr_device_poisoned", False)
        and not _request_finished(scheduler, job.req)
        and not getattr(runner, "lookahead_dropped", False)
        and not getattr(runner, "lookahead_inflight", False)
        and job.state is not None
        and hasattr(runner, "_alloc_layer_slots")
        and int(runner.cached_len)
        == int(runner.prefix_len) + int(job.state.depth) * int(job.state.config.width)
    )


def _bonus_graph(scheduler):
    """CUDA graph runner that can replay a shared verify graph, or None."""
    device = torch.device(scheduler.req_to_token_pool.device)
    graph = getattr(_bonus_model_runner(scheduler), "graph_runner", None)
    if (
        device.type != "cuda"
        or graph is None
        or getattr(graph, "require_mlp_tp_gather", False)
        or getattr(graph, "require_attn_tp_gather", False)
        or getattr(graph, "require_gathered_buffer", False)
    ):
        return None
    return graph


def _replay_bs(graph, bs: int):
    """The batch the shared replay uses: bisect pad, or exact without padding."""
    capture = sorted(int(b) for b in graph.capture_bs)
    if getattr(graph, "disable_padding", False):
        return int(bs) if int(bs) in capture else None
    index = bisect.bisect_left(capture, int(bs))
    return capture[index] if index < len(capture) else None


def _bonus_width(scheduler, bs: int, need: int) -> Tuple[int, bool]:
    """Smallest captured verify width >= need at the batch replay pads to.

    Bonus shares ``r{G}_{bs}`` graphs with layer forwards. The capture loop
    skips ``bs * G > max_num_token`` (for example ``r65_4``), and shared
    can_run does not check the padded key, so check it here. Too narrow:
    widest, and the caller splits.
    """
    from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
        BONUS_NTPBS,
        LAYER_WIDTHS,
    )

    graph = _bonus_graph(scheduler)
    replay_bs = _replay_bs(graph, bs) if graph is not None else None
    if replay_bs is not None:
        stream = None
        if getattr(graph, "enable_pdmux", False):
            from sglang.srt.multiplex.pdmux_context import get_current_stream_idx

            stream = get_current_stream_idx()
        widths = sorted(
            w
            for w in {int(w) for w in (*BONUS_NTPBS, *LAYER_WIDTHS)}
            if w > 1 and graph._make_graph_key(replay_bs, stream, w) in graph.graphs
        )
        if widths:
            return next((w for w in widths if w >= need), widths[-1]), True
    return int(need), False


@dataclass
class _BonusBatchOwner:
    """One allocation, retained until every row is released or transferred."""

    entries: list
    width: int
    loc: Optional[torch.Tensor] = None
    # After padding release this holds only the real slots, never freed rows.
    owned: Optional[torch.Tensor] = None
    mapped: bool = False
    complete: bool = False
    release_unknown: bool = False
    previous_guesses: object = None


def _quarantine_bonus_batch(scheduler, owner) -> None:
    from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
        quarantine,
    )

    scheduler._sr_device_poisoned = True
    clear = getattr(scheduler, "_sr_clear_stamps_after_poison", None)
    if callable(clear):
        clear()
    record = quarantine(scheduler).retain(owner)
    # No event/units: neither an unknown consumer nor a failed free may be
    # retried by a later quarantine poll. The owner retains the device ids.
    record.release_error = RuntimeError("ClearTree bonus batch ownership unresolved")


def _clear_bonus_batch_map(scheduler, owner) -> None:
    if not owner.mapped:
        return
    for job, _chunk in owner.entries:
        end = int(job.runner.cached_len)
        _clear_req_to_token_span(scheduler, job.req, end, end + owner.width)
        job.req.kv_allocated_len = end
    owner.mapped = False


def _cleanup_bonus_batch(scheduler, owner) -> None:
    if owner.loc is None:
        return
    if getattr(scheduler, "_sr_device_poisoned", False) or owner.release_unknown:
        _quarantine_bonus_batch(scheduler, owner)
        return
    try:
        if not owner.complete and owner.loc.device.type != "cpu":
            torch.get_device_module(owner.loc.device).current_stream().synchronize()
        owner.complete = True
        _clear_bonus_batch_map(scheduler, owner)
        if owner.owned is not None:
            # Mark before submission: a throwing allocator may have freed a
            # subset already. Never retry that submission or free siblings.
            owner.release_unknown = True
            scheduler.token_to_kv_pool_allocator.free(owner.owned)
            owner.release_unknown = False
            owner.owned = None
    except Exception:
        _quarantine_bonus_batch(scheduler, owner)
        raise


def _run_bonus_group(scheduler, entries, width: int, executor=None) -> bool:
    """One transactional bonus allocation; real guesses append across groups."""
    from sglang.srt.mem_cache.common import alloc_token_slots
    from sglang.srt.speculative.clear_tree_remote.clear_tree_lease import (
        OptBonusGuess,
    )
    from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

    device = scheduler.req_to_token_pool.device
    pool = scheduler.req_to_token_pool.req_to_token
    count = len(entries)
    ends = [int(job.runner.cached_len) for job, _chunk in entries]
    rows = [int(job.req.req_pool_idx) for job, _chunk in entries]
    if (
        len(set(rows)) != count
        or any(row < 0 or row >= pool.shape[0] for row in rows)
        or any(end < 0 or end + width > pool.shape[1] for end in ends)
    ):
        _opt_draft_count(scheduler, "opt_bonus_skipped_map")
        _opt_draft_count(scheduler, "opt_bonus_skipped", count)
        return False
    allocator = scheduler.tree_cache.token_to_kv_pool_allocator
    # SWA alloc() can remove slots from one sub-pool and then fail before the
    # caller receives them. Bonus is optional, so skip rather than risk a leak.
    # Detect by dual sub-pools to avoid importing the SWA module on CPU checks.
    if hasattr(allocator, "full_attn_allocator") and hasattr(
        allocator, "swa_attn_allocator"
    ):
        _opt_draft_count(scheduler, "opt_bonus_skipped_allocator")
        _opt_draft_count(scheduler, "opt_bonus_skipped", count)
        return False
    # Do not include evictable_size: bonus must not evict useful prefix KV.
    if int(allocator.available_size()) < count * width:
        _opt_draft_count(scheduler, "opt_bonus_skipped_kv")
        _opt_draft_count(scheduler, "opt_bonus_skipped", count)
        return False
    owner = _BonusBatchOwner(entries, width)
    try:
        owner.loc = alloc_token_slots(scheduler.tree_cache, count * width)
        owner.owned = owner.loc
        if owner.loc is None or owner.loc.numel() != count * width:
            raise RuntimeError("ClearTree bonus batch allocation has wrong size")
        starts = _bonus_long_tensor(ends, device)
        # Assign may partially submit before throwing, so cleanup owns every
        # request span before calling it.
        owner.mapped = True
        for job, _chunk in entries:
            job.req.kv_allocated_len = int(job.runner.cached_len) + width
        assign_req_to_token_pool_func(
            _bonus_long_tensor(rows, device),
            pool,
            starts,
            starts + width,
            owner.loc,
            count,
        )
        tokens = _bonus_long_tensor(
            [p.token for _job, chunk in entries for p in chunk], device
        ).reshape(count, width)
        positions = _bonus_long_tensor(
            [p.position for _job, chunk in entries for p in chunk], device
        )
        masks = [
            bonus_attention_mask(
                job.runner.prefix_len, end, [p.chain for p in chunk], query_width=width
            ).reshape(-1)
            for (job, chunk), end in zip(entries, ends)
        ]
        mask = _copy_bonus_host(torch.cat(masks), device)
        if executor is None:
            executor = _bonus_executor(entries, _BonusSingle())
        logits, path = executor(
            scheduler, entries, width, tokens, positions, mask, owner.loc, ends
        )
        _opt_draft_count(scheduler, "opt_bonus_" + path)
        slot_rows = _read_bonus_slot_ids(list(owner.loc.reshape(count, width)))
        owner.complete = True
        _clear_bonus_batch_map(scheduler, owner)
        guesses = []
        real_indices, dummy_indices = [], []
        for i, ((_job, chunk), slots) in enumerate(zip(entries, slot_rows)):
            kept = []
            for j, (plan, slot) in enumerate(zip(chunk, slots)):
                if not plan.keep:
                    dummy_indices.append(i * width + j)
                    continue
                real_indices.append(i * width + j)
                kept.append(
                    OptBonusGuess(
                        parent=int(plan.parent),
                        token=int(plan.token),
                        slot=int(slot),
                        logits=logits[i][j].detach().clone(),
                        end_rank=int(plan.end_rank),
                    )
                )
            guesses.append(kept)
        # Build everything (including indexed device tensors) before any free.
        real_loc = owner.loc[real_indices]
        dummy_loc = owner.loc[dummy_indices]
        # Allocate merged Python lists before freeing anything. No fallible
        # tensor work may happen between padding free and ownership transfer.
        merged = [
            list(getattr(job.req, "opt_bonus_guesses", None) or []) + kept
            for (job, _chunk), kept in zip(entries, guesses)
        ]
        if dummy_indices:
            owner.release_unknown = True
            scheduler.token_to_kv_pool_allocator.free(dummy_loc)
            owner.release_unknown = False
        owner.owned = real_loc
        # No further fallible device operations before ownership transfer.
        for (job, _chunk), kept in zip(entries, merged):
            job.req.opt_bonus_guesses = kept
        owner.owned = None
        _opt_draft_count(scheduler, "opt_bonus_requests", count)
        _opt_draft_count(scheduler, "opt_bonus_query_rows", count * width)
        _opt_draft_count(scheduler, "opt_bonus_padding_rows", len(dummy_indices))
        return True
    except Exception as exc:
        try:
            _reraise_device_context(scheduler, exc)
        finally:
            _cleanup_bonus_batch(scheduler, owner)
        raise


def _release_previous_bonus(scheduler, selected) -> None:
    from sglang.srt.speculative.clear_tree_remote.clear_tree_lease import (
        release_opt_bonus_guesses,
    )

    for job, _plans in selected:
        previous = getattr(job.req, "opt_bonus_guesses", None)
        try:
            release_opt_bonus_guesses(scheduler, job.req)
        except Exception:
            # release clears the field before submitting free. Preserve the
            # detached guesses if the allocator may have released only a part.
            owner = _BonusBatchOwner([(job, [])], 0, previous_guesses=previous)
            owner.release_unknown = True
            _quarantine_bonus_batch(scheduler, owner)
            raise


def maybe_attach_bonus_guesses_many(scheduler, jobs) -> None:
    """Predict all eligible requests, including settled lookahead, group by group."""
    jobs = list(jobs)
    try:
        if not bonus_prediction_enabled():
            return
        selected = []
        for job in jobs:
            if not _bonus_job_ready(scheduler, job):
                _opt_draft_count(scheduler, "opt_bonus_skipped")
                _opt_draft_count(scheduler, "opt_bonus_skipped_unready")
                continue
            plans = select_bonus_candidates(job.state, job.window, job.runner)
            if plans:
                selected.append((job, sorted(plans, key=lambda p: p.end_rank)))
            else:
                _opt_draft_count(scheduler, "opt_bonus_skipped")
                _opt_draft_count(scheduler, "opt_bonus_skipped_empty")
        if not selected:
            return
        # Validate before releasing old guesses or writing any temporary map.
        rows = [int(job.req.req_pool_idx) for job, _plans in selected]
        if len(rows) != len(set(rows)):
            _opt_draft_count(scheduler, "opt_bonus_skipped_map")
            _opt_draft_count(scheduler, "opt_bonus_skipped", len(selected))
            return
        _release_previous_bonus(scheduler, selected)
        _opt_draft_count(scheduler, "opt_bonus_rounds", len(selected))
        single = _BonusSingle()
        split = set()
        while selected:
            width, _expected = _bonus_width(
                scheduler, len(selected), max(len(plans) for _, plans in selected)
            )
            entries = []
            remaining = []
            for job, plans in selected:
                entries.append((job, bonus_verify_chunks(plans[:width], width)[0]))
                if len(plans) > width:
                    if id(job) not in split:
                        split.add(id(job))
                        _opt_draft_count(scheduler, "opt_bonus_groups_split")
                    remaining.append((job, plans[width:]))
            if not _run_bonus_group(
                scheduler, entries, width, _bonus_executor(entries, single)
            ):
                break
            selected = remaining
    except Exception as exc:
        _reraise_device_context(scheduler, exc)
        if getattr(scheduler, "_sr_device_poisoned", False):
            raise
        _opt_draft_count(scheduler, "opt_bonus_failed")
        logger.warning("[ClearTree] bonus group failed: %s", exc)
    finally:
        for job in jobs:
            _drop_stored_logits(job.runner)


def maybe_attach_bonus_guesses(scheduler, job: OptDeferredTree) -> None:
    """Compatibility entry point; singleton uses the same ownership path."""
    maybe_attach_bonus_guesses_many(scheduler, [job])


def final_layer_step(state) -> ClearTreeStep:
    """The layer ``step_after_layer`` closed without forwarding."""
    return ClearTreeStep(
        is_final=False,
        tokens=state.tokens[state.depth - 1].clone(),
        positions=_layer_positions(state),
        ancestor_mask=state.anc_mask,
    )


def _clear_req_to_token_span(scheduler, req, start: int, end: int) -> None:
    row = getattr(req, "req_pool_idx", None)
    pool = getattr(getattr(scheduler, "req_to_token_pool", None), "req_to_token", None)
    if row is None or not torch.is_tensor(pool):
        return
    start = int(start)
    end = min(int(end), int(pool.shape[1]))
    if start < 0 or end <= start or int(row) < 0 or int(row) >= int(pool.shape[0]):
        return
    pool[int(row), start:end] = -1


def _drop_unfinished_layer(scheduler, job: OptDeferredTree) -> None:
    """Pop a last layer whose forward did not finish.

    ``local_rollback`` writes ``kv_committed_len`` to the fork point. That
    point is the tree length, past the committed prefix, so OPT puts the
    committed length back. The shared rollbacker is unchanged.
    """
    runner = job.runner
    if runner is None:
        return
    depth = int(job.state.depth)
    width = int(job.state.config.width)
    if int(runner.cached_len) == int(runner.prefix_len) + depth * width:
        return
    if len(runner.layer_slots) < depth:
        return
    if getattr(scheduler, "_sr_device_poisoned", False):
        # Already isolated: the slots stay mapped and poisoned cleanup owns
        # them. Do not try a rollback that would only be refused.
        return
    popped = runner.layer_slots.pop()
    cached = int(runner.cached_len)
    try:
        scheduler._clear_rollback(job.req, cached)
    except Exception as exc:
        # A partial free arrives as a device-context error and poisons here,
        # so the later publish does not roll the same span back again.
        runner.layer_slots.append(popped)
        _reraise_device_context(scheduler, exc)
        logger.warning(
            "[OPT] rollback of unfinished layer failed for %s",
            getattr(job.req, "rid", None),
            exc_info=True,
        )
        _quarantine_lookahead(
            scheduler, [runner], exc, counter="opt_deferred_unsettled"
        )
        return
    job.req.kv_committed_len = int(job.keep)
    _clear_req_to_token_span(scheduler, job.req, cached, cached + width)


class LookaheadFallback(RuntimeError):
    """Leave the overlapped path and grow this tree the original way."""


class LookaheadUnsettled(RuntimeError):
    """A failed lookahead could not prove its slots were rolled back.

    Not a ``LookaheadFallback``: the round must not be rebuilt on slots whose
    owner is unknown. The slots stay on the request and in quarantine, and
    the device is treated as poisoned.
    """


class _PinnedLayer:
    """One host buffer for an in-flight layer copy.

    Two of these alternate. The host clones a slot only after its event, and
    the next copy into that slot waits until that clone has happened.
    """

    def __init__(self, width: int):
        n = int(width)
        pin = {"device": "cpu", "pin_memory": True}
        self.scores = torch.empty(n, dtype=torch.float32, **pin)
        self.tokens = torch.empty(n, dtype=torch.long, **pin)
        self.parents = torch.empty(n, dtype=torch.long, **pin)
        self.ids = (
            torch.empty(n, n + 1, dtype=torch.long, **pin)
            if bonus_prediction_enabled()
            else None
        )
        self.event = torch.cuda.Event()
        self.kind = "seed"
        self.ready = False

    def copy_seed(self, scores: torch.Tensor, ids: torch.Tensor) -> None:
        n = int(self.scores.shape[0])
        flat_ids = ids.reshape(-1)
        self.scores.copy_(scores.reshape(-1)[:n].float(), non_blocking=True)
        self.tokens.copy_(flat_ids[:n].to(dtype=torch.long), non_blocking=True)
        if self.ids is not None:
            self.ids[0, : int(flat_ids.numel())].copy_(
                flat_ids.to(dtype=torch.long), non_blocking=True
            )
        self.kind = "seed"
        self.event.record()
        self.ready = True

    def copy_expand(
        self,
        top_scores: torch.Tensor,
        tokens: torch.Tensor,
        parents: torch.Tensor,
        vocab_ids: torch.Tensor,
    ) -> None:
        self.scores.copy_(top_scores.reshape(-1).float(), non_blocking=True)
        self.tokens.copy_(tokens.reshape(-1).to(dtype=torch.long), non_blocking=True)
        self.parents.copy_(parents.reshape(-1).to(dtype=torch.long), non_blocking=True)
        if self.ids is not None:
            self.ids.copy_(vocab_ids.to(dtype=torch.long), non_blocking=True)
        self.kind = "expand"
        self.event.record()
        self.ready = True

    def clone_cpu(self):
        if not self.ready:
            raise RuntimeError("lookahead pinned slot was read before a copy")
        self.event.synchronize()
        self.ready = False
        return (
            self.scores.clone(),
            self.tokens.clone(),
            self.parents.clone(),
            self.ids.clone() if self.ids is not None else None,
            self.kind,
        )


class _PinnedBonus:
    """Top-(N+1) ids of the speculative last layer. Not one of the two layer slots."""

    def __init__(self, width: int):
        n = int(width)
        self.ids = torch.empty(
            n, n + 1, dtype=torch.long, device="cpu", pin_memory=True
        )
        self.event = torch.cuda.Event()
        self.ready = False

    def copy(self, vocab_ids: torch.Tensor) -> None:
        self.ids.copy_(vocab_ids.to(dtype=torch.long), non_blocking=True)
        self.event.record()
        self.ready = True

    def clone_cpu(self) -> torch.Tensor:
        if not self.ready:
            raise RuntimeError("lookahead bonus ids were read before the copy")
        self.event.synchronize()
        self.ready = False
        return self.ids.clone()


def _vocab_topk(logits: torch.Tensor, k: int):
    """Temperature-1 top-k on the probability device. No host copy."""
    work = logits.detach()
    if work.dim() == 3 and int(work.shape[0]) == 1:
        work = work[0]
    probs = torch.softmax(work.float(), dim=-1)
    width = int(probs.shape[-1])
    k = min(max(int(k), 1), width)
    return torch.topk(probs, k=k, dim=-1)


def _lookahead_backend_is_triton(scheduler) -> bool:
    args = getattr(scheduler, "server_args", None)
    name = str(getattr(args, "attention_backend", None) or "").lower()
    if name == "triton":
        return True
    worker = getattr(scheduler, "model_worker", None)
    runner = getattr(worker, "model_runner", None)
    backend = getattr(runner, "attn_backend", None)
    return "triton" in type(backend).__name__.lower()


def lookahead_mode(scheduler) -> bool:
    """Whether this process runs Draft lookahead. Cached; it does not change.

    The ledger tags layer samples with it, so ordinary layers in a lookahead
    process stay out of the lookahead layer model.
    """
    cached = getattr(scheduler, "_opt_la_mode", None)
    if not isinstance(cached, bool):
        try:
            cached = bool(lookahead_supported(scheduler))
        except Exception:
            cached = False
        try:
            scheduler._opt_la_mode = cached
        except Exception:
            pass
    return bool(cached)


def lookahead_supported(scheduler) -> bool:
    """CUDA page-1 Triton only. Anything else keeps ``grow_active_layers``."""
    if not opt_draft_lookahead_enabled():
        return False
    device = getattr(scheduler, "device", None)
    try:
        if device is None or torch.device(device).type != "cuda":
            return False
    except Exception:
        return False
    if _opt_page_size(scheduler) != 1:
        return False
    return _lookahead_backend_is_triton(scheduler)


def _lookahead_kv_need(scheduler, ingest_slots: int = 0) -> int:
    """Slots held together: ingest, every layer, and an optional bonus group.

    ``depth_rule`` stops once ``depth`` reaches ``max_depth - 1``. Lookahead
    forwards that last layer before the host decides to stop, so the tree
    takes ``(max_depth - 1) * n`` slots. Before K is known, reserve the
    largest configured candidate set. Narrow singleton graphs may need one
    padded tail while earlier real guesses remain allocated. This is an
    estimate, not a reservation; every group rechecks free space. Admission
    reserves only ``_lookahead_tree_need``; the bonus part is not evicted for.
    """
    return _lookahead_tree_need(scheduler, ingest_slots) + _lookahead_bonus_slots(
        scheduler
    )


def _lookahead_tree_need(scheduler, ingest_slots: int = 0) -> int:
    """Ingest plus every lookahead layer; the part admission reserves."""
    n = _round_width(scheduler)
    depth = int(scheduler.server_args.clear_tree_max_depth)
    return int(ingest_slots) + (depth - 1) * n


def _lookahead_bonus_slots(scheduler) -> int:
    """Largest bonus group one request can take; 0 when bonus is off.

    Not part of the admission reservation: ``_run_bonus_group`` uses free
    slots only and skips the bonus when they do not fit.
    """
    if not bonus_prediction_enabled():
        return 0
    from sglang.srt.speculative.clear_tree_remote.clear_tree_width import BONUS_NTPBS

    maximum = max(BONUS_NTPBS)
    width, _expected = _bonus_width(scheduler, 1, maximum)
    return ((maximum + width - 1) // width) * width


def _lookahead_ingest_slots(runner) -> int:
    """Tokens ingest will allocate. ``prefix_len`` is already the committed length."""
    committed = int(runner.prefix_len)
    have = int(getattr(runner.req, "kv_committed_len", 0) or 0)
    return max(committed - have, 0)


def _lookahead_would_evict(scheduler, ingest_slots: int = 0) -> bool:
    """True when the free list alone cannot cover ingest, the tree and a bonus.

    A pure check (``available_size`` is ``len`` of the device free list) kept
    for tests and diagnostics. Admission does not call it; it reserves with
    ``_lookahead_reserve_kv``.
    """
    tree = getattr(scheduler, "tree_cache", None)
    if tree is None:
        return False
    if getattr(tree, "token_to_kv_pool_allocator", None) is None:
        return True
    try:
        need = _lookahead_kv_need(scheduler, ingest_slots)
    except Exception:
        return True
    return _lookahead_short_of(scheduler, need)


# Admission evicts this many times the round's need, so a full radix cache
# does not evict again on every round.
_LOOKAHEAD_EVICT_FACTOR = 2


def _plain_radix_eviction(tree) -> bool:
    """True when evicting from ``tree`` cannot synchronize.

    Exactly ``RadixCache`` over ``TokenToKVPoolAllocator``: ``evict`` pops a
    host heap and ``free`` concatenates device index tensors. Subclasses
    (HiRadix, LMCache, Mamba, SWA) and wrappers may write back or sync, so
    they keep falling back on a short free list.
    """
    try:
        from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
        from sglang.srt.mem_cache.radix_cache import RadixCache
    except Exception:
        return False
    allocator = getattr(tree, "token_to_kv_pool_allocator", None)
    return type(tree) is RadixCache and type(allocator) is TokenToKVPoolAllocator


def _lookahead_reserve_kv(scheduler, need: int) -> None:
    """Make the free list cover ``need`` before ingest, or raise ``LookaheadFallback``.

    ``need`` is ingest plus the tree (``_lookahead_tree_need``), never the
    bonus: bonus keeps its free-slots-only rule.

    Finished requests stay in the radix cache, so a long run fills the pool
    and the free list holds only what the last rollback returned. Layer
    allocations go through ``alloc_token_slots``, which evicts on demand
    without syncing, so a short free list was never a reason to give up
    lookahead. Admission instead evicts ``_LOOKAHEAD_EVICT_FACTOR * need``
    tokens once, with the same ``evict_from_tree_cache``, so layers do not
    evict a little at a time inside the overlapped loop. Eviction only
    removes unlocked leaves, so running requests keep their prefixes. Other
    caches, or free plus evictable below ``need``, fall back with the sizes
    in the reason.
    """
    tree = getattr(scheduler, "tree_cache", None)
    if tree is None:
        return
    allocator = getattr(tree, "token_to_kv_pool_allocator", None)
    if allocator is None:
        raise LookaheadFallback("kv allocator missing")
    need = int(need)
    try:
        free = int(allocator.available_size())
    except Exception:
        raise LookaheadFallback("kv free size unknown")
    if free >= need:
        return
    plain = _plain_radix_eviction(tree)
    evictable = 0
    if plain:
        try:
            evictable = int(tree.evictable_size())
        except Exception:
            evictable = 0
    if not plain or free + evictable < need:
        raise LookaheadFallback(
            f"kv short need={need} free={free} evictable={evictable}"
            + ("" if plain else " cache=not-plain-radix")
        )
    try:
        from sglang.srt.mem_cache.common import evict_from_tree_cache

        evict_from_tree_cache(tree, _LOOKAHEAD_EVICT_FACTOR * need)
        after = int(allocator.available_size())
    except Exception as exc:
        _reraise_device_context(scheduler, exc)
        logger.warning("[OPT] lookahead eviction failed", exc_info=True)
        raise LookaheadFallback("kv evict failed") from exc
    if after > free:
        _opt_draft_count(scheduler, "opt_lookahead_kv_evictions")
        _opt_draft_count(scheduler, "opt_lookahead_kv_evicted_tokens", after - free)
    if after < need:
        try:
            left = int(tree.evictable_size())
        except Exception:
            left = -1
        raise LookaheadFallback(
            f"kv short after evict need={need} free={after} evictable={left}"
        )


def _lookahead_short_of(scheduler, need: int) -> bool:
    """True when the free list holds fewer than ``need`` slots. Does not sync."""
    tree = getattr(scheduler, "tree_cache", None)
    if tree is None:
        return False
    allocator = getattr(tree, "token_to_kv_pool_allocator", None)
    if allocator is None:
        return True
    try:
        available = allocator.available_size()
    except Exception:
        return True
    return int(available) < int(need)


def _lookahead_forward_template(scheduler, batch, raise_errors: bool = False):
    """One ``ForwardBatch`` for every later layer, or None when replay cannot run.

    ``init_new`` may synchronize while copying a pageable M-RoPE delta. That
    happens here, before the layer loop. Later layers copy this object and
    do not call ``init_new`` again.

    An exception from ``init_new`` or ``can_run`` gives None, so the caller
    sees the same result as a missing graph. ``raise_errors=True`` re-raises
    it instead; the group cache stores a miss for every request of that
    size and must not store one caused by a single request.
    """
    worker = getattr(scheduler, "model_worker", None)
    model_runner = getattr(worker, "model_runner", None)
    graph_runner = getattr(model_runner, "graph_runner", None)
    if model_runner is None or graph_runner is None:
        return None
    try:
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        forward_batch = ForwardBatch.init_new(
            batch.get_model_worker_batch(), model_runner
        )
    except Exception:
        if raise_errors:
            raise
        logger.warning("[OPT] lookahead graph check failed", exc_info=True)
        return None
    if not forward_batch.forward_mode.is_cuda_graph():
        return None
    if getattr(forward_batch, "is_sr_tail_extend", False):
        return None
    try:
        ready = bool(graph_runner.can_run(forward_batch))
    except Exception:
        if raise_errors:
            raise
        logger.warning("[OPT] lookahead can_run failed", exc_info=True)
        return None
    if not ready:
        return None
    return forward_batch


def _lookahead_graph_ready(scheduler, batch) -> bool:
    """False when this layer would miss the captured verify graph and run eager."""
    return _lookahead_forward_template(scheduler, batch) is not None


def _mrope_delta_from_template(forward_batch):
    """Device ``[1, 1]`` M-RoPE delta taken from the admission forward.

    ``None`` when the model has no M-RoPE. Later layers add this tensor on
    device. They do not copy the CPU delta, which would synchronize the stream.
    """
    mrope = getattr(forward_batch, "mrope_positions", None)
    positions = getattr(forward_batch, "positions", None)
    if mrope is None or positions is None:
        return None
    pos = positions.detach().reshape(-1)
    rows = mrope.detach().reshape(3, -1)
    if pos.numel() < 1 or rows.shape[1] < 1:
        return None
    return (rows[0, :1] - pos[:1]).to(dtype=torch.int64).reshape(1, 1)


def _request_mrope_delta(req) -> Optional[int]:
    """This request's target-verify M-RoPE delta as a host int.

    Same value ``_compute_spec_mrope_positions`` adds. 0 for a request
    without multimodal inputs. None when it has multimodal inputs but no
    delta: harmless without M-RoPE, but admission falls back when the
    template uses M-RoPE, since the reference would fail there. The
    scheduler leaves the delta on the host; one on the device, or one that
    is not a single value, raises ``LookaheadFallback``.
    """
    mm = getattr(req, "multimodal_inputs", None)
    if mm is None:
        return 0
    delta = getattr(mm, "mrope_position_delta", None)
    if delta is None:
        return None
    device = getattr(delta, "device", None)
    if getattr(delta, "is_cuda", False) or getattr(device, "type", "cpu") != "cpu":
        raise LookaheadFallback("mrope delta on device")
    if not torch.is_tensor(delta):
        return int(delta)
    if int(delta.numel()) != 1:
        raise LookaheadFallback(f"mrope delta shape {tuple(delta.shape)}")
    return int(delta.reshape(-1)[0])


def _lookahead_mrope_positions(positions, delta) -> torch.Tensor:
    """Target-verify M-RoPE: each request's device delta added to its positions.

    ``delta`` is ``[rows, 1]``; positions are ``rows`` equal segments.
    """
    rows = int(delta.shape[0]) if delta.dim() == 2 else 1
    seq_positions = positions.detach().reshape(rows, -1)
    return (seq_positions + delta).flatten().unsqueeze(0).repeat(3, 1)


def _refresh_lookahead_forward(template, batch, delta):
    """Shallow copy with this layer's tokens, lengths, mask, and positions.

    The template object keeps its own tensors. ``req_pool_indices``,
    ``forward_mode``, ``sampling_info``, and the attention backend stay shared.
    ``seq_lens_cpu`` is the new pageable tensor already installed on ``batch``.
    ``TARGET_VERIFY`` ``init_new`` does not fill ``extend_num_tokens`` or the
    extend CPU lists, so this copy leaves those admission values in place.
    """
    fb = copy.copy(template)
    fb.input_ids = batch.input_ids
    fb.seq_lens = batch.seq_lens
    fb.seq_lens_cpu = batch.seq_lens_cpu
    fb.orig_seq_lens = batch.orig_seq_lens
    fb.seq_lens_sum = batch.seq_lens_sum
    fb.out_cache_loc = batch.out_cache_loc
    fb.spec_info = batch.spec_info
    fb.positions = batch.spec_info.positions
    if delta is not None:
        fb.mrope_positions = _lookahead_mrope_positions(fb.positions, delta)
    return fb


def _lookahead_width_cache(scheduler, width: int, device):
    """Pinned slots, events, and device scratch for one tree width.

    The first call allocates. Later steps reuse the same objects. A slot is
    copied again only after ``clone_cpu`` clears ``ready``; the previous
    step's settle or reset has already synchronized that event.
    """
    cache = getattr(scheduler, "_opt_la_cache", None)
    if cache is None:
        cache = {}
        scheduler._opt_la_cache = cache
    width = int(width)
    hit = cache.get(width)
    if hit is not None:
        return hit
    hit = {
        "slots": (_PinnedLayer(width), _PinnedLayer(width)),
        "bonus": _PinnedBonus(width) if bonus_prediction_enabled() else None,
        "positions": torch.empty(width, dtype=torch.long, device=device),
        "dummy_tokens": torch.zeros(width, dtype=torch.long, device=device),
        "launch_event": torch.cuda.Event(),
    }
    cache[width] = hit
    return hit


def _lookahead_batch(scheduler, req):
    """Reuse one decode batch for this request.

    A new ``rid`` or a new KV pool index rebuilds the batch and its sampling
    info. The same request only keeps the existing length tensors; later
    ``fill_`` writes the current lengths.
    """
    batches = getattr(scheduler, "_opt_la_batches", None)
    if batches is None:
        batches = {}
        scheduler._opt_la_batches = batches
    rid = getattr(req, "rid", None)
    pool = getattr(req, "req_pool_idx", None)
    hit = batches.get(rid)
    if hit is not None and hit["pool_idx"] == pool and hit["batch"] is not None:
        batch = hit["batch"]
        if not batch.reqs or batch.reqs[0] is not req:
            batch.reqs[0] = req
        return batch
    batch = scheduler._sr_make_decode_batch([req])
    _keep_raw_logits(batch)
    batches[rid] = {"pool_idx": pool, "batch": batch}
    return batch


def drop_lookahead_request(scheduler, rid) -> None:
    """Drop a finished request's cached decode batch.

    Group templates hold no request, so they stay.
    """
    batches = getattr(scheduler, "_opt_la_batches", None)
    if batches is not None:
        batches.pop(rid, None)


def prepare_lookahead_runtime(scheduler, runner):
    """Attach the cached batch and pinned slots while the GPU is still idle.

    ``_sr_make_decode_batch`` copies sampling tensors with ``torch.tensor``.
    That has to happen before the ingest forward is queued, and only when
    this request does not already have a batch.
    """
    batch = _lookahead_batch(scheduler, runner.req)
    width = _round_width(scheduler)
    device = scheduler.req_to_token_pool.device
    shared = _lookahead_width_cache(scheduler, width, device)
    runner._la_batch = batch
    runner._la_seq = batch.seq_lens
    runner._la_seq_cpu = batch.seq_lens_cpu
    runner._la_orig = batch.orig_seq_lens
    runner._la_pool = batch.req_pool_indices
    runner._la_positions = shared["positions"]
    runner._la_dummy_tokens = shared["dummy_tokens"]
    runner._lookahead_tail = None
    runner._lookahead_bonus = None
    runner._lookahead_launch_event = shared["launch_event"]
    runner._lookahead_queued = False
    runner.lookahead_inflight = False
    runner.lookahead_dropped = False
    runner._lookahead_satisfied = False
    runner._last_logits = None
    runner.bonus_ids = []
    runner._la_forward_template = None
    runner._la_mrope_delta = None
    return batch, shared["slots"], shared["bonus"]


def _reset_lookahead_runner(scheduler, runner) -> None:
    """``_settle_lookahead_rollback`` for one runner."""
    _settle_lookahead_rollback(scheduler, [runner])


def _forget_lookahead_tree(runner, keep: int) -> None:
    """Drop the tree's bookkeeping. Only after its slots were rolled back."""
    runner.layer_slots.clear()
    runner.cached_len = keep
    runner.req.kv_allocated_len = keep
    runner.req.kv_committed_len = keep
    runner.lookahead_inflight = False
    runner._lookahead_queued = False
    runner._lookahead_bonus = None
    _drop_stored_logits(runner)
    runner._last_logits = None
    runner._la_forward_template = None
    runner._la_mrope_delta = None


def _settle_lookahead_rollback(scheduler, runners) -> None:
    """Wait for queued lookahead kernels once, then roll every tree back.

    A runner's slots and lengths are cleared only after its rollback
    succeeded. An ordinary sync failure, or a rollback that raised, leaves
    those runners' slots on the request, quarantines them and raises
    ``LookaheadUnsettled`` so the caller does not rebuild on top of them.
    ``_clear_rollback`` raises here only when ``local_rollback`` refused
    without freeing; a rollback that freed part of the span raises
    ``SRTransferUnresolved``, which is a device-context error. Device-context
    errors poison and re-raise, and the remaining runners are left as they
    are: a poisoned device never returns their slots. A device already
    poisoned must not return slots, so nothing is rolled back.
    """
    from sglang.srt.speculative.standalone_remote.sr_align import (
        is_device_context_error,
    )

    runners = [runner for runner in runners if runner is not None]
    if not runners:
        return
    if getattr(scheduler, "_sr_device_poisoned", False):
        _quarantine_lookahead(scheduler, runners, None)
        raise LookaheadUnsettled("device poisoned before lookahead rollback")
    try:
        torch.get_device_module(scheduler.device).synchronize()
    except Exception as exc:
        if is_device_context_error(exc):
            _reraise_device_context(scheduler, exc)
        logger.warning("[OPT] lookahead reset sync failed", exc_info=True)
        _quarantine_lookahead(scheduler, runners, exc)
        raise LookaheadUnsettled(f"lookahead sync failed: {exc}") from exc
    failed = []
    error = None
    for runner in runners:
        keep = int(runner.prefix_len)
        end = int(getattr(runner.req, "kv_allocated_len", keep) or keep)
        try:
            scheduler._clear_rollback(runner.req, keep)
        except Exception as exc:
            if is_device_context_error(exc):
                _reraise_device_context(scheduler, exc)
            logger.warning(
                "[OPT] lookahead rollback failed for %s",
                getattr(runner.req, "rid", None),
                exc_info=True,
            )
            failed.append(runner)
            error = exc
            continue
        _forget_lookahead_tree(runner, keep)
        _clear_req_to_token_span(scheduler, runner.req, keep, end)
    if failed:
        _quarantine_lookahead(scheduler, failed, error)
        raise LookaheadUnsettled(
            f"lookahead rollback failed for {len(failed)} requests: {error}"
        ) from error


class _UnsettledLookahead:
    """Quarantine owner for lookahead slots whose release is unconfirmed."""

    def __init__(self, entries, error):
        self.entries = entries
        self.error = error


def _quarantine_lookahead(
    scheduler, runners, exc, counter: str = "opt_lookahead_unsettled"
) -> None:
    """Keep these runners' slots, poison the device and record the owner.

    Same rule as ``_quarantine_bonus_batch``: the record carries a
    ``release_error`` so no later quarantine poll frees it. Slots stay mapped
    on each request up to ``kv_allocated_len``; poisoned cleanup owns them.
    """
    from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
        quarantine,
    )

    scheduler._sr_device_poisoned = True
    clear = getattr(scheduler, "_sr_clear_stamps_after_poison", None)
    if callable(clear):
        clear()
    entries = []
    for runner in runners:
        req = getattr(runner, "req", None)
        entries.append(
            (
                req,
                int(getattr(req, "kv_allocated_len", 0) or 0),
                list(getattr(runner, "layer_slots", []) or []),
            )
        )
        runner.lookahead_inflight = False
        runner._lookahead_queued = False
    try:
        record = quarantine(scheduler).retain(_UnsettledLookahead(entries, exc))
        record.release_error = RuntimeError("ClearTree lookahead rollback unresolved")
    except Exception:
        logger.warning("[OPT] lookahead quarantine failed", exc_info=True)
    _opt_draft_count(scheduler, counter, len(runners))


def _confirm_lookahead_done(scheduler, runner) -> bool:
    """Wait for a queued last layer and its bonus copy. True once completion is known.

    An ordinary wait error is confirmed with a device sync. If that fails
    too, completion is unknown: the device is poisoned, and the caller must
    neither publish nor recycle the slots. A device-context error poisons and
    re-raises. ``bonus.ready`` is cleared either way.
    """
    if runner is None:
        return True
    bonus = getattr(runner, "_lookahead_bonus", None)
    try:
        _await_lookahead_event(scheduler, getattr(runner, "_lookahead_launch_event", None))
        if bonus is not None:
            _await_lookahead_event(scheduler, getattr(bonus, "event", None))
            bonus.ready = False
        return True
    except Exception as exc:
        if bonus is not None:
            bonus.ready = False
        _reraise_device_context(scheduler, exc)
        logger.warning("[OPT] lookahead wait failed: %s", exc)
    if getattr(scheduler, "_sr_device_poisoned", False):
        return False
    try:
        torch.get_device_module(scheduler.device).synchronize()
        return True
    except Exception:
        scheduler._sr_device_poisoned = True
        clear = getattr(scheduler, "_sr_clear_stamps_after_poison", None)
        if callable(clear):
            clear()
        return False


def _await_lookahead_event(scheduler, event) -> None:
    if event is None:
        return
    from sglang.srt.speculative.standalone_remote.sr_align import (
        is_device_context_error,
    )

    try:
        event.synchronize()
    except Exception as exc:
        if is_device_context_error(exc):
            _reraise_device_context(scheduler, exc)
        raise


def _private_seed_logits(runner, tail_logits) -> torch.Tensor:
    """Copy the seed row before any verify replay can reuse its buffer.

    EXTEND ingest returns a view of the graph output. A later fallback must
    read this copy, not that view.
    """
    cloned = tail_logits.detach().clone()
    runner._lookahead_tail = cloned
    logits = cloned
    if logits.dim() == 2 and int(logits.shape[0]) == 1:
        logits = logits[0]
    if logits.dim() != 1:
        raise LookaheadFallback(
            f"lookahead seed logits must be one row, got {tuple(tail_logits.shape)}"
        )
    return logits


def _admit_lookahead(scheduler, runner, batch) -> None:
    """Reuse or build the layer template, then reserve KV for the tree.

    Call this before ingest is queued. ``prefix_len`` and ``cached_len`` are
    already the committed length, and ingest does not change them. The
    reservation covers the slots ingest still has to allocate plus the tree,
    not the bonus. It runs after the template so a graph miss does not evict.
    ``_lookahead_reserve_kv`` evicts from a plain radix cache when the free
    list is short; when even that is not enough it raises
    ``LookaheadFallback``, clears the runner's template and leaves the stored
    one in place.

    A template and its M-RoPE delta stay on ``_opt_la_batches[rid]`` until the
    KV pool index changes or the request is dropped. A hit skips ``init_new``
    and ``can_run``. Later layers still report a graph miss through
    ``opt_lookahead_eager_layers``.
    """
    n = _round_width(scheduler)
    _admit_lookahead_rows(scheduler, [runner], n)
    try:
        need = _lookahead_tree_need(scheduler, _lookahead_ingest_slots(runner))
    except Exception as exc:
        raise LookaheadFallback("kv need unknown") from exc
    cached = _load_layer_admit(scheduler, runner.req)
    if cached is not None:
        runner._la_forward_template, runner._la_mrope_delta = cached
        _reserve_lookahead_runner(scheduler, runner, need)
        return
    device = scheduler.req_to_token_pool.device
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )

    runner._la_positions.fill_(int(runner.prefix_len))
    runner._la_seq.fill_(int(runner.cached_len))
    runner._la_seq_cpu.fill_(int(runner.cached_len))
    runner._la_orig.fill_(int(runner.cached_len))
    mask = torch.ones(n * (int(runner.prefix_len) + n), dtype=torch.bool, device=device)
    batch.forward_mode = ForwardMode.TARGET_VERIFY
    batch.global_forward_mode = ForwardMode.TARGET_VERIFY
    batch.input_ids = runner._la_dummy_tokens
    batch.output_ids = None
    batch.seq_lens = runner._la_seq
    batch.seq_lens_cpu = runner._la_seq_cpu
    batch.orig_seq_lens = runner._la_orig
    batch.seq_lens_sum = int(runner.cached_len)
    batch.out_cache_loc = runner._la_dummy_tokens
    batch.extend_lens = [n]
    batch.prefix_lens = [int(runner.cached_len)]
    batch.extend_num_tokens = n
    batch.extend_logprob_start_lens = [int(runner.cached_len)]
    batch.spec_info = ClearLayerSpec(
        custom_mask=mask,
        positions=runner._la_positions,
        draft_token_num=n,
        num_tokens_per_req=n,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )
    runner._la_forward_template = None
    runner._la_mrope_delta = None
    template = _lookahead_forward_template(scheduler, batch)
    if template is None:
        raise LookaheadFallback("cuda graph miss")
    delta = _mrope_delta_from_template(template)
    runner._la_forward_template = template
    runner._la_mrope_delta = delta
    _store_layer_admit(scheduler, runner.req, template, delta)
    _reserve_lookahead_runner(scheduler, runner, need)


def _admit_lookahead_rows(scheduler, runners, width: int) -> None:
    """Lookahead launches layer 1 before any stop; it must fit every row.

    Deeper layers are kept inside the row by ``row_max_depths``. A row with
    no room for one layer leaves lookahead; the ordinary path then sends a
    seed-only tree that writes no layer.
    """
    if _req_row_width(scheduler) is None:
        return
    for runner in runners:
        cap = row_layer_cap(scheduler, int(runner.prefix_len), int(width))
        if cap is not None and cap < 1:
            _opt_draft_count(scheduler, "opt_lookahead_row_short")
            raise LookaheadFallback(
                f"row short rid={getattr(runner.req, 'rid', None)} "
                f"prefix={int(runner.prefix_len)} width={int(width)} "
                f"row={_req_row_width(scheduler)}"
            )


def _reserve_lookahead_runner(scheduler, runner, need: int) -> None:
    """``_lookahead_reserve_kv``; a fallback also clears the runner's template."""
    try:
        _lookahead_reserve_kv(scheduler, need)
    except LookaheadFallback:
        runner._la_forward_template = None
        runner._la_mrope_delta = None
        raise


def grow_lookahead(scheduler, runner, config: ClearTreeConfig, tail_logits, runtime):
    """Grow one tree with the next layer already queued.

    The host mirrors whatever the device top-k selected. It does not run a
    second top-k. A graph miss or an eviction falls back to the caller.
    """
    batch, slots, bonus = runtime
    predict_bonus = bonus_prediction_enabled()
    state = ClearTreeState.create(config)
    caps = row_max_depths(scheduler, [runner], config)
    if caps is not None:
        _apply_depth_cap(state, config, caps[0])
    state.decider = getattr(scheduler, "_clear_round_plan", None)
    rids = list(getattr(scheduler, "_clear_round_rids", None) or [])
    prefixes = list(getattr(scheduler, "_clear_round_prefixes", None) or [])
    if rids:
        state.rid = rids[0]
    if prefixes:
        state.prefix = prefixes[0]
    n = int(config.width)
    turn = 0
    layers = 0
    try:
        logits = _private_seed_logits(runner, tail_logits)
        scores, ids = _vocab_topk(logits, n + 1)
        top_scores, tokens, anc = seed_on_device(scores, ids, n)
        slot = slots[turn]
        turn ^= 1
        slot.copy_seed(scores, ids)
        runner._la_weight = top_scores
        runner._la_anc = anc
        runner.launch_layer(tokens, anc, 1, batch)
        layers += 1
        guard = 0
        while True:
            guard += 1
            if guard > int(config.max_depth) + 1:
                raise RuntimeError(
                    f"OPT lookahead did not finish within max_depth {config.max_depth}"
                )
            cpu_scores, cpu_tokens, _cpu_parents, cpu_ids, kind = slot.clone_cpu()
            if kind == "seed":
                mirror_layer(state, cpu_scores, cpu_tokens, None, seed=True)
                if predict_bonus:
                    runner.bonus_ids.append(cpu_ids[0, : n + 1].contiguous())
            else:
                mirror_layer(state, cpu_scores, cpu_tokens, _cpu_parents, seed=False)
                if predict_bonus:
                    runner.bonus_ids.append(cpu_ids.contiguous())
            # Same hook as ``_commit_prob_rows``: the plan sees the written
            # layer and updates the next-layer gain calibration.
            decider = getattr(state, "decider", None)
            if decider is not None and hasattr(decider, "begin_layer"):
                decider.begin_layer([state])
            step = step_after_layer(state)
            if step.is_final:
                if predict_bonus:
                    bonus_scores, bonus_ids = _vocab_topk(runner._last_logits, n + 1)
                    del bonus_scores
                    bonus.copy(bonus_ids)
                    runner._lookahead_bonus = bonus
                runner.lookahead_inflight = True
                return [_verify_window(step)], state, layers
            select = _begin_select_timing(scheduler)
            child_scores, child_ids = _vocab_topk(runner._last_logits, n + 1)
            top_scores, tokens, parents, anc = expand_on_device(
                runner._la_weight, runner._la_anc, child_scores, child_ids, n
            )
            slot = slots[turn]
            turn ^= 1
            slot.copy_expand(top_scores, tokens, parents, child_ids)
            _finish_select_timing(scheduler, select)
            next_depth = int(state.depth) + 1
            runner._la_weight = top_scores
            runner._la_anc = anc
            runner.launch_layer(tokens, anc, next_depth, batch)
            layers += 1
    except LookaheadFallback:
        scheduler._opt_select_events = []
        _opt_draft_count(scheduler, "opt_lookahead_fallbacks")
        _reset_lookahead_runner(scheduler, runner)
        raise
    except Exception as exc:
        scheduler._opt_select_events = []
        _reraise_device_context(scheduler, exc)
        _opt_draft_count(scheduler, "opt_lookahead_fallbacks")
        _reset_lookahead_runner(scheduler, runner)
        raise LookaheadFallback(str(exc)) from exc


def _free_unrecorded_slots(scheduler, slots, exc) -> None:
    """Return slots that were allocated but never reached ``req_to_token``.

    Rollback only frees what ``kv_allocated_len`` covers, so these would leak.
    A dead device context keeps them for the poisoned-device cleanup.
    """
    if slots is None or getattr(scheduler, "_sr_device_poisoned", False):
        return
    from sglang.srt.speculative.standalone_remote.sr_align import (
        is_device_context_error,
    )

    if is_device_context_error(exc):
        return
    try:
        scheduler.token_to_kv_pool_allocator.free(slots)
    except Exception:
        logger.warning("[OPT] freeing unrecorded layer slots failed", exc_info=True)


def _small_to_device(values, device, dtype=torch.int64) -> torch.Tensor:
    """A small host list on ``device`` without a blocking copy.

    CUDA targets go through pinned memory with ``non_blocking``; PyTorch's
    pinned allocator keeps the block until the copy finishes. Other devices
    copy directly.
    """
    device = torch.device(device)
    if device.type == "cuda":
        return torch.tensor(values, dtype=dtype, pin_memory=True).to(
            device, non_blocking=True
        )
    return torch.tensor(values, dtype=dtype, device=device)


class _PinnedLayerMany:
    """``_PinnedLayer`` for up to ``max_bs`` trees that share a depth.

    Two alternate per width. Row ``i`` is the ``i``-th tree launched in that
    layer; ``rows`` is how many are filled.
    """

    def __init__(self, max_bs: int, width: int, pin_memory: bool = True):
        b = int(max_bs)
        n = int(width)
        pin = {"device": "cpu", "pin_memory": bool(pin_memory)}
        self.scores = torch.empty(b, n, dtype=torch.float32, **pin)
        self.tokens = torch.empty(b, n, dtype=torch.long, **pin)
        self.parents = torch.empty(b, n, dtype=torch.long, **pin)
        predict = bonus_prediction_enabled()
        self.seed_ids = (
            torch.empty(b, n + 1, dtype=torch.long, **pin) if predict else None
        )
        self.ids = torch.empty(b, n, n + 1, dtype=torch.long, **pin) if predict else None
        self.event = torch.cuda.Event()
        self.kind = "seed"
        self.rows = 0
        self.ready = False

    def copy_seed(self, scores: torch.Tensor, ids: torch.Tensor) -> None:
        """``scores`` and ``ids`` are the ``[m, n + 1]`` root top-k rows."""
        m = int(scores.shape[0])
        n = int(self.scores.shape[1])
        self.scores[:m].copy_(scores[:, :n].float(), non_blocking=True)
        self.tokens[:m].copy_(ids[:, :n].to(dtype=torch.long), non_blocking=True)
        if self.seed_ids is not None:
            self.seed_ids[:m].copy_(ids.to(dtype=torch.long), non_blocking=True)
        self.kind = "seed"
        self.rows = m
        self.event.record()
        self.ready = True

    def copy_expand(self, top_scores, tokens, parents, vocab_ids) -> None:
        """``[m, n]`` choices and the ``[m, n, n + 1]`` child ids they came from."""
        m = int(top_scores.shape[0])
        self.scores[:m].copy_(top_scores.float(), non_blocking=True)
        self.tokens[:m].copy_(tokens.to(dtype=torch.long), non_blocking=True)
        self.parents[:m].copy_(parents.to(dtype=torch.long), non_blocking=True)
        if self.ids is not None:
            self.ids[:m].copy_(vocab_ids.to(dtype=torch.long), non_blocking=True)
        self.kind = "expand"
        self.rows = m
        self.event.record()
        self.ready = True

    def clone_cpu(self):
        if not self.ready:
            raise RuntimeError("lookahead pinned group was read before a copy")
        self.event.synchronize()
        self.ready = False
        m = self.rows
        if self.kind == "seed":
            ids = self.seed_ids[:m].clone() if self.seed_ids is not None else None
        else:
            ids = self.ids[:m].clone() if self.ids is not None else None
        return (
            self.scores[:m].clone(),
            self.tokens[:m].clone(),
            self.parents[:m].clone(),
            ids,
            self.kind,
        )


def _lookahead_group_cache(scheduler, width: int, max_bs: int):
    """Pinned group slots and per-row bonus buffers for one width.

    Grows when a larger batch shows up; allocates nothing on a hit.
    """
    cache = getattr(scheduler, "_opt_la_group_cache", None)
    if cache is None:
        cache = {}
        scheduler._opt_la_group_cache = cache
    width = int(width)
    max_bs = int(max_bs)
    hit = cache.get(width)
    if hit is not None and hit["max_bs"] >= max_bs:
        return hit
    hit = {
        "max_bs": max_bs,
        "slots": (_PinnedLayerMany(max_bs, width), _PinnedLayerMany(max_bs, width)),
        "bonus": (
            [_PinnedBonus(width) for _ in range(max_bs)]
            if bonus_prediction_enabled()
            else None
        ),
    }
    cache[width] = hit
    return hit


# Request-bound ``ForwardBatch`` fields. A template that carries any of
# them cannot be replayed for other requests of the same size.
# ``out_cache_loc_swa`` is derived from the admission placeholder slots, so a
# hybrid-SWA model would read stale slots in an eager layer.
_GROUP_REQUEST_FIELDS = (
    "mamba_track_indices",
    "mamba_track_mask",
    "encoder_lens",
    "input_embeds",
    "token_type_ids",
    "ngram_embedding_info",
    "out_cache_loc_swa",
)


def _group_template_shareable(scheduler, template) -> bool:
    """False when a group template would carry one request's state to another."""
    server_args = getattr(scheduler, "server_args", None)
    if bool(getattr(server_args, "enable_lora", False)):
        return False
    return all(
        getattr(template, name, None) is None for name in _GROUP_REQUEST_FIELDS
    )


def _lookahead_group_template(scheduler, runners, width: int):
    """The ``ForwardBatch`` template for ``len(runners)`` trees of ``width``.

    Keyed by ``(width, request count)``. ``runners`` only supply shapes for
    the first build: each layer sets its own pools, M-RoPE positions,
    ``mm_inputs`` and ``rids`` (``_refresh_group_forward``). Only admission
    calls this; layers use ``_cached_group_template``. Building calls
    ``init_new`` and may synchronize.

    A size without a captured graph is cached as a miss. So is a template
    with request-bound fields (LoRA, encoder, mamba, input embeddings, token
    types, ngram embeddings, SWA slots). A build that raised, including in
    ``init_new`` or ``can_run``, is not cached and is retried at the next
    admission.
    """
    cache = getattr(scheduler, "_opt_la_group_templates", None)
    if cache is None:
        cache = {}
        scheduler._opt_la_group_templates = cache
    key = (int(width), len(runners))
    if key in cache:
        return cache[key]
    template = None
    failed = False
    try:
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardMode,
        )

        n = int(width)
        m = len(runners)
        device = scheduler.req_to_token_pool.device
        cached = [int(runner.cached_len) for runner in runners]
        prefixes = [int(runner.prefix_len) for runner in runners]
        batch = scheduler._sr_make_decode_batch([runner.req for runner in runners])
        _keep_raw_logits(batch)
        mask = torch.ones(
            sum(n * (length + n) for length in cached), dtype=torch.bool, device=device
        )
        positions = torch.tensor(
            [prefix for prefix in prefixes for _ in range(n)],
            dtype=torch.long,
            device=device,
        )
        dummy = torch.zeros(m * n, dtype=torch.long, device=device)
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.global_forward_mode = ForwardMode.TARGET_VERIFY
        batch.input_ids = dummy
        batch.output_ids = None
        batch.seq_lens = torch.tensor(cached, dtype=torch.int64, device=device)
        batch.seq_lens_cpu = torch.tensor(cached, dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(cached, dtype=torch.int32, device=device)
        batch.seq_lens_sum = sum(cached)
        batch.out_cache_loc = dummy
        batch.extend_lens = [n] * m
        batch.prefix_lens = list(cached)
        batch.extend_num_tokens = n * m
        batch.extend_logprob_start_lens = list(cached)
        batch.spec_info = ClearLayerSpec(
            custom_mask=mask,
            positions=positions,
            draft_token_num=n,
            num_tokens_per_req=n,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        template = _lookahead_forward_template(scheduler, batch, raise_errors=True)
    except Exception as exc:
        _reraise_device_context(scheduler, exc)
        logger.warning("[OPT] lookahead group template failed", exc_info=True)
        template = None
        failed = True
    if failed:
        _opt_draft_count(scheduler, "opt_lookahead_group_template_errors")
        return None
    if template is not None:
        if _group_template_shareable(scheduler, template):
            # Do not keep the first group's image features or ids alive.
            template.mm_inputs = None
            template.rids = None
        else:
            _opt_draft_count(scheduler, "opt_lookahead_group_template_unshared")
            template = None
    cache[key] = template
    if template is None:
        _opt_draft_count(scheduler, "opt_lookahead_group_template_misses")
    else:
        _opt_draft_count(scheduler, "opt_lookahead_group_templates")
    return template


def _cached_group_template(scheduler, width: int, count: int):
    """The group template admission built, or None. Never builds one."""
    cache = getattr(scheduler, "_opt_la_group_templates", None)
    if not cache:
        return None
    return cache.get((int(width), int(count)))


def _admit_lookahead_many(scheduler, runners, width: int) -> None:
    """KV budget, per-request M-RoPE deltas and every group size, before ingest.

    The KV reservation is every tree's ingest plus layers together, not the
    bonus. It runs last, so a graph miss or a missing M-RoPE delta does not
    evict; ``_lookahead_reserve_kv`` evicts from a plain radix cache to
    cover it.

    Every row must also hold layer 1 (``_admit_lookahead_rows``); deeper
    layers stay inside the row through ``row_max_depths``.

    Templates for ``1 .. len(runners)`` trees are built here so no layer
    builds one. Only a miss at the full size falls back; a smaller size
    without a template runs that layer through ``_verify_batch_many``.
    """
    _admit_lookahead_rows(scheduler, runners, width)
    try:
        need = sum(
            _lookahead_tree_need(scheduler, _lookahead_ingest_slots(runner))
            for runner in runners
        )
    except Exception as exc:
        raise LookaheadFallback("kv need unknown") from exc
    for runner in runners:
        runner._la_group_delta = _request_mrope_delta(runner.req)
    template = None
    for m in range(1, len(runners) + 1):
        template = _lookahead_group_template(scheduler, runners[:m], width)
    if template is None:
        raise LookaheadFallback("cuda graph miss")
    if getattr(template, "mrope_positions", None) is not None and any(
        runner._la_group_delta is None for runner in runners
    ):
        raise LookaheadFallback("mrope delta missing")
    _lookahead_reserve_kv(scheduler, need)


def _refresh_group_forward(template, source, pools, deltas, reqs):
    """``_refresh_lookahead_forward`` plus this layer's requests.

    The template is shared by every group of the same size, so the copy
    gets these requests' pool rows, ``mm_inputs`` and ``rids``, and M-RoPE
    positions from their host deltas (``deltas`` is ``[m]`` on device).
    """
    fb = _refresh_lookahead_forward(template, source, None)
    fb.req_pool_indices = pools
    fb.mm_inputs = [getattr(req, "multimodal_inputs", None) for req in reqs]
    fb.rids = [req.rid for req in reqs]
    if getattr(template, "mrope_positions", None) is not None:
        fb.mrope_positions = _lookahead_mrope_positions(
            fb.positions, deltas.reshape(-1, 1)
        )
    return fb


def _launch_layer_many(scheduler, runners, tokens_dev, anc_dev, depth: int, width: int, event):
    """Queue one layer for every tree in ``runners``. Returns ``[m * n, vocab]``.

    ``tokens_dev`` is ``[m, n]`` and ``anc_dev`` is ``[m, n, depth * n]``, both
    on device. Lengths, pool rows and M-RoPE deltas go through one pinned
    copy, so nothing here synchronizes once the group template exists.
    """
    from sglang.srt.mem_cache.common import alloc_token_slots
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
    from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

    n = int(width)
    m = len(runners)
    depth = int(depth)
    _require_layer_width(n)
    if tuple(tokens_dev.shape) != (m, n):
        raise RuntimeError(f"OPT group tokens {tuple(tokens_dev.shape)} != [{m}, {n}]")
    if tuple(anc_dev.shape) != (m, n, depth * n):
        raise RuntimeError(
            f"OPT group ancestor mask {tuple(anc_dev.shape)} != [{m}, {n}, {depth * n}]"
        )
    for runner in runners:
        if runner.cached_len + n != runner.prefix_len + depth * n:
            raise RuntimeError(
                f"OPT cache length {runner.cached_len} is not layer {depth} "
                f"after prefix {runner.prefix_len}"
            )
    timing = _begin_layer_timing(scheduler)
    device = scheduler.req_to_token_pool.device
    cached = [int(runner.cached_len) for runner in runners]
    prefixes = [int(runner.prefix_len) for runner in runners]
    pools = [int(runner.req.req_pool_idx) for runner in runners]
    template = _cached_group_template(scheduler, n, m)
    mrope = template is not None and getattr(template, "mrope_positions", None) is not None
    deltas = []
    for runner in runners:
        if hasattr(runner, "_la_group_delta"):
            delta = runner._la_group_delta
        else:
            delta = _request_mrope_delta(runner.req)
        if delta is None and mrope:
            raise RuntimeError(f"OPT lookahead request {runner.req.rid} has no M-RoPE delta")
        deltas.append(0 if delta is None else int(delta))
    staged = _small_to_device([cached, prefixes, pools, deltas], device)
    seq_dev = staged[0]
    positions = (staged[1] + (depth - 1)).repeat_interleave(n)
    pool_dev = staged[2]
    ancestors = [anc_dev[i] for i in range(m)]
    if _cascade_on(runners[0]):
        mask = anc_dev.reshape(-1).to(dtype=torch.bool)
    else:
        # Committed prefix of ones, then this tree's ancestor columns.
        mask = torch.cat(
            [
                layer_attention_mask(prefix, ancestor, device)
                for prefix, ancestor in zip(prefixes, ancestors)
            ]
        )
    _require_row_room(scheduler, runners, n, starts=cached)
    out_cache_loc = alloc_token_slots(scheduler.tree_cache, m * n)
    try:
        if int(out_cache_loc.numel()) != m * n:
            raise RuntimeError(
                f"OPT group layer allocated {int(out_cache_loc.numel())} slots for {m * n}"
            )
        assign_req_to_token_pool_func(
            pool_dev,
            scheduler.req_to_token_pool.req_to_token,
            seq_dev,
            seq_dev + n,
            out_cache_loc,
            m,
        )
    except BaseException as exc:
        _free_unrecorded_slots(scheduler, out_cache_loc, exc)
        raise
    # Cover every row by rollback first; a later copy failure then leaks nothing.
    for runner in runners:
        runner.req.kv_allocated_len = runner.cached_len + n
    for i, runner in enumerate(runners):
        runner.layer_slots.append(out_cache_loc[i * n : (i + 1) * n].clone())
    spec = ClearLayerSpec(
        custom_mask=mask,
        positions=positions,
        draft_token_num=n,
        num_tokens_per_req=n,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )
    bind_cascade(spec, runners, ancestors, n, cached_lens=cached)
    _close_layer_host(timing)
    _mark_layer_launch(timing)
    if template is not None:
        from types import SimpleNamespace

        from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
            remember_layer_path,
        )

        source = SimpleNamespace(
            input_ids=tokens_dev.reshape(-1).to(dtype=torch.int64),
            seq_lens=seq_dev,
            seq_lens_cpu=torch.tensor(cached, dtype=torch.int64),
            orig_seq_lens=seq_dev.to(dtype=torch.int32),
            seq_lens_sum=sum(cached),
            out_cache_loc=out_cache_loc,
            spec_info=spec,
        )
        forward_batch = _refresh_group_forward(
            template, source, pool_dev, staged[3], [runner.req for runner in runners]
        )
        out = scheduler.model_worker.model_runner.forward(forward_batch)
        remember_layer_path(scheduler, out)
        if not bool(getattr(out, "can_run_graph", True)):
            _opt_draft_count(scheduler, "opt_lookahead_eager_layers")
        logits = getattr(getattr(out, "logits_output", None), "next_token_logits", None)
        if logits is None:
            raise RuntimeError("OPT lookahead group forward returned no logits")
    else:
        _opt_draft_count(scheduler, "opt_lookahead_template_fallbacks")
        batch = _verify_batch_many(
            scheduler,
            runners,
            [tokens_dev[i] for i in range(m)],
            positions,
            mask,
            out_cache_loc,
            cached,
        )
        bind_cascade(batch.spec_info, runners, ancestors, n, cached_lens=cached)
        logits = _run_batch(scheduler, batch)
    event.record()
    _queue_ledger_layer(scheduler, timing, runners, depth, n)
    _finish_layer_timing(scheduler, timing)
    for runner in runners:
        runner.cached_len += n
    _opt_draft_count(scheduler, "opt_lookahead_layers")
    flat = _flatten_logits(logits)
    if int(flat.shape[0]) != m * n:
        raise RuntimeError(f"OPT group layer logits rows {int(flat.shape[0])} != {m * n}")
    return flat


def _reset_lookahead_group(scheduler, runners) -> None:
    """One sync for the batch, then per-request rollback (``_settle_lookahead_rollback``)."""
    for runner in runners:
        runner.bonus_ids = []
    _settle_lookahead_rollback(scheduler, runners)


def grow_lookahead_many(scheduler, runners, config: ClearTreeConfig, tails, states_out):
    """Grow every tree with the next layer already queued.

    Same decisions as ``grow_active_layers``: each written layer is mirrored on
    the host, the round plan judges all trees together, and stopped trees are
    cut to one K at the end. Layer ``d`` is launched for every live tree before
    that judgment, so a tree that stops at ``d`` already has its last layer
    queued, as in ``grow_lookahead``. ``states_out`` receives the trees in
    request order. Returns ``(windows, launched)``. Any failure resets every
    runner and raises ``LookaheadFallback``; the caller then clears
    ``states_out`` and grows from each runner's ``_lookahead_tail``.
    """
    n = int(config.width)
    count = len(runners)
    predict_bonus = bonus_prediction_enabled()
    results: List[Optional[tuple]] = [None] * count
    pending: List[int] = []
    launched = 0
    try:
        # Setup sits inside the try: a pinned-memory or plan failure here must
        # also fall back instead of failing the whole batch expand.
        cache = _lookahead_group_cache(scheduler, n, count)
        slots = cache["slots"]
        bonuses = cache["bonus"]
        states, plan = _round_states(
            scheduler, config, count, row_max_depths(scheduler, runners, config)
        )
        states_out.extend(states)
        for runner in runners:
            runner.bonus_ids = []
            runner.lookahead_inflight = False
            runner.lookahead_dropped = False
            runner._lookahead_satisfied = False
            runner._lookahead_bonus = None
            runner._lookahead_launch_event = None
            runner._last_logits = None
        rows = torch.stack(
            [_flatten_logits(tail.detach())[-1] for tail in tails], dim=0
        ).clone()
        for index, runner in enumerate(runners):
            runner._lookahead_tail = rows[index]
        scores, ids = _vocab_topk(rows, n + 1)
        weight, tokens, anc = seed_on_device_many(scores, ids, n)
        turn = 0
        slot = slots[turn]
        turn ^= 1
        slot.copy_seed(scores, ids)
        active = list(range(count))
        event = torch.cuda.Event()
        logits = _launch_layer_many(
            scheduler, [runners[i] for i in active], tokens, anc, 1, n, event
        )
        launched += 1
        guard = 0
        while True:
            guard += 1
            if guard > int(config.max_depth) + 1:
                raise RuntimeError(
                    f"OPT group lookahead did not finish within max_depth {config.max_depth}"
                )
            cpu_scores, cpu_tokens, cpu_parents, cpu_ids, kind = slot.clone_cpu()
            touched = []
            for row, index in enumerate(active):
                state = states[index]
                if kind == "seed":
                    mirror_layer(state, cpu_scores[row], cpu_tokens[row], None, seed=True)
                else:
                    mirror_layer(
                        state, cpu_scores[row], cpu_tokens[row], cpu_parents[row], seed=False
                    )
                if predict_bonus and cpu_ids is not None:
                    runners[index].bonus_ids.append(cpu_ids[row].contiguous())
                touched.append(state)
            if plan is not None and hasattr(plan, "begin_layer"):
                plan.begin_layer(touched)
            layer_rows = logits.reshape(len(active), n, -1)
            keep_rows = []
            keep = []
            for row, index in enumerate(active):
                step = step_after_layer(states[index])
                if not step.is_final:
                    keep_rows.append(row)
                    keep.append(index)
                    continue
                if step.verify is None:
                    pending.append(index)
                else:
                    results[index] = _verify_window(step)
                runner = runners[index]
                if predict_bonus and bonuses is not None:
                    # Queued before the next launch can overwrite this view.
                    _scores, final_ids = _vocab_topk(layer_rows[row], n + 1)
                    del _scores
                    bonuses[index].copy(final_ids)
                    runner._lookahead_bonus = bonuses[index]
                runner._lookahead_launch_event = event
                runner.lookahead_inflight = True
            if not keep:
                break
            select = _begin_select_timing(scheduler)
            if len(keep_rows) == len(active):
                chosen = layer_rows
                prev_weight = weight
                prev_anc = anc
            else:
                picked = _small_to_device(keep_rows, layer_rows.device)
                chosen = layer_rows.index_select(0, picked)
                prev_weight = weight.index_select(0, picked)
                prev_anc = anc.index_select(0, picked)
            child_scores, child_ids = _vocab_topk(chosen.reshape(-1, chosen.shape[-1]), n + 1)
            child_scores = child_scores.reshape(len(keep), n, n + 1)
            child_ids = child_ids.reshape(len(keep), n, n + 1)
            weight, tokens, parents, anc = expand_on_device_many(
                prev_weight, prev_anc, child_scores, child_ids, n
            )
            slot = slots[turn]
            turn ^= 1
            slot.copy_expand(weight, tokens, parents, child_ids)
            _finish_select_timing(scheduler, select)
            active = keep
            depth = int(states[active[0]].depth) + 1
            event = torch.cuda.Event()
            logits = _launch_layer_many(
                scheduler, [runners[i] for i in active], tokens, anc, depth, n, event
            )
            launched += 1
        # Same order as ``grow_active_layers``: T_K starts at the last stop,
        # so the K cut is not charged to the last layer's wall.
        if scheduler is not None:
            from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
                mark_stop,
            )

            mark_stop(scheduler)
        _cut_deferred(plan, states, pending, results)
        if any(window is None for window in results):
            raise RuntimeError("OPT group lookahead finished without every window")
        return results, launched
    except LookaheadFallback:
        scheduler._opt_select_events = []
        _opt_draft_count(scheduler, "opt_lookahead_fallbacks")
        _reset_lookahead_group(scheduler, runners)
        raise
    except Exception as exc:
        scheduler._opt_select_events = []
        _reraise_device_context(scheduler, exc)
        _opt_draft_count(scheduler, "opt_lookahead_fallbacks")
        _reset_lookahead_group(scheduler, runners)
        raise LookaheadFallback(str(exc)) from exc


def _settle_lookahead_job(scheduler, job: OptDeferredTree) -> None:
    """Install the speculative last layer, or roll the lookahead tree back to ``keep``."""
    runner = job.runner
    if runner is None or not getattr(runner, "lookahead_inflight", False):
        return
    poisoned = bool(getattr(scheduler, "_sr_device_poisoned", False))
    drop = (
        poisoned
        or _request_finished(scheduler, job.req)
        or _opt_page_size(scheduler) != 1
        or int(runner.cached_len)
        != int(runner.prefix_len) + int(job.state.depth) * int(job.state.config.width)
    )
    if poisoned:
        runner.lookahead_dropped = True
        runner.lookahead_inflight = False
        return
    if drop:
        runner.lookahead_dropped = True
        if not _confirm_lookahead_done(scheduler, runner):
            # Completion unknown: neither recycle nor clear the tree's slots.
            _quarantine_lookahead(scheduler, [runner], None)
            return
        # ``keep`` is the committed prefix. Rollback frees the whole tree,
        # and local_rollback then sets ``kv_allocated_len`` to ``keep``.
        end = int(getattr(job.req, "kv_allocated_len", job.keep) or job.keep)
        try:
            scheduler._clear_rollback(job.req, int(job.keep))
        except Exception as exc:
            _reraise_device_context(scheduler, exc)
            logger.warning(
                "[OPT] rollback of the lookahead tree failed for %s",
                getattr(job.req, "rid", None),
                exc_info=True,
            )
            # Release is unconfirmed: keep the slots and lengths on the request.
            _quarantine_lookahead(scheduler, [runner], exc)
            return
        _clear_req_to_token_span(scheduler, job.req, int(job.keep), end)
        runner.layer_slots.clear()
        runner.cached_len = int(runner.prefix_len)
        job.req.kv_allocated_len = int(job.keep)
        job.req.kv_committed_len = int(job.keep)
        runner.lookahead_dropped = True
        runner.lookahead_inflight = False
        _opt_draft_count(scheduler, "opt_lookahead_rollbacks")
        return
    bonus = getattr(runner, "_lookahead_bonus", None)
    bonus_failed = False
    try:
        if bonus is not None:
            runner.bonus_ids.append(bonus.clone_cpu())
        else:
            # With prediction off there is no bonus D2H event to wait on.
            # Settle the queued final layer before publishing its KV slots.
            # Unknown completion poisons the device; poisoned cleanup then
            # owns the slots and publish keeps them.
            if not _confirm_lookahead_done(scheduler, runner):
                raise RuntimeError("lookahead final layer completion unknown")
    except Exception as exc:
        _reraise_device_context(scheduler, exc)
        # A failed bonus-copy event is not proof that the queued final
        # layer completed. Confirm completion before publishing tree slots.
        if bonus is not None and not getattr(scheduler, "_sr_device_poisoned", False):
            try:
                torch.get_device_module(scheduler.device).synchronize()
            except Exception:
                scheduler._sr_device_poisoned = True
                scheduler._sr_clear_stamps_after_poison()
        bonus_failed = True
        logger.warning(
            "[OPT] lookahead settle failed for %s: %s",
            getattr(getattr(runner, "req", None), "rid", None),
            exc,
        )
    finally:
        # The layer forward is already queued. A second forward would allocate
        # another N slots on top of it. A failed id copy still counts as
        # settled, and the dropped flag skips the bonus forward.
        runner.lookahead_inflight = False
        runner._lookahead_satisfied = True
        if bonus_failed:
            runner.lookahead_dropped = True


def forward_deferred_final_layers(scheduler, jobs) -> None:
    """Forward each deferred last layer, then publish or roll it back.

    Layers forwarded in this function are marked post-reply for the ledger.

    When enabled, bonus guesses are written while the tree is still mapped.
    Publish clears those columns, so the guess forward cannot follow it.
    A lookahead job already queued that last layer. It waits for completion
    (and installs bonus ids when enabled), unless it must drop the slots. Each
    finished ``_forward_verify_batch`` increments
    ``opt_layer_forwards_deferred``. A depth group that fits in one forward
    still counts once. A group split by the KV budget counts once per
    sub-batch. A failed group adds nothing.
    """
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        set_post_reply,
    )

    set_post_reply(scheduler, True)
    try:
        for job in jobs:
            try:
                _settle_lookahead_job(scheduler, job)
            except Exception as exc:
                _reraise_device_context(scheduler, exc)
                logger.warning("[OPT] lookahead settle failed: %s", exc)
        by_depth = {}
        skipped = []
        for job in jobs:
            runner = job.runner
            if (
                getattr(runner, "lookahead_dropped", False)
                or _request_finished(scheduler, job.req)
                or getattr(scheduler, "_sr_device_poisoned", False)
            ):
                skipped.append(job)
                continue
            if getattr(runner, "_lookahead_satisfied", False):
                continue
            room = row_layer_cap(
                scheduler, int(runner.cached_len), int(job.state.config.width)
            )
            if room is not None and room < 1:
                # Only a seed-only tree on a full row gets here
                # (``row_max_depths``); with no layer slots it rolls back.
                _opt_draft_count(scheduler, "opt_row_final_skipped")
                skipped.append(job)
                continue
            by_depth.setdefault(int(job.state.depth), []).append(job)
        for group in by_depth.values():
            try:
                _rows, n_forwards = _forward_verify_fitting(
                    scheduler,
                    [job.runner for job in group],
                    [final_layer_step(job.state) for job in group],
                    consume=_append_layer_bonus_ids,
                )
            except Exception as exc:
                for job in group:
                    _drop_unfinished_layer(scheduler, job)
                _reraise_device_context(scheduler, exc)
                logger.warning("[OPT] deferred final layer failed: %s", exc)
            else:
                _opt_draft_count(scheduler, "opt_layer_forwards_deferred", n_forwards)
        skipped_ids = {id(job) for job in skipped}
        for job in skipped:
            _drop_stored_logits(job.runner)
        maybe_attach_bonus_guesses_many(
            scheduler, [job for job in jobs if id(job) not in skipped_ids]
        )
    finally:
        set_post_reply(scheduler, False)
        for job in jobs:
            try:
                _publish_or_rollback(
                    scheduler, job.req, job.runner, job.keep, job.window, True
                )
            finally:
                if not _request_finished(scheduler, job.req):
                    scheduler._sr_pause_req(job.req)
        scheduler.last_batch = None
        scheduler.cur_batch = None


def finish_opt_expand_req(
    scheduler, req, runner, keep, window, expanded: bool, state=None
) -> Optional[OptDeferredTree]:
    """Publish this expand, or return a job for the post-reply forward.

    A returned job is not paused and its slots stay on the request until
    ``commit_deferred_expands`` queues it.
    """
    wait_error = None
    if (
        runner is not None
        and getattr(runner, "lookahead_inflight", False)
        and not (
            _deferrable_state(state) and _can_defer(scheduler, req, expanded, window)
        )
    ):
        # This publish frees slots before the reply. The extra layer is still
        # queued, so wait for it first. The deferred path waits after the reply.
        # A failed wait still publishes or rolls back below: unknown completion
        # poisons the device, and publish then keeps the slots.
        try:
            if not _retire_lookahead_before_publish(scheduler, runner):
                wait_error = LookaheadUnsettled(
                    "lookahead final layer completion unknown before publish"
                )
        except Exception as exc:
            wait_error = exc
    if (
        runner is not None
        and _deferrable_state(state)
        and _can_defer(scheduler, req, expanded, window)
    ):
        return OptDeferredTree(
            req=req,
            runner=runner,
            keep=int(keep),
            window=window,
            state=state,
        )
    publish_error = None
    try:
        _publish_or_rollback(scheduler, req, runner, keep, window, expanded)
    except Exception as exc:
        publish_error = exc
    if not _request_finished(scheduler, req):
        scheduler._sr_pause_req(req)
    if publish_error is not None:
        raise publish_error
    if wait_error is not None:
        raise wait_error
    return None


def _retire_lookahead_before_publish(scheduler, runner) -> bool:
    """Wait out a queued lookahead last layer before its slots are published.

    Shared by an immediate publish and by cancelling deferred jobs. Returns
    whether completion is known (``_confirm_lookahead_done``); when it is
    not, the device is poisoned and publish keeps the slots. The runner no
    longer counts as in flight either way.
    """
    if runner is None or not getattr(runner, "lookahead_inflight", False):
        return True
    try:
        return _confirm_lookahead_done(scheduler, runner)
    finally:
        runner.lookahead_inflight = False
        runner.lookahead_dropped = True


def commit_deferred_expands(scheduler, jobs, *, keep: bool) -> Optional[BaseException]:
    """Queue jobs after a clean expand, or publish them when expand is failing.

    A failing expand can still hold a lookahead last layer in flight (the
    tree finished, then something later raised). That layer is waited out
    before its slots are published or rolled back.
    """
    if keep:
        bucket = getattr(scheduler, "_opt_deferred", None)
        if isinstance(bucket, list):
            bucket.extend(jobs)
            return None
    error = None
    for job in jobs:
        try:
            _retire_lookahead_before_publish(scheduler, job.runner)
            _publish_or_rollback(
                scheduler, job.req, job.runner, job.keep, job.window, True
            )
        except Exception as exc:
            logger.warning(
                "[OPT] rollback after deferred expand failed for %s: %s",
                getattr(job.req, "rid", None),
                exc,
            )
            error = exc
        if not _request_finished(scheduler, job.req):
            scheduler._sr_pause_req(job.req)
    return error


def _publish_or_rollback(scheduler, req, runner, keep, window, expanded: bool) -> None:
    """Keep a page-1 tree lease, or roll slots back to ``keep``.

    The caller turns a rollback error into the expand failure when no earlier
    exception is already propagating.
    """
    from sglang.srt.speculative.clear_tree_remote.clear_tree_lease import (
        publish_clear_tree_lease,
        release_opt_bonus_guesses,
    )

    published = False
    finished = False
    checker = getattr(scheduler, "_sr_is_finished", None)
    if callable(checker):
        try:
            finished = bool(checker(req))
        except Exception:
            finished = True
    if (
        expanded
        and not finished
        and not getattr(scheduler, "_sr_device_poisoned", False)
    ):
        try:
            published = publish_clear_tree_lease(scheduler, req, runner, keep, window)
        except Exception as exc:
            logger.warning(
                "[OPT] lease publish failed for %s: %s",
                getattr(req, "rid", None),
                exc,
            )
        if not published and _opt_page_size(scheduler) == 1:
            _opt_draft_count(scheduler, "opt_lease_publish_failed")
    if published or getattr(scheduler, "_sr_device_poisoned", False):
        return
    release_opt_bonus_guesses(scheduler, req)
    scheduler._clear_rollback(req, keep)


def grow_batch(scheduler, reqs):
    """Grow every request together.

    A missing output that fits in one chunk shares one EXTEND. A longer output
    is extended in ``chunked_prefill_size`` pieces, and only the last piece's
    row is kept. One or two missing tokens on page size 1 are DECODE steps.
    Requests that still have the same depth share one TARGET_VERIFY. On page
    size 1, a clean finish keeps forwarded slots in ``req.clear_tree_lease``.
    When post-reply reuse is on, that publish waits until the last layer has
    been forwarded. Every other finish rolls those slots back to the committed
    length.
    """
    if not reqs:
        return []
    from sglang.srt.speculative.clear_tree_remote.clear_tree_lease import (
        consume_clear_tree_lease,
    )
    from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
        get_sr_round_metrics,
    )

    committed = [_committed_len(req) for req in reqs]
    runners: List[OptLayerForward] = []
    states: List[ClearTreeState] = []
    windows = None
    expanded = False
    metrics = get_sr_round_metrics(scheduler, "Draft")
    try:
        resume_start = time.perf_counter()
        for req in reqs:
            scheduler._sr_resume_req(req)
        scheduler._sr_isolate_need({req.rid for req in reqs})
        metrics.add_host("opt_resume", time.perf_counter() - resume_start)
        for req in reqs:
            consume_clear_tree_lease(scheduler, req)
        for req in reqs:
            scheduler._clear_release_uncommitted(req)
        from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import (
            attach_round,
            remember_choice,
        )

        plan = attach_round(scheduler, reqs)
        config = ClearTreeConfig(
            width=int(plan.width),
            tau=float(scheduler.server_args.clear_tree_tau),
            max_depth=int(scheduler.server_args.clear_tree_max_depth),
        )
        runners = [
            OptLayerForward(scheduler, req, keep) for req, keep in zip(reqs, committed)
        ]
        scheduler._opt_lookahead_active = False
        scheduler._opt_defer_layer_drain = False
        if len(runners) > 1 and lookahead_supported(scheduler):
            t_admit = time.perf_counter()
            try:
                # Before ingest: the budget counts the slots ingest will take.
                _admit_lookahead_many(scheduler, runners, int(plan.width))
                scheduler._opt_lookahead_active = True
            except LookaheadFallback as exc:
                logger.info("[OPT] group lookahead admit fell back: %s", exc)
                metrics.counts["opt_lookahead_fallbacks"] += 1
            except Exception as exc:
                # Admission has not allocated KV, so the layer-by-layer path
                # can still grow this step.
                _reraise_device_context(scheduler, exc)
                logger.info("[OPT] group lookahead admit fell back: %s", exc)
                metrics.counts["opt_lookahead_fallbacks"] += 1
            finally:
                if metrics.active:
                    metrics.counts["opt_lookahead_admit_us"] += int(
                        (time.perf_counter() - t_admit) * 1e6
                    )

        def _sync():
            # Lookahead times layers with events; a sync here would also wait
            # out the ingest forward the seed overlaps with.
            if metrics.active and not scheduler._opt_lookahead_active:
                torch.get_device_module(scheduler.device).synchronize()

        phases = OptPhaseClock()
        t0 = time.perf_counter()
        tail_logits = ingest_tail_logits(scheduler, runners, phases=phases)
        _sync()
        phases.lap("opt_ingest_forward")
        flush_opt_ingest_phases(metrics, phases.parts, time.perf_counter() - t0)
        for runner, keep in zip(runners, committed):
            written = int(getattr(runner.req, "kv_committed_len", 0) or 0)
            if written != keep:
                raise RuntimeError(
                    f"OPT draft {runner.req.rid} committed KV {written} != {keep} tokens"
                )
        layers = 0

        def forward_active(steps, indices):
            nonlocal layers
            rows, n_forwards = _forward_verify_fitting(
                scheduler, [runners[i] for i in indices], steps
            )
            layers += n_forwards
            return rows

        t1 = time.perf_counter()
        topk_out = [[] for _ in runners] if bonus_prediction_enabled() else None
        windows = None
        seed_tails = tail_logits
        if scheduler._opt_lookahead_active:
            from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import (
                restore_round_learning,
                snapshot_round_learning,
            )

            round_plan = getattr(scheduler, "_clear_round_plan", None)
            learning = snapshot_round_learning(round_plan)
            try:
                windows, launched = grow_lookahead_many(
                    scheduler, runners, config, tail_logits, states
                )
                layers += launched
                metrics.counts["opt_layer_forwards_critical"] += critical_layer_forwards(
                    launched
                )
                # The last layers are still queued; read their events after the reply.
                scheduler._opt_defer_layer_drain = True
            except LookaheadFallback as exc:
                logger.info(
                    "[OPT] group lookahead fell back for %d requests: %s",
                    len(runners),
                    exc,
                )
                # The failed attempt already queued layer samples and policy
                # notes; keep this round out of the fits, undo its gain
                # observations, and skip its width replay.
                from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
                    mark_round_fallback,
                )

                mark_round_fallback(scheduler)
                restore_round_learning(round_plan, learning)
                scheduler._clear_round_retried = True
                windows = None
                states.clear()
                scheduler._opt_lookahead_active = False
                # Graph replays may have overwritten the ingest views.
                seed_tails = [
                    saved if saved is not None else tail
                    for saved, tail in zip(
                        [getattr(runner, "_lookahead_tail", None) for runner in runners],
                        tail_logits,
                    )
                ]
            except LookaheadUnsettled:
                # Slots are quarantined; the round fails without a rebuild.
                from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
                    mark_round_fallback,
                )

                mark_round_fallback(scheduler)
                restore_round_learning(round_plan, learning)
                scheduler._clear_round_retried = True
                raise
        if windows is None:
            windows = grow_active_layers(
                config,
                seed_tails,
                forward_active,
                scheduler,
                states_out=states,
                topk_out=topk_out,
                depth_caps=row_max_depths(scheduler, runners, config),
            )
            for index, runner in enumerate(runners):
                runner.bonus_ids = topk_out[index] if topk_out is not None else []
        metrics.add_host("opt_grow", time.perf_counter() - t1)
        metrics.counts["opt_layer_forwards"] += layers
        metrics.counts["opt_rounds"] += 1
        expanded = True
        from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
            mark_stop,
        )

        if windows:
            remember_choice(scheduler, len(windows[0][0]), plan.width)
            from sglang.srt.speculative.clear_tree_remote.clear_tree_replay import (
                queue_hold_replay,
            )

            count_row_capped(scheduler, states, windows)
            note_forwarded_layers(states, runners, config.width)
            queue_hold_replay(scheduler, reqs, states, plan)
        mark_stop(scheduler)
        return windows
    finally:
        scheduler._opt_lookahead_active = False
        if not getattr(scheduler, "_opt_defer_layer_drain", False):
            drain_opt_layer_gpu_us(scheduler, metrics)
        finish_start = time.perf_counter()
        try:
            pending = sys.exc_info()[1]
            rollback_error = None
            deferred = []
            for index, (req, keep) in enumerate(zip(reqs, committed)):
                clear_prefill_tail_logits(req)
                runner = runners[index] if index < len(runners) else None
                window = None
                if expanded and windows is not None and index < len(windows):
                    window = windows[index]
                state = states[index] if index < len(states) else None
                try:
                    job = finish_opt_expand_req(
                        scheduler, req, runner, keep, window, expanded, state
                    )
                except Exception as rollback_exc:
                    logger.warning(
                        "[OPT] rollback after batch expand failed for %s: %s",
                        getattr(req, "rid", None),
                        rollback_exc,
                    )
                    rollback_error = rollback_exc
                    job = None
                if job is not None:
                    deferred.append(job)
            cancel_error = commit_deferred_expands(
                scheduler,
                deferred,
                keep=pending is None and rollback_error is None,
            )
            if cancel_error is not None and rollback_error is None:
                rollback_error = cancel_error
            scheduler.last_batch = None
            scheduler.cur_batch = None
            if rollback_error is not None and pending is None:
                raise rollback_error
        finally:
            metrics.add_host("opt_finish", time.perf_counter() - finish_start)
