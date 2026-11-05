"""One OPT-Tree forward per layer.

The committed tail is scored once. Each non-final ``update`` then runs N
tokens at one shared position under ``ancestor_mask``. Speculative KV is
appended past the committed prefix. On CUDA page size 1 a successful expand
keeps those slots in ``req.opt_tree_lease``. Every other path drops them.

Several requests that are still growing share one forward at the same depth.
A single request, and any batch that fails for a reason other than a device
context error, still grow through ``_opt_expand_one``. A device context error
poisons the draft and propagates. On page size 1 the last layer's KV can be
forwarded after the reply; the verify window is already fixed by then.
"""

from __future__ import annotations

import copy
import logging
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.opt_tree_remote.opt_tree_config import (
    OptTreeConfig,
    opt_draft_lookahead_enabled,
    opt_probe_layer_wait_enabled,
)
from sglang.srt.speculative.opt_tree_remote.opt_tree_grow import grow_opt_tree
from sglang.srt.speculative.opt_tree_remote.opt_tree_ops import (
    OptTreeState,
    OptTreeStep,
    _layer_positions,
    expand_from_topk,
    expand_on_device,
    host_topk_rows,
    mirror_layer,
    seed_from_topk,
    seed_on_device,
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
        name: max(0.0, float(parts.get(name, 0.0) or 0.0))
        for name in OPT_INGEST_PHASES
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


def pack_verify_attention(
    prefix_lens: Sequence[int],
    ancestors: Sequence[torch.Tensor],
    depth: int,
    width: int,
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenate per-request TARGET_VERIFY masks and positions on ``device``.

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
            getattr(reqs[i], "rid", None)
            for i, row in enumerate(rows)
            if row is None
        ]
        raise RuntimeError(
            f"OPT draft reprefill returned no tail logits for {missing_rids}"
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
    from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
        get_sr_round_metrics,
    )

    windows = [None] * len(reqs)
    batch_failed = False
    try:
        windows = list(grow_batch(scheduler, reqs))
    except Exception as exc:
        _reraise_device_context(scheduler, exc)
        logger.warning("[OPT] batch tree expand failed, falling back: %s", exc)
        batch_failed = True
    if batch_failed or any(window is None for window in windows):
        get_sr_round_metrics(scheduler, "Draft").counts["opt_batch_fallbacks"] += 1
    for i, window in enumerate(windows):
        if window is None:
            windows[i] = scheduler._opt_expand_one(reqs[i])
    return windows


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
      ``SGLANG_OPT_PROBE_LAYER_WAIT`` is set. The probe synchronizes the
      stream before ``_run`` and records that stall here, outside ``host_us``
      and ``launch_us``. That schedule is not the online path: the host waits
      for the previous layer to finish before starting ``_run``.
    - ``opt_layer_gpu_us`` is ``ev0.elapsed_time(ev1)``. ``ev0`` is recorded at
      the start of the ``_run`` window, so the interval includes host
      preparation while the stream is idle. It is not GPU-busy time. Confirm
      busy and idle with nsys; do not subtract ``launch_us``.
    - ``opt_layer_forwards`` counts every queued layer, including the last one
      lookahead submits early. ``opt_layer_forwards_critical`` is that count
      minus one, matching the old path's pre-reply forwards.
    - ``opt_ingest_gpu_us`` is the ingest forward on the device. Its events are
      read in the same drain as the layer events.

    ``opt_seed_d2h_wait_us`` is once per grow and is not divided by
    ``opt_layer_forwards``. ``opt_layer_d2h_wait_us`` is the later ``.cpu()``
    of each layer. Logits stay on device until that copy. Warmup samples
    include the first replay and are reference only.
    """

    __slots__ = ("metrics", "t0", "t_launch", "ev0", "ev1", "opened", "host_closed")

    def __init__(self, metrics, t0, ev0, ev1):
        self.metrics = metrics
        self.t0 = t0
        self.t_launch = None
        self.ev0 = ev0
        self.ev1 = ev1
        self.opened = False
        self.host_closed = False


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


def _begin_layer_timing(scheduler):
    """Start the host clock. Event creation failure drops this sample only."""
    metrics = _draft_metrics(scheduler)
    if metrics is None:
        return None
    try:
        factory = torch.get_device_module(scheduler.device).Event
        ev0 = factory(enable_timing=True)
        ev1 = factory(enable_timing=True)
    except Exception:
        return None
    return _LayerTiming(metrics, time.perf_counter(), ev0, ev1)


def _close_layer_host(timing) -> None:
    """Stop the preparation clock before ``_run``."""
    if timing is None or timing.host_closed:
        return
    now = time.perf_counter()
    timing.metrics.counts["opt_layer_host_us"] += int((now - timing.t0) * 1e6)
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
        timing.metrics.counts["opt_layer_host_us"] += int((now - timing.t0) * 1e6)
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
    ``SGLANG_OPT_PROBE_LAYER_WAIT`` is set, and it sits outside this window.
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
    events = getattr(scheduler, "_opt_layer_events", None)
    if events is None:
        events = []
        scheduler._opt_layer_events = events
    events.append((timing.ev0, timing.ev1))


def critical_layer_forwards(launched: int) -> int:
    """Layers on the pre-reply path. The early final launch is excluded."""
    return max(int(launched) - 1, 0)


def _begin_ingest_gpu(scheduler) -> None:
    """Queue the start of the ingest GPU interval. No host synchronize."""
    scheduler._opt_ingest_events = None
    if _draft_metrics(scheduler) is None:
        return
    try:
        event = torch.get_device_module(scheduler.device).Event(enable_timing=True)
        event.record()
    except Exception:
        return
    scheduler._opt_ingest_events = event


def _finish_ingest_gpu(scheduler) -> None:
    """Queue the end of the ingest GPU interval. No host synchronize."""
    start = getattr(scheduler, "_opt_ingest_events", None)
    if start is None or isinstance(start, tuple):
        return
    try:
        end = torch.get_device_module(scheduler.device).Event(enable_timing=True)
        end.record()
    except Exception:
        scheduler._opt_ingest_events = None
        return
    scheduler._opt_ingest_events = (start, end)


def _note_lookahead_launch_wait(runner) -> None:
    """Optionally time an explicit wait for the previous layer.

    Default does not synchronize. The template path does not call
    ``init_new``. A missing template falls back to ``run_batch``, and that
    path still waits inside ``ForwardBatch.init_new``.
    ``SGLANG_OPT_PROBE_LAYER_WAIT`` inserts ``current_stream().synchronize()``
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


def drain_opt_layer_gpu_us(scheduler, metrics) -> None:
    """Read queued verify and ingest events once, after the grow clock has stopped.

    An empty queue does not synchronize. A poisoned device drops the events.
    ``elapsed_time`` is milliseconds. Other timing errors increment
    ``device_timing_dropped_opt_layer`` and do not replace a business result
    unless the accelerator context is already dead.
    """
    events = list(getattr(scheduler, "_opt_layer_events", None) or [])
    ingest = getattr(scheduler, "_opt_ingest_events", None)
    scheduler._opt_layer_events = []
    scheduler._opt_ingest_events = None
    ingest_pair = ingest if isinstance(ingest, tuple) and len(ingest) == 2 else None
    if not events and ingest_pair is None:
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
    if ingest_pair is None:
        return
    try:
        metrics.counts["opt_ingest_gpu_us"] += int(
            ingest_pair[0].elapsed_time(ingest_pair[1]) * 1000
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


def grow_active_layers(
    config: OptTreeConfig,
    tail_logits,
    forward_active: Callable,
    scheduler=None,
    states_out=None,
    topk_out=None,
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
    """
    tails = list(tail_logits)
    states = [OptTreeState.create(config) for _ in tails]
    if states_out is not None:
        states_out.extend(states)
    steps = [None] * len(tails)
    results = [None] * len(tails)
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
    # Earlier layers sit in the KV prefix, so every TARGET_VERIFY reads that
    # part of the mask. ``is_bonus`` only selects the N+1 graph key.
    apply_prefix_custom_mask: bool = True
    is_bonus: bool = False
    # OPT draft cascade only. Other algorithms leave this unset.
    target_verify_attention: object = None

    def is_draft_input(self) -> bool:
        return False

    def get_spec_adjusted_global_num_tokens(self, batch):
        return batch.global_num_tokens, batch.global_num_tokens_for_logprob


# Graph-key suffix for the bonus verify graph. Layer forwards stay on the
# unsuffixed ``r{N}_{bs}`` graph. Both graphs apply the prefix mask.
OPT_BONUS_GRAPH_EXTRA = 0


def opt_bonus_graph_extra(spec_info):
    """Extra graph key for a bonus forward, or None for a layer forward."""
    if getattr(spec_info, "is_bonus", False):
        return OPT_BONUS_GRAPH_EXTRA
    return None


def opt_draft_bonus_ntpb(server_args):
    """Query width of the single bonus graph, or None when this role has none.

    The bonus forward carries the root guess plus one guess per verify node,
    so its width is ``opt_tree_width + 1``. Layer graphs stay at ``N``.
    """
    if getattr(server_args, "opt_tree_role", None) != "draft":
        return None
    width = int(getattr(server_args, "opt_tree_width", 0) or 0)
    if width <= 1:
        return None
    return width + 1


def opt_draft_capture_extra_keys(server_args, ntpb) -> list:
    """Layer graphs stay unsuffixed. Only the N+1 bonus graph uses extra 0.

    ``ntpb == 1`` is the DECODE graph and is not duplicated. The width-N
    verify graph is the layer forward. Both graphs apply the prefix mask;
    the suffix only separates the bonus width.
    """
    if getattr(server_args, "opt_tree_role", None) != "draft":
        return [None]
    bonus = opt_draft_bonus_ntpb(server_args)
    if bonus is not None and ntpb is not None and int(ntpb) == int(bonus):
        return [OPT_BONUS_GRAPH_EXTRA]
    return [None]


def opt_layer_verify_graph_matches(spec_info, actual_ntpb: int) -> bool:
    """True when this spec may replay a TARGET_VERIFY graph of this width.

    The captured ntpb must equal this spec's query width. The ntpb=1 dual
    graph is DECODE, so width 1 stays eager. A bonus spec uses width N+1 and
    the runner selects that graph via ``opt_bonus_graph_extra``.
    """
    if not isinstance(spec_info, OptLayerSpec):
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
        timing = _begin_layer_timing(self.scheduler)
        positions = layer_positions(self.prefix_len, depth, n).to(device)
        mask = layer_attention_mask(self.prefix_len, ancestor, device)
        out_cache_loc = self._alloc_layer_slots(n)
        self.req.kv_allocated_len = self.cached_len + n
        batch = self._verify_batch(tokens.to(device), positions, mask, out_cache_loc)
        from sglang.srt.speculative.opt_tree_remote.opt_tree_attention import (
            bind_opt_cascade,
        )

        bind_opt_cascade(batch.spec_info, self, out_cache_loc, ancestor)
        _mark_layer_launch(timing)
        logits = self._run(batch)
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
        width = int(self.scheduler.server_args.opt_tree_width)
        if n != width:
            raise RuntimeError(
                f"OPT layer has {n} tokens but attention was built for {width}"
            )
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
        mask = layer_attention_mask(self.prefix_len, ancestor, device)
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

        batch.spec_info = OptLayerSpec(
            custom_mask=mask.reshape(-1),
            positions=self._la_positions.reshape(-1),
            draft_token_num=n,
            num_tokens_per_req=n,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        from sglang.srt.speculative.opt_tree_remote.opt_tree_attention import (
            bind_opt_cascade,
        )

        bind_opt_cascade(batch.spec_info, self, out_cache_loc, ancestor)
        _close_layer_host(timing)
        _note_lookahead_launch_wait(self)
        _mark_layer_launch(timing)
        self._lookahead_queued = True
        logits = self._run_lookahead(batch)
        if self._lookahead_launch_event is not None:
            self._lookahead_launch_event.record()
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
        self._la_seq.fill_(self.cached_len)
        out_cache_loc = alloc_token_slots(scheduler.tree_cache, n)
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
    batch.spec_info = OptLayerSpec(
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
    logits = _flatten_logits(_run_batch(scheduler, batch))
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
        rows = _extend_request_rows(_run_batch(scheduler, batch), len(runners))
        phases.lap("opt_ingest_forward")
    except Exception:
        for runner, start, _end in prepared:
            scheduler._opt_rollback(runner.req, start)
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
    """
    if not runners:
        raise RuntimeError("OPT verify batch is empty")
    page = int(getattr(scheduler.tree_cache, "page_size", 1) or 1)
    width = int(scheduler.server_args.opt_tree_width)
    pending = list(zip(runners, steps))
    rows = []
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
        group_rows = _as_layer_rows(logits, count, width)
        if consume is not None:
            consume(group_runners, group_rows)
        if count < len(pending):
            group_rows = [row.clone() for row in group_rows]
        rows.extend(group_rows)
        pending = pending[count:]
    return rows


def _forward_verify_batch(scheduler, runners, steps) -> torch.Tensor:
    """One TARGET_VERIFY for requests that still share a depth."""
    if not runners:
        raise RuntimeError("OPT verify batch is empty")
    timing = _begin_layer_timing(scheduler)
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
        prefix_lens, ancestors, depth, width, device
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
        flat_positions,
        flat_mask,
        torch.cat(locs),
        cached,
    )
    if len(runners) == 1:
        from sglang.srt.speculative.opt_tree_remote.opt_tree_attention import (
            bind_opt_cascade,
        )

        bind_opt_cascade(
            batch.spec_info,
            runners[0],
            locs[0],
            ancestors[0],
            cached_len=cached[0],
        )
    _mark_layer_launch(timing)
    logits = _run_batch(scheduler, batch)
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
        logits = _flatten_logits(_run_batch(scheduler, batch))
        phases.lap("opt_ingest_forward")
        if int(logits.shape[0]) != len(runners):
            raise RuntimeError(
                f"OPT decode append rows {int(logits.shape[0])} != {len(runners)}"
            )
        rows = [logits[i].clone() for i in range(len(runners))]
    except Exception:
        for runner, start, _end in prepared:
            scheduler._opt_rollback(runner.req, start)
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
            "OPT decode append left "
            + ", ".join(runner.req.rid for runner in pending)
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


def _committed_len(req) -> int:
    return len(req.origin_input_ids or []) + len(req.output_ids or [])


@dataclass
class OptDeferredTree:
    """One request whose last layer is forwarded after the reply is sent."""

    req: object
    runner: "OptLayerForward"
    keep: int
    window: tuple
    state: OptTreeState


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
        and envs.SGLANG_OPT_TREE_KV_REUSE.get()
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
    """Store one host ``[N, N + 1]`` id tensor per request in this sub-batch."""
    for runner, row in zip(runners, rows):
        if runner is None or not torch.is_tensor(row):
            continue
        width = int(runner.scheduler.server_args.opt_tree_width)
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
    """First top-(N+1) id at the root and every verify node that is not a child."""
    if state is None or window is None or len(window) != 3 or runner is None:
        return []
    tokens, parents, cols = window
    if not tokens or not (len(tokens) == len(parents) == len(cols)):
        return []
    from sglang.srt.speculative.opt_tree_remote.opt_tree_lease import node_depths

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
        plans.append(_BonusCandidate(int(parent), int(token), int(position), tuple(chain)))

    if bonus_ids:
        banned = [int(tokens[i]) for i, parent in enumerate(parents) if int(parent) == -1]
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
        banned = [int(tokens[j]) for j, parent in enumerate(parents) if int(parent) == index]
        add(
            index,
            row[col],
            banned,
            prefix + depth + 1,
            ancestor_columns(state, prefix, depth, col),
        )
    return plans


def _free_bonus_ids(scheduler, slot_ids) -> None:
    from sglang.srt.speculative.opt_tree_remote.opt_tree_lease import _free_slot_ids

    _free_slot_ids(scheduler, slot_ids)


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


@dataclass
class _BonusLaunch:
    """One bonus group whose slot ids are still on device."""

    loc: Optional[torch.Tensor]
    logits: torch.Tensor
    chunk: list
    tree_end: int
    width: int


def _alloc_bonus_slots(scheduler, runner, n: int, start: int) -> torch.Tensor:
    """Page-1 bonus slots. Length tensors live on the scheduler.

    ``fill_`` writes the current prefix and, when it changes, the pool index.
    The two length-1 buffers are keyed by device in ``_opt_la_cache`` and
    survive a new ``OptLayerForward``. No ``tree_cache`` keeps the caller's
    ``_alloc_layer_slots`` stub. These slots are not appended to ``layer_slots``.
    """
    tree_cache = getattr(scheduler, "tree_cache", None)
    token_pool = getattr(scheduler, "req_to_token_pool", None)
    pool = getattr(token_pool, "req_to_token", None)
    page = int(getattr(tree_cache, "page_size", 1) or 1) if tree_cache is not None else 1
    pool_idx = getattr(getattr(runner, "req", None), "req_pool_idx", None)
    if (
        tree_cache is None
        or token_pool is None
        or pool is None
        or page != 1
        or pool_idx is None
    ):
        return runner._alloc_layer_slots(n, start=start)
    from sglang.srt.mem_cache.common import alloc_token_slots
    from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

    device = token_pool.device
    cache = _bonus_len_cache(scheduler, device)
    pool_idx = int(pool_idx)
    if cache["pool_idx"] != pool_idx:
        cache["pool"].fill_(pool_idx)
        cache["pool_idx"] = pool_idx
    cache["seq"].fill_(int(start))
    out_cache_loc = alloc_token_slots(tree_cache, n)
    if out_cache_loc is None or int(out_cache_loc.numel()) != n:
        raise RuntimeError(
            f"OPT bonus allocated "
            f"{0 if out_cache_loc is None else int(out_cache_loc.numel())} "
            f"slots for {n} tokens"
        )
    assign_req_to_token_pool_func(
        cache["pool"],
        pool,
        cache["seq"],
        cache["seq"] + n,
        out_cache_loc,
        1,
    )
    return out_cache_loc


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
    """Split one host copy of every group's slot ids."""
    pieces = []
    for loc in locs:
        if loc is None:
            raise RuntimeError("OPT bonus slot ids were read after the buffer was cleared")
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


def _abort_bonus_slots(scheduler, runner, loc, tree_end: int, width: int) -> None:
    """Sync one failed group's ids, clear the map, and free those slots."""
    slots = _read_bonus_slot_ids([loc])[0]
    _clear_req_to_token_span(scheduler, runner.req, tree_end, tree_end + width)
    runner.req.kv_allocated_len = tree_end
    _free_bonus_ids(scheduler, slots)


def _free_open_bonus_groups(scheduler, runner, groups: Sequence[_BonusLaunch]) -> None:
    for group in groups:
        if group.loc is None:
            continue
        loc = group.loc
        group.loc = None
        _abort_bonus_slots(scheduler, runner, loc, group.tree_end, group.width)


def _bonus_model_runner(scheduler):
    worker = getattr(scheduler, "model_worker", None)
    return getattr(worker, "model_runner", None)


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
    batch.spec_info = OptLayerSpec(
        custom_mask=mask.reshape(-1),
        positions=positions.reshape(-1).to(dtype=torch.int64),
        draft_token_num=int(width),
        num_tokens_per_req=int(width),
        capture_hidden_mode=CaptureHiddenMode.NULL,
        apply_prefix_custom_mask=True,
        is_bonus=True,
    )
    mask_rows = mask.reshape(int(width), int(tree_end) + int(width))
    prefix = int(getattr(runner, "prefix_len", 0) or 0)
    from sglang.srt.speculative.opt_tree_remote.opt_tree_attention import (
        bind_opt_cascade,
    )

    prefix_rows = None
    if prefix > 0:
        prefix_rows = mask_rows[:, 0]
    bind_opt_cascade(
        batch.spec_info,
        runner,
        loc,
        mask_rows[:, prefix:],
        prefix_rows=prefix_rows,
        cached_len=int(tree_end),
    )
    return batch


def _bonus_fallback_logits(
    scheduler, runner, tokens, positions, mask, loc, tree_end: int, width: int
):
    batch = _verify_batch_many(
        scheduler,
        [runner],
        [tokens],
        positions,
        mask.reshape(-1),
        loc.reshape(-1),
        [tree_end],
        is_bonus=True,
    )
    return _as_layer_rows(_run_batch(scheduler, batch), 1, width)[0]


def _bonus_template_forward(scheduler, batch, template, delta, width: int):
    """Replay the N+1 bonus graph. A false ``can_run_graph`` already ran eager.

    Do not call ``run_batch`` after that forward: the slots were just written.
    """
    forward_batch = _refresh_lookahead_forward(template, batch, delta)
    out = scheduler.model_worker.model_runner.forward(forward_batch)
    if getattr(out, "can_run_graph", True) is False:
        _opt_draft_count(scheduler, "opt_bonus_eager_groups")
    logits = getattr(getattr(out, "logits_output", None), "next_token_logits", None)
    if logits is None:
        raise RuntimeError("OPT bonus forward returned no logits")
    return _as_layer_rows(logits, 1, width)[0]


def _bonus_batch_hit(scheduler, req):
    """The lookahead batch entry for this request and its current pool."""
    batches = getattr(scheduler, "_opt_la_batches", None)
    if not batches:
        return None
    hit = batches.get(getattr(req, "rid", None))
    if hit is None or hit.get("pool_idx") != getattr(req, "req_pool_idx", None):
        return None
    return hit


def _load_bonus_admit(scheduler, req):
    """Return a cached bonus template, or None when this pool has none."""
    hit = _bonus_batch_hit(scheduler, req)
    if hit is None:
        return None
    template = hit.get("bonus_template")
    if template is None:
        return None
    return template, hit.get("bonus_delta")


def _store_bonus_admit(scheduler, req, template, delta) -> None:
    """Keep a successful admission. A miss is not stored."""
    if template is None:
        return
    hit = _bonus_batch_hit(scheduler, req)
    if hit is None:
        return
    hit["bonus_template"] = template
    hit["bonus_delta"] = delta


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


def _bonus_group_logits(
    scheduler, runner, tokens, positions, mask, loc, tree_end: int, width: int, admit: _BonusAdmit
):
    """One group. Reuse the cached N+1 template, or admit it once.

    ``_fill_bonus_schedule_batch`` refreshes the lookahead batch first, so a
    pool change has already dropped the previous template. Later groups in
    this round only copy lengths, the mask, and positions onto that template.
    """
    model_runner = _bonus_model_runner(scheduler)
    if model_runner is not None and (not admit.tried or admit.template is not None):
        batch = _fill_bonus_schedule_batch(
            scheduler, runner, tokens, positions, mask, loc, tree_end, width
        )
        if not admit.tried:
            admit.tried = True
            cached = _load_bonus_admit(scheduler, runner.req)
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
                    scheduler, runner.req, admit.template, admit.delta
                )
        if admit.template is not None:
            return _bonus_template_forward(
                scheduler, batch, admit.template, admit.delta, width
            )
    _opt_draft_count(scheduler, "opt_bonus_template_fallbacks")
    return _bonus_fallback_logits(
        scheduler, runner, tokens, positions, mask, loc, tree_end, width
    )


def _launch_bonus_group(scheduler, runner, chunk, tree_end: int, prefix: int, admit: _BonusAdmit):
    """Forward one group and leave its slot ids on device."""
    width = len(chunk)
    device = scheduler.req_to_token_pool.device
    chains = [plan.chain for plan in chunk]
    mask = bonus_attention_mask(prefix, tree_end, chains, device=device)
    positions = _bonus_long_tensor([plan.position for plan in chunk], device)
    tokens = _bonus_long_tensor([plan.token for plan in chunk], device)
    loc = _alloc_bonus_slots(scheduler, runner, width, tree_end)
    runner.req.kv_allocated_len = tree_end + width
    try:
        logits = _bonus_group_logits(
            scheduler, runner, tokens, positions, mask, loc, tree_end, width, admit
        )
        # The next replay overwrites this graph output buffer.
        logits = logits.detach().clone()
    except Exception as exc:
        _reraise_device_context(scheduler, exc)
        _abort_bonus_slots(scheduler, runner, loc, tree_end, width)
        raise
    _clear_req_to_token_span(scheduler, runner.req, tree_end, tree_end + width)
    runner.req.kv_allocated_len = tree_end
    return _BonusLaunch(
        loc=loc,
        logits=logits,
        chunk=list(chunk),
        tree_end=int(tree_end),
        width=int(width),
    )


def _materialize_bonus_guesses(scheduler, group: _BonusLaunch, slots: Sequence[int]):
    """Turn host slot ids into guesses and free the padded ones."""
    from sglang.srt.speculative.opt_tree_remote.opt_tree_lease import OptBonusGuess

    if len(slots) != len(group.chunk):
        raise RuntimeError(
            f"OPT bonus returned {len(slots)} slots for {len(group.chunk)} queries"
        )
    guesses = []
    dummy_slots = []
    for index, plan in enumerate(group.chunk):
        if not plan.keep:
            dummy_slots.append(int(slots[index]))
            continue
        guesses.append(
            OptBonusGuess(
                parent=int(plan.parent),
                token=int(plan.token),
                slot=int(slots[index]),
                logits=group.logits[index].detach().clone(),
            )
        )
    try:
        _free_bonus_ids(scheduler, dummy_slots)
    except Exception:
        _free_bonus_ids(scheduler, [guess.slot for guess in guesses])
        raise
    return guesses


def _forward_one_bonus_group(scheduler, runner, chunk, tree_end: int, prefix: int):
    """Forward one bonus group. Return guesses and free the padded slots."""
    group = _launch_bonus_group(scheduler, runner, chunk, tree_end, prefix, _BonusAdmit())
    slots = _read_bonus_slot_ids([group.loc])[0]
    group.loc = None
    return _materialize_bonus_guesses(scheduler, group, slots)


def _forward_bonus_candidates(scheduler, runner, plans: Sequence[_BonusCandidate]) -> None:
    """Write guess KV past the tree, then detach those columns from the map.

    One forward has ``N + 1`` queries: the root guess plus one per verify
    node. That matches the bonus CUDA graph. A short tail still repeats the
    last real candidate. Publish sees only the tree span. Real guess slots
    stay allocated until the next step installs one or frees them.

    Every group of this round shares one admission attempt. A successful
    template stays on the lookahead batch until the pool index changes or
    the request is dropped. Slot ids stay on device until the last group
    returns, then one copy reads them all.
    """
    from sglang.srt.speculative.opt_tree_remote.opt_tree_lease import (
        release_opt_bonus_guesses,
    )

    width = int(scheduler.server_args.speculative_num_draft_tokens) + 1
    tree_end = int(runner.cached_len)
    prefix = int(runner.prefix_len)
    launched: List[_BonusLaunch] = []
    kept = []
    admit = _BonusAdmit()
    try:
        for chunk in bonus_verify_chunks(plans, width):
            launched.append(
                _launch_bonus_group(scheduler, runner, chunk, tree_end, prefix, admit)
            )
        id_rows = _read_bonus_slot_ids([group.loc for group in launched])
        for group, slots in zip(launched, id_rows):
            group.loc = None
            kept.extend(_materialize_bonus_guesses(scheduler, group, slots))
    except Exception as exc:
        _reraise_device_context(scheduler, exc)
        _free_open_bonus_groups(scheduler, runner, launched)
        if not getattr(scheduler, "_sr_device_poisoned", False):
            release_opt_bonus_guesses(scheduler, runner.req, kept)
        raise
    release_opt_bonus_guesses(scheduler, runner.req)
    runner.req.opt_bonus_guesses = kept


def maybe_attach_bonus_guesses(scheduler, job: OptDeferredTree) -> None:
    """Predict bonus tokens while the tree KV is still mapped.

    Page size other than 1 never writes these slots. Logits are dropped either
    way so a deferred runner does not keep the whole vocabulary.
    """
    runner = job.runner
    try:
        if runner is None or _opt_page_size(scheduler) != 1:
            _opt_draft_count(scheduler, "opt_bonus_skipped")
            return
        if getattr(scheduler, "_sr_device_poisoned", False) or _request_finished(
            scheduler, job.req
        ):
            _opt_draft_count(scheduler, "opt_bonus_skipped")
            return
        if not hasattr(runner, "_alloc_layer_slots"):
            _opt_draft_count(scheduler, "opt_bonus_skipped")
            return
        plans = select_bonus_candidates(job.state, job.window, runner)
        if not plans:
            _opt_draft_count(scheduler, "opt_bonus_skipped")
            return
        _opt_draft_count(scheduler, "opt_bonus_rounds")
        try:
            _forward_bonus_candidates(scheduler, runner, plans)
        except Exception as exc:
            _reraise_device_context(scheduler, exc)
            logger.warning("[OPT] bonus guess forward failed: %s", exc)
    finally:
        _drop_stored_logits(runner)


def final_layer_step(state) -> OptTreeStep:
    """The layer ``step_after_layer`` closed without forwarding."""
    return OptTreeStep(
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
    if (
        start < 0
        or end <= start
        or int(row) < 0
        or int(row) >= int(pool.shape[0])
    ):
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
    runner.layer_slots.pop()
    cached = int(runner.cached_len)
    try:
        scheduler._opt_rollback(job.req, cached)
    except Exception:
        logger.warning(
            "[OPT] rollback of unfinished layer failed for %s",
            getattr(job.req, "rid", None),
            exc_info=True,
        )
        return
    job.req.kv_committed_len = int(job.keep)
    _clear_req_to_token_span(scheduler, job.req, cached, cached + width)


class LookaheadFallback(RuntimeError):
    """Leave the overlapped path and grow this tree the original way."""


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
        self.ids = torch.empty(n, n + 1, dtype=torch.long, **pin)
        self.event = torch.cuda.Event()
        self.kind = "seed"
        self.ready = False

    def copy_seed(self, scores: torch.Tensor, ids: torch.Tensor) -> None:
        n = int(self.scores.shape[0])
        flat_ids = ids.reshape(-1)
        self.scores.copy_(scores.reshape(-1)[:n].float(), non_blocking=True)
        self.tokens.copy_(flat_ids[:n].to(dtype=torch.long), non_blocking=True)
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
            self.ids.clone(),
            self.kind,
        )


class _PinnedBonus:
    """Top-(N+1) ids of the speculative last layer. Not one of the two layer slots."""

    def __init__(self, width: int):
        n = int(width)
        self.ids = torch.empty(n, n + 1, dtype=torch.long, device="cpu", pin_memory=True)
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
    """Slots held together: ingest, every forwarded layer, and one bonus group.

    ``depth_rule`` stops once ``depth`` reaches ``max_depth - 1``. Lookahead
    forwards that last layer before the host decides to stop, so the tree
    takes ``(max_depth - 1) * n`` slots. The bonus forward then allocates
    ``n + 1`` while those slots are still mapped. ``ingest_slots`` is the
    span ingest still has to allocate. Admission runs before that alloc.
    """
    n = int(scheduler.server_args.opt_tree_width)
    depth = int(scheduler.server_args.opt_tree_max_depth)
    return int(ingest_slots) + (depth - 1) * n + (n + 1)


def _lookahead_ingest_slots(runner) -> int:
    """Tokens ingest will allocate. ``prefix_len`` is already the committed length."""
    committed = int(runner.prefix_len)
    have = int(getattr(runner.req, "kv_committed_len", 0) or 0)
    return max(committed - have, 0)


def _lookahead_would_evict(scheduler, ingest_slots: int = 0) -> bool:
    """True when the free list cannot cover ingest plus the lookahead tree.

    ``available_size`` is ``len`` of the device free list and does not sync.
    A short radix free list would evict, and that eviction can sync. A short
    chunk-cache free list fails the alloc and still resets the step. Either
    case falls back before ingest is queued.
    """
    tree = getattr(scheduler, "tree_cache", None)
    if tree is None:
        return False
    allocator = getattr(tree, "token_to_kv_pool_allocator", None)
    if allocator is None:
        return True
    try:
        available = allocator.available_size()
        need = _lookahead_kv_need(scheduler, ingest_slots)
    except Exception:
        return True
    return int(available) < int(need)


def _lookahead_forward_template(scheduler, batch):
    """One ``ForwardBatch`` for every later layer, or None when replay cannot run.

    ``init_new`` may synchronize while copying a pageable M-RoPE delta. That
    happens here, before the layer loop. Later layers copy this object and
    do not call ``init_new`` again.
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
        logger.warning("[OPT] lookahead graph check failed", exc_info=True)
        return None
    if not forward_batch.forward_mode.is_cuda_graph():
        return None
    if getattr(forward_batch, "is_sr_tail_extend", False):
        return None
    try:
        ready = bool(graph_runner.can_run(forward_batch))
    except Exception:
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


def _lookahead_mrope_positions(positions, delta) -> torch.Tensor:
    """Target-verify M-RoPE: one device delta added to every position, three rows."""
    seq_positions = positions.detach().reshape(1, -1)
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


def _bonus_len_cache(scheduler, device):
    """Length-1 seq and pool-index buffers for bonus slot assignment.

    Stored on the scheduler next to the width cache and keyed by device.
    A new ``OptLayerForward`` each step still reuses the same pair.
    """
    cache = getattr(scheduler, "_opt_la_cache", None)
    if cache is None:
        cache = {}
        scheduler._opt_la_cache = cache
    by_device = cache.get("_bonus_len")
    if not isinstance(by_device, dict):
        by_device = {}
        cache["_bonus_len"] = by_device
    key = device if isinstance(device, torch.device) else torch.device(device)
    hit = by_device.get(key)
    if (
        hit is None
        or not torch.is_tensor(hit.get("seq"))
        or not torch.is_tensor(hit.get("pool"))
    ):
        hit = {
            "seq": torch.empty(1, dtype=torch.int64, device=key),
            "pool": torch.empty(1, dtype=torch.int64, device=key),
            "pool_idx": None,
        }
        by_device[key] = hit
    return hit


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
        "bonus": _PinnedBonus(width),
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
    if (
        hit is not None
        and hit["pool_idx"] == pool
        and hit["batch"] is not None
    ):
        batch = hit["batch"]
        if not batch.reqs or batch.reqs[0] is not req:
            batch.reqs[0] = req
        return batch
    batch = scheduler._sr_make_decode_batch([req])
    _keep_raw_logits(batch)
    batches[rid] = {"pool_idx": pool, "batch": batch}
    return batch


def drop_lookahead_request(scheduler, rid) -> None:
    """Drop a finished request's cached decode batch."""
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
    width = int(scheduler.server_args.opt_tree_width)
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
    """Wait for queued lookahead kernels, then free every uncommitted tree slot."""
    from sglang.srt.speculative.standalone_remote.sr_align import (
        is_device_context_error,
    )

    try:
        torch.get_device_module(scheduler.device).synchronize()
    except Exception as exc:
        if is_device_context_error(exc):
            _reraise_device_context(scheduler, exc)
        logger.warning("[OPT] lookahead reset sync failed", exc_info=True)
    keep = int(runner.prefix_len)
    end = int(getattr(runner.req, "kv_allocated_len", keep) or keep)
    try:
        scheduler._opt_rollback(runner.req, keep)
    except Exception:
        logger.warning(
            "[OPT] lookahead rollback failed for %s",
            getattr(runner.req, "rid", None),
            exc_info=True,
        )
    runner.layer_slots.clear()
    runner.cached_len = keep
    runner.req.kv_allocated_len = keep
    runner.req.kv_committed_len = keep
    _clear_req_to_token_span(scheduler, runner.req, keep, end)
    runner.lookahead_inflight = False
    runner._lookahead_queued = False
    runner._last_logits = None
    runner._la_forward_template = None
    runner._la_mrope_delta = None


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
    """KV-budget check, then reuse or build the layer template.

    Call this before ingest is queued. ``prefix_len`` and ``cached_len`` are
    already the committed length, and ingest does not change them. The budget
    includes the slots ingest still has to allocate. ``available_size`` is the
    free-list length and does not sync. A short list raises
    ``LookaheadFallback`` and leaves a cached template in place.

    A template and its M-RoPE delta stay on ``_opt_la_batches[rid]`` until the
    KV pool index changes or the request is dropped. A hit skips ``init_new``
    and ``can_run``. Later layers still report a graph miss through
    ``opt_lookahead_eager_layers``.
    """
    n = int(scheduler.server_args.opt_tree_width)
    if _lookahead_would_evict(scheduler, _lookahead_ingest_slots(runner)):
        raise LookaheadFallback("kv evict")
    cached = _load_layer_admit(scheduler, runner.req)
    if cached is not None:
        runner._la_forward_template, runner._la_mrope_delta = cached
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
    mask = torch.ones(
        n * (int(runner.prefix_len) + n), dtype=torch.bool, device=device
    )
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
    batch.spec_info = OptLayerSpec(
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


def grow_lookahead(scheduler, runner, config: OptTreeConfig, tail_logits, runtime):
    """Grow one tree with the next layer already queued.

    The host mirrors whatever the device top-k selected. It does not run a
    second top-k. A graph miss or an eviction falls back to the caller.
    """
    batch, slots, bonus = runtime
    state = OptTreeState.create(config)
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
                runner.bonus_ids.append(cpu_ids[0, : n + 1].contiguous())
            else:
                mirror_layer(
                    state, cpu_scores, cpu_tokens, _cpu_parents, seed=False
                )
                runner.bonus_ids.append(cpu_ids.contiguous())
            step = step_after_layer(state)
            if step.is_final:
                bonus_scores, bonus_ids = _vocab_topk(runner._last_logits, n + 1)
                del bonus_scores
                bonus.copy(bonus_ids)
                runner._lookahead_bonus = bonus
                runner.lookahead_inflight = True
                return [_verify_window(step)], state, layers
            child_scores, child_ids = _vocab_topk(runner._last_logits, n + 1)
            top_scores, tokens, parents, anc = expand_on_device(
                runner._la_weight, runner._la_anc, child_scores, child_ids, n
            )
            slot = slots[turn]
            turn ^= 1
            slot.copy_expand(top_scores, tokens, parents, child_ids)
            next_depth = int(state.depth) + 1
            runner._la_weight = top_scores
            runner._la_anc = anc
            runner.launch_layer(tokens, anc, next_depth, batch)
            layers += 1
    except LookaheadFallback:
        _opt_draft_count(scheduler, "opt_lookahead_fallbacks")
        _reset_lookahead_runner(scheduler, runner)
        raise
    except Exception as exc:
        _reraise_device_context(scheduler, exc)
        _opt_draft_count(scheduler, "opt_lookahead_fallbacks")
        _reset_lookahead_runner(scheduler, runner)
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
        or int(runner.cached_len) != int(runner.prefix_len) + int(job.state.depth) * int(job.state.config.width)
    )
    if poisoned:
        runner.lookahead_dropped = True
        runner.lookahead_inflight = False
        return
    if drop:
        try:
            _await_lookahead_event(
                scheduler, getattr(runner, "_lookahead_launch_event", None)
            )
            bonus = getattr(runner, "_lookahead_bonus", None)
            if bonus is not None:
                _await_lookahead_event(scheduler, getattr(bonus, "event", None))
                bonus.ready = False
            # ``keep`` is the committed prefix. Rollback frees the whole tree,
            # and local_rollback then sets ``kv_allocated_len`` to ``keep``.
            end = int(getattr(job.req, "kv_allocated_len", job.keep) or job.keep)
            scheduler._opt_rollback(job.req, int(job.keep))
            _clear_req_to_token_span(scheduler, job.req, int(job.keep), end)
        except Exception as exc:
            _reraise_device_context(scheduler, exc)
            logger.warning(
                "[OPT] rollback of the lookahead tree failed for %s",
                getattr(job.req, "rid", None),
                exc_info=True,
            )
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
    except Exception as exc:
        _reraise_device_context(scheduler, exc)
        bonus_failed = True
        logger.warning(
            "[OPT] lookahead bonus ids failed for %s: %s",
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

    Bonus guesses are written while the tree columns are still mapped.
    Publish clears those columns, so the guess forward cannot follow it.
    A lookahead job already queued that last layer. It only installs the
    top-k ids, unless the request can no longer keep the slots.
    """
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
            by_depth.setdefault(int(job.state.depth), []).append(job)
        for group in by_depth.values():
            try:
                _forward_verify_fitting(
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
        skipped_ids = {id(job) for job in skipped}
        for job in skipped:
            _drop_stored_logits(job.runner)
        for job in jobs:
            if id(job) in skipped_ids:
                continue
            try:
                maybe_attach_bonus_guesses(scheduler, job)
            except Exception as exc:
                _reraise_device_context(scheduler, exc)
                logger.warning("[OPT] bonus attach failed: %s", exc)
                _opt_draft_count(scheduler, "opt_bonus_attach_failed")
    finally:
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
    if (
        runner is not None
        and getattr(runner, "lookahead_inflight", False)
        and not (
            _deferrable_state(state) and _can_defer(scheduler, req, expanded, window)
        )
    ):
        # This publish frees slots before the reply. The extra layer is still
        # queued, so wait for it first. The bonus copy is queued after that
        # event; drop it here so a later clone_cpu cannot read the old ids.
        # The deferred path waits after the reply.
        _await_lookahead_event(
            scheduler, getattr(runner, "_lookahead_launch_event", None)
        )
        bonus = getattr(runner, "_lookahead_bonus", None)
        if bonus is not None:
            _await_lookahead_event(scheduler, getattr(bonus, "event", None))
            bonus.ready = False
        runner.lookahead_inflight = False
        runner.lookahead_dropped = True
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
    return None


def commit_deferred_expands(scheduler, jobs, *, keep: bool) -> Optional[BaseException]:
    """Queue jobs after a clean expand, or publish them when expand is failing."""
    if keep:
        bucket = getattr(scheduler, "_opt_deferred", None)
        if isinstance(bucket, list):
            bucket.extend(jobs)
            return None
    error = None
    for job in jobs:
        try:
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
    from sglang.srt.speculative.opt_tree_remote.opt_tree_lease import (
        publish_opt_tree_lease,
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
            published = publish_opt_tree_lease(scheduler, req, runner, keep, window)
        except Exception as exc:
            logger.warning(
                "[OPT] lease publish failed for %s: %s",
                getattr(req, "rid", None),
                exc,
            )
    if published or getattr(scheduler, "_sr_device_poisoned", False):
        return
    release_opt_bonus_guesses(scheduler, req)
    scheduler._opt_rollback(req, keep)


def grow_batch(scheduler, reqs):
    """Grow every request together.

    A missing output that fits in one chunk shares one EXTEND. A longer output
    is extended in ``chunked_prefill_size`` pieces, and only the last piece's
    row is kept. One or two missing tokens on page size 1 are DECODE steps.
    Requests that still have the same depth share one TARGET_VERIFY. On page
    size 1, a clean finish keeps forwarded slots in ``req.opt_tree_lease``.
    When post-reply reuse is on, that publish waits until the last layer has
    been forwarded. Every other finish rolls those slots back to the committed
    length.
    """
    if not reqs:
        return []
    from sglang.srt.speculative.opt_tree_remote.opt_tree_lease import (
        consume_opt_tree_lease,
    )

    from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
        get_sr_round_metrics,
    )

    committed = [_committed_len(req) for req in reqs]
    runners: List[OptLayerForward] = []
    states: List[OptTreeState] = []
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
            consume_opt_tree_lease(scheduler, req)
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

        def _sync():
            if metrics.active:
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
            layers += 1
            return _forward_verify_fitting(
                scheduler, [runners[i] for i in indices], steps
            )

        t1 = time.perf_counter()
        topk_out = [[] for _ in runners]
        windows = grow_active_layers(
            config,
            tail_logits,
            forward_active,
            scheduler,
            states_out=states,
            topk_out=topk_out,
        )
        for runner, ids in zip(runners, topk_out):
            runner.bonus_ids = ids
        metrics.add_host("opt_grow", time.perf_counter() - t1)
        metrics.counts["opt_layer_forwards"] += layers
        metrics.counts["opt_rounds"] += 1
        expanded = True
        return windows
    finally:
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
