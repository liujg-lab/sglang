"""Local SR round timings; device events are polled, never synchronized."""

import logging
import math
import time
from collections import Counter, defaultdict, deque
from contextlib import contextmanager, nullcontext

from sglang.srt.speculative.standalone_remote.sr_align import is_device_context_error

logger = logging.getLogger(__name__)


def metrics_phase(metrics, name, *, device=False, stream=None):
    """Return ``metrics.phase`` or an empty context when metrics are absent."""
    phase = getattr(metrics, "phase", None)
    if not callable(phase):
        return nullcontext()
    return phase(name, device=device, stream=stream)

GRAPH_HOST_STAGES = (
    "lengths",
    "payload_fill",
    "update_call",
    "replay_call",
    "submit_envelope",
    "prepare_submit",
)
GRAPH_HOST_STAGE_LOG = {
    "lengths": "lengths_host_ms",
    "payload_fill": "payload_fill_host_ms",
    "update_call": "update_call_host_ms",
    "replay_call": "replay_call_host_ms",
    "submit_envelope": "submit_envelope_host_ms",
    "prepare_submit": "prepare_submit_total_host_ms",
}
_GRAPH_HOST_WARN_MAX = 8
_graph_host_warns = 0
_sr_tree_path_export_warned = False


def _ns_to_ms(duration_ns):
    return duration_ns / 1e6


def _warn_graph_host(where, exc):
    global _graph_host_warns
    try:
        if _graph_host_warns >= _GRAPH_HOST_WARN_MAX:
            return
        _graph_host_warns += 1
        logger.warning("SR graph host %s failed: %s", where, exc)
    except Exception:
        return


def note_sr_tree_path_export_error(metrics, exc) -> None:
    """Count every tree-path export failure. Warn once per process.

    Clearing the path is safe: Draft treats a missing candidate path as a
    lease miss and falls back to tail-extend.
    """
    global _sr_tree_path_export_warned
    if metrics is not None and hasattr(metrics, "counts"):
        metrics.counts["sr_tree_path_export_errors"] += 1
    if _sr_tree_path_export_warned:
        return
    _sr_tree_path_export_warned = True
    logger.warning(
        "[SR] accepted tree path export failed; clearing paths so Draft uses tail-extend: %s",
        exc,
    )


def begin_graph_host_sample(context):
    return GraphHostSample(context)


def _diag(where, fn):
    """Run a diagnostics callback. Ordinary failures drop the stat only."""
    try:
        return fn()
    except Exception as exc:
        _warn_graph_host(where, exc)
        return None


def measure_call(sample, stage, fn):
    """Time ``fn`` once.

    Sample creation, clock reads, and stage records are diagnostics. Their
    ordinary exceptions are discarded. ``fn`` still runs exactly once, and its
    return value, original exception, and interrupt propagate unchanged.
    """
    if sample is None:
        return fn()
    start = _diag("clock", time.perf_counter_ns)
    try:
        result = fn()
    except BaseException:
        if start is not None:
            _record_stage_safely(sample, stage, start, completed=False)
        raise
    if start is not None:
        _record_stage_safely(sample, stage, start, completed=True)
    return result


def _record_stage_safely(sample, stage, start_ns, completed):
    def _record():
        duration = _ns_to_ms(time.perf_counter_ns() - start_ns)
        sample.record_stage(stage, duration, completed=completed)

    _diag("stage", _record)


def mark_graph_host_failed(sample):
    if sample is None:
        return
    _diag("mark_failed", sample.mark_failed_or_interrupted)


def bind_graph_host_metrics(runner, metrics):
    """Attach metrics for one forward. ``None`` when there is nothing to restore."""
    if runner is None or metrics is None:
        return None
    had = hasattr(runner, "_sr_graph_host_metrics")
    previous = getattr(runner, "_sr_graph_host_metrics", None) if had else None
    runner._sr_graph_host_metrics = metrics
    return (runner, had, previous)


def restore_graph_host_metrics(token):
    """Restore the previous attribute, or delete one this bind created."""
    if token is None:
        return
    runner, had, previous = token
    if had:
        runner._sr_graph_host_metrics = previous
    elif hasattr(runner, "_sr_graph_host_metrics"):
        del runner._sr_graph_host_metrics


def record_graph_host_sample_safely(metrics, sample):
    if sample is None:
        return
    try:
        if metrics is None:
            return
        metrics.record_graph_host_sample(sample)
    except Exception as exc:
        _warn_graph_host("record", exc)


class GraphHostSample:
    def __init__(self, context):
        self.context = dict(context)
        self.stages = {}
        self.stage_completed = {}
        self.outcome = "ok"
        self._recorded = False

    def mark_failed_or_interrupted(self):
        self.outcome = "failed"

    def record_stage(self, name, duration_ms, completed=True):
        self.stages[name] = duration_ms
        self.stage_completed[name] = bool(completed)

    def group_key(self):
        ctx = self.context
        return (
            ctx.get("graph_phase", "draft_tree"),
            ctx.get("graph_key"),
            ctx.get("raw_bs"),
            ctx.get("capture_bs"),
            ctx.get("implementation"),
            ctx.get("overlap"),
            ctx.get("kv_bucket"),
        )


class SRCommMetrics:
    """Bounded, per-action host telemetry owned by the socket's TP rank.

    A window contains at most 32 completed calls. Event counters (e.g. stale
    replies) do not complete calls. Percentiles use nearest-rank selection.
    """

    def __init__(self, role, transport):
        self.role = role
        self.transport = transport
        self.windows = {}
        self.first_success = set()

    def _window(self, action):
        # Limit cardinality even for malformed input from peers.
        action = getattr(action, "value", action)
        action = (
            action if action in ("prefill", "step", "finish", "abort") else "unknown"
        )
        if action not in self.windows:
            self.windows[action] = {"calls": 0, "counts": Counter(), "values": {}}
        return action, self.windows[action]

    @staticmethod
    def summarize(values):
        ordered = sorted(values)
        n = len(ordered)
        return {
            "n": n,
            "mean": round(sum(ordered) / n, 3),
            "p50": round(ordered[math.ceil(n * 0.50) - 1], 3),
            "p95": round(ordered[math.ceil(n * 0.95) - 1], 3),
            "max": round(ordered[-1], 3),
        }

    def count(self, action, event, n=1):
        _, window = self._window(action)
        window["counts"][event] += n

    def record(self, action, outcome, values=None):
        action, window = self._window(action)
        values = values or {}
        window["calls"] += 1
        window["counts"][outcome] += 1
        for name, value in values.items():
            window["values"].setdefault(name, []).append(value)
        if outcome == "success" and action not in self.first_success:
            self.first_success.add(action)
            self._emit(
                action, "sample", 1, {outcome: 1}, {k: [v] for k, v in values.items()}
            )
        if window["calls"] >= 32:
            self.flush(action)

    def _emit(self, action, kind, calls, counts, values):
        logger.info(
            "[SR Comm %s] transport=%s action=%s kind=%s calls=%d "
            "counts=%s stats=%s bytes_total=%s",
            self.role,
            self.transport,
            action,
            kind,
            calls,
            dict(counts),
            {k: self.summarize(v) for k, v in values.items()},
            {k: sum(v) for k, v in values.items() if k.endswith("_bytes")},
        )

    def flush(self, action=None):
        for key in list(self.windows) if action is None else [action]:
            window = self.windows.pop(key, None)
            if window and (window["calls"] or window["counts"]):
                self._emit(
                    key, "window", window["calls"], window["counts"], window["values"]
                )


WINDOW_LIMIT = 32
DEVICE_SAMPLE_ROUND = 15  # 0-based index of the 16th metered round
IDLE_FLUSH_S = 2.0


class SRRoundMetrics:
    def __init__(self, role, device_module=None, tp_rank=0):
        self.role = role
        self.device_module = device_module
        self.tp_rank = int(tp_rank or 0)
        self.rounds = 0
        self.skipped_batches = 0
        self.host = Counter()
        self.host_max = Counter()
        self._round_host = Counter()
        self.counts = Counter()
        self.paths = Counter()
        self.device_ms = Counter()
        self.device_samples = Counter()
        self.pending = deque()
        self.active = False
        self._graph_host_samples = []
        self._window_start = None
        self._window_end = None
        self._last_activity = None
        self._pending_graph_rounds = None

    def _timing_failure(self, where, exc):
        if is_device_context_error(exc):
            raise exc
        self.counts["device_timing_dropped_" + where] += 1
        _warn_graph_host("device timing " + where, exc)

    def add_host(self, name, seconds):
        if not self.active:
            return
        self.host[name] += seconds
        self._round_host[name] += seconds
        # A phase may execute several times in one round (e.g. KV moves).
        # Both mean and max describe the sum of those calls per round.
        self.host_max[name] = max(self.host_max[name], self._round_host[name])

    def poll(self):
        # Query only completed events. Bound the queue by skipping new samples
        # when a device has not caught up, rather than forcing synchronization.
        for _ in range(len(self.pending)):
            name, start, end = self.pending.popleft()
            try:
                ready = end.query()
            except Exception as exc:
                self._timing_failure("query", exc)
                continue
            if not ready:
                self.pending.append((name, start, end))
                continue
            try:
                duration = start.elapsed_time(end)
            except Exception as exc:
                self._timing_failure("elapsed", exc)
                continue
            self.device_ms[name] += duration
            self.device_samples[name] += 1

    @contextmanager
    def round(self):
        self.poll()
        self._round_host.clear()
        self.active = True
        start = time.perf_counter()
        if self._window_start is None:
            self._window_start = start
        try:
            yield self
        except BaseException:
            self.counts["failed_rounds"] += 1
            raise
        finally:
            end = time.perf_counter()
            self.add_host("total", end - start)
            self.active = False
            self._round_host.clear()
            self.rounds += 1
            self._window_end = end
            self._last_activity = end
            if self.rounds + self.skipped_batches >= WINDOW_LIMIT:
                # A diagnostics error must not replace a business exception.
                import sys

                if sys.exc_info()[0] is None:
                    self.poll()
                try:
                    self._emit_window("full")
                finally:
                    self._flush_graph_host_safely()

    @contextmanager
    def phase(self, name, *, device=False, stream=None):
        if not self.active:
            yield
            return
        start_time = time.perf_counter()
        event_pair = None
        factory = getattr(self.device_module, "Event", None)
        if (
            device
            and self.tp_rank == 0
            and factory is not None
            and self.rounds == DEVICE_SAMPLE_ROUND
            and len(self.pending) < 16
        ):
            try:
                start, end = factory(enable_timing=True), factory(enable_timing=True)
                if stream is None:
                    start.record()
                else:
                    start.record(stream)
                event_pair = (start, end)
            except Exception as exc:
                self._timing_failure("start", exc)
        completed = False
        try:
            yield
            completed = True
        finally:
            self.add_host(name, time.perf_counter() - start_time)
            # Do not submit instrumentation after a failed device operation or
            # replace its original exception with a timing-event error.
            if event_pair is not None and completed:
                start, end = event_pair
                try:
                    if stream is None:
                        end.record()
                    else:
                        end.record(stream)
                except Exception as exc:
                    self._timing_failure("end", exc)
                else:
                    self.pending.append((name, start, end))

    def note_skip(self, reason, n_reqs):
        """Count one unmetered AR batch. Does not open a host-timed round."""
        if reason not in ("breaker", "high_bs"):
            reason = "high_bs"
        self.counts[f"skip_rpc_{reason}_batches"] += 1
        self.counts[f"skip_rpc_{reason}_requests"] += int(n_reqs)
        self.skipped_batches += 1
        self._last_activity = time.perf_counter()
        if not self.active and self.rounds + self.skipped_batches >= WINDOW_LIMIT:
            self.flush("full")

    def idle_flush(self, now=None):
        if self.active or (self.rounds == 0 and self.skipped_batches == 0):
            return
        now = time.perf_counter() if now is None else now
        if self._last_activity is None or now - self._last_activity < IDLE_FLUSH_S:
            return
        self.flush("idle")

    def flush(self, reason):
        if self.active or (self.rounds == 0 and self.skipped_batches == 0):
            return
        try:
            self._emit_window(reason)
        finally:
            self._flush_graph_host_safely()

    def _fallback_requests(self):
        return (
            self.counts["normal_decode_fallback_requests"]
            + self.counts["skip_rpc_breaker_requests"]
            + self.counts["skip_rpc_high_bs_requests"]
        )

    def _emit_window(self, reason):
        n = self.rounds
        skipped = self.skipped_batches
        wall_ms = None
        if self._window_start is not None and self._window_end is not None:
            wall_ms = round((self._window_end - self._window_start) * 1000, 3)
        busy_ms = round(self.host["total"] * 1000, 3) if n else None
        host_mean = (
            {k: round(v * 1000 / n, 3) for k, v in self.host.items()} if n else {}
        )
        verify_requests = self.counts["verify_requests"]
        fallback_requests = self._fallback_requests()
        step_denom = verify_requests + fallback_requests
        accept_len = (
            round(self.counts["accepted_tokens_including_bonus"] / verify_requests, 3)
            if verify_requests
            else None
        )
        produced = self.counts["accepted_tokens_including_bonus"] + fallback_requests
        tokens_per_step = round(produced / step_denom, 3) if step_denom else None
        fallback_ratio = (
            round(fallback_requests / step_denom, 3) if step_denom else None
        )
        header = (
            f"[SR %s round] rounds={n} reason={reason} "
            f"window_wall_ms={wall_ms} busy_ms={busy_ms} tp_rank={self.tp_rank} "
            f"skipped_batches={skipped} "
        )
        message = (
            header + "host_mean_ms=%s host_max_ms=%s "
            "device_sample_mean_ms=%s device_samples=%s "
            "device_pending=%s counters=%s tail_attention=%s"
        )
        args = [
            self.role,
            host_mean,
            {k: round(v * 1000, 3) for k, v in self.host_max.items()},
            {
                k: round(v / self.device_samples[k], 3)
                for k, v in self.device_ms.items()
            },
            dict(self.device_samples),
            len(self.pending),
            dict(self.counts),
            dict(self.paths),
        ]
        if self.role == "Target":
            message += (
                " accept_len_mean=%s tokens_per_req_step=%s "
                "fallback_ratio=%s verify_rows_without_draft=%s"
            )
            args.extend(
                [
                    accept_len,
                    tokens_per_step,
                    fallback_ratio,
                    self.counts["verify_rows_without_draft"],
                ]
            )
        self._pending_graph_rounds = n
        self.rounds = 0
        self.skipped_batches = 0
        self.host.clear()
        self.host_max.clear()
        self.counts.clear()
        self.paths.clear()
        self.device_ms.clear()
        self.device_samples.clear()
        self._window_start = None
        self._window_end = None
        if self.tp_rank == 0:
            logger.info(message, *args)

    def record_graph_host_sample(self, sample):
        if sample is None or sample._recorded:
            return
        self._graph_host_samples.append(sample)
        sample._recorded = True

    def _flush_graph_host_safely(self):
        samples = self._graph_host_samples
        self._graph_host_samples = []
        window_rounds = self._pending_graph_rounds
        self._pending_graph_rounds = None
        if window_rounds is None:
            window_rounds = self.rounds
        if not samples:
            return
        try:
            self._emit_graph_host_window(samples, window_rounds)
        except Exception as exc:
            _warn_graph_host("flush", exc)

    def _emit_graph_host_window(self, samples, window_rounds):
        grouped = {}
        for sample in samples:
            key = sample.group_key()
            bucket = grouped.get(key)
            if bucket is None:
                ctx = sample.context
                bucket = {
                    "graph_phase": ctx.get("graph_phase", "draft_tree"),
                    "graph_key": ctx.get("graph_key"),
                    "raw_bs": ctx.get("raw_bs"),
                    "capture_bs": ctx.get("capture_bs"),
                    "implementation": ctx.get("implementation"),
                    "overlap": ctx.get("overlap"),
                    "kv_bucket": ctx.get("kv_bucket"),
                    "ok": 0,
                    "failed": 0,
                    "ok_times": defaultdict(list),
                    "failed_times": defaultdict(list),
                    "stage_status": defaultdict(lambda: {"ok": 0, "failed": 0}),
                }
                grouped[key] = bucket
            if sample.outcome == "ok":
                bucket["ok"] += 1
                times = bucket["ok_times"]
            else:
                bucket["failed"] += 1
                times = bucket["failed_times"]
            for name, duration_ms in sample.stages.items():
                times[name].append(duration_ms)
                status = "ok" if sample.stage_completed.get(name, False) else "failed"
                bucket["stage_status"][name][status] += 1
        groups = []
        for key in sorted(grouped, key=lambda item: tuple(repr(part) for part in item)):
            bucket = grouped[key]
            groups.append(
                {
                    "graph_phase": bucket["graph_phase"],
                    "graph_key": bucket["graph_key"],
                    "raw_bs": bucket["raw_bs"],
                    "capture_bs": bucket["capture_bs"],
                    "implementation": bucket["implementation"],
                    "overlap": bucket["overlap"],
                    "kv_bucket": bucket["kv_bucket"],
                    "ok": bucket["ok"],
                    "failed": bucket["failed"],
                    "ok_stats": {
                        GRAPH_HOST_STAGE_LOG[name]: SRCommMetrics.summarize(
                            bucket["ok_times"][name]
                        )
                        for name in GRAPH_HOST_STAGES
                        if bucket["ok_times"].get(name)
                    },
                    "failed_stats": {
                        GRAPH_HOST_STAGE_LOG[name]: SRCommMetrics.summarize(
                            bucket["failed_times"][name]
                        )
                        for name in GRAPH_HOST_STAGES
                        if bucket["failed_times"].get(name)
                    },
                    "stage_status": {
                        GRAPH_HOST_STAGE_LOG[name]: dict(bucket["stage_status"][name])
                        for name in GRAPH_HOST_STAGES
                        if name in bucket["stage_status"]
                    },
                }
            )
        if self.tp_rank == 0:
            logger.info(
                "[SR %s graph host] window_rounds=%s groups=%s",
                self.role,
                window_rounds,
                groups,
            )


def get_sr_round_metrics(owner, role):
    metrics = getattr(owner, "_sr_round_metrics", None)
    if metrics is None:
        metrics = SRRoundMetrics(
            role,
            getattr(owner, "device_module", None),
            tp_rank=getattr(owner, "tp_rank", 0),
        )
        owner._sr_round_metrics = metrics
    return metrics


def flush_sr_round_metrics(owner, reason):
    metrics = getattr(owner, "_sr_round_metrics", None)
    if metrics is not None:
        metrics.flush(reason)


def idle_sr_round_metrics(owner):
    metrics = getattr(owner, "_sr_round_metrics", None)
    if metrics is not None:
        metrics.idle_flush()
