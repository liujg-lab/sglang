"""Local SR round timings; device events are polled, never synchronized."""

import logging
import math
import time
from collections import Counter, defaultdict, deque
from contextlib import contextmanager

logger = logging.getLogger(__name__)

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


class SRRoundMetrics:
    def __init__(self, role, device_module=None):
        self.role = role
        self.device_module = device_module
        self.rounds = 0
        self.host = Counter()
        self.host_max = Counter()
        self.counts = Counter()
        self.paths = Counter()
        self.device_ms = Counter()
        self.device_samples = Counter()
        self.pending = deque()
        self.active = False
        self._graph_host_samples = []

    def add_host(self, name, seconds):
        if not self.active:
            return
        self.host[name] += seconds
        current = self.host_max[name]
        if seconds > current:
            self.host_max[name] = seconds

    def poll(self):
        # Query only completed events. Bound the queue by skipping new samples
        # when a device has not caught up, rather than forcing synchronization.
        for _ in range(len(self.pending)):
            name, start, end = self.pending.popleft()
            if end.query():
                self.device_ms[name] += start.elapsed_time(end)
                self.device_samples[name] += 1
            else:
                self.pending.append((name, start, end))

    @contextmanager
    def round(self):
        self.poll()
        self.active = True
        start = time.perf_counter()
        try:
            yield self
        except BaseException:
            self.counts["failed_rounds"] += 1
            raise
        finally:
            self.add_host("total", time.perf_counter() - start)
            self.active = False
            self.rounds += 1
            if self.rounds % 32 == 0:
                self.poll()
                try:
                    logger.info(
                        "[SR %s round] rounds=32 host_mean_ms=%s host_max_ms=%s "
                        "device_sample_mean_ms=%s device_samples=%s "
                        "device_pending=%s counters=%s tail_attention=%s accept_len_mean=%s",
                        self.role,
                        {k: round(v * 1000 / 32, 3) for k, v in self.host.items()},
                        {k: round(v * 1000, 3) for k, v in self.host_max.items()},
                        {
                            k: round(v / self.device_samples[k], 3)
                            for k, v in self.device_ms.items()
                        },
                        dict(self.device_samples),
                        len(self.pending),
                        dict(self.counts),
                        dict(self.paths),
                        (
                            round(
                                self.counts["accepted_tokens_including_bonus"]
                                / self.counts["verify_requests"],
                                3,
                            )
                            if self.counts["verify_requests"]
                            else None
                        ),
                    )
                    self.host.clear()
                    self.host_max.clear()
                    self.counts.clear()
                    self.paths.clear()
                    self.device_ms.clear()
                    self.device_samples.clear()
                finally:
                    self._flush_graph_host_safely()

    @contextmanager
    def phase(self, name, *, device=False):
        if not self.active:
            yield
            return
        start_time = time.perf_counter()
        event_pair = None
        factory = getattr(self.device_module, "Event", None)
        if (
            device
            and factory is not None
            and self.rounds % 32 == 0
            and len(self.pending) < 16
        ):
            start, end = factory(enable_timing=True), factory(enable_timing=True)
            start.record()
            event_pair = (start, end)
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
                end.record()
                self.pending.append((name, start, end))

    def record_graph_host_sample(self, sample):
        if sample is None or sample._recorded:
            return
        self._graph_host_samples.append(sample)
        sample._recorded = True

    def _flush_graph_host_safely(self):
        samples = self._graph_host_samples
        self._graph_host_samples = []
        if not samples:
            return
        try:
            self._emit_graph_host_window(samples)
        except Exception as exc:
            _warn_graph_host("flush", exc)

    def _emit_graph_host_window(self, samples):
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
        logger.info(
            "[SR %s graph host] window_rounds=32 groups=%s", self.role, groups
        )


def get_sr_round_metrics(owner, role):
    metrics = getattr(owner, "_sr_round_metrics", None)
    if metrics is None:
        metrics = SRRoundMetrics(role, getattr(owner, "device_module", None))
        owner._sr_round_metrics = metrics
    return metrics
