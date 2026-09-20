"""Local SR round timings; device events are polled, never synchronized."""

import logging
import math
import time
from collections import Counter, deque
from contextlib import contextmanager

logger = logging.getLogger(__name__)


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


def get_sr_round_metrics(owner, role):
    metrics = getattr(owner, "_sr_round_metrics", None)
    if metrics is None:
        metrics = SRRoundMetrics(role, getattr(owner, "device_module", None))
        owner._sr_round_metrics = metrics
    return metrics
