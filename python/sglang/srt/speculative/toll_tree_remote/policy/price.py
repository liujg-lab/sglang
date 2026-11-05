"""Token price λ = slow mean tokens / slow mean step time, bucketed by b."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Bucket:
    sum_tokens: float = 0.0
    sum_dt: float = 0.0
    count: int = 0
    fast_tokens: float = 0.0
    fast_dt: float = 0.0
    fast_ready: bool = False
    diverge_streak: int = 0
    anomaly_streak: int = 0

    @property
    def slow_lambda(self) -> float:
        if self.sum_dt <= 0.0 or self.count <= 0:
            return 0.0
        return self.sum_tokens / self.sum_dt

    @property
    def fast_lambda(self) -> float:
        if not self.fast_ready or self.fast_dt <= 0.0:
            return 0.0
        return self.fast_tokens / self.fast_dt


class PriceTracker:
    """One clock for every batch size.

    The first timestamp only starts the clock. Each later step's duration is
    charged to the ``b`` of that ``on_step``, not to a per-bucket clock.
    ``continuous=False`` moves the clock and does not move the means. λ is the
    ratio of means. One step longer than three times the slow mean duration is
    dropped. After ``persist`` consecutive anomalies, that step is written as
    the new level of its ``b`` bucket. A fast EMA that stays 30% away from the
    slow mean also replaces the slow mean.
    """

    def __init__(
        self,
        alpha: float = 0.02,
        anomaly_factor: float = 3.0,
        diverge: float = 0.30,
        persist: int = 8,
        min_anomaly_samples: int = 8,
    ) -> None:
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if int(persist) < 1:
            raise ValueError("persist must be >= 1")
        self.alpha = float(alpha)
        self.anomaly_factor = float(anomaly_factor)
        self.diverge = float(diverge)
        self.persist = int(persist)
        self.min_anomaly_samples = int(min_anomaly_samples)
        self._last_ns: int | None = None
        self._buckets: dict[int, _Bucket] = {}

    def _bucket(self, bs: int) -> _Bucket:
        key = int(bs)
        if key < 1:
            raise ValueError(f"bs must be >= 1, got {bs}")
        slot = self._buckets.get(key)
        if slot is None:
            slot = _Bucket()
            self._buckets[key] = slot
        return slot

    def on_step(self, now_ns: int, committed: int, bs: int, continuous: bool) -> None:
        slot = self._bucket(bs)
        now = int(now_ns)
        if self._last_ns is None:
            self._last_ns = now
            return
        dt_s = (now - self._last_ns) / 1e9
        self._last_ns = now
        if not continuous or dt_s <= 0.0:
            return
        slow_dt = slot.sum_dt / slot.count if slot.count else 0.0
        is_anomaly = (
            slot.count >= self.min_anomaly_samples
            and slow_dt > 0.0
            and dt_s > self.anomaly_factor * slow_dt
        )
        if is_anomaly:
            slot.anomaly_streak += 1
            if slot.anomaly_streak < self.persist:
                return
            self._adopt_level(slot, float(committed), dt_s)
            return
        slot.anomaly_streak = 0
        self._blend(slot, float(committed), dt_s)

    def _adopt_level(self, slot: _Bucket, tokens: float, dt_s: float) -> None:
        slot.sum_tokens = tokens
        slot.sum_dt = dt_s
        slot.count = 1
        slot.fast_tokens = tokens
        slot.fast_dt = dt_s
        slot.fast_ready = True
        slot.anomaly_streak = 0
        slot.diverge_streak = 0

    def _blend(self, slot: _Bucket, tokens: float, dt_s: float) -> None:
        if not slot.fast_ready:
            slot.fast_tokens = tokens
            slot.fast_dt = dt_s
            slot.fast_ready = True
        else:
            keep = 1.0 - self.alpha
            slot.fast_tokens = keep * slot.fast_tokens + self.alpha * tokens
            slot.fast_dt = keep * slot.fast_dt + self.alpha * dt_s
        slot.sum_tokens += tokens
        slot.sum_dt += dt_s
        slot.count += 1
        if self._diverged(slot):
            slot.diverge_streak += 1
        else:
            slot.diverge_streak = 0
        if slot.diverge_streak >= self.persist and slot.fast_dt > 0.0:
            slot.sum_tokens = slot.fast_tokens
            slot.sum_dt = slot.fast_dt
            slot.count = 1
            slot.diverge_streak = 0

    def _diverged(self, slot: _Bucket) -> bool:
        slow = slot.slow_lambda
        fast = slot.fast_lambda
        if slow <= 0.0 or not slot.fast_ready:
            return False
        return abs(fast - slow) / slow >= self.diverge

    def price(self, bs: int, model_prior: float) -> float:
        """λ for this batch size. An empty bucket returns ``model_prior``."""
        slot = self._buckets.get(int(bs))
        if slot is None or slot.count <= 0 or slot.sum_dt <= 0.0:
            return float(model_prior)
        return slot.slow_lambda
