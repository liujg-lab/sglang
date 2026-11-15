"""Flat layer width and online verify count for one ClearTree round.

W is a fixed table: batch size 1 uses 8, every other batch uses 4. K is
chosen among 8, 16, and 32. Target graphs are captured at K + 1, so a short
tree is padded only inside the graph of the K that was chosen.

Warmup, probe rounds, and rounds whose verify or layer sample is still short
keep the tau stop. A probe round also locks K. Other rounds stop by comparing
the value of one more layer with the value of verifying now.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

VERIFY_KS = (8, 16, 32)
TARGET_CAPACITY = 33
DRAFT_CAPACITY = 8
LAYER_WIDTHS = (4, 8)
BONUS_NTPBS = (5, 9)
WARMUP_ROUNDS = 10
PROBE_PER_K = 2
CALIBRATE_LAMBDA = 3.0
PAD_PARENT = -2

_clock: Dict[tuple, list] = {}
_clock_count: Dict[int, int] = {}
_price_tok: Dict[int, float] = {}
_price_ms: Dict[int, float] = {}
_sum_a: Dict[str, float] = {}
_sum_e: Dict[str, float] = {}
_total_a = 0.0
_total_e = 0.0
_pending: Dict[str, dict] = {}
_probes: Dict[int, Dict[int, int]] = {}
_rounds = 0


def reset_policy() -> None:
    """Drop prices, accept ratios, probe counts, and verify samples."""
    global _total_a, _total_e, _rounds
    _clock.clear()
    _clock_count.clear()
    _price_tok.clear()
    _price_ms.clear()
    _sum_a.clear()
    _sum_e.clear()
    _pending.clear()
    _probes.clear()
    _total_a = 0.0
    _total_e = 0.0
    _rounds = 0


def flat_width(batch_size: int) -> int:
    """Layer width. One request uses 8; every larger batch uses 4."""
    return 8 if int(batch_size) == 1 else 4


def capture_ntpb(role: Optional[str]) -> list:
    """Graph widths for this role, widest first, including the 1-token decode graph."""
    if role == "target":
        return [TARGET_CAPACITY, 17, 9, 1]
    if role == "draft":
        return [DRAFT_CAPACITY, 4, 1]
    return [1]


def bonus_ntpbs() -> tuple:
    """Draft bonus graphs. Each is one wider than a layer width."""
    return BONUS_NTPBS


def select_k(values: Dict[int, float]) -> int:
    """Argmax of J. Equal values keep the larger K."""
    if not values:
        raise ValueError("select_k needs at least one J")
    return max(values, key=lambda key: (float(values[key]), int(key)))


def layer_should_continue(stop_value: float, next_value: float) -> bool:
    """True when one more layer is worth more than verifying now."""
    return float(next_value) > float(stop_value)


def ingest_verify_samples(samples) -> None:
    """Record samples carried on one ClearTree request. ``device_us`` is microseconds."""
    for sample in list(samples or []):
        if not isinstance(sample, (list, tuple)) or len(sample) != 4:
            continue
        batch, k, bucket, device_us = sample
        if device_us is None:
            continue
        observe_verify(int(batch), int(k), int(bucket), float(device_us) / 1000.0)


def locked_k_from_env() -> Optional[int]:
    """Fixed verify count for a baseline run. Unset means the online rule."""
    raw = os.environ.get("SGLANG_CLEAR_LOCK_K", "").strip()
    if not raw:
        return None
    return int(raw)


def observe_verify(batch: int, k: int, bucket: Optional[int], ms: float) -> None:
    """One clean graph verify. Eager and over-bucket rows are not recorded."""
    if int(k) not in VERIFY_KS or bucket is None or ms is None:
        return
    key = (int(batch), int(k), int(bucket))
    _clock.setdefault(key, []).append(float(ms))
    _clock_count[int(k)] = _clock_count.get(int(k), 0) + 1


def verify_ready(k: int) -> bool:
    """A tier is usable once it has been seen and the table has a full sample."""
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        VERIFY_MIN_SAMPLES,
    )

    if _clock_count.get(int(k), 0) < 1:
        return False
    return sum(_clock_count.values()) >= VERIFY_MIN_SAMPLES


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    mid = count // 2
    if count % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _steady_verify_samples(rows: Sequence[float]) -> list:
    """Drop one high spike, then average what remains.

    The comparison is against the median of the other samples. With only two
    or three probes, that median would include the spike and hide it.
    """
    kept = [float(value) for value in rows]
    if len(kept) < 2:
        return kept
    largest = max(kept)
    rest = list(kept)
    rest.remove(largest)
    if largest > _median(rest) * 1.5:
        return rest
    return kept


def predict_verify_ms(batch: int, k: int, bucket: Optional[int]) -> Optional[float]:
    """Device time for this batch, K, and KV bucket, after one high spike is dropped."""
    if bucket is None or not verify_ready(k):
        return None
    rows = _clock.get((int(batch), int(k), int(bucket)))
    if not rows:
        return None
    kept = _steady_verify_samples(rows)
    return sum(kept) / float(len(kept))


def observe_price(batch: int, tokens: float, ms: float) -> None:
    """Tokens per millisecond on one clean decode round."""
    if ms is None or float(ms) <= 0 or tokens is None or float(tokens) < 0:
        return
    key = int(batch)
    _price_tok[key] = _price_tok.get(key, 0.0) + float(tokens)
    _price_ms[key] = _price_ms.get(key, 0.0) + float(ms)


def theta(batch: int) -> Optional[float]:
    """Measured tokens per millisecond at this batch size."""
    key = int(batch)
    spent = _price_ms.get(key, 0.0)
    if spent <= 0:
        return None
    return _price_tok.get(key, 0.0) / spent


def observe_accept(rid: str, accepted: float, expected: float) -> None:
    """One settled round. ``expected`` is the path mass of the verified nodes."""
    global _total_a, _total_e
    key = str(rid)
    taken = max(float(accepted), 0.0)
    mass = max(float(expected), 0.0)
    if taken == 0.0 and mass == 0.0:
        return
    _sum_a[key] = _sum_a.get(key, 0.0) + taken
    _sum_e[key] = _sum_e.get(key, 0.0) + mass
    _total_a += taken
    _total_e += mass


def accept_ratio(rid: str) -> float:
    """Shrink the request ratio toward the pooled ratio. Empty data stays 1."""
    key = str(rid)
    taken = _sum_a.get(key, 0.0)
    mass = _sum_e.get(key, 0.0)
    pooled = (_total_a / _total_e) if _total_e > 0 else 1.0
    return (taken + CALIBRATE_LAMBDA * pooled) / (mass + CALIBRATE_LAMBDA)


def note_verified(rid: str, prefix: int, expected: float) -> None:
    """Remember this round until the next packet reports how many tokens committed."""
    _pending[str(rid)] = {"prefix": int(prefix), "e": float(expected)}


def settle_verified(rids: Sequence[str], prefixes: Sequence[int]) -> None:
    """Turn the pending path mass into an accept count. Missing requests drop."""
    found = {str(rid): int(prefix) for rid, prefix in zip(rids, prefixes)}
    for rid in list(_pending):
        item = _pending.pop(rid)
        if rid not in found:
            continue
        observe_accept(rid, found[rid] - int(item["prefix"]), float(item["e"]))


def _take_probe(bucket: Optional[int], max_ell: int, buckets: Sequence[int]) -> Optional[int]:
    """Lock the next K that still fits this prefix. An eager K keeps its quota."""
    if bucket is None:
        return None
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        smallest_bucket,
    )

    row = _probes.get(int(bucket))
    if row is None:
        row = {int(k): PROBE_PER_K for k in VERIFY_KS}
        _probes[int(bucket)] = row
    for k in VERIFY_KS:
        if smallest_bucket(int(max_ell) + int(k) + 1, buckets) is None:
            continue
        if row[k] > 0:
            row[k] -= 1
            return int(k)
    return None


def fitting_ks(max_ell: int, buckets: Sequence[int]) -> list:
    """Captured K whose KV bucket exists. An empty result still keeps K = 8."""
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        smallest_bucket,
    )

    found = []
    for k in VERIFY_KS:
        if smallest_bucket(int(max_ell) + int(k) + 1, buckets) is not None:
            found.append(int(k))
    return found or [VERIFY_KS[0]]


class RoundPlan:
    """Stop and verify choice for the trees grown in this round."""

    def __init__(
        self,
        *,
        width: int,
        k: int,
        use_tau: bool,
        locked_k: Optional[int],
        theta_value: Optional[float],
        layer_ms: Optional[float],
        max_ell: int,
        batch: int,
        buckets: Sequence[int],
    ) -> None:
        self.width = int(width)
        self.k = int(k)
        self.use_tau = bool(use_tau)
        self.locked_k = None if locked_k is None else int(locked_k)
        self.theta_value = theta_value
        self.layer_ms = layer_ms
        self.max_ell = int(max_ell)
        self.batch = int(batch)
        self.buckets = tuple(buckets)
        self._peers = []
        self._at = None
        self._grow = False

    def begin_layer(self, states) -> None:
        """The layer was just written. Judge these trees together."""
        self._peers = list(states)
        self._at = None
        self._grow = False

    def should_grow(self, state) -> bool:
        from sglang.srt.speculative.clear_tree_remote.clear_tree_ops import depth_rule

        if int(state.depth) >= int(state.config.max_depth) - 1:
            return False
        if self.use_tau or self.theta_value is None or self.layer_ms is None:
            return depth_rule(state)
        if self._at != int(state.depth):
            self._grow = self._continue(self._peers or [state])
            self._at = int(state.depth)
        return bool(self._grow)

    def verify_k(self, state) -> int:
        if self.locked_k is not None:
            return int(self.locked_k)
        if self.use_tau:
            return int(self.k)
        return self._select(self._peers or [state])

    def note(self, state) -> None:
        rid = getattr(state, "rid", None)
        prefix = getattr(state, "prefix", None)
        if rid is None or prefix is None:
            return
        scores = _path_scores(state)
        note_verified(str(rid), int(prefix), _prefix_mass(scores, self.verify_k(state)))

    def _time(self, k: int) -> Optional[float]:
        from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
            smallest_bucket,
        )

        bucket = smallest_bucket(self.max_ell + int(k) + 1, self.buckets)
        return predict_verify_ms(self.batch, int(k), bucket)

    def _select(self, states) -> int:
        values = self._values(states, include_gain=False)
        if not values:
            return int(self.k)
        return select_k(values)

    def _continue(self, states) -> bool:
        current = self._values(states, include_gain=False)
        nxt = self._values(states, include_gain=True)
        if not current or not nxt or self.theta_value is None or self.layer_ms is None:
            return False
        stop_value = max(current.values())
        next_value = max(nxt.values()) - float(self.theta_value) * float(self.layer_ms)
        return layer_should_continue(stop_value, next_value)

    def _values(self, states, *, include_gain: bool) -> Dict[int, float]:
        if self.theta_value is None:
            return {}
        values = {}
        for k in VERIFY_KS:
            spent = self._time(k)
            if spent is None:
                continue
            benefit = 0.0
            for state in states:
                ratio = accept_ratio(str(getattr(state, "rid", "")))
                scores = _path_scores(state)
                benefit += ratio * _prefix_mass(scores, k)
                if include_gain:
                    benefit += ratio * _marginal_gain(state, k)
            values[int(k)] = benefit - float(self.theta_value) * float(spent)
        return values


def make_round(scheduler, reqs) -> RoundPlan:
    """Choose W and the warmup, probe, or value rule for these requests."""
    global _rounds
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        configured_buckets,
        predict_layer_ms,
        smallest_bucket,
    )

    batch = max(len(list(reqs)), 1)
    prefixes = [
        len(getattr(req, "origin_input_ids", None) or [])
        + len(getattr(req, "output_ids", None) or [])
        for req in reqs
    ]
    max_ell = max(prefixes) if prefixes else 0
    buckets = configured_buckets()
    _rounds += 1
    regime = smallest_bucket(max_ell + 1, buckets)
    locked_env = locked_k_from_env()
    probe = (
        None
        if locked_env is not None
        else _take_probe(regime, max_ell, buckets)
    )
    price = theta(batch)
    width = flat_width(batch)
    layer_ms = predict_layer_ms(scheduler, width, batch, sum(prefixes))
    ready = any(
        verify_ready(k) and _time_for(batch, max_ell, k, buckets) is not None
        for k in VERIFY_KS
    )
    use_tau = (
        locked_env is not None
        or _rounds <= WARMUP_ROUNDS
        or probe is not None
        or price is None
        or layer_ms is None
        or not ready
    )
    if locked_env is not None:
        chosen = int(locked_env)
        locked = chosen
    elif probe is not None:
        chosen = int(probe)
        locked = chosen
    else:
        chosen = max(fitting_ks(max_ell, buckets))
        locked = chosen if use_tau else None
    return RoundPlan(
        width=width,
        k=chosen,
        use_tau=use_tau,
        locked_k=locked,
        theta_value=price,
        layer_ms=layer_ms,
        max_ell=max_ell,
        batch=batch,
        buckets=buckets,
    )


def attach_round(scheduler, reqs) -> RoundPlan:
    """Store the round plan where the grow loop can see it."""
    plan = make_round(scheduler, reqs)
    scheduler._clear_round_plan = plan
    scheduler._clear_round_rids = [getattr(req, "rid", "") for req in reqs]
    scheduler._clear_round_prefixes = [
        len(getattr(req, "origin_input_ids", None) or [])
        + len(getattr(req, "output_ids", None) or [])
        for req in reqs
    ]
    return plan


def remember_choice(scheduler, k: int, width: int) -> None:
    """The open ledger round should record the K and W that were actually used."""
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import bind_ledger

    ledger = bind_ledger(scheduler)
    if ledger is None or ledger.open is None:
        return
    ledger.open.k = int(k)
    ledger.open.w = int(width)


def _time_for(batch, max_ell, k, buckets) -> Optional[float]:
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        smallest_bucket,
    )

    bucket = smallest_bucket(int(max_ell) + int(k) + 1, buckets)
    return predict_verify_ms(batch, int(k), bucket)


def _path_scores(state) -> list:
    if int(getattr(state, "depth", 0) or 0) < 1:
        return []
    flat = state.weight[: int(state.depth)].reshape(-1)
    return [float(value) for value in flat.tolist()]


def _prefix_mass(scores: Sequence[float], k: int) -> float:
    ordered = sorted((float(value) for value in scores), reverse=True)
    return float(sum(ordered[: int(k)]))


def _marginal_gain(state, k: int) -> float:
    """How much of the newest layer still clears the current K-th score."""
    scores = _path_scores(state)
    if not scores or int(state.depth) < 1:
        return 0.0
    ordered = sorted(scores, reverse=True)
    theta_k = ordered[int(k) - 1] if len(ordered) >= int(k) else 0.0
    latest = [float(value) for value in state.weight[int(state.depth) - 1].tolist()]
    return float(sum(max(0.0, score - theta_k) for score in latest))
