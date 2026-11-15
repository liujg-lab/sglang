"""Flat layer width and online verify count for one ClearTree round.

W starts from a fixed table: batch size 1 uses 8, every other batch uses 4.
A shadow replay may replace that width only after the measured 8-versus-4
bias is inside 3 percent. K is chosen among 8, 16, and 32. Target graphs
are captured at K + 1, so a short tree is padded only inside the graph of
the K that was chosen.

Warmup, probe rounds, and rounds whose verify or layer sample is still short
keep the tau stop. A probe round also locks K. Other rounds stop by comparing
one J across verify counts: keep the prefix of requests whose next layer is
worth more than verifying now. Every request in the round, including one that
already stopped growing, stays in that J. Verify time stays on the full batch.
The layer cost is only the requests that would be forwarded.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional, Sequence

VERIFY_KS = (8, 16, 32)
TARGET_CAPACITY = 33
DRAFT_CAPACITY = 8
LAYER_WIDTHS = (4, 8)
BONUS_NTPBS = (5, 9)
WARMUP_ROUNDS = 10
PROBE_PER_K = 2
CANDIDATE_WIDTHS = (2, 4, 8)
REPLAY_EVERY = 20
REPLAY_GAIN = 0.03
NARROW_MIN_ROUNDS = 30
WIDE_MIN_ROUNDS = 10
BIAS_MIN_ROUNDS = 30
BIAS_LIMIT = 0.03
CALIBRATE_LAMBDA = 3.0
PRICE_EMA_FLOOR = 0.05
PRICE_PRIOR_N0 = 10.0
PRICE_PRIOR_EXP = 0.8
PAD_PARENT = -2

_clock: Dict[tuple, list] = {}
_clock_count: Dict[int, int] = {}
_price_n: Dict[int, int] = {}
_price_tok: Dict[int, float] = {}
_price_ms: Dict[int, float] = {}
_sum_a: Dict[str, float] = {}
_sum_e: Dict[str, float] = {}
_total_a = 0.0
_total_e = 0.0
_pending: Dict[str, dict] = {}
_probes: Dict[tuple, Dict[int, int]] = {}
_rounds = 0
_persistent_w: Dict[int, int] = {}
_eligible_n: Dict[int, int] = {}
_replay_n: Dict[int, int] = {}
_replay_arms: Dict[int, Dict[int, Dict[str, float]]] = {}
_would_w: Dict[int, int] = {}
_bias_n8: Dict[int, int] = {}
_bias_n4: Dict[int, int] = {}
_bias_tok8: Dict[int, float] = {}
_bias_ms8: Dict[int, float] = {}
_bias_replay_tok: Dict[int, float] = {}
_bias_replay_ms: Dict[int, float] = {}
_bias_tok4: Dict[int, float] = {}
_bias_ms4: Dict[int, float] = {}


def reset_policy() -> None:
    """Drop prices, accept ratios, probe counts, verify samples, and replay."""
    global _total_a, _total_e, _rounds
    _clock.clear()
    _clock_count.clear()
    _price_n.clear()
    _price_tok.clear()
    _price_ms.clear()
    _sum_a.clear()
    _sum_e.clear()
    _pending.clear()
    _probes.clear()
    _persistent_w.clear()
    _eligible_n.clear()
    _replay_n.clear()
    _replay_arms.clear()
    _would_w.clear()
    _bias_n8.clear()
    _bias_n4.clear()
    _bias_tok8.clear()
    _bias_ms8.clear()
    _bias_replay_tok.clear()
    _bias_replay_ms.clear()
    _bias_tok4.clear()
    _bias_ms4.clear()
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


def request_terms(state, k: int) -> tuple:
    """Path mass inside K, and how much of the newest layer still clears it."""
    return _prefix_mass(_path_scores(state), int(k)), _marginal_gain(state, int(k))


def select_continuing(
    all_states,
    eligible,
    *,
    theta: float,
    verify_times: Dict[int, float],
    layer_cost,
    kappa: Optional[float],
) -> Optional[list]:
    """Requests to forward, or None when no candidate has a layer cost.

    ``J`` uses every tree in the round. ``T`` is the caller's verify time for
    the full batch. The layer cost is charged only for the ``m`` requests that
    would continue. An empty list means stop. Ties keep the larger K, then
    the larger group.
    """
    states = list(all_states)
    growing = list(eligible)
    if not growing:
        return []
    price = float(theta)
    slope = 0.0 if kappa is None else float(kappa)
    joint = {}
    gains = {}
    for k, spent in verify_times.items():
        if spent is None or int(k) not in VERIFY_KS:
            continue
        benefit = 0.0
        per_state = {}
        for state in states:
            ratio = accept_ratio(str(getattr(state, "rid", "")))
            mass, gain = request_terms(state, int(k))
            benefit += ratio * mass
            per_state[id(state)] = ratio * gain
        joint[int(k)] = benefit - price * float(spent)
        gains[int(k)] = per_state
    if not joint:
        return None
    stop_value = max(joint.values())
    best = None
    any_cost = False
    for k, base in joint.items():
        ranked = []
        for state in growing:
            gain = float(gains[k].get(id(state), 0.0))
            ell = int(getattr(state, "prefix", 0) or 0)
            ranked.append((gain - price * slope * ell, gain, ell, state))
        ranked.sort(key=lambda item: item[0], reverse=True)
        running_gain = 0.0
        running_ell = 0
        for m, (_net, gain, ell, _state) in enumerate(ranked, start=1):
            running_gain += gain
            running_ell += ell
            cost = layer_cost(m, running_ell)
            if cost is None:
                continue
            any_cost = True
            value = base + running_gain - price * float(cost)
            choice = (value, int(k), int(m))
            if best is None or choice > best[0]:
                best = (choice, [item[3] for item in ranked[:m]])
    if not any_cost or best is None:
        return None
    if best[0][0] > stop_value:
        return list(best[1])
    return []


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


def _observe_price_into(counts, tokens_ema, ms_ema, batch, tokens, ms) -> bool:
    """One EMA step. The first sample uses step 1; later steps stay at least 0.05."""
    if ms is None or float(ms) <= 0 or tokens is None or float(tokens) < 0:
        return False
    key = int(batch)
    if key < 1:
        return False
    count = int(counts.get(key, 0)) + 1
    step = max(PRICE_EMA_FLOOR, 1.0 / float(count))
    counts[key] = count
    tokens_ema[key] = (1.0 - step) * float(tokens_ema.get(key, 0.0)) + step * float(
        tokens
    )
    ms_ema[key] = (1.0 - step) * float(ms_ema.get(key, 0.0)) + step * float(ms)
    return True


def _raw_theta(counts, tokens_ema, ms_ema, batch: int) -> Optional[float]:
    key = int(batch)
    spent = float(ms_ema.get(key, 0.0))
    if int(counts.get(key, 0)) <= 0 or spent <= 0:
        return None
    return float(tokens_ema[key]) / spent


def _price_prior(counts, tokens_ema, ms_ema, batch: int) -> Optional[float]:
    """log Θ = α + β·log b from the other batch sizes. One neighbor scales by b^0.8."""
    if int(batch) < 1:
        return None
    usable = []
    for key in counts:
        if int(key) == int(batch):
            continue
        raw = _raw_theta(counts, tokens_ema, ms_ema, int(key))
        if raw is None or raw <= 0:
            continue
        usable.append((int(key), float(raw)))
    if not usable:
        return None
    if len(usable) == 1:
        other_b, other_theta = usable[0]
        return other_theta * (float(batch) / float(other_b)) ** PRICE_PRIOR_EXP
    xs = [math.log(float(other_b)) for other_b, _theta in usable]
    ys = [math.log(other_theta) for _other_b, other_theta in usable]
    mean_x = sum(xs) / float(len(xs))
    mean_y = sum(ys) / float(len(ys))
    var_x = sum((item - mean_x) ** 2 for item in xs)
    if var_x <= 0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = cov / var_x
    intercept = mean_y - slope * mean_x
    return math.exp(intercept + slope * math.log(float(batch)))


def _theta_from(counts, tokens_ema, ms_ema, batch: int) -> Optional[float]:
    raw = _raw_theta(counts, tokens_ema, ms_ema, batch)
    prior = _price_prior(counts, tokens_ema, ms_ema, batch)
    if raw is None and prior is None:
        return None
    if raw is None:
        return prior
    if prior is None:
        return raw
    count = float(counts.get(int(batch), 0))
    weight = count / (count + PRICE_PRIOR_N0)
    return weight * raw + (1.0 - weight) * prior


def observe_price(batch: int, tokens: float, ms: float) -> None:
    """One clean decode round updates this batch size's token and time EMAs."""
    _observe_price_into(_price_n, _price_tok, _price_ms, batch, tokens, ms)


def theta(batch: int) -> Optional[float]:
    """Pure-decode tokens per millisecond, blended with the other-batch prior."""
    return _theta_from(_price_n, _price_tok, _price_ms, int(batch))


def compare_price(rounds, gaps) -> Dict[int, dict]:
    """Replay both price estimates. This does not change the SR protocol.

    Dropping a round uses only ``price_kept`` samples. Subtracting the period
    adds PREFILL and new-request rounds after removing the matched Target
    PREFILL, and adds a round that only followed idle. FINISH and ABORT stay
    out of both. Alignment is ``rpc_seq``, not the wall clock.
    """
    gap_wall = {}
    for gap in list(gaps or []):
        if not isinstance(gap, dict):
            continue
        seq = gap.get("rpc_seq")
        wall = gap.get("wall_us")
        if seq is None or wall is None:
            continue
        gap_wall[int(seq)] = gap_wall.get(int(seq), 0) + int(wall)
    dropped_n: Dict[int, int] = {}
    dropped_tok: Dict[int, float] = {}
    dropped_ms: Dict[int, float] = {}
    deducted_n: Dict[int, int] = {}
    deducted_tok: Dict[int, float] = {}
    deducted_ms: Dict[int, float] = {}
    seen = set()
    for record in list(rounds or []):
        if not isinstance(record, dict):
            continue
        tokens = record.get("price_tokens")
        spent = record.get("price_ms")
        if tokens is None or spent is None:
            continue
        batch = int(record.get("b", 0) or 0)
        if batch < 1:
            continue
        seen.add(batch)
        reasons = {str(item) for item in (record.get("exclude_reason") or [])}
        if record.get("price_kept"):
            _observe_price_into(dropped_n, dropped_tok, dropped_ms, batch, tokens, spent)
            _observe_price_into(
                deducted_n, deducted_tok, deducted_ms, batch, tokens, spent
            )
            continue
        if reasons.intersection(("finish", "abort")):
            continue
        if "prefill" in reasons or "new_rid" in reasons:
            matched = 0
            for seq in list(record.get("gap_rpc") or []):
                matched += int(gap_wall.get(int(seq), 0))
            adjusted = float(spent) - matched / 1000.0
            if adjusted <= 0:
                continue
            _observe_price_into(
                deducted_n, deducted_tok, deducted_ms, batch, tokens, adjusted
            )
            continue
        if "idle_start" in reasons:
            _observe_price_into(
                deducted_n, deducted_tok, deducted_ms, batch, tokens, spent
            )
    report = {}
    for batch in sorted(seen):
        discard = _theta_from(dropped_n, dropped_tok, dropped_ms, batch)
        deduct = _theta_from(deducted_n, deducted_tok, deducted_ms, batch)
        relative = None
        if discard not in (None, 0.0) and deduct is not None:
            relative = abs(float(discard) - float(deduct)) / abs(float(discard))
        report[int(batch)] = {
            "discard": discard,
            "deduct": deduct,
            "relative": relative,
        }
    return report


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


def _take_probe(
    bucket: Optional[int], max_ell: int, buckets: Sequence[int], batch: int
) -> Optional[int]:
    """Lock the next K that still fits this prefix.

    Each batch size keeps its own quota inside a KV bucket, so a new
    concurrency starts again at K = 8.
    """
    if bucket is None:
        return None
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        smallest_bucket,
    )

    key = (int(bucket), int(batch))
    row = _probes.get(key)
    if row is None:
        row = {int(k): PROBE_PER_K for k in VERIFY_KS}
        _probes[key] = row
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
        scheduler=None,
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
        self._scheduler = scheduler
        self._peers = []
        self._all = []
        self._at = None
        self._keep_ids = set()
        self._frozen_k = None

    @property
    def defer_finish(self) -> bool:
        """Cut every window after the round, so one K is used for all of them.

        A single request is cut where it stops. Lookahead only grows one tree,
        and that path reads the window immediately.
        """
        return (
            self.batch > 1
            and not self.use_tau
            and self.locked_k is None
            and self.theta_value is not None
            and self.layer_ms is not None
        )

    def bind_states(self, states) -> None:
        """Every tree in the round, including one that later stops growing."""
        self._all = list(states)

    def begin_layer(self, states) -> None:
        """The layer was just written. Judge these trees together."""
        self._peers = list(states)
        self._at = None
        self._keep_ids = set()

    def should_grow(self, state) -> bool:
        from sglang.srt.speculative.clear_tree_remote.clear_tree_ops import depth_rule

        if int(state.depth) >= int(state.config.max_depth) - 1:
            return False
        if self.use_tau or self.theta_value is None or self.layer_ms is None:
            return depth_rule(state)
        if self._at != int(state.depth):
            peers = list(self._peers or [state])
            eligible = [
                item
                for item in peers
                if int(item.depth) < int(item.config.max_depth) - 1
            ]
            self._keep_ids = {id(item) for item in self._choose_keep(eligible)}
            self._at = int(state.depth)
        return id(state) in self._keep_ids

    def verify_k(self, state) -> int:
        if self._frozen_k is not None:
            return int(self._frozen_k)
        if self.locked_k is not None:
            return int(self.locked_k)
        if self.use_tau:
            return int(self.k)
        return self._select(self._all or self._peers or [state])

    def freeze_k(self, states) -> int:
        """One verify count for every tree, chosen after growth has stopped."""
        chosen = self._select(list(states))
        self._frozen_k = int(chosen)
        return int(self._frozen_k)

    def note(self, state) -> None:
        rid = getattr(state, "rid", None)
        prefix = getattr(state, "prefix", None)
        if rid is None or prefix is None:
            return
        scores = _path_scores(state)
        note_verified(str(rid), int(prefix), _prefix_mass(scores, self.verify_k(state)))

    def _choose_keep(self, eligible) -> list:
        times = {}
        for k in VERIFY_KS:
            spent = self._time(k)
            if spent is not None:
                times[int(k)] = float(spent)
        kept = select_continuing(
            self._all or eligible,
            eligible,
            theta=float(self.theta_value),
            verify_times=times,
            layer_cost=self._layer_cost,
            kappa=self._kappa(),
        )
        if kept is None:
            if self._continue(eligible):
                return list(eligible)
            return []
        return list(kept)

    def _layer_cost(self, active: int, sum_ell: int) -> Optional[float]:
        if self._scheduler is None:
            return None
        from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
            predict_layer_ms,
        )

        return predict_layer_ms(
            self._scheduler, self.width, int(active), int(sum_ell)
        )

    def _kappa(self) -> Optional[float]:
        if self._scheduler is None:
            return None
        from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
            layer_sum_ell_slope,
        )

        return layer_sum_ell_slope(self._scheduler, self.width)

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
        else _take_probe(regime, max_ell, buckets, batch)
    )
    price = theta(batch)
    eligible = (
        _rounds > WARMUP_ROUNDS
        and probe is None
        and locked_env is None
        and _requests_greedy(reqs)
    )
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        ledger_enabled,
    )

    width, kind = _cached_build_width(
        scheduler,
        batch,
        ledger_on=ledger_enabled(),
        eligible=eligible,
    )
    layer_ms = predict_layer_ms(scheduler, width, batch, sum(prefixes))
    usable = [
        int(k)
        for k in VERIFY_KS
        if smallest_bucket(int(max_ell) + int(k) + 1, buckets) is not None
    ]
    ready = bool(usable) and all(
        _time_for(batch, max_ell, k, buckets) is not None for k in usable
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
    plan = RoundPlan(
        width=width,
        k=chosen,
        use_tau=use_tau,
        locked_k=locked,
        theta_value=price,
        layer_ms=layer_ms,
        max_ell=max_ell,
        batch=batch,
        buckets=buckets,
        scheduler=scheduler,
    )
    plan.replay_kind = kind
    plan.replay_eligible = eligible
    return plan


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


def batch_bucket(batch_size: int) -> int:
    """Replay state is shared inside 1, 2–4, and 5 or more."""
    size = max(int(batch_size), 1)
    if size == 1:
        return 1
    if size <= 4:
        return 2
    return 5


def persistent_width(batch_size: int) -> int:
    """Width served after a gated switch, or the flat table before one."""
    bucket = batch_bucket(batch_size)
    if bucket in _persistent_w:
        return int(_persistent_w[bucket])
    return flat_width(batch_size)


def choose_build_width(batch_size: int, *, ledger_on: bool, eligible: bool) -> tuple:
    """Flat or switched width, plus one temporary departure when the ledger is on.

    A batch larger than one spends every 20th eligible round at width 8.
    A single request, already at 8, spends that round at width 4 instead.
    Neither departure happens while the ledger is off, or on a warmup, probe,
    or locked round.
    """
    base = persistent_width(batch_size)
    if not ledger_on or not eligible:
        return base, "ordinary"
    bucket = batch_bucket(batch_size)
    _eligible_n[bucket] = int(_eligible_n.get(bucket, 0)) + 1
    if _eligible_n[bucket] % REPLAY_EVERY != 0:
        return base, "ordinary"
    if bucket == 1 and base == 8:
        return 4, "measure"
    if bucket != 1 and base == 4:
        return 8, "explore"
    return base, "ordinary"


def _cached_build_width(scheduler, batch: int, *, ledger_on: bool, eligible: bool):
    """One width per packet. A fallback expand must not take a second departure."""
    token = None if scheduler is None else getattr(scheduler, "_clear_build_token", None)
    cached = None if scheduler is None else getattr(scheduler, "_clear_build_cache", None)
    if token is not None and isinstance(cached, tuple) and cached[0] == token:
        return cached[1]
    chosen = choose_build_width(batch, ledger_on=ledger_on, eligible=eligible)
    if token is not None and scheduler is not None:
        scheduler._clear_build_cache = (token, chosen)
    return chosen


def _requests_greedy(reqs) -> bool:
    """A round is greedy when every request would commit the top token.

    ``SamplingParams`` rewrites temperature 0 into temperature 1 and top_k 1.
    A missing temperature, a literal 0, or that top_k all count. Any other
    temperature leaves the flat table.
    """
    for req in reqs:
        params = getattr(req, "sampling_params", None)
        if params is None:
            continue
        top_k = getattr(params, "top_k", None)
        if top_k is not None and int(top_k) == 1:
            continue
        temperature = getattr(params, "temperature", None)
        if temperature is None or float(temperature) == 0.0:
            continue
        return False
    return True


def truncate_columns(layers, width: int) -> list:
    """Per layer, the columns kept when the tree is cut down to ``width``.

    Layer 0 keeps the highest scores. A later layer keeps the highest scores
    among children of the columns kept above it.
    """
    kept = []
    previous = None
    limit = max(int(width), 1)
    for depth, nodes in enumerate(layers):
        ranked = []
        for col, node in enumerate(nodes):
            if depth > 0 and int(node["parent"]) not in previous:
                continue
            ranked.append((float(node["score"]), -int(col), int(col)))
        ranked.sort(reverse=True)
        chosen = [col for _score, _neg, col in ranked[:limit]]
        previous = set(chosen)
        kept.append(chosen)
    return kept


def matched_depth(layers, kept, tokens) -> int:
    """How many committed tokens walk the kept columns, from the root."""
    parent = None
    matched = 0
    sequence = [int(token) for token in tokens]
    for depth, cols in enumerate(kept):
        if matched >= len(sequence):
            break
        found = None
        for col in cols:
            node = layers[depth][col]
            if depth > 0 and int(node["parent"]) != parent:
                continue
            if int(node["token"]) == sequence[matched]:
                found = int(col)
                break
        if found is None:
            break
        parent = found
        matched += 1
    return matched


def replay_accept(layers, width: int, tokens, stop: Optional[int] = None) -> int:
    """Accepted draft tokens on the truncated tree, plus one bonus when present."""
    kept = truncate_columns(layers, width)
    depth = len(layers) if stop is None else min(len(layers), max(int(stop), 0))
    matched = matched_depth(layers[:depth], kept[:depth], tokens)
    if len(list(tokens)) > matched:
        return matched + 1
    return matched


def _kept_scores(layers, kept, end: int) -> list:
    scores = []
    for depth in range(min(int(end), len(kept))):
        for col in kept[depth]:
            scores.append(float(layers[depth][col]["score"]))
    return scores


def _list_marginal(scores, latest, k: int) -> float:
    if not latest:
        return 0.0
    ordered = sorted((float(value) for value in scores), reverse=True)
    theta_k = ordered[int(k) - 1] if len(ordered) >= int(k) else 0.0
    return float(sum(max(0.0, float(score) - theta_k) for score in latest))


def candidate_stop_depth(
    layers,
    kept,
    *,
    theta,
    verify_ms: Dict[int, Optional[float]],
    layer_costs: Sequence[Optional[float]],
    ratio: float,
) -> int:
    """Built depth, or up to two layers earlier when the truncated J says stop."""
    depth = len(layers)
    if depth < 2 or theta is None:
        return depth
    earliest = max(1, depth - 2)
    for stop in range(earliest, depth):
        cost = layer_costs[stop] if stop < len(layer_costs) else None
        if cost is None:
            return depth
        if not _truncated_worth(
            layers, kept, stop, float(theta), verify_ms, float(cost), float(ratio)
        ):
            return stop
    return depth


def _truncated_worth(layers, kept, stop, theta, verify_ms, layer_cost, ratio) -> bool:
    """True when one more truncated layer is worth more than verifying now."""
    current = {}
    nxt = {}
    scores = _kept_scores(layers, kept, stop)
    latest = [
        float(layers[stop][col]["score"])
        for col in kept[stop]
    ] if stop < len(kept) else []
    for k, spent in verify_ms.items():
        if spent is None:
            continue
        mass = _prefix_mass(scores, int(k))
        current[int(k)] = float(ratio) * mass - float(theta) * float(spent)
        nxt[int(k)] = (
            float(ratio) * (mass + _list_marginal(scores, latest, int(k)))
            - float(theta) * float(spent)
        )
    if not current or not nxt:
        return True
    next_value = max(nxt.values()) - float(theta) * float(layer_cost)
    return layer_should_continue(max(current.values()), next_value)


def _schedule_ms(layer_rows, width: int, n_layers: int, verify_ms, k: int):
    spent = None if verify_ms is None else verify_ms.get(int(k))
    if spent is None:
        return None
    total = float(spent)
    for row in list(layer_rows)[: int(n_layers)]:
        pred = row.get("pred", {}).get(int(width))
        if pred is None:
            return None
        total += float(pred)
    return total


def _select_replay_k(score_rows, ratios, theta, verify_ms, default_k: int) -> int:
    if theta is None:
        return int(default_k)
    values = {}
    for k, spent in verify_ms.items():
        if spent is None:
            continue
        benefit = 0.0
        for scores, ratio in zip(score_rows, ratios):
            benefit += float(ratio) * _prefix_mass(scores, int(k))
        values[int(k)] = benefit - float(theta) * float(spent)
    if not values:
        return int(default_k)
    return select_k(values)


def _blank_arm() -> Dict[str, float]:
    return {"tok": 0.0, "ms": 0.0, "cur_tok": 0.0, "cur_ms": 0.0, "n": 0.0}


def _arm(bucket: int, width: int) -> Dict[str, float]:
    row = _replay_arms.get(bucket)
    if row is None:
        row = {}
        _replay_arms[bucket] = row
    found = row.get(int(width))
    if found is None:
        found = _blank_arm()
        row[int(width)] = found
    return found


def _clear_arms(bucket: int) -> None:
    _replay_arms.pop(int(bucket), None)
    _replay_n[int(bucket)] = 0


def replay_bias(bucket: int) -> Optional[float]:
    """Window gap between replayed width-4 throughput and real width-4 throughput.

    Both are divided by the real width-8 throughput of the same bucket. None
    until each side has 30 rounds.
    """
    key = int(bucket)
    if int(_bias_n8.get(key, 0)) < BIAS_MIN_ROUNDS:
        return None
    if int(_bias_n4.get(key, 0)) < BIAS_MIN_ROUNDS:
        return None
    ms8 = float(_bias_ms8.get(key, 0.0))
    replay_ms = float(_bias_replay_ms.get(key, 0.0))
    ms4 = float(_bias_ms4.get(key, 0.0))
    if ms8 <= 0.0 or replay_ms <= 0.0 or ms4 <= 0.0:
        return None
    base = float(_bias_tok8.get(key, 0.0)) / ms8
    if base <= 0.0:
        return None
    replay_ratio = (float(_bias_replay_tok.get(key, 0.0)) / replay_ms) / base - 1.0
    real_ratio = (float(_bias_tok4.get(key, 0.0)) / ms4) / base - 1.0
    return float(replay_ratio - real_ratio)


def bias_gate_open(bucket: int) -> bool:
    """True when the 8-versus-4 gap is known and inside the 3 percent limit."""
    gap = replay_bias(bucket)
    return gap is not None and abs(float(gap)) <= BIAS_LIMIT


def _note_bias(bucket: int, built: int, actual_tokens, actual_ms, replayed) -> None:
    if float(actual_ms) <= 0.0:
        return
    key = int(bucket)
    if int(built) == 8 and 4 in replayed and replayed[4][1] is not None:
        _bias_n8[key] = int(_bias_n8.get(key, 0)) + 1
        _bias_tok8[key] = float(_bias_tok8.get(key, 0.0)) + float(actual_tokens)
        _bias_ms8[key] = float(_bias_ms8.get(key, 0.0)) + float(actual_ms)
        _bias_replay_tok[key] = float(_bias_replay_tok.get(key, 0.0)) + float(replayed[4][0])
        _bias_replay_ms[key] = float(_bias_replay_ms.get(key, 0.0)) + float(replayed[4][1])
    elif int(built) == 4:
        _bias_n4[key] = int(_bias_n4.get(key, 0)) + 1
        _bias_tok4[key] = float(_bias_tok4.get(key, 0.0)) + float(actual_tokens)
        _bias_ms4[key] = float(_bias_ms4.get(key, 0.0)) + float(actual_ms)


def _judge(bucket: int, current: int, *, allow_apply: bool) -> tuple:
    """Best candidate under the 20-round, 3 percent, 30/10 rule. Width 2 stays a shadow."""
    best = int(current)
    best_gain = 0.0
    for cand in CANDIDATE_WIDTHS:
        if int(cand) == int(current):
            continue
        arm = _arm(bucket, cand)
        needed = NARROW_MIN_ROUNDS if int(cand) < int(current) else WIDE_MIN_ROUNDS
        if (
            arm["n"] < needed
            or arm["ms"] <= 0.0
            or arm["cur_ms"] <= 0.0
            or arm["tok"] <= 0.0
            or arm["cur_tok"] <= 0.0
        ):
            continue
        gain = (arm["tok"] / arm["ms"]) / (arm["cur_tok"] / arm["cur_ms"]) - 1.0
        if gain > best_gain and gain > REPLAY_GAIN:
            best = int(cand)
            best_gain = float(gain)
    applied = False
    if (
        allow_apply
        and best != int(current)
        and best in (4, 8)
        and int(current) in (4, 8)
        and bias_gate_open(bucket)
    ):
        _persistent_w[int(bucket)] = int(best)
        _would_w[int(bucket)] = int(best)
        _clear_arms(bucket)
        applied = True
    else:
        _would_w[int(bucket)] = int(best)
    return best, best_gain, applied


def observe_replay(
    batch: int,
    built: int,
    kind: str,
    actual_tokens: float,
    actual_ms: float,
    replayed: Optional[Dict[int, tuple]] = None,
    *,
    counted: bool = True,
    apply: bool = True,
) -> dict:
    """Accumulate one labeled round and return the shadow decision.

    ``counted`` is false for warmup, probe, and locked rounds. A measure round
    updates only the width-4 side of the bias. Width 2 can win the shadow
    comparison and still cannot change the served width.
    """
    pairs = {} if replayed is None else {int(key): value for key, value in replayed.items()}
    bucket = batch_bucket(batch)
    current = persistent_width(batch)
    if counted:
        _note_bias(bucket, int(built), actual_tokens, actual_ms, pairs)
    gain = 0.0
    applied = False
    judged_rounds = int(_replay_n.get(bucket, 0))
    if counted and kind != "measure":
        arms = {}
        if int(built) == int(current):
            arms[int(current)] = (float(actual_tokens), float(actual_ms))
            for width, pair in pairs.items():
                if int(width) < int(current) and pair[1] is not None:
                    arms[int(width)] = (float(pair[0]), float(pair[1]))
        elif int(built) > int(current):
            arms[int(built)] = (float(actual_tokens), float(actual_ms))
            for width, pair in pairs.items():
                if pair[1] is not None:
                    arms[int(width)] = (float(pair[0]), float(pair[1]))
        if int(current) in arms and float(arms[int(current)][1]) > 0.0:
            cur_tok, cur_ms = arms[int(current)]
            paired = False
            for width, pair in arms.items():
                if int(width) == int(current) or float(pair[1]) <= 0.0:
                    continue
                arm = _arm(bucket, width)
                arm["tok"] += float(pair[0])
                arm["ms"] += float(pair[1])
                arm["cur_tok"] += float(cur_tok)
                arm["cur_ms"] += float(cur_ms)
                arm["n"] += 1.0
                paired = True
            if paired:
                _replay_n[bucket] = int(_replay_n.get(bucket, 0)) + 1
                judged_rounds = int(_replay_n[bucket])
                if judged_rounds % REPLAY_EVERY == 0:
                    _best, gain, applied = _judge(bucket, current, allow_apply=apply)
                    current = persistent_width(batch)
    rounds = judged_rounds
    record = {
        "bucket": int(bucket),
        "b": int(batch),
        "w": int(built),
        "kind": str(kind),
        "candidates": _candidate_log(int(built), actual_tokens, actual_ms, pairs),
        "rounds": rounds,
        "gain": float(gain),
        "would_w": int(_would_w.get(bucket, current)),
        "bias": replay_bias(bucket),
        "gate": bias_gate_open(bucket),
        "applied": bool(applied),
    }
    return record


def _candidate_log(built, actual_tokens, actual_ms, replayed) -> dict:
    logged = {
        str(int(built)): {"tokens": float(actual_tokens), "ms": float(actual_ms)}
    }
    for width, pair in replayed.items():
        logged[str(int(width))] = {
            "tokens": float(pair[0]),
            "ms": None if pair[1] is None else float(pair[1]),
        }
    return logged


def _layers_of(state) -> list:
    depth = int(getattr(state, "depth", 0) or 0)
    if depth < 1 or not hasattr(state, "tokens"):
        return []
    width = int(state.tokens.shape[1])
    tokens = state.tokens[:depth].tolist()
    parents = state.parents[:depth].tolist()
    scores = state.weight[:depth].tolist()
    layers = []
    for row in range(depth):
        layers.append(
            [
                {
                    "token": int(tokens[row][col]),
                    "parent": int(parents[row][col]),
                    "score": float(scores[row][col]),
                }
                for col in range(width)
            ]
        )
    return layers


def _freeze_layer_rows(scheduler, states) -> list:
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        predict_layer_ms,
    )

    depth = max((int(getattr(state, "depth", 0) or 0) for state in states), default=0)
    rows = []
    for layer in range(depth):
        active_states = [
            state for state in states if int(getattr(state, "depth", 0) or 0) > layer
        ]
        pred = {}
        active = len(active_states)
        sum_ell = sum(int(getattr(state, "prefix", 0) or 0) for state in active_states)
        for width in (4, 8):
            try:
                pred[width] = predict_layer_ms(scheduler, width, active, sum_ell)
            except Exception:
                pred[width] = None
        rows.append({"active": active, "sum_ell": sum_ell, "pred": pred})
    return rows


def hold_replay(scheduler, reqs, states, plan) -> None:
    """Keep this round's trees until the next STEP brings their committed text."""
    try:
        trees = {}
        usable = []
        for req, state in zip(reqs, states):
            layers = _layers_of(state)
            if not layers:
                continue
            rid = str(getattr(req, "rid", "") or getattr(state, "rid", ""))
            output = list(getattr(req, "output_ids", None) or [])
            trees[rid] = {
                "layers": layers,
                "output_len": len(output),
                "prefix": int(getattr(state, "prefix", 0) or 0),
                "ratio": float(accept_ratio(rid)),
            }
            usable.append(state)
        if not trees:
            return
        verify = {}
        for k in VERIFY_KS:
            try:
                verify[int(k)] = plan._time(k)
            except Exception:
                verify[int(k)] = None
        if getattr(plan, "_frozen_k", None) is not None:
            k_actual = int(plan._frozen_k)
        elif plan.locked_k is not None:
            k_actual = int(plan.locked_k)
        else:
            k_actual = int(plan.verify_k(usable[0]))
        token = getattr(scheduler, "_clear_build_token", None)
        existing = getattr(scheduler, "_clear_replay_hold", None)
        if (
            existing is None
            or existing.get("token") != token
            or len(list(reqs)) > 1
        ):
            existing = {
                "token": token,
                "trees": {},
                "width": int(plan.width),
                "kind": str(getattr(plan, "replay_kind", "ordinary")),
                "eligible": bool(getattr(plan, "replay_eligible", False)),
                "batch": int(plan.batch),
                "theta": plan.theta_value,
                "verify_ms": verify,
                "k": k_actual,
                "layers": _freeze_layer_rows(scheduler, usable),
            }
        existing["trees"].update(trees)
        scheduler._clear_replay_hold = existing
    except Exception:
        import logging

        logging.getLogger(__name__).warning(
            "[CLEAR ledger] replay hold failed", exc_info=True
        )


def _replayed_counts(hold, tokens_by_rid) -> Dict[int, tuple]:
    built = int(hold["width"])
    theta = hold.get("theta")
    verify_ms = hold.get("verify_ms") or {}
    layer_rows = list(hold.get("layers") or [])
    trees = hold["trees"]
    out = {}
    for width in CANDIDATE_WIDTHS:
        if int(width) >= built:
            continue
        accepts = []
        scores = []
        ratios = []
        stops = []
        for rid, tree in trees.items():
            tokens = tokens_by_rid.get(rid)
            if tokens is None:
                return {}
            layers = tree["layers"]
            kept = truncate_columns(layers, width)
            costs = [
                row.get("pred", {}).get(int(width)) for row in layer_rows
            ]
            stop = candidate_stop_depth(
                layers,
                kept,
                theta=theta,
                verify_ms=verify_ms,
                layer_costs=costs,
                ratio=float(tree["ratio"]),
            )
            accepts.append(replay_accept(layers, width, tokens, stop=stop))
            scores.append(_kept_scores(layers, kept, stop))
            ratios.append(float(tree["ratio"]))
            stops.append(int(stop))
        k_prime = _select_replay_k(scores, ratios, theta, verify_ms, int(hold["k"]))
        model_actual = _schedule_ms(layer_rows, built, len(layer_rows), verify_ms, int(hold["k"]))
        model_cand = _schedule_ms(
            layer_rows, int(width), max(stops) if stops else 0, verify_ms, k_prime
        )
        measured = hold.get("measured_ms")
        if measured is None or model_actual is None or model_cand is None:
            out[int(width)] = (float(sum(accepts)), None)
            continue
        counterfactual = float(measured) - float(model_actual) + float(model_cand)
        if counterfactual <= 0.0:
            out[int(width)] = (float(sum(accepts)), None)
        else:
            out[int(width)] = (float(sum(accepts)), counterfactual)
    return out


def settle_replay(scheduler, reqs) -> Optional[dict]:
    """Replay the previous tree with this STEP's committed tokens."""
    hold = getattr(scheduler, "_clear_replay_hold", None)
    closed = getattr(scheduler, "_clear_replay_closed", None)
    scheduler._clear_replay_hold = None
    scheduler._clear_replay_closed = None
    if not isinstance(hold, dict) or not isinstance(closed, dict):
        return None
    try:
        if not hold.get("eligible"):
            return None
        if closed.get("excluded") or not closed.get("price_kept"):
            return None
        measured = closed.get("price_ms")
        if measured is None or float(measured) <= 0.0:
            return None
        by_rid = {str(getattr(req, "rid", "")): req for req in reqs}
        tokens_by_rid = {}
        actual = 0
        for rid, tree in hold["trees"].items():
            req = by_rid.get(rid)
            if req is None:
                return None
            output = [int(token) for token in list(getattr(req, "output_ids", None) or [])]
            old = int(tree["output_len"])
            if old > len(output):
                return None
            fresh = output[old:]
            tokens_by_rid[rid] = fresh
            actual += len(fresh)
        if not tokens_by_rid:
            return None
        hold["measured_ms"] = float(measured)
        replayed = _replayed_counts(hold, tokens_by_rid)
        size = int(getattr(scheduler, "tp_size", 1) or 1)
        record = observe_replay(
            int(hold["batch"]),
            int(hold["width"]),
            str(hold["kind"]),
            float(actual),
            float(measured),
            replayed,
            counted=True,
            apply=size <= 1,
        )
        _emit_replay(scheduler, record)
        return record
    except Exception:
        import logging

        logging.getLogger(__name__).warning(
            "[CLEAR ledger] replay settle failed", exc_info=True
        )
        return None


def _emit_replay(scheduler, record: dict) -> None:
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        _emit,
        ledger_enabled,
    )

    if not ledger_enabled():
        return
    rank = 0 if scheduler is None else int(getattr(scheduler, "tp_rank", 0) or 0)
    if rank != 0:
        return
    _emit("d_replay", record)
