"""Flat layer width and online verify count for one ClearTree round.

W is a fixed table: batch size 1 uses 8, every other batch uses 4. K is
chosen among 8, 16, and 32. Target graphs are captured at K + 1, so a short
tree is padded only inside the graph of the K that was chosen.

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


def reset_policy() -> None:
    """Drop prices, accept ratios, probe counts, and verify samples."""
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
    width = flat_width(batch)
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
        scheduler=scheduler,
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
