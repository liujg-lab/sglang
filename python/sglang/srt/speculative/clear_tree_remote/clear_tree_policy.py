"""Flat layer width and online verify count for one ClearTree round.

W and the tau verify count start from a fixed table. Batch size 1 uses width
8 and K 16. Every other batch uses width 4 and K 8. Eligible rounds first
build the next narrower width when that width has not run for 300 rounds.
That burst lasts until the width has four layer samples, at most four
rounds, and two rounds once the fit exists. A narrow width that already has
four samples but cannot price this round's prefix sum is built for one
round after 20 eligible rounds. Every 20th eligible round that is not one
of those builds the next wider candidate. A shadow replay may replace the
served width when that candidate's gain clears 3 percent. Either replacement
needs 30 paired rounds, and a narrower one is only the next smaller
candidate. Measured bias is logged and does not gate the switch. Online K
stays inside the tiers that can hold the tree and that already have a
sample. The default candidates are 2, 4, 8, 16, 32, and 64. Target graphs
are captured at K + 1, so a short tree is padded only inside the graph of
the K that was chosen.

K-probe rounds, and rounds whose flat tier or layer sample is still short,
keep the tau stop and the flat K. A K-probe round locks the probed K.
Probes start only after that flat tier has a sample, skip the tier, and
take the nearest other tier first. Each of those tiers is probed until it
has two samples. The flat tier above the cover is replaced by the largest
tier inside it. Past ``max(2 * K tiers, rounds / 20)`` probes, the round
keeps choosing among tiers that already have a sample. A cell whose flat
tier still lacks a sample stays on tau and does not probe. Ordinary rounds
choose K only inside the covering tier of the width built for that round.
Other rounds stop by comparing one J across verify counts at or below that
tier: keep the prefix of requests whose next layer is worth more than
verifying now. The final K is also at or below the tier that covers the nodes
actually built. Every request in the round, including one that already
stopped growing, stays in that J. Verify time stays on the full batch. The
layer cost is only the requests that would be forwarded.
"""

from __future__ import annotations

import logging
import math
import os
from collections import deque
from typing import Callable, Dict, Optional, Sequence, Union

logger = logging.getLogger(__name__)

from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import policy_timed
from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
    BONUS_NTPBS,
    CANDIDATE_WIDTHS,
    DRAFT_CAPACITY,
    LAYER_WIDTHS,
    TARGET_CAPACITY,
    VERIFY_KS,
    batch_bucket,
    bonus_ntpbs,
    capture_ntpb,
    _next_wider,
    flat_k,
    flat_width,
    persistent_width,
)

EXPLORE_EVERY = 20
PROBE_RUNS = 2
VERIFY_WINDOW = 16
CALIBRATE_LAMBDA = 3.0
PRICE_EMA_FLOOR = 0.05
PRICE_PRIOR_N0 = 10.0

_clock: Dict[tuple, list] = {}
_curve_cache: Dict[tuple, dict] = {}
_q_raw: Dict[tuple, deque] = {}
Q_RIDGE = (0.0, 1.0)
Q_RES_WINDOW = 32
_price_n: Dict[int, int] = {}
_price_tok: Dict[int, float] = {}
_price_ms: Dict[int, float] = {}
_sum_a: Dict[str, float] = {}
_sum_e: Dict[str, float] = {}
_total_a = 0.0
_total_e = 0.0
_pending: Dict[str, dict] = {}
_rounds = 0
_probe_rounds = 0
_bucket_warned = False
_gain_pred = 0.0
_gain_actual = 0.0


def on_external_event(scheduler, batch=None) -> None:
    """Target session advanced.

    The same Target header keeps H, Q, and the layer fit. Anything else
    clears those fits and then tries the file for the new header. Neither
    path writes the file. A missing batch keeps the full clear.
    """
    global _probe_rounds
    from sglang.srt.speculative.clear_tree_remote.clear_tree_fit import (
        on_target_info,
        target_headers_match,
    )
    from sglang.srt.speculative.clear_tree_remote.clear_tree_replay import (
        reset_replay_state,
    )

    info = getattr(batch, "target_info", None) if batch is not None else None
    current = getattr(scheduler, "_clear_target_header", None)
    ledger = getattr(scheduler, "_clear_ledger", None)
    if target_headers_match(info, current):
        _pending.clear()
        reset_replay_state()
        if ledger:
            ledger.drop_open()
        scheduler._clear_replay_to_hold = []
        if isinstance(info, dict):
            on_target_info(scheduler, info)
        return
    _clock.clear()
    _curve_cache.clear()
    _q_raw.clear()
    _probe_rounds = 0
    _pending.clear()
    reset_replay_state()
    if ledger:
        ledger.drop_fitted()
        ledger.split_capable = False
    scheduler._clear_replay_to_hold = []
    scheduler._clear_fit_installed = False
    scheduler._clear_prefit = None
    if isinstance(info, dict):
        on_target_info(scheduler, info)


def reset_gain_calibration() -> None:
    """Drop the next-layer gain ratio. An external event does not call this."""
    global _gain_pred, _gain_actual
    _gain_pred = 0.0
    _gain_actual = 0.0


def reset_policy(keep_fits: bool = False) -> None:
    """Drop prices, accept ratios, probe counts, and replay.

    ``keep_fits`` leaves the H table, the Q samples, and the layer regressions.
    """
    global _total_a, _total_e, _rounds, _probe_rounds
    global _bucket_warned
    if not keep_fits:
        _clock.clear()
        _q_raw.clear()
    _curve_cache.clear()
    _price_n.clear()
    _price_tok.clear()
    _price_ms.clear()
    _sum_a.clear()
    _sum_e.clear()
    _pending.clear()
    _bucket_warned = False
    reset_gain_calibration()
    from sglang.srt.speculative.clear_tree_remote.clear_tree_replay import (
        reset_replay_state,
    )
    from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
        reset_persistent_widths,
    )

    reset_persistent_widths()
    reset_replay_state()
    _total_a = 0.0
    _total_e = 0.0
    _rounds = 0
    _probe_rounds = 0


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


def keep_by_net_benefit(
    requests,
    *,
    theta: float,
    verify_times: Dict[int, float],
    layer_cost,
    kappa: Optional[Union[float, Callable]],
) -> Optional[list]:
    """Requests to forward, or None when no candidate has a layer cost.

    Each request maps ``e`` and ``g`` from K to path mass and marginal gain,
    plus ``ell``, ``c``, ``item``, and ``growing``. ``J`` uses every request.
    The layer cost is charged only for the ``m`` requests that would continue.
    ``kappa`` is a prefix slope, or a function of ``m``. Each distinct slope
    is sorted once, and ``m`` uses that slope's own prefix. An empty list
    means stop. Ties keep the larger K, then the larger group.
    """
    rows = list(requests)
    growing = [row for row in rows if row.get("growing", True)]
    if not growing:
        return []
    price = float(theta)
    slope_of = (
        kappa
        if callable(kappa)
        else (lambda _m: 0.0 if kappa is None else float(kappa))
    )
    joint = {}
    gains = {}
    for k, spent in verify_times.items():
        if spent is None or int(k) not in VERIFY_KS:
            continue
        benefit = 0.0
        per_row = {}
        for row in rows:
            ratio = float(row["c"])
            benefit += ratio * float(row["e"].get(int(k), 0.0))
            per_row[id(row)] = ratio * float(row["g"].get(int(k), 0.0))
        joint[int(k)] = benefit - price * float(spent)
        gains[int(k)] = per_row
    if not joint:
        return None
    stop_value = max(joint.values())
    best = None
    any_cost = False
    for k, base in joint.items():
        orders = {}
        for m in range(1, len(growing) + 1):
            slope = slope_of(m)
            if slope is None:
                slope = 0.0
            slope = float(slope)
            ranked = orders.get(slope)
            if ranked is None:
                ranked = []
                for row in growing:
                    gain = float(gains[k].get(id(row), 0.0))
                    ell = int(row["ell"])
                    ranked.append(
                        (gain - price * slope * ell, gain, ell, row["item"])
                    )
                ranked.sort(key=lambda item: item[0], reverse=True)
                orders[slope] = ranked
            chosen = ranked[:m]
            running_gain = sum(item[1] for item in chosen)
            running_ell = sum(item[2] for item in chosen)
            cost = layer_cost(m, running_ell)
            if cost is None:
                continue
            any_cost = True
            value = base + running_gain - price * float(cost)
            choice = (value, int(k), int(m))
            if best is None or choice > best[0]:
                best = (choice, [item[3] for item in chosen])
    if not any_cost or best is None:
        return None
    if best[0][0] > stop_value:
        return list(best[1])
    return []


def select_continuing(
    all_states,
    eligible,
    *,
    theta: float,
    verify_times: Dict[int, float],
    layer_cost,
    kappa: Optional[Union[float, Callable]],
) -> Optional[list]:
    """Requests to forward, or None when no candidate has a layer cost.

    Path mass at a tier is a prefix of the benefit order. When ``kappa``
    depends on how many requests continue, each count uses its own order.
    Tiers with no verify time are left out.
    """
    growing_ids = {id(state) for state in eligible}
    timed = []
    for key, spent in verify_times.items():
        if spent is None or int(key) not in VERIFY_KS:
            continue
        timed.append(int(key))
    ratio = gain_ratio()
    requests = []
    for state in list(all_states):
        scores = _path_scores(state)
        ordered = sorted(scores, reverse=True)
        prefix = []
        running = 0.0
        for value in ordered:
            running += float(value)
            prefix.append(running)
        latest = _latest_scores(state)
        mass = {}
        gain = {}
        count = len(ordered)
        for k in timed:
            if count == 0:
                mass[k] = 0.0
                gain[k] = 0.0
                continue
            mass[k] = prefix[min(k, count) - 1]
            theta_k = ordered[k - 1] if count >= k else 0.0
            raw = sum(max(0.0, score - theta_k) for score in latest)
            gain[k] = raw * ratio
        requests.append(
            {
                "e": mass,
                "g": gain,
                "ell": int(getattr(state, "prefix", 0) or 0),
                "c": accept_ratio(str(getattr(state, "rid", ""))),
                "item": state,
                "growing": id(state) in growing_ids,
            }
        )
    return keep_by_net_benefit(
        requests,
        theta=theta,
        verify_times=verify_times,
        layer_cost=layer_cost,
        kappa=kappa,
    )


def ingest_verify_samples(samples) -> None:
    """Parse samples carried on one ClearTree request.

    ``device_us`` is microseconds and is only logged. The verify table is
    filled from the Draft ledger's ``T_K``.
    """
    for sample in list(samples or []):
        if not isinstance(sample, (list, tuple)) or len(sample) != 4:
            continue
        batch, k, bucket, device_us = sample
        _warn_foreign_bucket(int(bucket))
        if device_us is None:
            continue
        logger.debug(
            "[CLEAR] verify_samples device_us b=%s k=%s bucket=%s device_us=%s",
            int(batch),
            int(k),
            int(bucket),
            int(device_us),
        )


def locked_k_from_env() -> Optional[int]:
    """Fixed verify count for a baseline run. Unset means the online rule."""
    raw = os.environ.get("SGLANG_CLEAR_LOCK_K", "").strip()
    if not raw:
        return None
    return int(raw)


def observe_verify(batch: int, k: int, bucket: Optional[int], ms: float) -> None:
    """One H sample. Eager and over-bucket rows are not recorded.

    Each cell keeps the latest ``VERIFY_WINDOW`` samples. One sample is enough
    to use that cell. The name stays ``observe_verify`` so the Target device
    fit can keep calling it; that process does not choose K.
    """
    if int(k) not in VERIFY_KS or bucket is None or ms is None:
        return
    key = (int(batch), int(k), int(bucket))
    rows = _clock.setdefault(key, [])
    rows.append(float(ms))
    if len(rows) > VERIFY_WINDOW:
        del rows[:-VERIFY_WINDOW]
    _curve_cache.pop((int(batch), int(bucket)), None)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    mid = count // 2
    if count % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _steady_verify_samples(rows: Sequence[float]) -> list:
    """Drop samples above 1.5 times the median, then average the rest.

    Two samples use the smaller one as the median, so one spike cannot hide
    inside the comparison. A single sample is kept as it is.
    """
    kept = [float(value) for value in rows]
    if len(kept) < 2:
        return kept
    center = min(kept) if len(kept) == 2 else _median(kept)
    filtered = [value for value in kept if value <= center * 1.5]
    return filtered or kept


def _pava(points: Sequence[tuple]) -> Dict[int, float]:
    """Weighted isotonic regression, non-decreasing in K.

    Each point is ``(k, value, weight)``. Adjacent violators merge into the
    weighted mean of the block.
    """
    stack = []
    for k, value, weight in points:
        stack.append([[int(k)], float(value), float(weight)])
        while len(stack) >= 2 and stack[-2][1] > stack[-1][1] + 1e-12:
            right = stack.pop()
            left = stack.pop()
            total = left[2] + right[2]
            if total <= 0:
                merged = right[1]
                total = 1.0
            else:
                merged = (left[1] * left[2] + right[1] * right[2]) / total
            stack.append([left[0] + right[0], merged, total])
    fitted = {}
    for ks, value, _weight in stack:
        for key in ks:
            fitted[int(key)] = float(value)
    return fitted


def _hold_curve(batch: int, bucket: int) -> Dict[int, float]:
    """Robust H means for one (batch, bucket), forced non-decreasing in K."""
    key = (int(batch), int(bucket))
    cached = _curve_cache.get(key)
    if cached is not None:
        return cached
    points = []
    for k in VERIFY_KS:
        rows = _clock.get((int(batch), int(k), int(bucket)))
        if not rows:
            continue
        kept = _steady_verify_samples(rows)
        if not kept:
            continue
        points.append((int(k), sum(kept) / float(len(kept)), float(len(kept))))
    fitted = _pava(points)
    _curve_cache[key] = fitted
    return fitted


def predict_hold_ms(batch: int, k: int, bucket: Optional[int]) -> Optional[float]:
    """Isotonic H for this cell, or None when the cell has no sample."""
    if bucket is None:
        return None
    return _hold_curve(int(batch), int(bucket)).get(int(k))


def _fit_q_line(points: Sequence[tuple]) -> tuple:
    """Intercept and slope for kept ``(k, ms)`` points. Ridge is ``Q_RIDGE``."""
    count = float(len(points))
    sum_k = 0.0
    sum_k2 = 0.0
    sum_y = 0.0
    sum_ky = 0.0
    for k, ms in points:
        value = float(k)
        measured = float(ms)
        sum_k += value
        sum_k2 += value * value
        sum_y += measured
        sum_ky += value * measured
    det = (count + Q_RIDGE[0]) * (sum_k2 + Q_RIDGE[1]) - sum_k * sum_k
    if abs(det) < 1e-12:
        return 0.0, 0.0
    intercept = ((sum_y * (sum_k2 + Q_RIDGE[1])) - (sum_k * sum_ky)) / det
    slope = (((count + Q_RIDGE[0]) * sum_ky) - (sum_k * sum_y)) / det
    return float(intercept), float(slope)


def _q_kept(batch: int) -> list:
    """Raw Q samples for one batch, filtered inside each K."""
    kept = []
    for k in VERIFY_KS:
        rows = _q_raw.get((int(batch), int(k)))
        if not rows:
            continue
        for ms in _steady_verify_samples(rows):
            kept.append((int(k), float(ms)))
    return kept


def observe_q(batch: int, k: int, ms: float) -> None:
    """Keep one raw Q sample for this ``(batch, K)``.

    Prediction filters inside that K, then fits the batch line. A rejected
    sample stays in the raw window and is left out of the fit.
    """
    if int(k) not in VERIFY_KS or ms is None:
        return
    key = (int(batch), int(k))
    window = _q_raw.setdefault(key, deque(maxlen=Q_RES_WINDOW))
    window.append(float(ms))


def predict_q_ms(batch: int, k: int) -> float:
    """Median residual plus the current K slope. An empty batch predicts 0."""
    kept = _q_kept(int(batch))
    if not kept:
        return 0.0
    _intercept, slope = _fit_q_line(kept)
    residuals = [ms - slope * float(item_k) for item_k, ms in kept]
    return float(_median(residuals) + slope * int(k))


def predict_verify_ms(batch: int, k: int, bucket: Optional[int]) -> Optional[float]:
    """H plus Q for this cell.

    A tier is usable only when its own ``(batch, K, bucket)`` H cell has a
    sample. An empty Q fit adds nothing, so a constant network delay that
    has not been seen yet does not change the ranking.
    """
    hold = predict_hold_ms(batch, k, bucket)
    if hold is None:
        return None
    return float(hold) + predict_q_ms(batch, k)


def replace_hold_cells(rows) -> None:
    """Install saved H samples. Later ``observe_verify`` calls extend them."""
    _clock.clear()
    _curve_cache.clear()
    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        samples = row.get("samples_ms")
        if not isinstance(samples, list) or not samples:
            continue
        key = (int(row["b"]), int(row["k"]), int(row["bucket"]))
        kept = [float(value) for value in samples][-VERIFY_WINDOW:]
        _clock[key] = kept


def export_hold_cells() -> list:
    """H cells in stable order, for the one prefit write."""
    rows = []
    for key in sorted(_clock):
        samples = _clock.get(key) or []
        if not samples:
            continue
        batch, k, bucket = key
        rows.append(
            {
                "b": int(batch),
                "k": int(k),
                "bucket": int(bucket),
                "samples_ms": [float(value) for value in samples],
            }
        )
    return rows


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
    """log Θ = α + β·log b from the other batch sizes. One neighbor scales by b/b′."""
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
        return other_theta * (float(batch) / float(other_b))
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
            _observe_price_into(
                dropped_n, dropped_tok, dropped_ms, batch, tokens, spent
            )
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


def note_verified(
    rid: str,
    prefix: int,
    expected: float,
    depth: Optional[int] = None,
) -> None:
    """Remember this round until the next packet reports how many tokens committed.

    ``depth`` is the written tree. Without it, a later prefix jump is not a
    target gap.
    """
    item = {"prefix": int(prefix), "e": float(expected)}
    if depth is not None:
        item["depth"] = int(depth)
    _pending[str(rid)] = item


def target_gap_open(rids: Sequence[str], prefixes: Sequence[int]) -> bool:
    """True when one noted request grew by more than its tree depth plus one."""
    found = {str(rid): int(prefix) for rid, prefix in zip(rids, prefixes)}
    for rid, item in _pending.items():
        if rid not in found or "depth" not in item:
            continue
        growth = int(found[rid]) - int(item["prefix"])
        if growth > int(item["depth"]) + 1:
            return True
    return False


def discard_pending() -> None:
    """Drop unsettled accept rows without writing them into c_i."""
    _pending.clear()


def settle_verified(rids: Sequence[str], prefixes: Sequence[int]) -> None:
    """Turn the pending path mass into an accept count. Missing requests drop."""
    found = {str(rid): int(prefix) for rid, prefix in zip(rids, prefixes)}
    for rid in list(_pending):
        item = _pending.pop(rid)
        if rid not in found:
            continue
        delta = int(found[rid]) - int(item["prefix"])
        observe_accept(rid, max(delta - 1, 0), float(item["e"]))


def forget_request(rid: str) -> None:
    """Drop one request's accept totals. The pooled shrink target stays."""
    key = str(rid)
    _sum_a.pop(key, None)
    _sum_e.pop(key, None)
    _pending.pop(key, None)


def _warn_foreign_bucket(bucket: int) -> None:
    """One warning when a sample bucket is missing from the Draft list."""
    global _bucket_warned
    if _bucket_warned:
        return
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        runtime_kv_buckets,
    )

    if int(bucket) in {int(item) for item in runtime_kv_buckets()}:
        return
    _bucket_warned = True
    logger.warning(
        "[CLEAR] verify sample bucket %s is not in the Draft KV bucket list",
        int(bucket),
    )


def _tier_for_nodes(need: int, fitting: Sequence[int]) -> int:
    """Smallest tier that can hold ``need`` nodes, else the largest tier."""
    tiers = [int(k) for k in fitting]
    if not tiers:
        tiers = [int(k) for k in VERIFY_KS]
    enough = [k for k in tiers if k >= int(need)]
    if enough:
        return min(enough)
    return max(tiers)


def covering_k(width: int, max_depth: int, fitting: Sequence[int]) -> int:
    """Smallest tier that holds one full tree, else the largest fitting tier.

    A full tree has ``width * (max_depth - 1)`` nodes. The value rule does not
    look past this tier.
    """
    need = int(width) * max(int(max_depth) - 1, 0)
    return _tier_for_nodes(need, fitting)


def _round_max_depth(scheduler) -> Optional[int]:
    args = getattr(scheduler, "server_args", None)
    depth = getattr(args, "clear_tree_max_depth", None)
    if depth is None:
        return None
    return int(depth)


def _cover_for(width: int, max_depth: Optional[int], fitting: Sequence[int]) -> int:
    """Covering tier, or the largest fitting tier when depth is unknown."""
    tiers = [int(k) for k in fitting] or [int(k) for k in VERIFY_KS]
    if max_depth is None:
        return max(tiers)
    return covering_k(int(width), int(max_depth), tiers)


def _flat_locked_k(batch: int, max_ell: int, buckets: Sequence[int]) -> int:
    """Flat K when its bucket fits, else the largest K that fits."""
    preferred = int(flat_k(batch))
    fitting = [int(k) for k in fitting_ks(max_ell, buckets)]
    if preferred in fitting:
        return preferred
    return max(fitting)


def _covered_flat_k(
    batch: int, max_ell: int, buckets: Sequence[int], cover: int
) -> int:
    """Flat K inside ``cover``, else the largest fitting tier at or below it."""
    preferred = _flat_locked_k(batch, max_ell, buckets)
    fitting = [int(k) for k in fitting_ks(max_ell, buckets) if int(k) <= int(cover)]
    if int(preferred) in fitting:
        return int(preferred)
    if fitting:
        return max(fitting)
    return int(preferred)


def _probe_limit() -> int:
    """Probe rounds allowed so far: two per K tier, else one in EXPLORE_EVERY."""
    return max(PROBE_RUNS * len(VERIFY_KS), _rounds // EXPLORE_EVERY)


def _probe_rank(flat: int, k: int) -> tuple:
    """Log2 distance from the flat tier. A tie keeps the smaller K."""
    if int(flat) <= 0 or int(k) <= 0:
        return (0, int(k))
    return (abs(math.log2(float(k)) - math.log2(float(flat))), int(k))


def _take_probe(
    max_ell: int,
    buckets: Sequence[int],
    batch: int,
    cover: int,
    flat: int,
) -> Optional[int]:
    """Send the nearest K whose own cell still has fewer than two samples.

    The flat tier is not probed. Tiers above ``cover`` are not probed. A
    round that was excluded writes no sample, so the same cell is eligible
    again on the next round.
    """
    global _probe_rounds
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        smallest_bucket,
    )

    ranked = sorted(
        (int(k) for k in VERIFY_KS if int(k) <= int(cover) and int(k) != int(flat)),
        key=lambda k: _probe_rank(flat, k),
    )
    for k in ranked:
        bucket = smallest_bucket(int(max_ell) + int(k) + 1, buckets)
        if bucket is None:
            continue
        key = (int(batch), int(k), int(bucket))
        if len(_clock.get(key, ())) >= PROBE_RUNS:
            continue
        _probe_rounds += 1
        return int(k)
    return None


def fitting_ks(max_ell: int, buckets: Sequence[int]) -> list:
    """Captured K whose KV bucket exists. An empty result keeps the smallest K."""
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
        self.cover_k = int(max(VERIFY_KS))
        self.fixed_depth = None
        self.prefit = None
        self._peers = []
        self._all = []
        self._at = None
        self._keep_ids = set()
        self._raw_gain: Dict[int, float] = {}
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

    @policy_timed
    def begin_layer(self, states) -> None:
        """The layer was just written. Judge these trees together."""
        self._peers = list(states)
        self._at = None
        self._keep_ids = set()
        self._note_layer_gain(states)

    def _note_layer_gain(self, states) -> None:
        """Compare the previous flat-K upper bound with this layer's mass.

        The stored prediction is the raw upper bound, before the ratio.
        Depth 1 only stores that bound for the next layer.
        """
        from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
            FORGET,
        )

        k = int(flat_k(self.batch))
        for state in list(states or []):
            depth = int(getattr(state, "depth", 0) or 0)
            key = id(state)
            if depth >= 2 and key in self._raw_gain:
                _observe_gain(self._raw_gain[key], _quality_increment(state, k), FORGET)
            self._raw_gain[key] = _raw_marginal(state, k)

    @policy_timed
    def should_grow(self, state) -> bool:
        from sglang.srt.speculative.clear_tree_remote.clear_tree_ops import depth_rule

        if self.fixed_depth is not None:
            cap = min(int(self.fixed_depth), int(state.config.max_depth) - 1)
            return int(state.depth) < cap
        if int(state.depth) >= int(state.config.max_depth) - 1:
            return False
        if self.use_tau or self.theta_value is None or self.layer_ms is None:
            keep = int(self.k) if self.use_tau else None
            return depth_rule(state, keep)
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

    @policy_timed
    def verify_k(self, state) -> int:
        if self._frozen_k is not None:
            return int(self._frozen_k)
        if self.locked_k is not None:
            return int(self.locked_k)
        if self.use_tau:
            return int(self.k)
        return self._select(self._all or self._peers or [state])

    @policy_timed
    def freeze_k(self, states) -> int:
        """One verify count for every tree, chosen after growth has stopped."""
        chosen = self._select(list(states))
        self._frozen_k = int(chosen)
        return int(self._frozen_k)

    def note(self, state) -> None:
        rid = getattr(state, "rid", None)
        if rid is None:
            return
        stored = getattr(self._scheduler, "_clear_target_prefix", None) or {}
        key = str(rid)
        if key in stored:
            prefix = int(stored[key])
        elif getattr(state, "prefix", None) is not None:
            prefix = int(state.prefix)
        else:
            return
        scores = _path_scores(state)
        note_verified(
            key,
            prefix,
            _prefix_mass(scores, self.verify_k(state)),
            int(state.depth),
        )

    def _choose_keep(self, eligible) -> list:
        times = {}
        for k in VERIFY_KS:
            if int(k) > int(self.cover_k):
                continue
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

        return predict_layer_ms(self._scheduler, self.width, int(active), int(sum_ell))

    def _kappa(self):
        if self._scheduler is None:
            return None
        from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
            layer_sum_ell_slope,
        )

        scheduler = self._scheduler
        width = self.width

        def slope(active: int):
            return layer_sum_ell_slope(scheduler, width, int(active))

        return slope

    def _time(self, k: int) -> Optional[float]:
        from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
            smallest_bucket,
        )

        bucket = smallest_bucket(self.max_ell + int(k) + 1, self.buckets)
        return predict_verify_ms(self.batch, int(k), bucket)

    def _actual_cover(self, states) -> int:
        """Smallest tier that holds the largest tree built in this round."""
        nodes = 0
        for state in states:
            depth = int(getattr(state, "depth", 0) or 0)
            cfg = getattr(state, "config", None)
            width = int(getattr(cfg, "width", self.width) or self.width)
            nodes = max(nodes, depth * width)
        fitting = [int(k) for k in VERIFY_KS if int(k) <= int(self.cover_k)]
        if nodes <= 0 or not fitting:
            return int(self.cover_k)
        return _tier_for_nodes(nodes, fitting)

    def _select(self, states) -> int:
        values = self._values(
            states, include_gain=False, limit=self._actual_cover(states)
        )
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

    def _values(
        self, states, *, include_gain: bool, limit: Optional[int] = None
    ) -> Dict[int, float]:
        if self.theta_value is None:
            return {}
        cap = int(self.cover_k if limit is None else limit)
        values = {}
        for k in VERIFY_KS:
            if int(k) > cap:
                continue
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


def _draft_prefix(req) -> int:
    """Draft KV length. Layer cost uses this, not the Target prefix."""
    return len(getattr(req, "origin_input_ids", None) or []) + len(
        getattr(req, "output_ids", None) or []
    )


def _plan_prefix(scheduler, req, draft_len: int) -> int:
    """Target committed length when this packet stored one, else the Draft length."""
    stored = getattr(scheduler, "_clear_target_prefix", None) or {}
    rid = str(getattr(req, "rid", ""))
    if rid in stored:
        return int(stored[rid])
    return int(draft_len)


def make_round(scheduler, reqs) -> RoundPlan:
    """Choose W and the probe or value rule for these requests."""
    global _rounds
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        LAYER_MIN_SAMPLES,
        layer_sample_count,
        ledger_enabled,
        predict_layer_ms,
        runtime_kv_buckets,
    )

    reqs = list(reqs)
    batch = max(len(reqs), 1)
    drafts = [_draft_prefix(req) for req in reqs]
    prefixes = [_plan_prefix(scheduler, req, draft) for req, draft in zip(reqs, drafts)]
    max_ell = max(prefixes) if prefixes else 0
    buckets = runtime_kv_buckets()
    _rounds += 1
    planned = _prefit_round(scheduler, reqs, max_ell, buckets, batch)
    if planned is not None:
        return planned
    greedy = _requests_greedy(reqs)
    ledger_on = ledger_enabled()
    locked_env = locked_k_from_env()
    fitting = fitting_ks(max_ell, buckets)
    max_depth = _round_max_depth(scheduler)
    # A K-probe round is not eligible, so its tree stays on the served width.
    # Only the served width's covering tier bounds K probes. A superset build
    # needs the wider width's own layer sample and flat-tier verify time.
    served = persistent_width(batch)
    probe_cover = _cover_for(served, max_depth, fitting)
    served_flat = _covered_flat_k(
        batch, max_ell, buckets, _cover_for(served, max_depth, fitting)
    )
    flat_ready = _time_for(batch, max_ell, served_flat, buckets) is not None
    probe = None
    if (
        locked_env is None
        and ledger_on
        and greedy
        and flat_ready
        and _probe_rounds < _probe_limit()
    ):
        probe = _take_probe(max_ell, buckets, batch, probe_cover, served_flat)
    price = theta(batch)
    eligible = probe is None and locked_env is None and greedy and ledger_on
    from sglang.srt.speculative.clear_tree_remote.clear_tree_replay import (
        _cached_build_width,
    )

    sum_ell = sum(drafts)

    def superset_ready(candidate):
        if layer_sample_count(scheduler, int(candidate), batch) < LAYER_MIN_SAMPLES:
            return True
        if price is None:
            return False
        if predict_layer_ms(scheduler, int(candidate), batch, sum_ell) is None:
            return False
        cover_k = _cover_for(int(candidate), max_depth, fitting)
        flat = _covered_flat_k(batch, max_ell, buckets, cover_k)
        return _time_for(batch, max_ell, flat, buckets) is not None

    width, kind = _cached_build_width(
        scheduler,
        batch,
        ledger_on=ledger_on,
        eligible=eligible,
        sum_ell=sum_ell,
        ready=superset_ready,
    )
    layer_ms = predict_layer_ms(scheduler, width, batch, sum(drafts))
    cover = _cover_for(width, max_depth, fitting)
    flat_tier = _covered_flat_k(batch, max_ell, buckets, cover)
    ready = _time_for(batch, max_ell, flat_tier, buckets) is not None
    use_tau = (
        locked_env is not None
        or not ledger_on
        or not greedy
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
        chosen = int(flat_tier)
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
    plan.cover_k = int(cover)
    plan.flat_k = int(flat_tier)
    plan.replay_kind = kind
    plan.replay_eligible = bool(eligible and not use_tau and kind != "probe_w")
    plan.fallback = False
    return plan


def _prefit_round(scheduler, reqs, max_ell, buckets, batch) -> Optional[RoundPlan]:
    """A locked prefit plan, or None when this packet is ordinary traffic."""
    from sglang.srt.speculative.clear_tree_remote.clear_tree_fit import prefit_plan

    info = prefit_plan(scheduler, reqs, batch, max_ell, buckets)
    if not info:
        return None
    use_tau = bool(info.get("use_tau"))
    if use_tau:
        width = int(info.get("width") or flat_width(batch))
        chosen = int(info.get("k") or flat_k(batch))
        depth = None
    else:
        width = int(info["width"])
        chosen = int(info["k"])
        depth = info.get("depth")
    plan = RoundPlan(
        width=width,
        k=chosen,
        use_tau=use_tau,
        locked_k=chosen,
        theta_value=theta(batch),
        layer_ms=None,
        max_ell=max_ell,
        batch=batch,
        buckets=buckets,
        scheduler=scheduler,
    )
    plan.cover_k = int(max(VERIFY_KS))
    plan.flat_k = int(flat_k(batch))
    plan.fixed_depth = None if depth is None else int(depth)
    plan.prefit = info
    plan.replay_kind = "prefit"
    plan.replay_eligible = False
    plan.fallback = False
    return plan


def make_fallback_round(scheduler, reqs) -> RoundPlan:
    """Tau and the flat K for the whole packet. No extra round or probe."""
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        runtime_kv_buckets,
    )
    from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
        persistent_width,
    )

    reqs = list(reqs)
    batch = max(len(reqs), 1)
    drafts = [_draft_prefix(req) for req in reqs]
    prefixes = [_plan_prefix(scheduler, req, draft) for req, draft in zip(reqs, drafts)]
    max_ell = max(prefixes) if prefixes else 0
    buckets = runtime_kv_buckets()
    chosen = _flat_locked_k(batch, max_ell, buckets)
    plan = RoundPlan(
        width=persistent_width(batch),
        k=chosen,
        use_tau=True,
        locked_k=chosen,
        theta_value=theta(batch),
        layer_ms=None,
        max_ell=max_ell,
        batch=batch,
        buckets=buckets,
        scheduler=scheduler,
    )
    plan.cover_k = _cover_for(
        plan.width, _round_max_depth(scheduler), fitting_ks(max_ell, buckets)
    )
    plan.flat_k = _covered_flat_k(batch, max_ell, buckets, plan.cover_k)
    plan.replay_kind = "ordinary"
    plan.replay_eligible = False
    plan.fallback = True
    return plan


def _bind_round(scheduler, plan, reqs) -> None:
    """Publish the plan. Prefixes stored here stay on the Draft length."""
    scheduler._clear_round_plan = plan
    scheduler._clear_round_rids = [getattr(req, "rid", "") for req in reqs]
    scheduler._clear_round_prefixes = [_draft_prefix(req) for req in reqs]
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import bind_ledger

    ledger = bind_ledger(scheduler)
    if ledger is not None and ledger.open is not None:
        ledger.open.prefit = getattr(plan, "prefit", None)


@policy_timed
def install_fallback_plan(scheduler, reqs) -> RoundPlan:
    """One degraded plan for every per-request rebuild of this packet."""
    token = getattr(scheduler, "_clear_build_token", None)
    plan = make_fallback_round(scheduler, reqs)
    scheduler._clear_plan_fallback = True
    scheduler._clear_fallback_plan = (token, plan)
    _bind_round(scheduler, plan, reqs)
    _mark_fallback(scheduler)
    return plan


@policy_timed
def attach_round(scheduler, reqs) -> RoundPlan:
    """Store the round plan where the grow loop can see it.

    The second attach for one ``_clear_build_token`` is the batch-build
    fallback: tau, K locked to the packet's flat tier, and no extra round or probe.
    """
    token = getattr(scheduler, "_clear_build_token", None)
    fallback = False
    if getattr(scheduler, "_clear_plan_fallback", False):
        cached = getattr(scheduler, "_clear_fallback_plan", None)
        if isinstance(cached, tuple) and cached[0] == token:
            plan = cached[1]
        else:
            plan = make_fallback_round(scheduler, reqs)
            scheduler._clear_fallback_plan = (token, plan)
        fallback = True
    else:
        cached = getattr(scheduler, "_clear_primary_plan", None)
        if isinstance(cached, tuple) and cached[0] == token:
            plan = make_fallback_round(scheduler, reqs)
            scheduler._clear_plan_fallback = True
            scheduler._clear_fallback_plan = (token, plan)
            fallback = True
        else:
            plan = make_round(scheduler, reqs)
            scheduler._clear_primary_plan = (token, plan)
    _bind_round(scheduler, plan, reqs)
    if fallback:
        _mark_fallback(scheduler)
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


def _latest_scores(state) -> list:
    depth = int(getattr(state, "depth", 0) or 0)
    weight = getattr(state, "weight", None)
    if depth < 1 or weight is None:
        return []
    return [float(value) for value in weight[depth - 1].tolist()]


def _raw_marginal(state, k: int) -> float:
    """Upper bound on the newest layer, before the gain ratio."""
    scores = _path_scores(state)
    if not scores or int(getattr(state, "depth", 0) or 0) < 1:
        return 0.0
    ordered = sorted(scores, reverse=True)
    theta_k = ordered[int(k) - 1] if len(ordered) >= int(k) else 0.0
    return float(sum(max(0.0, score - theta_k) for score in _latest_scores(state)))


def _quality_increment(state, k: int) -> float:
    """Top-K mass this layer added, at one flat verify count."""
    scores = _path_scores(state)
    latest = _latest_scores(state)
    if not scores or not latest:
        return 0.0
    previous = scores[: -len(latest)] if len(latest) <= len(scores) else []
    return _prefix_mass(scores, k) - _prefix_mass(previous, k)


def _observe_gain(predicted: float, actual: float, forget: float) -> None:
    global _gain_pred, _gain_actual
    _gain_pred = _gain_pred * float(forget) + float(predicted)
    _gain_actual = _gain_actual * float(forget) + float(actual)


def gain_ratio() -> float:
    """How much of the upper bound the next layer has actually kept.

    No observations leave the bound unchanged. The ratio never exceeds 1.
    """
    if _gain_pred == 0.0 and _gain_actual == 0.0:
        return 1.0
    raw = (_gain_actual + CALIBRATE_LAMBDA) / (_gain_pred + CALIBRATE_LAMBDA)
    return min(1.0, float(raw))


def _marginal_gain(state, k: int) -> float:
    """How much of the newest layer still clears the current K-th score."""
    return _raw_marginal(state, k) * gain_ratio()


def _mark_fallback(scheduler) -> None:
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        mark_round_fallback,
    )

    mark_round_fallback(scheduler)


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


_REPLAY_EXPORTS = frozenset(
    {
        "choose_build_width",
        "truncate_columns",
    }
)


def __getattr__(name):
    if name not in _REPLAY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from sglang.srt.speculative.clear_tree_remote import clear_tree_replay as replay

    value = getattr(replay, name)
    globals()[name] = value
    return value
