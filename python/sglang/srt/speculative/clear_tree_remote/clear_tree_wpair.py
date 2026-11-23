"""Block-wise paired width rule for one ClearTree width key.

The default rule (``simple``) keeps one state per (batch bucket, KV bucket of
the round's longest prefix). A block builds the next wider candidate on a
share ``p`` of eligible rounds. A superset round records an up sample (wider
over served) and a down sample (narrower over served) from its frozen tree.
An ordinary round records a down sample from its own tree, but only when the
wider width is also priced on that round, so both directions cover the same
rounds; the widest tier has no wider width and always records. The block ends
at 30 up samples, or at a virtual count of 30 on the widest tier; a block
still short of that after ``ROUND_CAP`` eligible rounds closes as a stay. An
ordinary round's down sample also needs every frozen layer row to price the
wider width, as a superset round's up sample does. Both sides share one set of
six round-index groups; the se is the larger of the spread of the group gains
and the per-row se. A side passes when ``gain - 2 se > 1%``. Exactly one passing side
switches and restarts the key at the opening rate. Otherwise the samples are
cleared and the rate halves down to the floor.

``SGLANG_CLEAR_WPAIR_RULE=legacy`` keeps the older rule: one state per batch
bucket, superset rounds only, judgments at 15 samples (``g - 2 se``) and 30
(``g - se``) against 0.5%, and a down side that needs 15 samples.

The default gain is the paired throughput ratio: on the same frozen tree,
``(E + B) / (spent + s)`` for the other width over the served width, minus 1.
``B`` is the packet's bonus tokens and ``s`` the recent seed time; both are
shared by the two widths. ``SGLANG_CLEAR_WPAIR_METRIC=value`` judges on the
older ``sum(dV) / sum(E)`` instead. Both are logged. This module does not
import the replay module at load time.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional, Sequence

from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import (
    VERIFY_KS,
    _prefix_mass,
    _tier_for_nodes,
)
from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
    _next_narrower,
    _next_wider,
    assign_persistent_width,
    batch_bucket,
    persistent_width,
)

P0 = 0.20
P_MIN = 0.05
N1 = 15
N2 = 30
Z = 2.0
BAND = 0.005
SIG_FLOOR = 0.01
MIN_DOWN = 15
BAND_SIMPLE = 0.01
GROUPS = 6
# A block that never gathers N2 up samples is closed as a stay after this many
# eligible rounds: twice the length of an N2 block at the floor rate.
ROUND_CAP = int(round(2 * N2 / P_MIN))
_COUNT_EPS = 1e-9

# Keyed by (batch bucket, KV bucket). The legacy rule uses a None KV bucket.
_blocks: Dict[tuple, dict] = {}


def _fresh(epoch: int = 0) -> dict:
    return {
        "up": [],
        "down": [],
        "up_ix": [],
        "down_ix": [],
        "acc": 0.0,
        "virt": 0.0,
        "p": P0,
        "n1": False,
        "rounds": 0,
        "epoch": int(epoch),
    }


def _key(bucket, kv: Optional[int] = None) -> tuple:
    """``(batch bucket, KV bucket)``. A plain bucket uses ``kv``."""
    if isinstance(bucket, tuple):
        bucket, kv = bucket
    return (int(bucket), None if kv is None else int(kv))


def _block(bucket, kv: Optional[int] = None) -> dict:
    key = _key(bucket, kv)
    found = _blocks.get(key)
    if found is None:
        found = _fresh()
        _blocks[key] = found
    return found


def reset_pair_state() -> None:
    _blocks.clear()


def wpair_rule() -> str:
    """``simple`` (default) or ``legacy`` from ``SGLANG_CLEAR_WPAIR_RULE``."""
    raw = os.environ.get("SGLANG_CLEAR_WPAIR_RULE", "").strip().lower()
    return "legacy" if raw == "legacy" else "simple"


def width_kv_bucket(max_ell: int, buckets: Sequence[int]) -> Optional[int]:
    """KV bucket that keys this round's width state.

    The smallest bucket holding ``max_ell + 1``, or -1 past every bucket. The
    legacy rule returns None, the batch-only key.
    """
    if wpair_rule() == "legacy":
        return None
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        smallest_bucket,
    )

    found = smallest_bucket(int(max_ell) + 1, [int(item) for item in buckets or ()])
    return -1 if found is None else int(found)


def round_tag(bucket, kv: Optional[int] = None) -> tuple:
    """``(epoch, index)`` of the eligible round just counted by ``superset_due``."""
    blk = _block(bucket, kv)
    return (int(blk["epoch"]), max(int(blk["rounds"]) - 1, 0))


def wpair_metric() -> str:
    """``ratio`` (default) or ``value`` from ``SGLANG_CLEAR_WPAIR_METRIC``."""
    raw = os.environ.get("SGLANG_CLEAR_WPAIR_METRIC", "").strip().lower()
    return "value" if raw == "value" else "ratio"


def _gs(rows: Sequence[tuple]):
    """Ratio of means and its standard error, with a floor on the per-round sd.

    Uses the first two entries of each row: ``(dV, E)``.
    """
    n = len(rows)
    if n < 2:
        return None, None
    total_d = sum(float(row[0]) for row in rows)
    total_e = sum(float(row[1]) for row in rows)
    if total_e <= 0.0:
        return None, None
    gain = total_d / total_e
    mean_e = total_e / n
    var = sum((float(row[0]) - gain * float(row[1])) ** 2 for row in rows) / (n - 1)
    se = math.sqrt(var / n) / mean_e
    return gain, max(se, SIG_FLOOR / math.sqrt(n))


def _ratio_gs(rows: Sequence[tuple]):
    """Paired throughput ratio minus 1 and its delta-method standard error.

    Each usable row ends with ``(N_alt, D_alt, N_srv, D_srv)``. Rows without
    them are skipped. The se has the same per-round floor as ``_gs``.
    """
    usable = [
        tuple(float(value) for value in row[2:6])
        for row in rows
        if len(row) >= 6 and all(value is not None for value in row[2:6])
    ]
    n = len(usable)
    if n < 2:
        return None, None
    means = [sum(row[i] for row in usable) / n for i in range(4)]
    if any(value <= 0.0 for value in means):
        return None, None
    ratio = (means[0] / means[1]) / (means[2] / means[3])
    terms = [
        row[0] / means[0] - row[1] / means[1] - row[2] / means[2] + row[3] / means[3]
        for row in usable
    ]
    center = sum(terms) / n
    var = sum((term - center) ** 2 for term in terms) / (n - 1)
    se = ratio * math.sqrt(var / n)
    return ratio - 1.0, max(se, SIG_FLOOR / math.sqrt(n))


def superset_due(bucket, base: int, ready, kv: Optional[int] = None) -> bool:
    """True when this eligible round should build the next wider candidate.

    ``ready`` false keeps the accumulated share for a later round. The widest
    candidate has no superset round; its block length is the same share, stored
    on ``virt``. Every call counts one eligible round of the block.
    """
    blk = _block(bucket, kv)
    blk["rounds"] = int(blk["rounds"]) + 1
    wider = _next_wider(base)
    if wider is None:
        blk["virt"] = float(blk["virt"]) + float(blk["p"])
        return False
    blk["acc"] = min(float(blk["acc"]) + float(blk["p"]), 2.0)
    is_ready = False if ready is None else bool(ready(int(wider)))
    if float(blk["acc"]) >= 1.0 and is_ready:
        blk["acc"] = float(blk["acc"]) - 1.0
        return True
    return False


def pair_values(frozen: dict, widths: Sequence[int]) -> Dict[int, tuple]:
    """``(V, E, spent)`` for each requested width on this frozen tree.

    ``E`` is the accept-ratio weighted prefix mass at the width's own K*.
    ``spent`` is the frozen layer schedule plus that K*'s verify time, and
    ``V`` subtracts theta times ``spent``. A width is omitted when theta,
    that verify time, or any layer prediction is missing.
    """
    from sglang.srt.speculative.clear_tree_remote.clear_tree_replay import (
        _kept_scores,
        _retained_nodes,
        _schedule_ms,
        _select_replay_k,
        truncate_columns,
    )

    theta = frozen.get("theta")
    if theta is None:
        return {}
    trees = frozen.get("trees") or {}
    if not trees:
        return {}
    layer_rows = list(frozen.get("layers") or [])
    verify_ms = frozen.get("verify_ms") or {}
    default_k = int(frozen.get("k", 1))
    flat = frozen.get("flat_k")
    found = {}
    for width in widths:
        score_rows = []
        ratios = []
        retained = 0
        for tree in trees.values():
            layers = tree["layers"]
            kept = truncate_columns(layers, int(width))
            score_rows.append(_kept_scores(layers, kept, len(layers)))
            ratios.append(float(tree["ratio"]))
            retained = max(retained, _retained_nodes({"kept": kept}, len(kept)))
        if retained <= 0:
            continue
        cap = _tier_for_nodes(retained, VERIFY_KS)
        k_star = _select_replay_k(
            score_rows,
            ratios,
            theta,
            verify_ms,
            default_k,
            cap,
            flat_k_value=None if flat is None else int(flat),
        )
        spent = _schedule_ms(
            layer_rows, int(width), len(layer_rows), verify_ms, int(k_star)
        )
        if spent is None:
            continue
        expect = 0.0
        for scores, ratio in zip(score_rows, ratios):
            expect += float(ratio) * _prefix_mass(scores, int(k_star))
        found[int(width)] = (
            expect - float(theta) * float(spent),
            expect,
            float(spent),
        )
    return found


def _sample(values: dict, left: int, right: int, bonus: float = 0.0, seed: float = 0.0):
    """``(dV, E_right, N_left, D_left, N_right, D_right)``, or None.

    ``N = E + bonus`` and ``D = spent + seed``. The last four are None when a
    value carries no ``spent``.
    """
    if int(left) not in values or int(right) not in values:
        return None
    entry_left = values[int(left)]
    entry_right = values[int(right)]
    gap = float(entry_left[0]) - float(entry_right[0])
    expect_right = float(entry_right[1])
    if len(entry_left) < 3 or len(entry_right) < 3:
        return (gap, expect_right, None, None, None, None)
    extra = float(bonus)
    shared = float(seed)
    return (
        gap,
        expect_right,
        float(entry_left[1]) + extra,
        float(entry_left[2]) + shared,
        expect_right + extra,
        float(entry_right[2]) + shared,
    )


def _shared_terms(frozen: dict) -> tuple:
    """Bonus tokens and seed time, the same for every width on this round."""
    return float(frozen.get("bonus", 0) or 0), float(frozen.get("seed_ms", 0.0) or 0.0)


def _record_superset(
    blk: dict, values: dict, served: int, narrow, wider, bonus=0.0, seed=0.0
) -> None:
    """Keep one round only when every compared width has a value."""
    up = _sample(values, int(wider), int(served), bonus, seed)
    if narrow is None:
        if up is not None:
            blk["up"].append(up)
        return
    down = _sample(values, int(narrow), int(served), bonus, seed)
    if up is None or down is None:
        return
    blk["up"].append(up)
    blk["down"].append(down)


def observe_pair(frozen: dict) -> Optional[dict]:
    """Record this frozen round's paired samples and judge the block."""
    if not isinstance(frozen, dict) or "width" not in frozen:
        return None
    if wpair_rule() == "legacy":
        return _observe_legacy(frozen)
    return _observe_simple(frozen)


def _observe_simple(frozen: dict) -> Optional[dict]:
    """Superset rounds give up and down; ordinary rounds give down."""
    tag = frozen.get("tag")
    if not isinstance(tag, (tuple, list)) or len(tag) != 2:
        return None
    batch = int(frozen.get("batch", 1))
    kv = frozen.get("kv")
    key = _key(batch_bucket(batch), kv)
    served = persistent_width(batch, key[1])
    blk = _block(key)
    if int(tag[0]) != int(blk["epoch"]):
        return None
    index = max(int(tag[1]), 0)
    built = int(frozen["width"])
    narrow = _next_narrower(served)
    wider = _next_wider(served)
    bonus, seed = _shared_terms(frozen)
    if wider is not None and built == int(wider):
        wanted = [int(served), int(wider)]
        if narrow is not None:
            wanted.append(int(narrow))
        n_up = len(blk["up"])
        n_down = len(blk["down"])
        _record_superset(
            blk, pair_values(frozen, wanted), served, narrow, wider, bonus, seed
        )
        if len(blk["up"]) > n_up:
            blk["up_ix"].append(index)
        if len(blk["down"]) > n_down:
            blk["down_ix"].append(index)
        return _block_check_simple(key, blk, served, batch)
    if built != int(served):
        return None
    if narrow is not None and _wider_priced_rows(frozen, wider):
        values = pair_values(frozen, [int(served), int(narrow)])
        sample = _sample(values, int(narrow), int(served), bonus, seed)
        if sample is not None:
            blk["down"].append(sample)
            blk["down_ix"].append(index)
    return _block_check_simple(key, blk, served, batch)


def _wider_priced_rows(frozen: dict, wider: Optional[int]) -> bool:
    """True when this ordinary round could have given an up sample.

    The plan's check uses the whole packet's prefix sum. A superset round
    also needs the wider width's prediction on every frozen layer row, whose
    active count and depth can differ. The widest tier has no wider width.
    """
    if wider is None:
        return True
    if not bool(frozen.get("wider_priced")):
        return False
    return all(
        (row.get("pred") or {}).get(int(wider)) is not None
        for row in list(frozen.get("layers") or [])
    )


def _observe_legacy(frozen: dict) -> Optional[dict]:
    """Record a superset round and judge at 15 and 30 paired samples."""
    batch = int(frozen.get("batch", 1))
    served = persistent_width(batch)
    built = int(frozen["width"])
    bucket = batch_bucket(batch)
    blk = _block(bucket)
    narrow = _next_narrower(served)
    wider = _next_wider(served)
    bonus, seed = _shared_terms(frozen)
    if wider is not None and built == int(wider):
        wanted = [int(served), int(wider)]
        if narrow is not None:
            wanted.append(int(narrow))
        _record_superset(
            blk, pair_values(frozen, wanted), served, narrow, wider, bonus, seed
        )
        return _block_check(bucket, blk, served, batch)
    if wider is None and built == int(served) and narrow is not None:
        values = pair_values(frozen, [int(served), int(narrow)])
        sample = _sample(values, int(narrow), int(served), bonus, seed)
        if sample is not None:
            blk["down"].append(sample)
        return _block_check(bucket, blk, served, batch)
    return None


def _bound_clears(rows: Sequence[tuple], zmul: float, metric: str = "value"):
    gain, se = _ratio_gs(rows) if metric == "ratio" else _gs(rows)
    if gain is None or se is None:
        return False, gain, se
    return gain - zmul * se > BAND, gain, se


def _block_check(bucket: int, blk: dict, served: int, batch: int) -> Optional[dict]:
    wider = _next_wider(served)
    has_up = wider is not None
    counted = float(len(blk["up"])) if has_up else float(blk["virt"])
    if counted + _COUNT_EPS < N1:
        return None
    at_thirty = counted + _COUNT_EPS >= N2
    if not at_thirty:
        if blk.get("n1"):
            return None
        blk["n1"] = True
        zmul = Z
    else:
        zmul = 1.0
    metric = wpair_metric()
    up_ok = False
    down_ok = False
    gain_up, se_up = (None, None)
    gain_down, se_down = (None, None)
    ratio_up, rse_up = (None, None)
    ratio_down, rse_down = (None, None)
    if has_up:
        up_ok, _gain, _se = _bound_clears(blk["up"], zmul, metric)
        gain_up, se_up = _gs(blk["up"])
        ratio_up, rse_up = _ratio_gs(blk["up"])
    narrow = _next_narrower(served)
    if narrow is not None and len(blk["down"]) >= MIN_DOWN:
        down_ok, _gain, _se = _bound_clears(blk["down"], zmul, metric)
        gain_down, se_down = _gs(blk["down"])
        ratio_down, rse_down = _ratio_gs(blk["down"])
    if not at_thirty and up_ok == down_ok:
        return None
    decision = "stay"
    target = int(served)
    if up_ok and not down_ok:
        decision = "up"
        target = int(wider)
    elif down_ok and not up_ok:
        decision = "down"
        target = int(narrow)
    record = {
        "rule": "legacy",
        "bucket": int(bucket),
        "b": int(batch),
        "w": int(served),
        "p": float(blk["p"]),
        "n_up": len(blk["up"]),
        "n_down": len(blk["down"]),
        "g_up": gain_up,
        "se_up": se_up,
        "g_dn": gain_down,
        "se_dn": se_down,
        "r_up": ratio_up,
        "rse_up": rse_up,
        "r_dn": ratio_down,
        "rse_dn": rse_down,
        "metric": metric,
        "decision": decision,
        "target": int(target),
    }
    if decision == "stay":
        rate = float(blk["p"])
        blk["up"] = []
        blk["down"] = []
        blk["virt"] = 0.0
        blk["n1"] = False
        if not (up_ok and down_ok):
            blk["p"] = max(rate / 2.0, P_MIN)
    else:
        assign_persistent_width(int(bucket), int(target))
        _blocks[_key(bucket)] = _fresh()
    return record


def _usable(row: tuple, metric: str) -> bool:
    if metric == "ratio":
        return len(row) >= 6 and all(value is not None for value in row[2:6])
    return True


def _gain_se(rows: Sequence[tuple], metric: str):
    return _ratio_gs(rows) if metric == "ratio" else _gs(rows)


def _grouped(rows: Sequence[tuple], ix: Sequence[int], rounds: int, metric: str):
    """Gain over all rows, its se, and the number of groups used.

    Rows fall into ``GROUPS`` equal spans of the block's eligible-round index,
    the same spans for both directions. A group needs two usable rows. With two
    or more groups the se is the spread of the group gains over sqrt(groups),
    but never below the per-row se (which carries the ``SIG_FLOOR / sqrt(n)``
    floor), since a few groups give an unstable spread. With fewer groups it
    is the per-row se.
    """
    gain, flat_se = _gain_se(rows, metric)
    if gain is None:
        return None, None, 0
    span = max(int(rounds), 1)
    groups: Dict[int, list] = {}
    for row, index in zip(rows, ix):
        if not _usable(row, metric):
            continue
        slot = min(GROUPS - 1, max(int(index), 0) * GROUPS // span)
        groups.setdefault(slot, []).append(row)
    gains = []
    for slot in sorted(groups):
        part, _ = _gain_se(groups[slot], metric)
        if part is not None:
            gains.append(float(part))
    count = len(gains)
    if count < 2:
        return gain, flat_se, count
    n = sum(1 for row in rows if _usable(row, metric))
    center = sum(gains) / count
    var = sum((value - center) ** 2 for value in gains) / (count - 1)
    se = math.sqrt(var / count)
    floor = SIG_FLOOR / math.sqrt(max(n, 1))
    return gain, max(se, floor, float(flat_se or 0.0)), count


def _block_check_simple(
    key: tuple, blk: dict, served: int, batch: int
) -> Optional[dict]:
    """Judge once at 30 up samples (virt on the widest tier).

    A block still short after ``ROUND_CAP`` eligible rounds closes as a stay.
    """
    wider = _next_wider(served)
    narrow = _next_narrower(served)
    has_up = wider is not None
    counted = float(len(blk["up"])) if has_up else float(blk["virt"])
    rounds = int(blk["rounds"])
    capped = counted + _COUNT_EPS < N2
    if capped and rounds < ROUND_CAP:
        return None
    metric = wpair_metric()
    empty = (None, None, 0)
    up_value = up_ratio = down_value = down_ratio = empty
    if has_up:
        up_value = _grouped(blk["up"], blk["up_ix"], rounds, "value")
        up_ratio = _grouped(blk["up"], blk["up_ix"], rounds, "ratio")
    if narrow is not None:
        down_value = _grouped(blk["down"], blk["down_ix"], rounds, "value")
        down_ratio = _grouped(blk["down"], blk["down_ix"], rounds, "ratio")
    up_judged = up_ratio if metric == "ratio" else up_value
    down_judged = down_ratio if metric == "ratio" else down_value

    def clears(found) -> bool:
        gain, se, _count = found
        if gain is None or se is None:
            return False
        return float(gain) - Z * float(se) > BAND_SIMPLE

    up_ok = clears(up_judged) and not capped
    down_ok = clears(down_judged) and not capped
    decision = "stay"
    target = int(served)
    if up_ok and not down_ok:
        decision = "up"
        target = int(wider)
    elif down_ok and not up_ok:
        decision = "down"
        target = int(narrow)
    record = {
        "rule": "simple",
        "bucket": int(key[0]),
        "kv_bucket": key[1],
        "b": int(batch),
        "w": int(served),
        "p": float(blk["p"]),
        "rounds": rounds,
        "n_up": len(blk["up"]),
        "n_down": len(blk["down"]),
        "groups_up": int(up_judged[2]),
        "groups_dn": int(down_judged[2]),
        "g_up": up_value[0],
        "se_up": up_value[1],
        "g_dn": down_value[0],
        "se_dn": down_value[1],
        "r_up": up_ratio[0],
        "rse_up": up_ratio[1],
        "r_dn": down_ratio[0],
        "rse_dn": down_ratio[1],
        "metric": metric,
        "capped": bool(capped),
        "decision": decision,
        "target": int(target),
    }
    epoch = int(blk["epoch"]) + 1
    if decision == "stay":
        fresh = _fresh(epoch)
        fresh["p"] = max(float(blk["p"]) / 2.0, P_MIN)
        fresh["acc"] = float(blk["acc"])
        _blocks[key] = fresh
    else:
        assign_persistent_width(int(key[0]), int(target), key[1])
        _blocks[key] = _fresh(epoch)
    return record
