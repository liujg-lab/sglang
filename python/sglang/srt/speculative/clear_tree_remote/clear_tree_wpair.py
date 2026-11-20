"""Block-wise paired width rule for one ClearTree batch bucket.

A block builds the next wider candidate on a share of eligible rounds. Ordinary
rounds price the next narrower cut. Superset rounds price both neighbors. The
block ends early when the widen samples are already decisive, and otherwise at
30. One judgment then moves to the neighbor with the larger lower bound, or
stays. This module does not import the replay module at load time.
"""

from __future__ import annotations

import math
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
_COUNT_EPS = 1e-9

_blocks: Dict[int, dict] = {}


def _fresh() -> dict:
    return {"up": [], "down": [], "acc": 0.0, "virt": 0.0, "p": P0}


def _block(bucket: int) -> dict:
    key = int(bucket)
    found = _blocks.get(key)
    if found is None:
        found = _fresh()
        _blocks[key] = found
    return found


def reset_pair_state() -> None:
    _blocks.clear()


def _gs(rows: Sequence[tuple]):
    """Ratio of means and its standard error, with a floor on the per-round sd."""
    n = len(rows)
    if n < 2:
        return None, None
    total_d = sum(float(d) for d, _e in rows)
    total_e = sum(float(e) for _d, e in rows)
    if total_e <= 0.0:
        return None, None
    gain = total_d / total_e
    mean_e = total_e / n
    var = sum((float(d) - gain * float(e)) ** 2 for d, e in rows) / (n - 1)
    se = math.sqrt(var / n) / mean_e
    return gain, max(se, SIG_FLOOR / math.sqrt(n))


def superset_due(bucket: int, base: int, ready) -> bool:
    """True when this eligible round should build the next wider candidate.

    ``ready`` false keeps the accumulated share for a later round. The widest
    candidate has no superset round; its block length is the same share, stored
    on ``virt``.
    """
    blk = _block(bucket)
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
    """``(V, E)`` for each requested width on this frozen tree.

    ``E`` is the accept-ratio weighted prefix mass at the width's own K*. ``V``
    subtracts theta times the frozen layer schedule and that K*'s verify time.
    A width is omitted when theta, that verify time, or any layer prediction
    is missing.
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
        found[int(width)] = (expect - float(theta) * float(spent), expect)
    return found


def _sample(values: dict, left: int, right: int):
    if int(left) not in values or int(right) not in values:
        return None
    value_left, _expect_left = values[int(left)]
    value_right, expect_right = values[int(right)]
    return (float(value_left) - float(value_right), float(expect_right))


def observe_pair(frozen: dict) -> Optional[dict]:
    """Record this round's paired values and judge the block when it ends."""
    if not isinstance(frozen, dict) or "width" not in frozen:
        return None
    batch = int(frozen.get("batch", 1))
    served = persistent_width(batch)
    built = int(frozen["width"])
    bucket = batch_bucket(batch)
    blk = _block(bucket)
    narrow = _next_narrower(served)
    wider = _next_wider(served)
    wanted = []
    if built == int(served) or (wider is not None and built == int(wider)):
        wanted.append(int(served))
        if narrow is not None:
            wanted.append(int(narrow))
        if wider is not None and built == int(wider):
            wanted.append(int(wider))
    values = pair_values(frozen, wanted) if wanted else {}
    if built == int(served) and narrow is not None:
        sample = _sample(values, narrow, served)
        if sample is not None:
            blk["down"].append(sample)
    elif wider is not None and built == int(wider):
        sample = _sample(values, wider, served)
        if sample is not None:
            blk["up"].append(sample)
        if narrow is not None:
            sample = _sample(values, narrow, served)
            if sample is not None:
                blk["down"].append(sample)
    return _block_check(bucket, blk, served, batch)


def _block_check(bucket: int, blk: dict, served: int, batch: int) -> Optional[dict]:
    wider = _next_wider(served)
    has_up = wider is not None
    counted = float(len(blk["up"])) if has_up else float(blk["virt"])
    if counted + _COUNT_EPS < N1:
        return None
    ended = counted + _COUNT_EPS >= N2
    if not ended and has_up:
        gain, se = _gs(blk["up"])
        if gain is not None and (
            gain - Z * se > BAND or gain + Z * se < -BAND
        ):
            ended = True
    if not ended:
        return None
    cands = []
    gain_up, se_up = (None, None)
    gain_down, se_down = (None, None)
    if has_up:
        gain_up, se_up = _gs(blk["up"])
        if gain_up is not None and gain_up - Z * se_up > BAND:
            cands.append((gain_up - Z * se_up, int(wider), "up"))
    narrow = _next_narrower(served)
    if narrow is not None and len(blk["down"]) >= MIN_DOWN:
        gain_down, se_down = _gs(blk["down"])
        if gain_down is not None and gain_down - Z * se_down > BAND:
            cands.append((gain_down - Z * se_down, int(narrow), "down"))
    decision = "stay"
    target = int(served)
    if cands:
        best = max(item[0] for item in cands)
        winners = [item for item in cands if abs(item[0] - best) <= 1e-12]
        if len(winners) == 1:
            _bound, target, decision = winners[0]
    record = {
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
        "decision": decision,
        "target": int(target),
    }
    if decision == "stay":
        rate = float(blk["p"])
        blk["up"] = []
        blk["down"] = []
        blk["virt"] = 0.0
        blk["p"] = max(rate / 2.0, P_MIN)
    else:
        assign_persistent_width(int(bucket), int(target))
        _blocks[int(bucket)] = _fresh()
    return record
