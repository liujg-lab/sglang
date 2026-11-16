"""Shadow replay for a ClearTree round.

The served width changes only after the measured bias is inside 3 percent.
This module may call the price policy and the ledger. The ledger does not
import it.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import (
    VERIFY_KS,
    _prefix_mass,
    accept_ratio,
    keep_by_net_benefit,
    select_k,
)
from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
    CANDIDATE_WIDTHS,
    assign_persistent_width,
    batch_bucket,
    persistent_width,
)

REPLAY_EVERY = 20
REPLAY_GAIN = 0.03
NARROW_MIN_ROUNDS = 30
WIDE_MIN_ROUNDS = 10
BIAS_MIN_ROUNDS = 30
BIAS_LIMIT = 0.03

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


def reset_replay_state() -> None:
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


def choose_build_width(batch_size: int, *, ledger_on: bool, eligible: bool) -> tuple:
    """Flat or switched width, plus one temporary departure when the ledger is on.

    Width 4 spends every 20th eligible round at 8. Width 8 spends that round
    at 4. The departure does not depend on the batch bucket. It does not
    happen while the ledger is off, or on a warmup, probe, or locked round.
    """
    base = persistent_width(batch_size)
    if not ledger_on or not eligible:
        return base, "ordinary"
    bucket = batch_bucket(batch_size)
    _eligible_n[bucket] = int(_eligible_n.get(bucket, 0)) + 1
    if _eligible_n[bucket] % REPLAY_EVERY != 0:
        return base, "ordinary"
    if base == 4:
        return 8, "explore"
    if base == 8:
        return 4, "measure"
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
    """Built depth, or up to two layers earlier when the batch J says stop.

    ``layer_costs[i]`` is the forward of written layer ``i``. The gain is the
    latest written layer, not the next layer's real nodes.
    """
    rows = [
        {"pred": {0: None if cost is None else float(cost)}}
        for cost in list(layer_costs or [])
    ]
    return batch_stop_depth(
        [
            {
                "layers": layers,
                "kept": kept,
                "ratio": float(ratio),
                "prefix": 0,
            }
        ],
        theta=theta,
        verify_ms=verify_ms,
        layer_rows=rows,
        width=0,
    )

def _top_kept_nodes(layers, kept, end: int, limit: int) -> set:
    """Top ``limit`` kept nodes: score desc, depth asc, column asc."""
    nodes = []
    for depth in range(min(int(end), len(kept))):
        for col in kept[depth]:
            nodes.append(
                (
                    float(layers[depth][col]["score"]),
                    int(depth),
                    int(col),
                )
            )
    nodes.sort(key=lambda item: (-item[0], item[1], item[2]))
    return {(depth, col) for _score, depth, col in nodes[: max(int(limit), 0)]}


def replay_accept_k(layers, kept, tokens, stop: int, k_prime: int) -> int:
    """Accepted drafts on the top ``k_prime`` nodes, plus one bonus when present."""
    chosen = _top_kept_nodes(layers, kept, stop, k_prime)
    parent = None
    matched = 0
    sequence = [int(token) for token in tokens]
    depth_limit = min(len(layers), max(int(stop), 0))
    for depth in range(depth_limit):
        if matched >= len(sequence):
            break
        found = None
        for col in kept[depth]:
            if (depth, int(col)) not in chosen:
                continue
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
    if len(sequence) > matched:
        return matched + 1
    return matched


def batch_stop_depth(trees, *, theta, verify_ms, layer_rows, width: int) -> int:
    """One kept depth for the batch. The gain uses the latest written layer.

    ``trees`` carry ``layers``, ``kept``, ``ratio``, and ``prefix``. A missing
    layer cost keeps the built depth. The comparison is the shared net-benefit
    rule, so one request does not pay the whole batch's layer cost alone.
    """
    if not trees or theta is None:
        return max((len(tree["layers"]) for tree in trees), default=0)
    depth = max(len(tree["layers"]) for tree in trees)
    if depth < 2:
        return depth
    earliest = max(1, depth - 2)
    times = {
        int(k): float(spent)
        for k, spent in (verify_ms or {}).items()
        if spent is not None
    }
    for stop in range(earliest, depth):
        row = layer_rows[stop - 1] if 0 <= stop - 1 < len(layer_rows) else None
        pred = None if row is None else row.get("pred", {}).get(int(width))
        if pred is None:
            return depth
        requests = []
        for tree in trees:
            end = min(int(stop), len(tree["layers"]))
            if end < 1:
                continue
            scores = _kept_scores(tree["layers"], tree["kept"], end)
            latest = [
                float(tree["layers"][end - 1][col]["score"])
                for col in tree["kept"][end - 1]
            ]
            requests.append(
                {
                    "e": {int(k): _prefix_mass(scores, int(k)) for k in VERIFY_KS},
                    "g": {
                        int(k): _list_marginal(scores, latest, int(k))
                        for k in VERIFY_KS
                    },
                    "ell": int(tree.get("prefix", 0) or 0),
                    "c": float(tree["ratio"]),
                    "item": tree,
                    "growing": True,
                }
            )
        if not requests:
            return depth

        def layer_cost(_count, _ell, _pred=float(pred)):
            return _pred

        kept_items = keep_by_net_benefit(
            requests,
            theta=float(theta),
            verify_times=times,
            layer_cost=layer_cost,
            kappa=0.0,
        )
        if kept_items is None:
            return depth
        if not kept_items:
            return stop
    return depth

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
    """Best servable width under the 20-round, 3 percent, 30/10 rule.

    Width 2 can win the shadow comparison and is written to ``would_w``. It
    cannot block a switch among 4 and 8, and it is never served.
    """
    best = int(current)
    best_gain = 0.0
    shadow = int(current)
    shadow_gain = 0.0
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
        if gain > shadow_gain and gain > REPLAY_GAIN:
            shadow = int(cand)
            shadow_gain = float(gain)
        if int(cand) not in (4, 8):
            continue
        if gain > best_gain and gain > REPLAY_GAIN:
            best = int(cand)
            best_gain = float(gain)
    _would_w[int(bucket)] = int(shadow)
    applied = False
    if (
        allow_apply
        and best != int(current)
        and best in (4, 8)
        and int(current) in (4, 8)
        and bias_gate_open(bucket)
    ):
        assign_persistent_width(int(bucket), int(best))
        _clear_arms(bucket)
        applied = True
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
            for width, pair in arms.items():
                if int(width) == int(current) or float(pair[1]) <= 0.0:
                    continue
                arm = _arm(bucket, width)
                arm["tok"] += float(pair[0])
                arm["ms"] += float(pair[1])
                arm["cur_tok"] += float(cur_tok)
                arm["cur_ms"] += float(cur_ms)
                arm["n"] += 1.0
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
    # The last written layer is verified and not forwarded.
    for layer in range(max(int(depth) - 1, 0)):
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
        prepared = []
        for rid, tree in trees.items():
            tokens = tokens_by_rid.get(rid)
            if tokens is None:
                return {}
            layers = tree["layers"]
            prepared.append(
                {
                    "layers": layers,
                    "kept": truncate_columns(layers, width),
                    "tokens": tokens,
                    "ratio": float(tree["ratio"]),
                    "prefix": int(tree.get("prefix", 0) or 0),
                }
            )
        stop = batch_stop_depth(
            prepared,
            theta=theta,
            verify_ms=verify_ms,
            layer_rows=layer_rows,
            width=int(width),
        )
        scores = []
        ratios = []
        for tree in prepared:
            end = min(int(stop), len(tree["layers"]))
            scores.append(_kept_scores(tree["layers"], tree["kept"], end))
            ratios.append(float(tree["ratio"]))
        k_prime = _select_replay_k(scores, ratios, theta, verify_ms, int(hold["k"]))
        accepts = [
            replay_accept_k(
                tree["layers"],
                tree["kept"],
                tree["tokens"],
                min(int(stop), len(tree["layers"])),
                k_prime,
            )
            for tree in prepared
        ]
        model_actual = _schedule_ms(layer_rows, built, len(layer_rows), verify_ms, int(hold["k"]))
        model_cand = _schedule_ms(
            layer_rows,
            int(width),
            max(int(stop) - 1, 0),
            verify_ms,
            k_prime,
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

