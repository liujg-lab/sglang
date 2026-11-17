"""Shadow replay for a ClearTree round.

The served width changes when a candidate's gain clears 3 percent. A move to
either neighbor needs 30 paired rounds, and a narrower move is only the next
smaller width. Measured bias is logged and does not decide the switch. When
the next narrower width has missed 300 eligible rounds, it is built until its
layer fit has four samples, at most four rounds, and for two rounds after
that. If that width already has four samples but cannot price this round's
prefix sum, it is built for one round after 20 eligible rounds. This module
may call the price policy and the ledger. The ledger does not import it.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import (
    EXPLORE_EVERY,
    PROBE_RUNS,
    VERIFY_KS,
    _prefix_mass,
    _tier_for_nodes,
    accept_ratio,
    gain_ratio,
    keep_by_net_benefit,
    select_k,
)
from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
    CANDIDATE_WIDTHS,
    _next_narrower,
    _next_wider,
    assign_persistent_width,
    batch_bucket,
    flat_k,
    persistent_width,
)

REPLAY_GAIN = 0.03
MIN_PAIRS = 30
PROBE_W_IDLE = 300

_eligible_n: Dict[int, int] = {}
_replay_n: Dict[int, int] = {}
_replay_arms: Dict[int, Dict[int, Dict[str, float]]] = {}
_would_w: Dict[int, int] = {}
_real_n: Dict[tuple, int] = {}
_real_tok: Dict[tuple, float] = {}
_real_ms: Dict[tuple, float] = {}
_pair_n: Dict[tuple, int] = {}
_pair_replay_tok: Dict[tuple, float] = {}
_pair_replay_ms: Dict[tuple, float] = {}
_pair_wide_tok: Dict[tuple, float] = {}
_pair_wide_ms: Dict[tuple, float] = {}
_narrow_idle: Dict[tuple, int] = {}
_probe_w_left: Dict[tuple, int] = {}
_probe_w_fill: Dict[tuple, bool] = {}


def reset_replay_state() -> None:
    _eligible_n.clear()
    _replay_n.clear()
    _replay_arms.clear()
    _would_w.clear()
    _real_n.clear()
    _real_tok.clear()
    _real_ms.clear()
    _pair_n.clear()
    _pair_replay_tok.clear()
    _pair_replay_ms.clear()
    _pair_wide_tok.clear()
    _pair_wide_ms.clear()
    _narrow_idle.clear()
    _probe_w_left.clear()
    _probe_w_fill.clear()


def _mark_width_ran(bucket: int, width: int) -> None:
    """This width was built, so its narrow-timing counter starts over."""
    key = (int(bucket), int(width))
    _narrow_idle[key] = 0
    _probe_w_left[key] = 0


def _layer_samples(scheduler, width: int) -> int:
    """Layer samples already stored for ``width``. No ledger counts as zero."""
    if scheduler is None:
        return 0
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        layer_sample_count,
    )

    return int(layer_sample_count(scheduler, width))


def _narrow_unpriced(scheduler, width: int, batch: int, sum_ell: int) -> bool:
    """True when this width cannot price the round's prefix sum."""
    if scheduler is None:
        return False
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        predict_layer_ms,
    )

    return (
        predict_layer_ms(scheduler, int(width), int(batch), int(sum_ell)) is None
    )


def choose_build_width(
    batch_size: int,
    *,
    ledger_on: bool,
    eligible: bool,
    scheduler=None,
    sum_ell: int = 0,
) -> tuple:
    """Flat or switched width, plus a narrow timing probe and a wider build.

    The next narrower width is built when it has gone 300 eligible rounds
    without a real build. The first bursts for that width run until its layer
    fit has four samples, and never more than four rounds. Later bursts are
    two rounds. A width that already has four samples, cannot price this
    round's prefix sum, and has waited 20 eligible rounds is built for one
    round. The counter starts at 300, so the first eligible round is a
    probe. Those rounds are ``probe_w`` and take the slot ahead of
    exploration. Every 20th remaining eligible round builds the next larger
    candidate. The widest candidate stays on that round. The departure does
    not happen while the ledger is off, or on a K-probe or locked round.
    """
    base = persistent_width(batch_size)
    if not ledger_on or not eligible:
        return base, "ordinary"
    bucket = batch_bucket(batch_size)
    _eligible_n[bucket] = int(_eligible_n.get(bucket, 0)) + 1
    narrow = _next_narrower(base)
    if narrow is not None:
        from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
            LAYER_MIN_SAMPLES,
        )

        key = (int(bucket), int(narrow))
        left = int(_probe_w_left.get(key, 0))
        idle = int(_narrow_idle.get(key, PROBE_W_IDLE))
        samples = _layer_samples(scheduler, narrow)
        if (
            left > 0
            and _probe_w_fill.get(key, False)
            and samples >= LAYER_MIN_SAMPLES
        ):
            _probe_w_left[key] = 0
            _probe_w_fill[key] = False
            left = 0
        early = (
            left <= 0
            and samples >= LAYER_MIN_SAMPLES
            and EXPLORE_EVERY <= idle < PROBE_W_IDLE
            and _narrow_unpriced(scheduler, narrow, batch_size, sum_ell)
        )
        if left > 0 or idle >= PROBE_W_IDLE or early:
            if left <= 0:
                if early:
                    left = 1
                    _probe_w_fill[key] = False
                else:
                    filling = samples < LAYER_MIN_SAMPLES
                    left = LAYER_MIN_SAMPLES if filling else PROBE_RUNS
                    _probe_w_fill[key] = filling
            _probe_w_left[key] = left - 1
            _narrow_idle[key] = 0
            return int(narrow), "probe_w"
        _narrow_idle[key] = idle + 1
    if _eligible_n[bucket] % EXPLORE_EVERY == 0:
        nxt = _next_wider(base)
        if nxt is not None:
            _mark_width_ran(bucket, nxt)
            return int(nxt), "explore"
    _mark_width_ran(bucket, base)
    return base, "ordinary"

def _cached_build_width(
    scheduler, batch: int, *, ledger_on: bool, eligible: bool, sum_ell: int = 0
):
    """One width per packet. A fallback expand must not take a second departure."""
    token = None if scheduler is None else getattr(scheduler, "_clear_build_token", None)
    cached = None if scheduler is None else getattr(scheduler, "_clear_build_cache", None)
    if token is not None and isinstance(cached, tuple) and cached[0] == token:
        return cached[1]
    chosen = choose_build_width(
        batch,
        ledger_on=ledger_on,
        eligible=eligible,
        scheduler=scheduler,
        sum_ell=sum_ell,
    )
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


def batch_stop_depth(
    trees, *, theta, verify_ms, layer_rows, width: int, gain_ratio: float = 1.0
) -> int:
    """One kept depth for the batch. The gain uses the latest written layer.

    ``trees`` carry ``layers``, ``kept``, ``ratio``, and ``prefix``. A missing
    layer cost keeps the built depth. The comparison is the shared net-benefit
    rule, so one request does not pay the whole batch's layer cost alone. A
    tree whose own build depth is already at or behind ``stop`` is not growing.
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
            scale = float(gain_ratio)
            requests.append(
                {
                    "e": {int(k): _prefix_mass(scores, int(k)) for k in VERIFY_KS},
                    "g": {
                        int(k): _list_marginal(scores, latest, int(k)) * scale
                        for k in VERIFY_KS
                    },
                    "ell": int(tree.get("prefix", 0) or 0),
                    "c": float(tree["ratio"]),
                    "item": tree,
                    "growing": len(tree["layers"]) > int(stop),
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

def _retained_nodes(tree, stop: int) -> int:
    """Kept nodes at or above the root and before ``stop``."""
    kept = tree.get("kept") or []
    end = min(int(stop), len(kept))
    return sum(len(kept[depth]) for depth in range(max(end, 0)))


def _clamp_replay_k(k: int, cap: Optional[int]) -> int:
    chosen = int(k)
    if cap is None:
        return chosen
    return min(chosen, int(cap))


def _select_replay_k(
    score_rows,
    ratios,
    theta,
    verify_ms,
    default_k: int,
    cap: Optional[int] = None,
    flat_k_value: Optional[int] = None,
) -> int:
    """Value-rule K at or below ``cap``. No priced tier inside ``cap`` uses the flat K.

    ``cap`` is the retained-node tier. ``flat_k_value`` is this round's flat
    tier and is not clamped to ``cap``.
    """
    if theta is None:
        if flat_k_value is not None:
            return int(flat_k_value)
        return _clamp_replay_k(default_k, cap)
    values = {}
    for k, spent in verify_ms.items():
        if spent is None:
            continue
        if cap is not None and int(k) > int(cap):
            continue
        benefit = 0.0
        for scores, ratio in zip(score_rows, ratios):
            benefit += float(ratio) * _prefix_mass(scores, int(k))
        values[int(k)] = benefit - float(theta) * float(spent)
    if not values:
        if flat_k_value is not None:
            return int(flat_k_value)
        return _clamp_replay_k(default_k, cap)
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

def _real_key(bucket: int, width: int) -> tuple:
    return (int(bucket), int(width))

def _pair_key(bucket: int, wide: int, narrow: int) -> tuple:
    return (int(bucket), int(wide), int(narrow))

def _ordered_pair(left: int, right: int) -> tuple:
    if int(left) >= int(right):
        return int(left), int(right)
    return int(right), int(left)

def replay_bias(bucket: int, wide: int, narrow: int) -> Optional[float]:
    """Gap between truncated throughput and real narrow throughput.

    Both are divided by the real wide throughput of rounds that replayed this
    pair. None until each side has ``MIN_PAIRS`` rounds.
    """
    real = _real_key(bucket, narrow)
    pair = _pair_key(bucket, wide, narrow)
    if int(_real_n.get(real, 0)) < MIN_PAIRS:
        return None
    if int(_pair_n.get(pair, 0)) < MIN_PAIRS:
        return None
    wide_ms = float(_pair_wide_ms.get(pair, 0.0))
    replay_ms = float(_pair_replay_ms.get(pair, 0.0))
    narrow_ms = float(_real_ms.get(real, 0.0))
    if wide_ms <= 0.0 or replay_ms <= 0.0 or narrow_ms <= 0.0:
        return None
    base = float(_pair_wide_tok.get(pair, 0.0)) / wide_ms
    if base <= 0.0:
        return None
    replay_ratio = (float(_pair_replay_tok.get(pair, 0.0)) / replay_ms) / base - 1.0
    real_ratio = (float(_real_tok.get(real, 0.0)) / narrow_ms) / base - 1.0
    return float(replay_ratio - real_ratio)

def _bias_report(bucket: int, served: int, would: int) -> Optional[float]:
    """Bias of the served width against ``would``. Diagnostic only."""
    if int(served) == int(would):
        return None
    wide, narrow = _ordered_pair(served, would)
    return replay_bias(bucket, wide, narrow)

def _note_bias(bucket: int, built: int, actual_tokens, actual_ms, replayed) -> None:
    if float(actual_ms) <= 0.0:
        return
    real = _real_key(bucket, built)
    _real_n[real] = int(_real_n.get(real, 0)) + 1
    _real_tok[real] = float(_real_tok.get(real, 0.0)) + float(actual_tokens)
    _real_ms[real] = float(_real_ms.get(real, 0.0)) + float(actual_ms)
    for width, pair in replayed.items():
        if int(width) >= int(built) or pair[1] is None:
            continue
        key = _pair_key(bucket, built, width)
        _pair_n[key] = int(_pair_n.get(key, 0)) + 1
        _pair_replay_tok[key] = float(_pair_replay_tok.get(key, 0.0)) + float(pair[0])
        _pair_replay_ms[key] = float(_pair_replay_ms.get(key, 0.0)) + float(pair[1])
        _pair_wide_tok[key] = float(_pair_wide_tok.get(key, 0.0)) + float(actual_tokens)
        _pair_wide_ms[key] = float(_pair_wide_ms.get(key, 0.0)) + float(actual_ms)

def _judge(bucket: int, current: int, *, allow_apply: bool) -> tuple:
    """Best width under the 20-round, 3 percent, 30-pair rule.

    Both a narrower and a wider candidate need 30 rounds. ``would_w`` is the
    largest gain, including a width that is not the adjacent narrower step.
    The served width moves only to that adjacent narrower candidate or to a
    wider one.
    """
    best = int(current)
    best_gain = 0.0
    shadow = int(current)
    shadow_gain = 0.0
    adjacent = _next_narrower(current)
    for cand in CANDIDATE_WIDTHS:
        if int(cand) == int(current):
            continue
        arm = _arm(bucket, cand)
        needed = MIN_PAIRS
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
        if int(cand) < int(current) and int(cand) != adjacent:
            continue
        if gain > best_gain and gain > REPLAY_GAIN:
            best = int(cand)
            best_gain = float(gain)
    _would_w[int(bucket)] = int(shadow)
    applied = False
    if allow_apply and best != int(current):
        assign_persistent_width(int(bucket), int(best))
        _clear_arms(bucket)
        applied = True
    return best, shadow_gain, applied

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

    ``counted`` is false for K-probe and locked rounds. A measure
    round updates bias only. ``would_w`` can name a candidate that is not the
    adjacent narrower step. The logged bias is the pair of the width served
    at the start of this round and ``would_w``.
    """
    pairs = {} if replayed is None else {int(key): value for key, value in replayed.items()}
    bucket = batch_bucket(batch)
    served = persistent_width(batch)
    current = served
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
        if judged_rounds % EXPLORE_EVERY == 0:
            _best, gain, applied = _judge(bucket, current, allow_apply=apply)
    rounds = judged_rounds
    would = int(_would_w.get(bucket, served))
    bias = _bias_report(bucket, served, would)
    record = {
        "bucket": int(bucket),
        "b": int(batch),
        "w": int(built),
        "kind": str(kind),
        "candidates": _candidate_log(int(built), actual_tokens, actual_ms, pairs),
        "rounds": rounds,
        "gain": float(gain),
        "would_w": would,
        "bias": bias,
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
        # Row i is the forward of layer i. A tree whose last written layer is
        # i was verified there and did not forward it.
        active_states = [
            state
            for state in states
            if int(getattr(state, "depth", 0) or 0) > layer + 1
        ]
        pred = {}
        active = len(active_states)
        sum_ell = sum(int(getattr(state, "prefix", 0) or 0) for state in active_states)
        for width in CANDIDATE_WIDTHS:
            try:
                pred[int(width)] = predict_layer_ms(
                    scheduler, int(width), active, sum_ell
                )
            except Exception:
                pred[int(width)] = None
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
                "flat_k": int(getattr(plan, "flat_k", flat_k(int(plan.batch)))),
                "layers": _freeze_layer_rows(scheduler, usable),
                "gain_ratio": float(gain_ratio()),
            }
        existing["trees"].update(trees)
        existing["gain_ratio"] = float(gain_ratio())
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
            gain_ratio=float(hold.get("gain_ratio", 1.0) or 1.0),
        )
        scores = []
        ratios = []
        retained = 0
        for tree in prepared:
            end = min(int(stop), len(tree["layers"]))
            scores.append(_kept_scores(tree["layers"], tree["kept"], end))
            ratios.append(float(tree["ratio"]))
            retained = max(retained, _retained_nodes(tree, stop))
        cap = None if retained <= 0 else _tier_for_nodes(retained, VERIFY_KS)
        stored_flat = hold.get("flat_k")
        if stored_flat is None:
            stored_flat = flat_k(int(hold.get("batch", 1)))
        k_prime = _select_replay_k(
            scores,
            ratios,
            theta,
            verify_ms,
            int(hold["k"]),
            cap,
            flat_k_value=int(stored_flat),
        )
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

