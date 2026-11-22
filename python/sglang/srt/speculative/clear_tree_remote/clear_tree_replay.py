"""Build width and frozen trees for one ClearTree round.

Eligible rounds sometimes build the next wider candidate. Paired values on
that tree and on ordinary trees decide the served width at the end of a
block. When the next narrower width has missed 300 eligible rounds, it is
built until its layer fit has four samples, at most four rounds, and for two
rounds after that. If that width already has four samples but cannot price
this round's prefix sum, it is built for one round after 20 eligible rounds.
This module may call the price policy and the ledger. The ledger does not
import it.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import (
    EXPLORE_EVERY,
    PROBE_RUNS,
    VERIFY_KS,
    _prefix_mass,
    _tier_for_nodes,
    accept_ratio,
    select_k,
)
from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
    CANDIDATE_WIDTHS,
    _next_narrower,
    _next_wider,
    batch_bucket,
    flat_k,
    persistent_width,
)

PROBE_W_IDLE = 300

_narrow_idle: Dict[tuple, int] = {}
_probe_w_left: Dict[tuple, int] = {}
_probe_w_fill: Dict[tuple, bool] = {}


def reset_replay_state() -> None:
    _narrow_idle.clear()
    _probe_w_left.clear()
    _probe_w_fill.clear()
    from sglang.srt.speculative.clear_tree_remote.clear_tree_wpair import (
        reset_pair_state,
    )

    reset_pair_state()


def _mark_width_ran(bucket: int, width: int) -> None:
    """This width was built, so its narrow-timing counter starts over."""
    key = (int(bucket), int(width))
    _narrow_idle[key] = 0
    _probe_w_left[key] = 0


def _layer_samples(scheduler, width: int, batch_size: int) -> int:
    """Layer samples for ``width`` at this round's capture batch.

    No ledger counts as zero.
    """
    if scheduler is None:
        return 0
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        layer_sample_count,
    )

    return int(layer_sample_count(scheduler, width, int(batch_size)))


def _narrow_unpriced(scheduler, width: int, batch: int, sum_ell: int) -> bool:
    """True when this width cannot price the round's prefix sum."""
    if scheduler is None:
        return False
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        predict_layer_ms,
    )

    return predict_layer_ms(scheduler, int(width), int(batch), int(sum_ell)) is None


def choose_build_width(
    batch_size: int,
    *,
    ledger_on: bool,
    eligible: bool,
    scheduler=None,
    sum_ell: int = 0,
    ready=None,
) -> tuple:
    """Flat or switched width, plus a narrow timing probe and a wider build.

    The next narrower width is built when it has gone 300 eligible rounds
    without a real build. The first bursts for that width run until its layer
    fit has four samples, and never more than four rounds. Later bursts are
    two rounds. A width that already has four samples, cannot price this
    round's prefix sum, and has waited 20 eligible rounds is built for one
    round. The counter starts at 300, so the first eligible round is a
    probe. Those rounds are ``probe_w`` and take the slot ahead of a superset
    build. A missing ``ready`` does not spend a superset round. The departure
    does not happen while the ledger is off, or on a K-probe or locked round.
    """
    base = persistent_width(batch_size)
    if not ledger_on or not eligible:
        return base, "ordinary"
    bucket = batch_bucket(batch_size)
    narrow = _next_narrower(base)
    if narrow is not None:
        from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
            LAYER_MIN_SAMPLES,
        )

        key = (int(bucket), int(narrow))
        left = int(_probe_w_left.get(key, 0))
        idle = int(_narrow_idle.get(key, PROBE_W_IDLE))
        samples = _layer_samples(scheduler, narrow, batch_size)
        if left > 0 and _probe_w_fill.get(key, False) and samples >= LAYER_MIN_SAMPLES:
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
    from sglang.srt.speculative.clear_tree_remote.clear_tree_wpair import (
        superset_due,
    )

    if superset_due(bucket, base, ready):
        nxt = _next_wider(base)
        if nxt is not None:
            _mark_width_ran(bucket, nxt)
            return int(nxt), "superset"
    _mark_width_ran(bucket, base)
    return base, "ordinary"


def _cached_build_width(
    scheduler,
    batch: int,
    *,
    ledger_on: bool,
    eligible: bool,
    sum_ell: int = 0,
    ready=None,
):
    """One width per packet. A fallback expand must not take a second departure."""
    token = (
        None if scheduler is None else getattr(scheduler, "_clear_build_token", None)
    )
    cached = (
        None if scheduler is None else getattr(scheduler, "_clear_build_cache", None)
    )
    if token is not None and isinstance(cached, tuple) and cached[0] == token:
        return cached[1]
    chosen = choose_build_width(
        batch,
        ledger_on=ledger_on,
        eligible=eligible,
        scheduler=scheduler,
        sum_ell=sum_ell,
        ready=ready,
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


def _kept_scores(layers, kept, end: int) -> list:
    scores = []
    for depth in range(min(int(end), len(kept))):
        for col in kept[depth]:
            scores.append(float(layers[depth][col]["score"]))
    return scores


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
                # Row ``layer`` is the forward with layer index ``layer + 1``.
                pred[int(width)] = predict_layer_ms(
                    scheduler, int(width), active, sum_ell, depth=layer + 1
                )
            except Exception:
                pred[int(width)] = None
        rows.append({"active": active, "sum_ell": sum_ell, "pred": pred})
    return rows


def queue_hold_replay(scheduler, reqs, states, plan) -> None:
    """Register references only; freeze tensors and prices after the reply.

    Capture the served width now. This packet's post-reply may switch it
    after the trees are scored.
    """
    queued = getattr(scheduler, "_clear_replay_to_hold", None)
    if queued is None:
        queued = []
        scheduler._clear_replay_to_hold = queued
    queued.append(
        (
            list(reqs),
            list(states),
            plan,
            getattr(scheduler, "_clear_build_token", None),
            int(persistent_width(int(plan.batch))),
            getattr(scheduler, "_clear_replay_packet", None),
        )
    )


def hold_replay(
    scheduler, reqs, states, plan, *, token=None, served=None, packet=None
):
    """Freeze this packet's trees for the width rule. Returns the dict, or None."""
    try:
        if token is None:
            token = getattr(scheduler, "_clear_build_token", None)
        if served is None:
            served = persistent_width(int(plan.batch))
        if packet is None:
            packet = getattr(scheduler, "_clear_replay_packet", None)
        trees = {}
        usable = []
        for req, state in zip(reqs, states):
            layers = _layers_of(state)
            if not layers:
                continue
            rid = str(getattr(req, "rid", "") or getattr(state, "rid", ""))
            trees[rid] = {
                "layers": layers,
                "prefix": int(getattr(state, "prefix", 0) or 0),
                "ratio": float(accept_ratio(rid)),
            }
            usable.append(state)
        if not trees:
            return None
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
        return {
            "token": token,
            "packet": packet,
            "served": int(served),
            "trees": trees,
            "width": int(plan.width),
            "kind": str(getattr(plan, "replay_kind", "ordinary")),
            "eligible": bool(getattr(plan, "replay_eligible", False)),
            "batch": int(plan.batch),
            "theta": plan.theta_value,
            "verify_ms": verify,
            "k": k_actual,
            "flat_k": int(getattr(plan, "flat_k", flat_k(int(plan.batch)))),
            "layers": _freeze_layer_rows(scheduler, usable),
        }
    except Exception:
        import logging

        logging.getLogger(__name__).warning(
            "[CLEAR ledger] replay hold failed", exc_info=True
        )
        return None


def run_post_reply(scheduler) -> None:
    """Freeze this packet, judge its width block, then flush Draft logs once."""
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        add_post_ms,
        flush_draft_logs,
        queue_draft_log,
    )
    from sglang.srt.speculative.clear_tree_remote.clear_tree_wpair import (
        observe_pair,
    )

    started = time.perf_counter()
    token = getattr(scheduler, "_clear_build_token", None)
    packet = getattr(scheduler, "_clear_replay_packet", None)
    post_key = (token, packet)
    if getattr(scheduler, "_clear_replay_post_done", None) == post_key:
        return
    scheduler._clear_replay_post_done = post_key
    previous_post = getattr(scheduler, "_clear_in_post_reply", False)
    scheduler._clear_in_post_reply = True
    try:
        queued = getattr(scheduler, "_clear_replay_to_hold", None) or []
        scheduler._clear_replay_to_hold = None
        try:
            if not getattr(scheduler, "_clear_plan_fallback", False):
                matched = [
                    item
                    for item in queued
                    if item[3] == token and item[5] == packet
                ]
                if matched:
                    served = int(matched[0][4])
                    plan = matched[0][2]
                    same = all(int(item[4]) == served for item in matched)
                    reqs = []
                    states = []
                    for item in matched:
                        reqs.extend(item[0])
                        states.extend(item[1])
                    frozen = None
                    if same:
                        frozen = hold_replay(
                            scheduler,
                            reqs,
                            states,
                            plan,
                            token=token,
                            served=served,
                            packet=packet,
                        )
                    if (
                        isinstance(frozen, dict)
                        and frozen.get("eligible")
                        and int(frozen["served"])
                        == persistent_width(int(frozen["batch"]))
                    ):
                        record = observe_pair(frozen)
                        if record is not None:
                            queue_draft_log(scheduler, "d_wpair", record)
        except Exception:
            import logging

            logging.getLogger(__name__).warning(
                "[CLEAR ledger] replay post failed", exc_info=True
            )
        flush_draft_logs(scheduler)
    finally:
        scheduler._clear_in_post_reply = previous_post
        add_post_ms(scheduler, (time.perf_counter() - started) * 1000.0)
