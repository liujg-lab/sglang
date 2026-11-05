"""One-step TollTree width, exit, and verify-bucket decisions."""

from __future__ import annotations

from dataclasses import dataclass

from sglang.srt.speculative.toll_tree_remote.config import TollConfig
from sglang.srt.speculative.toll_tree_remote.policy.cost import CostPoint
from sglang.srt.speculative.toll_tree_remote.policy.gain import GainTable


@dataclass(frozen=True)
class LayerDecision:
    """Whether to forward the current frontier, and at which physical width."""

    grow: bool
    width: int
    gains: tuple[float, ...]
    active: tuple[int, ...]
    exited_now: tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class KChoice:
    """Verify width. ``k`` is always one of the configured buckets, or 0."""

    k: int
    score: float


def beta_tilde(cost: CostPoint, cfg: TollConfig) -> float:
    """β̃ = β1 + s1/BW. ``bytes_per_token`` converts the byte slope."""
    return float(cost.beta) + float(cost.ms_per_byte) * float(cfg.bytes_per_token)


def _bucket_at_least(buckets: tuple[int, ...], need: int) -> int:
    for bucket in buckets:
        if int(bucket) >= int(need):
            return int(bucket)
    return int(buckets[-1])


def _kth_largest(values: list[float], k: int) -> float:
    """1-based order statistic. Missing ranks contribute 0."""
    if k <= 0 or len(values) < k:
        return 0.0
    ordered = sorted(values, reverse=True)
    return float(ordered[k - 1])


def _frontier(tree) -> list:
    return [tree.nodes[i] for i in tree.frontier]


def decide_layer(trees, lam: float, cost: CostPoint, gain: GainTable, cfg: TollConfig, *, kv_ok: bool = True) -> LayerDecision:
    """Stop when the shared layer toll is not covered, a request's surplus is ≤ 0, depth, or KV.

    ``cut`` is ``max(λ·β̃, π̂_(K_max))`` on nodes already in the tree. ``F`` is
    the current layer above that cut, capped at ``W_max``. ``s`` subtracts only
    ``λ·(κ·ℓ + ε·|F|)``. The round stops when the positive surplus does not
    beat ``λ·w``.
    """
    if not kv_ok:
        return LayerDecision(False, 0, tuple(0.0 for _ in trees), (), (), "kv")
    cut_floor = float(lam) * beta_tilde(cost, cfg)
    gains = []
    surplus = []
    chosen = []
    exited_now = []
    depth_blocked = False
    for index, tree in enumerate(trees):
        if tree.exited:
            gains.append(0.0)
            surplus.append(0.0)
            chosen.append([])
            continue
        if tree.frontier and tree.nodes[tree.frontier[0]].depth >= int(cfg.max_depth):
            gains.append(0.0)
            surplus.append(0.0)
            chosen.append([])
            depth_blocked = True
            continue
        # The order statistic uses nodes already kept, not the frontier being cut.
        on_frontier = set(tree.frontier)
        existing = [
            float(node.pi)
            for index_node, node in enumerate(tree.nodes)
            if index_node not in on_frontier
        ]
        cut = max(cut_floor, _kth_largest(existing, int(cfg.k_max)))
        ranked = sorted(_frontier(tree), key=lambda node: float(node.pi), reverse=True)
        kept = [node for node in ranked if float(node.pi) > cut][: int(cfg.w_max)]
        total = 0.0
        depth = int(kept[0].depth) if kept else 0
        for node in kept:
            total += float(node.pi) * float(gain.estimate(float(node.pi), cut, depth))
        variable = float(cost.kappa) * float(tree.prefix_len) + float(cost.eps) * float(len(kept))
        score = total - float(lam) * variable
        gains.append(total)
        surplus.append(score)
        chosen.append(kept)
        if score <= 0.0 and not cfg.no_exit:
            tree.mark_exited()
            exited_now.append(index)
    active = [
        index
        for index, tree in enumerate(trees)
        if not tree.exited and chosen[index]
    ]
    positive = sum(score for score in surplus if score > 0.0)
    if not active:
        reason = "max_depth" if depth_blocked else "no_active"
        return LayerDecision(False, 0, tuple(gains), (), tuple(exited_now), reason)
    if positive <= float(lam) * float(cost.w):
        return LayerDecision(
            False, 0, tuple(gains), tuple(active), tuple(exited_now), "layer_toll"
        )
    width_need = max(len(nodes) for nodes in chosen if nodes)
    width = _bucket_at_least(tuple(cfg.width_buckets), width_need)
    return LayerDecision(
        True, width, tuple(gains), tuple(active), tuple(exited_now), "grow"
    )


def _top_mass(tree, k: int) -> float:
    ordered = sorted((float(node.pi) for node in tree.nodes), reverse=True)
    return float(sum(ordered[: min(int(k), len(ordered))]))


def choose_k(trees, lam: float, cost: CostPoint, cfg: TollConfig) -> KChoice:
    """Pick the verify bucket maximizing mass minus the linear and hinge toll."""
    if cfg.fixed_k is not None:
        return KChoice(int(cfg.fixed_k), 0.0)
    live = [tree for tree in trees if not tree.exited and tree.nodes]
    if not live:
        return KChoice(0, 0.0)
    batch = float(len(live))
    slope = beta_tilde(cost, cfg)
    best_k = 0
    best_score = float("-inf")
    for bucket in cfg.verify_buckets:
        mass = sum(_top_mass(tree, int(bucket)) for tree in live)
        extra = max(batch * float(bucket) - float(cost.node_knee), 0.0)
        score = (
            mass
            - float(lam) * slope * batch * float(bucket)
            - float(lam) * float(cost.beta_plus) * extra
        )
        if score > best_score or (score == best_score and int(bucket) < best_k):
            best_score = score
            best_k = int(bucket)
    return KChoice(best_k, best_score)


def should_speculate(lam: float, bs: int, t_ar: float, delta: float = 0.0) -> bool:
    """Speculate only when λ covers (1+δ) autoregressive tokens per second."""
    if float(t_ar) <= 0.0 or int(bs) < 1:
        return False
    return float(lam) >= (1.0 + float(delta)) * float(bs) / float(t_ar)
