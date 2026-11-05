"""One-step TollTree width, exit, and verify-bucket decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sglang.srt.speculative.toll_tree_remote.config import TollConfig
from sglang.srt.speculative.toll_tree_remote.policy.cost import CostPoint
from sglang.srt.speculative.toll_tree_remote.policy.gain import GainTable


@dataclass(frozen=True)
class LayerDecision:
    """Whether to forward the current frontier, and at which physical width.

    ``forward`` lists the tree indices to run when the round grows.
    ``cuts`` is the cut used for that request, including when the round stops.
    """

    grow: bool
    width: int
    gains: tuple[float, ...]
    active: tuple[int, ...]
    exited_now: tuple[int, ...]
    reason: str
    forward: tuple[tuple[int, ...], ...]
    cuts: tuple[float, ...]


@dataclass(frozen=True)
class KChoice:
    """Verify width. ``k`` is always one of the configured buckets, or 0."""

    k: int
    score: float


def beta_tilde(cost: CostPoint, cfg: TollConfig) -> float:
    """β̃ = β1 + s1/BW. ``bytes_per_token`` converts the byte slope."""
    return float(cost.beta) + float(cost.ms_per_byte) * float(cfg.bytes_per_token)


def uses_fixed_policy(model, path: str) -> bool:
    """An unready cost path is not a decision input. The caller uses ``fixed``."""
    return not model.ready(path)


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


def _layer_toll(lam: float, cost: CostPoint, n_active: int, width: int) -> float:
    """λ·w + λ·ε⁺·(rows − M̂)⁺. Rows are active requests times the width bucket."""
    rows = float(n_active) * float(width)
    hinge = max(rows - float(cost.row_knee), 0.0)
    return float(lam) * float(cost.w) + float(lam) * float(cost.eps_plus) * hinge


def _empty_forward(n: int) -> tuple[tuple[int, ...], ...]:
    return tuple(() for _ in range(n))


def decide_layer(
    trees,
    lam: float,
    cost: CostPoint,
    gain: GainTable,
    cfg: TollConfig,
    *,
    kv_ok: bool = True,
) -> LayerDecision:
    """Stop when the shared layer toll is not covered, a request's surplus is ≤ 0, depth, or KV.

    ``cut`` is ``max(λ·β̃, π̂_(K_max))`` over every node already in the tree,
    including the frontier being cut. ``F`` is the current layer above that
    cut, capped at ``W_max``. ``s`` subtracts only ``λ·(κ·ℓ + ε·|F|)``. The
    round stops when the positive surplus does not beat
    ``λ·w + λ·ε⁺·(rows − M̂)⁺``.
    """
    n = len(trees)
    if not kv_ok:
        return LayerDecision(
            False,
            0,
            tuple(0.0 for _ in trees),
            (),
            (),
            "kv",
            _empty_forward(n),
            tuple(0.0 for _ in trees),
        )
    cut_floor = float(lam) * beta_tilde(cost, cfg)
    gains = []
    surplus = []
    chosen = []
    cuts = []
    exited_now = []
    depth_blocked = False
    for index, tree in enumerate(trees):
        if tree.exited:
            gains.append(0.0)
            surplus.append(0.0)
            chosen.append([])
            cuts.append(0.0)
            continue
        if tree.frontier.size and int(tree.depth[int(tree.frontier[0])]) >= int(cfg.max_depth):
            gains.append(0.0)
            surplus.append(0.0)
            chosen.append([])
            cuts.append(0.0)
            depth_blocked = True
            continue
        masses = [float(v) for v in tree.pi.tolist()]
        cut = max(cut_floor, _kth_largest(masses, int(cfg.k_max)))
        ranked = sorted(
            (int(i) for i in tree.frontier.tolist()),
            key=lambda i: float(tree.pi[i]),
            reverse=True,
        )
        kept = [i for i in ranked if float(tree.pi[i]) > cut][: int(cfg.w_max)]
        total = 0.0
        depth = int(tree.depth[kept[0]]) if kept else 0
        for node in kept:
            pi = float(tree.pi[node])
            total += pi * float(gain.estimate(pi, cut, depth))
        variable = float(cost.kappa) * float(tree.prefix_len) + float(cost.eps) * float(len(kept))
        score = total - float(lam) * variable
        gains.append(total)
        surplus.append(score)
        chosen.append(kept)
        cuts.append(cut)
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
        return LayerDecision(
            False,
            0,
            tuple(gains),
            (),
            tuple(exited_now),
            reason,
            _empty_forward(n),
            tuple(cuts),
        )
    width_need = max(len(chosen[index]) for index in active)
    width = _bucket_at_least(tuple(cfg.width_buckets), width_need)
    toll = _layer_toll(lam, cost, len(active), width)
    if positive <= toll:
        return LayerDecision(
            False,
            0,
            tuple(gains),
            tuple(active),
            tuple(exited_now),
            "layer_toll",
            _empty_forward(n),
            tuple(cuts),
        )
    forward = tuple(tuple(chosen[index]) if index in active else () for index in range(n))
    return LayerDecision(
        True,
        width,
        tuple(gains),
        tuple(active),
        tuple(exited_now),
        "grow",
        forward,
        tuple(cuts),
    )


def _top_mass(tree, k: int) -> float:
    if len(tree) == 0:
        return 0.0
    ordered = np.sort(np.asarray(tree.pi, dtype=np.float64))[::-1]
    return float(ordered[: min(int(k), ordered.size)].sum())


def choose_k(trees, lam: float, cost: CostPoint, cfg: TollConfig) -> KChoice:
    """Pick the verify bucket maximizing mass minus the linear and hinge toll.

    ``b`` is ``len(trees)``, including requests that already exited. Their
    nodes still contribute path mass. An empty forest returns K = 0.
    """
    if cfg.fixed_k is not None:
        return KChoice(int(cfg.fixed_k), 0.0)
    if not trees or not any(len(tree) for tree in trees):
        return KChoice(0, 0.0)
    batch = float(len(trees))
    slope = beta_tilde(cost, cfg)
    best_k = 0
    best_score = float("-inf")
    for bucket in cfg.verify_buckets:
        mass = sum(_top_mass(tree, int(bucket)) for tree in trees)
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
