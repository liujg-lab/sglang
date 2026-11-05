"""One TollTree round driven by callbacks. No device code."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sglang.srt.speculative.toll_tree_remote.metrics import TollMetrics
from sglang.srt.speculative.toll_tree_remote.policy.baselines import FixedPolicy
from sglang.srt.speculative.toll_tree_remote.policy.decide import choose_k, decide_layer
from sglang.srt.speculative.toll_tree_remote.policy.price import price_per_ms
from sglang.srt.speculative.toll_tree_remote.tree.state import TollTree


@dataclass
class LayerRecord:
    rows: int
    active: int
    ms: float
    stop_reason: str
    gain: float
    width: int


@dataclass
class RoundResult:
    verify: list
    k: int
    layers: list[LayerRecord] = field(default_factory=list)
    expected_alen: float = 0.0


def _clip_prob(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _seed_tree(tree: TollTree, scores, ids) -> None:
    indices = []
    for score, token in zip(np.asarray(scores).reshape(-1), np.asarray(ids).reshape(-1)):
        prob = _clip_prob(float(score))
        indices.append(tree.add_child(-1, int(token), prob, prob))
    tree.set_frontier(indices)


def _append_fixed_layer(tree: TollTree, policy: FixedPolicy) -> None:
    state = policy.state
    depth = int(state.depth) - 1
    previous = [int(node) for node in tree.frontier.tolist()]
    indices = []
    for col in range(state.width):
        parent_col = int(state.parents[depth, col])
        parent = previous[parent_col]
        path = float(state.weight[depth, col])
        base = float(tree.pi[parent])
        cond = _clip_prob(path / base if base > 0.0 else 0.0)
        indices.append(tree.add_child(parent, int(state.tokens[depth, col]), cond, cond))
    tree.set_frontier(indices)


class TollRound:
    """Run fixed warmup or toll decisions and keep one tree per request.

    ``forward(trees, frontiers, width)`` returns one ``(scores, ids, ms)`` per
    request. Scores are already top-k. Calibration and refits happen in
    ``after_reply``.
    """

    def __init__(self, cfg, price, cost, calib, gain, path: str) -> None:
        self.cfg = cfg
        self.price = price
        self.cost = cost
        self.calib = calib
        self.gain = gain
        self.path = path
        self.trees: list[TollTree] = []
        self.last_k = 0
        self.last_prefix_sum = 0.0
        self.last_result: RoundResult | None = None
        self.last_lam = 0.0
        self.last_net_ms = 0.0
        self.last_target_ms = 0.0
        self.metrics = TollMetrics()

    def on_step(self, now_ns, committed, bs, continuous, net_ms, target_ms) -> None:
        self.price.on_step(int(now_ns), int(committed), int(bs), bool(continuous))
        self.last_net_ms = float(net_ms)
        self.last_target_ms = float(target_ms)
        if self.last_k > 0:
            rows = float(bs) * float(self.last_k)
            self.cost.observe_target(self.last_prefix_sum, rows, float(target_ms))
        self.cost.observe_net(max(float(net_ms), 0.0), float(net_ms))

    def run(self, prefix_lens, root_topk, forward, kv_ok: bool = True) -> RoundResult:
        prefixes = [int(value) for value in prefix_lens]
        self.trees = [TollTree(length) for length in prefixes]
        self.last_prefix_sum = float(sum(prefixes))
        lam = self.price.price(len(prefixes), self.cost.lambda_prior(len(prefixes), self.last_prefix_sum))
        self.last_lam = float(lam)
        lam_ms = price_per_ms(lam)
        if self.cfg.policy == "toll" and self.cost.ready(self.path):
            result = self._run_toll(root_topk, forward, lam_ms, kv_ok)
        else:
            result = self._run_fixed(root_topk, forward, lam_ms)
        self.last_k = int(result.k)
        self.last_result = result
        self._log(0.0)
        return result

    def after_reply(self, committed_tokens_by_req) -> None:
        actual = 0
        for tree, committed in zip(self.trees, committed_tokens_by_req):
            tokens = [int(token) for token in committed]
            actual += len(tokens)
            self.calib.observe(tree, tokens)
            self._observe_phi(tree)
        self.cost.refresh()
        self._log(float(actual))

    def _log(self, actual_gain: float) -> None:
        result = self.last_result
        if result is None:
            return
        reasons: dict[str, int] = {}
        for layer in result.layers:
            reasons[layer.stop_reason] = reasons.get(layer.stop_reason, 0) + 1
        point = self.cost.snapshot(self.path)
        self.metrics.record(
            lam=self.last_lam,
            elapsed_ms=sum(layer.ms for layer in result.layers) + self.last_target_ms + self.last_net_ms,
            expected_alen=result.expected_alen,
            actual_gain=actual_gain,
            layers=result.layers,
            k=result.k,
            stop_reasons=reasons,
            rtt_ms=self.last_net_ms,
            target_ms=self.last_target_ms,
            ready=self.cost.ready(self.path),
            cost=point,
        )

    def _finish(self, lam_ms: float, layers: list[LayerRecord]) -> RoundResult:
        point = self.cost.snapshot(self.path)
        choice = choose_k(self.trees, lam_ms, point, self.cfg)
        limit = int(choice.k)
        verify = [tree.finish(min(limit, len(tree))) for tree in self.trees]
        return RoundResult(
            verify=verify,
            k=limit,
            layers=layers,
            expected_alen=float(sum(layer.gain for layer in layers)),
        )

    def _run_fixed(self, root_topk, forward, lam_ms: float) -> RoundResult:
        width = int(self.cfg.fixed_width or self.cfg.width_buckets[0])
        tau = 0.0 if self.cfg.fixed_tau is None else float(self.cfg.fixed_tau)
        policies = []
        for tree, top in zip(self.trees, root_topk):
            policy = FixedPolicy(width, tau, int(self.cfg.max_depth))
            scores, ids = top[0], top[1]
            policy.seed_topk(scores, ids)
            _seed_tree(tree, policy.state.weight[0], policy.state.tokens[0])
            policies.append(policy)
        width_rows = len(self.trees) * width
        self.cost.observe_layer(self.path, self.last_prefix_sum, float(width_rows), 0.0)
        layers: list[LayerRecord] = [
            LayerRecord(width_rows, len(self.trees), 0.0, "seed", 0.0, width)
        ]
        guard = 0
        while any(policy.should_grow() for policy in policies) and guard < int(self.cfg.max_depth):
            guard += 1
            outputs = forward(self.trees, [tree.frontier for tree in self.trees], width)
            grew = False
            layer_ms = 0.0
            for policy, tree, output in zip(policies, self.trees, outputs):
                if not policy.should_grow():
                    continue
                scores, ids, ms = output
                policy.expand_topk(scores, ids)
                _append_fixed_layer(tree, policy)
                layer_ms = max(layer_ms, float(ms))
                grew = True
            if not grew:
                break
            rows = len(self.trees) * width
            self.cost.observe_layer(self.path, self.last_prefix_sum, float(rows), layer_ms)
            layers.append(LayerRecord(rows, len(self.trees), layer_ms, "grow", 0.0, width))
        if not layers:
            layers.append(LayerRecord(0, 0, 0.0, "fixed", 0.0, width))
        return self._finish(lam_ms, layers)

    def _run_toll(self, root_topk, forward, lam_ms: float, kv_ok: bool) -> RoundResult:
        limit = int(self.cfg.child_topk)
        for tree, top in zip(self.trees, root_topk):
            scores = np.asarray(top[0]).reshape(-1)[:limit]
            ids = np.asarray(top[1]).reshape(-1)[:limit]
            _seed_tree(tree, scores, ids)
        layers: list[LayerRecord] = []
        for _ in range(int(self.cfg.max_depth)):
            point = self.cost.snapshot(self.path)
            decision = decide_layer(
                self.trees, lam_ms, point, self.gain, self.cfg, kv_ok=kv_ok
            )
            gain = float(sum(decision.gains))
            if not decision.grow:
                layers.append(LayerRecord(0, len(decision.active), 0.0, decision.reason, gain, 0))
                break
            outputs = forward(self.trees, decision.forward, decision.width)
            layer_ms = 0.0
            for tree, kept, output in zip(self.trees, decision.forward, outputs):
                if tree.exited:
                    continue
                scores, ids, ms = output
                layer_ms = max(layer_ms, float(ms))
                scores = np.asarray(scores, dtype=np.float64)
                ids = np.asarray(ids, dtype=np.int64)
                if scores.ndim == 1:
                    scores = scores.reshape(1, -1)
                    ids = ids.reshape(1, -1)
                fresh = []
                for local, parent in enumerate(kept):
                    for score, token in zip(scores[local], ids[local]):
                        prob = _clip_prob(float(score))
                        if prob <= 0.0:
                            continue
                        fresh.append(tree.add_child(int(parent), int(token), prob, prob))
                tree.set_frontier(fresh)
            rows = len(decision.active) * int(decision.width)
            self.cost.observe_layer(self.path, self.last_prefix_sum, float(rows), layer_ms)
            layers.append(
                LayerRecord(rows, len(decision.active), layer_ms, "grow", gain, int(decision.width))
            )
        return self._finish(lam_ms, layers)

    def _observe_phi(self, tree: TollTree) -> None:
        grouped: dict[int, list[float]] = {}
        for index in range(len(tree)):
            grouped.setdefault(int(tree.parent[index]), []).append(float(tree.a[index]))
        for parent, siblings in grouped.items():
            depth = 0 if parent < 0 else int(tree.depth[parent])
            self.gain.observe(siblings, depth)
