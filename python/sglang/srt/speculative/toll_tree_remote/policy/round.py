"""One TollTree round driven by callbacks. No device code."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sglang.srt.speculative.toll_tree_remote.metrics import TollMetrics
from sglang.srt.speculative.toll_tree_remote.policy.baselines import FixedPolicy
from sglang.srt.speculative.toll_tree_remote.policy.decide import choose_k, decide_layer
from sglang.srt.speculative.toll_tree_remote.policy.price import price_per_ms
from sglang.srt.speculative.toll_tree_remote.policy.warmup import WarmupSchedule
from sglang.srt.speculative.toll_tree_remote.tree.state import TollTree


@dataclass(frozen=True)
class RoundReq:
    rid: str
    prefix_len: int


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
    ghat: float = 0.0


def _as_rows(scores, ids):
    taken = np.asarray(scores, dtype=np.float64)
    tokens = np.asarray(ids, dtype=np.int64)
    if taken.ndim == 1:
        taken = taken.reshape(1, -1)
        tokens = tokens.reshape(1, -1)
    return taken, tokens


class TollRound:
    """Run one draft round and keep each request's tree under its rid.

    ``on_step`` only stores the tokens and timings from the step that just
    finished. ``run`` builds a new tree per rid. ``after_reply`` labels the
    previous trees, updates φ, refreshes the cost fit, then keeps the new trees.
    """

    def __init__(self, cfg, price, cost, calib, gain, path: str) -> None:
        self.cfg = cfg
        self.price = price
        self.cost = cost
        self.calib = calib
        self.gain = gain
        self.path = path
        self.warmup = WarmupSchedule(cfg)
        self._pending: dict[str, TollTree] = {}
        self._current: dict[str, TollTree] = {}
        self._commits: dict[str, list[int]] = {}
        self.last_k = 0
        self.last_prefix_sum = 0.0
        self.last_result: RoundResult | None = None
        self.last_lam = 0.0
        self.last_net_ms = 0.0
        self.last_target_ms = 0.0
        self.metrics = TollMetrics()

    def on_step(
        self,
        now_ns,
        committed_by_rid,
        bs_prev,
        continuous,
        net_ms,
        net_bytes,
        target_ms,
    ) -> None:
        """Store the finished step. Calibration waits for ``after_reply``."""
        self._commits = {
            str(rid): [int(token) for token in tokens]
            for rid, tokens in dict(committed_by_rid).items()
        }
        committed = sum(len(tokens) for tokens in self._commits.values())
        self.price.on_step(int(now_ns), int(committed), int(bs_prev), bool(continuous))
        self.last_net_ms = float(net_ms)
        self.last_target_ms = float(target_ms)
        if self.last_k > 0:
            rows = float(bs_prev) * float(self.last_k)
            self.cost.observe_target(self.last_prefix_sum, rows, float(target_ms))
        self.cost.observe_net(float(net_bytes), float(net_ms))

    def run(self, reqs, root_topk, forward, kv_ok: bool = True) -> RoundResult:
        ordered = [req if isinstance(req, RoundReq) else RoundReq(str(req[0]), int(req[1])) for req in reqs]
        self._current = {req.rid: TollTree(req.prefix_len) for req in ordered}
        self.last_prefix_sum = float(sum(req.prefix_len for req in ordered))
        lam = self.price.price(
            len(ordered),
            self.cost.lambda_prior(len(ordered), self.last_prefix_sum),
        )
        self.last_lam = float(lam)
        lam_ms = price_per_ms(lam)
        if self.cfg.policy == "fixed":
            result = self._run_fixed(ordered, root_topk, forward)
        elif not self.cost.ready(self.path):
            result = self._run_warmup(ordered, root_topk, forward)
        else:
            result = self._run_toll(ordered, root_topk, forward, lam_ms, kv_ok)
        self.last_k = int(result.k)
        self.last_result = result
        return result

    def after_reply(self) -> None:
        actual = 0
        for rid, tree in self._pending.items():
            tokens = self._commits.get(rid, [])
            actual += len(tokens)
            self.calib.observe(tree, tokens)
            self._observe_phi(tree)
        self.cost.refresh()
        self._log(float(actual))
        self._pending = self._current
        self._current = {}

    def _trees(self, reqs) -> list[TollTree]:
        return [self._current[req.rid] for req in reqs]

    def _log(self, actual_gain: float) -> None:
        result = self.last_result
        if result is None:
            return
        reasons: dict[str, int] = {}
        for layer in result.layers:
            reasons[layer.stop_reason] = reasons.get(layer.stop_reason, 0) + 1
        self.metrics.record(
            lam=self.last_lam,
            elapsed_ms=sum(layer.ms for layer in result.layers) + self.last_target_ms + self.last_net_ms,
            expected_alen=result.expected_alen,
            ghat=result.ghat,
            actual_gain=actual_gain,
            layers=result.layers,
            k=result.k,
            stop_reasons=reasons,
            rtt_ms=self.last_net_ms,
            target_ms=self.last_target_ms,
            ready=self.cost.ready(self.path),
            cost=self.cost.snapshot(self.path),
        )

    def _alen(self, reqs) -> float:
        mass = 0.0
        for tree in self._trees(reqs):
            if len(tree):
                mass += float(np.sum(tree.pi))
        return float(len(reqs)) + mass

    def _run_fixed(self, reqs, root_topk, forward) -> RoundResult:
        width = int(self.cfg.fixed_width)
        tau = float(self.cfg.fixed_tau)
        policies = []
        for req, top in zip(reqs, root_topk):
            policy = FixedPolicy(width, tau, int(self.cfg.max_depth))
            policy.seed_topk(top[0], top[1])
            self._install_fixed_seed(self._current[req.rid], policy)
            policies.append(policy)
        layers: list[LayerRecord] = []
        guard = 0
        while any(policy.should_grow() for policy in policies) and guard < int(self.cfg.max_depth):
            guard += 1
            outputs = forward(self._trees(reqs), [tree.frontier for tree in self._trees(reqs)], width)
            grew = False
            layer_ms = 0.0
            for policy, req, output in zip(policies, reqs, outputs):
                if not policy.should_grow():
                    continue
                scores, ids, ms = output
                policy.expand_topk(scores, ids)
                self._install_fixed_children(self._current[req.rid], policy, scores, ids)
                layer_ms = max(layer_ms, float(ms))
                grew = True
            if not grew:
                break
            rows = len(reqs) * width
            self.cost.observe_layer(self.path, self.last_prefix_sum, float(rows), layer_ms)
            layers.append(LayerRecord(rows, len(reqs), layer_ms, "grow", 0.0, width))
        verify = [policy.finish() for policy in policies]
        return RoundResult(verify, width, layers, self._alen(reqs), 0.0)

    def _run_warmup(self, reqs, root_topk, forward) -> RoundResult:
        width, bucket = self.warmup.next()
        tau = 0.0 if self.cfg.fixed_tau is None else float(self.cfg.fixed_tau)
        policies = []
        for req, top in zip(reqs, root_topk):
            policy = FixedPolicy(int(width), tau, int(self.cfg.max_depth))
            policy.seed_topk(top[0], top[1])
            self._install_fixed_seed(self._current[req.rid], policy)
            policies.append(policy)
        layers: list[LayerRecord] = []
        guard = 0
        while any(policy.should_grow() for policy in policies) and guard < int(self.cfg.max_depth):
            guard += 1
            outputs = forward(self._trees(reqs), [tree.frontier for tree in self._trees(reqs)], width)
            grew = False
            layer_ms = 0.0
            for policy, req, output in zip(policies, reqs, outputs):
                if not policy.should_grow():
                    continue
                scores, ids, ms = output
                policy.expand_topk(scores, ids)
                self._install_fixed_children(self._current[req.rid], policy, scores, ids)
                layer_ms = max(layer_ms, float(ms))
                grew = True
            if not grew:
                break
            rows = len(reqs) * int(width)
            self.cost.observe_layer(self.path, self.last_prefix_sum, float(rows), layer_ms)
            layers.append(LayerRecord(rows, len(reqs), layer_ms, "warmup", 0.0, int(width)))
        verify = [policy.finish() for policy in policies]
        return RoundResult(verify, int(bucket), layers, self._alen(reqs), 0.0)

    def _run_toll(self, reqs, root_topk, forward, lam_ms: float, kv_ok: bool) -> RoundResult:
        limit = int(self.cfg.child_topk)
        trees = self._trees(reqs)
        for tree, top in zip(trees, root_topk):
            scores = np.asarray(top[0], dtype=np.float64).reshape(-1)[:limit]
            ids = np.asarray(top[1], dtype=np.int64).reshape(-1)[:limit]
            mapped = self.calib.map(scores, depth=1)
            for score, token, cond in zip(scores, ids, mapped):
                tree.add_child(-1, int(token), float(score), float(cond))
            tree.set_frontier(list(range(len(tree))))
        layers: list[LayerRecord] = []
        ghat = 0.0
        for _ in range(int(self.cfg.max_depth)):
            point = self.cost.snapshot(self.path)
            decision = decide_layer(trees, lam_ms, point, self.gain, self.cfg, kv_ok=kv_ok)
            ghat += float(sum(decision.gains))
            if not decision.grow:
                layers.append(LayerRecord(0, len(decision.active), 0.0, decision.reason, float(sum(decision.gains)), 0))
                break
            outputs = forward(trees, decision.forward, decision.width)
            layer_ms = 0.0
            for tree, kept, cut, output in zip(trees, decision.forward, decision.cuts, outputs):
                if tree.exited or not kept:
                    continue
                scores, ids, ms = output
                layer_ms = max(layer_ms, float(ms))
                self._admit_children(tree, kept, scores, ids, float(cut), int(decision.width))
            rows = len(decision.active) * int(decision.width)
            self.cost.observe_layer(self.path, self.last_prefix_sum, float(rows), layer_ms)
            layers.append(LayerRecord(rows, len(decision.active), layer_ms, "grow", float(sum(decision.gains)), int(decision.width)))
        choice = choose_k(trees, lam_ms, self.cost.snapshot(self.path), self.cfg)
        verify = [tree.finish(min(int(choice.k), len(tree))) for tree in trees]
        return RoundResult(verify, int(choice.k), layers, self._alen(reqs), ghat)

    def _admit_children(self, tree, kept, scores, ids, cut: float, limit: int) -> None:
        taken, tokens = _as_rows(scores, ids)
        if taken.shape[0] != len(kept):
            raise ValueError(f"child rows {taken.shape[0]} != forwarded nodes {len(kept)}")
        raw = taken.reshape(-1)
        token_ids = tokens.reshape(-1)
        parents = np.repeat(np.asarray(kept, dtype=np.int64), taken.shape[1])
        depths = np.asarray([int(tree.depth[int(parent)]) + 1 for parent in parents], dtype=np.int64)
        mapped = np.concatenate(self.calib.map_batch(list(taken), list(depths[:: taken.shape[1]])))
        admitted = tree.filter_candidates(parents, token_ids, raw, mapped, cut, limit)
        tree.set_frontier(admitted.tolist())

    def _mapped_group(self, tree: TollTree, parent: int, raw_row) -> np.ndarray:
        depth = 1 if parent < 0 else int(tree.depth[parent]) + 1
        return self.calib.map(raw_row, depth)

    def _install_fixed_seed(self, tree: TollTree, policy: FixedPolicy) -> None:
        raw = np.asarray(policy.state.weight[0], dtype=np.float64)
        tokens = np.asarray(policy.state.tokens[0], dtype=np.int64)
        mapped = self._mapped_group(tree, -1, raw)
        for score, token, cond in zip(raw, tokens, mapped):
            tree.add_child(-1, int(token), float(score), float(cond))
        tree.set_frontier(list(range(len(tree))))

    def _install_fixed_children(self, tree: TollTree, policy: FixedPolicy, scores, ids) -> None:
        state = policy.state
        depth = int(state.depth) - 1
        previous = [int(node) for node in tree.frontier.tolist()]
        taken, tokens = _as_rows(scores, ids)
        chosen = {}
        for col in range(state.width):
            key = (int(state.parents[depth, col]), int(state.tokens[depth, col]))
            chosen[key] = chosen.get(key, 0) + 1
        fresh = []
        for parent_col in range(taken.shape[0]):
            parent = previous[parent_col]
            mapped = self._mapped_group(tree, parent, taken[parent_col])
            for raw, token, cond in zip(taken[parent_col], tokens[parent_col], mapped):
                key = (parent_col, int(token))
                if chosen.get(key, 0) > 0:
                    chosen[key] -= 1
                    fresh.append(tree.add_child(parent, int(token), float(raw), float(cond)))
                else:
                    tree.reject(parent, int(token), float(raw), float(cond))
        tree.set_frontier(fresh)

    def _observe_phi(self, tree: TollTree) -> None:
        grouped: dict[int, list[float]] = {}
        for index in range(len(tree)):
            grouped.setdefault(int(tree.parent[index]), []).append(float(tree.a[index]))
        for index in range(int(tree.rejected_a.size)):
            grouped.setdefault(int(tree.rejected_parent[index]), []).append(float(tree.rejected_a[index]))
        for parent, siblings in grouped.items():
            depth = 0 if parent < 0 else int(tree.depth[parent])
            self.gain.observe(siblings, depth)
