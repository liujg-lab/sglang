"""Fake-model TollTree rounds. No device and no serving loop."""

import unittest

import numpy as np
import torch

from sglang.srt.speculative.opt_tree_remote import OptTreeConfig, OptTreeState, update
from sglang.srt.speculative.toll_tree_remote.config import TollConfig
from sglang.srt.speculative.toll_tree_remote.policy.calib import Calibrator
from sglang.srt.speculative.toll_tree_remote.policy.cost import TierCostModel
from sglang.srt.speculative.toll_tree_remote.policy.gain import GainTable
from sglang.srt.speculative.toll_tree_remote.policy.price import PriceTracker
from sglang.srt.speculative.toll_tree_remote.policy.round import RoundReq, TollRound
from sglang.srt.speculative.toll_tree_remote.tree.state import TollTree
from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=30, suite="stage-a-test-cpu")


def _driver(cfg, cost=None, calib=None, gain=None, price=None):
    return TollRound(
        cfg,
        price or PriceTracker(),
        cost or TierCostModel(probe_interval=1, warmup_rounds=1),
        calib or Calibrator(),
        gain or GainTable(),
        "sync",
    )


def _ready_cost(layer_ms=0.2, beta=0.05):
    model = TierCostModel(probe_interval=1, warmup_rounds=1)
    for rows in (2.0, 8.0, 16.0):
        model.observe_layer("sync", 1.0, rows, layer_ms)
        model.observe_target(1.0, rows, beta * rows)
    model.refresh()
    return model


class TestTollRound(CustomTestCase):
    def test_fixed_round_matches_opt_and_uses_width_as_k(self):
        rng = np.random.default_rng(5)
        width, max_depth, vocab, tau = 8, 4, 16, 0.05
        cfg = TollConfig(policy="fixed", fixed_width=width, fixed_tau=tau, max_depth=max_depth)
        root = rng.random(vocab).astype(np.float32)
        expands = [rng.random((width, vocab)).astype(np.float32) for _ in range(max_depth)]
        order = np.argsort(-root, kind="stable")[:width]
        cursor = {"i": 0}

        def forward(trees, frontiers, asked):
            matrix = expands[cursor["i"]]
            cursor["i"] += 1
            child_order = np.argsort(-matrix, axis=-1, kind="stable")[:, :width]
            child_scores = np.take_along_axis(matrix, child_order, axis=-1)
            return [(child_scores, child_order, 1.0)]

        driver = _driver(cfg)
        driver.on_step(0, {}, bs_prev=1, continuous=True, net_ms=1.0, net_bytes=128.0, target_ms=1.0)
        result = driver.run([RoundReq("r", 4)], [(root[order], order)], forward)
        driver.after_reply()
        state = OptTreeState.create(OptTreeConfig(width=width, tau=tau, max_depth=max_depth))
        step = update(state, torch.tensor(root))
        opt_cursor = 0
        while not step.is_final:
            step = update(state, torch.tensor(expands[opt_cursor]))
            opt_cursor += 1
        verify = step.verify
        self.assertEqual(result.k, width)
        self.assertEqual(result.verify[0].tokens, tuple(int(v) for v in verify.tokens.tolist()))
        self.assertEqual(result.verify[0].parents, tuple(int(v) for v in verify.parents.tolist()))

    def test_bs1_warmup_recovers_the_layer_fee(self):
        true_w = 3.5
        cfg = TollConfig(
            policy="toll",
            fixed_tau=0.0,
            warmup_rounds=1,
            probe_interval=1,
            max_depth=3,
            child_topk=4,
        )
        cost = TierCostModel(probe_interval=1, warmup_rounds=1)
        driver = _driver(cfg, cost=cost)
        scores = np.full(32, 0.05, dtype=np.float64)
        ids = np.arange(32)

        def forward(trees, frontiers, width):
            width = int(width)
            rows = []
            for _front in frontiers:
                child_scores = np.full((width, width), 0.02, dtype=np.float64)
                child_ids = np.arange(width * width).reshape(width, width)
                rows.append((child_scores, child_ids, true_w))
            return rows

        now = 0
        nets = [(0.0, 1.0), (100.0, 2.0), (200.0, 3.0)]
        driver.on_step(now, {}, bs_prev=1, continuous=True, net_ms=1.0, net_bytes=0.0, target_ms=1.0)
        for step in range(2):
            driver.run([RoundReq("a", 4)], [(scores, ids)], forward)
            driver.after_reply()
            now += 1_000_000
            _bytes, _ms = nets[step + 1]
            driver.on_step(
                now,
                {"a": []},
                bs_prev=1,
                continuous=True,
                net_ms=_ms,
                net_bytes=_bytes,
                target_ms=1.0,
            )
            driver.after_reply()
        self.assertTrue(cost.ready("sync"))
        self.assertAlmostEqual(cost.snapshot("sync").w, true_w, places=4)
        self.assertAlmostEqual(cost.snapshot("sync").ms_per_byte, 0.01, places=4)

    def test_mapped_acceptance_differs_from_the_raw_probability(self):
        calib = Calibrator(
            alpha=1.0,
            min_samples=1,
            prior_count=0.0,
            scale_min=0.0,
            scale_max=1e6,
            edges=[1e-3, 0.5, 1.0],
        )
        seeded = TollTree(prefix_len=1)
        seeded.add_child(-1, token=1, raw_p=0.2, a=0.2)
        calib.observe(seeded, [1])
        cfg = TollConfig(policy="toll", warmup_rounds=1, probe_interval=1, max_depth=2, child_topk=2)
        driver = _driver(cfg, cost=_ready_cost(), calib=calib)
        driver.on_step(0, {}, bs_prev=1, continuous=True, net_ms=1.0, net_bytes=32.0, target_ms=1.0)
        driver.run([RoundReq("a", 1)], [(np.array([0.4]), np.array([7]))], lambda *args: [])
        tree = driver._current["a"]
        self.assertAlmostEqual(float(tree.raw_p[0]), 0.4)
        self.assertGreater(float(tree.a[0]), float(tree.raw_p[0]))

    def test_commit_tokens_follow_the_request_id(self):
        calib = Calibrator(
            alpha=1.0,
            min_samples=1,
            prior_count=0.0,
            scale_min=0.0,
            scale_max=1e6,
            edges=[1e-3, 0.5, 1.0],
        )
        cfg = TollConfig(policy="fixed", fixed_width=8, fixed_tau=10.0, max_depth=2)
        driver = _driver(cfg, calib=calib)

        def quiet(trees, frontiers, width):
            return []

        def top(token):
            scores = np.full(8, 0.01, dtype=np.float64)
            scores[0] = 0.2
            ids = np.arange(8)
            ids[0] = token
            return scores, ids

        driver.on_step(0, {}, bs_prev=2, continuous=True, net_ms=1.0, net_bytes=16.0, target_ms=1.0)
        driver.run(
            [RoundReq("A", 2), RoundReq("B", 2)],
            [top(10), top(20)],
            quiet,
        )
        # tau is large, so the seed is not expanded. Plant the rejected child on A.
        driver._current["A"].reject(0, token=11, raw_p=0.01, a=0.01)
        driver.after_reply()
        driver.on_step(
            1_000_000,
            {"A": [10, 11], "B": [20]},
            bs_prev=2,
            continuous=True,
            net_ms=1.0,
            net_bytes=16.0,
            target_ms=1.0,
        )
        driver.run(
            [RoundReq("A", 2), RoundReq("C", 2)],
            [top(50), top(60)],
            quiet,
        )
        driver.after_reply()
        self.assertAlmostEqual(float(calib.map([0.01], depth=2)[0]), 1.0)

    def test_rejected_real_token_is_labeled_one(self):
        calib = Calibrator(
            alpha=1.0,
            min_samples=1,
            prior_count=0.0,
            scale_min=0.0,
            scale_max=1e6,
            edges=[1e-3, 0.5, 1.0],
        )
        cfg = TollConfig(policy="toll", warmup_rounds=1, probe_interval=1, max_depth=3, child_topk=2)
        price = PriceTracker()
        price.on_step(0, 0, bs=1, continuous=True)
        price.on_step(10_000_000, 10, bs=1, continuous=True)
        gain = GainTable()
        gain.observe([1.0], depth=0)
        driver = _driver(cfg, cost=_ready_cost(layer_ms=0.05, beta=0.2), calib=calib, gain=gain, price=price)

        def forward(trees, frontiers, width):
            return [
                (
                    np.array([[0.9, 0.01]]),
                    np.array([[3, 99]]),
                    0.05,
                )
                for _front in frontiers
            ]

        driver.on_step(20_000_000, {}, bs_prev=1, continuous=False, net_ms=1.0, net_bytes=8.0, target_ms=1.0)
        driver.run([RoundReq("a", 1)], [(np.array([0.8]), np.array([7]))], forward)
        self.assertIn(99, driver._current["a"].rejected_token.tolist())
        driver.after_reply()
        driver.on_step(
            1_000_000,
            {"a": [7, 99]},
            bs_prev=1,
            continuous=True,
            net_ms=1.0,
            net_bytes=8.0,
            target_ms=1.0,
        )
        driver.after_reply()
        self.assertAlmostEqual(float(calib.map([0.01], depth=2)[0]), 1.0)

    def test_token_dependent_children_match_opt(self):
        width, vocab, tau, max_depth = 8, 32, 0.0, 4
        cfg = TollConfig(policy="fixed", fixed_width=width, fixed_tau=tau, max_depth=max_depth)

        def row_for(token):
            center = (int(token) + 1) % vocab
            raw = np.asarray(
                [
                    1.0 / (1.0 + abs(index - center)) + (vocab - index) * 1e-4
                    for index in range(vocab)
                ],
                dtype=np.float64,
            )
            return (raw / raw.sum()).astype(np.float32)

        def matrix_for(parent_tokens):
            return np.stack([row_for(token) for token in parent_tokens])

        def top_rows(matrix):
            order = np.argsort(-matrix, axis=-1, kind="stable")[:, :width]
            scores = np.take_along_axis(matrix, order, axis=-1)
            return scores, order

        root = row_for(3)
        root_order = np.argsort(-root, kind="stable")[:width]

        def forward(trees, frontiers, asked):
            tree = trees[0]
            parents = [int(tree.token[int(node)]) for node in list(frontiers[0])]
            self.assertEqual(len(parents), width)
            return [(*top_rows(matrix_for(parents)), 1.0)]

        driver = _driver(cfg)
        driver.on_step(0, {}, bs_prev=1, continuous=True, net_ms=1.0, net_bytes=0.0, target_ms=1.0)
        result = driver.run([RoundReq("r", 4)], [(root[root_order], root_order)], forward)
        state = OptTreeState.create(OptTreeConfig(width=width, tau=tau, max_depth=max_depth))
        step = update(state, torch.tensor(root))
        while not step.is_final:
            parents = state.tokens[state.depth - 1].tolist()
            step = update(state, torch.tensor(matrix_for(parents)))
        verify = step.verify
        self.assertEqual(result.verify[0].tokens, tuple(int(v) for v in verify.tokens.tolist()))
        self.assertEqual(result.verify[0].parents, tuple(int(v) for v in verify.parents.tolist()))

    def test_target_cost_uses_the_previous_round(self):
        cfg = TollConfig(policy="fixed", fixed_width=8, fixed_tau=10.0, max_depth=2)
        cost = TierCostModel(probe_interval=1, warmup_rounds=0)
        driver = _driver(cfg, cost=cost)
        scores = np.full(8, 0.1, dtype=np.float64)
        ids = np.arange(8)
        driver.on_step(0, {}, bs_prev=1, continuous=True, net_ms=0.0, net_bytes=0.0, target_ms=0.0)
        driver.run([RoundReq("a", 3)], [(scores, ids)], lambda *args: [])
        driver.on_step(
            1_000_000,
            {"a": []},
            bs_prev=1,
            continuous=True,
            net_ms=1.0,
            net_bytes=0.0,
            target_ms=2.5,
        )
        prefix, rows, ms = cost._target[-1]
        self.assertEqual(prefix, 3.0)
        self.assertEqual(rows, 8.0)
        self.assertEqual(ms, 2.5)


if __name__ == "__main__":
    unittest.main()
