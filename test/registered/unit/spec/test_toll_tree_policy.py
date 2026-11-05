"""CPU checks for the TollTree price, cost, and tree rules.

Hand example, one request, φ = 1, λ = 1, w = 0.25, κ = 0, ε = 0, β̃ = 0.005.
Path masses are milli-units divided by 1000. Each layer is eight nodes, so the
physical width is 8. Children stay at or below their parent.

Layer 1 sums to 0.739 and is forwarded.
Layer 2 sums to 0.586 and is forwarded.
Layer 3 sums to 0.415 and is forwarded.
Layer 4 sums to 0.246, and 0.246 ≤ 0.25, so the fourth judgment stops.
The 32 path masses then pick verify bucket 32:
top-8 mass 0.980 scores 0.940, top-16 mass 1.475 scores 1.395,
all 32 mass 1.986 scores 1.826.

K_max = 16 uses the 16th previous mass 0.036 as the cut on layer 3, leaving
0.330. Two requests show one exit and one survivor.
"""

import unittest

import numpy as np
import torch

from sglang.srt.speculative.opt_tree_remote import (
    OptTreeConfig,
    OptTreeState,
    update,
)
from sglang.srt.speculative.toll_tree_remote.config import TollConfig
from sglang.srt.speculative.toll_tree_remote.policy.baselines import fixed_verify
from sglang.srt.speculative.toll_tree_remote.policy.calib import Calibrator
from sglang.srt.speculative.toll_tree_remote.policy.cost import CostPoint, TierCostModel
from sglang.srt.speculative.toll_tree_remote.policy.decide import choose_k, decide_layer, should_speculate
from sglang.srt.speculative.toll_tree_remote.policy.gain import GainTable
from sglang.srt.speculative.toll_tree_remote.policy.price import PriceTracker
from sglang.srt.speculative.toll_tree_remote.tree.state import TollTree
from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=30, suite="stage-a-test-cpu")

_L1 = (200, 150, 100, 90, 70, 50, 40, 39)
_L2 = (150, 110, 80, 70, 55, 45, 40, 36)
_L3 = (100, 80, 60, 50, 40, 35, 30, 20)
_L4 = (60, 50, 40, 30, 25, 20, 12, 9)


def _milli(values):
    return [v / 1000.0 for v in values]


def _hand_cost() -> CostPoint:
    return CostPoint(
        w=0.25,
        kappa=0.0,
        eps=0.0,
        eps_plus=0.0,
        row_knee=10**9,
        beta=0.005,
        beta_plus=0.0,
        node_knee=10**9,
        rtt_ms=0.0,
        ms_per_byte=0.0,
    )


def _add_layer(tree: TollTree, parents: list[int], millis, token_base: int) -> list[int]:
    indices = []
    for slot, milli in enumerate(millis):
        target = milli / 1000.0
        parent = parents[slot]
        base = 1.0 if parent < 0 else tree.nodes[parent].pi
        indices.append(tree.add_child(parent, token_base + slot, target / base))
    tree.set_frontier(indices)
    return indices


def _hand_tree() -> TollTree:
    tree = TollTree(prefix_len=4)
    roots = [-1] * 8
    layer1 = _add_layer(tree, roots, _L1, 100)
    layer2 = _add_layer(tree, layer1, _L2, 200)
    layer3 = _add_layer(tree, layer2, _L3, 300)
    _add_layer(tree, layer3, _L4, 400)
    return tree


class TestTollPrice(CustomTestCase):
    def test_lambda_is_ratio_of_means(self):
        # Two counted steps, 10 tokens in 1s and 10 tokens in 9s.
        # E[tokens] / E[dt] = 10 / 5 = 2. E[tokens/dt] would be about 5.556.
        tracker = PriceTracker(alpha=0.02)
        tracker.on_step(0, 0, bs=1, continuous=True)
        tracker.on_step(1_000_000_000, 10, bs=1, continuous=True)
        tracker.on_step(10_000_000_000, 10, bs=1, continuous=True)
        self.assertAlmostEqual(tracker.price(1, model_prior=0.0), 2.0)

    def test_idle_gap_and_new_bucket_prior(self):
        tracker = PriceTracker()
        tracker.on_step(0, 0, bs=1, continuous=True)
        tracker.on_step(1_000_000_000, 10, bs=1, continuous=True)
        tracker.on_step(10_000_000_000, 10, bs=1, continuous=True)
        tracker.on_step(50_000_000_000, 100, bs=1, continuous=False)
        self.assertAlmostEqual(tracker.price(1, model_prior=0.0), 2.0)
        self.assertAlmostEqual(tracker.price(2, model_prior=2.0), 2.0)

    def test_anomaly_and_level_switch(self):
        tracker = PriceTracker(alpha=0.02, min_anomaly_samples=8)
        tracker.on_step(0, 0, bs=1, continuous=True)
        now = 0
        for _ in range(8):
            now += 1_000_000_000
            tracker.on_step(now, 10, bs=1, continuous=True)
        now += 10_000_000_000
        tracker.on_step(now, 10, bs=1, continuous=True)
        self.assertAlmostEqual(tracker.price(1, model_prior=0.0), 10.0)

        switched = PriceTracker(alpha=1.0, persist=2, min_anomaly_samples=100)
        switched.on_step(0, 0, bs=1, continuous=True)
        switched.on_step(1_000_000_000, 10, bs=1, continuous=True)
        switched.on_step(21_000_000_000, 10, bs=1, continuous=True)
        switched.on_step(41_000_000_000, 10, bs=1, continuous=True)
        self.assertAlmostEqual(switched.price(1, model_prior=0.0), 0.5)


class TestTollCostAndCalibration(CustomTestCase):
    def test_layer_target_and_net_fit(self):
        model = TierCostModel()
        for prefix in range(4):
            for rows in (2, 8, 16):
                ms = 1.0 + 0.2 * prefix + 0.3 * rows
                model.observe_layer("sync", prefix, rows, ms)
        point = model.snapshot("sync")
        self.assertAlmostEqual(point.w, 1.0, places=6)
        self.assertAlmostEqual(point.kappa, 0.2, places=6)
        self.assertAlmostEqual(point.eps, 0.3, places=6)
        self.assertAlmostEqual(point.eps_plus, 0.0, places=6)

        for prefix in (0.0, 1.0, 2.0):
            for rows in (0.0, 5.0, 10.0, 15.0, 20.0):
                knee = max(rows - 10.0, 0.0)
                ms = 0.5 * prefix + 0.1 * rows + 0.4 * knee
                model.observe_target(prefix, rows, ms)
        point = model.snapshot("sync")
        self.assertAlmostEqual(point.beta, 0.1, places=5)
        self.assertAlmostEqual(point.beta_plus, 0.4, places=5)
        self.assertEqual(point.node_knee, 10)

        model.observe_net(0.0, 3.0)
        model.observe_net(10.0, 3.0 + 0.25 * 10.0)
        model.observe_net(40.0, 3.0 + 0.25 * 40.0)
        point = model.snapshot("sync")
        self.assertAlmostEqual(point.rtt_ms, 3.0, places=6)
        self.assertAlmostEqual(point.ms_per_byte, 0.25, places=6)
        with self.assertRaises(ValueError):
            model.observe_layer("missing", 0.0, 1.0, 1.0)

    def test_sibling_cap_and_censored_labels(self):
        calibrator = Calibrator()
        mapped = calibrator.map([0.8, 0.8], depth=1)
        self.assertLessEqual(float(mapped.sum()), 1.0 + 1e-12)
        self.assertAlmostEqual(float(mapped[0]), 0.5)
        tree = TollTree(prefix_len=1)
        kept = tree.add_child(-1, token=7, a=0.6)
        tree.add_child(-1, token=8, a=0.2)
        tree.add_child(kept, token=9, a=0.5)
        dropped = tree.add_child(-1, token=1, a=0.1)
        tree.add_child(dropped, token=2, a=0.4)
        calibrator.observe(tree, [7, 9])
        self.assertEqual(calibrator.labeled, 4)
        self.assertEqual(calibrator.censored, 1)
        learned = calibrator.map([1.0, 1.0], depth=1)
        self.assertLessEqual(float(learned.sum()), 1.0 + 1e-12)

    def test_gain_interpolates(self):
        table = GainTable()
        self.assertAlmostEqual(table.estimate(0.5, 0.25, depth=1), 1.0)
        table.observe(1.0, depth=1, x=0.0)
        table.observe(0.0, depth=1, x=1.0)
        self.assertAlmostEqual(table.estimate(0.4, 0.2, depth=1), 0.5)


class TestTollDecide(CustomTestCase):
    def test_hand_example_stops_on_fourth_judgment_and_picks_k_32(self):
        cfg = TollConfig()
        cost = _hand_cost()
        gain = GainTable()
        tree = TollTree(prefix_len=4)
        expected = (_L1, _L2, _L3, _L4)
        parents = [-1] * 8
        token_base = 100
        for layer, millis in enumerate(expected, start=1):
            parents = _add_layer(tree, parents, millis, token_base)
            token_base += 100
            decision = decide_layer([tree], 1.0, cost, gain, cfg)
            self.assertAlmostEqual(decision.gains[0], sum(_milli(millis)), places=9)
            if layer < 4:
                self.assertTrue(decision.grow)
                self.assertEqual(decision.width, 8)
                self.assertEqual(decision.reason, "grow")
                self.assertFalse(tree.exited)
            else:
                self.assertFalse(decision.grow)
                self.assertEqual(decision.reason, "layer_toll")
        choice = choose_k([tree], 1.0, cost, cfg)
        self.assertEqual(choice.k, 32)
        self.assertIn(choice.k, cfg.verify_buckets)

    def test_k_max_16_cuts_layer_three(self):
        cfg = TollConfig(verify_buckets=(8, 16))
        cost = _hand_cost()
        tree = TollTree(prefix_len=4)
        layer1 = _add_layer(tree, [-1] * 8, _L1, 100)
        layer2 = _add_layer(tree, layer1, _L2, 200)
        _add_layer(tree, layer2, _L3, 300)
        decision = decide_layer([tree], 1.0, cost, GainTable(), cfg)
        self.assertAlmostEqual(decision.gains[0], 0.330, places=9)
        self.assertEqual(decision.width, 8)
        self.assertTrue(decision.grow)

    def test_two_requests_one_exits(self):
        cfg = TollConfig()
        cost = CostPoint(
            w=0.25,
            kappa=0.01,
            eps=0.0,
            eps_plus=0.0,
            row_knee=10**9,
            beta=0.005,
            beta_plus=0.0,
            node_knee=10**9,
            rtt_ms=0.0,
            ms_per_byte=0.0,
        )
        first = TollTree(prefix_len=10)
        second = TollTree(prefix_len=40)
        first.set_frontier([first.add_child(-1, token=1, a=0.50)])
        second.set_frontier([second.add_child(-1, token=2, a=0.30)])
        decision = decide_layer([first, second], 1.0, cost, GainTable(), cfg)
        self.assertAlmostEqual(decision.gains[0], 0.50, places=9)
        self.assertAlmostEqual(decision.gains[1], 0.30, places=9)
        self.assertEqual(decision.exited_now, (1,))
        self.assertEqual(decision.active, (0,))
        self.assertTrue(decision.grow)
        self.assertEqual(decision.width, 2)
        self.assertTrue(second.exited)
        with self.assertRaises(RuntimeError):
            second.add_child(-1, token=3, a=0.1)
        again = decide_layer([first, second], 1.0, cost, GainTable(), cfg)
        self.assertNotIn(1, again.active)
        self.assertNotIn(1, again.exited_now)

    def test_depth_matches_brute_force_on_random_chains(self):
        cfg = TollConfig(max_depth=8)
        cost = _hand_cost()
        gain = GainTable()
        rng = np.random.default_rng(11)
        for _ in range(20):
            draw = np.sort(rng.random(5))[::-1]
            masses = 0.05 + 0.9 * draw
            tree = TollTree(prefix_len=1)
            parent = -1
            grown = 0
            marginals = []
            for depth, mass in enumerate(masses, start=1):
                base = 1.0 if parent < 0 else tree.nodes[parent].pi
                parent = tree.add_child(parent, token=depth, a=float(mass) / base)
                tree.set_frontier([parent])
                decision = decide_layer([tree], 1.0, cost, gain, cfg)
                marginals.append(decision.gains[0] - cost.w)
                if decision.grow:
                    grown += 1
                else:
                    break
            best = 0
            total = 0.0
            best_depth = 0
            for depth, marginal in enumerate(marginals, start=1):
                total += marginal
                if total > best:
                    best = total
                    best_depth = depth
            self.assertEqual(grown, best_depth)

    def test_should_speculate_threshold(self):
        self.assertFalse(should_speculate(0.5, bs=2, t_ar=1.0))
        self.assertTrue(should_speculate(2.0, bs=2, t_ar=1.0))
        self.assertFalse(should_speculate(2.0, bs=2, t_ar=1.0, delta=0.1))


class TestTollTree(CustomTestCase):
    def test_finish_is_ancestor_closed_for_every_k(self):
        tree = _hand_tree()
        for k in range(0, len(tree.nodes) + 1):
            verify = tree.finish(k)
            self.assertEqual(len(verify.tokens), k)
            for index, parent in enumerate(verify.parents):
                if parent == -1:
                    self.assertEqual(verify.depths[index], 1)
                else:
                    self.assertLess(parent, index)
                    self.assertGreaterEqual(verify.pis[parent], verify.pis[index] - 1e-12)
        with self.assertRaises(ValueError):
            tree.add_child(0, token=1, a=1.1)

    def test_config_rejects_bad_buckets_rpd_and_dp(self):
        with self.assertRaises(ValueError):
            TollConfig(width_buckets=(1, 4))
        with self.assertRaises(ValueError):
            TollConfig(verify_buckets=())
        with self.assertRaises(ValueError):
            TollConfig(verify_mode="rpd")
        with self.assertRaises(ValueError):
            TollConfig(dp_attention=True)
        with self.assertRaises(ValueError):
            TollConfig(draft_on_npu=True)


class TestFixedMatchesOpt(CustomTestCase):
    def test_random_inputs_match_opt_verify_nodes(self):
        rng = np.random.default_rng(3)
        for trial in range(6):
            width = 4
            max_depth = 6
            vocab = 24
            tau = 0.05 + 0.01 * trial
            root = rng.random(vocab).astype(np.float32)
            expands = [
                rng.random((width, vocab)).astype(np.float32) for _ in range(max_depth)
            ]
            got = fixed_verify(root, expands, width, tau, max_depth)
            state = OptTreeState.create(
                OptTreeConfig(width=width, tau=tau, max_depth=max_depth)
            )
            step = update(state, torch.tensor(root))
            cursor = 0
            while not step.is_final:
                step = update(state, torch.tensor(expands[cursor]))
                cursor += 1
            verify = step.verify
            self.assertIsNotNone(verify)
            self.assertEqual(got.tokens, tuple(int(v) for v in verify.tokens.tolist()))
            self.assertEqual(got.parents, tuple(int(v) for v in verify.parents.tolist()))


if __name__ == "__main__":
    unittest.main()
