"""CPU checks for the TollTree price, cost, and tree rules.

Hand example, one request, λ = 1, w = 0.25, κ = 0, ε = 0, β̃ = 0.03125.
φ is installed per depth, not left at the empty-table prior of 0.

Layers 1–3 each have eight nodes at π̂ = 0.25. Fewer than K_max = 32 nodes
are present, so the cut is the floor λ·β̃ = 0.03125 and x = cut/π̂ = 0.125.
φ(0.125) is 0.739/2, 0.586/2, 0.415/2. Eight nodes give
ĝ = 8 · 0.25 · φ = 0.739, 0.586, 0.415. Each beats 0.25, so those layers
forward at width 8.

Layer 4 adds seven nodes at 0.25 and one at 0.0625. The tree then holds 32
nodes, so the order statistic includes this layer and the 32nd mass is
0.0625. The cut rises to 0.0625. The seven nodes above it have
x = 0.25 and φ(0.25) = 0.246/1.75, so ĝ = 1.75 · φ = 0.246. That does not
beat 0.25, and the fourth judgment stops.

Verify scores at β̃ = 0.03125, b = 1:
top-8 mass 2 scores 1.75, top-16 mass 4 scores 3.5,
all 32 mass 7.8125 scores 6.8125, so K = 32.

K_max = 16 on one layer of 8×0.25, 8×0.125, 4×0.0625: the 16th mass is
0.125, eight nodes of 0.25 remain, φ(0.5) = 0.5, and ĝ = 1.
"""

import time
import unittest

import numpy as np
import torch

from sglang.srt.speculative.opt_tree_remote import (
    OptTreeConfig,
    OptTreeState,
    update,
)
from sglang.srt.speculative.toll_tree_remote.config import TollConfig
from sglang.srt.speculative.toll_tree_remote.policy.baselines import (
    FixedPolicy,
    fixed_verify,
)
from sglang.srt.speculative.toll_tree_remote.policy.calib import Calibrator
from sglang.srt.speculative.toll_tree_remote.policy.cost import CostPoint, TierCostModel
from sglang.srt.speculative.toll_tree_remote.policy.decide import (
    choose_k,
    decide_layer,
    should_speculate,
    uses_fixed_policy,
)
from sglang.srt.speculative.toll_tree_remote.policy.gain import GainTable
from sglang.srt.speculative.toll_tree_remote.policy.price import PriceTracker
from sglang.srt.speculative.toll_tree_remote.tree.state import TollTree
from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=30, suite="stage-a-test-cpu")

_HAND_GRID = np.array([0.0, 0.014, 0.05, 0.1, 0.125, 0.25, 0.5, 1.0])
_PI = 0.25
_BETA = 0.03125
_LAYER_GAINS = (0.739, 0.586, 0.415, 0.246)
_LAYER_X = (0.125, 0.125, 0.125, 0.25)
_LAYER_PHI = (0.739 / 2.0, 0.586 / 2.0, 0.415 / 2.0, 0.246 / 1.75)


def _hand_cost() -> CostPoint:
    return CostPoint(
        w=0.25,
        kappa=0.0,
        eps=0.0,
        eps_plus=0.0,
        row_knee=10**9,
        beta=_BETA,
        beta_plus=0.0,
        node_knee=10**9,
        rtt_ms=0.0,
        ms_per_byte=0.0,
    )


def _hand_gain() -> GainTable:
    table = GainTable(grid=_HAND_GRID)
    for depth, (x, phi) in enumerate(zip(_LAYER_X, _LAYER_PHI), start=1):
        table.observe([phi + x], depth)
    return table


def _add_equal(tree: TollTree, parents: list[int], pi: float, token_base: int) -> list[int]:
    indices = []
    for slot, parent in enumerate(parents):
        base = 1.0 if parent < 0 else float(tree.pi[parent])
        cond = float(pi) / base
        indices.append(tree.add_child(parent, token_base + slot, cond, cond))
    tree.set_frontier(indices)
    return indices


def _hand_tree() -> TollTree:
    tree = TollTree(prefix_len=4)
    layer = _add_equal(tree, [-1] * 8, _PI, 100)
    layer = _add_equal(tree, layer, _PI, 200)
    layer = _add_equal(tree, layer, _PI, 300)
    kept = []
    for slot in range(7):
        kept.append(tree.add_child(layer[slot], 400 + slot, 1.0, 1.0))
    kept.append(tree.add_child(layer[7], 407, 0.25, 0.25))
    tree.set_frontier(kept)
    return tree


def _unit_gain(*depths: int) -> GainTable:
    table = GainTable(grid=_HAND_GRID)
    for depth in depths:
        table.observe([1.0], depth)
    return table


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

    def test_persist_anomalies_adopt_the_new_level(self):
        tracker = PriceTracker(persist=3, min_anomaly_samples=4)
        tracker.on_step(0, 0, bs=1, continuous=True)
        now = 0
        for _ in range(4):
            now += 1_000_000_000
            tracker.on_step(now, 10, bs=1, continuous=True)
        self.assertAlmostEqual(tracker.price(1, model_prior=0.0), 10.0)
        for _ in range(2):
            now += 10_000_000_000
            tracker.on_step(now, 10, bs=1, continuous=True)
            self.assertAlmostEqual(tracker.price(1, model_prior=0.0), 10.0)
        now += 10_000_000_000
        tracker.on_step(now, 10, bs=1, continuous=True)
        self.assertAlmostEqual(tracker.price(1, model_prior=0.0), 1.0)

    def test_global_clock_charges_the_interval_to_the_current_batch(self):
        tracker = PriceTracker()
        tracker.on_step(0, 0, bs=1, continuous=True)
        tracker.on_step(1_000_000_000, 10, bs=1, continuous=True)
        tracker.on_step(2_000_000_000, 10, bs=2, continuous=True)
        tracker.on_step(12_000_000_000, 10, bs=1, continuous=True)
        self.assertAlmostEqual(tracker.price(2, model_prior=0.0), 10.0)
        # b=1 sees 1s then 10s. A per-bucket clock would have charged 11s.
        self.assertAlmostEqual(tracker.price(1, model_prior=0.0), 20.0 / 11.0)


class TestTollCostAndCalibration(CustomTestCase):
    def test_layer_target_and_net_fit(self):
        model = TierCostModel(probe_interval=1, warmup_rounds=1)
        for prefix in range(4):
            for rows in (2, 8, 16):
                ms = 1.0 + 0.2 * prefix + 0.3 * rows
                model.observe_layer("sync", prefix, rows, ms)
        point = model.snapshot("sync")
        self.assertAlmostEqual(point.w, 1.0, places=6)
        self.assertAlmostEqual(point.kappa, 0.2, places=6)
        self.assertAlmostEqual(point.eps, 0.3, places=6)
        self.assertAlmostEqual(point.eps_plus, 0.0, places=6)
        self.assertEqual(point.row_knee, 10**9)

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

    def test_ready_clamp_and_insignificant_knee(self):
        cold = TierCostModel(probe_interval=1, warmup_rounds=8)
        for rows in (0.0, 2.0, 4.0):
            cold.observe_layer("sync", 0.0, rows, 1.0 + 0.3 * rows)
        self.assertFalse(cold.ready("sync"))
        self.assertTrue(uses_fixed_policy(cold, "sync"))

        negative = TierCostModel(probe_interval=1, warmup_rounds=1)
        for rows in (0.0, 2.0, 4.0, 6.0, 8.0):
            negative.observe_layer("eager", 0.0, rows, 1.0 - 0.2 * rows)
        point = negative.snapshot("eager")
        self.assertAlmostEqual(point.eps, 0.0)
        self.assertGreaterEqual(point.w, 0.0)
        self.assertGreaterEqual(point.kappa, 0.0)
        self.assertGreaterEqual(point.eps_plus, 0.0)

        noisy = TierCostModel(probe_interval=1, warmup_rounds=1)
        for rows in range(8):
            noisy.observe_layer("batch", 0.0, float(rows), 1.0 + 0.3 * rows)
        point = noisy.snapshot("batch")
        self.assertEqual(point.row_knee, 10**9)
        self.assertAlmostEqual(point.eps, 0.3, places=6)
        rng = np.random.default_rng(0)
        shaken = TierCostModel(probe_interval=1, warmup_rounds=1)
        for prefix in range(3):
            for level in np.linspace(0.0, 20.0, 10):
                ms = 1.0 + 0.2 * prefix + 0.3 * level + float(rng.normal(0.0, 0.01))
                shaken.observe_layer("lookahead_cascade", float(prefix), float(level), ms)
        shaken_point = shaken.snapshot("lookahead_cascade")
        self.assertEqual(shaken_point.row_knee, 10**9)
        self.assertAlmostEqual(shaken_point.eps, 0.3, places=3)

        cached = TierCostModel(probe_interval=4, warmup_rounds=4)
        samples = ((0.0, 2.0), (0.0, 8.0), (0.0, 16.0))
        for prefix, rows in samples:
            cached.observe_layer("sync", prefix, rows, 1.0 + 0.2 * prefix + 0.3 * rows)
        self.assertFalse(cached.ready("sync"))
        self.assertEqual(cached.snapshot("sync").w, 0.0)
        cached.observe_layer("sync", 1.0, 4.0, 1.0 + 0.2 * 1.0 + 0.3 * 4.0)
        self.assertTrue(cached.ready("sync"))
        self.assertFalse(uses_fixed_policy(cached, "sync"))
        fitted = cached.snapshot("sync")
        self.assertAlmostEqual(fitted.w, 1.0, places=5)
        self.assertAlmostEqual(fitted.kappa, 0.2, places=5)
        self.assertAlmostEqual(fitted.eps, 0.3, places=5)
        cached.observe_layer("sync", 0.0, 2.0, 100.0)
        self.assertAlmostEqual(cached.snapshot("sync").w, fitted.w)

    def test_sibling_cap_and_censored_labels(self):
        calibrator = Calibrator()
        mapped = calibrator.map([0.8, 0.8], depth=1)
        self.assertLessEqual(float(mapped.sum()), 1.0 + 1e-12)
        self.assertAlmostEqual(float(mapped[0]), 0.5)
        tree = TollTree(prefix_len=1)
        kept = tree.add_child(-1, token=7, raw_p=0.6, a=0.6)
        tree.add_child(-1, token=8, raw_p=0.2, a=0.2)
        tree.add_child(kept, token=9, raw_p=0.5, a=0.5)
        dropped = tree.add_child(-1, token=1, raw_p=0.1, a=0.1)
        tree.add_child(dropped, token=2, raw_p=0.4, a=0.4)
        calibrator.observe(tree, [7, 9])
        self.assertEqual(calibrator.labeled, 4)
        self.assertEqual(calibrator.censored, 1)
        learned = calibrator.map([1.0, 1.0], depth=1)
        self.assertLessEqual(float(learned.sum()), 1.0 + 1e-12)

    def test_upward_calibration_uses_a_and_does_not_compound(self):
        calibrator = Calibrator(alpha=1.0, edges=[1e-3, 0.5, 1.0])
        tree = TollTree(prefix_len=1)
        parent = tree.add_child(-1, token=7, raw_p=1.0, a=1.0)
        tree.add_child(parent, token=9, raw_p=0.2, a=0.2)
        calibrator.observe(tree, [7, 9])
        once = calibrator.map([0.1], depth=2)
        twice = calibrator.map([0.1], depth=2)
        self.assertAlmostEqual(float(once[0]), 0.5)
        self.assertTrue(np.allclose(once, twice))
        child = tree.add_child(parent, token=3, raw_p=0.1, a=float(once[0]))
        self.assertAlmostEqual(float(tree.pi[child]), float(tree.pi[parent]) * float(once[0]))
        self.assertNotAlmostEqual(float(tree.pi[child]), float(tree.pi[parent]) * 0.1, places=6)
        self.assertAlmostEqual(float(tree.a[child]), 0.5)
        capped = calibrator.map([0.4, 0.4], depth=2)
        self.assertLessEqual(float(capped.sum()), 1.0 + 1e-12)

    def test_gain_log_grid_ema_and_empty_prior(self):
        table = GainTable()
        self.assertNotEqual(table.bin_index(0.014), table.bin_index(0.1))
        self.assertAlmostEqual(table.estimate(0.5, 0.25, depth=1), 0.0)
        siblings = np.array([0.4, 0.3, 0.2])
        exact = GainTable(alpha=1.0)
        exact.observe(siblings, depth=3)
        for x in (0.014, 0.1):
            expected = float(np.maximum(siblings - x, 0.0).sum())
            self.assertAlmostEqual(exact.estimate(1.0, x, depth=3), expected)
        grid = exact.grid
        knot = int(np.where(np.isclose(grid, 0.1))[0][0])
        x0 = float(grid[knot])
        x1 = float(grid[knot + 1])
        mid = 0.5 * (x0 + x1)
        y0 = float(np.maximum(siblings - x0, 0.0).sum())
        y1 = float(np.maximum(siblings - x1, 0.0).sum())
        weight = (mid - x0) / (x1 - x0)
        self.assertAlmostEqual(
            exact.estimate(1.0, mid, depth=3), y0 + weight * (y1 - y0)
        )
        blended = GainTable(alpha=0.5, grid=[0.0, 0.1, 1.0])
        blended.observe([0.6], depth=1)
        blended.observe([0.2], depth=1)
        self.assertAlmostEqual(blended.estimate(1.0, 0.1, depth=1), 0.3)


class TestTollDecide(CustomTestCase):
    def test_hand_example_stops_on_fourth_judgment_and_picks_k_32(self):
        cfg = TollConfig()
        cost = _hand_cost()
        gain = _hand_gain()
        tree = TollTree(prefix_len=4)
        parents = [-1] * 8
        token_base = 100
        for layer, expected in enumerate(_LAYER_GAINS, start=1):
            if layer < 4:
                parents = _add_equal(tree, parents, _PI, token_base)
            else:
                kept = []
                for slot in range(7):
                    kept.append(tree.add_child(parents[slot], token_base + slot, 1.0, 1.0))
                kept.append(tree.add_child(parents[7], token_base + 7, 0.25, 0.25))
                tree.set_frontier(kept)
            token_base += 100
            decision = decide_layer([tree], 1.0, cost, gain, cfg)
            self.assertAlmostEqual(decision.gains[0], expected, places=9)
            if layer < 4:
                self.assertTrue(decision.grow)
                self.assertEqual(decision.width, 8)
                self.assertEqual(decision.reason, "grow")
                self.assertEqual(len(decision.forward[0]), 8)
                self.assertAlmostEqual(decision.cuts[0], _BETA, places=9)
                self.assertFalse(tree.exited)
            else:
                self.assertFalse(decision.grow)
                self.assertEqual(decision.reason, "layer_toll")
                self.assertEqual(decision.forward[0], ())
                self.assertAlmostEqual(decision.cuts[0], 0.0625, places=9)
        choice = choose_k([tree], 1.0, cost, cfg)
        self.assertEqual(choice.k, 32)
        self.assertAlmostEqual(choice.score, 6.8125, places=9)
        self.assertIn(choice.k, cfg.verify_buckets)

    def test_k_max_16_includes_the_current_layer(self):
        cfg = TollConfig(verify_buckets=(8, 16))
        cost = _hand_cost()
        gain = GainTable(grid=_HAND_GRID)
        gain.observe([1.0], depth=1)
        tree = TollTree(prefix_len=4)
        nodes = []
        for pi, count, token in ((0.25, 8, 10), (0.125, 8, 20), (0.0625, 4, 30)):
            for slot in range(count):
                nodes.append(tree.add_child(-1, token + slot, pi, pi))
        tree.set_frontier(nodes)
        decision = decide_layer([tree], 1.0, cost, gain, cfg)
        self.assertAlmostEqual(decision.cuts[0], 0.125, places=9)
        self.assertAlmostEqual(decision.gains[0], 1.0, places=9)
        self.assertEqual(len(decision.forward[0]), 8)
        self.assertEqual(decision.width, 8)
        self.assertTrue(decision.grow)

    def test_exited_request_still_selects_k(self):
        tree = _hand_tree()
        tree.mark_exited()
        choice = choose_k([tree], 1.0, _hand_cost(), TollConfig())
        self.assertEqual(choice.k, 32)
        self.assertAlmostEqual(choice.score, 6.8125, places=9)
        empty = TollTree(prefix_len=1)
        empty.mark_exited()
        wider = choose_k([tree, empty], 1.0, _hand_cost(), TollConfig())
        self.assertEqual(wider.k, 32)
        self.assertAlmostEqual(wider.score, 5.8125, places=9)

    def test_row_hinge_raises_the_layer_toll(self):
        cfg = TollConfig()
        gain = _unit_gain(1)
        tree = TollTree(prefix_len=1)
        tree.set_frontier(
            [tree.add_child(-1, token=slot, raw_p=0.05, a=0.05) for slot in range(8)]
        )
        plain = CostPoint(
            w=0.05,
            kappa=0.0,
            eps=0.0,
            eps_plus=0.0,
            row_knee=4,
            beta=0.0,
            beta_plus=0.0,
            node_knee=10**9,
            rtt_ms=0.0,
            ms_per_byte=0.0,
        )
        hinged = CostPoint(
            w=0.05,
            kappa=0.0,
            eps=0.0,
            eps_plus=0.1,
            row_knee=4,
            beta=0.0,
            beta_plus=0.0,
            node_knee=10**9,
            rtt_ms=0.0,
            ms_per_byte=0.0,
        )
        self.assertTrue(decide_layer([tree], 1.0, plain, gain, cfg).grow)
        stopped = decide_layer([tree], 1.0, hinged, gain, cfg)
        self.assertFalse(stopped.grow)
        self.assertEqual(stopped.reason, "layer_toll")
        self.assertAlmostEqual(stopped.gains[0], 0.40, places=9)

    def test_hard_request_exit_stops_the_round_when_the_other_cannot_pay(self):
        cfg = TollConfig()
        cost = CostPoint(
            w=0.25,
            kappa=0.01,
            eps=0.0,
            eps_plus=0.0,
            row_knee=10**9,
            beta=0.0,
            beta_plus=0.0,
            node_knee=10**9,
            rtt_ms=0.0,
            ms_per_byte=0.0,
        )
        gain = _unit_gain(1)
        hard = TollTree(prefix_len=100)
        easy = TollTree(prefix_len=1)
        hard.set_frontier([hard.add_child(-1, token=1, raw_p=0.20, a=0.20)])
        easy.set_frontier([easy.add_child(-1, token=2, raw_p=0.10, a=0.10)])
        decision = decide_layer([hard, easy], 1.0, cost, gain, cfg)
        self.assertEqual(decision.exited_now, (0,))
        self.assertEqual(decision.active, (1,))
        self.assertFalse(decision.grow)
        self.assertEqual(decision.reason, "layer_toll")
        self.assertTrue(hard.exited)
        with self.assertRaises(RuntimeError):
            hard.add_child(-1, token=3, raw_p=0.1, a=0.1)
        self.assertFalse(easy.exited)

    def test_two_small_surpluses_pay_the_layer_and_take_the_wider_bucket(self):
        cfg = TollConfig()
        cost = CostPoint(
            w=0.20,
            kappa=0.0,
            eps=0.0,
            eps_plus=0.0,
            row_knee=10**9,
            beta=0.0,
            beta_plus=0.0,
            node_knee=10**9,
            rtt_ms=0.0,
            ms_per_byte=0.0,
        )
        gain = _unit_gain(1)
        wide = TollTree(prefix_len=2)
        narrow = TollTree(prefix_len=2)
        wide.set_frontier(
            [wide.add_child(-1, token=slot, raw_p=0.05, a=0.05) for slot in range(3)]
        )
        narrow.set_frontier([narrow.add_child(-1, token=9, raw_p=0.15, a=0.15)])
        decision = decide_layer([wide, narrow], 1.0, cost, gain, cfg)
        self.assertAlmostEqual(decision.gains[0], 0.15, places=9)
        self.assertAlmostEqual(decision.gains[1], 0.15, places=9)
        self.assertEqual(decision.exited_now, ())
        self.assertEqual(decision.active, (0, 1))
        self.assertTrue(decision.grow)
        self.assertEqual(decision.width, 4)
        self.assertEqual(len(decision.forward[0]), 3)
        self.assertEqual(len(decision.forward[1]), 1)

    def test_depth_matches_brute_force_on_random_chains(self):
        cfg = TollConfig(max_depth=8)
        cost = _hand_cost()
        gain = _unit_gain(*range(1, 9))
        rng = np.random.default_rng(11)
        for _ in range(20):
            draw = np.sort(rng.random(5))[::-1]
            masses = 0.05 + 0.9 * draw
            tree = TollTree(prefix_len=1)
            parent = -1
            grown = 0
            marginals = []
            for depth, mass in enumerate(masses, start=1):
                base = 1.0 if parent < 0 else float(tree.pi[parent])
                cond = float(mass) / base
                parent = tree.add_child(parent, token=depth, raw_p=cond, a=cond)
                tree.set_frontier([parent])
                decision = decide_layer([tree], 1.0, cost, gain, cfg)
                marginals.append(decision.gains[0] - cost.w)
                if decision.grow:
                    grown += 1
                else:
                    break
            best = 0.0
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
        for k in range(0, len(tree) + 1):
            verify = tree.finish(k)
            self.assertEqual(len(verify.tokens), k)
            for index, parent in enumerate(verify.parents):
                if parent == -1:
                    self.assertEqual(verify.depths[index], 1)
                else:
                    self.assertLess(parent, index)
                    self.assertGreaterEqual(verify.pis[parent], verify.pis[index] - 1e-12)
        with self.assertRaises(ValueError):
            tree.add_child(0, token=1, raw_p=1.0, a=1.1)

    def test_below_cut_candidates_stay_out_of_finish(self):
        tree = TollTree(prefix_len=1)
        roots = [tree.add_child(-1, token=i, raw_p=1.0, a=1.0) for i in range(32)]
        parent_ids = np.repeat(np.asarray(roots, dtype=np.int64), 16)
        tokens = np.arange(1000, 1512)
        raw = np.full(512, 0.5)
        accepted = np.tile(np.linspace(0.95, 0.05, 16), 32)
        started = time.perf_counter()
        kept = tree.filter_candidates(parent_ids, tokens, raw, accepted, cut=0.5, limit=32)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 0.25)
        self.assertEqual(len(kept), 32)
        self.assertTrue(np.all(tree.pi[kept] > 0.5))
        self.assertEqual(int(tree.rejected_token.size), 480)
        finished = set(tree.finish(len(tree)).tokens)
        for token in tree.rejected_token.tolist():
            self.assertNotIn(int(token), finished)
        calibrator = Calibrator()
        calibrator.observe(tree, [int(tree.token[roots[0]])])
        self.assertGreater(calibrator.labeled, 0)
        self.assertGreater(int(tree.rejected_token.size), 0)

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
    def test_fixed_policy_steps_match_fixed_verify(self):
        rng = np.random.default_rng(1)
        width, max_depth, vocab, tau = 4, 5, 16, 0.02
        root = rng.random(vocab).astype(np.float32)
        expands = [rng.random((width, vocab)).astype(np.float32) for _ in range(max_depth)]
        policy = FixedPolicy(width, tau, max_depth)
        policy.seed(root)
        cursor = 0
        while policy.should_grow() and cursor < len(expands):
            policy.expand(expands[cursor])
            cursor += 1
        got = policy.finish()
        ref = fixed_verify(root, expands, width, tau, max_depth)
        self.assertEqual(got.tokens, ref.tokens)
        self.assertEqual(got.parents, ref.parents)

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
