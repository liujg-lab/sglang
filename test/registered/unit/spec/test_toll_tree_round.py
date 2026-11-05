"""Fake-model TollTree rounds. No device and no serving loop."""

import unittest

import numpy as np

from sglang.srt.speculative.toll_tree_remote.config import TollConfig
from sglang.srt.speculative.toll_tree_remote.policy.calib import Calibrator
from sglang.srt.speculative.toll_tree_remote.policy.cost import TierCostModel
from sglang.srt.speculative.toll_tree_remote.policy.decide import choose_k
from sglang.srt.speculative.toll_tree_remote.policy.gain import GainTable
from sglang.srt.speculative.toll_tree_remote.policy.price import PriceTracker, price_per_ms
from sglang.srt.speculative.toll_tree_remote.policy.round import TollRound
from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=30, suite="stage-a-test-cpu")


def _cfg():
    return TollConfig(
        policy="toll",
        fixed_width=8,
        fixed_tau=10.0,
        warmup_rounds=1,
        probe_interval=1,
        max_depth=4,
        child_topk=4,
    )


def _root(prob=0.2, width=8):
    scores = np.full(width, prob, dtype=np.float64)
    ids = np.arange(width, dtype=np.int64)
    return scores, ids


def _children(frontier, prob=0.4):
    scores = np.full((len(frontier), 4), prob, dtype=np.float64)
    ids = np.arange(len(frontier) * 4, dtype=np.int64).reshape(len(frontier), 4)
    return scores, ids, 0.01


class TestTollRound(CustomTestCase):
    def _round(self):
        return TollRound(
            _cfg(),
            PriceTracker(),
            TierCostModel(probe_interval=1, warmup_rounds=1),
            Calibrator(min_samples=1, prior_count=0.0, scale_min=0.0, scale_max=20.0),
            GainTable(),
            "sync",
        )

    def test_warmup_fills_phi_and_toll_can_grow(self):
        driver = self._round()
        self.assertFalse(driver.cost.ready("sync"))

        def hold(trees, frontiers, width):
            return [_children(front) for front in frontiers]

        driver.run([4], [_root()], hold)
        driver.after_reply([[]])
        driver.on_step(0, 0, bs=1, continuous=True, net_ms=1.0, target_ms=1.0)
        driver.on_step(1_000_000, 1, bs=1, continuous=True, net_ms=1.0, target_ms=1.0)
        driver.run([2, 2], [_root(), _root(0.1)], hold)
        driver.on_step(2_000_000, 1, bs=2, continuous=True, net_ms=1.0, target_ms=1.0)
        driver.after_reply([[], []])
        self.assertTrue(driver.gain._ema)
        self.assertTrue(driver.cost.ready("sync"))
        grown = driver.run([4], [_root(0.5)], hold)
        self.assertTrue(any(layer.stop_reason == "grow" for layer in grown.layers))
        self.assertIn("[TOLL round]", driver.metrics.lines[-1])
        self.assertIn("ready=True", driver.metrics.lines[-1])

    def test_slower_rtt_does_not_shorten_the_tree(self):
        cfg = TollConfig(policy="toll", warmup_rounds=1, probe_interval=1, max_depth=4, child_topk=2)
        cost = TierCostModel(probe_interval=1, warmup_rounds=1)
        for rows in (2.0, 8.0):
            cost.observe_layer("sync", 4.0, rows, 4.55)
            cost.observe_target(4.0, rows, 1.0)
        cost.refresh()
        price = PriceTracker(min_anomaly_samples=100)
        price.on_step(0, 0, bs=1, continuous=True)
        now = 0
        for _ in range(4):
            now += 20_000_000
            price.on_step(now, 1, bs=1, continuous=True)
        gain = GainTable()
        gain.observe([1.0], depth=0)
        gain.observe([1.0], depth=1)
        driver = TollRound(cfg, price, cost, Calibrator(), gain, "sync")

        def one_child(trees, frontiers, width):
            return [_children(front, prob=0.2) for front in frontiers]

        fast = driver.run([4], [_root(0.15, width=2)], one_child)
        fast_grown = sum(layer.stop_reason == "grow" for layer in fast.layers)
        for _ in range(4):
            now += 2_000_000_000
            price.on_step(now, 1, bs=1, continuous=True)
        slow = driver.run([4], [_root(0.15, width=2)], one_child)
        slow_grown = sum(layer.stop_reason == "grow" for layer in slow.layers)
        self.assertGreaterEqual(slow_grown, fast_grown)
        self.assertLess(price_per_ms(price.price(1, 0.0)), price_per_ms(50.6))

    def test_exited_request_still_counts_toward_k(self):
        cfg = TollConfig(
            policy="toll",
            warmup_rounds=1,
            probe_interval=1,
            max_depth=3,
            child_topk=2,
            verify_buckets=(8, 16, 32),
        )
        cost = TierCostModel(probe_interval=1, warmup_rounds=1)
        for prefix, rows in ((0.0, 2.0), (10.0, 8.0), (40.0, 16.0)):
            cost.observe_layer("sync", prefix, rows, 0.05 * prefix + 0.001 * rows)
            cost.observe_target(prefix, rows, 0.1 * rows)
        cost.refresh()
        price = PriceTracker()
        price.on_step(0, 0, bs=2, continuous=True)
        price.on_step(2_000_000, 2, bs=2, continuous=True)
        gain = GainTable()
        gain.observe([1.0], depth=0)
        driver = TollRound(cfg, price, cost, Calibrator(), gain, "sync")

        def stop(trees, frontiers, width):
            return [_children(front, prob=0.0) for front in frontiers]

        hard = (np.array([0.9, 0.05]), np.array([11, 12]))
        easy = (np.array([0.3, 0.05]), np.array([21, 22]))
        result = driver.run([40, 1], [hard, easy], stop)
        self.assertTrue(driver.trees[0].exited)
        self.assertGreater(len(driver.trees[0]), 0)
        point = cost.snapshot("sync")
        lam_ms = price_per_ms(price.price(2, 0.0))
        both = choose_k(driver.trees, lam_ms, point, cfg)
        easy_only = choose_k([driver.trees[1]], lam_ms, point, cfg)
        self.assertEqual(result.k, both.k)
        self.assertNotAlmostEqual(both.score, easy_only.score, places=4)


if __name__ == "__main__":
    unittest.main()
