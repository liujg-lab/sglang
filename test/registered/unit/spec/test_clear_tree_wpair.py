"""CPU checks for the ClearTree block-wise paired width rule."""

import math
import os
import unittest
from unittest.mock import patch

from sglang.srt.speculative.clear_tree_remote import clear_tree_wpair as wpair
from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import reset_policy
from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
    assign_persistent_width,
    persistent_width,
)
from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=20, suite="stage-a-test-cpu")


def _node(token, parent, score):
    return {"token": token, "parent": parent, "score": score}


def _tree():
    return {
        "layers": [
            [
                _node(1, 0, 1.0),
                _node(2, 0, 4.0),
                _node(3, 0, 2.0),
                _node(4, 0, 3.0),
            ],
            [
                _node(5, 1, 0.5),
                _node(6, 1, 0.9),
                _node(7, 3, 0.2),
                _node(8, 0, 8.0),
            ],
        ],
        "ratio": 1.0,
        "prefix": 4,
    }


def _frozen(**overrides):
    frozen = {
        "batch": 1,
        "width": 4,
        "theta": 1.0,
        "k": 8,
        "flat_k": 16,
        "verify_ms": {2: 5.0, 4: 5.5, 8: 6.0},
        "layers": [{"pred": {4: 3.0, 2: 1.0}}],
        "trees": {"a": _tree()},
    }
    frozen.update(overrides)
    return frozen


class TestPairValues(CustomTestCase):
    def test_narrow_k_is_smaller_and_value_matches_the_hand_calc(self):
        values = wpair.pair_values(_frozen(), [4, 2])
        self.assertAlmostEqual(values[4][1], 19.6)
        self.assertAlmostEqual(values[4][0], 19.6 - 9.0)
        self.assertAlmostEqual(values[2][1], 8.4)
        self.assertAlmostEqual(values[2][0], 8.4 - 6.5)

    def test_missing_price_layer_or_verify_omits_the_width(self):
        self.assertEqual(wpair.pair_values(_frozen(theta=None), [4, 2]), {})
        missing_layer = _frozen(layers=[{"pred": {4: 3.0}}])
        partial = wpair.pair_values(missing_layer, [4, 2])
        self.assertIn(4, partial)
        self.assertNotIn(2, partial)
        unpriced = _frozen(verify_ms={}, flat_k=16)
        self.assertEqual(wpair.pair_values(unpriced, [4, 2]), {})

    def test_spent_is_the_layer_schedule_plus_verify(self):
        values = wpair.pair_values(_frozen(), [4, 2])
        self.assertAlmostEqual(values[4][2], 9.0)
        self.assertAlmostEqual(values[2][2], 6.5)


class TestPairBlocks(CustomTestCase):
    """Block rules on the older value gain; samples here carry no spent."""

    def setUp(self):
        reset_policy()
        assign_persistent_width(1, 8)
        self._saved_metric = os.environ.get("SGLANG_CLEAR_WPAIR_METRIC")
        os.environ["SGLANG_CLEAR_WPAIR_METRIC"] = "value"

    def tearDown(self):
        if self._saved_metric is None:
            os.environ.pop("SGLANG_CLEAR_WPAIR_METRIC", None)
        else:
            os.environ["SGLANG_CLEAR_WPAIR_METRIC"] = self._saved_metric

    def _observe(self, built, values):
        frozen = _frozen(width=built)
        with patch.object(wpair, "pair_values", return_value=values):
            return wpair.observe_pair(frozen)

    def test_fifteen_decisive_up_samples_switch_and_reset_p(self):
        wpair.superset_due(1, 8, lambda _w: False)
        wpair.superset_due(1, 8, lambda _w: False)
        wpair.superset_due(1, 8, lambda _w: False)
        self.assertAlmostEqual(wpair._blocks[1]["acc"], 0.6)
        last = None
        values = {16: (3.0, 1.0), 8: (0.0, 100.0), 4: (-1.0, 1.0)}
        for _ in range(15):
            last = self._observe(16, values)
        self.assertEqual(last["decision"], "up")
        self.assertEqual(last["target"], 16)
        self.assertEqual(persistent_width(1), 16)
        self.assertAlmostEqual(wpair._blocks[1]["p"], wpair.P0)
        self.assertEqual(wpair._blocks[1]["acc"], 0.0)

    def test_negative_up_samples_wait_until_thirty_then_stay(self):
        for _ in range(3):
            self.assertFalse(wpair.superset_due(1, 8, lambda _w: False))
        values = {16: (-3.0, 1.0), 8: (0.0, 100.0), 4: (-3.0, 1.0)}
        last = None
        for _ in range(29):
            last = self._observe(16, values)
            self.assertIsNone(last)
        last = self._observe(16, values)
        self.assertEqual(last["decision"], "stay")
        self.assertEqual(last["target"], 8)
        self.assertEqual(last["n_up"], 30)
        self.assertEqual(persistent_width(1), 8)
        self.assertAlmostEqual(wpair._blocks[1]["p"], 0.10)
        self.assertAlmostEqual(wpair._blocks[1]["acc"], 0.6)
        self.assertEqual(wpair._blocks[1]["up"], [])

    def test_undecided_samples_wait_until_thirty_then_stay(self):
        values = {16: (0.0, 1.0), 8: (0.0, 100.0), 4: (0.0, 1.0)}
        last = None
        for _ in range(29):
            last = self._observe(16, values)
            self.assertIsNone(last)
        last = self._observe(16, values)
        self.assertEqual(last["decision"], "stay")
        self.assertEqual(last["n_up"], 30)
        self.assertAlmostEqual(wpair._blocks[1]["p"], 0.10)

    def test_stay_halves_p_down_to_the_floor(self):
        values = {16: (0.0, 1.0), 8: (0.0, 100.0), 4: (0.0, 1.0)}
        rates = []
        for _ in range(3):
            for _step in range(30):
                record = self._observe(16, values)
            rates.append(record["p"])
            self.assertEqual(record["decision"], "stay")
        self.assertEqual(rates, [0.20, 0.10, 0.05])
        self.assertAlmostEqual(wpair._blocks[1]["p"], 0.05)

    def test_ordinary_rounds_record_nothing(self):
        for _ in range(15):
            record = self._observe(8, {4: (8.0, 1.0), 8: (0.0, 100.0)})
            self.assertIsNone(record)
        self.assertEqual(wpair._blocks[1]["down"], [])
        self.assertEqual(wpair._blocks[1]["up"], [])
        record = self._observe(4, {4: (8.0, 1.0), 8: (0.0, 100.0), 16: (3.0, 1.0)})
        self.assertIsNone(record)
        self.assertEqual(wpair._blocks[1]["down"], [])
        for _ in range(10):
            record = self._observe(
                16, {16: (0.0, 1.0), 8: (0.0, 100.0), 4: (8.0, 1.0)}
            )
            self.assertIsNone(record)
        self.assertEqual(len(wpair._blocks[1]["down"]), 10)
        self.assertEqual(len(wpair._blocks[1]["up"]), 10)
        self.assertEqual(persistent_width(1), 8)

    def test_both_directions_keep_probing_until_thirty(self):
        values = {16: (3.0, 1.0), 8: (0.0, 100.0), 4: (8.0, 1.0)}
        last = None
        for _ in range(15):
            last = self._observe(16, values)
        self.assertIsNone(last)
        self.assertEqual(len(wpair._blocks[1]["up"]), 15)
        self.assertEqual(len(wpair._blocks[1]["down"]), 15)
        self.assertEqual(persistent_width(1), 8)
        self.assertAlmostEqual(wpair._blocks[1]["p"], 0.20)
        for _ in range(14):
            last = self._observe(16, values)
            self.assertIsNone(last)
        last = self._observe(16, values)
        self.assertEqual(last["decision"], "stay")
        self.assertEqual(last["target"], 8)
        self.assertEqual(last["n_up"], 30)
        self.assertEqual(persistent_width(1), 8)
        self.assertEqual(wpair._blocks[1]["up"], [])
        self.assertEqual(wpair._blocks[1]["down"], [])
        self.assertAlmostEqual(wpair._blocks[1]["p"], 0.20)

    def test_only_down_switches(self):
        last = None
        for _ in range(15):
            last = self._observe(
                16, {16: (0.0, 1.0), 8: (0.0, 100.0), 4: (8.0, 1.0)}
            )
        self.assertEqual(last["decision"], "down")
        self.assertEqual(last["target"], 4)
        self.assertEqual(persistent_width(1), 4)

    def test_equal_lower_bounds_keep_the_rate_at_thirty(self):
        values = {16: (3.0, 1.0), 8: (0.0, 100.0), 4: (3.0, 1.0)}
        last = None
        for _ in range(15):
            last = self._observe(16, values)
        self.assertIsNone(last)
        self.assertAlmostEqual(wpair._blocks[1]["p"], 0.20)
        for _ in range(15):
            last = self._observe(16, values)
        self.assertEqual(last["decision"], "stay")
        self.assertEqual(persistent_width(1), 8)
        self.assertEqual(wpair._blocks[1]["up"], [])
        self.assertAlmostEqual(wpair._blocks[1]["p"], 0.20)

    def test_identical_samples_use_one_se_at_thirty(self):
        gain, se = wpair._gs([(1.0, 100.0)] * 15)
        self.assertAlmostEqual(gain, 0.01)
        self.assertGreaterEqual(se, 0.01 / math.sqrt(15) - 1e-12)
        self.assertLessEqual(gain - wpair.Z * se, wpair.BAND)
        gain30, se30 = wpair._gs([(1.0, 100.0)] * 30)
        self.assertGreater(gain30 - se30, wpair.BAND)
        values = {16: (1.0, 1.0), 8: (0.0, 100.0), 4: (0.0, 1.0)}
        last = None
        for _ in range(29):
            last = self._observe(16, values)
            self.assertIsNone(last)
        last = self._observe(16, values)
        self.assertEqual(last["decision"], "up")
        self.assertEqual(last["target"], 16)
        self.assertEqual(last["n_up"], 30)

    def test_widest_width_uses_virtual_superset_rounds(self):
        assign_persistent_width(1, 64)
        last = None
        for _ in range(75):
            self.assertFalse(wpair.superset_due(1, 64, lambda _w: True))
            last = self._observe(64, {32: (8.0, 1.0), 64: (0.0, 100.0)})
        self.assertEqual(last["decision"], "down")
        self.assertEqual(last["target"], 32)
        self.assertEqual(last["n_down"], 75)
        self.assertIsNone(last["g_up"])
        self.assertEqual(persistent_width(1), 32)

    def test_widest_width_uses_one_se_at_virt_thirty(self):
        assign_persistent_width(1, 64)
        values = {32: (0.7, 1.0), 64: (0.0, 100.0)}
        last = None
        for _ in range(149):
            self.assertFalse(wpair.superset_due(1, 64, lambda _w: True))
            last = self._observe(64, values)
            self.assertIsNone(last)
        self.assertFalse(wpair.superset_due(1, 64, lambda _w: True))
        last = self._observe(64, values)
        self.assertEqual(last["decision"], "down")
        self.assertEqual(last["target"], 32)
        self.assertEqual(last["n_down"], 150)
        self.assertEqual(persistent_width(1), 32)

    def test_unready_superset_is_spent_on_the_next_ready_round(self):
        for _ in range(5):
            self.assertFalse(wpair.superset_due(1, 8, lambda _w: False))
        self.assertAlmostEqual(wpair._blocks[1]["acc"], 1.0)
        self.assertTrue(wpair.superset_due(1, 8, lambda _w: True))
        self.assertAlmostEqual(wpair._blocks[1]["acc"], 0.2)

    def test_reset_restores_the_opening_rate(self):
        values = {16: (0.0, 1.0), 8: (0.0, 100.0), 4: (0.0, 1.0)}
        for _ in range(30):
            self._observe(16, values)
        self.assertAlmostEqual(wpair._blocks[1]["p"], 0.10)
        wpair.reset_pair_state()
        self.assertFalse(wpair.superset_due(1, 8, lambda _w: False))
        self.assertAlmostEqual(wpair._blocks[1]["p"], 0.20)
        self.assertAlmostEqual(wpair._blocks[1]["acc"], 0.20)

    def test_missing_side_records_nothing(self):
        self._observe(16, {16: (3.0, 1.0), 8: (0.0, 100.0)})
        self.assertEqual(wpair._blocks[1]["up"], [])
        self.assertEqual(wpair._blocks[1]["down"], [])
        self._observe(16, {8: (0.0, 100.0), 4: (2.0, 1.0)})
        self.assertEqual(wpair._blocks[1]["up"], [])
        self.assertEqual(wpair._blocks[1]["down"], [])
        self._observe(16, {16: (3.0, 1.0), 8: (0.0, 100.0), 4: (2.0, 1.0)})
        self.assertEqual(len(wpair._blocks[1]["up"]), 1)
        self.assertEqual(len(wpair._blocks[1]["down"]), 1)
        assign_persistent_width(1, 2)
        self._observe(4, {4: (3.0, 1.0)})
        self.assertEqual(len(wpair._blocks[1]["up"]), 1)
        self._observe(4, {4: (3.0, 1.0), 2: (0.0, 100.0)})
        self.assertEqual(len(wpair._blocks[1]["up"]), 2)
        self.assertEqual(len(wpair._blocks[1]["down"]), 1)


class TestPairRatio(CustomTestCase):
    """Paired throughput ratio: (E + B) / (spent + s), other width over served."""

    _VALUES = {
        8: (0.0, 10.0, 50.0),
        16: (0.2, 10.3, 51.0),
        4: (-1.0, 9.0, 49.0),
    }

    def setUp(self):
        reset_policy()
        assign_persistent_width(1, 8)
        self._saved_metric = os.environ.pop("SGLANG_CLEAR_WPAIR_METRIC", None)

    def tearDown(self):
        os.environ.pop("SGLANG_CLEAR_WPAIR_METRIC", None)
        if self._saved_metric is not None:
            os.environ["SGLANG_CLEAR_WPAIR_METRIC"] = self._saved_metric

    def _observe(self, values):
        frozen = _frozen(width=16, bonus=4, seed_ms=10.0)
        with patch.object(wpair, "pair_values", return_value=values):
            return wpair.observe_pair(frozen)

    def test_default_metric_is_ratio(self):
        self.assertEqual(wpair.wpair_metric(), "ratio")
        os.environ["SGLANG_CLEAR_WPAIR_METRIC"] = "value"
        self.assertEqual(wpair.wpair_metric(), "value")

    def test_sample_adds_bonus_and_seed(self):
        sample = wpair._sample(self._VALUES, 16, 8, 4.0, 10.0)
        expected = (0.2, 10.0, 14.3, 61.0, 14.0, 60.0)
        for got, want in zip(sample, expected):
            self.assertAlmostEqual(got, want)
        bare = wpair._sample({16: (0.2, 10.3), 8: (0.0, 10.0)}, 16, 8, 4.0, 10.0)
        self.assertAlmostEqual(bare[0], 0.2)
        self.assertAlmostEqual(bare[1], 10.0)
        self.assertEqual(bare[2:], (None, None, None, None))
        self.assertIsNone(wpair._sample({8: (0.0, 10.0, 50.0)}, 16, 8))

    def test_ratio_matches_the_hand_calc(self):
        same = [(0.0, 0.0, 14.3, 61.0, 14.0, 60.0)] * 15
        gain, se = wpair._ratio_gs(same)
        self.assertAlmostEqual(gain, (14.3 / 61.0) / (14.0 / 60.0) - 1.0)
        self.assertAlmostEqual(se, wpair.SIG_FLOOR / math.sqrt(15))
        rows = [
            (0.0, 0.0, 10.0, 20.0, 9.0, 20.0),
            (0.0, 0.0, 14.0, 30.0, 12.0, 30.0),
            (0.0, 0.0, None, None, None, None),
            (0.0, 0.0),
        ]
        gain, se = wpair._ratio_gs(rows)
        ratio = 12.0 / 10.5
        self.assertAlmostEqual(gain, ratio - 1.0)
        self.assertAlmostEqual(se, ratio / 42.0)
        self.assertEqual(wpair._ratio_gs(rows[1:]), (None, None))

    def test_value_switches_where_ratio_stays(self):
        os.environ["SGLANG_CLEAR_WPAIR_METRIC"] = "value"
        last = None
        for _ in range(15):
            last = self._observe(self._VALUES)
        self.assertEqual(last["decision"], "up")
        self.assertEqual(last["metric"], "value")
        self.assertAlmostEqual(last["g_up"], 0.02)
        self.assertAlmostEqual(last["r_up"], (14.3 / 61.0) / (14.0 / 60.0) - 1.0)
        self.assertEqual(persistent_width(1), 16)

        reset_policy()
        assign_persistent_width(1, 8)
        os.environ.pop("SGLANG_CLEAR_WPAIR_METRIC", None)
        for _ in range(29):
            self.assertIsNone(self._observe(self._VALUES))
        last = self._observe(self._VALUES)
        self.assertEqual(last["decision"], "stay")
        self.assertEqual(last["metric"], "ratio")
        self.assertEqual(last["n_up"], 30)
        self.assertAlmostEqual(last["r_up"], (14.3 / 61.0) / (14.0 / 60.0) - 1.0)
        self.assertLess(last["r_dn"], 0.0)
        self.assertIsNotNone(last["rse_up"])
        self.assertEqual(persistent_width(1), 8)
        self.assertAlmostEqual(wpair._blocks[1]["p"], 0.10)

    def test_ratio_without_spent_never_clears(self):
        values = {16: (3.0, 1.0), 8: (0.0, 100.0), 4: (-1.0, 1.0)}
        last = None
        for _ in range(30):
            last = self._observe(values)
        self.assertEqual(last["decision"], "stay")
        self.assertIsNone(last["r_up"])
        self.assertIsNotNone(last["g_up"])


if __name__ == "__main__":
    unittest.main()
