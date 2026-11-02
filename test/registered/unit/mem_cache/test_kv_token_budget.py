"""CPU tests for KV pool sizing when free memory is far below device capacity."""

import unittest

from sglang.srt.model_executor.kv_token_budget import (
    KV_FALLBACK_MIN_HEADROOM_GB,
    resolve_kv_token_budget,
)
from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

# Qwen3-VL-2B text KV: 8 kv heads * (128 + 128) * 28 layers * bf16.
QWEN3_VL_2B_CELL = 8 * (128 + 128) * 28 * 2
LOG_PRE_GB = 5.98
LOG_POST_GB = 1.57
LOG_MEM_FRACTION = 0.7524
LOG_PAGE_SIZE = 128
LOG_CHUNKED_PREFILL = 8192
LOG_CUDA_GRAPH_BS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    10,
    12,
    14,
    16,
    32,
    64,
]


def _static_rest_gb(pre_gb, post_gb, mem_fraction_static, extra_reserve_gb=0.0):
    return post_gb - pre_gb * (1.0 - mem_fraction_static) - extra_reserve_gb


def _profiled_tokens(rest_gb, cell_size):
    return int(rest_gb * (1 << 30)) // cell_size


class TestKVTokenBudget(CustomTestCase):
    def test_empty_device_keeps_static_fraction_formula(self):
        pre_gb = 64.0
        post_gb = 58.0
        mem_fraction = 0.8
        rest_gb = _static_rest_gb(pre_gb, post_gb, mem_fraction)
        expected = _profiled_tokens(rest_gb, QWEN3_VL_2B_CELL)

        budget = resolve_kv_token_budget(
            pre_gb=pre_gb,
            post_gb=post_gb,
            mem_fraction_static=mem_fraction,
            cell_size=QWEN3_VL_2B_CELL,
            page_size=1,
            chunked_prefill_size=8192,
            cuda_graph_bs=[1, 2, 4, 8, 64],
        )

        self.assertFalse(budget.used_fallback)
        self.assertEqual(budget.profiled_tokens, expected)
        self.assertEqual(budget.aligned_tokens, expected)
        self.assertIsNone(budget.chunked_prefill_size)
        self.assertIsNone(budget.cuda_graph_bs)
        self.assertFalse(budget.disable_cuda_graph)
        self.assertAlmostEqual(
            budget.activation_reserve_gb, pre_gb * (1.0 - mem_fraction)
        )

    def test_logged_npu_sample_stays_on_formula_when_one_page_fits(self):
        rest_gb = _static_rest_gb(LOG_PRE_GB, LOG_POST_GB, LOG_MEM_FRACTION)
        expected = _profiled_tokens(rest_gb, QWEN3_VL_2B_CELL)
        aligned = expected // LOG_PAGE_SIZE * LOG_PAGE_SIZE

        budget = resolve_kv_token_budget(
            pre_gb=LOG_PRE_GB,
            post_gb=LOG_POST_GB,
            mem_fraction_static=LOG_MEM_FRACTION,
            cell_size=QWEN3_VL_2B_CELL,
            page_size=LOG_PAGE_SIZE,
            chunked_prefill_size=LOG_CHUNKED_PREFILL,
            cuda_graph_bs=LOG_CUDA_GRAPH_BS,
            total_capacity_gb=61.3,
        )

        self.assertGreaterEqual(aligned, LOG_PAGE_SIZE)
        self.assertFalse(budget.used_fallback)
        self.assertEqual(budget.profiled_tokens, expected)
        self.assertEqual(budget.aligned_tokens, aligned)
        self.assertGreater(budget.aligned_tokens, 0)
        self.assertEqual(budget.aligned_tokens % LOG_PAGE_SIZE, 0)
        self.assertIsNone(budget.chunked_prefill_size)
        self.assertIsNone(budget.cuda_graph_bs)
        self.assertFalse(budget.disable_cuda_graph)

    def test_under_one_page_uses_post_load_activation_reserve(self):
        # Same weights and fraction as the NPU log, with pre-load free memory
        # high enough that the static reserve leaves less than one KV page.
        pre_gb = 6.4
        budget = resolve_kv_token_budget(
            pre_gb=pre_gb,
            post_gb=LOG_POST_GB,
            mem_fraction_static=LOG_MEM_FRACTION,
            cell_size=QWEN3_VL_2B_CELL,
            page_size=LOG_PAGE_SIZE,
            chunked_prefill_size=LOG_CHUNKED_PREFILL,
            cuda_graph_bs=LOG_CUDA_GRAPH_BS,
            total_capacity_gb=61.3,
        )

        static_rest = _static_rest_gb(pre_gb, LOG_POST_GB, LOG_MEM_FRACTION)
        static_aligned = (
            _profiled_tokens(static_rest, QWEN3_VL_2B_CELL) // LOG_PAGE_SIZE
            * LOG_PAGE_SIZE
        )
        self.assertLess(static_aligned, LOG_PAGE_SIZE)

        reserve_bytes = int(KV_FALLBACK_MIN_HEADROOM_GB * (1 << 30))
        kv_bytes = int(LOG_POST_GB * (1 << 30)) - reserve_bytes
        fallback_tokens = kv_bytes // QWEN3_VL_2B_CELL
        fallback_aligned = fallback_tokens // LOG_PAGE_SIZE * LOG_PAGE_SIZE
        activation_mb = KV_FALLBACK_MIN_HEADROOM_GB * 1024
        expected_chunked = max(
            LOG_PAGE_SIZE, min(LOG_CHUNKED_PREFILL, int(activation_mb / 1.5))
        )

        self.assertTrue(budget.used_fallback)
        self.assertTrue(budget.disable_cuda_graph)
        self.assertEqual(budget.profiled_tokens, fallback_tokens)
        self.assertEqual(budget.aligned_tokens, fallback_aligned)
        self.assertGreater(budget.aligned_tokens, 0)
        self.assertEqual(budget.aligned_tokens % LOG_PAGE_SIZE, 0)
        self.assertGreaterEqual(
            budget.activation_reserve_gb, KV_FALLBACK_MIN_HEADROOM_GB
        )
        self.assertEqual(budget.chunked_prefill_size, expected_chunked)
        self.assertLess(budget.chunked_prefill_size, LOG_CHUNKED_PREFILL)
        self.assertEqual(budget.cuda_graph_bs, [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(budget.cuda_graph_max_bs, 8)
        self.assertIn("Skipping NPU graph", budget.warning)
        self.assertIn(f"mem_fraction_static={LOG_MEM_FRACTION}", budget.warning)
        self.assertIn(f"pre_free_gb={pre_gb:.4f}", budget.warning)
        self.assertIn(f"post_free_gb={LOG_POST_GB:.4f}", budget.warning)
        self.assertIn(f"cell_size={QWEN3_VL_2B_CELL}", budget.warning)
        self.assertIn(f"page_size={LOG_PAGE_SIZE}", budget.warning)

    def test_logged_tight_profile_skips_npu_graph(self):
        pre_gb = 5.9881
        post_gb = 1.4671
        static_rest = _static_rest_gb(pre_gb, post_gb, LOG_MEM_FRACTION)
        static_aligned = (
            _profiled_tokens(static_rest, QWEN3_VL_2B_CELL) // LOG_PAGE_SIZE
            * LOG_PAGE_SIZE
        )
        self.assertLess(static_aligned, LOG_PAGE_SIZE)

        budget = resolve_kv_token_budget(
            pre_gb=pre_gb,
            post_gb=post_gb,
            mem_fraction_static=LOG_MEM_FRACTION,
            cell_size=QWEN3_VL_2B_CELL,
            page_size=LOG_PAGE_SIZE,
            chunked_prefill_size=LOG_CHUNKED_PREFILL,
            cuda_graph_bs=LOG_CUDA_GRAPH_BS,
            total_capacity_gb=61.27,
        )

        reserve_bytes = int(KV_FALLBACK_MIN_HEADROOM_GB * (1 << 30))
        kv_bytes = int(post_gb * (1 << 30)) - reserve_bytes
        expected_tokens = kv_bytes // QWEN3_VL_2B_CELL
        expected_aligned = expected_tokens // LOG_PAGE_SIZE * LOG_PAGE_SIZE

        self.assertTrue(budget.used_fallback)
        self.assertTrue(budget.disable_cuda_graph)
        self.assertGreaterEqual(
            budget.activation_reserve_gb, KV_FALLBACK_MIN_HEADROOM_GB
        )
        self.assertEqual(budget.profiled_tokens, expected_tokens)
        self.assertEqual(budget.aligned_tokens, expected_aligned)
        self.assertGreater(budget.aligned_tokens, 0)
        self.assertEqual(budget.aligned_tokens % LOG_PAGE_SIZE, 0)
        self.assertIn("Skipping NPU graph", budget.warning)

    def test_graph_buckets_above_cap_fall_back_to_one(self):
        budget = resolve_kv_token_budget(
            pre_gb=6.4,
            post_gb=LOG_POST_GB,
            mem_fraction_static=LOG_MEM_FRACTION,
            cell_size=QWEN3_VL_2B_CELL,
            page_size=LOG_PAGE_SIZE,
            chunked_prefill_size=None,
            cuda_graph_bs=[16, 32, 64],
        )

        self.assertTrue(budget.used_fallback)
        self.assertTrue(budget.disable_cuda_graph)
        self.assertEqual(budget.cuda_graph_bs, [1])
        self.assertEqual(budget.cuda_graph_max_bs, 1)
        self.assertGreaterEqual(budget.chunked_prefill_size, LOG_PAGE_SIZE)

    def test_post_load_free_below_one_page_reports_budget_inputs(self):
        with self.assertRaises(RuntimeError) as ctx:
            resolve_kv_token_budget(
                pre_gb=LOG_PRE_GB,
                post_gb=0.001,
                mem_fraction_static=LOG_MEM_FRACTION,
                cell_size=QWEN3_VL_2B_CELL,
                page_size=LOG_PAGE_SIZE,
                total_capacity_gb=61.3,
            )

        message = str(ctx.exception)
        self.assertNotIn("Please try to increase --mem-fraction-static", message)
        for field in (
            "mem_fraction_static=",
            "pre_free_gb=",
            "post_free_gb=",
            "rest_gb=",
            "cell_size=",
            "page_size=",
            "aligned_tokens=",
            "total_capacity_gb=61.3000",
            "available_gb=",
        ):
            self.assertIn(field, message)


if __name__ == "__main__":
    unittest.main()
