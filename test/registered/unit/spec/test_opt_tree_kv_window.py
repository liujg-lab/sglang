"""CPU checks for the OPT Draft KV window shadow row."""

import os
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import pack_verify_attention
from sglang.srt.speculative.opt_tree_remote.opt_tree_kv_window import (
    fill_shadow,
    fill_shadow_delta,
    index_mismatch,
    prepare_layer_window,
    release_kv_window,
    window_layout,
    write_new_slots,
)
from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


class TestOptKvWindow(CustomTestCase):
    def test_truncated_layout_and_mask(self):
        layout = window_layout(4096, 4096 + 16, 64, 512)
        self.assertEqual(layout.mask_prefix, 64 + 512)
        self.assertEqual(layout.seq_len, 64 + 512 + 16)
        self.assertEqual(layout.position_prefix, 4096)
        self.assertEqual(layout.skipped, 4096 - 64 - 512)
        self.assertEqual(layout.new_at, layout.seq_len)

        table = torch.arange(2 * 4200, dtype=torch.int32).reshape(2, 4200)
        fill_shadow(table, 0, 1, layout)
        shadow = table[1]
        self.assertTrue(torch.equal(shadow[:64], table[0, :64]))
        self.assertTrue(torch.equal(shadow[64 : 64 + 512], table[0, 4096 - 512 : 4096]))
        self.assertTrue(torch.equal(shadow[64 + 512 : 64 + 512 + 16], table[0, 4096 : 4112]))

        slots = torch.tensor([9, 8, 7, 6], dtype=torch.int32)
        write_new_slots(table, 1, layout, slots)
        self.assertTrue(torch.equal(shadow[layout.new_at : layout.new_at + 4], slots))

        ancestor = torch.ones(4, 8, dtype=torch.bool)
        mask, positions = pack_verify_attention(
            [layout.mask_prefix],
            [ancestor],
            depth=2,
            width=4,
            device="cpu",
            position_lens=[layout.position_prefix],
        )
        self.assertEqual(mask.numel(), 4 * (layout.mask_prefix + 8))
        self.assertTrue(torch.all(positions == 4096 + 2 - 1))

    def test_delta_keeps_the_prefix_and_appends_tree_columns(self):
        layout = window_layout(4096, 4112, 64, 512)
        table = torch.arange(2 * 4200, dtype=torch.int32).reshape(2, 4200)
        fill_shadow(table, 0, 1, layout)
        table[1, 0] = -1
        grown = window_layout(4096, 4120, 64, 512)
        fill_shadow_delta(table, 0, 1, grown, 4112)
        self.assertEqual(int(table[1, 0]), -1)
        self.assertTrue(torch.equal(table[1, 64 + 512 + 16 : 64 + 512 + 24], table[0, 4112:4120]))

    def test_truncated_kept_span_matches_the_real_row(self):
        layout = window_layout(4096, 4096 + 16, 64, 512)
        table = torch.arange(2 * 4200, dtype=torch.int32).reshape(2, 4200)
        fill_shadow(table, 0, 1, layout)
        slots = torch.tensor([9, 8, 7, 6], dtype=torch.int32)
        write_new_slots(table, 1, layout, slots)
        mismatch, first = index_mismatch(table[0], table[1], layout, slots)
        self.assertEqual(mismatch, 0)
        self.assertIsNone(first)
        table[1, 0] = -1
        mismatch, first = index_mismatch(table[0], table[1], layout, slots)
        self.assertGreater(mismatch, 0)
        self.assertEqual(first, 0)
        wide = window_layout(4352, 4352, 64, 8192)
        self.assertEqual(wide.skipped, 0)

    def test_full_window_copies_the_real_prefix(self):
        layout = window_layout(4096, 4112, 64, "full")
        self.assertEqual(layout.mask_prefix, 4096)
        self.assertEqual(layout.seq_len, 4112)
        self.assertEqual(layout.position_prefix, 4096)
        self.assertEqual(layout.skipped, 0)
        table = torch.arange(2 * 4112, dtype=torch.int32).reshape(2, 4112)
        fill_shadow(table, 0, 1, layout)
        self.assertTrue(torch.equal(table[1, :4112], table[0, :4112]))

    def test_page_size_128_stays_off(self):
        old = os.environ.get("SGLANG_OPT_DRAFT_KV_WINDOW")
        os.environ["SGLANG_OPT_DRAFT_KV_WINDOW"] = "512"
        try:
            runners = [
                SimpleNamespace(prefix_len=4096, cached_len=4096, req=SimpleNamespace(req_pool_idx=0)),
                SimpleNamespace(prefix_len=4096, cached_len=4096, req=SimpleNamespace(req_pool_idx=1)),
            ]
            pool = SimpleNamespace(
                size=4,
                free_slots=[2, 3],
                req_to_token=torch.zeros(4, 8, dtype=torch.int32),
            )
            scheduler = SimpleNamespace(req_to_token_pool=pool)
            self.assertIsNone(prepare_layer_window(scheduler, runners, page_size=128))
            self.assertEqual(pool.free_slots, [2, 3])
        finally:
            if old is None:
                os.environ.pop("SGLANG_OPT_DRAFT_KV_WINDOW", None)
            else:
                os.environ["SGLANG_OPT_DRAFT_KV_WINDOW"] = old

    def test_borrow_and_release(self):
        old = os.environ.get("SGLANG_OPT_DRAFT_KV_WINDOW")
        os.environ["SGLANG_OPT_DRAFT_KV_WINDOW"] = "full"
        try:
            reqs = [SimpleNamespace(req_pool_idx=0), SimpleNamespace(req_pool_idx=1)]
            runners = [
                SimpleNamespace(prefix_len=4096, cached_len=4100, req=reqs[0]),
                SimpleNamespace(prefix_len=100, cached_len=100, req=reqs[1]),
            ]
            table = torch.arange(4 * 4100, dtype=torch.int32).reshape(4, 4100)
            pool = SimpleNamespace(size=4, free_slots=[2, 3], req_to_token=table)
            scheduler = SimpleNamespace(req_to_token_pool=pool)
            plan = prepare_layer_window(scheduler, runners, page_size=1)
            self.assertIsNotNone(plan)
            self.assertEqual(plan.requests[0].pool_index, 2)
            self.assertIsNone(plan.requests[1].pool_index)
            self.assertEqual(plan.seq_lens(), [4100, 100])
            self.assertEqual(plan.mask_prefixes(), [4096, 100])
            self.assertTrue(torch.equal(table[2, :4100], table[0, :4100]))
            release_kv_window(scheduler, reqs[0])
            self.assertIsNone(getattr(reqs[0], "opt_kv_window_idx", None))
            self.assertIn(2, pool.free_slots)
        finally:
            if old is None:
                os.environ.pop("SGLANG_OPT_DRAFT_KV_WINDOW", None)
            else:
                os.environ["SGLANG_OPT_DRAFT_KV_WINDOW"] = old


if __name__ == "__main__":
    unittest.main()
