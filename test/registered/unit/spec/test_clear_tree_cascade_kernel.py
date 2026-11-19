"""CPU checks for the stateless ClearTree cascade, plus a CUDA compare when present."""

import unittest

import torch

from sglang.srt.speculative.clear_tree_remote.clear_tree_cascade_kernel import (
    fill_cascade_meta,
    prefix_split_bounds,
    reference,
    splits_for,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=20, suite="stage-a-test-cpu")
register_cuda_ci(est_time=120, suite="stage-b-test-1-gpu-small")


class ClearTreeCascadeMetaTest(CustomTestCase):
    def test_splits_follow_batch_and_stay_in_range(self):
        self.assertEqual(splits_for(1, 2, 82), 32)
        self.assertEqual(splits_for(4, 2, 82), 21)
        self.assertEqual(splits_for(100, 8, 82), 1)
        self.assertEqual(splits_for(1, 1, 1), 2)

    def test_prefix_splits_cover_the_prefix_once(self):
        for length in (0, 1, 2, 7, 32, 40, 4096):
            for parts in (1, 3, 21, 31):
                covered = []
                for split in range(parts):
                    start, end = prefix_split_bounds(split, length, parts)
                    covered.extend(range(start, end))
                self.assertEqual(covered, list(range(length)))

    def test_fill_clears_mask_tails_and_pads_graph_rows(self):
        width = 4
        padded_bs = 3
        raw_bs = 2
        cols = 6
        prefix_len = torch.full((4,), 9, dtype=torch.int32)
        tree_mask = torch.full((padded_bs * width, 8), 7, dtype=torch.uint8)
        prefix_rows = torch.zeros(padded_bs * width, dtype=torch.int32)
        real_mask = torch.tensor(
            [
                [1, 0, 1, 0],
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [1, 0, 0, 1],
                [1, 1, 1, 0],
                [0, 0, 1, 1],
                [1, 0, 1, 1],
                [0, 1, 0, 1],
            ],
            dtype=torch.uint8,
        )
        real_rows = torch.tensor([1, 1, 0, 1, 1, 0, 1, 1], dtype=torch.int32)
        seq_lens = torch.tensor([20, 11, 1], dtype=torch.int32)
        fill_cascade_meta(
            prefix_len,
            tree_mask,
            prefix_rows,
            prefix_lens=[5, 8],
            tree_mask=real_mask,
            prefix_rows=real_rows,
            seq_lens=seq_lens,
            width=width,
            padded_bs=padded_bs,
        )
        self.assertEqual(prefix_len[:3].tolist(), [5, 8, 1])
        self.assertEqual(prefix_len[3].item(), 9)
        self.assertEqual(prefix_rows[:8].tolist(), real_rows.tolist())
        self.assertEqual(prefix_rows[8:].tolist(), [1] * width)
        self.assertTrue(torch.equal(tree_mask[:8, :4], real_mask))
        self.assertTrue(torch.all(tree_mask[:8, 4:] == 0))
        self.assertTrue(torch.all(tree_mask[8:12] == 0))
        self.assertTrue(torch.all(tree_mask[12:] == 7))

    def test_capture_fill_uses_seq_len_and_hides_the_tree(self):
        prefix_len = torch.zeros(2, dtype=torch.int32)
        tree_mask = torch.ones((8, 3), dtype=torch.uint8)
        prefix_rows = torch.zeros(8, dtype=torch.int32)
        seq_lens = torch.tensor([1, 1], dtype=torch.int32)
        fill_cascade_meta(
            prefix_len,
            tree_mask,
            prefix_rows,
            prefix_lens=None,
            tree_mask=None,
            prefix_rows=None,
            seq_lens=seq_lens,
            width=4,
            padded_bs=2,
        )
        self.assertEqual(prefix_len.tolist(), [1, 1])
        self.assertTrue(torch.all(prefix_rows == 1))
        self.assertTrue(torch.all(tree_mask == 0))

    def test_reference_respects_per_request_prefix_and_hidden_rows(self):
        torch.manual_seed(3)
        width = 2
        bs = 2
        heads, dim = 2, 4
        q = torch.randn(bs * width, heads, dim)
        k_cache = torch.randn(16, 1, dim)
        v_cache = torch.randn(16, 1, dim)
        req_to_token = torch.arange(32, dtype=torch.int32).reshape(2, 16) % 16
        prefix_len = torch.tensor([2, 0], dtype=torch.int32)
        seq_lens = torch.tensor([3, 1], dtype=torch.int32)
        tree_mask = torch.zeros((bs * width, 4), dtype=torch.uint8)
        tree_mask[0, 0] = 1
        tree_mask[1, 1] = 1
        tree_mask[2, 0] = 1
        tree_mask[3, 2] = 1
        prefix_rows = torch.tensor([1, 0, 1, 1], dtype=torch.int32)
        got = reference(
            q,
            k_cache,
            v_cache,
            req_to_token,
            torch.tensor([0, 1]),
            seq_lens,
            prefix_len,
            tree_mask,
            prefix_rows,
            dim**-0.5,
            width,
        )
        # Request 0 row 1 hides the prefix, so it attends only tree column 1,
        # which is req_to_token[0, 2 + 1].
        slot = int(req_to_token[0, 3])
        only = (q[1, 0].float() * k_cache[slot, 0].float()).sum() * (dim**-0.5)
        weight = torch.softmax(only.reshape(1), dim=0)
        expect = (weight[0] * v_cache[slot, 0].float()).to(dtype=q.dtype)
        torch.testing.assert_close(got[1, 0], expect)
        self.assertFalse(torch.allclose(got[0], got[2]))

    def test_bonus_and_layer_rows_match_req_to_token_columns(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from sglang.srt.speculative.clear_tree_remote import (
            clear_tree_attention as attention,
        )
        from sglang.srt.speculative.clear_tree_remote.clear_tree_forward import (
            _bind_bonus_cascade,
            bind_cascade,
            bonus_attention_mask,
        )

        scheduler = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(device="cpu"),
        )
        runners = [
            SimpleNamespace(prefix_len=4, cached_len=8, scheduler=scheduler),
            SimpleNamespace(prefix_len=2, cached_len=2, scheduler=scheduler),
        ]
        pool = torch.full((2, 32), -1, dtype=torch.int32)
        layer = torch.tensor([10, 11, 12, 13, 20, 21, 22, 23], dtype=torch.int32)
        pool[0, 4:8] = layer[:4]
        pool[1, 2:6] = layer[4:]
        spec = SimpleNamespace()
        ancestors = [
            torch.tensor(
                [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 0, 1]],
                dtype=torch.uint8,
            ),
            torch.tensor(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                dtype=torch.uint8,
            ),
        ]
        with patch.object(attention, "cascade_enabled", return_value=True):
            bind_cascade(spec, runners, ancestors, 4, cached_lens=[4, 2])
        rows = spec.target_verify_attention.rows
        self.assertEqual(rows.prefix_lens, [4, 2])
        self.assertEqual(tuple(rows.tree_mask.shape), (8, 4))
        self.assertTrue(torch.equal(rows.tree_mask[:4], ancestors[0]))
        self.assertEqual(
            pool[0, 4 : 4 + 4].tolist(),
            layer[:4].tolist(),
        )
        self.assertEqual(pool[1, 2 : 2 + 4].tolist(), layer[4:].tolist())

        masks = [
            bonus_attention_mask(4, 8, [(), (5,), (6,)], query_width=3).reshape(-1),
            bonus_attention_mask(2, 2, [(), (), ()], query_width=3).reshape(-1),
        ]
        bonus = SimpleNamespace(spec_info=SimpleNamespace())
        with patch.object(attention, "cascade_enabled", return_value=True):
            _bind_bonus_cascade(bonus, runners, torch.cat(masks), [8, 2], 3)
        bonus_rows = bonus.spec_info.target_verify_attention.rows
        self.assertEqual(tuple(bonus_rows.tree_mask.shape), (6, 7))
        self.assertEqual(bonus_rows.tree_mask[:3, 4:].sum().item(), 3)
        self.assertTrue(torch.all(bonus_rows.tree_mask[3:, 3:] == 0))
        bonus_slots = torch.tensor([30, 31, 32, 40, 41, 42], dtype=torch.int32)
        pool[0, 8:11] = bonus_slots[:3]
        pool[1, 2:5] = bonus_slots[3:]
        self.assertEqual(pool[0, 4:11].tolist(), [10, 11, 12, 13, 30, 31, 32])
        self.assertEqual(pool[1, 2:5].tolist(), [40, 41, 42])


def _require_cuda():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("cuda")
    try:
        import triton  # noqa: F401
    except Exception as exc:
        raise unittest.SkipTest(f"triton: {exc}")


class ClearTreeCascadeCudaTest(CustomTestCase):
    def test_kernel_matches_reference_across_batch_and_width(self):
        _require_cuda()
        from sglang.srt.speculative.clear_tree_remote.clear_tree_cascade_kernel import (
            launch,
        )

        device = "cuda"
        sm = torch.cuda.get_device_properties(device).multi_processor_count
        cases = (
            (1, 2, 1, torch.float32),
            (1, 64, None, torch.float32),
            (2, 8, 1, torch.float32),
            (4, 16, None, torch.float32),
            (4, 64, 21, torch.bfloat16),
        )
        for bs, width, forced_splits, dtype in cases:
            with self.subTest(bs=bs, width=width, splits=forced_splits, dtype=dtype):
                self._compare(device, sm, bs, width, forced_splits, dtype, pad=False)

    def test_kernel_matches_reference_with_padding_row(self):
        _require_cuda()
        device = "cuda"
        sm = torch.cuda.get_device_properties(device).multi_processor_count
        self._compare(device, sm, 2, 4, 1, torch.float32, pad=True)

    def _compare(self, device, sm, bs, width, forced_splits, dtype, pad):
        from sglang.srt.speculative.clear_tree_remote.clear_tree_cascade_kernel import (
            launch,
        )

        torch.manual_seed(11 + bs * 17 + width)
        kv_heads, group, dim = 2, 2, 32
        heads = kv_heads * group
        prefixes = [3 + 5 * i for i in range(bs)]
        trees = [2 + (i % 5) for i in range(bs)]
        if pad:
            prefixes[-1] = 1
            trees[-1] = width
        seqs = [prefix + tree for prefix, tree in zip(prefixes, trees)]
        # The kernel's tree span is seq + width - prefix, so the stored mask
        # must cover the already-written tree plus this forward's width.
        mask_cols = [tree + width for tree in trees]
        max_cols = max(mask_cols)
        context = max(seq + width for seq in seqs) + 4
        pools = bs + 1
        req_to_token = torch.randint(
            0, 64, (pools, context), dtype=torch.int32, device=device
        )
        tokens = int(req_to_token.max().item()) + 1
        k_cache = torch.randn(tokens, kv_heads, dim, device=device, dtype=dtype)
        v_cache = torch.randn(tokens, kv_heads, dim, device=device, dtype=dtype)
        q = torch.randn(bs * width, heads, dim, device=device, dtype=dtype)
        tree_mask = torch.zeros((bs * width, max_cols), dtype=torch.uint8, device=device)
        prefix_rows = torch.ones(bs * width, dtype=torch.int32, device=device)
        for batch in range(bs):
            cols = mask_cols[batch]
            block = tree_mask[batch * width : (batch + 1) * width, :cols]
            block[:, 0] = 1
            for row in range(width):
                block[row, row % cols] = 1
                block[row, cols - 1] = 1
            if batch == bs - 1 and not pad:
                prefix_rows[batch * width + width - 1] = 0
            if pad and batch == bs - 1:
                block.zero_()
                prefix_rows[batch * width : (batch + 1) * width] = 1
                req_to_token[batch, : context] = 0
        prefix_len = torch.tensor(prefixes, dtype=torch.int32, device=device)
        seq_lens = torch.tensor(seqs, dtype=torch.int32, device=device)
        pools_idx = torch.arange(bs, dtype=torch.int64, device=device)
        splits = forced_splits or splits_for(bs, kv_heads, sm)
        logits = torch.empty(
            (bs * width, heads, splits, dim), dtype=torch.float32, device=device
        )
        lse = torch.full(
            (bs * width, heads, splits), -1.0e20, dtype=torch.float32, device=device
        )
        got = torch.empty_like(q)
        launch(
            q,
            got,
            k_cache,
            v_cache,
            req_to_token,
            pools_idx,
            seq_lens,
            prefix_len,
            tree_mask,
            prefix_rows,
            logits,
            lse,
            dim**-0.5,
            kv_heads,
            width,
            splits,
            bs,
        )
        ref = reference(
            q,
            k_cache,
            v_cache,
            req_to_token,
            pools_idx,
            seq_lens,
            prefix_len,
            tree_mask,
            prefix_rows,
            dim**-0.5,
            width,
        )
        atol = 2e-2 if dtype == torch.bfloat16 else 2e-3
        torch.testing.assert_close(got, ref, atol=atol, rtol=atol)
        self.assertTrue(torch.isfinite(got).all())

    def test_assign_writes_layer_and_bonus_columns_in_mask_order(self):
        _require_cuda()
        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        device = "cuda"
        pool = torch.full((2, 32), -1, dtype=torch.int32, device=device)
        # Two requests, prefixes 4 and 7. A layer of width 4, then a bonus of width 3.
        starts = torch.tensor([4, 7], dtype=torch.int32, device=device)
        layer = torch.arange(100, 108, dtype=torch.int32, device=device)
        assign_req_to_token_pool_func(
            torch.tensor([0, 1], dtype=torch.int64, device=device),
            pool,
            starts,
            starts + 4,
            layer,
            2,
        )
        bonus_at = starts + 4
        bonus = torch.arange(200, 206, dtype=torch.int32, device=device)
        assign_req_to_token_pool_func(
            torch.tensor([0, 1], dtype=torch.int64, device=device),
            pool,
            bonus_at,
            bonus_at + 3,
            bonus,
            2,
        )
        torch.cuda.synchronize()
        self.assertEqual(pool[0, 4:11].tolist(), [100, 101, 102, 103, 200, 201, 202])
        self.assertEqual(pool[1, 7:14].tolist(), [104, 105, 106, 107, 203, 204, 205])
        self.assertTrue(torch.all(pool[0, :4] == -1))
        self.assertTrue(torch.all(pool[1, :7] == -1))


if __name__ == "__main__":
    unittest.main()
