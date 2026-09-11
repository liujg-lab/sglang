"""CPU tests for Ascend tree-mask polarity conversion."""

import unittest

import torch

from sglang.srt.speculative.tree_attn_mask import (
    FIA_TREE_MASK_CONTRACT,
    custom_mask_to_ascend_masked,
    iter_full_mask_rows,
)
from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestTreeAttnMask(CustomTestCase):
    def test_cuda_true_attend_becomes_ascend_masked_false(self):
        # One request, seq_len=2, num_draft=2. Flattened FULL_MASK rows of length 4.
        # Query 0 attends prefix+self: [T, T, T, F]
        # Query 1 attends prefix+self+ancestor: [T, T, T, T]
        custom = torch.tensor(
            [True, True, True, False, True, True, True, True], dtype=torch.bool
        )
        seq_lens = torch.tensor([2], dtype=torch.int32)
        masked = custom_mask_to_ascend_masked(custom, seq_lens, num_draft=2)
        self.assertEqual(tuple(masked.shape), (2, 4))
        # Ascend True = masked
        self.assertEqual(masked[0].tolist(), [False, False, False, True])
        self.assertEqual(masked[1].tolist(), [False, False, False, False])

    def test_sibling_column_stays_masked_for_other_branch(self):
        # seq_len=1, draft=3. Query 1 cannot see draft token 2 (sibling).
        custom = torch.tensor(
            [
                True, True, False, False,  # q0: prefix + self
                True, True, True, False,  # q1: prefix + root + self, not sibling
                True, True, False, True,  # q2: prefix + root + self, not sibling 1
            ],
            dtype=torch.bool,
        )
        seq_lens = torch.tensor([1], dtype=torch.int32)
        masked = custom_mask_to_ascend_masked(custom, seq_lens, num_draft=3)
        self.assertTrue(masked[1, 3].item())  # sibling masked
        self.assertTrue(masked[2, 2].item())  # other sibling masked
        rows = list(iter_full_mask_rows(custom, [1], 3))
        self.assertEqual(len(rows), 3)

    def test_fia_contract_and_padding_is_masked(self):
        self.assertEqual(FIA_TREE_MASK_CONTRACT["ascend_true"], "masked")
        self.assertEqual(FIA_TREE_MASK_CONTRACT["sparse_mode"], 0)
        custom = torch.tensor([True, True, True], dtype=torch.bool)
        seq_lens = torch.tensor([1], dtype=torch.int32)
        masked = custom_mask_to_ascend_masked(
            custom, seq_lens, num_draft=2, max_kv=5
        )
        self.assertEqual(tuple(masked.shape), (2, 5))
        self.assertTrue(masked[:, 3:].all())


if __name__ == "__main__":
    unittest.main()
