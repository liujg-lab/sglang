"""CPU unit tests for portable tree greedy / target_only verification."""

import unittest

import torch

from sglang.srt.speculative.tree_verify import (
    check_tree_verify_tensors,
    tree_speculative_sampling_target_only_ref,
    verify_tree_greedy_ref,
)
from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


def _empty_io(bs, num_draft, spec_steps, vocab=8, device="cpu"):
    predicts = torch.full((bs * num_draft,), -1, dtype=torch.int32, device=device)
    accept_index = torch.full((bs, spec_steps), -1, dtype=torch.int32, device=device)
    accept_token_num = torch.zeros((bs,), dtype=torch.int32, device=device)
    return predicts, accept_index, accept_token_num


class TestTreeVerifyRef(CustomTestCase):
    def test_greedy_accepts_sibling_not_linear_chain(self):
        # Root -> child1 (token 10) sibling child2 (token 20). Target argmax at
        # root is 20, so a chain-only verifier would reject; sibling walk accepts.
        candidates = torch.tensor([[7, 10, 20]], dtype=torch.int64)
        retrive_index = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        retrive_next_token = torch.tensor([[1, -1, -1]], dtype=torch.int64)
        retrive_next_sibling = torch.tensor([[-1, 2, -1]], dtype=torch.int64)
        target_predict = torch.tensor([[20, 3, 4]], dtype=torch.int64)
        predicts, accept_index, accept_len = _empty_io(1, 3, 2)
        verify_tree_greedy_ref(
            predicts,
            accept_index,
            accept_len,
            candidates,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            target_predict,
        )
        self.assertEqual(int(accept_len[0]), 1)
        self.assertEqual(accept_index[0, :2].tolist(), [0, 2])
        self.assertEqual(int(predicts[0]), 20)

    def test_greedy_all_reject_keeps_bonus_from_root(self):
        candidates = torch.tensor([[7, 10, 20]], dtype=torch.int64)
        retrive_index = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        retrive_next_token = torch.tensor([[1, -1, -1]], dtype=torch.int64)
        retrive_next_sibling = torch.tensor([[-1, 2, -1]], dtype=torch.int64)
        target_predict = torch.tensor([[99, 3, 4]], dtype=torch.int64)
        predicts, accept_index, accept_len = _empty_io(1, 3, 2)
        verify_tree_greedy_ref(
            predicts,
            accept_index,
            accept_len,
            candidates,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            target_predict,
        )
        self.assertEqual(int(accept_len[0]), 0)
        self.assertEqual(int(predicts[0]), 99)
        self.assertEqual(int(accept_index[0, 0]), 0)

    def test_target_only_fixed_coins_accepts_high_prob_child(self):
        vocab = 8
        candidates = torch.tensor([[7, 1, 2]], dtype=torch.int64)
        retrive_index = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        retrive_next_token = torch.tensor([[1, -1, -1]], dtype=torch.int64)
        retrive_next_sibling = torch.tensor([[-1, 2, -1]], dtype=torch.int64)
        target_probs = torch.zeros((1, 3, vocab), dtype=torch.float32)
        target_probs[0, 0, 1] = 0.9
        target_probs[0, 0, 2] = 0.1
        target_probs[0, 1, 3] = 1.0
        target_probs[0, 2, 4] = 1.0
        draft_probs = torch.zeros_like(target_probs)
        coins = torch.tensor([[0.5, 0.1, 0.1]], dtype=torch.float32)
        coins_final = torch.tensor([0.0], dtype=torch.float32)
        predicts, accept_index, accept_len = _empty_io(1, 3, 2)
        tree_speculative_sampling_target_only_ref(
            predicts,
            accept_index,
            accept_len,
            candidates,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            coins,
            coins_final,
            target_probs,
            draft_probs,
            threshold_single=1.0,
            threshold_acc=1.0,
        )
        self.assertEqual(int(accept_len[0]), 1)
        self.assertEqual(accept_index[0, :2].tolist(), [0, 1])
        self.assertEqual(int(predicts[0]), 1)

    def test_greedy_linear_chain_accepts_all(self):
        candidates = torch.tensor([[7, 10, 20]], dtype=torch.int64)
        retrive_index = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        retrive_next_token = torch.tensor([[1, 2, -1]], dtype=torch.int64)
        retrive_next_sibling = torch.tensor([[-1, -1, -1]], dtype=torch.int64)
        target_predict = torch.tensor([[10, 20, 4]], dtype=torch.int64)
        predicts, accept_index, accept_len = _empty_io(1, 3, 3)
        verify_tree_greedy_ref(
            predicts,
            accept_index,
            accept_len,
            candidates,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            target_predict,
        )
        self.assertEqual(int(accept_len[0]), 2)
        self.assertEqual(accept_index[0, :3].tolist(), [0, 1, 2])
        self.assertEqual(int(predicts[0]), 10)
        self.assertEqual(int(predicts[1]), 20)

    def test_greedy_repeated_tokens_still_walk_siblings(self):
        candidates = torch.tensor([[5, 5, 5]], dtype=torch.int64)
        retrive_index = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        retrive_next_token = torch.tensor([[1, -1, -1]], dtype=torch.int64)
        retrive_next_sibling = torch.tensor([[-1, 2, -1]], dtype=torch.int64)
        target_predict = torch.tensor([[5, 9, 9]], dtype=torch.int64)
        predicts, accept_index, accept_len = _empty_io(1, 3, 2)
        verify_tree_greedy_ref(
            predicts,
            accept_index,
            accept_len,
            candidates,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            target_predict,
        )
        self.assertEqual(int(accept_len[0]), 1)
        self.assertEqual(accept_index[0, :2].tolist(), [0, 1])

    def test_target_only_all_reject_uses_relu_bonus(self):
        vocab = 4
        candidates = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        retrive_index = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        retrive_next_token = torch.tensor([[1, -1, -1]], dtype=torch.int64)
        retrive_next_sibling = torch.tensor([[-1, 2, -1]], dtype=torch.int64)
        target_probs = torch.zeros((1, 3, vocab), dtype=torch.float32)
        target_probs[0, 0, 3] = 1.0
        draft_probs = torch.zeros_like(target_probs)
        coins = torch.tensor([[0.99, 0.99, 0.99]], dtype=torch.float32)
        coins_final = torch.tensor([0.0], dtype=torch.float32)
        predicts, accept_index, accept_len = _empty_io(1, 3, 2)
        tree_speculative_sampling_target_only_ref(
            predicts,
            accept_index,
            accept_len,
            candidates,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            coins,
            coins_final,
            target_probs,
            draft_probs,
            threshold_single=1.0,
            threshold_acc=1.0,
        )
        self.assertEqual(int(accept_len[0]), 0)
        self.assertEqual(int(predicts[0]), 3)

    def test_check_tree_verify_tensors_rejects_shape_mismatch(self):
        candidates = torch.zeros((2, 3), dtype=torch.long)
        retrive_index = torch.zeros((2, 2), dtype=torch.long)
        with self.assertRaises(ValueError):
            check_tree_verify_tensors(
                candidates=candidates, retrive_index=retrive_index
            )


if __name__ == "__main__":
    unittest.main()
