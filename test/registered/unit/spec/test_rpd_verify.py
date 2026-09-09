"""CPU unit tests for Relative Probability Drop (RPD) verification."""

import math
import unittest

import torch

from sglang.srt.speculative.rpd_verify import (
    edge_valid_from_logits,
    rpd_gap_max,
    verify_tree_rpd,
)

try:
    from sglang.srt.server_args import ServerArgs
except ImportError:
    ServerArgs = None

try:
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase

    register_cpu_ci(est_time=10, suite="stage-a-test-cpu")
except ImportError:
    CustomTestCase = unittest.TestCase


def _run_rpd(
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    logits,
    tau,
    spec_width=None,
):
    bs, n = candidates.shape
    tot = int(retrive_index.max().item()) + 1
    if spec_width is None:
        spec_width = n
    predicts = torch.full((tot,), -1, dtype=torch.int32)
    accept_index = torch.full((bs, spec_width), -1, dtype=torch.int32)
    accept_length = torch.zeros((bs,), dtype=torch.int32)
    return verify_tree_rpd(
        predicts,
        accept_index,
        accept_length,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        logits,
        tau,
    )


class TestRpdMath(CustomTestCase):
    def test_gap_max_formula(self):
        self.assertEqual(rpd_gap_max(0.0), 0.0)
        self.assertAlmostEqual(rpd_gap_max(0.2), -math.log(0.8), places=6)

    def test_gap_max_rejects_invalid_tau(self):
        with self.assertRaises(ValueError):
            rpd_gap_max(1.0)
        with self.assertRaises(ValueError):
            rpd_gap_max(-0.1)

    def test_edge_valid_matches_gap_threshold(self):
        tau = 0.2
        g_max = rpd_gap_max(tau)
        z_star = 10.0
        z_ok = z_star - g_max + 1e-6
        z_bad = z_star - g_max - 1e-6
        self.assertTrue(edge_valid_from_logits(z_star, z_ok, tau))
        self.assertFalse(edge_valid_from_logits(z_star, z_bad, tau))
        self.assertTrue(edge_valid_from_logits(z_star, z_star - g_max, tau))


class TestRpdLongestPath(CustomTestCase):
    def _branching_tree(self):
        """root: A(c*)->C->D(invalid) vs B->E->F (all valid for tau=0.2)."""
        n = 7
        candidates = torch.tensor([[0, 1, 3, 4, 2, 5, 6]], dtype=torch.int64)
        retrive_index = torch.arange(n, dtype=torch.int64).unsqueeze(0)
        retrive_next_token = torch.tensor(
            [[1, 2, 3, -1, 5, 6, -1]], dtype=torch.int64
        )
        retrive_next_sibling = torch.tensor(
            [[-1, 4, -1, -1, -1, -1, -1]], dtype=torch.int64
        )
        logits = torch.zeros(n, 8, dtype=torch.float32)
        logits[0, 1] = 10.0
        logits[0, 2] = 9.85
        logits[1, 3] = 10.0
        logits[2, 7] = 10.0
        logits[4, 5] = 10.0
        logits[5, 6] = 10.0
        logits[6, 7] = 10.0
        return (
            candidates,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            logits,
        )

    def test_longest_path_beats_top1_branch(self):
        tensors = self._branching_tree()
        predicts, accept_index, accept_length = _run_rpd(*tensors, tau=0.2)
        self.assertEqual(accept_length.tolist(), [3])
        self.assertEqual(accept_index[0, :4].tolist(), [0, 4, 5, 6])
        self.assertEqual(int(predicts[0]), 2)
        self.assertEqual(int(predicts[4]), 5)
        self.assertEqual(int(predicts[5]), 6)
        self.assertEqual(int(predicts[6]), 7)

    def test_tau_zero_uses_argmax_equality(self):
        tensors = self._branching_tree()
        predicts, accept_index, accept_length = _run_rpd(*tensors, tau=0.0)
        self.assertEqual(accept_length.tolist(), [2])
        self.assertEqual(accept_index[0, :3].tolist(), [0, 1, 2])
        self.assertEqual(int(predicts[0]), 1)
        self.assertEqual(int(predicts[1]), 3)
        self.assertEqual(int(predicts[2]), 7)


class TestRpdGreedyChain(CustomTestCase):
    def test_tau_zero_chain_matches_left_cstar_walk(self):
        n = 4
        candidates = torch.tensor([[9, 10, 11, 12]], dtype=torch.int64)
        retrive_index = torch.arange(n, dtype=torch.int64).unsqueeze(0)
        retrive_next_token = torch.tensor([[1, 2, 3, -1]], dtype=torch.int64)
        retrive_next_sibling = torch.full((1, n), -1, dtype=torch.int64)
        logits = torch.zeros(n, 16, dtype=torch.float32)
        logits[0, 10] = 5.0
        logits[1, 11] = 5.0
        logits[2, 12] = 5.0
        logits[3, 1] = 5.0
        predicts, accept_index, accept_length = _run_rpd(
            candidates,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            logits,
            tau=0.0,
        )
        self.assertEqual(accept_length.tolist(), [3])
        self.assertEqual(accept_index[0, :4].tolist(), [0, 1, 2, 3])
        self.assertEqual(int(predicts[0]), 10)
        self.assertEqual(int(predicts[1]), 11)
        self.assertEqual(int(predicts[2]), 12)
        self.assertEqual(int(predicts[3]), 1)

    def test_verify_rejects_invalid_tau(self):
        n = 2
        candidates = torch.tensor([[0, 1]], dtype=torch.int64)
        retrive_index = torch.arange(n, dtype=torch.int64).unsqueeze(0)
        next_token = torch.tensor([[1, -1]], dtype=torch.int64)
        sibling = torch.full((1, n), -1, dtype=torch.int64)
        logits = torch.zeros(n, 4, dtype=torch.float32)
        logits[0, 1] = 1.0
        with self.assertRaises(ValueError):
            _run_rpd(
                candidates,
                retrive_index,
                next_token,
                sibling,
                logits,
                tau=1.0,
            )


class TestRpdServerArgs(CustomTestCase):
    @unittest.skipIf(ServerArgs is None, "sglang.srt.server_args is not importable")
    def test_defaults_keep_auto_mode(self):
        args = ServerArgs(model_path="dummy")
        self.assertEqual(args.speculative_verify_mode, "auto")
        self.assertEqual(args.speculative_rpd_tau, 0.2)

    @unittest.skipIf(ServerArgs is None, "sglang.srt.server_args is not importable")
    def test_invalid_tau_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            ServerArgs(model_path="dummy", speculative_rpd_tau=1.0)
        self.assertIn("speculative-rpd-tau", str(ctx.exception))

    @unittest.skipIf(ServerArgs is None, "sglang.srt.server_args is not importable")
    def test_invalid_verify_mode_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            ServerArgs(model_path="dummy", speculative_verify_mode="nope")
        self.assertIn("speculative-verify-mode", str(ctx.exception))

    @unittest.skipIf(ServerArgs is None, "sglang.srt.server_args is not importable")
    def test_rpd_mode_accepted(self):
        args = ServerArgs(
            model_path="dummy",
            speculative_verify_mode="rpd",
            speculative_rpd_tau=0.0,
        )
        self.assertEqual(args.speculative_verify_mode, "rpd")
        self.assertEqual(args.speculative_rpd_tau, 0.0)


if __name__ == "__main__":
    unittest.main()
