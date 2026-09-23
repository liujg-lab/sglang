"""CPU unit tests for Relative Probability Drop (RPD) verification."""

import math
import unittest

import torch

import sglang.srt.speculative.rpd_verify as rpd
from sglang.srt.speculative.rpd_verify import (
    _rpd_backend_name,
    _verify_tree_rpd_compact,
    _verify_tree_rpd_cpu,
    edge_valid_from_logits,
    reset_compact_cross_device_bytes,
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

    register_cpu_ci(est_time=40, suite="stage-a-test-cpu")
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


def _tree_buffers(
    candidates,
    retrive_index,
    spec_width=None,
    predict_len=None,
    predict_fill=-1,
    accept_fill=-1,
):
    batch, width = candidates.shape
    if predict_len is None:
        if retrive_index.numel() == 0:
            predict_len = 1
        else:
            predict_len = max(int(retrive_index.max().item()) + 1, 1)
    if spec_width is None:
        spec_width = max(int(width), 1)
    predicts = torch.full((predict_len,), predict_fill, dtype=torch.int32)
    accept_index = torch.full((batch, spec_width), accept_fill, dtype=torch.int32)
    accept_length = torch.zeros((batch,), dtype=torch.int32)
    return predicts, accept_index, accept_length


def _run_reference_and_compact(
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    logits,
    tau,
    spec_width=None,
    predict_len=None,
    predict_fill=-1,
):
    gap_max = rpd_gap_max(tau)
    use_equality = float(tau) == 0.0
    kwargs = dict(
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        logits=logits,
        gap_max=gap_max,
        use_equality=use_equality,
    )
    ref = _tree_buffers(
        candidates,
        retrive_index,
        spec_width=spec_width,
        predict_len=predict_len,
        predict_fill=predict_fill,
    )
    compact = _tree_buffers(
        candidates,
        retrive_index,
        spec_width=spec_width,
        predict_len=predict_len,
        predict_fill=predict_fill,
    )
    _verify_tree_rpd_cpu(*ref, **kwargs)
    _verify_tree_rpd_compact(*compact, **kwargs)
    return ref, compact


def _assert_same_result(test, ref, compact):
    for got, exp in zip(compact, ref):
        test.assertEqual(got.dtype, exp.dtype)
        test.assertEqual(tuple(got.shape), tuple(exp.shape))
        test.assertTrue(torch.equal(got, exp), f"{got.tolist()} != {exp.tolist()}")


class TestRpdCompactMatchesReference(CustomTestCase):
    def _branching(self, dtype=torch.float32, vocab=8):
        case = TestRpdLongestPath()._branching_tree()
        candidates, retrive, nxt, sibling, logits = case
        wide = torch.zeros(logits.shape[0], vocab, dtype=torch.float32)
        wide[:, : logits.shape[1]] = logits
        return candidates, retrive, nxt, sibling, wide.to(dtype)

    def test_branching_tree_fp32_matches_known_path(self):
        tensors = self._branching()
        ref, compact = _run_reference_and_compact(*tensors, tau=0.2)
        _assert_same_result(self, ref, compact)
        self.assertEqual(compact[2].tolist(), [3])
        self.assertEqual(compact[1][0, :4].tolist(), [0, 4, 5, 6])
        self.assertEqual(int(compact[0][0]), 2)
        self.assertEqual(int(compact[0][4]), 5)
        self.assertEqual(int(compact[0][6]), 7)

    def test_tau_zero_matches_argmax_walk(self):
        tensors = self._branching()
        ref, compact = _run_reference_and_compact(*tensors, tau=0.0)
        _assert_same_result(self, ref, compact)
        self.assertEqual(compact[2].tolist(), [2])
        self.assertEqual(compact[1][0, :3].tolist(), [0, 1, 2])

    def test_equal_length_prefers_smaller_cumulative_gap(self):
        candidates = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int64)
        retrive = torch.arange(5, dtype=torch.int64).unsqueeze(0)
        nxt = torch.tensor([[1, 2, -1, 4, -1]], dtype=torch.int64)
        sibling = torch.tensor([[-1, 3, -1, -1, -1]], dtype=torch.int64)
        logits = torch.zeros(5, 8, dtype=torch.float32)
        logits[0, 0] = 10.0
        logits[0, 1] = 9.6
        logits[0, 3] = 9.7
        logits[1, 0] = 10.0
        logits[1, 2] = 9.6
        logits[3, 0] = 10.0
        logits[3, 4] = 9.7
        ref, compact = _run_reference_and_compact(
            candidates, retrive, nxt, sibling, logits, tau=0.5
        )
        _assert_same_result(self, ref, compact)
        self.assertEqual(compact[1][0, :3].tolist(), [0, 3, 4])
        self.assertEqual(compact[2].tolist(), [2])

    def test_equal_gap_keeps_earlier_sibling(self):
        candidates = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int64)
        retrive = torch.arange(5, dtype=torch.int64).unsqueeze(0)
        nxt = torch.tensor([[1, 2, -1, 4, -1]], dtype=torch.int64)
        sibling = torch.tensor([[-1, 3, -1, -1, -1]], dtype=torch.int64)
        logits = torch.zeros(5, 8, dtype=torch.float32)
        logits[0, 0] = 10.0
        logits[0, 1] = 9.7
        logits[0, 3] = 9.7
        logits[1, 2] = 10.0
        logits[3, 4] = 10.0
        ref, compact = _run_reference_and_compact(
            candidates, retrive, nxt, sibling, logits, tau=0.5
        )
        _assert_same_result(self, ref, compact)
        self.assertEqual(compact[1][0, :3].tolist(), [0, 1, 2])

    def test_threshold_boundary(self):
        gap = rpd_gap_max(0.2)
        star = torch.tensor(8.0, dtype=torch.float32)
        low = torch.tensor(0.0, dtype=torch.float32)
        high = star.clone()
        for _ in range(60):
            mid = torch.tensor((float(low) + float(high)) / 2, dtype=torch.float32)
            if float(star) - float(mid) <= gap:
                high = mid
            else:
                low = mid
        self.assertLess(float(low), float(high))
        self.assertLessEqual(float(star) - float(high), gap)
        self.assertGreater(float(star) - float(low), gap)
        candidates = torch.tensor([[0, 2]], dtype=torch.int64)
        retrive = torch.arange(2, dtype=torch.int64).unsqueeze(0)
        nxt = torch.tensor([[1, -1]], dtype=torch.int64)
        sibling = torch.full((1, 2), -1, dtype=torch.int64)

        def logits_with(candidate):
            logits = torch.zeros(2, 4, dtype=torch.float32)
            logits[0, 1] = star
            logits[0, 2] = candidate
            logits[1, 0] = 1.0
            return logits

        ref, compact = _run_reference_and_compact(
            candidates, retrive, nxt, sibling, logits_with(high), tau=0.2
        )
        _assert_same_result(self, ref, compact)
        self.assertEqual(compact[2].tolist(), [1])

        ref, compact = _run_reference_and_compact(
            candidates, retrive, nxt, sibling, logits_with(low), tau=0.2
        )
        _assert_same_result(self, ref, compact)
        self.assertEqual(compact[2].tolist(), [0])

    def test_tie_argmax_uses_first_index(self):
        candidates = torch.tensor([[0, 2]], dtype=torch.int64)
        retrive = torch.arange(2, dtype=torch.int64).unsqueeze(0)
        nxt = torch.tensor([[1, -1]], dtype=torch.int64)
        sibling = torch.full((1, 2), -1, dtype=torch.int64)
        logits = torch.zeros(2, 4, dtype=torch.float32)
        logits[0, 1] = 5.0
        logits[0, 2] = 5.0
        ref, compact = _run_reference_and_compact(
            candidates, retrive, nxt, sibling, logits, tau=0.0
        )
        _assert_same_result(self, ref, compact)
        self.assertEqual(compact[2].tolist(), [0])
        self.assertEqual(int(compact[0][0]), 1)

    def test_dtypes_and_random_values_match_reference(self):
        generator = torch.Generator().manual_seed(7)
        base = torch.randn(6, 32, generator=generator)
        candidates = torch.tensor([[0, 3, 5, 1, 4, 2]], dtype=torch.int64)
        retrive = torch.arange(6, dtype=torch.int64).unsqueeze(0)
        nxt = torch.tensor([[1, 2, -1, 4, 5, -1]], dtype=torch.int64)
        sibling = torch.tensor([[-1, 3, -1, -1, -1, -1]], dtype=torch.int64)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for tau in (0.0, 0.2, 0.8):
                ref, compact = _run_reference_and_compact(
                    candidates,
                    retrive,
                    nxt,
                    sibling,
                    base.to(dtype),
                    tau=tau,
                )
                _assert_same_result(self, ref, compact)

    def test_special_values_match_reference(self):
        candidates = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
        retrive = torch.arange(4, dtype=torch.int64).unsqueeze(0)
        nxt = torch.tensor([[1, 2, 3, -1]], dtype=torch.int64)
        sibling = torch.full((1, 4), -1, dtype=torch.int64)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            logits = torch.zeros(4, 8, dtype=dtype)
            logits[0, 1] = 2.0
            logits[0, 4] = float("nan")
            logits[1, 2] = 3.0
            logits[1, 0] = float("inf")
            logits[1, 6] = float("-inf")
            logits[2, :] = float("-inf")
            logits[3, 3] = float("nan")
            logits[3, 5] = float("inf")
            for tau in (0.0, 0.2):
                ref, compact = _run_reference_and_compact(
                    candidates, retrive, nxt, sibling, logits, tau=tau
                )
                _assert_same_result(self, ref, compact)

    def test_root_only_and_no_edges(self):
        candidates = torch.tensor([[4]], dtype=torch.int64)
        retrive = torch.tensor([[2]], dtype=torch.int64)
        nxt = torch.tensor([[-1]], dtype=torch.int64)
        sibling = torch.tensor([[-1]], dtype=torch.int64)
        logits = torch.zeros(4, 6, dtype=torch.float32)
        logits[2, 3] = 2.0
        ref, compact = _run_reference_and_compact(
            candidates,
            retrive,
            nxt,
            sibling,
            logits,
            tau=0.2,
            spec_width=3,
            predict_len=5,
            predict_fill=42,
        )
        _assert_same_result(self, ref, compact)
        self.assertEqual(compact[2].tolist(), [0])
        self.assertEqual(compact[1][0].tolist(), [2, -1, -1])
        self.assertEqual(int(compact[0][2]), 3)
        self.assertEqual(compact[0][:2].tolist(), [42, 42])
        self.assertEqual(compact[0][3:].tolist(), [42, 42])

    def test_empty_batch_skips_reduction_and_keeps_outputs(self):
        candidates = torch.zeros((0, 3), dtype=torch.int64)
        retrive = torch.zeros((0, 3), dtype=torch.int64)
        nxt = torch.full((0, 3), -1, dtype=torch.int64)
        sibling = nxt.clone()
        logits = torch.ones(2, 4, dtype=torch.int32)
        predicts = torch.tensor([8, 9], dtype=torch.int32)
        accept_index = torch.full((0, 3), 7, dtype=torch.int32)
        accept_length = torch.zeros((0,), dtype=torch.int32)
        original = (predicts.clone(), accept_index.clone(), accept_length.clone())

        def forbid_max(*args, **kwargs):
            raise AssertionError("empty batch reduced logits")

        original_max = torch.max
        torch.max = forbid_max
        try:
            _verify_tree_rpd_compact(
                predicts,
                accept_index,
                accept_length,
                candidates,
                retrive,
                nxt,
                sibling,
                logits,
                0.0,
                True,
            )
        finally:
            torch.max = original_max
        self.assertTrue(torch.equal(predicts, original[0]))
        self.assertTrue(torch.equal(accept_index, original[1]))
        self.assertTrue(torch.equal(accept_length, original[2]))

    def test_empty_logit_rows_do_not_clamp_or_reduce(self):
        candidates = torch.zeros((1, 2), dtype=torch.int64)
        retrive = torch.zeros((1, 2), dtype=torch.int64)
        nxt = torch.tensor([[1, -1]], dtype=torch.int64)
        sibling = torch.full((1, 2), -1, dtype=torch.int64)
        logits = torch.zeros((0, 4), dtype=torch.float32)
        predicts, accept_index, accept_length = _tree_buffers(candidates, retrive)

        def forbid_max(*args, **kwargs):
            raise AssertionError("empty rows reduced logits")

        def forbid_clamp(self, *args, **kwargs):
            raise AssertionError("empty rows were clamped")

        original_max = torch.max
        original_clamp = torch.Tensor.clamp
        torch.max = forbid_max
        torch.Tensor.clamp = forbid_clamp
        try:
            with self.assertRaises(ValueError):
                _verify_tree_rpd_compact(
                    predicts,
                    accept_index,
                    accept_length,
                    candidates,
                    retrive,
                    nxt,
                    sibling,
                    logits,
                    0.0,
                    True,
                )
        finally:
            torch.max = original_max
            torch.Tensor.clamp = original_clamp

    def test_multi_request_and_last_duplicate_write(self):
        candidates = torch.tensor([[0, 1], [0, 2]], dtype=torch.int64)
        retrive = torch.tensor([[0, 1], [0, 2]], dtype=torch.int64)
        nxt = torch.tensor([[-1, -1], [1, -1]], dtype=torch.int64)
        sibling = torch.full((2, 2), -1, dtype=torch.int64)
        logits = torch.zeros(3, 6, dtype=torch.float32)
        logits[0, 1] = 10.0
        logits[0, 2] = 9.9
        logits[2, 4] = 3.0
        ref, compact = _run_reference_and_compact(
            candidates,
            retrive,
            nxt,
            sibling,
            logits,
            tau=0.5,
            predict_fill=11,
        )
        _assert_same_result(self, ref, compact)
        self.assertEqual(int(compact[0][0]), 2)
        self.assertEqual(int(compact[0][2]), 4)
        self.assertEqual(int(compact[0][1]), 11)

    def test_same_row_bonus_overwrites_edge_token(self):
        candidates = torch.tensor([[0, 2]], dtype=torch.int64)
        retrive = torch.tensor([[0, 0]], dtype=torch.int64)
        nxt = torch.tensor([[1, -1]], dtype=torch.int64)
        sibling = torch.tensor([[-1, -1]], dtype=torch.int64)
        logits = torch.zeros(1, 4, dtype=torch.float32)
        logits[0, 1] = 10.0
        logits[0, 2] = 9.95
        ref, compact = _run_reference_and_compact(
            candidates, retrive, nxt, sibling, logits, tau=0.5
        )
        _assert_same_result(self, ref, compact)
        self.assertEqual(compact[1][0].tolist(), [0, 0])
        self.assertEqual(int(compact[0][0]), 1)

    def test_noncontiguous_retrieve_and_logits_layouts(self):
        candidates = torch.tensor([[0, 1, 3]], dtype=torch.int64)
        nxt = torch.tensor([[1, 2, -1]], dtype=torch.int64)
        sibling = torch.full((1, 3), -1, dtype=torch.int64)
        base_retrive = torch.arange(12, dtype=torch.int64).reshape(1, 12)
        retrive = base_retrive[:, ::2][:, :3]
        self.assertFalse(retrive.is_contiguous())
        logits = torch.zeros(12, 8, dtype=torch.float32)
        logits[0, 1] = 4.0
        logits[2, 3] = 4.0
        logits[4, 5] = 4.0
        ref, compact = _run_reference_and_compact(
            candidates, retrive, nxt, sibling, logits, tau=0.0
        )
        _assert_same_result(self, ref, compact)

        storage = torch.zeros(12, 16, dtype=torch.float32)
        logits_nc = storage[:, ::2]
        self.assertFalse(logits_nc.is_contiguous())
        logits_nc[:, :8] = logits
        ref, compact = _run_reference_and_compact(
            candidates, retrive, nxt, sibling, logits_nc, tau=0.0
        )
        _assert_same_result(self, ref, compact)

        logits_3d = logits.reshape(3, 4, 8)
        ref, compact = _run_reference_and_compact(
            candidates, retrive, nxt, sibling, logits_3d, tau=0.0
        )
        _assert_same_result(self, ref, compact)

        base_3d = torch.zeros(3, 4, 16, dtype=torch.float32)
        logits_3d_nc = base_3d[:, :, ::2]
        self.assertFalse(logits_3d_nc.is_contiguous())
        logits_3d_nc.copy_(logits.reshape(3, 4, 8))
        ref, compact = _run_reference_and_compact(
            candidates, retrive, nxt, sibling, logits_3d_nc, tau=0.0
        )
        _assert_same_result(self, ref, compact)

    def test_compact_keeps_output_identity(self):
        tensors = self._branching()
        predicts, accept_index, accept_length = _tree_buffers(tensors[0], tensors[1])
        ptrs = (
            predicts.data_ptr(),
            accept_index.data_ptr(),
            accept_length.data_ptr(),
        )
        dtypes = (predicts.dtype, accept_index.dtype, accept_length.dtype)
        shapes = (tuple(predicts.shape), tuple(accept_index.shape), tuple(accept_length.shape))
        _verify_tree_rpd_compact(
            predicts,
            accept_index,
            accept_length,
            *tensors,
            rpd_gap_max(0.2),
            False,
        )
        self.assertEqual(
            (predicts.data_ptr(), accept_index.data_ptr(), accept_length.data_ptr()),
            ptrs,
        )
        self.assertEqual((predicts.dtype, accept_index.dtype, accept_length.dtype), dtypes)
        self.assertEqual(
            (tuple(predicts.shape), tuple(accept_index.shape), tuple(accept_length.shape)),
            shapes,
        )
        returned = verify_tree_rpd(
            predicts, accept_index, accept_length, *tensors, 0.2
        )
        self.assertIs(returned[0], predicts)
        self.assertIs(returned[1], accept_index)
        self.assertIs(returned[2], accept_length)

    def test_illegal_edge_is_skipped_and_matches_reference(self):
        candidates = torch.tensor([[0, 99, 1]], dtype=torch.int64)
        retrive = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        nxt = torch.tensor([[1, -1, -1]], dtype=torch.int64)
        sibling = torch.tensor([[-1, 2, -1]], dtype=torch.int64)
        logits = torch.zeros(3, 8, dtype=torch.float32)
        logits[0, 1] = 4.0
        logits[2, 3] = 1.0
        ref, compact = _run_reference_and_compact(
            candidates,
            retrive,
            nxt,
            sibling,
            logits,
            tau=0.0,
            predict_fill=5,
        )
        _assert_same_result(self, ref, compact)
        self.assertEqual(compact[2].tolist(), [1])
        self.assertEqual(int(compact[0][1]), 5)

    def test_accepted_retrieve_out_of_range_fails_before_upload(self):
        cases = (
            torch.tensor([[-1, 0]], dtype=torch.int64),
            torch.tensor([[0, -1]], dtype=torch.int64),
            torch.tensor([[0, 9]], dtype=torch.int64),
        )
        for retrive in cases:
            candidates = torch.tensor([[1, 1]], dtype=torch.int64)
            nxt = torch.tensor([[1, -1]], dtype=torch.int64)
            sibling = torch.full((1, 2), -1, dtype=torch.int64)
            logits = torch.zeros(2, 4, dtype=torch.float32)
            logits[0, 1] = 5.0
            logits[1, 2] = 5.0
            predicts, accept_index, accept_length = _tree_buffers(
                candidates, torch.tensor([[0, 1]]), predict_fill=23, accept_fill=23
            )
            before = (
                predicts.clone(),
                accept_index.clone(),
                accept_length.clone(),
            )
            calls = []
            original = rpd._rpd_compact_apply

            def record(*args, **kwargs):
                calls.append(1)
                return original(*args, **kwargs)

            rpd._rpd_compact_apply = record
            try:
                with self.assertRaises(ValueError):
                    _verify_tree_rpd_compact(
                        predicts,
                        accept_index,
                        accept_length,
                        candidates,
                        retrive,
                        nxt,
                        sibling,
                        logits,
                        0.0,
                        True,
                    )
            finally:
                rpd._rpd_compact_apply = original
            self.assertEqual(calls, [])
            self.assertTrue(torch.equal(predicts, before[0]))
            self.assertTrue(torch.equal(accept_index, before[1]))
            self.assertTrue(torch.equal(accept_length, before[2]))

    def test_short_accept_buffer_fails_before_upload(self):
        candidates = torch.tensor([[1, 1, 1]], dtype=torch.int64)
        retrive = torch.arange(3, dtype=torch.int64).unsqueeze(0)
        nxt = torch.tensor([[1, 2, -1]], dtype=torch.int64)
        sibling = torch.full((1, 3), -1, dtype=torch.int64)
        logits = torch.zeros(3, 4, dtype=torch.float32)
        logits[0, 1] = 2.0
        logits[1, 1] = 2.0
        logits[2, 0] = 2.0
        predicts, accept_index, accept_length = _tree_buffers(
            candidates, retrive, spec_width=2, predict_fill=8
        )
        before = (predicts.clone(), accept_index.clone(), accept_length.clone())
        calls = []
        original = rpd._rpd_compact_apply

        def record(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        rpd._rpd_compact_apply = record
        try:
            with self.assertRaises(RuntimeError):
                _verify_tree_rpd_compact(
                    predicts,
                    accept_index,
                    accept_length,
                    candidates,
                    retrive,
                    nxt,
                    sibling,
                    logits,
                    0.0,
                    True,
                )
        finally:
            rpd._rpd_compact_apply = original
        self.assertEqual(calls, [])
        self.assertTrue(torch.equal(predicts, before[0]))
        self.assertTrue(torch.equal(accept_index, before[1]))
        self.assertTrue(torch.equal(accept_length, before[2]))

    def test_unsupported_dtype_and_width_raise(self):
        candidates = torch.zeros((1, 1), dtype=torch.int64)
        retrive = torch.zeros((1, 1), dtype=torch.int64)
        nxt = torch.full((1, 1), -1, dtype=torch.int64)
        sibling = nxt.clone()
        predicts, accept_index, accept_length = _tree_buffers(candidates, retrive)
        with self.assertRaises(ValueError):
            _verify_tree_rpd_compact(
                predicts,
                accept_index,
                accept_length,
                candidates,
                retrive,
                nxt,
                sibling,
                torch.zeros(1, 4, dtype=torch.int32),
                0.0,
                True,
            )
        with self.assertRaises(ValueError):
            _verify_tree_rpd_compact(
                predicts,
                accept_index,
                accept_length,
                torch.zeros((1, 0), dtype=torch.int64),
                torch.zeros((1, 0), dtype=torch.int64),
                torch.full((1, 0), -1, dtype=torch.int64),
                torch.full((1, 0), -1, dtype=torch.int64),
                torch.zeros(1, 4, dtype=torch.float32),
                0.0,
                True,
            )


class TestRpdCompactTransfers(CustomTestCase):
    def _run_with_stub(self, tensors, tau, spec_width=None, predict_len=None):
        reads = []
        writes = []
        original_read = rpd._compact_host_read
        original_write = rpd._compact_device_write

        def read(tensor):
            reads.append(tensor.detach().cpu().clone())
            return original_read(tensor)

        def write(tensor, device):
            writes.append(tensor.detach().cpu().clone())
            return original_write(tensor, device)

        rpd._compact_host_read = read
        rpd._compact_device_write = write
        try:
            ref, compact = _run_reference_and_compact(
                *tensors, tau=tau, spec_width=spec_width, predict_len=predict_len
            )
        finally:
            rpd._compact_host_read = original_read
            rpd._compact_device_write = original_write
        return reads, writes, ref, compact

    def test_cpu_cross_device_bytes_are_zero(self):
        tensors = TestRpdCompactMatchesReference()._branching(vocab=64)
        reset_compact_cross_device_bytes()
        _run_reference_and_compact(*tensors, tau=0.2)
        from sglang.srt.speculative.rpd_verify import compact_cross_device_bytes

        self.assertEqual(compact_cross_device_bytes(), 0)

    def test_edged_batch_reads_tree_then_stats_once(self):
        tensors = TestRpdCompactMatchesReference()._branching()
        rpd.reset_compact_stat_waits()
        reads, writes, ref, compact = self._run_with_stub(tensors, tau=0.2)
        _assert_same_result(self, ref, compact)
        self.assertEqual(rpd.compact_stat_waits(), 1)
        self.assertEqual(len(reads), 3)
        self.assertEqual(len(writes), 2)
        self.assertEqual(tuple(reads[0].shape), (4, 1, 7))
        self.assertEqual(tuple(reads[1].shape), (1, 7))
        self.assertEqual(int(writes[0].shape[0]), 2)
        self.assertEqual(tuple(reads[2].shape), (2, int(writes[0].shape[1])))
        self.assertEqual(reads[2].dtype, tensors[-1].dtype)
        self.assertGreater(writes[0].shape[1], 0)
        logits = tensors[-1]
        for tensor in reads + writes:
            self.assertNotEqual(tuple(tensor.shape), tuple(logits.shape))
            self.assertNotEqual(int(tensor.numel()), int(logits.numel()))
            self.assertFalse(
                tensor.dim() == 2 and int(tensor.shape[-1]) == int(logits.shape[-1])
            )

    def test_tree_read_starts_before_vocab_max(self):
        tensors = TestRpdCompactMatchesReference()._branching()
        order = []
        original_max = torch.max
        original_read = rpd._compact_host_read

        def wrapped_max(*args, **kwargs):
            order.append("max")
            return original_max(*args, **kwargs)

        def wrapped_read(tensor):
            order.append(("read", tuple(tensor.shape)))
            return original_read(tensor)

        torch.max = wrapped_max
        rpd._compact_host_read = wrapped_read
        try:
            _run_reference_and_compact(*tensors, tau=0.2)
        finally:
            torch.max = original_max
            rpd._compact_host_read = original_read
        self.assertEqual(order[0], ("read", (4, 1, 7)))
        self.assertLess(order.index(("read", (4, 1, 7))), order.index("max"))

    def test_root_only_skips_edge_transfers(self):
        candidates = torch.tensor([[4]], dtype=torch.int64)
        retrive = torch.tensor([[0]], dtype=torch.int64)
        nxt = torch.tensor([[-1]], dtype=torch.int64)
        sibling = torch.tensor([[-1]], dtype=torch.int64)
        logits = torch.zeros(1, 32, dtype=torch.float32)
        logits[0, 3] = 1.0
        rpd.reset_compact_stat_waits()
        reads, writes, _ref, _compact = self._run_with_stub(
            (candidates, retrive, nxt, sibling, logits), tau=0.2, spec_width=2
        )
        self.assertEqual(rpd.compact_stat_waits(), 1)
        self.assertEqual(len(reads), 2)
        self.assertEqual(len(writes), 1)
        self.assertEqual(tuple(reads[0].shape), (4, 1, 1))
        self.assertEqual(tuple(reads[1].shape), (1, 1))
        self.assertFalse(any(tuple(tensor.shape)[:1] == (2,) and tensor.dim() == 2 for tensor in reads + writes))

    def test_transfer_bytes_do_not_grow_with_vocab(self):
        small = TestRpdCompactMatchesReference()._branching(vocab=16)
        large = TestRpdCompactMatchesReference()._branching(vocab=128)
        reads_s, writes_s, _, _ = self._run_with_stub(small, tau=0.2)
        reads_l, writes_l, _, _ = self._run_with_stub(large, tau=0.2)

        def nbytes(chunks):
            return sum(int(t.numel()) * int(t.element_size()) for t in chunks)

        self.assertEqual(nbytes(reads_s + writes_s), nbytes(reads_l + writes_l))
        self.assertNotEqual(small[-1].numel(), large[-1].numel())

    def test_compact_does_not_materialize_full_logits(self):
        tensors = TestRpdCompactMatchesReference()._branching(vocab=24)
        logits = tensors[-1]
        full = tuple(logits.shape)
        seen = []
        original_float = torch.Tensor.float
        original_cpu = torch.Tensor.cpu
        original_contig = torch.Tensor.contiguous
        original_reshape = torch.Tensor.reshape
        original_getitem = torch.Tensor.__getitem__

        def wrap_float(self, *args, **kwargs):
            if tuple(self.shape) == full:
                seen.append(("float", full))
            return original_float(self, *args, **kwargs)

        def wrap_cpu(self, *args, **kwargs):
            if tuple(self.shape) == full:
                seen.append(("cpu", full))
            return original_cpu(self, *args, **kwargs)

        def wrap_contig(self, *args, **kwargs):
            if tuple(self.shape) == full:
                seen.append(("contiguous", full))
            return original_contig(self, *args, **kwargs)

        def wrap_reshape(self, *args, **kwargs):
            if tuple(self.shape) == full:
                seen.append(("reshape", full))
            return original_reshape(self, *args, **kwargs)

        def wrap_getitem(self, key):
            if tuple(self.shape) == full and not isinstance(key, tuple):
                seen.append(("row", full))
            return original_getitem(self, key)

        torch.Tensor.float = wrap_float
        torch.Tensor.cpu = wrap_cpu
        torch.Tensor.contiguous = wrap_contig
        torch.Tensor.reshape = wrap_reshape
        torch.Tensor.__getitem__ = wrap_getitem
        try:
            predicts, accept_index, accept_length = _tree_buffers(tensors[0], tensors[1])
            _verify_tree_rpd_compact(
                predicts,
                accept_index,
                accept_length,
                *tensors,
                rpd_gap_max(0.2),
                False,
            )
        finally:
            torch.Tensor.float = original_float
            torch.Tensor.cpu = original_cpu
            torch.Tensor.contiguous = original_contig
            torch.Tensor.reshape = original_reshape
            torch.Tensor.__getitem__ = original_getitem
        self.assertEqual(seen, [])

    def test_dispatch_by_device_type(self):
        self.assertEqual(_rpd_backend_name("npu", True), "npu")
        self.assertEqual(_rpd_backend_name("npu", False), "npu")
        self.assertEqual(_rpd_backend_name("cuda", True), "cuda")
        self.assertEqual(_rpd_backend_name("cuda", False), "cuda")
        self.assertEqual(_rpd_backend_name("cpu", False), "cpu")

        called = []
        original = rpd._verify_tree_rpd_compact

        def boom(**kwargs):
            called.append(1)
            raise AssertionError("cpu dispatch entered compact")

        rpd._verify_tree_rpd_compact = boom
        try:
            case = TestRpdLongestPath()._branching_tree()
            _run_rpd(*case, tau=0.2)
        finally:
            rpd._verify_tree_rpd_compact = original
        self.assertEqual(called, [])

    @unittest.skipUnless(torch.cuda.is_available(), "cuda not available")
    def test_cuda_dispatch_uses_kernel_and_import_error_falls_back(self):
        case = [t.cuda() for t in TestRpdLongestPath()._branching_tree()]
        batch, width = case[0].shape
        total = int(case[1].max().item()) + 1
        predicts = torch.full((total,), -1, dtype=torch.int32, device="cuda")
        accept_index = torch.full((batch, width), -1, dtype=torch.int32, device="cuda")
        accept_length = torch.zeros((batch,), dtype=torch.int32, device="cuda")
        entered = []
        original_cuda = rpd._verify_tree_rpd_cuda
        original_compact = rpd._verify_tree_rpd_compact

        def fake_cuda(**kwargs):
            entered.append(kwargs["logits"].device.type)

        def forbid_compact(**kwargs):
            raise AssertionError("cuda dispatch entered compact")

        rpd._verify_tree_rpd_cuda = fake_cuda
        rpd._verify_tree_rpd_compact = forbid_compact
        try:
            verify_tree_rpd(
                predicts, accept_index, accept_length, *case, 0.2
            )
            self.assertEqual(entered, ["cuda"])
            entered.clear()

            def missing(**kwargs):
                raise ImportError("kernel missing")

            rpd._verify_tree_rpd_cuda = missing
            verify_tree_rpd(
                predicts, accept_index, accept_length, *case, 0.2
            )
        finally:
            rpd._verify_tree_rpd_cuda = original_cuda
            rpd._verify_tree_rpd_compact = original_compact
        self.assertEqual(entered, [])
        cpu_predicts, cpu_index, cpu_length = _run_rpd(
            *[t.cpu() for t in TestRpdLongestPath()._branching_tree()], tau=0.2
        )
        self.assertEqual(accept_length.tolist(), cpu_length.tolist())
        self.assertEqual(accept_index[0, :4].tolist(), cpu_index[0, :4].tolist())
        self.assertEqual(int(predicts[0].item()), int(cpu_predicts[0].item()))


if __name__ == "__main__":
    unittest.main()
