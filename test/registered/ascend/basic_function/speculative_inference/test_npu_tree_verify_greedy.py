"""NPU sibling-walk greedy verify vs the CPU reference.

Requires torch_npu. Missing NPU runtime skips locally; NPU CI must run these
tests rather than treat a missing runtime as a pass of the device kernel.
"""

from __future__ import annotations

import time
import unittest

import torch

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=120, suite="stage-b-test-1-npu-a2")

_SENTINEL = 777


def _npu_available() -> bool:
    try:
        return bool(getattr(torch, "npu", None) and torch.npu.is_available())
    except Exception:
        return False


def _npu_device():
    if not _npu_available():
        raise unittest.SkipTest("torch_npu is not available")
    return torch.device("npu:0")


def _empty_outputs(bs, width, path_cap, extra, device):
    predicts = torch.full(
        (bs * width + extra,), _SENTINEL, dtype=torch.int32, device=device
    )
    accept_index = torch.full((bs, path_cap), -1, dtype=torch.int32, device=device)
    accept_token_num = torch.zeros((bs,), dtype=torch.int32, device=device)
    return predicts, accept_index, accept_token_num


def _base_tree(bs, width, device):
    candidates = torch.zeros((bs, width), dtype=torch.int64, device=device)
    retrive_index = torch.arange(bs * width, device=device, dtype=torch.int64).reshape(
        bs, width
    )
    next_token = torch.full((bs, width), -1, dtype=torch.int64, device=device)
    sibling = torch.full((bs, width), -1, dtype=torch.int64, device=device)
    target = torch.zeros((bs, width), dtype=torch.int64, device=device)
    return candidates, retrive_index, next_token, sibling, target


def _clone_io(predicts, accept_index, accept_token_num):
    return predicts.clone(), accept_index.clone(), accept_token_num.clone()


def _run_ref_and_npu(predicts, accept_index, accept_token_num, **tree):
    from sglang.srt.speculative.tree_verify import verify_tree_greedy_ref
    from sglang.srt.speculative.tree_verify_npu import verify_tree_greedy_npu

    ref_p, ref_a, ref_n = _clone_io(predicts, accept_index, accept_token_num)
    npu_p, npu_a, npu_n = _clone_io(predicts, accept_index, accept_token_num)
    verify_tree_greedy_ref(
        ref_p,
        ref_a,
        ref_n,
        tree["candidates"],
        tree["retrive_index"],
        tree["retrive_next_token"],
        tree["retrive_next_sibling"],
        tree["target_predict"],
    )
    verify_tree_greedy_npu(
        npu_p,
        npu_a,
        npu_n,
        tree["candidates"],
        tree["retrive_index"],
        tree["retrive_next_token"],
        tree["retrive_next_sibling"],
        tree["target_predict"],
    )
    torch.testing.assert_close(npu_p.cpu(), ref_p.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(npu_a.cpu(), ref_a.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(npu_n.cpu(), ref_n.cpu(), atol=0, rtol=0)
    return npu_p, npu_a, npu_n, ref_p, ref_a, ref_n


def _sibling_second_hit(bs, device, extra=0, width=3, path_cap=2):
    predicts, accept_index, accept_token_num = _empty_outputs(
        bs, width, path_cap, extra, device
    )
    candidates, retrive_index, next_token, sibling, target = _base_tree(
        bs, width, device
    )
    candidates[:, 0] = 7
    candidates[:, 1] = 10
    candidates[:, 2] = 20
    next_token[:, 0] = 1
    sibling[:, 1] = 2
    target[:, 0] = 20
    target[:, 1] = 3
    target[:, 2] = 4
    return predicts, accept_index, accept_token_num, dict(
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=next_token,
        retrive_next_sibling=sibling,
        target_predict=target,
    )


class TestNpuTreeVerifyGreedy(CustomTestCase):
    def test_handwritten_sibling_and_bonus(self):
        device = _npu_device()
        for bs in (1, 2, 3, 4):
            with self.subTest(bs=bs):
                predicts, acc, num, tree = _sibling_second_hit(bs, device)
                npu_p, npu_a, npu_n, *_ = _run_ref_and_npu(predicts, acc, num, **tree)
                self.assertTrue(torch.all(npu_n == 1).item())
                self.assertEqual(npu_a[0, :2].cpu().tolist(), [0, 2])
                self.assertEqual(int(npu_p[0].cpu()), 20)

    def test_dynamic_batch_order(self):
        device = _npu_device()
        for bs in (1, 2, 4, 3, 1):
            predicts, acc, num, tree = _sibling_second_hit(bs, device)
            _run_ref_and_npu(predicts, acc, num, **tree)

    def test_w15_l6_chain_root_and_widths(self):
        device = _npu_device()
        bs, width, path_cap = 2, 15, 6
        predicts, acc, num = _empty_outputs(bs, width, path_cap, 0, device)
        candidates, retrive_index, next_token, sibling, target = _base_tree(
            bs, width, device
        )
        for node in range(width - 1):
            next_token[:, node] = node + 1
        candidates[:, :] = torch.arange(width, device=device, dtype=torch.int64)
        for node in range(path_cap - 1):
            target[:, node] = node + 1
        _run_ref_and_npu(
            predicts,
            acc,
            num,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=next_token,
            retrive_next_sibling=sibling,
            target_predict=target,
        )

        predicts, acc, num = _empty_outputs(1, 1, 1, 0, device)
        candidates, retrive_index, next_token, sibling, target = _base_tree(
            1, 1, device
        )
        target[:, 0] = 9
        npu_p, npu_a, npu_n, *_ = _run_ref_and_npu(
            predicts,
            acc,
            num,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=next_token,
            retrive_next_sibling=sibling,
            target_predict=target,
        )
        self.assertEqual(int(npu_n[0].cpu()), 0)
        self.assertEqual(int(npu_p[0].cpu()), 9)

    def test_last_and_deep_sibling_and_repeat_token(self):
        device = _npu_device()
        predicts, acc, num = _empty_outputs(1, 4, 3, 0, device)
        candidates, retrive_index, next_token, sibling, target = _base_tree(
            1, 4, device
        )
        candidates[0] = torch.tensor([7, 10, 11, 20], device=device)
        next_token[0, 0] = 1
        sibling[0, 1] = 2
        sibling[0, 2] = 3
        target[0, 0] = 20
        _run_ref_and_npu(
            predicts,
            acc,
            num,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=next_token,
            retrive_next_sibling=sibling,
            target_predict=target,
        )

        predicts, acc, num = _empty_outputs(1, 5, 3, 0, device)
        candidates, retrive_index, next_token, sibling, target = _base_tree(
            1, 5, device
        )
        candidates[0] = torch.tensor([1, 2, 3, 4, 5], device=device)
        next_token[0, 0] = 1
        next_token[0, 2] = 3
        sibling[0, 1] = 2
        sibling[0, 3] = 4
        target[0, 0] = 3
        target[0, 2] = 5
        _run_ref_and_npu(
            predicts,
            acc,
            num,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=next_token,
            retrive_next_sibling=sibling,
            target_predict=target,
        )

        predicts, acc, num = _empty_outputs(1, 3, 2, 0, device)
        candidates, retrive_index, next_token, sibling, target = _base_tree(
            1, 3, device
        )
        candidates.fill_(5)
        next_token[0, 0] = 1
        sibling[0, 1] = 2
        target[0, 0] = 5
        npu_p, npu_a, npu_n, *_ = _run_ref_and_npu(
            predicts,
            acc,
            num,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=next_token,
            retrive_next_sibling=sibling,
            target_predict=target,
        )
        self.assertEqual(npu_a[0, :2].cpu().tolist(), [0, 1])

    def test_reject_accept_leaf_cap_bonus_padding(self):
        device = _npu_device()
        predicts, acc, num, tree = _sibling_second_hit(1, device)
        tree["target_predict"].fill_(99)
        npu_p, npu_a, npu_n, *_ = _run_ref_and_npu(predicts, acc, num, **tree)
        self.assertEqual(int(npu_n[0].cpu()), 0)
        self.assertEqual(int(npu_p[0].cpu()), 99)
        self.assertEqual(int(npu_p[1].cpu()), _SENTINEL)

        predicts, acc, num = _empty_outputs(1, 4, 2, 0, device)
        candidates, retrive_index, next_token, sibling, target = _base_tree(
            1, 4, device
        )
        candidates[0] = torch.tensor([1, 2, 3, 4], device=device)
        next_token[0, 0] = 1
        next_token[0, 1] = 2
        target[0, 0] = 2
        target[0, 1] = 3
        npu_p, npu_a, npu_n, *_ = _run_ref_and_npu(
            predicts,
            acc,
            num,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=next_token,
            retrive_next_sibling=sibling,
            target_predict=target,
        )
        self.assertEqual(int(npu_n[0].cpu()), 1)
        self.assertEqual(npu_a[0].cpu().tolist(), [0, 1])

    def test_nonlinear_index_offset_and_reuse(self):
        device = _npu_device()
        predicts = torch.full((10,), _SENTINEL, dtype=torch.int32, device=device)
        acc = torch.full((2, 2), -1, dtype=torch.int32, device=device)
        num = torch.zeros((2,), dtype=torch.int32, device=device)
        candidates = torch.tensor([[7, 10, 20], [7, 10, 20]], dtype=torch.int64, device=device)
        retrive_index = torch.tensor([[4, 1, 6], [7, 2, 9]], dtype=torch.int64, device=device)
        next_token = torch.tensor([[1, -1, -1], [1, -1, -1]], dtype=torch.int64, device=device)
        sibling = torch.tensor([[-1, 2, -1], [-1, 2, -1]], dtype=torch.int64, device=device)
        target = torch.zeros((2, 3), dtype=torch.int64, device=device)
        target_flat = torch.full((10,), 3, dtype=torch.int64, device=device)
        target_flat[4] = 20
        target_flat[6] = 8
        target_flat[7] = 20
        target_flat[9] = 8
        npu_p, npu_a, npu_n, *_ = _run_ref_and_npu(
            predicts,
            acc,
            num,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=next_token,
            retrive_next_sibling=sibling,
            target_predict=target_flat,
        )
        self.assertEqual(int(npu_p[3].cpu()), _SENTINEL)
        npu_p.fill_(_SENTINEL)
        npu_a.fill_(-1)
        npu_n.fill_(0)
        _run_ref_and_npu(
            npu_p,
            npu_a,
            npu_n,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=next_token,
            retrive_next_sibling=sibling,
            target_predict=target_flat,
        )

    def test_caller_predicts_lengths_and_noncontiguous_fallback(self):
        device = _npu_device()
        predicts, acc, num, tree = _sibling_second_hit(1, device, extra=1)
        npu_p, *_rest = _run_ref_and_npu(predicts, acc, num, **tree)
        self.assertEqual(int(npu_p[-1].cpu()), _SENTINEL)

        from sglang.srt.speculative.eagle_utils import (
            greedy_verify_path_info,
            verify_tree_greedy_func,
        )

        base = torch.zeros((1, 6), dtype=torch.int64, device=device)
        noncontig = base[:, ::2]
        self.assertFalse(noncontig.is_contiguous())
        noncontig.copy_(tree["candidates"])
        case = dict(tree)
        case["candidates"] = noncontig
        predicts, acc, num = _empty_outputs(1, 3, 2, 0, device)
        verify_tree_greedy_func(
            predicts,
            acc,
            num,
            case["candidates"],
            case["retrive_index"],
            case["retrive_next_token"],
            case["retrive_next_sibling"],
            case["target_predict"],
        )
        path, reason = greedy_verify_path_info()
        self.assertEqual(path, "cpu_reference")
        self.assertEqual(reason, "noncontiguous layout")

    def test_random_legal_trees(self):
        device = _npu_device()
        rng = torch.Generator(device="cpu")
        rng.manual_seed(20260921)
        for bs, width, path_cap in ((1, 8, 4), (2, 15, 6), (4, 6, 3)):
            predicts, acc, num = _empty_outputs(bs, width, path_cap, 0, device)
            candidates, retrive_index, next_token, sibling, target = _base_tree(
                bs, width, device
            )
            candidates.copy_(
                torch.randint(1, 30, (bs, width), generator=rng, dtype=torch.int64)
            )
            target.copy_(
                torch.randint(1, 30, (bs, width), generator=rng, dtype=torch.int64)
            )
            for b in range(bs):
                unused = list(range(1, width))
                parents = [0]
                for node in unused:
                    parent = parents[(node - 1) % len(parents)]
                    if int(next_token[b, parent].cpu()) < 0:
                        next_token[b, parent] = node
                    else:
                        cur = int(next_token[b, parent].cpu())
                        while int(sibling[b, cur].cpu()) >= 0:
                            cur = int(sibling[b, cur].cpu())
                        sibling[b, cur] = node
                    parents.append(node)
            _run_ref_and_npu(
                predicts,
                acc,
                num,
                candidates=candidates,
                retrive_index=retrive_index,
                retrive_next_token=next_token,
                retrive_next_sibling=sibling,
                target_predict=target,
            )

    def test_wall_clock_microbench_same_metric(self):
        device = _npu_device()
        from sglang.srt.speculative.tree_verify import verify_tree_greedy_ref
        from sglang.srt.speculative.tree_verify_npu import verify_tree_greedy_npu

        predicts, acc, num = _empty_outputs(4, 15, 6, 0, device)
        candidates, retrive_index, next_token, sibling, target = _base_tree(
            4, 15, device
        )
        next_token[:, 0] = 1
        sibling[:, 1] = 2
        candidates[:, 1] = 10
        candidates[:, 2] = 20
        target[:, 0] = 20
        tree = dict(
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=next_token,
            retrive_next_sibling=sibling,
            target_predict=target,
        )
        p0, a0, n0 = _clone_io(predicts, acc, num)
        verify_tree_greedy_npu(p0, a0, n0, **tree)
        torch.npu.synchronize()
        iters = 20
        p1, a1, n1 = _clone_io(predicts, acc, num)
        t0 = time.perf_counter()
        for _ in range(iters):
            p1.fill_(_SENTINEL)
            a1.fill_(-1)
            n1.zero_()
            verify_tree_greedy_ref(p1, a1, n1, **tree)
        torch.npu.synchronize()
        cpu_ms = (time.perf_counter() - t0) * 1000 / iters
        p2, a2, n2 = _clone_io(predicts, acc, num)
        t0 = time.perf_counter()
        for _ in range(iters):
            p2.fill_(_SENTINEL)
            a2.fill_(-1)
            n2.zero_()
            verify_tree_greedy_npu(p2, a2, n2, **tree)
        torch.npu.synchronize()
        npu_ms = (time.perf_counter() - t0) * 1000 / iters
        print(
            f"greedy verify wall-clock ready-to-complete cpu_ref={cpu_ms:.3f}ms "
            f"npu_kernel={npu_ms:.3f}ms (JIT excluded)"
        )
        self.assertGreater(cpu_ms, 0.0)
        self.assertGreater(npu_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
