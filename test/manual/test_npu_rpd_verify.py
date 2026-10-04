"""NPU checks for the RPD compact readback.

Numerical and transfer contract checks run only when Torch NPU is configured.
``--bench`` records segmented compact cost against the current reference
path. CPU timings from that command are structural only and are not a TPOT
result.

Run with:
  PYTHONPATH=python python3 test/manual/test_npu_rpd_verify.py
  PYTHONPATH=python python3 test/manual/test_npu_rpd_verify.py --bench
"""

from __future__ import annotations

import sys
import time
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.speculative.rpd_verify as rpd
from sglang.srt.speculative.rpd_verify import (
    _compact_await_stats,
    _rpd_compact_apply,
    _rpd_compact_edge_values,
    _rpd_compact_edges,
    _rpd_compact_reduce,
    _rpd_compact_select,
    _rpd_compact_tree,
    _rpd_slot_argmax,
    _verify_tree_rpd_compact,
    _verify_tree_rpd_cpu,
    compact_cross_device_bytes,
    reset_compact_cross_device_bytes,
    rpd_gap_max,
    verify_tree_rpd,
)


def _npu_ready() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    npu = getattr(torch, "npu", None)
    if npu is None:
        return False
    try:
        return bool(npu.is_available())
    except Exception:
        return False


def _chain(batch, width, vocab, accept_len, dtype, device):
    """Build a chain whose first ``accept_len`` edges match argmax."""
    rows = batch * width
    candidates = torch.zeros((batch, width), dtype=torch.int64, device=device)
    retrive = torch.arange(rows, device=device, dtype=torch.int64).reshape(batch, width)
    nxt = torch.full((batch, width), -1, dtype=torch.int64, device=device)
    sibling = torch.full((batch, width), -1, dtype=torch.int64, device=device)
    logits = torch.zeros((rows, vocab), dtype=dtype, device=device)
    child = 1 if vocab > 1 else 0
    for b in range(batch):
        for slot in range(width - 1):
            nxt[b, slot] = slot + 1
        for slot in range(width):
            row = b * width + slot
            if slot < accept_len:
                candidates[b, slot + 1] = child
                logits[row, child] = 4
            else:
                logits[row, 0] = 4
                if slot + 1 < width:
                    candidates[b, slot + 1] = child
    return candidates, retrive, nxt, sibling, logits


def _buffers(candidates, retrive):
    batch, width = candidates.shape
    total = int(retrive.max().item()) + 1
    predicts = torch.full((total,), -1, dtype=torch.int32, device=candidates.device)
    accept_index = torch.full(
        (batch, width), -1, dtype=torch.int32, device=candidates.device
    )
    accept_length = torch.zeros((batch,), dtype=torch.int32, device=candidates.device)
    return predicts, accept_index, accept_length


def _sync(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _special_logits(dtype, device):
    logits = torch.zeros((4, 8), dtype=dtype, device=device)
    logits[0, 1] = 2
    logits[0, 4] = float("nan")
    logits[1, 2] = 3
    logits[1, 0] = float("inf")
    logits[1, 6] = float("-inf")
    logits[2, :] = float("-inf")
    logits[3, 3] = float("nan")
    logits[3, 5] = float("inf")
    return logits


@unittest.skipUnless(_npu_ready(), "Torch NPU is not configured")
class TestNpuRpdCompact(unittest.TestCase):
    def _device(self):
        return torch.device("npu")

    def test_native_max_matches_reference_argmax(self):
        device = self._device()
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            logits = torch.randn(6, 128, dtype=dtype, device=device)
            special = _special_logits(dtype, device)
            for values in (logits, special):
                _max, index = torch.max(values, dim=-1)
                reference = torch.argmax(values.float(), dim=-1)
                self.assertEqual(index.cpu().tolist(), reference.cpu().tolist())

    def test_compact_matches_reference_including_specials(self):
        device = self._device()
        original = rpd._verify_tree_rpd_cpu

        def forbid_reference(**kwargs):
            raise AssertionError("NPU dispatch entered the CPU reference")

        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for tau, accept_len in ((0.0, 0), (0.0, 3), (0.2, 1)):
                tensors = [
                    t if t.dtype.is_floating_point else t
                    for t in _chain(2, 8, 64, accept_len, dtype, device)
                ]
                tensors[-1] = tensors[-1].to(dtype)
                ref_tensors = [t.detach().cpu() for t in tensors]
                ref = _buffers(ref_tensors[0], ref_tensors[1])
                _verify_tree_rpd_cpu(
                    *ref,
                    *ref_tensors,
                    rpd_gap_max(tau),
                    float(tau) == 0.0,
                )
                got = _buffers(tensors[0], tensors[1])
                rpd._verify_tree_rpd_cpu = forbid_reference
                try:
                    returned = verify_tree_rpd(*got, *tensors, tau)
                finally:
                    rpd._verify_tree_rpd_cpu = original
                self.assertIs(returned[0], got[0])
                for actual, expect in zip(got, ref):
                    self.assertEqual(actual.cpu().tolist(), expect.tolist())

            special = _special_logits(dtype, device)
            candidates = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64, device=device)
            retrive = torch.arange(4, device=device, dtype=torch.int64).unsqueeze(0)
            nxt = torch.tensor([[1, 2, 3, -1]], dtype=torch.int64, device=device)
            sibling = torch.full((1, 4), -1, dtype=torch.int64, device=device)
            for tau in (0.0, 0.2):
                ref = _buffers(candidates.cpu(), retrive.cpu())
                _verify_tree_rpd_cpu(
                    *ref,
                    candidates.cpu(),
                    retrive.cpu(),
                    nxt.cpu(),
                    sibling.cpu(),
                    special.cpu(),
                    rpd_gap_max(tau),
                    float(tau) == 0.0,
                )
                got = _buffers(candidates, retrive)
                _verify_tree_rpd_compact(
                    *got,
                    candidates,
                    retrive,
                    nxt,
                    sibling,
                    special,
                    rpd_gap_max(tau),
                    float(tau) == 0.0,
                )
                for actual, expect in zip(got, ref):
                    self.assertEqual(actual.cpu().tolist(), expect.tolist())

    def test_transfer_bytes_stay_below_full_logits_and_ignore_vocab(self):
        device = self._device()

        def measure(vocab):
            tensors = _chain(2, 8, vocab, 3, torch.float16, device)
            buffers = _buffers(tensors[0], tensors[1])
            reset_compact_cross_device_bytes()
            _verify_tree_rpd_compact(
                *buffers, *tensors, rpd_gap_max(0.0), True
            )
            _sync(device)
            return compact_cross_device_bytes(), int(tensors[-1].numel()) * int(
                tensors[-1].element_size()
            )

        small, small_full = measure(128)
        large, large_full = measure(4096)
        self.assertGreater(small, 0)
        self.assertEqual(small, large)
        self.assertLess(small, small_full)
        self.assertLess(large, large_full)

    def test_sr_host_input_statistics_and_commit(self):
        """Actual NPU tree, typed D2H and KV commit, without profiler APIs."""
        from sglang.srt.speculative.eagle_utils import build_tree_kernel_efficient
        from sglang.srt.speculative.standalone_remote.sr_rpd import (
            SRRPDWorkspace,
            rpd_batch_key,
            verify_sr_rpd_host,
        )
        from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
            VerifyInputPacket,
        )
        from sglang.srt.speculative.standalone_remote.verifier.sr_fixed_accept import (
            SRFixedAcceptState,
        )

        # Reuse CPU request/allocator contracts, but put all live tensors on
        # NPU and execute the production packing/commit kernels.
        sys.path.insert(
            0, str(Path(__file__).resolve().parents[1] / "registered/unit/spec")
        )
        try:
            import test_sr_fixed_accept as fixtures
        finally:
            sys.path.pop(0)
        device = self._device()
        workspace, packet = SRRPDWorkspace(), VerifyInputPacket()
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for bs in (1, 2, 3, 4, 2, 1):
                for tau in (0.0, 0.2, 0.5):
                    reqs = [
                        fixtures._Req(finish_at=1 if b % 2 else None) for b in range(bs)
                    ]
                    verified, parents, selected, tokens = packet.load(
                        [0] * bs,
                        [[1, 2, 3]] * bs,
                        [None] * bs,
                        [None] * bs,
                        2,
                        3,
                        4,
                        device,
                        rpd_vocab=32,
                        rpd_batch_key=rpd_batch_key(reqs),
                    )
                    context = packet.rpd_input
                    seq = torch.ones(bs, dtype=torch.int64, device=device)
                    built = build_tree_kernel_efficient(
                        verified, parents, selected, tokens, seq, bs, 2, 3, 4
                    )
                    for actual, expected in zip(built[2:5], context.tree[1:]):
                        torch.testing.assert_close(actual.cpu(), expected)
                    logits = torch.randn(bs * 4, 32, dtype=dtype, device=device)
                    predict = torch.zeros(bs * 4 + 1, dtype=torch.int32, device=device)
                    index = torch.full((bs, 4), -1, dtype=torch.int32, device=device)
                    length = torch.zeros(bs, dtype=torch.int32, device=device)
                    _verify_tree_rpd_compact(
                        predict,
                        index,
                        length,
                        context.tree[0].to(device),
                        *built[2:5],
                        logits,
                        rpd_gap_max(tau),
                        tau == 0,
                    )
                    metrics = SimpleNamespace(counts=Counter())
                    self.assertIsNone(workspace.prepare(logits, context, metrics))
                    with patch.object(
                        rpd, "_rpd_compact_tree", side_effect=AssertionError("tree D2H")
                    ), patch.object(
                        rpd,
                        "_rpd_compact_apply",
                        side_effect=AssertionError("result H2D"),
                    ):
                        plan = verify_sr_rpd_host(logits, context, workspace, tau, 4)
                    self.assertEqual(plan.rows, index.cpu().tolist())
                    self.assertEqual(plan.pre_lengths, length.cpu().tolist())
                    self.assertEqual(metrics.counts["rpd_host_stats_waits"], 1)
                    self.assertEqual(metrics.counts["rpd_host_stats_d2h_count"], 2)
                    self.assertEqual(metrics.counts["rpd_host_edge_gather"], 1)
                    self.assertEqual(
                        metrics.counts["rpd_host_stats_d2h_bytes"],
                        bs * 4 * 8 + 2 * len(context.edges) * logits.element_size(),
                    )
                    alloc = fixtures.NPUPagedTokenToKVPoolAllocator(4)
                    alloc.kv_buffer = torch.arange(
                        2 * (bs + 1) * 4, dtype=torch.float32, device=device
                    ).reshape(2, 1, bs + 1, 4, 1, 1)
                    alloc.free_pages = alloc.free_pages.to(device)
                    alloc.release_pages = alloc.release_pages.to(device)
                    batch, _ = fixtures.FixedAcceptFinalizeTest()._batch(
                        reqs,
                        [1] * bs,
                        torch.arange(bs * 4, device=device),
                        alloc.kv_buffer,
                    )
                    batch.device = device
                    batch.seq_lens = batch.seq_lens.to(device)
                    batch.req_pool_indices = batch.req_pool_indices.to(device)
                    state = SRFixedAcceptState(bs, 4, 4, 4, device)
                    expected = fixtures.FixedAcceptFinalizeTest()._run(
                        plan.rows,
                        plan.tokens,
                        plan.pre_lengths,
                        [1] * bs,
                        [1 if b % 2 else None for b in range(bs)],
                    )
                    with patch.object(
                        state, "_pack", side_effect=AssertionError("greedy pack")
                    ), patch.object(
                        state,
                        "_submit_accept_readback",
                        side_effect=AssertionError("accept D2H"),
                    ):
                        result = state.finalize_from_host(
                            batch, SimpleNamespace(), 4, 2, alloc, plan
                        )
                    self.assertEqual(state._h2d_count, 1)
                    self.assertEqual(
                        [req.output_ids for req in reqs],
                        [req.output_ids for req in expected[3]],
                    )
                    torch.testing.assert_close(
                        result.verified_id.cpu(), expected[4].verified_id
                    )
                    torch.testing.assert_close(alloc.kv_buffer.cpu(), expected[5])

    def test_edge_gather_bits_and_empty_edges(self):
        """Kernel stores original-dtype bits into the live stats view."""
        from sglang.srt.speculative.standalone_remote.sr_rpd import (
            SRRPDWorkspace,
            build_sr_rpd_input,
            verify_sr_rpd_host,
        )
        from sglang.srt.speculative.standalone_remote.sr_rpd_kernels_npu import (
            gather_edge_stats,
        )

        device = self._device()
        bit_dtype = {
            torch.float16: torch.int16,
            torch.bfloat16: torch.int16,
            torch.float32: torch.int32,
        }
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for edges in (3, 130):
                rows, vocab = 6, 17
                parent = torch.arange(edges, dtype=torch.int64, device=device) % rows
                token = torch.arange(edges, dtype=torch.int64, device=device) % vocab
                token[-1] = vocab - 1
                parent[-1] = rows - 1
                edge_index = torch.stack((parent, token))
                values = torch.randn(rows, dtype=dtype, device=device)
                values[0] = float("nan")
                values[rows - 1] = float("inf")
                logits = torch.randn(rows, vocab, dtype=dtype, device=device)
                logits[0, int(token[0])] = float("nan")
                logits[rows - 1, vocab - 1] = float("-inf")
                logits[int(parent[1]), int(token[1])] = float("inf")
                capacity = 1 << (2 * edges - 1).bit_length()
                backing = torch.full((capacity,), 7, dtype=dtype, device=device)
                stats = backing[: 2 * edges].view(2, edges)
                self.assertTrue(gather_edge_stats(values, logits, edge_index, stats))
                _sync(device)
                ref = torch.empty((2, edges), dtype=dtype)
                parent_cpu, token_cpu = parent.cpu(), token.cpu()
                ref[0].copy_(values.cpu().index_select(0, parent_cpu))
                ref[1].copy_(logits.cpu()[parent_cpu, token_cpu])
                got = stats.cpu()
                self.assertTrue(
                    torch.equal(got.view(bit_dtype[dtype]), ref.view(bit_dtype[dtype]))
                )
                if capacity > 2 * edges:
                    tail = backing[2 * edges :].cpu()
                    self.assertTrue(torch.all(tail == 7))
            self.assertFalse(
                gather_edge_stats(
                    torch.empty(1, dtype=dtype, device=device),
                    torch.empty((1, 1), dtype=dtype, device=device),
                    torch.empty((2, 0), dtype=torch.int64, device=device),
                    torch.empty((2, 0), dtype=dtype, device=device),
                )
            )

        owner = SimpleNamespace(generation=1, unresolved=False, rpd_input=None)
        context = build_sr_rpd_input(
            torch.zeros(2, dtype=torch.int64),
            torch.empty((2, 0), dtype=torch.int64),
            torch.full((2, 1), -1, dtype=torch.int64),
            torch.empty((2, 0), dtype=torch.int64),
            topk=2,
            steps=1,
            width=1,
            vocab=4,
            owner=owner,
            generation=1,
        )
        owner.rpd_input = context
        context.edge_index = torch.empty((2, 0), dtype=torch.int64, device=device)
        logits = torch.randn(2, 4, dtype=torch.float16, device=device)
        workspace = SRRPDWorkspace()
        metrics = SimpleNamespace(counts=Counter())
        self.assertIsNone(workspace.prepare(logits, context, metrics))
        plan = verify_sr_rpd_host(logits, context, workspace, 0.0, 2)
        self.assertEqual(plan.pre_lengths, [0, 0])
        self.assertEqual(metrics.counts["rpd_host_edge_gather"], 0)
        self.assertEqual(metrics.counts["rpd_host_stats_waits"], 1)
        self.assertEqual(metrics.counts["rpd_host_stats_d2h_count"], 1)
        self.assertEqual(metrics.counts["rpd_host_stats_d2h_bytes"], 2 * 8)


def benchmark_rpd_compact_stages(device: torch.device | None = None) -> None:
    """Time compact stages. This is not called from ``verify_tree_rpd``."""
    if device is None:
        device = torch.device("npu") if _npu_ready() else torch.device("cpu")
    npu = device.type == "npu"
    print(
        "RPD compact benchmark"
        if npu
        else "RPD compact benchmark on CPU; not an NPU TPOT result"
    )
    print(f"device={device}")
    if npu:
        print(
            "Baseline is the current reference path running on NPU. "
            "Byte counts passing do not by themselves show a TPOT gain."
        )
    cases = (
        ("root_only", 2, 1, 1024, 0),
        ("sr_short", 1, 15, 1024, 0),
        ("sr_mid", 4, 15, 4096, 2),
        ("sr_long", 8, 15, 8192, 5),
        ("vocab_small", 2, 8, 128, 1),
        ("vocab_large", 2, 8, 8192, 1),
        ("sr_vocab", 1, 15, 151936, 3),
    )
    header = (
        f"{'case':<12} {'tree':>10} {'reduce':>10} {'edges':>10} "
        f"{'gather':>10} {'stats':>10} {'select':>10} {'write':>10} "
        f"{'compact':>10} {'reference':>10} {'xdev':>10} {'logical':>10} {'peak':>12}"
    )
    print(header)
    for name, batch, width, vocab, accept_len in cases:
        dtype = torch.bfloat16 if npu else torch.float32
        tensors = _chain(batch, width, vocab, accept_len, dtype, device)
        candidates, retrive, nxt, sibling, logits = tensors
        rows = int(logits.shape[0])
        _sync(device)
        started = time.perf_counter()
        tree = _rpd_compact_tree(candidates, retrive, nxt, sibling)
        _sync(device)
        tree_s = time.perf_counter() - started
        started = time.perf_counter()
        z_star, argmax = _rpd_compact_reduce(logits)
        _sync(device)
        reduce_s = time.perf_counter() - started
        started = time.perf_counter()
        edges, edge_index = _rpd_compact_edges(tree, rows, int(logits.shape[-1]))
        _sync(device)
        edges_s = time.perf_counter() - started
        started = time.perf_counter()
        edge_values = None
        if edge_index is not None:
            edge_values = _rpd_compact_edge_values(logits, z_star, edge_index, None)
        _sync(device)
        gather_s = time.perf_counter() - started
        started = time.perf_counter()
        star, stats = _compact_await_stats(
            _rpd_slot_argmax(argmax, retrive, rows), edge_values
        )
        _sync(device)
        stats_s = time.perf_counter() - started
        started = time.perf_counter()
        payload = _rpd_compact_select(
            tree,
            edges,
            star,
            stats,
            rpd_gap_max(0.0),
            True,
            width,
            rows,
            int(retrive.max().item()) + 1,
        )
        _sync(device)
        select_s = time.perf_counter() - started
        buffers = _buffers(candidates, retrive)
        started = time.perf_counter()
        _rpd_compact_apply(payload, *buffers, device)
        _sync(device)
        write_s = time.perf_counter() - started

        ref = _buffers(candidates, retrive)
        _sync(device)
        started = time.perf_counter()
        _verify_tree_rpd_cpu(
            *ref, *tensors, rpd_gap_max(0.0), True
        )
        _sync(device)
        reference_s = time.perf_counter() - started
        for actual, expect in zip(buffers, ref):
            if actual.detach().cpu().tolist() != expect.detach().cpu().tolist():
                raise RuntimeError(f"{name} compact result diverged from reference")

        reset_compact_cross_device_bytes()
        total_buffers = _buffers(candidates, retrive)
        if device.type == "npu":
            torch.npu.synchronize()
            torch.npu.reset_peak_memory_stats()
            torch.npu.empty_cache()
            baseline = torch.npu.memory_allocated()
        elif device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            baseline = torch.cuda.memory_allocated()
        else:
            baseline = None
        _sync(device)
        started = time.perf_counter()
        _verify_tree_rpd_compact(
            *total_buffers, *tensors, rpd_gap_max(0.0), True
        )
        _sync(device)
        compact_s = time.perf_counter() - started
        moved = compact_cross_device_bytes()
        if device.type == "npu":
            peak = torch.npu.max_memory_allocated() - baseline
        elif device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() - baseline
        else:
            peak = -1
        n_edges = 0 if edge_index is None else int(edge_index.shape[1])
        n_writes = int(payload[2].numel())
        logical = 32 * batch * width + 8 * batch * width
        if n_edges:
            logical += 16 * n_edges + 2 * n_edges * int(logits.element_size())
        logical += 8 * (batch * width + batch + 2 * n_writes)
        peak_text = "n/a" if peak < 0 else str(int(peak))
        print(
            f"{name:<12} {tree_s:10.6f} {reduce_s:10.6f} {edges_s:10.6f} "
            f"{gather_s:10.6f} {stats_s:10.6f} {select_s:10.6f} {write_s:10.6f} "
            f"{compact_s:10.6f} {reference_s:10.6f} {moved:10d} "
            f"{logical:10d} {peak_text:>12}"
        )


if __name__ == "__main__":
    if "--bench" in sys.argv:
        sys.argv = [arg for arg in sys.argv if arg != "--bench"]
        benchmark_rpd_compact_stages()
    else:
        unittest.main()
