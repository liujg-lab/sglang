"""NPU checks for the RPD compact readback.

Numerical checks and the profiler run only when Torch NPU is configured.
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

import torch

import sglang.srt.speculative.rpd_verify as rpd
from sglang.srt.speculative.rpd_verify import (
    _rpd_compact_apply,
    _compact_await_stats,
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

    def test_profiler_has_no_full_logits_host_copy(self):
        from torch_npu.profiler import ProfilerActivity, profile

        device = self._device()
        vocab = 4096
        tensors = _chain(1, 8, vocab, 2, torch.float16, device)
        logits = tensors[-1]
        full_bytes = int(logits.numel()) * int(logits.element_size())
        buffers = _buffers(tensors[0], tensors[1])
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            _verify_tree_rpd_compact(*buffers, *tensors, rpd_gap_max(0.0), True)
            _sync(device)
        offenders = []
        for evt in prof.key_averages():
            name = str(getattr(evt, "key", "")).lower()
            mem = max(
                int(getattr(evt, "self_cpu_memory_usage", 0) or 0),
                int(getattr(evt, "cpu_memory_usage", 0) or 0),
            )
            host_copy = any(
                token in name
                for token in ("copy", "to_copy", "clone", "contiguous", "memcpy")
            )
            if host_copy and mem >= full_bytes:
                offenders.append(f"{evt.key} cpu_mem={mem}")
        self.assertEqual(
            offenders,
            [],
            "NPU compact moved a full logits tensor to the host",
        )


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
