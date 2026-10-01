"""Real-device KV scratch checks. Run with PYTHONPATH=python.

Uses ordinary unittest entrypoints; no model weights or custom runner needed.
Full service/TP performance comparisons are described in SR_KV_COPY.md.
"""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from sglang.srt.speculative.standalone_remote import sr_kv_copy as kv

try:
    import torch_npu
except ImportError:
    torch_npu = None


def new_paged(dtype, device):
    # Odd width exercises the partial contiguous tile; values expose swaps.
    return NS(
        kv_buffer=torch.arange(2 * 3 * 8 * 8 * 2 * 67)
        .reshape(2, 3, 8, 8, 2, 67)
        .to(dtype=dtype, device=device)
    )


def expected_move(initial, src, dst):
    flat = initial.view(2, 3, 64, 134)
    result = flat.clone()
    result.index_copy_(2, dst.long(), flat.index_select(2, src.long()))
    return result.view_as(initial)


@unittest.skipUnless(torch_npu is not None, "requires torch_npu and NPU")
class TestNPUKVMove(unittest.TestCase):
    def test_nd_private_warmup_scratch_growth_and_reuse(self):
        # Reproduce the failing four-dimensional private allocation using real
        # NPU descriptors. CPU metadata mocks cannot certify this view behavior.
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.uint8):
            pool = new_paged(dtype, "npu")
            initial = pool.kv_buffer.cpu()
            address = pool.kv_buffer.data_ptr()
            allocations = []
            original = kv._empty_nd

            def allocate(*args, **kwargs):
                view, backing = original(*args, **kwargs)
                allocations.append((kwargs["role"], view, backing))
                return view, backing

            def check_formats():
                for role, view, backing in allocations:
                    self.assertEqual(torch_npu.get_npu_format(view), 2, role)
                    self.assertEqual(torch_npu.get_npu_format(backing), 2, role)
                    self.assertEqual(view.data_ptr(), backing.data_ptr(), role)
                    self.assertEqual(backing.ndim, 1, role)
                    self.assertEqual(view.dtype, dtype, role)
                    self.assertEqual(view.device, pool.kv_buffer.device, role)
                    self.assertTrue(view.is_contiguous(), role)

            with patch.object(kv, "_empty_nd", side_effect=allocate):
                kv.warm_private_slot_move(pool)
                self.assertEqual(
                    [a[0] for a in allocations], ["private_warmup", "scratch"]
                )
                check_formats()
                kv.warm_private_slot_move(pool)
                self.assertEqual(len(allocations), 2)
                self.assertTrue(torch.equal(initial, pool.kv_buffer.cpu()))
                self.assertEqual(address, pool.kv_buffer.data_ptr())
                ws = kv.prepare_kv_move(pool, 2)
                old_scratch = ws.scratch[0]
                src = torch.tensor([1, 0], device="npu")
                dst = torch.tensor([0, 1], device="npu")
                kv.move_kv_slots_(ws, src, dst)
                ws.reserve(17)
                check_formats()
                self.assertNotEqual(ws.scratch[0].data_ptr(), old_scratch.data_ptr())
                self.assertIs(ws.scratch_backings[0], allocations[-1][2])
                count = len(allocations)
                self.assertIs(ws, kv.prepare_kv_move(pool, 16))
                self.assertEqual(len(allocations), count)
                kv.move_kv_slots_(ws, src, dst)
                # Two swaps restore every bit; growth must retain the first
                # submitted consumer's storage without changing the live pool.
                self.assertTrue(torch.equal(initial, pool.kv_buffer.cpu()))
                self.assertEqual(ws.counts["grow"], 2)

    def test_explicit_move_and_native_pool_entry(self):
        from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool

        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.uint8):
            pool = new_paged(dtype, "npu")
            ws = kv.prepare_kv_move(pool, 8)
            self.assertEqual(ws.backend, "npu_paged6")
            pointers = [x.data_ptr() for x in ws.scratch]
            for source, target in (
                ([1, 0], [0, 1]),
                ([1, 2, 3, 0], [0, 1, 2, 3]),
                ([2, 2, 2], [0, 1, 2]),
                ([7, 8, 9], [15, 16, 17]),
            ):
                initial = pool.kv_buffer.cpu()
                src, dst = torch.tensor(source, device="npu"), torch.tensor(
                    target, device="npu"
                )
                kv.move_kv_slots_(ws, src, dst)
                self.assertTrue(
                    torch.equal(
                        pool.kv_buffer.cpu(),
                        expected_move(initial, src.cpu(), dst.cpu()),
                    )
                )
                self.assertEqual(pointers, [x.data_ptr() for x in ws.scratch])
            # EAGLE/N-gram reach the unchanged pool API, not an SR wrapper.
            kv.warm_private_slot_move(
                pool, index_dtype=torch.int64, dst_index_dtype=torch.int32
            )
            initial = pool.kv_buffer.cpu()
            src, dst = torch.tensor([1, 0], device="npu"), torch.tensor(
                [0, 1], dtype=torch.int32, device="npu"
            )
            NPUMHATokenToKVPool.move_kv_cache(pool, dst, src)
            self.assertTrue(
                torch.equal(
                    pool.kv_buffer.cpu(), expected_move(initial, src.cpu(), dst.cpu())
                )
            )
            # Prefix-tail fallback can have the reverse mixed pointer types.
            kv.warm_private_slot_move(
                pool, index_dtype=torch.int32, dst_index_dtype=torch.int64
            )
            initial = pool.kv_buffer.cpu()
            kv.move_kv_slots_(ws, src.to(torch.int32), dst.to(torch.int64))
            self.assertTrue(
                torch.equal(
                    pool.kv_buffer.cpu(), expected_move(initial, src.cpu(), dst.cpu())
                )
            )

    def test_graph_reads_live_tree_indices_and_padding(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.uint8):
            pool = new_paged(dtype, "npu")
            ws = kv.prepare_kv_move(pool, 36, domain="graph", graph=True)
            # B=4,K=3, three historical levels; replay includes B=3 padding.
            slots_cpu = torch.arange(1, 37).reshape(3, 12)
            slots = slots_cpu.to("npu")
            parents = torch.arange(12, device="npu")
            active = torch.ones(12, dtype=torch.bool, device="npu")
            # Compile before graph capture, against this private pool only.
            kv.remap_tree_kv_(ws, slots, parents, 3, active)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with ws.capture_scope(), torch.npu.graph(graph, auto_dispatch_capture=True):
                kv.remap_tree_kv_(ws, slots, parents, 3, active)
            scratch_address = ws.scratch[0].data_ptr()
            for bs in (1, 2, 3, 4, 2, 1):
                for duplicate in (False, True):
                    cpu_slots = slots_cpu.clone()
                    cpu_slots[:, bs * 3 :] = 0
                    cpu_parent = torch.arange(12)
                    for b in range(bs):
                        cpu_parent[b * 3 : b * 3 + 3] = b * 3 + torch.tensor(
                            [1, 2, 0] if not duplicate else [2, 2, 2]
                        )
                    cpu_active = torch.arange(12) < bs * 3
                    # Change the physical mapping too, preserving unique live targets.
                    cpu_slots[cpu_slots > 0] += 8
                    slots.copy_(cpu_slots)
                    parents.copy_(cpu_parent)
                    active.copy_(cpu_active)
                    initial = pool.kv_buffer.cpu().view(2, 3, 64, 134)
                    expected = initial.clone()
                    for step in range(3):
                        for row in range(bs * 3):
                            expected[:, :, cpu_slots[step, row]].copy_(
                                initial[:, :, cpu_slots[step, cpu_parent[row]]]
                            )
                    graph.replay()
                    self.assertTrue(
                        torch.equal(pool.kv_buffer.cpu().view_as(expected), expected)
                    )
                    self.assertEqual(ws.scratch[0].data_ptr(), scratch_address)
            self.assertEqual(ws.counts["grow"], 1)

    def test_runtime_chunk_scalars_capacity_and_cross_chunk_cycle(self):
        from unittest.mock import patch

        # 56 groups x 5 column tiles makes 381 slots span two chunks. Every
        # source snapshot precedes writes, including a cycle across the boundary.
        for dtype in (torch.bfloat16, torch.uint8):
            pool = NS(
                kv_buffer=(torch.arange(2 * 28 * 64 * 8 * 1025) % 127)
                .reshape(2, 28, 64, 8, 1, 1025)
                .to(dtype=dtype, device="npu")
            )
            live_before_warmup = pool.kv_buffer.cpu()
            kv.warm_private_slot_move(pool)
            self.assertTrue(torch.equal(live_before_warmup, pool.kv_buffer.cpu()))
            for n in (17, 381, 256, 1, 381):
                ws = kv.prepare_kv_move(pool, n)
                initial = pool.kv_buffer.cpu().view(2, 28, 512, 1025)
                dst = torch.arange(n, dtype=torch.int64)
                src = (dst + 1) % n
                gold = initial.clone()
                gold.index_copy_(2, dst, initial.index_select(2, src))
                with patch.object(
                    torch.Tensor, "fill_", side_effect=AssertionError("scalar fill")
                ):
                    kv.move_kv_slots_(ws, src.to("npu"), dst.to("npu"))
                self.assertTrue(torch.equal(pool.kv_buffer.cpu().view_as(gold), gold))
            self.assertEqual(ws.counts["grow"], 2)

    def test_graph_explicit_move_uses_runtime_indices_without_scalar_fills(self):
        from unittest.mock import patch

        pool = new_paged(torch.bfloat16, "npu")
        kv.warm_private_slot_move(pool, index_dtype=torch.int32, src_stride=2)
        ws = kv.prepare_kv_move(pool, 8, graph=True, domain="explicit_graph")
        backing = torch.arange(8, dtype=torch.int32, device="npu")
        src = backing[::2]
        dst = torch.arange(4, dtype=torch.int32, device="npu")
        kv.move_kv_slots_(ws, src, dst)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with patch.object(
            torch.Tensor, "fill_", side_effect=AssertionError("scalar fill")
        ), patch.object(
            kv, "_empty_nd", side_effect=AssertionError("allocation during capture")
        ):
            with ws.capture_scope(), torch.npu.graph(graph, auto_dispatch_capture=True):
                kv.move_kv_slots_(ws, src, dst)
        for source, target in (
            ([1, 0, 3, 2], [0, 1, 2, 3]),
            ([6, 6, 6, 6], [4, 5, 6, 7]),
        ):
            src.copy_(torch.tensor(source, dtype=torch.int32, device="npu"))
            dst.copy_(torch.tensor(target, dtype=torch.int32, device="npu"))
            initial = pool.kv_buffer.cpu()
            graph.replay()
            self.assertTrue(
                torch.equal(
                    pool.kv_buffer.cpu(), expected_move(initial, src.cpu(), dst.cpu())
                )
            )

    def test_mla_index_buffer_and_copy_stream(self):
        pool = NS(
            k_buffer=torch.randn(3, 8, 8, 1, 67, device="npu"),
            v_buffer=torch.randn(3, 8, 8, 1, 31, device="npu"),
            index_k_buffer=torch.randn(3, 8, 8, 1, 17, device="npu"),
        )
        before = [x.tensor.cpu() for x in kv.KVMoveLayout.from_pool(pool).buffers]
        ready = torch.npu.Event()
        ready.record()
        stream = torch.npu.Stream()
        stream.wait_event(ready)
        with torch.npu.stream(stream):
            ws = kv.prepare_kv_move(pool, 3, domain="lease_copy")
            src = torch.tensor([2, 2, 0], device="npu")
            dst = torch.tensor([0, 1, 2], device="npu")
            kv.move_kv_slots_(ws, src, dst)
            done = torch.npu.Event()
            done.record()
        torch.npu.current_stream().wait_event(done)
        for b, initial in zip(ws.layout.buffers, before):
            gold = initial.clone()
            gold.index_copy_(b.axis, dst.cpu(), initial.index_select(b.axis, src.cpu()))
            self.assertTrue(torch.equal(b.tensor.cpu(), gold))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestCUDAKVMove(unittest.TestCase):
    def test_list_layout_and_live_graph(self):
        pool = NS(
            k_buffer=[torch.randn(64, 2, 67, device="cuda") for _ in range(3)],
            v_buffer=[torch.randn(64, 2, 67, device="cuda") for _ in range(3)],
        )
        ws = kv.prepare_kv_move(pool, 4, domain="graph", graph=True)
        src, dst = torch.tensor([1, 2, 3, 0], device="cuda"), torch.arange(
            4, device="cuda"
        )
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm), ws.capture_scope():
            kv.move_kv_slots_(ws, src, dst)
        torch.cuda.current_stream().wait_stream(warm)
        graph = torch.cuda.CUDAGraph()
        with ws.capture_scope(), torch.cuda.graph(graph):
            kv.move_kv_slots_(ws, src, dst)
        for sr in ([3, 2, 1, 0], [2, 2, 2, 2], [1, 2, 3, 0]):
            src.copy_(torch.tensor(sr, device="cuda"))
            before = [b.tensor.cpu() for b in ws.layout.buffers]
            graph.replay()
            for b, initial in zip(ws.layout.buffers, before):
                expected = initial.clone()
                expected.index_copy_(
                    b.axis, dst.cpu(), initial.index_select(b.axis, src.cpu())
                )
                self.assertTrue(torch.equal(b.tensor.cpu(), expected))


if __name__ == "__main__":
    unittest.main()
