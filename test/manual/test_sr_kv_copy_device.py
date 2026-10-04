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
    def test_identity_sources_and_cross_chunk_reorders(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            pool = new_paged(dtype, "npu")
            ws = kv.prepare_kv_move(pool, 33)
            destination = torch.arange(33, device="npu")
            for mapping in (
                list(range(33)),
                [0] * 33,
                [32] * 33,
                list(range(1, 33)) + [0],
            ):
                before = pool.kv_buffer.cpu()
                source = torch.tensor(mapping, device="npu")
                with patch.object(kv, "_PAGED_GRID_LIMIT", 63):
                    kv.move_kv_slots_(ws, source, destination)
                torch.npu.synchronize()
                self.assertTrue(
                    torch.equal(
                        pool.kv_buffer.cpu(),
                        expected_move(before, source.cpu(), destination.cpu()),
                    )
                )

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
                src, dst = (
                    torch.tensor(source, device="npu"),
                    torch.tensor(target, device="npu"),
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
            src, dst = (
                torch.tensor([1, 0], device="npu"),
                torch.tensor([0, 1], dtype=torch.int32, device="npu"),
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
        with (
            patch.object(
                torch.Tensor, "fill_", side_effect=AssertionError("scalar fill")
            ),
            patch.object(
                kv, "_empty_nd", side_effect=AssertionError("allocation during capture")
            ),
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
        src, dst = (
            torch.tensor([1, 2, 3, 0], device="cuda"),
            torch.arange(4, device="cuda"),
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


@unittest.skipUnless(torch_npu is not None, "requires torch_npu and NPU")
class TestNPUCandidates(unittest.TestCase):
    def test_unindexed_device_resolves_to_tensor_device(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_candidates import (
            SRTreeCandidateWorkspace,
        )

        ws = SRTreeCandidateWorkspace("npu", 1, 2, 2, 3, 37, torch.float32)
        self.assertEqual(ws.device, torch.device("npu", torch.npu.current_device()))
        self.assertEqual(ws.device, ws.probs.device)
        ws.warm()
        logits = torch.randn(2, 37, device=ws.device)
        values, indices = ws.probabilities(logits, 0)
        expected = torch.topk(torch.softmax(logits, -1), 2, -1)
        torch.testing.assert_close(values, expected.values, rtol=0, atol=0)
        self.assertTrue(torch.equal(indices, expected.indices))

    def episode(self, workspace, seed_p, seed_i, logits):
        from sglang.srt.speculative.spec_utils import select_top_k_tokens
        from sglang.srt.speculative.eagle_utils import organize_draft_results

        rp, ri, wp, wi = seed_p, seed_i, seed_p, seed_i
        rs = ws = None
        scores, tokens, parents = [], [], []
        for step in range(workspace.steps):
            ids, _, rs, info, rows = select_top_k_tokens(
                step, rp, ri, None, rs, workspace.topk
            )
            wids, ws, nodes, wrows = workspace.select(step, wp, wi, ws)
            torch.testing.assert_close(wids, ids, rtol=0, atol=0)
            torch.testing.assert_close(ws, rs, rtol=0, atol=0, equal_nan=True)
            torch.testing.assert_close(nodes, info[2], rtol=0, atol=0)
            if rows is not None:
                torch.testing.assert_close(wrows, rows, rtol=0, atol=0)
            scores.append(info[0])
            tokens.append(info[1])
            parents.append(info[2])
            if step < workspace.steps - 1:
                p = torch.softmax(logits[step], dim=-1)
                top = (
                    torch.max(p, -1, keepdim=True)
                    if workspace.topk == 1
                    else torch.topk(p, workspace.topk, -1)
                )
                rp, ri = top.values, top.indices
                wp, wi = workspace.probabilities(logits[step], step)
        ref = organize_draft_results(scores, tokens, parents, workspace.width)
        got = workspace.finish(seed_p.shape[0])
        for actual, expected in zip(got, ref):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return got

    def test_candidate_dtypes_production_vocab_and_padding(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_candidates import (
            SRTreeCandidateWorkspace,
        )

        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            device = torch.device("npu", torch.npu.current_device())
            workspace = SRTreeCandidateWorkspace(device, 4, 3, 5, 15, 151936, dtype)
            workspace.warm()
            key = workspace._storage_key()
            for batch in (1, 2, 3, 4, 3, 1):
                seed = torch.softmax(
                    torch.randn(batch, 151936, dtype=dtype, device="npu"), -1
                )
                top = torch.topk(seed, 3, -1)
                logits = torch.randn(4, batch * 3, 151936, dtype=dtype, device="npu")
                self.episode(workspace, top.values, top.indices, logits)
                self.assertEqual(workspace._storage_key(), key)
            torch.npu.synchronize()

    def test_candidate_graph_dynamic_inputs(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_candidates import (
            SRTreeCandidateWorkspace,
        )

        workspace = SRTreeCandidateWorkspace(
            torch.device("npu", torch.npu.current_device()),
            4,
            3,
            5,
            15,
            257,
            torch.float32,
            graph=True,
        )
        workspace.warm()
        seed_p = torch.rand(4, 3, device="npu")
        seed_i = torch.arange(12, device="npu").reshape(4, 3)
        logits = torch.randn(4, 12, 257, device="npu")

        def run():
            p, ids, scores = seed_p, seed_i, None
            for step in range(5):
                _, scores, _, _ = workspace.select(step, p, ids, scores)
                if step < 4:
                    p, ids = workspace.probabilities(logits[step], step)
            return workspace.finish(4)

        run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with workspace.capture_scope(), torch.npu.graph(graph):
            outputs = run()
        for raw_batch in (3, 1, 4, 2, 3):
            seed_p.copy_(torch.rand_like(seed_p))
            seed_i.copy_(torch.arange(12, device="npu").reshape(4, 3).flip(1))
            if raw_batch < 4:
                seed_p[raw_batch:].zero_()
            logits.copy_(torch.randn_like(logits))
            graph.replay()
            saved = tuple(t.clone() for t in outputs)
            expected = self.episode(workspace, seed_p, seed_i, logits)
            for a, b in zip(saved, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.npu.synchronize()

    def test_persistent_thread_binds_real_device_and_stream(self):
        from sglang.srt.speculative.standalone_remote.sr_graph_update import (
            SRGraphUpdateWorker,
        )
        from sglang.srt.speculative.spec_utils import run_npu_graph_update_and_replay

        device = torch.npu.current_device()
        worker = SRGraphUpdateWorker(device)
        stream = torch.npu.Stream(device=device)
        observed = []
        try:
            with torch.npu.stream(stream):
                for _ in range(4):
                    run_npu_graph_update_and_replay(
                        lambda: observed.append(
                            (torch.npu.current_device(), torch.npu.current_stream())
                        ),
                        lambda: None,
                        overlap=True,
                        update_worker=worker,
                    )
            self.assertTrue(all(d == device and s == stream for d, s in observed))
        finally:
            worker.close()


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestCUDASRPaths(unittest.TestCase):
    def test_actual_backend_eager_graph_metadata_binding_and_reuse(self):
        from sglang.srt.layers.attention.triton_backend import (
            TritonMultiStepDraftBackend,
        )

        backend = TritonMultiStepDraftBackend.__new__(TritonMultiStepDraftBackend)
        backend._sr_cuda_metadata = True
        backend._sr_eager_metadata = {}
        backend.device = "cuda"
        backend.metadata_steps, backend.speculative_num_steps = 4, 5
        backend.topk, backend.max_context_len, backend.pool_len, backend.page_size = (
            3,
            64,
            64,
            1,
        )
        backend.kv_indptr = torch.zeros(4, 13, dtype=torch.int32, device="cuda")
        backend.cuda_graph_kv_indices = torch.empty(
            4, 12 * 64, dtype=torch.int64, device="cuda"
        )
        views = []
        backend.attn_backends = [
            NS(
                init_forward_metadata=lambda b: views.append(
                    (b.spec_info.kv_indices, b.spec_info.kv_indptr)
                )
            )
            for _ in range(4)
        ]
        backend.attn_backends[-1].cuda_graph_num_kv_splits = torch.zeros(
            12, dtype=torch.int32, device="cuda"
        )
        backend.attn_backends[-1].get_num_kv_splits = lambda *a: None
        mapping = torch.arange(4 * 64, dtype=torch.int32, device="cuda").reshape(4, 64)
        for b in (1, 2, 3, 4, 3, 1):
            seq = torch.arange(5, 5 + b, device="cuda")
            batch = NS(
                batch_size=b,
                seq_lens=seq,
                seq_lens_sum=int(seq.sum()),
                req_pool_indices=torch.arange(b - 1, -1, -1, device="cuda"),
                req_to_token_pool=NS(req_to_token=mapping),
                positions=seq.repeat_interleave(3),
                spec_info=NS(),
            )
            views.clear()
            backend.init_forward_metadata(batch)
            eager = [(i.clone(), p.clone()) for i, p in views]
            pointers = [(i.data_ptr(), p.data_ptr()) for i, p in views]
            views.clear()
            backend.init_forward_metadata(batch)
            self.assertEqual(pointers, [(i.data_ptr(), p.data_ptr()) for i, p in views])
            backend.init_forward_metadata_replay_cuda_graph(batch, b)
            for step, (indices, indptr) in enumerate(eager):
                self.assertTrue(
                    torch.equal(
                        indices, backend.cuda_graph_kv_indices[step, : indices.numel()]
                    )
                )
                self.assertTrue(
                    torch.equal(indptr, backend.kv_indptr[step, : indptr.numel()])
                )
            self.assertIs(backend._sr_metadata_holds[1], backend.cuda_graph_kv_indices)
        self.assertEqual(len(backend._sr_eager_metadata), 1)

    def test_masked_padding_keeps_identity_source_read_by_live_row(self):
        k = torch.arange(16, device="cuda", dtype=torch.float32).view(16, 1, 1)
        v = k + 100
        pool = NS(k_buffer=k.clone(), v_buffer=v.clone())
        ws = kv.prepare_kv_move(pool, 4)
        self.assertEqual(ws.backend, "cuda_layered")
        slots = torch.tensor([[5, 6, 7, 5]], device="cuda")
        parents = torch.tensor([0, 0, 1, 2], device="cuda")
        active = torch.tensor([True, True, True, False], device="cuda")
        initial_k = k.clone()
        kv.remap_tree_kv_(ws, slots, parents, 1, active)
        torch.cuda.synchronize()
        actual = pool.k_buffer.cpu()
        self.assertTrue(torch.equal(actual[5], initial_k.cpu()[5]))
        self.assertTrue(torch.equal(actual[6], initial_k.cpu()[5]))
        self.assertTrue(torch.equal(actual[7], initial_k.cpu()[6]))
        self.assertTrue(torch.equal(pool.v_buffer.cpu()[5], (initial_k + 100).cpu()[5]))

    def test_layered_overlap_across_cuda_grid_chunk_boundary(self):
        # This cycle crosses the y-axis chunk boundary. Chunk-interleaved
        # gather/scatter would lose the last source before it is read.
        n = 65537
        tensors = [
            torch.arange(n * 3, device="cuda", dtype=torch.float32).view(n, 1, 3) + j
            for j in range(4)
        ]
        pool = NS(k_buffer=tensors[:2], v_buffer=tensors[2:])
        ws = kv.prepare_kv_move(pool, n)
        dst = torch.arange(n, device="cuda")
        src = dst.roll(1)
        snapshots = [t.clone() for t in tensors]
        kv.move_kv_slots_(ws, src, dst)
        for initial, actual in zip(snapshots, tensors):
            self.assertTrue(torch.equal(actual, initial.index_select(0, src)))

    def test_token_accept_admission_commit_and_stop(self):
        from sglang.srt.speculative.standalone_remote.verifier.sr_fixed_accept import (
            build_fixed_accept_state,
        )
        from sglang.srt.speculative.standalone_remote.sr_rpd import (
            SRRPDHostPlan,
            rpd_batch_key,
        )

        class MHATokenToKVPool:
            row_dim = 0

            def __init__(self):
                self.k_buffer = [torch.randn(64, 2, 7, device="cuda")]
                self.v_buffer = [torch.randn(64, 2, 7, device="cuda")]

        class TokenToKVPoolAllocator:
            page_size, is_not_in_free_group = 1, True

            def __init__(self):
                self.pool, self.freed = MHATokenToKVPool(), []

            def get_kvcache(self):
                return self.pool

            def free(self, slots):
                self.freed.append(slots.clone())

        def request(stop):
            req = NS(
                output_ids=[],
                require_reasoning=False,
                kv_committed_len=5,
                spec_verify_ct=0,
                spec_accepted_tokens=0,
            )
            req.finished = lambda: stop is not None and len(req.output_ids) >= stop
            req.check_finished = lambda: None
            req.update_spec_acceptance_histogram = lambda n: None
            return req

        for stops in ((None, None), (1, None), (1, 2)):
            for host in (False, True):
                alloc = TokenToKVPoolAllocator()
                worker = NS(
                    device="cuda",
                    topk=2,
                    page_size=1,
                    speculative_num_steps=2,
                    speculative_num_draft_tokens=4,
                    _verify_max_bs=2,
                    _hybrid_needs_hidden=False,
                    token_to_kv_pool_allocator=alloc,
                )
                state = build_fixed_accept_state(worker)
                self.assertIsNotNone(state)
                state.warmup_scratch()
                batch = NS(
                    reqs=[request(n) for n in stops],
                    seq_lens=torch.tensor([5, 7], device="cuda"),
                    seq_lens_cpu=torch.tensor([5, 7]),
                    req_pool_indices=torch.tensor([2, 0], device="cuda"),
                    req_to_token_pool=NS(
                        req_to_token=torch.zeros(
                            4, 64, dtype=torch.int32, device="cuda"
                        )
                    ),
                    out_cache_loc=torch.tensor(
                        [11, 15, 17, 19, 21, 23, 25, 27], device="cuda"
                    ),
                    model_config=NS(think_end_id=None),
                    spec_algorithm=NS(is_standalone_remote=lambda: True),
                )
                rows, tokens = [[0, 2, -1], [4, 7, 5]], [[31, 32, 0], [41, 42, 43]]
                before = alloc.pool.k_buffer[0].clone()
                if host:
                    plan = SRRPDHostPlan(
                        rows, tokens, [1, 2], 100, rpd_batch_key(batch.reqs)
                    )
                    out = state.finalize_from_host(batch, NS(), 1, 2, alloc, plan)
                    self.assertEqual(state._d2h_count, 0)
                else:
                    pred = torch.tensor(
                        [31, 0, 32, 0, 41, 43, 0, 42, 0],
                        dtype=torch.int32,
                        device="cuda",
                    )
                    out = state.finalize(
                        batch,
                        NS(),
                        1,
                        2,
                        alloc,
                        torch.tensor(rows, device="cuda"),
                        pred,
                        torch.tensor([1, 2], device="cuda"),
                    )
                counts = [1 if stops[0] else 2, stops[1] or 3]
                kept = rows[0][: counts[0]] + rows[1][: counts[1]]
                cache = [11, 15, 17, 19, 21, 23, 25, 27]
                self.assertEqual(out.accepted_indices.cpu().tolist(), kept)
                self.assertEqual(
                    alloc.freed[0].cpu().tolist(),
                    [cache[i] for i in range(8) if i not in kept],
                )
                self.assertEqual(
                    batch.seq_lens.cpu().tolist(), [5 + counts[0], 7 + counts[1]]
                )
                self.assertTrue(torch.equal(before, alloc.pool.k_buffer[0]))
                self.assertEqual(state._h2d_count, 1)
                self.assertFalse(hasattr(alloc.pool, "_kv_move_workspaces"))

    def test_metadata_graph_reads_live_mapping_and_keeps_five_step_span(self):
        from sglang.srt.speculative.spec_utils import generate_draft_decode_kv_indices
        from sglang.srt.speculative.standalone_remote.sr_cuda_metadata import (
            SRDraftDecodeMetadataWorkspace,
        )

        ws = SRDraftDecodeMetadataWorkspace("cuda", 4, 64)
        indices, indptr = ws.reserve(12)
        mapping = torch.arange(4 * 64, device="cuda").reshape(4, 64)
        mapping[0].zero_()
        req = torch.tensor([1, 2, 3, 0], device="cuda")
        prefix = torch.tensor([5, 7, 9, 0], device="cuda")
        positions = prefix.repeat_interleave(3)

        def run():
            generate_draft_decode_kv_indices[(4, 4, 3)](
                req,
                mapping,
                prefix,
                indices,
                indptr,
                positions,
                64,
                indices.shape[1],
                indptr.shape[1],
                4,
                8,
                16,
                1,
                branch_steps=5,
            )

        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for lens in ([5, 7, 9, 0], [0, 11, 3, 0]):
            prefix.copy_(torch.tensor(lens, device="cuda"))
            positions.copy_(prefix.repeat_interleave(3))
            req[:3].copy_(torch.tensor([3, 1, 2], device="cuda"))
            graph.replay()
            actual, pointers = indices.cpu(), indptr.cpu()
            cpu_map, cpu_req = mapping.cpu(), req.cpu()
            for step in range(4):
                gold, ends = [], [0]
                for b, length in enumerate(lens):
                    for branch in range(3):
                        gold.extend(cpu_map[cpu_req[b], :length].tolist())
                        start = length + branch * 5
                        gold.extend(
                            cpu_map[cpu_req[b], start : start + step + 1].tolist()
                        )
                        ends.append(len(gold))
                self.assertEqual(actual[step, : len(gold)].tolist(), gold)
                self.assertEqual(pointers[step, :13].tolist(), ends)

    def test_pack_and_token_slots_strides_and_storage(self):
        from sglang.srt.speculative.standalone_remote.verifier.sr_fixed_accept_kernels import (
            pack_accept,
            gather_token_slots,
        )

        for dtype in (torch.int32, torch.int64):
            index = torch.tensor(
                [[0, 4, -1, -1], [8, 10, 14, -1]], dtype=dtype, device="cuda"
            )[:, ::2]
            pred = torch.arange(15, dtype=dtype, device="cuda")
            lengths = torch.tensor([0, 1], dtype=dtype, device="cuda")
            out = torch.empty(2, 6, dtype=torch.int64, device="cuda")
            gold = torch.empty(2, 6, dtype=torch.int64)
            pack_accept(index.cpu(), pred.cpu(), lengths.cpu(), gold)
            pack_accept(index, pred, lengths, out)
            self.assertTrue(torch.equal(gold, out.cpu()))
            cache = torch.arange(40, dtype=dtype, device="cuda")[::2]
            kept = torch.tensor([0, 3, 7], device="cuda")
            free = torch.tensor([1, 4, 9], device="cuda")
            slots, released = gather_token_slots(cache, kept, free)
            self.assertEqual(slots.cpu().tolist(), [0, 6, 14])
            self.assertEqual(released.cpu().tolist(), [2, 8, 18])
            self.assertEqual(slots.storage_offset(), 0)
            self.assertNotEqual(
                slots.untyped_storage().data_ptr(),
                released.untyped_storage().data_ptr(),
            )

    def test_layered_bits_strides_overlap_and_growth(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.uint8):
            k = [
                torch.arange(32 * 2 * 67, device="cuda").reshape(32, 2, 67).to(dtype)
                for _ in range(3)
            ]
            v = [t.transpose(1, 2) for t in k]  # distinct views need separate storage
            v = [t.clone() for t in v]
            pool = NS(k_buffer=k, v_buffer=v)
            ws = kv.prepare_kv_move(pool, 8)
            self.assertEqual(ws.backend, "cuda_layered")
            for src in (
                [0, 1, 2, 3, 4, 5, 6, 7],
                [1, 2, 3, 4, 5, 6, 7, 0],
                [0, 0, 2, 2, 0, 2, 6, 6],
            ):
                before = [b.tensor.clone() for b in ws.layout.buffers]
                s = torch.tensor(src, dtype=torch.int32, device="cuda")
                d = torch.arange(8, device="cuda")
                kv.move_kv_slots_(ws, s, d)
                for b, original in zip(ws.layout.buffers, before):
                    gold = original.index_copy(0, d, original.index_select(0, s.long()))
                    self.assertTrue(torch.equal(b.tensor, gold))
            kv.prepare_kv_move(pool, 17)
            self.assertEqual(ws.capacity, 32)
            table_ptrs = [row[0].data_ptr() for row in ws.cuda_tables]
            kv.prepare_kv_move(pool, 12)
            self.assertEqual(table_ptrs, [row[0].data_ptr() for row in ws.cuda_tables])
            kv.warm_private_slot_move(pool)

    def test_tree_graph_dynamic_parents_slots_and_active(self):
        pool = NS(
            k_buffer=[torch.randn(64, 2, 67, device="cuda") for _ in range(3)],
            v_buffer=[torch.randn(64, 2, 67, device="cuda") for _ in range(3)],
        )
        ws = kv.prepare_kv_move(pool, 36, domain="test_tree", graph=True)
        slots = torch.arange(36, device="cuda").reshape(3, 12).t().contiguous().t()
        parents = torch.arange(12, device="cuda")
        active = torch.tensor([True] * 9 + [False] * 3, device="cuda")
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm), ws.capture_scope():
            kv.remap_tree_kv_(ws, slots, parents, 3, active)
        torch.cuda.current_stream().wait_stream(warm)
        graph = torch.cuda.CUDAGraph()
        with ws.capture_scope(), torch.cuda.graph(graph):
            kv.remap_tree_kv_(ws, slots, parents, 3, active)
        for p in ([1, 2, 0] * 4, [2, 2, 2, 3, 4, 5, 8, 6, 7, 9, 10, 11]):
            parents.copy_(torch.tensor(p, device="cuda"))
            slots.add_(1)
            before = [b.tensor.clone() for b in ws.layout.buffers]
            graph.replay()
            for b, original in zip(ws.layout.buffers, before):
                gold = original.clone()
                for step in range(3):
                    gold.index_copy_(
                        0,
                        slots[step, :9],
                        original.index_select(0, slots[step, parents[:9]]),
                    )
                self.assertTrue(torch.equal(gold, b.tensor))

    def test_rpd_host_plan_matches_old_compact_and_single_wait(self):
        from collections import Counter
        from sglang.srt.speculative import rpd_verify as rpd
        from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
            VerifyInputPacket,
        )
        from sglang.srt.speculative.standalone_remote.sr_rpd import (
            SRRPDWorkspace,
            verify_sr_rpd_host,
        )

        ws = SRRPDWorkspace()
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for tau in (0.0, 0.2, 0.5):
                packet = VerifyInputPacket()
                packet.load(
                    [2, 2],
                    [[1, 2, 3, 4, 5]] * 2,
                    [[-1, 0, 1, 2, 3]] * 2,
                    [list(range(5))] * 2,
                    2,
                    3,
                    6,
                    "cuda",
                    rpd_vocab=32,
                )
                ctx = packet.rpd_input
                logits = torch.randn(12, 32, dtype=dtype, device="cuda")
                logits[0, 0] = float("nan")
                metrics = NS(counts=Counter())
                self.assertIsNone(ws.prepare(logits, ctx, metrics))
                tree = ctx.tree.to("cuda")
                pred = torch.zeros(13, dtype=torch.int32, device="cuda")
                rows = torch.full((2, 4), -1, dtype=torch.int32, device="cuda")
                lengths = torch.zeros(2, dtype=torch.int32, device="cuda")
                rpd.verify_tree_rpd(pred, rows, lengths, *tree, logits, tau)
                plan = verify_sr_rpd_host(logits, ctx, ws, tau, 4)
                self.assertEqual(plan.rows, rows.cpu().tolist())
                self.assertEqual(plan.pre_lengths, lengths.cpu().tolist())
                gold_tokens = [
                    [int(pred[i]) if i >= 0 else 0 for i in row] for row in plan.rows
                ]
                self.assertEqual(plan.tokens, gold_tokens)
                self.assertEqual(metrics.counts["rpd_host_stats_waits"], 1)
                self.assertEqual(metrics.counts["rpd_host_stats_d2h_count"], 2)

    def test_candidates_cuda_preserve_compiled_selection_and_graph(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_candidates import (
            SRTreeCandidateWorkspace,
        )
        from sglang.srt.speculative.spec_utils import select_top_k_tokens
        from sglang.srt.speculative.eagle_utils import organize_draft_results

        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            # Separate model dtypes must not share a test-only accumulated
            # compile guard cache and silently exceed Dynamo's default limit.
            torch._dynamo.reset()
            ws = SRTreeCandidateWorkspace(
                "cuda", 4, 3, 5, 15, 151936, dtype, graph=True
            )
            ws.warm()
            self.assertFalse(ws.selection_out)
            logits = torch.zeros(4, 12, 151936, dtype=dtype, device="cuda")
            seed = torch.topk(torch.softmax(logits[0, :4], -1), 3, -1)

            def run():
                p, ids, scores = seed.values, seed.indices, None
                for step in range(5):
                    _, scores, _, _ = ws.select(step, p, ids, scores)
                    if step < 4:
                        p, ids = ws.probabilities(logits[step], step)
                return ws.finish(4)

            warm = torch.cuda.Stream()
            warm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm), ws.capture_scope():
                run()
            torch.cuda.current_stream().wait_stream(warm)
            graph = torch.cuda.CUDAGraph()
            with ws.capture_scope(), torch.cuda.graph(graph):
                out = run()
            for value in (0.0, 0.01, float("nan"), float("inf"), -float("inf")):
                logits[:, :, 10:13].fill_(value)
                graph.replay()
                p, ids, scores = seed.values, seed.indices, None
                ss, tt, pp = [], [], []
                for step in range(5):
                    _, _, scores, info, _ = select_top_k_tokens(
                        step, p, ids, None, scores, 3
                    )
                    ss.append(info[0])
                    tt.append(info[1])
                    pp.append(info[2])
                    torch.testing.assert_close(
                        ws.scores[step], scores, rtol=0, atol=0, equal_nan=True
                    )
                    if step < 4:
                        top = torch.topk(torch.softmax(logits[step], -1), 3, -1)
                        p, ids = top.values, top.indices
                gold = organize_draft_results(ss, tt, pp, 15)
                for actual, expected in zip(out, gold):
                    self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
