"""Real NPU metadata/graph checks; run with PYTHONPATH=python, no model needed."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.speculative.standalone_remote.sr_round_metrics import SRRoundMetrics

from sglang.srt.speculative.standalone_remote.sr_paged_metadata import (
    SRDraftPagedPlan,
    SRTargetPagedPlan,
    allocate_draft_view,
    empty_metadata,
    fill_draft_paged_metadata_,
    fill_target_paged_metadata_,
    warm_draft_paged_metadata,
    warm_target_paged_metadata,
)
from sglang.srt.speculative.standalone_remote.drafter.sr_tree_paged_layout import (
    prepare_tree_paged_view,
)
from sglang.srt.speculative.standalone_remote.verifier.sr_target_tree_fia import (
    SRTargetTreeFiaMetadata,
    fill_target_tree_fia_metadata_,
)

try:
    import torch_npu
except ImportError:
    torch_npu = None


class LaunchRecorder:
    def __init__(self, kernel):
        self.kernel, self.calls = kernel, 0

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            self.calls += 1
            return self.kernel[grid](*args, **kwargs)

        return run


def to_nd(tensor):
    result = empty_metadata(tuple(tensor.shape), tensor.dtype, "npu")
    result.copy_(tensor)
    return result


@unittest.skipUnless(torch_npu is not None, "requires torch_npu and NPU")
class TestPagedMetadataNPU(unittest.TestCase):
    def test_target_reference_strides_and_launches(self):
        from sglang.srt.speculative.standalone_remote import (
            sr_paged_metadata_kernels_npu as kernels,
        )

        for dtype in (torch.int32, torch.int64):
            mapping = torch.arange(6 * 1024 * 2, dtype=dtype).reshape(6, 2048)
            dev_mapping = to_nd(mapping)[:, ::2]
            mapping = mapping[:, ::2]
            for prefixes in ((), (0,), (127, 128), (129, 0, 127), (0, 127, 128, 129)):
                with self.subTest(dtype=dtype, prefixes=prefixes):
                    p = SRTargetPagedPlan.build(prefixes, 4, 15, 3, 128)
                    md = SRTargetTreeFiaMetadata.allocate(4, 15, 3, 128, "npu")
                    metrics = SRRoundMetrics("Target")
                    md.workspace.metrics = metrics
                    ref = SRTargetTreeFiaMetadata.allocate(4, 15, 3, 128, "cpu")
                    pool_base = torch.tensor([5, 0, 2, 0, 4, 0, 1, 0], dtype=dtype)
                    pool = pool_base[::2][: len(prefixes)]
                    dev_pool = to_nd(pool_base)[::2][: len(prefixes)]
                    n = sum(15 * (x + 15) for x in prefixes)
                    mask_base = torch.arange(n * 2) % 3 != 0
                    mask, dev_mask = mask_base[::2], to_nd(mask_base)[::2]
                    with patch.object(
                        kernels, "_target_pages", LaunchRecorder(kernels._target_pages)
                    ) as a, patch.object(
                        kernels, "_target_mask", LaunchRecorder(kernels._target_mask)
                    ) as b:
                        fill_target_paged_metadata_(
                            md, dev_mapping, dev_pool, dev_mask, p, 7
                        )
                        self.assertEqual((a.calls, b.calls), (1, 1))
                    self.assertEqual(
                        metrics.counts["paged_metadata_target_h2d_count"], 1
                    )
                    self.assertEqual(
                        metrics.counts["paged_metadata_target_h2d_bytes"], 4 * 3 * 8
                    )
                    md.workspace.consumed()
                    md.workspace.consumer_event.synchronize()
                    fill_target_tree_fia_metadata_(
                        ref,
                        mapping,
                        pool,
                        mask,
                        prefixes,
                        15,
                        len(prefixes),
                        dummy_page=7,
                    )
                    for name in ("block_tables", "blocked_mask", "active_rows"):
                        self.assertTrue(
                            torch.equal(getattr(md, name).cpu(), getattr(ref, name)),
                            name,
                        )
                        self.assertEqual(torch_npu.get_npu_format(getattr(md, name)), 2)
                    self.assertEqual(md.kv_lens_cpu, ref.kv_lens_cpu)

    def test_draft_reserved_pages_strides_and_launches(self):
        from sglang.srt.speculative.standalone_remote import (
            sr_paged_metadata_kernels_npu as kernels,
        )

        for dtype in (torch.int32, torch.int64):
            for k, s in ((2, 2), (3, 5), (4, 9)):
                for prefixes in ((0,), (127, 128), (124, 129, 0), (0, 127, 128, 129)):
                    p = SRDraftPagedPlan.build(prefixes, 4, k, s, 4, 128)
                    view = allocate_draft_view(p, "npu")
                    metrics = SRRoundMetrics("Draft")
                    view.workspace.metrics = metrics
                    mapping = torch.arange(4 * 1024, dtype=dtype).view(4, 1024) * 2
                    pool = torch.tensor([3, 0, 2, 1], dtype=dtype)[: len(prefixes)]
                    storage = (
                        torch.arange(len(prefixes) * k * s * 2, dtype=dtype).reshape(
                            len(prefixes), k, s * 2
                        )
                        * 128
                    )
                    slots = storage[:, :, ::2]
                    with patch.object(
                        kernels, "_draft", LaunchRecorder(kernels._draft)
                    ) as launch:
                        fill_draft_paged_metadata_(
                            view,
                            to_nd(mapping),
                            to_nd(pool),
                            to_nd(storage)[:, :, ::2],
                            p,
                            7,
                        )
                        self.assertEqual(launch.calls, 1)
                    self.assertEqual(
                        metrics.counts["paged_metadata_draft_h2d_count"], 1
                    )
                    self.assertEqual(
                        metrics.counts["paged_metadata_draft_h2d_bytes"], 4 * 5 * 8
                    )
                    view.workspace.consumed()
                    view.workspace.consumer_event.synchronize()
                    ref = prepare_tree_paged_view(
                        mapping,
                        pool,
                        slots,
                        prefixes,
                        128,
                        k,
                        s,
                        dummy_page=7,
                        max_pages=4,
                    )
                    n = len(prefixes)
                    self.assertTrue(
                        torch.equal(view.block_tables[: n * k].cpu(), ref[0])
                    )
                    self.assertTrue(
                        torch.equal(
                            view.branch_pages[:n, :, : ref[2].shape[2]].cpu(), ref[2]
                        )
                    )
                    self.assertTrue(torch.all(view.block_tables[n * k :].cpu() == 7))
                    self.assertFalse(view.active_rows[n * k :].cpu().any())

    def test_graph_dynamic_mapping_prefix_active_and_stable_outputs(self):
        # Kernels normally execute outside the model graph. Capture them here
        # too to detect frozen input values, then consume their fixed outputs.
        from sglang.srt.speculative.standalone_remote import (
            sr_paged_metadata_kernels_npu as kernels,
        )

        p = SRDraftPagedPlan.build([0] * 4, 4, 3, 5, 4, 128, "graph")
        view = allocate_draft_view(p, "npu")
        mapping_cpu = torch.arange(4 * 1024).reshape(4, 1024)
        mapping = to_nd(mapping_cpu)
        pool = to_nd(torch.arange(4))
        slots = to_nd(torch.arange(4 * 3 * 5).reshape(4, 3, 5) * 128)
        fill_draft_paged_metadata_(view, mapping, pool, slots, p)
        view.workspace.consumed()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            kernels.draft(view, mapping, pool, slots, p, 0)
        pointers = [x.data_ptr() for x in view.workspace.outputs]
        for bs in (1, 3, 4, 2, 3):
            prefixes = [0, 127, 128, 129][:bs]
            current = SRDraftPagedPlan.build(prefixes, 4, 3, 5, 4, 128, "graph")
            cpu_pool = torch.arange(3, -1, -1)
            cpu_slots = torch.arange(4 * 3 * 5).reshape(4, 3, 5) * 256 + bs * 128
            pool.copy_(cpu_pool)
            slots.copy_(cpu_slots)
            mapping.copy_(mapping_cpu + bs * 128)
            view.workspace.upload(current.params, (mapping, pool, slots))
            graph.replay()
            view.workspace.ready(1)
            view.workspace.consumed()
            view.workspace.consumer_event.synchronize()
            ref = prepare_tree_paged_view(
                mapping_cpu + bs * 128,
                cpu_pool[:bs],
                cpu_slots[:bs],
                prefixes,
                128,
                3,
                5,
                max_pages=4,
            )
            self.assertTrue(torch.equal(view.block_tables[: bs * 3].cpu(), ref[0]))
            self.assertTrue(
                torch.equal(view.branch_pages[:bs, :, : ref[2].shape[2]].cpu(), ref[2])
            )
            self.assertFalse(view.active_rows[bs * 3 :].cpu().any())
            self.assertEqual([x.data_ptr() for x in view.workspace.outputs], pointers)

    def test_target_graph_consumes_dynamic_preparation(self):
        md = SRTargetTreeFiaMetadata.allocate(4, 15, 4, 128, "npu", domain="graph")
        mapping_cpu = torch.arange(4 * 1024).reshape(4, 1024)
        mapping = to_nd(mapping_cpu)
        pool = to_nd(torch.arange(4))
        # Capture the consumer, as in production: preparation remains outside
        # the graph and writes exactly the addresses captured below.
        outputs = [
            empty_metadata(tuple(t.shape), t.dtype, "npu") for t in md.workspace.outputs
        ]
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for dst, src in zip(outputs, md.workspace.outputs):
                dst.copy_(src)
        for bs in (1, 3, 4, 2, 3):
            prefixes = [129, 0, 127, 128][:bs]
            plan = SRTargetPagedPlan.build(prefixes, 4, 15, 4, 128)
            mask = torch.arange(sum(15 * (x + 15) for x in prefixes)) % (bs + 2) != 0
            cpu_pool = torch.arange(3, -1, -1)
            pool.copy_(cpu_pool)
            mapping.copy_(mapping_cpu + bs * 128)
            fill_target_paged_metadata_(md, mapping, pool, to_nd(mask), plan, 7)
            graph.replay()
            md.workspace.consumed()
            md.workspace.consumer_event.synchronize()
            ref = SRTargetTreeFiaMetadata.allocate(4, 15, 4, 128, "cpu")
            fill_target_tree_fia_metadata_(
                ref,
                mapping_cpu + bs * 128,
                cpu_pool,
                mask,
                prefixes,
                15,
                bs,
                dummy_page=7,
            )
            for value, expected in zip(
                outputs, (ref.block_tables, ref.blocked_mask, ref.active_rows)
            ):
                self.assertTrue(torch.equal(value.cpu(), expected))

    def test_private_warmup_isolation_and_idempotence(self):
        mapping = to_nd(torch.arange(4 * 1024).reshape(4, 1024))
        before = mapping.cpu().clone()
        md = SRTargetTreeFiaMetadata.allocate(4, 15, 4, 128, "npu", domain="graph")
        p = SRDraftPagedPlan.build((), 4, 3, 5, 4, 128, "graph")
        view = allocate_draft_view(p, "npu")
        warm_target_paged_metadata(md, mapping)
        warm_draft_paged_metadata(view, mapping)
        with patch(
            "sglang.srt.speculative.standalone_remote.sr_paged_metadata.empty_metadata",
            side_effect=AssertionError("repeat allocation"),
        ):
            warm_target_paged_metadata(md, mapping)
            warm_draft_paged_metadata(view, mapping)
        self.assertTrue(torch.equal(before, mapping.cpu()))
        self.assertTrue(md._paged_warm_complete)
        self.assertTrue(view._paged_warm_complete)


if __name__ == "__main__":
    unittest.main()
