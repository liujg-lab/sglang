"""CPU planning/lifetime contracts. Kernel numerics require the NPU manual test."""

import ast
import pathlib
import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.speculative.standalone_remote.sr_paged_metadata import (
    SRDraftPagedPlan,
    SRPagedMetadataCache,
    SRPagedMetadataSubmittedError,
    SRPagedMetadataWorkspace,
    SRTargetPagedPlan,
    allocate_draft_view,
    fill_draft_paged_metadata_,
    fill_target_paged_metadata_,
    finish_paged_metadata,
    poison_paged_metadata,
    report_paged_metadata,
)
from sglang.srt.speculative.standalone_remote.sr_round_metrics import SRRoundMetrics
from sglang.srt.speculative.standalone_remote.verifier.sr_target_tree_fia import (
    SRTargetTreeFiaMetadata,
    fill_target_tree_fia_metadata_,
)
from sglang.srt.speculative.standalone_remote.drafter.sr_tree_paged_layout import (
    prepare_tree_paged_view,
)
from sglang.srt.speculative.standalone_remote.drafter.sr_tree_paged_layout import (
    SRTreePagedMetadata,
    build_step_context_lens,
    context_lens_list,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")
KERNELS = "sglang.srt.speculative.standalone_remote.sr_paged_metadata_kernels_npu"
ROOT = pathlib.Path(__file__).resolve().parents[4]


def target_fixture(prefixes=(127, 128, 129), capacity=4, q=15, pages=2):
    plan = SRTargetPagedPlan.build(prefixes, capacity, q, pages, 128)
    md = SRTargetTreeFiaMetadata.allocate(capacity, q, pages, 128, "cpu")
    md.workspace = SRPagedMetadataWorkspace(
        "cpu", capacity, 3, (md.block_tables, md.blocked_mask, md.active_rows)
    )
    mapping = torch.arange(5 * 1024).reshape(5, 1024) * 2
    pool = torch.tensor([4, 1, 3, 0, 2])[: len(prefixes)]
    mask = torch.arange(sum(q * (p + q) for p in prefixes)) % 3 != 0
    return md, mapping, pool, mask, plan


def target_cpu_adapter(md, mapping, pool, mask, plan, dummy):
    # Test stand-in for the device consumer: read the uploaded packet, not the
    # immutable CPU plan, so stale/wrong packets cannot pass these tests.
    for b, (prefix, start, active) in enumerate(md.workspace.params.tolist()):
        md.active_rows[b] = bool(active)
        for j in range(plan.pages):
            md.block_tables[b, j] = (
                mapping[pool[b], j * plan.page_size] // plan.page_size
                if active and j * plan.page_size < prefix + plan.queries
                else dummy
            )
        for q in range(plan.queries):
            md.blocked_mask[b, 0, q].fill_(True)
            if active:
                md.blocked_mask[b, 0, q, : prefix + plan.queries] = ~mask[
                    start
                    + q * (prefix + plan.queries) : start
                    + (q + 1) * (prefix + plan.queries)
                ]
            else:
                md.blocked_mask[b, 0, q, 0] = False


def draft_cpu_adapter(view, mapping, pool, slots, plan, dummy):
    for b, (rem, shared, query, reserved, active) in enumerate(
        view.workspace.params.tolist()
    ):
        for k in range(plan.topk):
            row = b * plan.topk + k
            view.active_rows[row] = bool(active)
            for j in range(plan.pages):
                if active and j < shared:
                    value = mapping[pool[b], j * plan.page_size] // plan.page_size
                elif active and j < shared + query:
                    step = min(
                        max((j - shared) * plan.page_size - rem, 0), plan.steps - 1
                    )
                    value = slots[b, k, step] // plan.page_size
                else:
                    value = dummy
                view.block_tables[row, j] = value
            for j in range(plan.branch_capacity):
                step = min(max(j * plan.page_size - rem, 0), plan.steps - 1)
                view.branch_pages[b, k, j] = (
                    slots[b, k, step] // plan.page_size
                    if active and j < reserved
                    else dummy
                )


class TestPagedMetadata(unittest.TestCase):
    def setUp(self):
        self.kernels = types.ModuleType(KERNELS)
        self.kernels.target = Mock(side_effect=target_cpu_adapter)
        self.kernels.draft = Mock(side_effect=draft_cpu_adapter)
        self.modules = patch.dict(sys.modules, {KERNELS: self.kernels})
        self.modules.start()

    def tearDown(self):
        self.modules.stop()

    def test_target_packet_and_mask_against_reference(self):
        for prefixes, cap in [((), 1), ((0,), 1), ((127, 128, 129), 4), ((256, 1), 2)]:
            with self.subTest(prefixes=prefixes):
                md, mapping, pool, mask, plan = target_fixture(prefixes, cap, pages=4)
                ref = SRTargetTreeFiaMetadata.allocate(
                    cap, plan.queries, plan.pages, 128, "cpu"
                )
                fill_target_tree_fia_metadata_(
                    ref,
                    mapping,
                    pool,
                    mask,
                    prefixes,
                    plan.queries,
                    len(prefixes),
                    dummy_page=7,
                )
                fill_target_paged_metadata_(md, mapping, pool, mask, plan, 7)
                for name in ("block_tables", "blocked_mask", "active_rows"):
                    self.assertTrue(
                        torch.equal(getattr(md, name), getattr(ref, name)), name
                    )
                self.assertEqual(md.kv_lens_cpu, ref.kv_lens_cpu)
                self.assertEqual(md.q_lens_cpu, ref.q_lens_cpu)

    def test_draft_reserved_pages_and_padding_against_reference(self):
        for prefixes, cap, k, s in [
            ((0,), 1, 1, 1),
            ((124,), 1, 3, 5),
            ((127, 128, 129), 4, 3, 5),
            ((1, 255), 2, 2, 130),
        ]:
            with self.subTest(prefixes=prefixes, steps=s):
                p = SRDraftPagedPlan.build(prefixes, cap, k, s, 4, 128)
                v = allocate_draft_view(p, "cpu")
                mapping = torch.arange(4 * 1024, dtype=torch.int32).reshape(4, 1024)
                pool = torch.arange(len(prefixes) - 1, -1, -1)
                slots = (
                    torch.arange(len(prefixes) * k * s * 2).reshape(
                        len(prefixes), k, s * 2
                    )[..., ::2]
                    + 512
                )
                expected = prepare_tree_paged_view(
                    mapping,
                    pool,
                    slots,
                    prefixes,
                    128,
                    k,
                    s,
                    dummy_page=9,
                    max_pages=4,
                    capture_rows=cap * k,
                )
                fill_draft_paged_metadata_(v, mapping, pool, slots, p, 9)
                self.assertTrue(torch.equal(v.block_tables, expected[0]))
                self.assertTrue(torch.equal(v.active_rows, expected[3]))
                self.assertTrue(
                    torch.equal(
                        v.branch_pages[: len(prefixes), :, : expected[2].shape[2]],
                        expected[2],
                    )
                )
                self.assertTrue((v.branch_pages[len(prefixes) :] == 9).all())
                if prefixes == (124,):
                    self.assertEqual(p.params[0][2:4], (1, 2))

    def test_target_strided_inputs_and_mask(self):
        md, mapping, pool, mask, plan = target_fixture((129,), 1)
        backing = torch.zeros(mask.numel() * 2, dtype=torch.bool)
        backing[::2] = mask
        mapping = mapping[:, ::2]
        fill_target_paged_metadata_(md, mapping, pool, backing[::2], plan)
        self.assertTrue(torch.equal(md.blocked_mask[0, 0, 0, :144], ~mask[:144]))

    def test_bad_input_does_not_upload_or_write(self):
        for failure in ("mask", "pool", "layout", "output"):
            md, mapping, pool, mask, p = target_fixture((1,), 1)
            md.block_tables.fill_(99)
            md.workspace.params.fill_(77)
            if failure == "mask":
                mask = mask[:-1]
            elif failure == "pool":
                pool = pool[:0]
            elif failure == "layout":
                mask = torch.empty((2, mask.numel() // 2), dtype=torch.bool).t()
            else:
                md.workspace.outputs = (
                    md.block_tables[:, :1],
                    md.blocked_mask,
                    md.active_rows,
                )
            with self.subTest(failure=failure), self.assertRaises(
                (ValueError, RuntimeError)
            ):
                fill_target_paged_metadata_(md, mapping, pool, mask, p)
            self.assertTrue((md.block_tables == 99).all())
            self.assertTrue((md.workspace.params == 77).all())
            self.assertEqual(md.workspace.generation, 0)
        self.kernels.target.assert_not_called()

    def test_plan_capacity_errors(self):
        for args in [
            ([1, 2], 1, 15, 2, 128),
            ([-1], 1, 15, 2, 128),
            ([255], 1, 15, 2, 128),
        ]:
            with self.assertRaises(ValueError):
                SRTargetPagedPlan.build(*args)
        with self.assertRaises(ValueError):
            SRDraftPagedPlan.build([255], 1, 3, 5, 1, 128)

    def test_same_capacity_reuse_and_dynamic_mapping(self):
        md, mapping, pool, mask, p = target_fixture((1,), 1)
        ptrs = [t.data_ptr() for t in (md.workspace.params, *md.workspace.outputs)]
        fill_target_paged_metadata_(md, mapping, pool, mask, p)
        first = md.block_tables.clone()
        finish_paged_metadata(
            types.SimpleNamespace(_sr_target_paged_workspace=md.workspace)
        )
        pool[0] = 0
        with patch("torch.empty", side_effect=AssertionError("steady allocation")):
            fill_target_paged_metadata_(md, mapping, pool, mask, p)
        self.assertFalse(torch.equal(first, md.block_tables))
        self.assertEqual(
            ptrs, [t.data_ptr() for t in (md.workspace.params, *md.workspace.outputs)]
        )
        self.assertEqual(md.workspace.generation, 2)

    def test_missing_consumer_completion_blocks_reuse(self):
        md, mapping, pool, mask, p = target_fixture((1,), 1)
        fill_target_paged_metadata_(md, mapping, pool, mask, p)
        with self.assertRaisesRegex(RuntimeError, "consumer completion"):
            fill_target_paged_metadata_(md, mapping, pool, mask, p)
        self.assertEqual(self.kernels.target.call_count, 1)

    def test_kernel_failure_poison_and_holds(self):
        md, mapping, pool, mask, p = target_fixture((1,), 1)
        self.kernels.target.side_effect = RuntimeError("launch")
        with self.assertRaises(SRPagedMetadataSubmittedError):
            fill_target_paged_metadata_(md, mapping, pool, mask, p)
        self.assertTrue(md.workspace.unresolved)
        self.assertIs(md.workspace.holds[0], mapping)
        with self.assertRaises(SRPagedMetadataSubmittedError):
            fill_target_paged_metadata_(md, mapping, pool, mask, p)
        self.assertEqual(self.kernels.target.call_count, 1)

    def test_ready_record_failure_poison(self):
        md, mapping, pool, mask, p = target_fixture((1,), 1)
        md.workspace.ready_event = Mock()
        md.workspace.ready_event.record.side_effect = RuntimeError("record")
        with self.assertRaises(SRPagedMetadataSubmittedError):
            fill_target_paged_metadata_(md, mapping, pool, mask, p)
        self.assertTrue(md.workspace.unresolved)

    def test_staging_wait_and_consumer_event_are_distinct(self):
        md, mapping, pool, mask, p = target_fixture((1,), 1)
        fill_target_paged_metadata_(md, mapping, pool, mask, p)
        event = md.workspace.consumer_event = Mock()
        md.workspace.consumed()
        event.record.assert_called_once()
        md.workspace.upload_pending = True
        upload = md.workspace.upload_event = Mock()
        upload.query.return_value = False
        fill_target_paged_metadata_(md, mapping, pool, mask, p)
        upload.synchronize.assert_called_once()
        event.synchronize.assert_not_called()

    def test_wait_and_consumer_failures_retain_resources(self):
        for kind in ("wait", "consumer", "retire"):
            md, mapping, pool, mask, p = target_fixture((1,), 1)
            fill_target_paged_metadata_(md, mapping, pool, mask, p)
            event = Mock()
            with self.subTest(kind=kind), self.assertRaises(
                SRPagedMetadataSubmittedError
            ):
                if kind == "consumer":
                    md.workspace.consumer_event = event
                    event.record.side_effect = RuntimeError("record")
                    md.workspace.consumed()
                elif kind == "retire":
                    md.workspace.consumed()
                    md.workspace.consumer_event = event
                    event.query.side_effect = RuntimeError("query")
                    md.workspace.completed()
                else:
                    md.workspace.consumed()
                    md.workspace.upload_event = event
                    md.workspace.upload_pending = True
                    event.query.return_value = False
                    event.synchronize.side_effect = RuntimeError("wait")
                    fill_target_paged_metadata_(md, mapping, pool, mask, p)
            self.assertTrue(md.workspace.unresolved)
            self.assertIs(md.workspace.holds[0], mapping)

    def test_wrong_stream_and_output_replacement_rejected(self):
        md, mapping, pool, mask, p = target_fixture((1,), 1)
        md.workspace.check()
        with patch(
            "sglang.srt.speculative.standalone_remote.sr_paged_metadata.stream_key",
            return_value=("cpu", 0, 9),
        ):
            with self.assertRaisesRegex(RuntimeError, "different stream"):
                fill_target_paged_metadata_(md, mapping, pool, mask, p)
        md.workspace.outputs = (
            md.block_tables.clone(),
            md.blocked_mask,
            md.active_rows,
        )
        with self.assertRaisesRegex(RuntimeError, "storage"):
            fill_target_paged_metadata_(md, mapping, pool, mask, p)
        self.kernels.target.assert_not_called()

    def test_eager_retirement_and_atomic_allocation(self):
        cache = SRPagedMetadataCache()
        p = SRDraftPagedPlan.build([0], 1, 3, 5, 2, 128)
        first = cache.get("stream", 1, lambda: allocate_draft_view(p, "cpu"))
        first.workspace.pending_consumer = True
        with self.assertRaises(MemoryError):
            cache.get("stream", 2, Mock(side_effect=MemoryError()))
        self.assertIs(cache.get("stream", 1, Mock()), first)
        second = cache.get("stream", 2, lambda: allocate_draft_view(p, "cpu"))
        self.assertIn(first, cache.retired)
        first.workspace.consumed()
        self.assertIs(cache.get("stream", 2, Mock()), second)
        self.assertEqual(cache.retired, [])

    def test_metric_counts_and_byte_gauges(self):
        metrics = SRRoundMetrics("Target")
        md, mapping, pool, mask, p = target_fixture((1,), 1)
        md.workspace.metrics = metrics
        with metrics.round():
            for _ in range(2):
                fill_target_paged_metadata_(md, mapping, pool, mask, p)
                md.workspace.consumed()
        self.assertEqual(metrics.counts["paged_metadata_target_kernel_calls"], 4)
        self.assertEqual(metrics.counts["paged_metadata_target_workspace_grow"], 1)
        owner = types.SimpleNamespace(_target_fia_graph_metadata={1: md})
        report_paged_metadata(owner, "target", metrics)
        self.assertEqual(
            metrics.counts["paged_metadata_target_current_bytes"], md.workspace.bytes
        )
        self.assertEqual(metrics.counts["paged_metadata_target_retired_bytes"], 0)

    def test_empty_capacity_no_launch(self):
        md, mapping, pool, mask, p = target_fixture((), 0)
        fill_target_paged_metadata_(md, mapping, pool, mask, p)
        self.kernels.target.assert_not_called()

    def test_source_contracts_no_graph_table_recopy(self):
        path = (
            ROOT / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        method = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "bind_sr_tree_paged_replay"
        )
        fused = next(
            n
            for n in method.body
            if isinstance(n, ast.If) and "view is not None" in ast.unparse(n.test)
        )
        calls = [
            n.func.attr
            for n in ast.walk(fused)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        ]
        self.assertFalse(set(calls) & {"copy_", "fill_", "zero_"})

    def backend_fixture(self):
        # Load the actual backend methods without requiring torch_npu/model
        # dependencies on CPU. Only the three device kernels are stand-ins.
        path = (
            ROOT / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef)
            and n.name == "AscendAttnMultiStepDraftBackend"
        )
        names = {
            "_prepare_fused_sr_tree_paged",
            "restore_sr_tree_paged_eager",
            "bind_sr_tree_paged_replay",
            "_sr_clear_paged_round_state",
        }
        nodes = [
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        ns = dict(
            torch=torch,
            NpuGraphPreparationError=RuntimeError,
            SRTreePagedMetadata=SRTreePagedMetadata,
            build_step_context_lens=build_step_context_lens,
            context_lens_list=context_lens_list,
            ForwardMetadata=types.SimpleNamespace,
        )
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
        backend = types.SimpleNamespace(
            topk=3,
            speculative_num_steps=5,
            page_size=128,
            paged_impl_selected=lambda: True,
            _paged_round_dummy=0,
            _paged_round_impl="paged_fia",
            _tree_replay_raw_bs=3,
        )
        for name in names:
            setattr(backend, name, types.MethodType(ns[name], backend))
        mapping = torch.arange(5 * 512).reshape(5, 512)
        backend.attn_backends = [
            types.SimpleNamespace(
                req_to_token=mapping,
                speculative_step_id=i,
                forward_metadata=None,
                bind_sr_tree_paged_metadata=Mock(),
            )
            for i in range(4)
        ]
        plan = SRDraftPagedPlan.build((), 4, 3, 5, 4, 128, "graph")
        view = allocate_draft_view(plan, "cpu")
        backend._sr_paged_graph_views = {(4, 4): view}
        backend._paged_graph_table_view = Mock(
            return_value=(view.block_tables, view.active_rows)
        )
        batch = types.SimpleNamespace(
            req_pool_indices=torch.tensor([4, 1, 3]),
            _sr_paged_replay_plan=types.SimpleNamespace(capture_bs=4, kv_bucket=4),
        )
        slots = torch.arange(3 * 3 * 5).reshape(3, 3, 5) * 128
        return backend, batch, slots, view

    def test_backend_selects_graph_output_and_fallback_never_copies_kv(self):
        backend, batch, slots, captured = self.backend_fixture()
        view = backend._prepare_fused_sr_tree_paged(
            batch, slots, [124, 128, 129], 2, 0, None
        )
        self.assertIs(view, captured)
        self.assertEqual(view.block_tables.shape, (12, 4))
        backend.bind_sr_tree_paged_replay(4, 4)
        self.assertEqual(self.kernels.draft.call_count, 1)
        self.assertTrue(backend.restore_sr_tree_paged_eager())
        eager = backend._sr_draft_paged_view
        self.assertEqual(eager.block_tables.shape, (9, 4))
        self.assertTrue(torch.equal(eager.block_tables, captured.block_tables[:9]))
        self.assertFalse(captured.workspace.pending_consumer)
        self.assertEqual(self.kernels.draft.call_count, 2)
        for inner in backend.attn_backends:
            self.assertIs(inner.forward_metadata.block_tables, eager.block_tables)
        self.assertFalse(backend.restore_sr_tree_paged_eager())
        self.assertEqual(self.kernels.draft.call_count, 2)
        finish_paged_metadata(backend)
        backend._sr_clear_paged_round_state()
        self.assertIsNone(backend._sr_draft_paged_view)

    def test_backend_uncertain_completion_blocks_fallback_and_clear(self):
        backend, batch, slots, _ = self.backend_fixture()
        view = backend._prepare_fused_sr_tree_paged(
            batch, slots, [0, 127, 129], 2, 0, None
        )
        with self.assertRaisesRegex(RuntimeError, "consumer"):
            backend._sr_clear_paged_round_state()
        view.workspace.poison()
        with self.assertRaises(SRPagedMetadataSubmittedError):
            backend.restore_sr_tree_paged_eager()
        self.assertIs(backend._sr_draft_paged_view, view)
        self.assertEqual(self.kernels.draft.call_count, 1)

    def test_generation_and_already_consumed_view_reject_replay(self):
        backend, batch, slots, view = self.backend_fixture()
        backend._prepare_fused_sr_tree_paged(batch, slots, [0, 127, 129], 2, 0, None)
        view.generation -= 1
        with self.assertRaisesRegex(RuntimeError, "generation"):
            backend.bind_sr_tree_paged_replay(4, 4)
        view.generation += 1
        with self.assertRaisesRegex(RuntimeError, "round generation"):
            backend.bind_sr_tree_paged_replay(4, 4, generation=view.generation - 1)
        finish_paged_metadata(backend)
        with self.assertRaisesRegex(RuntimeError, "consumed"):
            backend.bind_sr_tree_paged_replay(4, 4)

    def test_kernel_failure_in_backend_does_not_select_eager(self):
        backend, batch, slots, captured = self.backend_fixture()
        self.kernels.draft.side_effect = RuntimeError("submitted")
        with self.assertRaises(SRPagedMetadataSubmittedError):
            backend._prepare_fused_sr_tree_paged(
                batch, slots, [0, 127, 129], 2, 0, None
            )
        self.assertTrue(captured.workspace.unresolved)
        self.assertFalse(hasattr(backend, "_sr_paged_eager_cache"))
        self.assertEqual(self.kernels.draft.call_count, 1)

    def test_nd_allocator_uses_flat_backing_and_rejects_wrong_view_format(self):
        from sglang.srt.speculative.standalone_remote import sr_paged_metadata as module

        fake = types.SimpleNamespace(
            empty_with_format=Mock(
                side_effect=lambda shape, **kwargs: torch.empty(
                    shape, dtype=kwargs["dtype"]
                )
            ),
            get_npu_format=Mock(return_value=2),
        )
        with patch.dict(sys.modules, {"torch_npu": fake}), patch.object(
            module.torch, "device", return_value=types.SimpleNamespace(type="npu")
        ):
            result = module.empty_metadata((2, 1, 3, 128), torch.bool, "npu")
            self.assertEqual(fake.empty_with_format.call_args.args[0], (768,))
            self.assertEqual(fake.empty_with_format.call_args.kwargs["acl_format"], 2)
            self.assertEqual(result.shape, (2, 1, 3, 128))
            self.assertEqual(result.data_ptr(), result._base.data_ptr())
            fake.get_npu_format.side_effect = [2, 0, 0]
            with self.assertRaisesRegex(RuntimeError, "view.*expected ND"):
                module.empty_metadata((2, 1, 3, 128), torch.bool, "npu")

    def test_plans_reject_non_host_lengths_without_reading_values(self):
        for builder, args in (
            (SRTargetPagedPlan.build, (1, 15, 4, 128)),
            (SRDraftPagedPlan.build, (1, 3, 5, 4, 128)),
        ):
            with self.assertRaisesRegex(ValueError, "CPU"):
                builder(torch.empty(1, device="meta"), *args)

    def test_private_warmup_failure_retains_ownership_and_never_marks_complete(self):
        from sglang.srt.speculative.standalone_remote import sr_paged_metadata as module

        md, mapping, _, _, _ = target_fixture((1,), 1)
        private, _, _, _, _ = target_fixture((0,), 1)
        private.workspace.consumer_event = Mock()
        private.workspace.consumer_event.synchronize.side_effect = RuntimeError(
            "completion"
        )
        with patch.object(SRTargetTreeFiaMetadata, "allocate", return_value=private):
            with self.assertRaisesRegex(SRPagedMetadataSubmittedError, "warmup"):
                module.warm_target_paged_metadata(md, mapping)
        self.assertIs(md._paged_warm_holds, private)
        self.assertTrue(private.workspace.unresolved)
        self.assertFalse(getattr(md, "_paged_warm_complete", False))

    def test_layout_warmup_preserves_primary_error_and_pending_resources(self):
        path = (
            ROOT
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_tree_drafter.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        method = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_sr_warm_layout_shapes"
        )
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_paged_layout import (
            ALLOC_LEASE,
            ALLOC_ORDINARY,
            tree_paged_shape_key,
        )

        log = Mock()
        ns = dict(
            torch=torch,
            SimpleNamespace=types.SimpleNamespace,
            logger=log,
            tree_paged_shape_key=tree_paged_shape_key,
            ALLOC_ORDINARY=ALLOC_ORDINARY,
            ALLOC_LEASE=ALLOC_LEASE,
        )
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
        for status in (
            "success",
            "pre_submit",
            "cleanup_failure",
            "pending",
            "unresolved",
        ):
            with self.subTest(status=status):
                backend, _, _, view = self.backend_fixture()
                backend._sr_draft_paged_view = view
                view.workspace.pending_consumer = status == "pending"
                view.workspace.unresolved = status == "unresolved"
                held = (object(),)
                view.workspace.holds = held
                primary = RuntimeError("original warmup failure")
                backend.prepare_sr_tree_paged_eager = Mock(
                    side_effect=None if status == "success" else primary
                )
                real_clear = backend._sr_clear_paged_round_state
                backend._sr_clear_paged_round_state = Mock(
                    wraps=real_clear,
                    side_effect=RuntimeError("secondary cleanup failure")
                    if status == "cleanup_failure"
                    else None,
                )
                worker = types.SimpleNamespace(
                    draft_attn_backend=backend,
                    req_to_token_pool=types.SimpleNamespace(
                        req_to_token=backend.attn_backends[0].req_to_token
                    ),
                    _paged_dummy_page=0,
                    page_size=128,
                    topk=3,
                    speculative_num_steps=5,
                    draft_model_runner=types.SimpleNamespace(token_to_kv_pool=None),
                    _sr_warmup_page_buckets=lambda: [1],
                    _sr_warmup_raw_batch_sizes=lambda: [1],
                )
                if status == "success":
                    self.assertTrue(ns[method.name](worker))
                else:
                    with self.assertRaises(RuntimeError) as caught:
                        ns[method.name](worker)
                    self.assertIs(caught.exception, primary)
                    backend.prepare_sr_tree_paged_eager.assert_called_once()
                if status in ("pending", "unresolved"):
                    backend._sr_clear_paged_round_state.assert_not_called()
                    self.assertIs(backend._sr_draft_paged_view, view)
                    self.assertIs(view.workspace.holds, held)
                    self.assertTrue(view.workspace.unresolved)
                else:
                    backend._sr_clear_paged_round_state.assert_called_once()
                    if status == "cleanup_failure":
                        self.assertIs(backend._sr_draft_paged_view, view)
                        self.assertIs(view.workspace.holds, held)
                        log.exception.assert_called()
                    else:
                        self.assertIsNone(backend._sr_draft_paged_view)


if __name__ == "__main__":
    unittest.main()
