"""Device-agnostic helpers for SPECTRE / STANDALONE_REMOTE dual-backend."""

import ast
import unittest
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=8, suite="stage-a-test-cpu")

_REPO = Path(__file__).resolve().parents[4]


def _noncontiguous_req_to_token(page_ids, page_size: int) -> torch.Tensor:
    """Fill req_to_token so physical page numbers are the given ids (not 0,1,2...)."""
    bs = len(page_ids)
    n_pages = max(len(p) for p in page_ids)
    ctx = n_pages * page_size
    req = torch.zeros((bs, ctx), dtype=torch.int32)
    for b, pages in enumerate(page_ids):
        for p, pid in enumerate(pages):
            start = p * page_size
            req[b, start : start + page_size] = int(pid) * page_size + torch.arange(
                page_size, dtype=torch.int32
            )
    return req


def _reference_tree_draft_block_tables(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
    topk: int,
    step_id: int,
    num_steps: int,
) -> torch.Tensor:
    bs = int(req_pool_indices.shape[0])
    kv_extra = int(step_id) + 1
    rows = []
    for b in range(bs):
        pool = int(req_pool_indices[b])
        seq = int(seq_lens[b])
        kv_len = seq + kv_extra
        n_pages = (kv_len + page_size - 1) // page_size
        if topk == 1:
            token_pos = [p * page_size for p in range(n_pages)]
            pages = [
                int(req_to_token[pool, pos]) // page_size if pos < kv_len else 0
                for pos in token_pos
            ]
            rows.append(pages)
            continue
        if page_size == 1:
            for k in range(topk):
                pages = []
                for p in range(n_pages):
                    if p < seq:
                        pos = p
                    else:
                        pos = seq + k * num_steps + (p - seq)
                    pages.append(
                        int(req_to_token[pool, pos]) if p < kv_len else 0
                    )
                rows.append(pages)
            continue
        last_page_len = seq % page_size
        prefix_base = seq - last_page_len
        num_new_pages = (last_page_len + num_steps + page_size - 1) // page_size
        n_shared = prefix_base // page_size
        for k in range(topk):
            pages = []
            for p in range(n_pages):
                if p < n_shared:
                    pos = p * page_size
                else:
                    pos = prefix_base + k * num_new_pages * page_size + (
                        p - n_shared
                    ) * page_size
                pages.append(
                    int(req_to_token[pool, pos]) // page_size if p < n_pages else 0
                )
            rows.append(pages)
    n_cols = max((len(r) for r in rows), default=0)
    out = torch.zeros((bs * topk, n_cols), dtype=torch.int32)
    for i, r in enumerate(rows):
        out[i, : len(r)] = torch.tensor(r, dtype=torch.int32)
    return out


def _load_tree_draft_helpers():
    helper_names = {
        "expand_seq_lens_for_spec_topk",
        "normalize_tree_draft_kv_lens",
        "build_tree_draft_block_tables",
        "build_draft_graph_step_kv_lens",
        "expand_fia_cpu_update_inputs",
        "resolve_fia_update_count",
        "_structured_op_name",
        "_structured_kwargs",
        "inspect_dispatch_record",
        "validate_tree_draft_fia_records",
        "validate_draft_graph_step_kv_lens",
        "normalize_fia_op_name",
        "_schema_has_kv_attr",
        "_dump_unreadable_dispatch_record",
        "tree_reselect_parent_rows",
    }
    class_names = {"NpuGraphReplaySubmittedError", "NpuGraphPreparationError"}
    assign_names = {"TREE_DRAFT_FIA_OP_NAMES", "_dumped_unreadable_dispatch_record"}
    try:
        from sglang.srt.speculative.spec_utils import (
            NpuGraphPreparationError,
            NpuGraphReplaySubmittedError,
            TREE_DRAFT_FIA_OP_NAMES,
            build_draft_graph_step_kv_lens,
            build_tree_draft_block_tables,
            expand_fia_cpu_update_inputs,
            expand_seq_lens_for_spec_topk,
            inspect_dispatch_record,
            normalize_fia_op_name,
            normalize_tree_draft_kv_lens,
            resolve_fia_update_count,
            tree_reselect_parent_rows,
            validate_draft_graph_step_kv_lens,
            validate_tree_draft_fia_records,
        )

        return SimpleNamespace(
            build_tree_draft_block_tables=build_tree_draft_block_tables,
            expand_seq_lens_for_spec_topk=expand_seq_lens_for_spec_topk,
            normalize_tree_draft_kv_lens=normalize_tree_draft_kv_lens,
            build_draft_graph_step_kv_lens=build_draft_graph_step_kv_lens,
            expand_fia_cpu_update_inputs=expand_fia_cpu_update_inputs,
            resolve_fia_update_count=resolve_fia_update_count,
            inspect_dispatch_record=inspect_dispatch_record,
            validate_tree_draft_fia_records=validate_tree_draft_fia_records,
            validate_draft_graph_step_kv_lens=validate_draft_graph_step_kv_lens,
            normalize_fia_op_name=normalize_fia_op_name,
            NpuGraphReplaySubmittedError=NpuGraphReplaySubmittedError,
            NpuGraphPreparationError=NpuGraphPreparationError,
            TREE_DRAFT_FIA_OP_NAMES=TREE_DRAFT_FIA_OP_NAMES,
            tree_reselect_parent_rows=tree_reselect_parent_rows,
        )
    except Exception:
        pass
    src_path = _REPO / "python/sglang/srt/speculative/spec_utils.py"
    tree = ast.parse(src_path.read_text())
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in helper_names:
            keep.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in class_names:
            keep.append(node)
        elif isinstance(node, ast.Assign):
            if any(
                isinstance(t, ast.Name) and t.id in assign_names for t in node.targets
            ):
                keep.append(node)
    mod = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {"torch": torch, "Optional": Optional, "Mapping": __import__("collections.abc", fromlist=["Mapping"]).Mapping, "logging": __import__("logging"), "logger": __import__("logging").getLogger("tree_draft_helpers")}
    exec(compile(mod, str(src_path), "exec"), ns)
    return SimpleNamespace(
        build_tree_draft_block_tables=ns["build_tree_draft_block_tables"],
        expand_seq_lens_for_spec_topk=ns["expand_seq_lens_for_spec_topk"],
        normalize_tree_draft_kv_lens=ns["normalize_tree_draft_kv_lens"],
        build_draft_graph_step_kv_lens=ns["build_draft_graph_step_kv_lens"],
        expand_fia_cpu_update_inputs=ns["expand_fia_cpu_update_inputs"],
        resolve_fia_update_count=ns["resolve_fia_update_count"],
        inspect_dispatch_record=ns["inspect_dispatch_record"],
        validate_tree_draft_fia_records=ns["validate_tree_draft_fia_records"],
        validate_draft_graph_step_kv_lens=ns["validate_draft_graph_step_kv_lens"],
        normalize_fia_op_name=ns["normalize_fia_op_name"],
        NpuGraphReplaySubmittedError=ns["NpuGraphReplaySubmittedError"],
        NpuGraphPreparationError=ns["NpuGraphPreparationError"],
        TREE_DRAFT_FIA_OP_NAMES=ns["TREE_DRAFT_FIA_OP_NAMES"],
        tree_reselect_parent_rows=ns["tree_reselect_parent_rows"],
    )


def _load_sr_tree_expand_methods():
    """CPU-safe expand_batch/_expand_one without importing SRTreeDrafter."""
    from sglang.srt.speculative.standalone_remote.sr_align import (
        is_device_context_error,
    )

    draft_helpers = _load_tree_draft_helpers()
    src_path = (
        _REPO
        / "python/sglang/srt/speculative/standalone_remote/drafter/sr_tree_drafter.py"
    )
    tree = ast.parse(src_path.read_text())
    methods = {}
    helper_nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_as_2d",
            "_wait_d2h_event",
            "_ensure_host_staging_slot",
        }:
            helper_nodes.append(node)
        if isinstance(node, ast.ClassDef) and node.name == "SRTreeDrafter":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in {
                    "expand_batch",
                    "_expand_one",
                    "_pack_tree_windows",
                    "_d2h_tree_outputs",
                    "_acquire_host_staging",
                }:
                    methods[item.name] = item
    future = ast.parse("from __future__ import annotations").body
    ns = {
        "NpuGraphReplaySubmittedError": draft_helpers.NpuGraphReplaySubmittedError,
        "is_device_context_error": is_device_context_error,
        "logger": __import__("logging").getLogger("sr_tree_expand"),
        "List": list,
        "SRTreeWindow": tuple,
        "torch": __import__("torch"),
    }
    mod = ast.Module(
        body=future + helper_nodes + list(methods.values()), type_ignores=[]
    )
    ast.fix_missing_locations(mod)
    exec(compile(mod, str(src_path), "exec"), ns)
    return SimpleNamespace(
        expand_batch=ns["expand_batch"],
        expand_one=ns["_expand_one"],
        NpuGraphReplaySubmittedError=draft_helpers.NpuGraphReplaySubmittedError,
    )


class TestRemoteSpecDevice(CustomTestCase):
    def _import_or_skip(self, fn):
        try:
            return fn()
        except Exception as e:
            self.skipTest(f"sglang runtime deps missing: {e}")
    def test_idle_input_uses_explicit_device(self):
        try:
            from sglang.srt.speculative.eagle_info import EagleVerifyInput
        except Exception as e:
            self.skipTest(f"sglang runtime deps missing: {e}")
        idle = EagleVerifyInput.create_idle_input(
            topk=2, spec_steps=3, num_verify_tokens=4, device="cpu"
        )
        self.assertEqual(idle.draft_token.device.type, "cpu")
        self.assertEqual(idle.custom_mask.device.type, "cpu")
        self.assertEqual(idle.retrive_index.shape[1], 4)

    def test_mm_serialize_moves_non_cpu_fake_via_cpu_tensor(self):
        try:
            from sglang.srt.speculative.spec_utils import tensor_on_accelerator
            from sglang.srt.speculative.spectre.spectre_mm_transport import (
                _tensor_gpu_bytes,
                _to_cpu_contiguous_tensor,
            )
        except Exception as e:
            self.skipTest(f"sglang runtime deps missing: {e}")

        tensor = torch.arange(4, dtype=torch.float32)
        out = _to_cpu_contiguous_tensor(tensor)
        self.assertEqual(out.device.type, "cpu")
        self.assertTrue(out.is_contiguous())
        self.assertEqual(_tensor_gpu_bytes(tensor), 0)
        self.assertFalse(tensor_on_accelerator(tensor))

        fake_npu = MagicMock(spec=torch.Tensor)
        fake_npu.device = SimpleNamespace(type="npu")
        fake_npu.numel.return_value = 8
        fake_npu.element_size.return_value = 4
        self.assertEqual(_tensor_gpu_bytes(fake_npu), 32)
        self.assertTrue(tensor_on_accelerator(fake_npu))

        fake_cuda = MagicMock(spec=torch.Tensor)
        fake_cuda.device = SimpleNamespace(type="cuda")
        fake_cuda.numel.return_value = 2
        fake_cuda.element_size.return_value = 2
        self.assertEqual(_tensor_gpu_bytes(fake_cuda), 4)

    def test_sr_mm_serialize_cpu(self):
        try:
            from sglang.srt.speculative.standalone_remote.sr_mm_payload import (
                _to_cpu_contiguous_tensor as sr_to_cpu,
            )
        except Exception as e:
            self.skipTest(f"sglang runtime deps missing: {e}")

        tensor = torch.ones(2, 2)
        out = sr_to_cpu(tensor)
        self.assertEqual(out.device.type, "cpu")

    def test_capability_table(self):
        try:
            from sglang.srt.speculative.spec_utils import (
                device_backend_key,
                tree_verify_method_available,
            )
        except Exception as e:
            self.skipTest(f"sglang runtime deps missing: {e}")

        self.assertTrue(tree_verify_method_available("greedy", "npu"))
        self.assertTrue(tree_verify_method_available("target_only", "npu"))
        self.assertTrue(tree_verify_method_available("target_only", "cuda"))
        self.assertFalse(tree_verify_method_available("target_only", "hip"))
        self.assertTrue(tree_verify_method_available("rpd", "npu"))
        self.assertEqual(device_backend_key("npu:0"), "npu")
        self.assertEqual(device_backend_key(torch.device("cpu")), "cpu")

    def test_page_aligned_rollback(self):
        try:
            from sglang.srt.speculative.standalone_remote.sr_kv_rollbacker import (
                SRKVRollbacker,
            )
        except Exception as e:
            self.skipTest(f"sglang runtime deps missing: {e}")

        pool = MagicMock()
        req_to_token = torch.arange(16, dtype=torch.int32).view(1, 16)
        pool.req_to_token = req_to_token
        allocator = MagicMock()
        rb = SRKVRollbacker(
            token_to_kv_pool_allocator=allocator,
            req_to_token_pool=pool,
            tree_cache=MagicMock(),
            page_size=4,
            tp_rank=0,
        )
        req = SimpleNamespace(
            prefix_indices=[],
            req_pool_idx=0,
            kv_allocated_len=12,
            kv_committed_len=12,
            rid="r0",
        )
        self.assertFalse(rb.can_local_rollback(req, fork_point=5))
        self.assertTrue(rb.can_local_rollback(req, fork_point=8))
        ok = rb.local_rollback(req, fork_point=8, current_kv_len=12)
        self.assertTrue(ok)
        allocator.free.assert_called_once()
        freed = allocator.free.call_args[0][0]
        self.assertEqual(freed.tolist(), [8, 9, 10, 11])
        self.assertEqual(req.kv_committed_len, 8)

    def test_device_context_error_classifier(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                _sr_is_device_context_error,
            )
        except Exception as e:
            self.skipTest(f"sglang runtime deps missing: {e}")

        class NPUError(Exception):
            pass

        self.assertTrue(_sr_is_device_context_error(NPUError("illegal memory access")))
        self.assertTrue(_sr_is_device_context_error(RuntimeError("NPU error: illegal")))
        self.assertFalse(_sr_is_device_context_error(RuntimeError("rpc timeout")))

    def test_npu_graph_runner_replay_uses_make_graph_key(self):
        src = (
            _REPO
            / "python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py"
        ).read_text()
        self.assertIn("self._make_graph_key", src)
        self.assertIn("TreeReplayPlan", src)
        self.assertIn("NpuGraphReplaySubmittedError", src)
        self.assertIn("actual_ntpb", src)
        self.assertNotIn("self.graphs[self.bs].replay()", src)
        self.assertNotIn("NPU graph miss", src)

    def test_page_physical_kv_copy_matches_slot_to_page_offset(self):
        from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
            copy_paged_kv_buffer_by_slot,
        )

        def _gold_6d(buf, src, tgt, page_size):
            src_page = torch.div(src, page_size, rounding_mode="floor")
            src_off = src % page_size
            tgt_page = torch.div(tgt, page_size, rounding_mode="floor")
            tgt_off = tgt % page_size
            out = buf.clone()
            out[:, :, tgt_page, tgt_off, :, :] = buf[:, :, src_page, src_off, :, :]
            return out

        page_size = 4
        buf = torch.arange(2 * 1 * 3 * 4 * 1 * 2, dtype=torch.float32).reshape(
            2, 1, 3, 4, 1, 2
        )
        disjoint_src = torch.tensor([5, 6], dtype=torch.int32)
        disjoint_tgt = torch.tensor([9, 10], dtype=torch.int32)
        got = buf.clone()
        copy_paged_kv_buffer_by_slot(got, disjoint_src, disjoint_tgt)
        torch.testing.assert_close(
            got, _gold_6d(buf, disjoint_src, disjoint_tgt, page_size)
        )

        overlap_src = torch.tensor([5, 6], dtype=torch.int32)
        overlap_tgt = torch.tensor([4, 5], dtype=torch.int32)
        got_overlap = buf.clone()
        copy_paged_kv_buffer_by_slot(got_overlap, overlap_src, overlap_tgt)
        torch.testing.assert_close(
            got_overlap, _gold_6d(buf, overlap_src, overlap_tgt, page_size)
        )

        layout_src = (
            _REPO
            / "python/sglang/srt/speculative/standalone_remote/sr_verify_layout.py"
        ).read_text()
        self.assertIn("def copy_paged_kv_buffer_by_slot", layout_src)
        self.assertIn("kv_buffer.view(kv2, layer, -1, head, dim)", layout_src)
        self.assertIn("flat.index_select(2, src)", layout_src)
        self.assertIn("flat.index_copy_(2, tgt, staged)", layout_src)
        self.assertNotIn("src_loc.reshape(-1).tolist()", layout_src)
        self.assertNotIn("for i, s in enumerate", layout_src)
        self.assertNotIn("[:, :, tgt_page, tgt_off", layout_src)

        npu_src = (
            _REPO / "python/sglang/srt/hardware_backend/npu/memory_pool_npu.py"
        ).read_text()
        self.assertIn("def move_kv_cache", npu_src)
        self.assertIn(
            "copy_paged_kv_buffer_by_slot(self.kv_buffer, src_loc, tgt_loc)",
            npu_src,
        )
        self.assertNotIn("def copy_paged_kv_buffer_by_slot", npu_src)
        self.assertNotIn("copy_all_layer_kv_cache_tiled", npu_src)
        self.assertIn("enable_kv_cache_copy=False", npu_src)
        self.assertNotIn("enable_kv_cache_copy=enable_kv_cache_copy", npu_src)
        self.assertIn("def _init_kv_copy_and_warmup", npu_src)

    def test_npu_tree_draft_triton_and_kv_restore_source_guards(self):
        spec_src = (
            _REPO / "python/sglang/srt/speculative/spec_utils.py"
        ).read_text()
        self.assertNotIn(
            "if ((page_size != 1) and (topk != 1)) and (duplicate_cache_len > 0):",
            spec_src,
        )
        self.assertIn("def build_paged_draft_cache_locs", spec_src)
        self.assertIn("def build_tree_draft_kv_slots", spec_src)
        self.assertNotIn("duplicate_cache_len: tl.constexpr", spec_src)
        drafter_src = (
            _REPO
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_tree_drafter.py"
        ).read_text()
        self.assertIn("def _alloc_tree_kv", drafter_src)
        self.assertIn("if token_to_kv_pool_state_backup is not None:", drafter_src)
        self.assertIn("except Exception:", drafter_src)
        self.assertNotIn(
            "if self.page_size > 1 and self.topk > 1 and duplicate_cache_len > 0:",
            drafter_src,
        )
        self.assertIn("if self.page_size > 1 and self.topk > 1:", drafter_src)
        self.assertIn("build_paged_draft_cache_locs", drafter_src)
        self.assertNotIn("if duplicate_cache_len > 0:", drafter_src)

    def test_normalize_tree_draft_kv_lens(self):
        normalize_tree_draft_kv_lens = (
            _load_tree_draft_helpers().normalize_tree_draft_kv_lens
        )

        self.assertEqual(normalize_tree_draft_kv_lens([7], 3, topk=3), [7, 7, 7])
        self.assertEqual(
            normalize_tree_draft_kv_lens([10, 20], 6, topk=3),
            [10, 10, 10, 20, 20, 20],
        )
        self.assertEqual(
            normalize_tree_draft_kv_lens([7, 7, 7], 3, topk=3),
            [7, 7, 7],
        )
        self.assertEqual(normalize_tree_draft_kv_lens([7], 1, topk=1), [7])
        with self.assertRaises(ValueError):
            normalize_tree_draft_kv_lens([7, 8], 3, topk=3)
        with self.assertRaises(ValueError):
            normalize_tree_draft_kv_lens(None, 3, topk=3)

    def test_expand_seq_lens_for_spec_topk(self):
        expand_seq_lens_for_spec_topk = (
            _load_tree_draft_helpers().expand_seq_lens_for_spec_topk
        )

        self.assertEqual(
            expand_seq_lens_for_spec_topk([10, 20], 6),
            [10, 10, 10, 20, 20, 20],
        )
        self.assertEqual(expand_seq_lens_for_spec_topk([10, 20], 2), [10, 20])
        self.assertEqual(expand_seq_lens_for_spec_topk([7], 4), [7])

    def test_build_draft_graph_step_kv_lens(self):
        helpers = _load_tree_draft_helpers()
        build = helpers.build_draft_graph_step_kv_lens

        self.assertEqual(
            build([128], capture_bs=1, topk=3, step_id=0),
            [129, 129, 129],
        )
        padded = build([128], capture_bs=4, topk=3, step_id=0)
        self.assertEqual(len(padded), 12)
        self.assertEqual(padded[:3], [129, 129, 129])
        self.assertEqual(padded[3:], [0] * 9)
        self.assertEqual(
            build([10, 20], capture_bs=2, topk=3, step_id=1),
            [12, 12, 12, 22, 22, 22],
        )
        self.assertEqual(build([128], capture_bs=1, topk=1, step_id=0), [129])
        with self.assertRaises(ValueError):
            build(None, capture_bs=1, topk=3, step_id=0)
        with self.assertRaises(ValueError):
            build([128, 64], capture_bs=1, topk=3, step_id=0)

    def test_expand_fia_cpu_update_inputs_is_step_major(self):
        helpers = _load_tree_draft_helpers()
        expand = helpers.expand_fia_cpu_update_inputs
        s0 = [129, 129, 129]
        s1 = [130, 130, 130]
        got = expand([s0, s1], num_layers=3, attr_name="actual_seq_lengths_kv")
        keys = [d["actual_seq_lengths_kv"] for d in got]
        self.assertEqual(keys, [s0, s0, s0, s1, s1, s1])
        self.assertNotEqual(keys, [s0, s1, s0, s1, s0, s1])
        # list * num_layers would produce the interleaved / repeated-block wrong order
        self.assertNotEqual(keys, ([s0, s1] * 3))

    def test_resolve_fia_update_count(self):
        helpers = _load_tree_draft_helpers()
        resolve = helpers.resolve_fia_update_count
        Prep = helpers.NpuGraphPreparationError
        self.assertEqual(resolve(112, 4, 28), 112)
        with self.assertRaises(Prep):
            resolve(None, 4, 28)
        with self.assertRaises(Prep):
            resolve(8, 4, 28)
        with self.assertRaises(Prep):
            resolve(224, 4, 28)

    def _fia_rec(self, op="npu_fused_infer_attention_score"):
        return SimpleNamespace(
            op_name=op,
            kwargs={"actual_seq_lengths_kv": [1]},
        )

    def test_validate_tree_draft_fia_records_all_fia(self):
        helpers = _load_tree_draft_helpers()
        validate = helpers.validate_tree_draft_fia_records
        n_steps, num_layers = 4, 3
        records = [self._fia_rec() for _ in range(n_steps * num_layers)]
        n_records, step_ids = validate(
            records, n_steps, num_layers, "actual_seq_lengths_kv"
        )
        self.assertEqual(n_records, n_steps * num_layers)
        self.assertEqual(step_ids, [i // num_layers for i in range(n_records)])
        expanded = helpers.expand_fia_cpu_update_inputs(
            [[129] * 3, [130] * 3, [131] * 3, [132] * 3],
            num_layers,
            "actual_seq_lengths_kv",
        )
        self.assertEqual(len(expanded), n_steps * num_layers)
        keys = [d["actual_seq_lengths_kv"] for d in expanded]
        self.assertEqual(
            keys,
            [[129] * 3] * 3
            + [[130] * 3] * 3
            + [[131] * 3] * 3
            + [[132] * 3] * 3,
        )

    def test_validate_tree_draft_fia_records_rejects_mixed_ops(self):
        helpers = _load_tree_draft_helpers()
        validate = helpers.validate_tree_draft_fia_records
        Prep = helpers.NpuGraphPreparationError
        records = [self._fia_rec() for _ in range(11)]
        records.insert(3, self._fia_rec(op="npu_rms_norm"))
        with self.assertRaises(Prep) as ctx:
            validate(records, 4, 3, "actual_seq_lengths_kv")
        self.assertEqual(ctx.exception.scope, "graph")

    def test_normalize_fia_op_name(self):
        normalize = _load_tree_draft_helpers().normalize_fia_op_name
        self.assertEqual(
            normalize("npu::npu_fused_infer_attention_score.default"),
            "npu_fused_infer_attention_score",
        )
        self.assertEqual(
            normalize("npu::npu_fused_infer_attention_score.out"),
            "npu_fused_infer_attention_score.out",
        )
        self.assertEqual(
            normalize("npu_fused_infer_attention_score"),
            "npu_fused_infer_attention_score",
        )

    def test_validate_tree_draft_fia_records_torchnpu_name(self):
        helpers = _load_tree_draft_helpers()
        validate = helpers.validate_tree_draft_fia_records
        n_steps, num_layers = 2, 2

        class KwargsMap(Mapping):
            def __init__(self, data):
                self._data = data

            def __getitem__(self, key):
                return self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

        entry = SimpleNamespace(
            **{
                "__name__": "npu::npu_fused_infer_attention_score.default",
            }
        )
        rec = SimpleNamespace(
            op_cache_entry=entry,
            kwargs=KwargsMap({"actual_seq_lengths_kv": [1]}),
        )
        n_records, step_ids = validate(
            [rec, rec, rec, rec], n_steps, num_layers, "actual_seq_lengths_kv"
        )
        self.assertEqual(n_records, 4)
        self.assertEqual(step_ids, [0, 0, 1, 1])

        out_entry = SimpleNamespace(
            **{"__name__": "npu::npu_fused_infer_attention_score.out"}
        )
        out_rec = SimpleNamespace(
            op_cache_entry=out_entry,
            kwargs={"actual_seq_lengths_kv": [1]},
        )
        n_records, _ = validate(
            [out_rec] * 4, n_steps, num_layers, "actual_seq_lengths_kv"
        )
        self.assertEqual(n_records, 4)

        schema_entry = SimpleNamespace(
            **{
                "__name__": "npu::npu_fused_infer_attention_score.default",
                "_schema": "actual_seq_lengths_kv: List[int]",
            }
        )
        schema_rec = SimpleNamespace(op_cache_entry=schema_entry)
        n_records, _ = validate(
            [schema_rec] * 4, n_steps, num_layers, "actual_seq_lengths_kv"
        )
        self.assertEqual(n_records, 4)

    def test_validate_tree_draft_fia_records_missing_api(self):
        helpers = _load_tree_draft_helpers()
        validate = helpers.validate_tree_draft_fia_records
        Prep = helpers.NpuGraphPreparationError
        with self.assertRaises(Prep) as ctx:
            validate(None, 4, 28, "actual_seq_lengths_kv")
        self.assertEqual(ctx.exception.scope, "format")
        with self.assertRaises(Prep) as ctx:
            inspect = helpers.inspect_dispatch_record
            inspect(SimpleNamespace(value="actual_seq_lengths_kv"), "actual_seq_lengths_kv")
        self.assertEqual(ctx.exception.scope, "format")

    def test_validate_draft_graph_step_kv_lens_padding(self):
        helpers = _load_tree_draft_helpers()
        build = helpers.build_draft_graph_step_kv_lens
        validate = helpers.validate_draft_graph_step_kv_lens
        Prep = helpers.NpuGraphPreparationError
        seq_lens = build([128], capture_bs=4, topk=3, step_id=0)
        got = validate(
            seq_lens,
            capture_bs=4,
            topk=3,
            raw_bs=1,
            prefix_lens=[128],
            step_id=0,
        )
        self.assertEqual(len(got), 12)
        self.assertEqual(got[:3], [129, 129, 129])
        self.assertEqual(got[3:], [0] * 9)
        with self.assertRaises(Prep):
            validate(
                [129, 129, 129] + [1] * 9,
                capture_bs=4,
                topk=3,
                raw_bs=1,
                prefix_lens=[128],
                step_id=0,
            )

    def test_expand_batch_reraises_graph_submitted(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_tree_drafter import (
                SRTreeDrafter,
            )
        except Exception as e:
            self.skipTest(f"sglang runtime deps missing: {e}")
        Submitted = _load_tree_draft_helpers().NpuGraphReplaySubmittedError
        drafter = object.__new__(SRTreeDrafter)
        called = {"expand_one": 0}

        def boom(*_args, **_kwargs):
            raise Submitted("already submitted")

        def expand_one(_req):
            called["expand_one"] += 1
            return ([], None, None)

        drafter._expand_tree = boom
        drafter._expand_one = expand_one
        drafter._stack_seeds = lambda _reqs: (None, None, None, None)
        req = SimpleNamespace(req_pool_idx=0, sr_tree_seed=object(), rid="r0")
        with self.assertRaises(Submitted):
            SRTreeDrafter.expand_batch(drafter, [req])
        self.assertEqual(called["expand_one"], 0)

    def test_expand_batch_reraises_device_context_error(self):
        methods = _load_sr_tree_expand_methods()

        class NPUError(Exception):
            pass

        drafter = SimpleNamespace()
        called = {"expand_one": 0}

        def boom(*_args, **_kwargs):
            raise NPUError("illegal memory access")

        def expand_one(_req):
            called["expand_one"] += 1
            return ([], None, None)

        drafter._expand_tree = boom
        drafter._expand_one = expand_one
        drafter._stack_seeds = lambda _reqs: (None, None, None, None)
        req = SimpleNamespace(req_pool_idx=0, sr_tree_seed=object(), rid="r0")
        with self.assertRaises(NPUError):
            methods.expand_batch(drafter, [req])
        self.assertEqual(called["expand_one"], 0)

        def boom_msg(*_args, **_kwargs):
            raise RuntimeError("NPU error: illegal memory access")

        drafter._expand_tree = boom_msg
        with self.assertRaises(RuntimeError):
            methods.expand_batch(drafter, [req])
        self.assertEqual(called["expand_one"], 0)

        def boom_one(*_args, **_kwargs):
            raise NPUError("illegal memory access")

        drafter._expand_tree = boom_one
        with self.assertRaises(NPUError):
            methods.expand_one(drafter, req)

    def test_scheduler_reraises_graph_submitted(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                StandaloneRemoteDraftSchedulerMixin,
            )
        except Exception as e:
            self.skipTest(f"sglang runtime deps missing: {e}")
        Submitted = _load_tree_draft_helpers().NpuGraphReplaySubmittedError

        class FakeDrafter:
            def expand_batch(self, _reqs):
                raise Submitted("already submitted")

        mixin = object.__new__(StandaloneRemoteDraftSchedulerMixin)
        mixin.sr_tree_drafter = FakeDrafter()
        mixin._sr_materialize_prefix_batch = lambda _reqs: None
        mixin._sr_run_tree_ingest = lambda _reqs: True
        mixin._sr_ensure_tree_seeds = lambda _reqs: None
        mixin._sr_is_degraded = lambda _rid: False
        mixin._sr_replay_grammars = lambda _reqs: None
        mixin._sr_resume_req = lambda _req: None
        mixin._sr_park_in_running_many = lambda _reqs: None
        mixin._sr_pause_req = lambda _req: None
        mixin._sr_is_finished = lambda _req: False
        mixin._sr_kv_len = lambda req: len(req.origin_input_ids) + len(
            req.output_ids or []
        )
        mixin._sr_mark_degraded = lambda *_a, **_k: None
        mixin.last_batch = object()
        req = SimpleNamespace(
            req_pool_idx=0,
            sr_tree_seed=object(),
            rid="r0",
            origin_input_ids=[1],
            output_ids=[],
            finished_reason=None,
            kv_committed_len=1,
        )
        with self.assertRaises(Submitted):
            mixin._sr_tree_expand_batch([req])

    def test_build_tree_draft_block_tables_matrix(self):
        build_tree_draft_block_tables = (
            _load_tree_draft_helpers().build_tree_draft_block_tables
        )

        cases = []
        for page_size in (1, 4, 128):
            for topk in (1, 3):
                for step_id in (0, 1, 4):
                    num_steps = 5
                    prefixes = {
                        1: (1, 2, 3),
                        4: (3, 4, 5, 7, 8, 9),
                        128: (127, 128, 129),
                    }[page_size]
                    for seq in prefixes:
                        cases.append((page_size, topk, step_id, num_steps, [seq]))
                    if page_size != 1:
                        cases.append(
                            (page_size, topk, step_id, num_steps, [prefixes[0], prefixes[-1]])
                        )

        for page_size, topk, step_id, num_steps, seqs in cases:
            with self.subTest(
                page_size=page_size, topk=topk, step_id=step_id, seqs=seqs
            ):
                last_page_len = [s % page_size for s in seqs]
                num_new = [
                    (lp + num_steps + page_size - 1) // page_size for lp in last_page_len
                ]
                n_logical = []
                for s, nnp in zip(seqs, num_new):
                    if topk == 1:
                        n_logical.append(s + step_id + 1 + page_size)
                    elif page_size == 1:
                        n_logical.append(s + topk * num_steps + 8)
                    else:
                        n_logical.append(
                            (s - s % page_size) + topk * nnp * page_size + page_size
                        )
                n_pages = [(n + page_size - 1) // page_size for n in n_logical]
                page_ids = [
                    [1000 + b * 50 + p * 7 for p in range(np)]
                    for b, np in enumerate(n_pages)
                ]
                req = _noncontiguous_req_to_token(page_ids, page_size)
                pool = torch.arange(len(seqs), dtype=torch.int64)
                seq_t = torch.tensor(seqs, dtype=torch.int64)
                got = build_tree_draft_block_tables(
                    req,
                    pool,
                    seq_t,
                    page_size=page_size,
                    topk=topk,
                    step_id=step_id,
                    num_steps=num_steps,
                )
                ref = _reference_tree_draft_block_tables(
                    req,
                    pool,
                    seq_t,
                    page_size=page_size,
                    topk=topk,
                    step_id=step_id,
                    num_steps=num_steps,
                )
                self.assertEqual(tuple(got.shape), tuple(ref.shape))
                self.assertTrue(torch.equal(got, ref), msg=f"got={got} ref={ref}")
                if topk > 1 and page_size > 1:
                    kv_len0 = seqs[0] + step_id + 1
                    n_pages0 = (kv_len0 + page_size - 1) // page_size
                    last = n_pages0 - 1
                    self.assertGreaterEqual(last, 0)
                    row0 = got[0]
                    row1 = got[1]
                    last_len = seqs[0] % page_size
                    n_shared = (seqs[0] - last_len) // page_size
                    if n_shared > 0:
                        self.assertEqual(int(row0[0]), int(row1[0]))
                    self.assertNotEqual(int(row0[last]), int(row1[last]))

    def test_build_tree_draft_block_tables_overflow_raises(self):
        build_tree_draft_block_tables = (
            _load_tree_draft_helpers().build_tree_draft_block_tables
        )

        page_size = 4
        req = torch.arange(8, dtype=torch.int32).view(1, 8)
        with self.assertRaises(RuntimeError):
            build_tree_draft_block_tables(
                req,
                torch.tensor([0]),
                torch.tensor([8]),
                page_size=page_size,
                topk=2,
                step_id=0,
                num_steps=5,
            )

    def test_build_tree_draft_block_tables_aligned(self):
        build_tree_draft_block_tables = (
            _load_tree_draft_helpers().build_tree_draft_block_tables
        )

        page_size = 4
        req_to_token = torch.arange(32, dtype=torch.int32).view(1, 32)
        tables = build_tree_draft_block_tables(
            req_to_token,
            torch.tensor([0]),
            torch.tensor([8]),
            page_size=page_size,
            topk=2,
            step_id=0,
            num_steps=5,
        )
        self.assertEqual(tables.shape[0], 2)
        self.assertTrue(torch.equal(tables[0, :-1], tables[1, :-1]))
        self.assertEqual(int(tables[0, -1]), 8 // page_size)
        self.assertEqual(int(tables[1, -1]), 16 // page_size)

    def test_build_tree_draft_block_tables_unaligned(self):
        build_tree_draft_block_tables = (
            _load_tree_draft_helpers().build_tree_draft_block_tables
        )

        page_size = 4
        req_to_token = torch.arange(32, dtype=torch.int32).view(1, 32)
        tables = build_tree_draft_block_tables(
            req_to_token,
            torch.tensor([0]),
            torch.tensor([5]),
            page_size=page_size,
            topk=2,
            step_id=0,
            num_steps=5,
        )
        self.assertEqual(tables.shape[0], 2)
        self.assertEqual(int(tables[0, 0]), int(tables[1, 0]))
        self.assertEqual(int(tables[0, 1]), 4 // page_size)
        self.assertEqual(int(tables[1, 1]), 12 // page_size)

    def test_build_tree_draft_block_tables_topk1(self):
        build_tree_draft_block_tables = (
            _load_tree_draft_helpers().build_tree_draft_block_tables
        )

        page_size = 4
        seq_len = 8
        step_id = 0
        kv_len = seq_len + step_id + 1
        req_to_token = torch.arange(32, dtype=torch.int32).view(1, 32)
        naive = req_to_token[:, :kv_len:page_size] // page_size
        tables = build_tree_draft_block_tables(
            req_to_token,
            torch.tensor([0]),
            torch.tensor([seq_len]),
            page_size=page_size,
            topk=1,
            step_id=step_id,
            num_steps=5,
        )
        self.assertEqual(tuple(tables.shape), (1, int(naive.shape[1])))
        self.assertTrue(torch.equal(tables, naive.to(torch.int32)))

    def test_npu_tree_draft_block_tables_source_guards(self):
        src = (
            _REPO
            / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
        ).read_text()
        self.assertIn("build_tree_draft_block_tables", src)
        self.assertIn("normalize_tree_draft_kv_lens", src)
        self.assertIn("max(int(max_bs), int(max_num_tokens))", src)
        self.assertIn("_tree_draft_table_rows(bs, num_tokens)", src)
        self.assertIn("draft_topk=topk", src)
        self.assertIn("draft_num_steps=speculative_num_steps", src)
        self.assertIn("metadata.block_tables.fill_(0)", src)
        self.assertIn("tables.shape[0] > dest.shape[0]", src)
        self.assertNotIn("n_rows = min(tables.shape[0], dest.shape[0])", src)
        self.assertIn("is_tree_draft = self._is_tree_draft(forward_batch)", src)
        graph_src = (
            _REPO
            / "python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_npu_graph_runner.py"
        ).read_text()
        self.assertIn("build_draft_graph_step_kv_lens", graph_src)
        self.assertIn("expand_fia_cpu_update_inputs", graph_src)
        self.assertIn("validate_tree_draft_fia_records", graph_src)
        self.assertIn("validate_draft_graph_step_kv_lens", graph_src)
        self.assertIn("graph_dispatch_records", graph_src)
        self.assertIn("NpuGraphReplaySubmittedError", graph_src)
        self.assertIn("NpuGraphPreparationError", graph_src)
        self.assertIn("seq_lens_cpu[: self.raw_bs]", graph_src)
        self.assertIn("_tree_fia_maps", graph_src)
        self.assertIn("tree_graph_disabled_reason", graph_src)
        self.assertIn("tree_graph_replay_count", graph_src)
        self.assertNotIn("normalize_tree_draft_kv_lens", graph_src)
        spec_src = (_REPO / "python/sglang/srt/speculative/spec_utils.py").read_text()
        self.assertIn("normalize_fia_op_name", spec_src)
        self.assertIn('getattr(obj, "__name__", None)', spec_src)
        self.assertIn("scope=\"format\"", spec_src)
        self.assertNotIn("count_fia_kv_len_records", graph_src)
        self.assertNotIn("if attr_name in str(rec)", graph_src)
        drafter_src = (
            _REPO
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_tree_drafter.py"
        ).read_text()
        self.assertIn("NpuGraphReplaySubmittedError", drafter_src)
        self.assertIn("NpuGraphPreparationError", drafter_src)
        self.assertIn("except NpuGraphReplaySubmittedError:\n            raise", drafter_src)
        self.assertNotIn("skipping per-req retry", drafter_src)
        self.assertIn("falling back to eager", drafter_src)
        self.assertIn("graph_submitted", drafter_src)
        self.assertIn("except NpuGraphPreparationError as e:", drafter_src)
        self.assertIn('getattr(e, "scope", "graph") == "format"', drafter_src)
        self.assertIn("tree_eager_fallback_count", drafter_src)
        self.assertIn("tree draft graphs disabled after capture", drafter_src)
        mixin_src = (
            _REPO
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_draft_scheduler_mixin.py"
        ).read_text()
        self.assertIn("except NpuGraphReplaySubmittedError:\n            raise", mixin_src)

    def test_npu_tree_draft_fia_alignment_source_guards(self):
        src = (
            _REPO
            / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
        ).read_text()
        self.assertIn("query = q.reshape(", src)
        self.assertIn("-1, 1, layer.tp_q_head_num, layer.qk_head_dim", src)
        self.assertIn("context_lens=context_lens", src)

    def test_advance_tree_draft_positions_for_step(self):
        from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
            advance_tree_draft_positions_for_step,
        )

        positions = torch.tensor([4, 5, 6], dtype=torch.int64)
        mrope = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.int64)
        advance_tree_draft_positions_for_step(0, positions, mrope)
        self.assertEqual(positions.tolist(), [4, 5, 6])
        self.assertEqual(mrope.tolist(), [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        advance_tree_draft_positions_for_step(1, positions, mrope)
        self.assertEqual(positions.tolist(), [5, 6, 7])
        self.assertEqual(mrope.tolist(), [[2, 3, 4], [5, 6, 7], [8, 9, 10]])

        advance_tree_draft_positions_for_step(-1, positions, mrope)
        self.assertEqual(positions.tolist(), [5, 6, 7])

    def test_tree_reselect_parent_rows_and_kv_remap(self):
        from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
            copy_paged_kv_buffer_by_slot,
        )

        tree_reselect_parent_rows = (
            _load_tree_draft_helpers().tree_reselect_parent_rows
        )
        topk = 2
        topk_cs_index = torch.tensor([[3, 1]], dtype=torch.int64)
        parent_rows = tree_reselect_parent_rows(topk_cs_index, num_hidden_rows=2, topk=topk)
        self.assertEqual(parent_rows.tolist(), [1, 0])

        page_size = 4
        buf = torch.arange(2 * 1 * 3 * 4 * 1 * 2, dtype=torch.float32).reshape(
            2, 1, 3, 4, 1, 2
        )
        out_cache_loc = torch.tensor([5, 6], dtype=torch.int64)
        src = out_cache_loc[parent_rows]
        tgt = out_cache_loc
        gold = buf.clone()
        src_page = torch.div(src, page_size, rounding_mode="floor")
        src_off = src % page_size
        tgt_page = torch.div(tgt, page_size, rounding_mode="floor")
        tgt_off = tgt % page_size
        gold[:, :, tgt_page, tgt_off, :, :] = buf[:, :, src_page, src_off, :, :]
        got = buf.clone()
        copy_paged_kv_buffer_by_slot(got, src, tgt)
        torch.testing.assert_close(got, gold)

    def test_copy_mha_kv_by_slot_matches_index_gold(self):
        from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
            copy_mha_kv_by_slot,
        )

        parent_rows = torch.tensor([1, 0], dtype=torch.int64)
        out_cache_loc = torch.tensor([5, 6], dtype=torch.int64)
        src = out_cache_loc[parent_rows]
        tgt = out_cache_loc

        k_buf = [
            torch.arange(12 * 1 * 2, dtype=torch.float32).reshape(12, 1, 2),
            torch.arange(12 * 1 * 2, dtype=torch.float32).reshape(12, 1, 2) + 100,
        ]
        v_buf = [b.clone() + 50 for b in k_buf]
        k_gold = []
        v_gold = []
        for kb, vb in zip(k_buf, v_buf):
            kg = kb.clone()
            vg = vb.clone()
            kg.index_copy_(0, tgt, kb.index_select(0, src))
            vg.index_copy_(0, tgt, vb.index_select(0, src))
            k_gold.append(kg)
            v_gold.append(vg)
        k_got = [b.clone() for b in k_buf]
        v_got = [b.clone() for b in v_buf]
        copy_mha_kv_by_slot(k_got, v_got, src, tgt)
        for got, gold in zip(k_got, k_gold):
            torch.testing.assert_close(got, gold)
        for got, gold in zip(v_got, v_gold):
            torch.testing.assert_close(got, gold)

        k5 = torch.arange(1 * 3 * 4 * 1 * 2, dtype=torch.float32).reshape(1, 3, 4, 1, 2)
        v5 = k5.clone() + 7
        k5_got = k5.clone()
        v5_got = v5.clone()
        copy_mha_kv_by_slot(k5_got, v5_got, src, tgt)
        k5_gold = k5.clone()
        v5_gold = v5.clone()
        k_flat = k5_gold.view(1, -1, 1, 2)
        v_flat = v5_gold.view(1, -1, 1, 2)
        k_flat.index_copy_(1, tgt, k5.view(1, -1, 1, 2).index_select(1, src))
        v_flat.index_copy_(1, tgt, v5.view(1, -1, 1, 2).index_select(1, src))
        torch.testing.assert_close(k5_got, k5_gold)
        torch.testing.assert_close(v5_got, v5_gold)

    def test_tree_draft_forward_rope_and_kv_remap_source_guards(self):
        drafter_src = (
            _REPO
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_tree_drafter.py"
        ).read_text()
        self.assertIn("advance_tree_draft_positions_for_step", drafter_src)
        self.assertIn("copy_kv_pool_by_slot", drafter_src)
        self.assertIn("seq_lens_sum_from_batch", drafter_src)
        self.assertIn("seq_lens_cpu_for_host", drafter_src)
        alloc_src = drafter_src[
            drafter_src.index("def _alloc_tree_kv") : drafter_src.index(
                "def _try_alloc_lease_tree_kv"
            )
        ]
        self.assertIn("paged_tree_mapping_fits(\n            seq_lens_cpu_for_host(batch)", alloc_src)
        self.assertNotIn("paged_tree_mapping_fits(\n            batch.seq_lens", alloc_src)
        eagle_src = (
            _REPO / "python/sglang/srt/speculative/eagle_worker.py"
        ).read_text()
        self.assertIn(
            "paged_tree_mapping_fits(\n            batch.seq_lens,",
            eagle_src,
        )
        self.assertIn("def _remap_tree_kv_to_parents", drafter_src)
        self.assertIn("parent_rows", drafter_src)
        self.assertNotIn("for s in range(n_prev_steps)", drafter_src)
        self.assertNotIn("move_kv_cache", drafter_src)
        self.assertNotIn("torch.sum(batch.seq_lens)", drafter_src)
        self.assertNotIn("advance_tree_draft_positions(", drafter_src)
        self.assertIn("if is_device_context_error(e):", drafter_src)
        self.assertNotIn("if kv_buffer is None:", drafter_src)

        spec_src = (_REPO / "python/sglang/srt/speculative/spec_utils.py").read_text()
        self.assertIn("def tree_reselect_parent_rows", spec_src)
        self.assertIn(
            "return input_ids, hidden_states, scores, tree_info, parent_rows",
            spec_src,
        )

    def test_eagle_verify_refuses_silent_greedy_for_remote_spec(self):
        eagle_src = (
            _REPO / "python/sglang/srt/speculative/eagle_info.py"
        ).read_text()
        self.assertIn("Refusing to fall back to greedy", eagle_src)
        self.assertIn("is_remote_spec_algorithm", eagle_src)

    def test_npu_does_not_call_chain_sgl_kernel_npu_greedy(self):
        src = (_REPO / "python/sglang/srt/speculative/eagle_utils.py").read_text()
        self.assertIn("verify_tree_greedy_ref", src)
        self.assertNotIn("sgl_kernel_npu", src)

    def test_copy_kv_pool_by_slot_validates_before_write(self):
        from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
            UnsupportedTreeKVLayout,
            copy_kv_pool_by_slot,
        )

        src = torch.tensor([1, 2], dtype=torch.int64)
        tgt = torch.tensor([3, 4], dtype=torch.int64)
        with self.assertRaises(UnsupportedTreeKVLayout):
            copy_kv_pool_by_slot(SimpleNamespace(), src, tgt)
        k = torch.arange(8, dtype=torch.float32).reshape(8, 1)
        pool = SimpleNamespace(k_buffer=k.clone(), v_buffer=None)
        with self.assertRaises(UnsupportedTreeKVLayout):
            copy_kv_pool_by_slot(pool, src, tgt)
        torch.testing.assert_close(pool.k_buffer, k)
        with self.assertRaisesRegex(RuntimeError, "length mismatch"):
            copy_kv_pool_by_slot(
                SimpleNamespace(k_buffer=k.clone(), v_buffer=k.clone()),
                torch.tensor([1]),
                tgt,
            )
        copy_kv_pool_by_slot(SimpleNamespace(), torch.tensor([]), torch.tensor([]))
        with self.assertRaisesRegex(RuntimeError, "length mismatch"):
            copy_kv_pool_by_slot(
                SimpleNamespace(k_buffer=k.clone(), v_buffer=k.clone()),
                torch.tensor([]),
                tgt,
            )

    def test_copy_kv_pool_by_slot_rejects_incomplete_index_before_write(self):
        from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
            UnsupportedTreeKVLayout,
            copy_kv_pool_by_slot,
        )

        src = torch.tensor([1, 0], dtype=torch.int64)
        tgt = torch.tensor([0, 1], dtype=torch.int64)
        k5 = torch.arange(1 * 2 * 2 * 1 * 2, dtype=torch.float32).reshape(1, 2, 2, 1, 2)
        v5 = k5.clone() + 3
        pool = SimpleNamespace(
            kv_buffer=None,
            k_buffer=k5.clone(),
            v_buffer=v5.clone(),
            index_k_buffer=torch.arange(4, dtype=torch.float32),
        )
        with self.assertRaises(UnsupportedTreeKVLayout):
            copy_kv_pool_by_slot(pool, src, tgt)
        torch.testing.assert_close(pool.k_buffer, k5)
        torch.testing.assert_close(pool.v_buffer, v5)

    def test_copy_kv_pool_by_slot_copies_list_and_mla5(self):
        from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
            copy_kv_pool_by_slot,
        )

        src = torch.tensor([1, 0], dtype=torch.int64)
        tgt = torch.tensor([0, 1], dtype=torch.int64)
        k_list = [
            torch.arange(6, dtype=torch.float32).reshape(6, 1),
            torch.arange(6, dtype=torch.float32).reshape(6, 1) + 10,
        ]
        v_list = [b.clone() + 1 for b in k_list]
        pool = SimpleNamespace(
            kv_buffer=None,
            k_buffer=[b.clone() for b in k_list],
            v_buffer=[b.clone() for b in v_list],
            index_k_buffer=None,
        )
        copy_kv_pool_by_slot(pool, src, tgt)
        torch.testing.assert_close(pool.k_buffer[0][0], k_list[0][1])
        torch.testing.assert_close(pool.k_buffer[0][1], k_list[0][0])
        k5 = torch.arange(1 * 2 * 2 * 1 * 2, dtype=torch.float32).reshape(1, 2, 2, 1, 2)
        v5 = k5.clone() + 3
        idx = k5.clone() + 9
        pool5 = SimpleNamespace(
            kv_buffer=None,
            k_buffer=k5.clone(),
            v_buffer=v5.clone(),
            index_k_buffer=idx.clone(),
        )
        copy_kv_pool_by_slot(pool5, src, tgt)
        flat_k = k5.view(1, -1, 1, 2)
        gold = flat_k.index_select(1, src)
        got = pool5.k_buffer.view(1, -1, 1, 2)
        torch.testing.assert_close(got.index_select(1, tgt), gold)
        torch.testing.assert_close(
            pool5.index_k_buffer.view(1, -1, 1, 2).index_select(1, tgt),
            idx.view(1, -1, 1, 2).index_select(1, src),
        )

        kv_list = [b.clone() for b in k_list]
        pool_kv = SimpleNamespace(kv_buffer=kv_list, k_buffer=None, v_buffer=None)
        copy_kv_pool_by_slot(pool_kv, src, tgt)
        torch.testing.assert_close(pool_kv.kv_buffer[0][0], k_list[0][1])

        paged = torch.arange(2 * 1 * 2 * 2 * 1 * 2, dtype=torch.float32).reshape(
            2, 1, 2, 2, 1, 2
        )
        pool6 = SimpleNamespace(kv_buffer=paged.clone(), k_buffer=None, v_buffer=None)
        copy_kv_pool_by_slot(pool6, src, tgt)
        flat = paged.view(2, 1, -1, 1, 2)
        torch.testing.assert_close(
            pool6.kv_buffer.view(2, 1, -1, 1, 2).index_select(2, tgt),
            flat.index_select(2, src),
        )


if __name__ == "__main__":
    unittest.main()
