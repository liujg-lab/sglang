"""Device-agnostic helpers for SPECTRE / STANDALONE_REMOTE dual-backend."""

import ast
import unittest
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
    }
    class_names = {"NpuGraphReplaySubmittedError"}
    try:
        from sglang.srt.speculative.spec_utils import (
            NpuGraphReplaySubmittedError,
            build_draft_graph_step_kv_lens,
            build_tree_draft_block_tables,
            expand_fia_cpu_update_inputs,
            expand_seq_lens_for_spec_topk,
            normalize_tree_draft_kv_lens,
            resolve_fia_update_count,
        )

        return SimpleNamespace(
            build_tree_draft_block_tables=build_tree_draft_block_tables,
            expand_seq_lens_for_spec_topk=expand_seq_lens_for_spec_topk,
            normalize_tree_draft_kv_lens=normalize_tree_draft_kv_lens,
            build_draft_graph_step_kv_lens=build_draft_graph_step_kv_lens,
            expand_fia_cpu_update_inputs=expand_fia_cpu_update_inputs,
            resolve_fia_update_count=resolve_fia_update_count,
            NpuGraphReplaySubmittedError=NpuGraphReplaySubmittedError,
        )
    except Exception:
        pass
    src_path = _REPO / "python/sglang/srt/speculative/spec_utils.py"
    tree = ast.parse(src_path.read_text())
    keep = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in helper_names)
        or (isinstance(node, ast.ClassDef) and node.name in class_names)
    ]
    mod = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {"torch": torch, "Optional": Optional}
    exec(compile(mod, str(src_path), "exec"), ns)
    return SimpleNamespace(
        build_tree_draft_block_tables=ns["build_tree_draft_block_tables"],
        expand_seq_lens_for_spec_topk=ns["expand_seq_lens_for_spec_topk"],
        normalize_tree_draft_kv_lens=ns["normalize_tree_draft_kv_lens"],
        build_draft_graph_step_kv_lens=ns["build_draft_graph_step_kv_lens"],
        expand_fia_cpu_update_inputs=ns["expand_fia_cpu_update_inputs"],
        resolve_fia_update_count=ns["resolve_fia_update_count"],
        NpuGraphReplaySubmittedError=ns["NpuGraphReplaySubmittedError"],
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
        self.assertIn("NPU graph miss", src)
        self.assertIn("actual_ntpb", src)
        self.assertNotIn("self.graphs[self.bs].replay()", src)

    def test_page_physical_kv_copy_matches_slot_to_page_offset(self):
        page_size = 4
        buf = torch.arange(2 * 1 * 3 * 4 * 1 * 2, dtype=torch.float32).reshape(
            2, 1, 3, 4, 1, 2
        )
        src = torch.tensor([5, 6], dtype=torch.int32)
        tgt = torch.tensor([9, 10], dtype=torch.int32)
        src_page = torch.div(src, page_size, rounding_mode="floor")
        src_off = src % page_size
        tgt_page = torch.div(tgt, page_size, rounding_mode="floor")
        tgt_off = tgt % page_size
        expected = buf[:, :, src_page, src_off, :, :].clone()
        buf[:, :, tgt_page, tgt_off, :, :] = buf[:, :, src_page, src_off, :, :]
        torch.testing.assert_close(buf[:, :, tgt_page, tgt_off, :, :], expected)
        npu_src = (
            _REPO / "python/sglang/srt/hardware_backend/npu/memory_pool_npu.py"
        ).read_text()
        self.assertIn("def move_kv_cache", npu_src)
        self.assertIn("tgt_page", npu_src)
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
        self.assertIn("if (page_size != 1) and (topk != 1):", spec_src)
        self.assertIn("Always run for paged tree draft", spec_src)
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
        self.assertIn("if duplicate_cache_len > 0:", drafter_src)

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
        resolve = _load_tree_draft_helpers().resolve_fia_update_count
        self.assertEqual(resolve(None, 4, 28), 112)
        self.assertEqual(resolve(112, 4, 28), 112)
        with self.assertRaises(RuntimeError):
            resolve(8, 4, 28)
        with self.assertRaises(RuntimeError):
            resolve(224, 4, 28)

    def test_count_fia_kv_len_records(self):
        src_path = (
            _REPO
            / "python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_npu_graph_runner.py"
        )
        tree = ast.parse(src_path.read_text())
        names = {"_iter_graph_dispatch_records", "count_fia_kv_len_records"}
        keep = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        mod = ast.Module(body=keep, type_ignores=[])
        ast.fix_missing_locations(mod)
        ns = {}
        exec(compile(mod, str(src_path), "exec"), ns)
        count = ns["count_fia_kv_len_records"]
        attr = "actual_seq_lengths_kv"
        self.assertIsNone(count(SimpleNamespace(), attr))
        graph = SimpleNamespace(
            graph_dispatch_mode=SimpleNamespace(
                graph_dispatch_records=[
                    SimpleNamespace(update_info={attr: [1]}),
                    SimpleNamespace(update_info={"other": [1]}),
                    SimpleNamespace(update_info={attr: [2]}),
                ]
            )
        )
        self.assertEqual(count(graph, attr), 2)
        empty = SimpleNamespace(graph_dispatch_records=[])
        self.assertEqual(count(empty, attr), 0)

    def test_expand_batch_skips_expand_one_after_graph_submitted(self):
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
        windows = SRTreeDrafter.expand_batch(drafter, [req])
        self.assertEqual(windows, [([], None, None)])
        self.assertEqual(called["expand_one"], 0)

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
        self.assertIn("resolve_fia_update_count", graph_src)
        self.assertIn("count_fia_kv_len_records", graph_src)
        self.assertIn("graph_dispatch_records", graph_src)
        self.assertIn("NpuGraphReplaySubmittedError", graph_src)
        self.assertIn("seq_lens_cpu[: self.raw_bs]", graph_src)
        self.assertNotIn("normalize_tree_draft_kv_lens", graph_src)
        drafter_src = (
            _REPO
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_tree_drafter.py"
        ).read_text()
        self.assertIn("NpuGraphReplaySubmittedError", drafter_src)
        self.assertIn("skipping per-req retry", drafter_src)
        self.assertIn("falling back to eager", drafter_src)

    def test_npu_tree_draft_fia_alignment_source_guards(self):
        src = (
            _REPO
            / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
        ).read_text()
        self.assertIn("query = q.reshape(", src)
        self.assertIn("-1, 1, layer.tp_q_head_num, layer.qk_head_dim", src)
        self.assertIn("context_lens=context_lens", src)

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


if __name__ == "__main__":
    unittest.main()
