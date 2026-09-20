"""CPU contracts for NPU STANDALONE_REMOTE paged tree attention layout."""

from __future__ import annotations

import ast
import pathlib
import sys
import types
import unittest
from types import MethodType, SimpleNamespace
from unittest import mock

import torch

from sglang.srt.speculative.standalone_remote.drafter.sr_tree_paged_layout import (
    ALLOC_LEASE,
    ALLOC_ORDINARY,
    IMPL_PAGED_ATB,
    IMPL_PAGED_FIA,
    SR_TREE_PAGED_ENV,
    SRTreeExpandTxn,
    SRTreePagedMetadata,
    build_step_context_lens,
    context_lens_list,
    fill_active_rows,
    fill_paged_cpu_update_payload,
    kv_buckets_to_page_buckets,
    make_dummy_block_tables,
    materialize_branch_pages,
    materialize_prefix_tail_copy_slots,
    materialize_shared_prefix_pages,
    max_query_pages_for_tree,
    pages_per_branch,
    plan_prefix_tail_copy_indices,
    prepare_tree_paged_view,
    query_page_count,
    read_sr_tree_paged_env,
    remainder,
    select_page_bucket,
    shared_page_count,
    validate_tree_draft_paged_records,
    visible_token_slots_from_pages,
)
from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=8, suite="stage-a-test-cpu")

_REPO = pathlib.Path(__file__).resolve().parents[4]
_LAYOUT = (
    _REPO
    / "python/sglang/srt/speculative/standalone_remote/drafter/sr_tree_paged_layout.py"
)
_BACKEND = (
    _REPO / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
)
_RUNNER = (
    _REPO
    / "python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_npu_graph_runner.py"
)
_DRAFTER = (
    _REPO
    / "python/sglang/srt/speculative/standalone_remote/drafter/sr_tree_drafter.py"
)


def _source(path: pathlib.Path) -> str:
    return path.read_text()


def _fn_source(path: pathlib.Path, name: str) -> str:
    tree = ast.parse(path.read_text())
    text = path.read_text()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node) or ast.unparse(node)
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == name:
                    return ast.get_source_segment(text, child) or ast.unparse(child)
    raise AssertionError(f"{name} not found in {path}")


def _req_to_token(page_ids, page_size: int, extra: int = 0) -> torch.Tensor:
    bs = len(page_ids)
    n_pages = max(len(p) for p in page_ids)
    ctx = n_pages * page_size + extra
    req = torch.zeros((bs, ctx), dtype=torch.int64)
    for b, pages in enumerate(page_ids):
        for j, pid in enumerate(pages):
            start = j * page_size
            req[b, start : start + page_size] = int(pid) * page_size + torch.arange(
                page_size, dtype=torch.int64
            )
    return req


def _draft_slots(prefixes, page, topk, steps, branch_page_ids) -> torch.Tensor:
    batch = len(prefixes)
    slots = torch.zeros((batch, topk, steps), dtype=torch.int64)
    for b, prefix in enumerate(prefixes):
        rem = remainder(prefix, page)
        for k in range(topk):
            for s in range(steps):
                j = (rem + s) // page
                off = (rem + s) % page
                slots[b, k, s] = int(branch_page_ids[b][k][j]) * page + off
    return slots


class TestSrTreePagedEnv(CustomTestCase):
    def test_default_on_when_unset(self):
        self.assertTrue(read_sr_tree_paged_env({}))

    def test_explicit_off(self):
        for raw in ("", "0", "false", "no", "off", "maybe"):
            self.assertFalse(read_sr_tree_paged_env({SR_TREE_PAGED_ENV: raw}))

    def test_truthy(self):
        for raw in ("1", "true", "YES", "on"):
            self.assertTrue(read_sr_tree_paged_env({SR_TREE_PAGED_ENV: raw}))


class TestPrefixTailCopyPlan(CustomTestCase):
    def test_topk_and_prefix_ordinary_vs_lease(self):
        page = 128
        for topk in (2, 3, 4):
            for prefix in (127, 128, 129):
                rem = remainder(prefix, page)
                ordinary = plan_prefix_tail_copy_indices(
                    [prefix], ALLOC_ORDINARY, topk, page
                )
                lease = plan_prefix_tail_copy_indices(
                    [prefix], ALLOC_LEASE, topk, page
                )
                if rem == 0:
                    self.assertEqual(len(ordinary), 0)
                    self.assertEqual(len(lease), 0)
                    continue
                self.assertEqual(len(ordinary), rem * (topk - 1))
                self.assertEqual(len(lease), rem * topk)
                self.assertFalse((ordinary.branch == 0).any())
                self.assertTrue((lease.branch == 0).any())

    def test_empty_prefix_and_aligned(self):
        for prefix in (0, 128, 256):
            for kind in (ALLOC_ORDINARY, ALLOC_LEASE):
                plan = plan_prefix_tail_copy_indices([prefix], kind, 3, 128)
                self.assertEqual(len(plan), 0)

    def test_materialize_matches_logical_and_skips_host_d2h_of_maps(self):
        page, topk, steps = 128, 3, 5
        prefixes = [127]
        prefix_pages = [[9]]
        branch_ids = [[[9, 30], [21, 31], [22, 32]]]
        req = _req_to_token(prefix_pages, page)
        pool = torch.tensor([0], dtype=torch.int64)
        slots = _draft_slots(prefixes, page, topk, steps, branch_ids)
        branch = materialize_branch_pages(slots, prefixes, page, topk, steps)
        plan = plan_prefix_tail_copy_indices(prefixes, ALLOC_LEASE, topk, page)
        src, dst = materialize_prefix_tail_copy_slots(
            req, pool, branch, plan, page
        )
        self.assertEqual(int(src.numel()), 127 * topk)
        self.assertTrue(torch.equal(src[:127], req[0, :127]))
        self.assertEqual(int(dst[0]), int(branch[0, 0, 0]) * page)
        self.assertNotIn(".cpu()", _fn_source(_LAYOUT, "materialize_prefix_tail_copy_slots"))
        self.assertNotIn(".tolist()", _fn_source(_LAYOUT, "materialize_prefix_tail_copy_slots"))


class TestQueryVsAllocPages(CustomTestCase):
    def test_last_reserved_slot_crosses_page_without_forward(self):
        prefix, page, steps = 124, 128, 5
        self.assertEqual(pages_per_branch(remainder(prefix, page), steps, page), 2)
        self.assertEqual(max_query_pages_for_tree([prefix], steps, page), [1])
        self.assertEqual(query_page_count(prefix, steps - 2, page), 1)
        last_ctx = prefix + (steps - 2) + 1
        self.assertEqual(last_ctx, 128)
        reserved = prefix + steps
        self.assertGreater(reserved, page)

    def test_prepare_uses_query_width(self):
        page, topk, steps = 128, 2, 5
        prefixes = [124]
        req = _req_to_token([[40]], page)
        pool = torch.tensor([0])
        branch_ids = [[[40, 77], [55, 88]]]
        slots = _draft_slots(prefixes, page, topk, steps, branch_ids)
        tables, _shared, branch, active, n_sh, n_q = prepare_tree_paged_view(
            req, pool, slots, prefixes, page, topk, steps, dummy_page=99
        )
        self.assertEqual(n_sh, [0])
        self.assertEqual(n_q, [1])
        self.assertEqual(int(tables.shape[1]), 1)
        self.assertEqual(int(branch.shape[2]), 2)
        self.assertTrue(torch.equal(tables[:, 0], torch.tensor([40, 55], dtype=torch.int32)))


class TestBlockTablesAndPadding(CustomTestCase):
    def test_noncontiguous_pages_and_ragged_batch(self):
        page, topk, steps = 128, 2, 5
        prefixes = [127, 129]
        req = _req_to_token([[11], [4, 7]], page, extra=256)
        pool = torch.tensor([0, 1])
        branch_ids = [
            [[11, 20], [21, 22]],
            [[7, 30], [31, 32]],
        ]
        slots = _draft_slots(prefixes, page, topk, steps, branch_ids)
        tables, shared, branch, active, n_sh, n_q = prepare_tree_paged_view(
            req, pool, slots, prefixes, page, topk, steps, dummy_page=99
        )
        self.assertEqual(n_sh, [0, 1])
        self.assertEqual(shared[1, 0].item(), 4)
        self.assertEqual(int(tables[0, 0]), 11)
        self.assertEqual(int(tables[2, 0]), 4)
        self.assertEqual(int(tables[2, 1]), 7)
        self.assertTrue(active[:4].all())

    def test_padding_rows_use_dummy_not_slot_zero_ownership(self):
        dummy = 17
        tables = make_dummy_block_tables(6, 3, dummy)
        active = fill_active_rows(raw_bs=2, topk=2, capture_rows=6)
        padded = torch.where(active.view(-1, 1), tables, torch.full_like(tables, dummy))
        self.assertTrue(torch.equal(padded[4:], torch.full((2, 3), dummy)))
        self.assertFalse(torch.equal(padded[4:], torch.zeros((2, 3), dtype=torch.int32)))

    def test_batch_shrink_then_grow_clears_leftover(self):
        dummy = 5
        dest = torch.full((8, 4), 123, dtype=torch.int32)
        small = torch.arange(8, dtype=torch.int32).reshape(2, 4)
        dest.fill_(dummy)
        dest[:2, :4].copy_(small)
        dest[2:].fill_(dummy)
        self.assertTrue((dest[2:] == dummy).all())
        dest.fill_(dummy)
        big = torch.arange(24, dtype=torch.int32).reshape(6, 4)
        dest[:6].copy_(big)
        self.assertTrue(torch.equal(dest[:6], big))
        self.assertTrue((dest[6:] == dummy).all())
        self.assertEqual(id(dest), id(dest))


class TestParentDuplicateAndVisibleSlots(CustomTestCase):
    def test_visible_slots_match_oracle_after_parent_zero(self):
        page, topk, steps = 128, 2, 5
        prefixes = [127]
        req = _req_to_token([[9]], page)
        pool = torch.tensor([0])
        branch_ids = [[[9, 30], [21, 31]]]
        slots = _draft_slots(prefixes, page, topk, steps, branch_ids)
        shared = materialize_shared_prefix_pages(req, pool, prefixes, page)
        branch = materialize_branch_pages(slots, prefixes, page, topk, steps)
        before = visible_token_slots_from_pages(
            shared, branch, prefixes, step_id=1, page_size=page
        )
        parent_rows = torch.zeros((topk,), dtype=torch.int64)
        hist = slots[0, :, :2]
        remapped = hist[parent_rows]
        self.assertTrue(torch.equal(remapped[0], remapped[1]))
        after = visible_token_slots_from_pages(
            shared, branch, prefixes, step_id=1, page_size=page
        )
        self.assertEqual(before, after)
        self.assertNotEqual(before[0], before[1])


class TestOncePerRoundAndTxn(CustomTestCase):
    def test_prepare_counts_once_and_skips_empty_copy(self):
        class FakeInner:
            def __init__(self):
                self._sr_tree_paged_prep_count = 0
                self._sr_tree_paged_copy_count = 0
                self.init_calls = 0

            def init_forward_metadata(self, *_a, **_k):
                self.init_calls += 1

        inner = FakeInner()
        prefixes = [128]
        page, topk, steps = 128, 2, 5
        req = _req_to_token([[3]], page)
        pool = torch.tensor([0])
        slots = _draft_slots(prefixes, page, topk, steps, [[[3, 8], [9, 10]]])
        inner._sr_tree_paged_prep_count += 1
        plan = plan_prefix_tail_copy_indices(prefixes, ALLOC_ORDINARY, topk, page)
        if len(plan):
            inner._sr_tree_paged_copy_count += 1
        inner.init_forward_metadata()
        inner.init_forward_metadata()
        self.assertEqual(inner._sr_tree_paged_prep_count, 1)
        self.assertEqual(inner._sr_tree_paged_copy_count, 0)
        self.assertEqual(inner.init_calls, 2)
        prepare_tree_paged_view(req, pool, slots, prefixes, page, topk, steps)

    def test_eager_submitted_skips_restore_and_retry(self):
        class Submitted(Exception):
            pass

        restored = []
        retried = []

        def expand(fail_after_submit=True):
            txn = SRTreeExpandTxn()
            txn.allocation_owned = True
            try:
                txn.mark_copy_begin()
                txn.mark_compute_begin()
                if fail_after_submit:
                    raise RuntimeError("device work unknown")
                txn.completion_confirmed = True
            except Submitted:
                raise
            except Exception:
                if txn.in_flight() and not txn.completion_confirmed:
                    raise Submitted("refuse rollback")
                if txn.may_rollback():
                    restored.append("rollback")
                    txn.rolled_back = True
                raise
            finally:
                abandon = txn.in_flight() and not txn.completion_confirmed
                if abandon:
                    restored.append("abandon")
                elif not txn.rolled_back:
                    restored.append("success-restore")

        with self.assertRaises(Exception):
            expand(True)
        self.assertEqual(restored, ["abandon"])
        self.assertEqual(retried, [])

        restored.clear()
        txn = SRTreeExpandTxn()
        txn.allocation_owned = True
        txn.completion_confirmed = True
        restored.append("success-restore")
        self.assertEqual(restored, ["success-restore"])
        self.assertTrue(txn.may_rollback() or txn.completion_confirmed)


class TestPagedGraphRecords(CustomTestCase):
    def test_record_count_type_and_attr_mismatch_refuse_replay(self):
        class Err(Exception):
            def __init__(self, msg, scope="graph"):
                super().__init__(msg)
                self.scope = scope

        fake = types.ModuleType("sglang.srt.speculative.spec_utils")
        fake.NpuGraphPreparationError = Err
        fake.inspect_dispatch_record = lambda rec, attr: (rec.name, rec.has)
        fake.normalize_fia_op_name = lambda name: name
        recs = [
            SimpleNamespace(name="npu_fused_infer_attention_score", has=True)
            for _ in range(4)
        ]
        with mock.patch.dict(sys.modules, {"sglang.srt.speculative.spec_utils": fake}):
            n, step_ids = validate_tree_draft_paged_records(
                recs, 2, 2, "actual_seq_lengths_kv", IMPL_PAGED_FIA
            )
            self.assertEqual(n, 4)
            self.assertEqual(step_ids, [0, 0, 1, 1])
            with self.assertRaises(Err):
                validate_tree_draft_paged_records(
                    recs[:3], 2, 2, "actual_seq_lengths_kv", IMPL_PAGED_FIA
                )
            bad = list(recs)
            bad[1] = SimpleNamespace(name="npu_paged_attention", has=True)
            with self.assertRaises(Err):
                validate_tree_draft_paged_records(
                    bad, 2, 2, "actual_seq_lengths_kv", IMPL_PAGED_FIA
                )
            missing = [
                SimpleNamespace(name="npu_fused_infer_attention_score", has=False)
                for _ in recs
            ]
            with self.assertRaises(Err):
                validate_tree_draft_paged_records(
                    missing, 2, 2, "actual_seq_lengths_kv", IMPL_PAGED_FIA
                )
            with self.assertRaises(Err):
                validate_tree_draft_paged_records(
                    recs, 2, 2, "actual_seq_lengths_kv", IMPL_PAGED_ATB
                )

    def test_independent_step_payloads(self):
        lens0 = build_step_context_lens([10, 12], topk=2, step_id=0, capture_rows=6)
        lens1 = build_step_context_lens([10, 12], topk=2, step_id=1, capture_rows=6)
        self.assertEqual(lens0.tolist(), [11, 11, 13, 13, 1, 1])
        self.assertEqual(lens1.tolist(), [12, 12, 14, 14, 1, 1])
        payload = [
            {"actual_seq_lengths_kv": [1, 1, 1, 1, 1, 1]},
            {"actual_seq_lengths_kv": torch.ones(6, dtype=torch.int32)},
        ]
        fill_paged_cpu_update_payload(
            payload,
            [lens0.tolist(), lens1.tolist()],
            [0, 1],
            "actual_seq_lengths_kv",
        )
        self.assertEqual(payload[0]["actual_seq_lengths_kv"], lens0.tolist())
        self.assertEqual(payload[1]["actual_seq_lengths_kv"].tolist(), lens1.tolist())

    def test_select_page_bucket_smallest_fit(self):
        pages = kv_buckets_to_page_buckets([128, 256, 512], 128)
        self.assertEqual(sorted(pages), [1, 2, 4])
        self.assertEqual(select_page_bucket(1, pages), 1)
        self.assertEqual(select_page_bucket(3, pages), 4)
        self.assertIsNone(select_page_bucket(8, pages))


class TestSourceGuards(CustomTestCase):
    def test_bind_validates_before_buffer_write(self):
        bind_src = _fn_source(_BACKEND, "bind_sr_tree_paged_replay")
        validate_src = _fn_source(_BACKEND, "_validate_sr_tree_paged_replay")
        self.assertNotIn("or []", bind_src)
        self.assertNotIn("or []", validate_src)
        self.assertNotIn("rows = min", bind_src)
        self.assertNotIn("cols = min", bind_src)
        self.assertLess(
            bind_src.find("_validate_sr_tree_paged_replay"),
            bind_src.find("fill_"),
        )
        self.assertLess(
            bind_src.find("_validate_sr_tree_paged_replay"),
            bind_src.find("copy_"),
        )
        for banned in (".cpu()", ".item()", ".tolist()"):
            self.assertNotIn(banned, validate_src)
        replay_src = _fn_source(_RUNNER, "replay")
        self.assertLess(
            replay_src.find("try:"), replay_src.find("bind_sr_tree_paged_replay")
        )
        self.assertIn("_snapshot_paged_eager_metadata", replay_src)
        self.assertIn("_restore_paged_eager_metadata", replay_src)
        self.assertIn("_paged_eager_restore_valid", replay_src)
        batch_src = _fn_source(_DRAFTER, "expand_batch")
        self.assertLess(
            batch_src.find("_tree_batch_isolate_count"),
            batch_src.find("_log_tree_failure"),
        )

    def test_backend_paged_path_reads_cache_only(self):
        run_src = _fn_source(_BACKEND, "_run_sr_tree_paged_attention")
        self.assertNotIn("gather_kv_into", run_src)
        self.assertNotIn("_ensure_tree_kv_scratch", run_src)
        self.assertNotIn("zero_gathered_kv_padding", run_src)
        self.assertNotIn("set_kv_buffer", run_src)
        decode_src = _fn_source(_BACKEND, "forward_decode")
        self.assertLess(
            decode_src.find("_can_run_sr_tree_paged"),
            decode_src.find("forward_decode_graph"),
        )
        self.assertIn("not self._paged_impl_selected()", decode_src)
        init_src = _fn_source(_BACKEND, "_init_tree_shared_prefix")
        self.assertIn("read_sr_tree_paged_env", init_src)
        self.assertIn("paged_capable", init_src)
        self.assertIn(SR_TREE_PAGED_ENV, _source(_LAYOUT))
        self.assertIn("prepare_sr_tree_paged_eager", _source(_BACKEND))
        self.assertIn("bind_sr_tree_paged_capture", _source(_BACKEND))
        self.assertIn("bind_sr_tree_paged_replay", _source(_BACKEND))
        prep_src = _fn_source(_BACKEND, "prepare_sr_tree_paged_eager")
        self.assertIn("_sr_tree_paged_prep_count += 1", prep_src)
        self.assertIn("plan_prefix_tail_copy_indices", prep_src)

    def test_graph_runner_paged_serial_and_validator(self):
        src = _source(_RUNNER)
        self.assertIn("validate_tree_draft_paged_records", src)
        self.assertIn("_tree_paged", src)
        self.assertIn("bind_sr_tree_paged_capture", src)
        self.assertIn("bind_sr_tree_paged_replay", src)
        self.assertIn("fill_paged_cpu_update_payload", src)
        self.assertIn("if getattr(self, \"_tree_paged\", False)", src)
        self.assertIn("SGLANG_NPU_TREE_FIA_SERIAL_UPDATE", src)
        skip_src = _fn_source(_RUNNER, "capture_one_batch_size")
        self.assertIn("not getattr(self, \"_tree_paged\", False)", skip_src)
        can_src = _fn_source(_RUNNER, "can_run")
        self.assertIn("_tree_fia_maps", can_src)
        self.assertIn("_tree_paged", can_src)

    def test_drafter_marks_before_submit_and_refuses_retry(self):
        src = _fn_source(_DRAFTER, "_expand_tree")
        self.assertLess(src.find("mark_copy_begin"), src.find("_prepare_paged_tree_round"))
        self.assertLess(src.find("mark_compute_begin"), src.find("replay"))
        self.assertIn("NpuGraphReplaySubmittedError", src)
        self.assertIn("abandon", src)
        can_src = _fn_source(_DRAFTER, "_can_run_tree_graph")
        self.assertIn("not getattr(runner, \"_tree_paged\", False)", can_src)
        batch_src = _fn_source(_DRAFTER, "expand_batch")
        self.assertIn("except NpuGraphReplaySubmittedError:\n            raise", batch_src)
        one_src = _fn_source(_DRAFTER, "_expand_one")
        self.assertIn("except NpuGraphReplaySubmittedError:\n            raise", one_src)

    def test_does_not_call_build_tree_draft_block_tables_for_paged(self):
        init_src = _fn_source(_BACKEND, "init_forward_metadata")
        self.assertIn("_paged_impl_selected", init_src)
        self.assertIn("sr_tree_paged", init_src)


class TestAssembleSharedAndBranch(CustomTestCase):
    def test_ordinary_k0_reuses_prefix_tail_page_only_when_remainder(self):
        page, topk, steps = 128, 2, 5
        prefixes = [127]
        req = _req_to_token([[9]], page)
        pool = torch.tensor([0])
        slots = _draft_slots(prefixes, page, topk, steps, [[[9, 30], [21, 31]]])
        branch = materialize_branch_pages(slots, prefixes, page, topk, steps)
        self.assertEqual(int(branch[0, 0, 0]), 9)
        self.assertEqual(int(branch[0, 1, 0]), 21)
        aligned = _draft_slots([128], page, topk, steps, [[[40], [41]]])
        req_a = _req_to_token([[3]], page)
        branch_a = materialize_branch_pages(
            aligned, [128], page, topk, steps
        )
        self.assertEqual(int(branch_a[0, 0, 0]), 40)
        self.assertEqual(int(branch_a[0, 1, 0]), 41)

    def test_shared_complete_pages_only(self):
        page = 128
        req = _req_to_token([[4, 7]], page)
        pool = torch.tensor([0])
        shared = materialize_shared_prefix_pages(req, pool, [129], page)
        self.assertEqual(int(shared.shape[1]), 1)
        self.assertEqual(int(shared[0, 0]), 4)
        self.assertEqual(shared_page_count(129, page), 1)
        self.assertEqual(shared_page_count(127, page), 0)


def _extract_class_methods(path: pathlib.Path, cls: str, names, ns):
    tree = ast.parse(path.read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    nodes = [
        n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    if len(nodes) != len(names):
        missing = set(names) - {n.name for n in nodes}
        raise AssertionError(f"missing {missing} in {cls}")
    ns = dict(ns)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *nodes], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        ns,
    )
    return {name: ns[name] for name in names}


def _load_npu_graph_prep_error():
    path = _REPO / "python/sglang/srt/speculative/spec_utils.py"
    tree = ast.parse(path.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "NpuGraphPreparationError"
    )
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    return ns["NpuGraphPreparationError"]


class _ForwardMetadata:
    def __init__(self):
        self.sr_tree_paged = None
        self.block_tables = None


class TestPagedReplayBind(CustomTestCase):
    def setUp(self):
        self.PrepError = _load_npu_graph_prep_error()
        fn = _extract_class_methods(
            _BACKEND,
            "AscendAttnMultiStepDraftBackend",
            ["_validate_sr_tree_paged_replay", "bind_sr_tree_paged_replay"],
            dict(
                torch=torch,
                NpuGraphPreparationError=self.PrepError,
                build_step_context_lens=build_step_context_lens,
                context_lens_list=context_lens_list,
                SRTreePagedMetadata=SRTreePagedMetadata,
                ForwardMetadata=_ForwardMetadata,
            ),
        )
        self._bind_fn = fn["bind_sr_tree_paged_replay"]
        self._validate_fn = fn["_validate_sr_tree_paged_replay"]

    def _inner(self, step, dest, dest_act, eager_meta=None, had_fm=True):
        inner = SimpleNamespace(
            speculative_step_id=step,
            tree_attention_impl="paged_atb",
            cuda_graph_paged_block_tables=dest,
            cuda_graph_paged_active=dest_act,
            _sr_tree_paged_meta=eager_meta,
            forward_metadata=_ForwardMetadata() if had_fm else None,
        )
        if had_fm:
            inner.forward_metadata.sr_tree_paged = eager_meta
            inner.forward_metadata.block_tables = (
                None if eager_meta is None else eager_meta.block_tables
            )
        inner.bind_sr_tree_paged_metadata = MethodType(
            lambda self, meta: setattr(self, "_sr_tree_paged_meta", meta),
            inner,
        )
        return inner

    def _eager_meta(self, tables, active, dummy=0, impl="paged_atb"):
        lens = torch.ones(int(tables.shape[0]), dtype=torch.int32)
        return SRTreePagedMetadata(
            block_tables=tables,
            active_rows=active,
            context_lens_cpu=lens,
            context_lens_list=context_lens_list(lens),
            dummy_page=dummy,
            max_pages=int(tables.shape[1]),
            impl=impl,
        )

    def _backend(self, raw_bs, capture_bs, topk, max_pages, prefix, src, src_act, dummy=0):
        dest = torch.full(
            (capture_bs * topk, max_pages), 777, dtype=torch.int32
        )
        dest_act = torch.full((capture_bs * topk,), 9, dtype=torch.int32)
        eager = self._eager_meta(src, src_act, dummy=dummy)
        inners = [
            self._inner(0, dest, dest_act, eager_meta=eager),
            self._inner(1, dest, dest_act, eager_meta=eager),
        ]
        backend = SimpleNamespace(
            topk=topk,
            speculative_num_steps=3,
            attn_backends=inners,
            _tree_replay_raw_bs=raw_bs,
            _tree_replay_capture_bs=capture_bs,
            _paged_round_prefix=prefix,
            _paged_round_tables=src,
            _paged_round_active=src_act,
            _paged_round_dummy=dummy,
            _paged_round_impl="paged_atb",
            paged_impl_selected=lambda: True,
        )
        backend.bind_sr_tree_paged_replay = MethodType(self._bind_fn, backend)
        backend._validate_sr_tree_paged_replay = MethodType(
            self._validate_fn, backend
        )
        return backend, dest, dest_act, eager, inners

    def test_int_prefix_dual_request_binds_step_lengths(self):
        topk, capture_bs, max_pages = 2, 2, 4
        src = torch.arange(16, dtype=torch.int32).reshape(4, 4)
        src_act = torch.tensor([1, 1, 1, 1], dtype=torch.int32)
        for dtype in (torch.int32, torch.int64):
            prefix = torch.tensor([10, 20], dtype=dtype)
            backend, dest, dest_act, _, inners = self._backend(
                2, capture_bs, topk, max_pages, prefix, src, src_act, dummy=99
            )
            backend.bind_sr_tree_paged_replay(capture_bs, max_pages)
            torch.testing.assert_close(dest[:4, :4], src.to(torch.int32))
            torch.testing.assert_close(dest_act[:4], src_act)
            self.assertEqual(
                list(inners[0]._sr_tree_paged_meta.context_lens_list),
                [11, 11, 21, 21],
            )
            self.assertEqual(
                list(inners[1]._sr_tree_paged_meta.context_lens_list),
                [12, 12, 22, 22],
            )
            self.assertEqual(
                inners[0]._sr_tree_paged_meta.block_tables.data_ptr(), dest.data_ptr()
            )
            self.assertIs(
                inners[0].forward_metadata.sr_tree_paged,
                inners[0]._sr_tree_paged_meta,
            )

    def test_zero_prefix_is_valid_not_missing(self):
        topk, capture_bs, max_pages = 2, 2, 3
        src = torch.ones((4, 2), dtype=torch.int32)
        src_act = torch.ones(4, dtype=torch.int32)
        prefix = torch.tensor([0, 0], dtype=torch.int32)
        backend, _, _, _, inners = self._backend(
            2, capture_bs, topk, max_pages, prefix, src, src_act
        )
        backend.bind_sr_tree_paged_replay(capture_bs, max_pages)
        self.assertEqual(
            list(inners[0]._sr_tree_paged_meta.context_lens_list),
            [1, 1, 1, 1],
        )

    def test_prefix_shape_error_does_not_write_buffers(self):
        topk, capture_bs, max_pages = 2, 2, 4
        src = torch.ones((4, 2), dtype=torch.int32)
        src_act = torch.ones(4, dtype=torch.int32)
        prefix = torch.tensor([10], dtype=torch.int32)
        backend, dest, dest_act, eager, inners = self._backend(
            2, capture_bs, topk, max_pages, prefix, src, src_act
        )
        with self.assertRaises(self.PrepError):
            backend.bind_sr_tree_paged_replay(capture_bs, max_pages)
        self.assertTrue(torch.equal(dest, torch.full_like(dest, 777)))
        self.assertTrue(torch.equal(dest_act, torch.full_like(dest_act, 9)))
        self.assertIs(inners[0]._sr_tree_paged_meta, eager)
        self.assertIs(inners[1]._sr_tree_paged_meta, eager)
        backend._paged_round_prefix = None
        with self.assertRaises(self.PrepError):
            backend.bind_sr_tree_paged_replay(capture_bs, max_pages)
        self.assertTrue(torch.equal(dest, torch.full_like(dest, 777)))

    def test_capacity_fail_before_any_graph_write(self):
        topk, capture_bs, max_pages = 2, 2, 4
        src = torch.arange(16, dtype=torch.int32).reshape(4, 4)
        src_act = torch.ones(4, dtype=torch.int32)
        prefix = torch.tensor([3, 5], dtype=torch.int32)
        backend, dest, dest_act, eager, inners = self._backend(
            2, capture_bs, topk, max_pages, prefix, src, src_act
        )
        dest.resize_(3, 4)
        writes = []
        orig_fill = torch.Tensor.fill_
        orig_copy = torch.Tensor.copy_

        def watch_fill(tensor, value):
            if tensor.data_ptr() in {dest.data_ptr(), dest_act.data_ptr()}:
                writes.append("fill")
            return orig_fill(tensor, value)

        def watch_copy(tensor, src_t, *args, **kwargs):
            if tensor.data_ptr() in {dest.data_ptr(), dest_act.data_ptr()}:
                writes.append("copy")
            return orig_copy(tensor, src_t, *args, **kwargs)

        with mock.patch.object(torch.Tensor, "fill_", watch_fill), mock.patch.object(
            torch.Tensor, "copy_", watch_copy
        ):
            with self.assertRaises(self.PrepError):
                backend.bind_sr_tree_paged_replay(capture_bs, max_pages)
        self.assertEqual(writes, [])
        self.assertIs(inners[0]._sr_tree_paged_meta, eager)

    def test_missing_capture_buffers_do_not_allocate_substitutes(self):
        topk, capture_bs, max_pages = 2, 2, 4
        src = torch.ones((4, 2), dtype=torch.int32)
        src_act = torch.ones(4, dtype=torch.int32)
        prefix = torch.tensor([4, 6], dtype=torch.int32)
        backend, dest, dest_act, _, inners = self._backend(
            2, capture_bs, topk, max_pages, prefix, src, src_act
        )
        inners[0].cuda_graph_paged_block_tables = None
        inners[0].cuda_graph_paged_active = None
        with self.assertRaises(self.PrepError):
            backend.bind_sr_tree_paged_replay(capture_bs, max_pages)
        self.assertIsNone(inners[0].cuda_graph_paged_block_tables)
        self.assertIsNone(inners[0].cuda_graph_paged_active)

    def test_raw_bs_shrink_grow_leaves_no_padding_residue(self):
        topk, capture_bs, max_pages, dummy = 2, 2, 4, 99
        dest_holder = {}

        def run(raw_bs, prefix, src, src_act):
            backend, dest, dest_act, _, inners = self._backend(
                raw_bs, capture_bs, topk, max_pages, prefix, src, src_act, dummy=dummy
            )
            if "dest" not in dest_holder:
                dest_holder["dest"] = dest
                dest_holder["dest_act"] = dest_act
            else:
                inners[0].cuda_graph_paged_block_tables = dest_holder["dest"]
                inners[0].cuda_graph_paged_active = dest_holder["dest_act"]
                inners[1].cuda_graph_paged_block_tables = dest_holder["dest"]
                inners[1].cuda_graph_paged_active = dest_holder["dest_act"]
            backend.bind_sr_tree_paged_replay(capture_bs, max_pages)
            return dest_holder["dest"], dest_holder["dest_act"], inners

        src2 = torch.arange(16, dtype=torch.int32).reshape(4, 4) + 10
        act2 = torch.tensor([1, 0, 1, 0], dtype=torch.int32)
        dest, dest_act, inners = run(
            2, torch.tensor([8, 16], dtype=torch.int32), src2, act2
        )
        ptr = dest.data_ptr()
        src1 = torch.arange(8, dtype=torch.int32).reshape(2, 4) + 50
        act1 = torch.tensor([1, 1], dtype=torch.int32)
        dest, dest_act, inners = run(
            1, torch.tensor([8], dtype=torch.int32), src1, act1
        )
        self.assertEqual(dest.data_ptr(), ptr)
        torch.testing.assert_close(dest[:2], src1)
        self.assertTrue(torch.equal(dest[2:], torch.full((2, 4), dummy, dtype=torch.int32)))
        self.assertTrue(torch.equal(dest_act[2:], torch.zeros(2, dtype=torch.int32)))
        self.assertEqual(
            list(inners[0]._sr_tree_paged_meta.context_lens_list),
            [9, 9, 1, 1],
        )
        dest, dest_act, inners = run(
            2, torch.tensor([8, 16], dtype=torch.int32), src2, act2
        )
        self.assertEqual(dest.data_ptr(), ptr)
        torch.testing.assert_close(dest[:4], src2)
        torch.testing.assert_close(dest_act[:4], act2)
        self.assertEqual(
            list(inners[0]._sr_tree_paged_meta.context_lens_list),
            [9, 9, 17, 17],
        )


if __name__ == "__main__":
    unittest.main()
