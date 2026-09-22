"""CPU contracts for NPU STANDALONE_REMOTE paged tree attention layout."""

from __future__ import annotations

import ast
import pathlib
import sys
import threading
import time
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
    SR_TAIL_UPDATE_OVERLAP_ENV,
    SR_TREE_UPDATE_OVERLAP_ENV,
    SR_TREE_WARMUP_ENV,
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
    quantize_page_width,
    query_page_count,
    read_sr_tree_paged_env,
    read_sr_tail_update_overlap_env,
    read_sr_tree_update_overlap_env,
    read_sr_tree_warmup_env,
    remainder,
    resolve_eager_page_buckets,
    select_page_bucket,
    shared_page_count,
    tree_paged_shape_key,
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

    def test_warmup_env_default_on(self):
        self.assertTrue(read_sr_tree_warmup_env({}))
        self.assertFalse(read_sr_tree_warmup_env({SR_TREE_WARMUP_ENV: "0"}))
        self.assertTrue(read_sr_tree_warmup_env({SR_TREE_WARMUP_ENV: "1"}))

    def test_update_overlap_env_default_on(self):
        self.assertTrue(read_sr_tree_update_overlap_env({}))
        self.assertTrue(read_sr_tree_update_overlap_env({SR_TREE_UPDATE_OVERLAP_ENV: "1"}))
        self.assertFalse(read_sr_tree_update_overlap_env({SR_TREE_UPDATE_OVERLAP_ENV: "0"}))
        self.assertFalse(
            read_sr_tree_update_overlap_env({SR_TREE_UPDATE_OVERLAP_ENV: "false"})
        )
        for raw in ("1", "true", "YES", "on"):
            self.assertTrue(
                read_sr_tree_update_overlap_env({SR_TREE_UPDATE_OVERLAP_ENV: raw})
            )

    def test_tail_update_overlap_env_default_on(self):
        self.assertTrue(read_sr_tail_update_overlap_env({}))
        self.assertTrue(
            read_sr_tail_update_overlap_env({SR_TAIL_UPDATE_OVERLAP_ENV: "1"})
        )
        self.assertFalse(
            read_sr_tail_update_overlap_env({SR_TAIL_UPDATE_OVERLAP_ENV: "0"})
        )
        self.assertFalse(
            read_sr_tail_update_overlap_env({SR_TAIL_UPDATE_OVERLAP_ENV: "false"})
        )
        for raw in ("1", "true", "YES", "on"):
            self.assertTrue(
                read_sr_tail_update_overlap_env({SR_TAIL_UPDATE_OVERLAP_ENV: raw})
            )


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

    def test_thread_start_failure_after_copy_refuses_rollback_when_confirm_fails(self):
        expand_src = _fn_source(_DRAFTER, "_expand_tree")
        self.assertLess(expand_src.find("mark_copy_begin"), expand_src.find("replay"))
        self.assertIn("_try_confirm_tree_completion", expand_src)
        self.assertIn("tree expand in-flight; refuse rollback", expand_src)

        path = (
            _REPO / "python/sglang/srt/speculative/spec_utils.py"
        )
        tree = ast.parse(path.read_text())
        names = {"NpuGraphReplaySubmittedError", "run_npu_graph_update_and_replay"}
        nodes = [
            n
            for n in tree.body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
        ]
        ns = {"threading": threading}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
        submitted_cls = ns["NpuGraphReplaySubmittedError"]
        run = ns["run_npu_graph_update_and_replay"]

        restored = []
        lease_freed = []
        retried = []
        replayed = []
        confirmed = []
        pending = ["lease"]

        class BoomThread:
            def __init__(self, target=None, daemon=None, **kwargs):
                self.daemon = daemon

            def start(self):
                raise RuntimeError("start boom")

            def join(self):
                raise AssertionError("must not join an unstarted thread")

        orig = threading.Thread
        threading.Thread = BoomThread
        graph_submitted = False
        try:
            txn = SRTreeExpandTxn()
            txn.allocation_owned = True
            txn.lease_state = {"page_slots": [7]}
            try:
                txn.mark_copy_begin()
                txn.mark_compute_begin()
                run(lambda: None, lambda: replayed.append("replay"), overlap=True)
            except submitted_cls:
                graph_submitted = True
                raise
            except Exception as exc:
                if txn.in_flight():
                    ok = False
                    confirmed.append(True)
                    if not ok:
                        graph_submitted = True
                        raise submitted_cls(
                            "tree expand in-flight; refuse rollback"
                        ) from exc
                    txn.completion_confirmed = True
                if txn.may_rollback() and not graph_submitted:
                    restored.append("allocator")
                    lease_freed.append("lease")
                    txn.rolled_back = True
                raise
            finally:
                abandon = graph_submitted or (
                    txn.in_flight() and not txn.completion_confirmed
                )
                if abandon:
                    pending = None
        except submitted_cls:
            pass
        finally:
            threading.Thread = orig

        self.assertEqual(replayed, [])
        self.assertEqual(confirmed, [True])
        self.assertEqual(restored, [])
        self.assertEqual(lease_freed, [])
        self.assertEqual(retried, [])
        self.assertIsNone(pending)


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
        list_dest = [1, 1, 1, 1, 1, 1]
        tensor_dest = torch.ones(6, dtype=torch.int32)
        payload = [
            {"actual_seq_lengths_kv": list_dest},
            {"actual_seq_lengths_kv": tensor_dest},
        ]
        fill_paged_cpu_update_payload(
            payload,
            [lens0.tolist(), lens1.tolist()],
            [0, 1],
            "actual_seq_lengths_kv",
        )
        self.assertIs(payload[0]["actual_seq_lengths_kv"], list_dest)
        self.assertIs(payload[1]["actual_seq_lengths_kv"], tensor_dest)
        self.assertEqual(payload[0]["actual_seq_lengths_kv"], lens0.tolist())
        self.assertEqual(payload[1]["actual_seq_lengths_kv"].tolist(), lens1.tolist())
        list_dest[0] = 99
        tensor_dest[0] = 99
        self.assertEqual(payload[0]["actual_seq_lengths_kv"][0], 99)
        self.assertEqual(int(payload[1]["actual_seq_lengths_kv"][0]), 99)

    def test_same_step_tensor_dests_are_independent(self):
        src = [11, 12, 13, 14]
        dests = [torch.ones(4, dtype=torch.int32) for _ in range(3)]
        payload = [{"context_lens": dest} for dest in dests]
        ids_before = [id(dest) for dest in dests]
        ptrs_before = [int(dest.data_ptr()) for dest in dests]
        fill_paged_cpu_update_payload(payload, [src], [0, 0, 0], "context_lens")
        self.assertEqual([id(rec["context_lens"]) for rec in payload], ids_before)
        self.assertEqual([int(d.data_ptr()) for d in dests], ptrs_before)
        self.assertEqual(len(set(ptrs_before)), 3)
        for dest in dests:
            self.assertEqual(dest.tolist(), src)
        dests[0].fill_(7)
        self.assertEqual(dests[1].tolist(), src)
        self.assertEqual(dests[2].tolist(), src)

    def test_shuffled_step_ids_fill_by_id(self):
        lens0 = [1, 1, 1, 1]
        lens1 = [2, 2, 2, 2]
        dests = [torch.zeros(4, dtype=torch.int32) for _ in range(3)]
        payload = [{"context_lens": dest} for dest in dests]
        fill_paged_cpu_update_payload(
            payload, [lens0, lens1], [1, 0, 1], "context_lens"
        )
        self.assertEqual(dests[0].tolist(), lens1)
        self.assertEqual(dests[1].tolist(), lens0)
        self.assertEqual(dests[2].tolist(), lens1)

    def test_list_dest_updated_in_place(self):
        dest0 = [0, 0, 0]
        dest1 = [0, 0, 0]
        payload = [
            {"actual_seq_lengths_kv": dest0},
            {"actual_seq_lengths_kv": dest1},
        ]
        fill_paged_cpu_update_payload(
            payload, [[4, 5, 6], [7, 8, 9]], [0, 1], "actual_seq_lengths_kv"
        )
        self.assertIs(payload[0]["actual_seq_lengths_kv"], dest0)
        self.assertIs(payload[1]["actual_seq_lengths_kv"], dest1)
        self.assertEqual(dest0, [4, 5, 6])
        self.assertEqual(dest1, [7, 8, 9])

    def test_tensor_ctor_once_per_step_key_with_wraps(self):
        from sglang.srt.speculative.standalone_remote.drafter import (
            sr_tree_paged_layout as layout_mod,
        )

        dests = [torch.ones(4, dtype=torch.int32) for _ in range(4)]
        payload = [{"context_lens": dest} for dest in dests]
        real_tensor = torch.tensor
        with mock.patch.object(
            layout_mod.torch, "tensor", wraps=real_tensor
        ) as tensor_ctor:
            fill_paged_cpu_update_payload(
                payload,
                [[1, 1, 1, 1], [2, 2, 2, 2]],
                [0, 0, 1, 1],
                "context_lens",
            )
        self.assertEqual(tensor_ctor.call_count, 2)
        for call in tensor_ctor.call_args_list:
            self.assertEqual(call.kwargs.get("device"), "cpu")
        self.assertEqual(dests[0].tolist(), [1, 1, 1, 1])
        self.assertEqual(dests[2].tolist(), [2, 2, 2, 2])

    def test_second_call_refreshes_values_same_dest_objects(self):
        dests = [torch.ones(4, dtype=torch.int32) for _ in range(2)]
        payload = [{"context_lens": dest} for dest in dests]
        ids_before = [id(dest) for dest in dests]
        fill_paged_cpu_update_payload(
            payload, [[1, 1, 1, 1], [2, 2, 2, 2]], [0, 1], "context_lens"
        )
        fill_paged_cpu_update_payload(
            payload, [[7, 7, 7, 7], [8, 8, 8, 8]], [0, 1], "context_lens"
        )
        self.assertEqual([id(rec["context_lens"]) for rec in payload], ids_before)
        self.assertEqual(dests[0].tolist(), [7, 7, 7, 7])
        self.assertEqual(dests[1].tolist(), [8, 8, 8, 8])

    def test_late_illegal_record_leaves_earlier_dests_unchanged(self):
        dest0 = torch.ones(4, dtype=torch.int32)
        dest1 = torch.ones(4, dtype=torch.int32)
        dest2 = torch.ones(3, dtype=torch.int32)
        before0 = dest0.clone()
        before1 = dest1.clone()
        payload = [
            {"context_lens": dest0},
            {"context_lens": dest1},
            {"context_lens": dest2},
        ]
        with self.assertRaises(ValueError):
            fill_paged_cpu_update_payload(
                payload, [[9, 9, 9, 9]], [0, 0, 0], "context_lens"
            )
        self.assertTrue(torch.equal(dest0, before0))
        self.assertTrue(torch.equal(dest1, before1))

        list0 = [1, 1, 1, 1]
        list1 = [1, 1, 1, 1]
        list_payload = [
            {"context_lens": list0},
            {"context_lens": list1},
            {"context_lens": torch.ones(4, dtype=torch.int32)},
        ]
        with self.assertRaises(ValueError):
            fill_paged_cpu_update_payload(
                list_payload, [[3, 3, 3, 3]], [0, 0, 5], "context_lens"
            )
        self.assertEqual(list0, [1, 1, 1, 1])
        self.assertEqual(list1, [1, 1, 1, 1])

    def test_select_page_bucket_smallest_fit(self):
        pages = kv_buckets_to_page_buckets([128, 256, 512], 128)
        self.assertEqual(sorted(pages), [1, 2, 4])
        self.assertEqual(select_page_bucket(1, pages), 1)
        self.assertEqual(select_page_bucket(3, pages), 4)
        self.assertIsNone(select_page_bucket(8, pages))

    def test_quantize_page_width_bucket_then_pow2(self):
        pages = kv_buckets_to_page_buckets([256, 512, 1024], 128)
        self.assertEqual(sorted(pages), [2, 4, 8])
        self.assertEqual(quantize_page_width(1, pages), 2)
        self.assertEqual(quantize_page_width(3, pages), 4)
        self.assertEqual(quantize_page_width(8, pages), 8)
        self.assertEqual(quantize_page_width(9, pages), 16)
        self.assertEqual(quantize_page_width(0, None), 1)
        self.assertEqual(quantize_page_width(-3, []), 1)
        for need in (1, 2, 3, 5, 9, 17):
            self.assertGreaterEqual(quantize_page_width(need, pages), need)
            self.assertGreaterEqual(quantize_page_width(need, None), need)

    def test_resolve_eager_page_buckets_matches_can_run(self):
        self.assertEqual(
            resolve_eager_page_buckets([256, 512, 1024], 128),
            kv_buckets_to_page_buckets([256, 512, 1024], 128),
        )
        self.assertEqual(resolve_eager_page_buckets(None, 128, max_pages=8), [8])
        self.assertIsNone(resolve_eager_page_buckets(None, 128))

    def test_prepare_view_widens_to_max_pages_with_dummy_cols(self):
        page, topk, steps = 128, 2, 5
        prefixes = [124]
        dummy = 99
        req = _req_to_token([[40]], page)
        pool = torch.tensor([0])
        branch_ids = [[[40, 77], [55, 88]]]
        slots = _draft_slots(prefixes, page, topk, steps, branch_ids)
        lens_before = build_step_context_lens(prefixes, topk, 0, topk)
        tables, _shared, _branch, _active, n_sh, n_q = prepare_tree_paged_view(
            req,
            pool,
            slots,
            prefixes,
            page,
            topk,
            steps,
            dummy_page=dummy,
            max_pages=4,
        )
        self.assertEqual(n_sh, [0])
        self.assertEqual(n_q, [1])
        self.assertEqual(int(tables.shape[1]), 4)
        self.assertTrue(torch.equal(tables[:, 0], torch.tensor([40, 55], dtype=torch.int32)))
        self.assertTrue(torch.equal(tables[:, 1:], torch.full((2, 3), dummy, dtype=torch.int32)))
        lens_after = build_step_context_lens(prefixes, topk, 0, topk)
        self.assertTrue(torch.equal(lens_before, lens_after))

    def test_tree_paged_shape_key_keeps_shared_and_branch_raw(self):
        page, topk, steps = 128, 3, 5
        buckets = kv_buckets_to_page_buckets([256, 512, 1024], page)
        self.assertEqual(sorted(buckets), [2, 4, 8])
        # Only the query width is bucketed: prefix 50 needs 1 page, snapped to 2.
        self.assertEqual(
            tree_paged_shape_key([200], page, topk, steps, buckets),
            (1, 1, 1, 2),
        )
        self.assertEqual(
            tree_paged_shape_key([50], page, topk, steps, buckets),
            (1, 0, 1, 2),
        )
        self.assertEqual(
            tree_paged_shape_key([50, 200], page, topk, steps, buckets),
            (2, 1, 1, 2),
        )
        self.assertEqual(
            tree_paged_shape_key([200], page, topk, steps, buckets)[2],
            pages_per_branch(remainder(200, page), steps, page),
        )
        self.assertEqual(
            tree_paged_shape_key([200], page, topk, steps, buckets, max_pages=8)[3], 8
        )

    def test_warmup_rem_ladder_covers_every_reachable_shape(self):
        page, topk = 128, 3
        buckets = kv_buckets_to_page_buckets([256, 512, 1024], page)
        for steps in (2, 5, 8):
            # Same ladder as _sr_warm_layout_shapes.
            rems = sorted({1, max(page - steps + 1, 1), max(page - 1, 1)})
            warm = {
                tree_paged_shape_key([s * page + r] * bs, page, topk, steps, buckets)
                for bs in (1, 2, 4)
                for s in range(max(buckets) + 1)
                for r in rems
            }
            for bs in (1, 2, 4):
                for prefix in range(1, max(buckets) * page):
                    key = tree_paged_shape_key(
                        [prefix] * bs, page, topk, steps, buckets
                    )
                    self.assertIn(key, warm, f"steps={steps} bs={bs} prefix={prefix}")

    def test_prepare_view_keeps_shared_width_data_dependent(self):
        page, topk, steps = 128, 2, 5
        prefixes = [200]
        dummy = 99
        req = _req_to_token([[11, 12]], page)
        pool = torch.tensor([0])
        branch_ids = [[[12, 30], [21, 31]]]
        slots = _draft_slots(prefixes, page, topk, steps, branch_ids)
        tables, shared, _branch, _act, n_sh, n_q = prepare_tree_paged_view(
            req,
            pool,
            slots,
            prefixes,
            page,
            topk,
            steps,
            dummy_page=dummy,
            max_pages=4,
        )
        # shared/branch stay at the exact page counts; only the block table is
        # widened to the quantized query width.
        self.assertEqual(int(shared.shape[1]), shared_page_count(200, page))
        self.assertEqual(n_sh, [1])
        self.assertEqual(n_q, [1])
        self.assertEqual(int(tables.shape[1]), 4)

    def test_prepare_view_keeps_empty_shared_when_prefix_below_page(self):
        page, topk, steps = 128, 2, 5
        prefixes = [50]
        dummy = 7
        req = _req_to_token([[40]], page)
        pool = torch.tensor([0])
        branch_ids = [[[40, 77], [55, 88]]]
        slots = _draft_slots(prefixes, page, topk, steps, branch_ids)
        _tables, shared, _branch, _active, n_sh, _n_q = prepare_tree_paged_view(
            req,
            pool,
            slots,
            prefixes,
            page,
            topk,
            steps,
            dummy_page=dummy,
        )
        self.assertEqual(n_sh, [0])
        self.assertEqual(tuple(shared.shape), (1, 0))


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
        self.assertIn("quantize_page_width", prep_src)
        self.assertIn("resolve_eager_page_buckets", prep_src)
        self.assertIn("max_pages=max_pages", prep_src)
        self.assertIn("metrics=None", prep_src)
        self.assertIn('metrics.add_host("tree_paged_view"', prep_src)
        self.assertIn('metrics.add_host("tree_paged_copy"', prep_src)
        self.assertIn('metrics.add_host("tree_paged_bind"', prep_src)
        clear_src = _fn_source(_BACKEND, "_sr_clear_paged_round_state")
        self.assertIn("_paged_round_tables = None", clear_src)
        self.assertIn("_sr_tree_paged_meta = None", clear_src)
        self.assertIn("metadata.sr_tree_paged = None", clear_src)
        self.assertNotIn("cuda_graph_paged_block_tables", prep_src)
        self.assertNotIn("cuda_graph_paged_active", prep_src)
        self.assertNotIn("exceed graph buffer", prep_src)
        view_src = _fn_source(_BACKEND, "_paged_graph_table_view")
        self.assertIn("NpuGraphPreparationError", view_src)
        self.assertNotIn("RuntimeError", view_src)
        self.assertIn("allow_alloc", view_src)
        self.assertIn("is_contiguous", view_src)
        self.assertIn("cuda_graph_paged_tables", view_src)
        self.assertIn("actives_map.get(key)", view_src)
        self.assertNotIn("actives_map.get(rows)", view_src)
        self.assertNotIn("[:rows, :pages]", view_src)
        self.assertNotIn("cuda_graph_paged_block_tables", view_src)
        init_graph_src = _fn_source(_BACKEND, "init_cuda_graph_state")
        self.assertIn("cuda_graph_paged_tables", init_graph_src)
        self.assertIn("_paged_graph_max_pages", init_graph_src)
        self.assertNotIn("cuda_graph_paged_block_tables", init_graph_src)
        can_src = _fn_source(_BACKEND, "tree_slot_graph_can_run")
        self.assertIn("_paged_graph_max_pages", can_src)
        self.assertNotIn("cuda_graph_paged_block_tables", can_src)
        runner_src = _source(_RUNNER)
        self.assertIn("_paged_graph_max_pages", runner_src)
        self.assertNotIn("cuda_graph_paged_block_tables", runner_src)

    def test_graph_runner_paged_serial_and_validator(self):
        src = _source(_RUNNER)
        self.assertIn("validate_tree_draft_paged_records", src)
        self.assertIn("_tree_paged", src)
        self.assertIn("bind_sr_tree_paged_capture", src)
        self.assertIn("bind_sr_tree_paged_replay", src)
        self.assertIn("fill_paged_cpu_update_payload", src)
        self.assertIn("if getattr(self, \"_tree_paged\", False)", src)
        self.assertIn("SGLANG_NPU_TREE_FIA_SERIAL_UPDATE", src)
        self.assertIn("sr_paged_overlap", src)
        self.assertIn("_npu_sr_tree_update_overlap", src)
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
        self.assertIn("record_tree_expand_admission", src)
        self.assertIn("tree_eager_prep_failed", src)
        self.assertIn("_tree_forward_calls", src)
        self.assertIn("reason=%s", src)
        self.assertIn("tree_make_batch", src)
        self.assertIn("tree_alloc_kv", src)
        self.assertIn("tree_prepare_meta", src)
        self.assertIn("tree_init_forward_batch", src)
        self.assertIn("tree_paged_eager", src)
        self.assertIn("tree_graph_key_first_use", src)
        prep_round_src = _fn_source(_DRAFTER, "_prepare_paged_tree_round")
        self.assertIn("tree_paged_shape_key", prep_round_src)
        self.assertIn("tree_shape_first_use", prep_round_src)
        self.assertIn("metrics=metrics", prep_round_src)
        warm_src = _fn_source(_DRAFTER, "_sr_warm_tree_shapes")
        self.assertIn("read_sr_tree_warmup_env", warm_src)
        self.assertIn("NpuGraphReplaySubmittedError", warm_src)
        self.assertIn("SRWarmupFatalError", warm_src)
        self.assertIn("is_device_context_error", warm_src)
        self.assertIn("_sr_warm_layout_shapes", warm_src)
        self.assertIn("warm_draft_alloc_mapping", warm_src)
        self.assertIn("_seen_tree_paged_shapes", warm_src)
        self.assertIn("tree warmup layout=", warm_src)
        self.assertNotIn("prepare_sr_tree_paged_eager", warm_src)
        layout_src = _fn_source(_DRAFTER, "_sr_warm_layout_shapes")
        self.assertIn("prepare_sr_tree_paged_eager", layout_src)
        self.assertIn("ALLOC_LEASE", layout_src)
        self.assertIn("ALLOC_ORDINARY", layout_src)
        self.assertIn("torch.arange", layout_src)
        self.assertIn("_sr_clear_paged_round_state", layout_src)
        self.assertIn("for shared in range(max_shared + 1)", layout_src)
        self.assertIn("for rem in rem_choices", layout_src)
        self.assertIn("max(page - steps + 1, 1)", layout_src)
        self.assertNotIn("prepare_tree_paged_view", layout_src)
        builder_src = _fn_source(_DRAFTER, "_sr_warm_layout_shapes_builders")
        self.assertIn("prepare_tree_paged_view", builder_src)
        init_src = _fn_source(_DRAFTER, "_init_cuda_graphs")
        self.assertIn("_sr_warm_tree_shapes", init_src)
        self.assertGreater(init_src.rfind("_sr_warm_tree_shapes"), init_src.rfind("_init_tail_graphs"))
        can_src = _fn_source(_DRAFTER, "_can_run_tree_graph")
        self.assertIn("not getattr(runner, \"_tree_paged\", False)", can_src)
        batch_src = _fn_source(_DRAFTER, "expand_batch")
        self.assertIn("except NpuGraphReplaySubmittedError:\n            raise", batch_src)
        one_src = _fn_source(_DRAFTER, "_expand_one")
        self.assertIn("except NpuGraphReplaySubmittedError:\n            raise", one_src)
        ctor_src = _fn_source(_DRAFTER, "__init__")
        self.assertIn("read_sr_tree_update_overlap_env", ctor_src)
        self.assertLess(
            ctor_src.find("npu_sr_tree_update_overlap_requested"),
            ctor_src.find("_init_cuda_graphs"),
        )
        expand_src = _fn_source(_DRAFTER, "_expand_tree")
        self.assertIn("_try_confirm_tree_completion", expand_src)
        confirm_src = _fn_source(_DRAFTER, "_try_confirm_tree_completion")
        self.assertIn("synchronize()", confirm_src)

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
            [
                "_paged_graph_table_view",
                "_validate_sr_tree_paged_replay",
                "bind_sr_tree_paged_replay",
            ],
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
        self._view_fn = fn["_paged_graph_table_view"]

    def _inner(self, step, dest, dest_act, eager_meta=None, had_fm=True):
        rows = int(dest.shape[0])
        pages = int(dest.shape[1])
        inner = SimpleNamespace(
            speculative_step_id=step,
            tree_attention_impl="paged_atb",
            device=dest.device,
            cuda_graph_paged_tables={(rows, pages): dest},
            cuda_graph_paged_actives={(rows, pages): dest_act},
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
        backend._paged_graph_table_view = MethodType(self._view_fn, backend)
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
        inners[0].cuda_graph_paged_tables = None
        inners[0].cuda_graph_paged_actives = None
        with self.assertRaises(self.PrepError):
            backend.bind_sr_tree_paged_replay(capture_bs, max_pages)
        self.assertIsNone(inners[0].cuda_graph_paged_tables)
        self.assertIsNone(inners[0].cuda_graph_paged_actives)

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
                key = (int(dest_holder["dest"].shape[0]), int(dest_holder["dest"].shape[1]))
                for inner in inners:
                    inner.cuda_graph_paged_tables = {key: dest_holder["dest"]}
                    inner.cuda_graph_paged_actives = {key: dest_holder["dest_act"]}
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


class TestPagedEagerPrep(CustomTestCase):
    def setUp(self):
        fn = _extract_class_methods(
            _BACKEND,
            "AscendAttnMultiStepDraftBackend",
            ["prepare_sr_tree_paged_eager"],
            dict(
                time=time,
                torch=torch,
                prepare_tree_paged_view=prepare_tree_paged_view,
                build_step_context_lens=build_step_context_lens,
                context_lens_list=context_lens_list,
                max_query_pages_for_tree=max_query_pages_for_tree,
                resolve_eager_page_buckets=resolve_eager_page_buckets,
                quantize_page_width=quantize_page_width,
                SRTreePagedMetadata=SRTreePagedMetadata,
                ForwardMetadata=_ForwardMetadata,
            ),
        )
        self._prep_fn = fn["prepare_sr_tree_paged_eager"]

    def _backend(self, raw_bs, dest_bs, topk=2, dest_pages=4, page_size=128, steps=3, dummy=99):
        dest = torch.full((dest_bs * topk, dest_pages), 777, dtype=torch.int32)
        dest_act = torch.full((dest_bs * topk,), 9, dtype=torch.int32)
        prefixes = [page_size] * raw_bs
        req = _req_to_token([[1]] * raw_bs, page_size)
        pool = torch.arange(raw_bs)
        branch_ids = [[[2] for _ in range(topk)] for _ in range(raw_bs)]
        slots = _draft_slots(prefixes, page_size, topk, steps, branch_ids)
        compact = slots.reshape(-1)
        inners = []
        for step in (0, 1):
            inner = SimpleNamespace(
                speculative_step_id=step,
                tree_attention_impl="paged_atb",
                cuda_graph_paged_block_tables=dest,
                cuda_graph_paged_active=dest_act,
                req_to_token=req,
                tree_kv_buckets=[],
                _paged_graph_max_pages=None,
                _sr_tree_paged_prep_count=0,
                _sr_tree_paged_copy_count=0,
                _sr_tree_paged_meta=None,
                forward_metadata=_ForwardMetadata(),
            )
            inner.bind_sr_tree_paged_metadata = MethodType(
                lambda self, meta: setattr(self, "_sr_tree_paged_meta", meta),
                inner,
            )
            inners.append(inner)
        backend = SimpleNamespace(
            topk=topk,
            page_size=page_size,
            speculative_num_steps=steps,
            attn_backends=inners,
            _paged_prep_count=0,
            _paged_copy_count=0,
            paged_impl_selected=lambda: True,
        )
        backend.prepare_sr_tree_paged_eager = MethodType(self._prep_fn, backend)
        forward_batch = SimpleNamespace(batch_size=raw_bs, req_pool_indices=pool)
        prefix = torch.tensor(prefixes, dtype=torch.int32)
        return backend, dest, dest_act, inners, forward_batch, compact, prefix, dummy

    def _assert_no_dest_writes(self, backend, dest, dest_act, forward_batch, compact, prefix, dummy):
        writes = []
        orig_fill = torch.Tensor.fill_
        orig_copy = torch.Tensor.copy_
        dest_ptrs = {dest.data_ptr(), dest_act.data_ptr()}

        def watch_fill(tensor, value):
            if tensor.data_ptr() in dest_ptrs:
                writes.append("fill")
            return orig_fill(tensor, value)

        def watch_copy(tensor, src_t, *args, **kwargs):
            if tensor.data_ptr() in dest_ptrs:
                writes.append("copy")
            return orig_copy(tensor, src_t, *args, **kwargs)

        with mock.patch.object(torch.Tensor, "fill_", watch_fill), mock.patch.object(
            torch.Tensor, "copy_", watch_copy
        ):
            copied = backend.prepare_sr_tree_paged_eager(
                forward_batch,
                compact,
                prefix,
                ALLOC_ORDINARY,
                kv_pool=None,
                dummy_page=dummy,
            )
        self.assertFalse(copied)
        self.assertEqual(writes, [])
        self.assertTrue(torch.equal(dest, torch.full_like(dest, 777)))
        self.assertTrue(torch.equal(dest_act, torch.full_like(dest_act, 9)))
        return backend.attn_backends

    def test_eager_over_graph_capacity_does_not_write_or_raise(self):
        topk = 2
        backend, dest, dest_act, _, forward_batch, compact, prefix, dummy = self._backend(
            raw_bs=4, dest_bs=2, topk=topk
        )
        inners = self._assert_no_dest_writes(
            backend, dest, dest_act, forward_batch, compact, prefix, dummy
        )
        need = 4 * topk
        self.assertEqual(int(backend._paged_round_tables.shape[0]), need)
        self.assertIs(inners[0]._sr_tree_paged_meta.block_tables, backend._paged_round_tables)
        self.assertEqual(int(inners[0]._sr_tree_paged_meta.block_tables.shape[0]), need)
        self.assertEqual(int(inners[0].forward_metadata.block_tables.shape[0]), need)

    def test_eager_fitting_capacity_still_skips_graph_buffers(self):
        topk = 2
        backend, dest, dest_act, _, forward_batch, compact, prefix, dummy = self._backend(
            raw_bs=2, dest_bs=2, topk=topk
        )
        inners = self._assert_no_dest_writes(
            backend, dest, dest_act, forward_batch, compact, prefix, dummy
        )
        need = 2 * topk
        self.assertEqual(int(backend._paged_round_tables.shape[0]), need)
        self.assertIs(inners[0]._sr_tree_paged_meta.block_tables, backend._paged_round_tables)

    def test_eager_width_snaps_to_graph_bucket_dummy_extra_cols(self):
        page, topk, steps = 128, 2, 3
        dummy = 99
        prefixes = [256]
        req = _req_to_token([[1, 2]], page)
        pool = torch.arange(1)
        branch_ids = [[[2, 3], [4, 5]]]
        slots = _draft_slots(prefixes, page, topk, steps, branch_ids)
        compact = slots.reshape(-1)
        dest = torch.full((2, 8), 777, dtype=torch.int32)
        dest_act = torch.full((2,), 9, dtype=torch.int32)
        inners = []
        for step in (0, 1):
            inner = SimpleNamespace(
                speculative_step_id=step,
                tree_attention_impl="paged_atb",
                req_to_token=req,
                tree_kv_buckets=[256, 512, 1024],
                _paged_graph_max_pages=8,
                _sr_tree_paged_prep_count=0,
                _sr_tree_paged_copy_count=0,
                _sr_tree_paged_meta=None,
                forward_metadata=_ForwardMetadata(),
            )
            inner.bind_sr_tree_paged_metadata = MethodType(
                lambda self, meta: setattr(self, "_sr_tree_paged_meta", meta),
                inner,
            )
            inners.append(inner)
        backend = SimpleNamespace(
            topk=topk,
            page_size=page,
            speculative_num_steps=steps,
            attn_backends=inners,
            _paged_prep_count=0,
            _paged_copy_count=0,
            paged_impl_selected=lambda: True,
        )
        backend.prepare_sr_tree_paged_eager = MethodType(self._prep_fn, backend)
        needed = max(max_query_pages_for_tree(prefixes, steps, page) or [1])
        self.assertEqual(needed, 3)
        lens_before = build_step_context_lens(prefixes, topk, 0, topk)
        backend.prepare_sr_tree_paged_eager(
            SimpleNamespace(batch_size=1, req_pool_indices=pool),
            compact,
            torch.tensor(prefixes, dtype=torch.int32),
            ALLOC_ORDINARY,
            kv_pool=None,
            dummy_page=dummy,
        )
        tables = backend._paged_round_tables
        self.assertEqual(int(tables.shape[1]), 4)
        self.assertTrue(torch.equal(tables[:, 3], torch.full((2,), dummy, dtype=torch.int32)))
        self.assertEqual(int(inners[0]._sr_tree_paged_meta.max_pages), 4)
        self.assertTrue(
            torch.equal(
                inners[0]._sr_tree_paged_meta.context_lens_cpu,
                lens_before,
            )
        )
        self.assertTrue(torch.equal(dest, torch.full_like(dest, 777)))
        self.assertTrue(torch.equal(dest_act, torch.full_like(dest_act, 9)))


class TestPagedGraphTableBuffers(CustomTestCase):
    def setUp(self):
        self.PrepError = _load_npu_graph_prep_error()
        fn = _extract_class_methods(
            _BACKEND,
            "AscendAttnMultiStepDraftBackend",
            ["_paged_graph_table_view"],
            dict(torch=torch, NpuGraphPreparationError=self.PrepError),
        )
        self._view_fn = fn["_paged_graph_table_view"]

    def _backend(self, topk=3):
        inner = SimpleNamespace(
            device=torch.device("cpu"),
            cuda_graph_paged_tables={},
            cuda_graph_paged_actives={},
        )
        backend = SimpleNamespace(topk=topk, attn_backends=[inner])
        backend._paged_graph_table_view = MethodType(self._view_fn, backend)
        return backend, inner

    def test_capture_allocates_independent_contiguous_buffers(self):
        backend, inner = self._backend()
        seen = []
        for rows, pages in ((3, 2), (3, 4), (6, 4)):
            tables, active = backend._paged_graph_table_view(
                rows // 3, pages, 0, allow_alloc=True
            )
            self.assertTrue(tables.is_contiguous())
            self.assertEqual(tuple(tables.stride()), (pages, 1))
            self.assertEqual(tuple(tables.shape), (rows, pages))
            self.assertEqual(int(active.numel()), rows)
            seen.append(tables.data_ptr())
        self.assertEqual(len(set(seen)), 3)
        again, _ = backend._paged_graph_table_view(1, 4, 0, allow_alloc=True)
        self.assertEqual(again.data_ptr(), seen[1])
        self.assertIs(again, inner.cuda_graph_paged_tables[(3, 4)])
        self.assertEqual(
            set(inner.cuda_graph_paged_tables), set(inner.cuda_graph_paged_actives)
        )
        act_s2 = inner.cuda_graph_paged_actives[(3, 2)]
        act_s4 = inner.cuda_graph_paged_actives[(3, 4)]
        self.assertIsNot(act_s2, act_s4)
        self.assertNotEqual(act_s2.data_ptr(), act_s4.data_ptr())

    def test_replay_missing_key_does_not_allocate(self):
        backend, inner = self._backend()
        with self.assertRaises(self.PrepError) as ctx:
            backend._paged_graph_table_view(1, 4, 0, allow_alloc=False)
        self.assertIn("missing", str(ctx.exception))
        self.assertEqual(inner.cuda_graph_paged_tables, {})
        self.assertEqual(inner.cuda_graph_paged_actives, {})

    def test_strided_stored_buffer_is_rejected(self):
        backend, inner = self._backend()
        shared = torch.zeros((6, 8), dtype=torch.int32)
        inner.cuda_graph_paged_tables[(3, 4)] = shared[:3, :4]
        inner.cuda_graph_paged_actives[(3, 4)] = torch.zeros((3,), dtype=torch.bool)
        with self.assertRaises(self.PrepError) as ctx:
            backend._paged_graph_table_view(1, 4, 0, allow_alloc=False)
        self.assertIn("contiguous", str(ctx.exception))


class TestSRWarmup(CustomTestCase):
    def test_raw_batch_sizes_include_three(self):
        from sglang.srt.speculative.standalone_remote.sr_warmup import (
            sr_warmup_raw_batch_sizes,
        )

        sizes, skip = sr_warmup_raw_batch_sizes([1, 2, 4], 16, 8, None)
        self.assertIsNone(skip)
        self.assertEqual(sizes, (1, 2, 3, 4))
        empty, reason = sr_warmup_raw_batch_sizes([], 16, 8, None)
        self.assertEqual(empty, ())
        self.assertIn("capture", reason)

    def test_steps_over_page_jump_sides(self):
        from sglang.srt.speculative.standalone_remote.sr_warmup import (
            warmup_nnp_jump_reachable,
            warmup_prefix_remainders,
        )

        rems = warmup_prefix_remainders(128, 133)
        self.assertIn(123, rems)
        self.assertIn(124, rems)
        self.assertEqual(pages_per_branch(123, 133, 128), 2)
        self.assertEqual(pages_per_branch(124, 133, 128), 3)
        self.assertTrue(warmup_nnp_jump_reachable(128, 133))
        self.assertFalse(warmup_nnp_jump_reachable(128, 129))

    def test_steps_le_page_prefixes(self):
        from sglang.srt.speculative.standalone_remote.sr_warmup import (
            warmup_prefix_candidates,
        )

        cands = warmup_prefix_candidates(128, 5, 4096, 2)
        self.assertEqual(cands, [128, 129, 252, 255])

    def test_alloc_extend_does_not_reown_prefix_pages(self):
        from sglang.srt.speculative.standalone_remote.sr_align import SRWarmupFatalError
        from sglang.srt.speculative.standalone_remote.sr_warmup import (
            SRWarmupTrackingAllocator,
            page_set,
            slots_to_pages,
        )

        class Inner:
            page_size = 4
            device = "cpu"
            evict_calls = 0

            def __init__(self):
                self.free_pages = torch.arange(1, 9, dtype=torch.int64)
                self.release_pages = torch.empty((0,), dtype=torch.int64)
                self.is_not_in_free_group = True
                self.free_group = []

            def available_size(self):
                return int(self.free_pages.numel()) * self.page_size

            def alloc(self, need):
                n = need // self.page_size
                pages = self.free_pages[:n]
                self.free_pages = self.free_pages[n:]
                return (
                    pages.unsqueeze(1) * self.page_size
                    + torch.arange(self.page_size)
                ).reshape(-1)

            def alloc_extend(self, *args, **kwargs):
                prefix = args[-1] if args else kwargs.get("last_loc")
                extra = self.alloc(self.page_size)
                return torch.cat([prefix.reshape(-1)[:1], extra])

            def free(self, idx):
                if torch.is_tensor(idx) and int(idx.numel()) == 0:
                    return
                pages = torch.unique(idx // self.page_size)
                self.free_pages = torch.cat([self.free_pages, pages])

            def backup_state(self):
                return (self.free_pages.clone(), self.release_pages.clone())

            def restore_state(self, state):
                self.free_pages, self.release_pages = state

        inner = Inner()
        before = page_set(inner)
        adapter = SRWarmupTrackingAllocator(inner)
        prefix = adapter.alloc(4)
        prefix_pages = set(adapter.owned_pages)
        extra = adapter.alloc_extend(prefix)
        self.assertTrue(prefix_pages <= adapter.owned_pages)
        self.assertEqual(len(adapter.owned_pages), len(prefix_pages) + 1)
        backed = adapter.backup_state()
        more = adapter.alloc(4)
        self.assertTrue(adapter.owned_pages > prefix_pages)
        adapter.restore_state(backed)
        self.assertEqual(adapter.owned_pages, prefix_pages | slots_to_pages(extra, 4))
        adapter.free(extra[1:])
        adapter.free(more[:0])
        self.assertEqual(adapter.owned_pages, prefix_pages)
        empty = torch.empty((0,), dtype=torch.int64)
        before_empty = page_set(inner)
        adapter.free(empty)
        self.assertEqual(page_set(inner), before_empty)
        adapter.free(prefix)
        self.assertEqual(adapter.owned_pages, set())
        self.assertEqual(page_set(inner), before)
        with self.assertRaises(SRWarmupFatalError):
            adapter.free(prefix)
        with self.assertRaises(SRWarmupFatalError):
            adapter.restore_state(inner.backup_state())

    def test_cache_adapter_never_evicts(self):
        from sglang.srt.speculative.standalone_remote.sr_warmup import (
            SRWarmupCacheAdapter,
        )

        calls = []

        class Prod:
            def evict(self, *a, **k):
                calls.append(1)

        cache = SRWarmupCacheAdapter(SimpleNamespace())
        cache.evict("anything")
        self.assertFalse(cache.is_chunk_cache())
        self.assertEqual(calls, [])
        Prod().evict()
        self.assertEqual(calls, [1])

    def test_outer_fatal_not_swallowed(self):
        import logging
        import time
        from types import MethodType

        from sglang.srt.speculative.standalone_remote.sr_align import SRWarmupFatalError

        src_path = _DRAFTER
        tree = ast.parse(src_path.read_text())
        fn = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "SRTreeDrafter":
                for child in node.body:
                    if (
                        isinstance(child, ast.FunctionDef)
                        and child.name == "_sr_warm_tree_shapes"
                    ):
                        fn = child
        self.assertIsNotNone(fn)
        ns = {
            "read_sr_tree_warmup_env": lambda: True,
            "SR_TREE_WARMUP_ENV": "SGLANG_NPU_SR_TREE_WARMUP",
            "warm_draft_alloc_mapping": lambda _d: (_ for _ in ()).throw(
                SRWarmupFatalError("ledger unknown")
            ),
            "NpuGraphReplaySubmittedError": type(
                "NpuGraphReplaySubmittedError", (Exception,), {}
            ),
            "SRWarmupFatalError": SRWarmupFatalError,
            "is_device_context_error": lambda _e: False,
            "logger": logging.getLogger("sr-warmup-fatal-test"),
            "time": time,
        }
        ast.fix_missing_locations(fn)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(src_path), "exec"), ns)

        class Dummy:
            sr_tree_paged = True
            _seen_tree_paged_shapes = None

            def _sr_warm_layout_shapes(self):
                return [(1, 0, 1, 2)]

        dummy = Dummy()
        dummy._sr_warm_tree_shapes = MethodType(ns["_sr_warm_tree_shapes"], dummy)
        with self.assertRaises(SRWarmupFatalError):
            dummy._sr_warm_tree_shapes()

    def test_unreachable_jump_is_not_failure(self):
        from sglang.srt.speculative.standalone_remote.sr_warmup import (
            warmup_nnp_jump_reachable,
        )

        self.assertFalse(warmup_nnp_jump_reachable(128, 129))
        src = _fn_source(
            _REPO
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_tree_warmup.py",
            "warm_draft_alloc_mapping",
        )
        self.assertIn('coverage["unreachable_jump"] = True', src)
        self.assertNotIn('coverage["skipped"] = "unreachable', src)


if __name__ == "__main__":
    unittest.main()
