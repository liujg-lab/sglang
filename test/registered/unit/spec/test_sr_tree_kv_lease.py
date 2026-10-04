"""CPU contracts for page-level SR tree KV leases."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
    IdentityRemapSubmittedError,
    IdentityRemapWorkspace,
    LEASE_PREFIX_WINDOW,
    MIN_TREE_KV_REUSE_DEPTH,
    SRAlignResult,
    SRTreeKVLease,
    SRTreeLeaseStore,
    build_tree_raw_slots,
    first_forward_node_ids,
    later_forward_node_ids,
    lease_budget_ok,
    lease_page_count,
    live_accept_prefix,
    lookup_candidate_slots,
    page_ids_from_alloc,
    pages_per_branch,
    physical_tree_slot,
    plan_paged_tree_layout,
    prefix_window_tokens,
    record_device_event,
    remap_slot_node_ids,
    prefix_window_from_committed,
    snapshot_sr_align,
    tree_raw_span_len,
    validate_lease_commit,
)
from sglang.srt.speculative.standalone_remote.drafter.sr_tree_reply_pack import (
    pack_tree_reply,
    plan_tree_reply,
    warm_tree_reply_pack,
)
from sglang.srt.speculative.standalone_remote.sr_kv_copy import KVMoveSubmittedError
from sglang.srt.speculative.standalone_remote.sr_protocol import (
    SRDraftReply,
    SRDraftRequest,
)
from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
    export_accepted_tree_candidate_indices,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="stage-a-test-cpu")


class DummyReq:
    def __init__(self, tokens, kv_len, revision=0):
        self.origin_input_ids = list(tokens)
        self.output_ids = []
        self.kv_committed_len = kv_len
        self.sr_prefix_revision = revision
        self.sr_padded_ids = None


class DummyDreq:
    def __init__(self, committed):
        self.committed_ids = list(committed)


class TestPagedTreeLayout(unittest.TestCase):
    def test_pages_for_example_r(self):
        topk, steps, page = 3, 5, 128
        cases = {
            0: (1, 3, 384),
            123: (1, 3, 261),
            124: (2, 6, 644),
            127: (2, 6, 641),
        }
        for r, (nnp, nreq, span) in cases.items():
            self.assertEqual(pages_per_branch(r, steps, page), nnp)
            layout = plan_paged_tree_layout([128 + r], page, topk, steps)
            self.assertEqual(layout.pages_per_req[0], nreq)
            self.assertEqual(tree_raw_span_len(r, topk, steps, page), span)

    def test_ragged_batch_prefix_sum(self):
        layout = plan_paged_tree_layout([123, 124, 127], 128, 3, 5)
        self.assertEqual(layout.pages_per_req, [3, 6, 6])
        self.assertEqual(sum(layout.pages_per_req), 15)

    def test_raw_slots_cover_span_not_just_nodes(self):
        page_ids = [10, 3, 7]
        r = 123
        slots = build_tree_raw_slots(page_ids, r, 3, 5, 128)
        self.assertEqual(slots.numel(), 261)
        self.assertEqual(int(slots[0]), 10 * 128 + 123)
        last_branch_start = physical_tree_slot(page_ids, 2, 0, r, 1, 128)
        self.assertIn(last_branch_start, slots.tolist())

    def test_cross_page_formula(self):
        page_ids = [1, 9, 2, 8, 3, 7]
        r, nnp, page = 124, 2, 128
        slot = physical_tree_slot(page_ids, 0, 5, r, nnp, page)
        self.assertEqual(slot, 9 * 128 + (124 + 5) % 128)
        naive = page_ids[0] * 128 + r + 5
        self.assertNotEqual(slot, naive)

    def test_vectorized_raw_slots_match_scalar(self):
        cases = [([10, 3, 7], 123, 3, 5, 128), ([1, 9, 2, 8, 3, 7], 124, 3, 5, 128)]
        for page_ids, r, topk, steps, page in cases:
            got = build_tree_raw_slots(page_ids, r, topk, steps, page)
            nnp = pages_per_branch(r, steps, page)
            span = tree_raw_span_len(r, topk, steps, page)
            expected = []
            for col in range(span):
                abs_off = r + col
                branch = abs_off // (nnp * page)
                within = abs_off % (nnp * page)
                page_index = branch * nnp + within // page
                expected.append(int(page_ids[page_index]) * page + within % page)
            self.assertEqual(got.tolist(), expected)

    def test_contiguous_alloc_slice_matches_helper(self):
        page_ids = [10, 3, 7]
        ps, r, topk, steps = 128, 123, 3, 5
        allocated = torch.cat(
            [
                torch.arange(p * ps, p * ps + ps, dtype=torch.int64)
                for p in page_ids
            ]
        )
        self.assertEqual(page_ids_from_alloc(allocated, ps), page_ids)
        helper = build_tree_raw_slots(
            page_ids_from_alloc(allocated, ps), r, topk, steps, ps
        )
        self.assertEqual(helper.tolist(), allocated[r:].tolist())


class TestIdentityAndLookup(unittest.TestCase):
    def test_parent_remap_overwrites_and_lookup(self):
        topk, steps, bs = 3, 4, 1
        ids = torch.full((steps, bs * topk), -1, dtype=torch.int64)
        phys = torch.arange(steps * bs * topk, dtype=torch.int64).reshape(steps, bs * topk)
        tmp = torch.empty((steps, bs * topk), dtype=torch.int64)
        ids[0].copy_(first_forward_node_ids(topk, bs))
        parent_rows = torch.tensor([1, 1, 2])
        remap_slot_node_ids(ids, parent_rows, 1, tmp)
        self.assertEqual(ids[0].tolist(), [1, 1, 2])
        ids[1].copy_(later_forward_node_ids(torch.tensor([[4, 5, 6]])))
        cands = torch.tensor([[0, 1, 2, 4]])
        slots = lookup_candidate_slots(ids, phys, cands, bs, topk)
        # node 0 was overwritten; 1 and 2 survive in copies; 4 is live
        self.assertEqual(int(slots[0, 0]), -1)
        self.assertGreaterEqual(int(slots[0, 1]), 0)
        self.assertEqual(int(slots[0, 3]), int(phys[1, 0]))

    def test_tagged_kv_after_real_remap(self):
        topk, steps = 3, 3
        kv = torch.full((steps, topk), -1.0)
        ids = torch.full((steps, topk), -1, dtype=torch.int64)
        tmp = torch.empty((steps, topk), dtype=torch.int64)
        ids[0] = torch.arange(topk)
        kv[0] = torch.tensor([100.0, 101.0, 102.0])
        parent_rows = torch.tensor([1, 1, 2])
        src = kv[0].index_select(0, parent_rows)
        kv[0].copy_(src)
        remap_slot_node_ids(ids, parent_rows, 1, tmp)
        phys = torch.arange(steps * topk).reshape(steps, topk)
        slots = lookup_candidate_slots(
            ids, phys, torch.tensor([[0, 1]]), 1, topk
        )
        self.assertEqual(int(slots[0, 0]), -1)
        live = int(slots[0, 1])
        copied = float(kv.reshape(-1)[live])
        self.assertEqual(copied, 101.0)

    def test_unforwarded_is_minus_one(self):
        ids = torch.full((5, 3), -1, dtype=torch.int64)
        ids[0] = torch.arange(3)
        phys = torch.arange(15).reshape(5, 3)
        # last-layer candidate 20 never written
        slots = lookup_candidate_slots(ids, phys, torch.tensor([[20]]), 1, 3)
        self.assertEqual(int(slots[0, 0]), -1)

    def test_lookup_stays_in_request_row(self):
        ids = torch.full((2, 6), -1, dtype=torch.int64)
        ids[0, 0:3] = torch.tensor([0, 1, 2])
        ids[0, 3:6] = torch.tensor([0, 1, 2])
        phys = torch.arange(12).reshape(2, 6)
        slots = lookup_candidate_slots(ids, phys, torch.tensor([[0], [0]]), 2, 3)
        self.assertEqual(int(slots[0, 0]), 0)
        self.assertEqual(int(slots[1, 0]), 3)

    def test_lookup_keeps_first_step_then_branch(self):
        # Same id is overwritten later in the scan. The earliest step/branch wins.
        ids = torch.tensor(
            [
                [4, 4],
                [4, 9],
            ],
            dtype=torch.int64,
        )
        phys = torch.tensor(
            [
                [10, 11],
                [20, 21],
            ],
            dtype=torch.int64,
        )
        slots = lookup_candidate_slots(ids, phys, torch.tensor([[4, 9, 7, -1]]), 1, 2)
        self.assertEqual(slots.tolist(), [[10, 21, -1, -1]])

    def test_lookup_matches_minus_one_node_id(self):
        ids = torch.full((1, 2), -1, dtype=torch.int64)
        ids[0, 1] = 3
        phys = torch.tensor([[8, 9]], dtype=torch.int64)
        slots = lookup_candidate_slots(ids, phys, torch.tensor([[-1, 3]]), 1, 2)
        self.assertEqual(slots.tolist(), [[8, 9]])

    def test_identity_reset_clears_padding(self):
        ids = torch.arange(8, dtype=torch.int64).reshape(2, 4)
        ids.fill_(-1)
        self.assertTrue(bool((ids == -1).all()))


def _expected_reply(tokens, parents, indices, node_ids, compact, batch, topk, steps, write_slots):
    parts = []
    for tensor in (tokens, parents, indices):
        if int(tensor.numel()) == 0:
            continue
        parts.append(tensor.detach().reshape(-1).to(dtype=torch.int64).cpu())
    if write_slots and int(indices.numel()) > 0:
        phys = (
            compact.detach()
            .reshape(int(batch), int(topk), int(steps))
            .permute(2, 0, 1)
            .reshape(int(steps), -1)
            .cpu()
        )
        slots = lookup_candidate_slots(
            node_ids.detach().cpu(), phys, indices.detach().cpu(), batch, topk
        )
        parts.append(slots.reshape(-1))
    if not parts:
        return torch.empty(0, dtype=torch.int64)
    return torch.cat(parts)


class TestTreeReplyPack(unittest.TestCase):
    def _assert_packed(self, tokens, parents, indices, node_ids, compact, batch, topk, steps, write_slots):
        segments, needed = plan_tree_reply(tokens, parents, indices, write_slots)
        buf = torch.empty(max(needed, 1), dtype=torch.int64)
        copies = {"n": 0}
        orig = torch.Tensor.copy_

        def counting_copy(self, src, *args, **kwargs):
            copies["n"] += 1
            return orig(self, src, *args, **kwargs)

        with patch.object(torch.Tensor, "copy_", counting_copy):
            pack_tree_reply(
                buf,
                tokens,
                parents,
                indices,
                node_ids,
                compact,
                batch=batch,
                topk=topk,
                steps=steps,
                write_slots=write_slots,
            )
        self.assertEqual(copies["n"], 0)
        expected = _expected_reply(
            tokens, parents, indices, node_ids, compact, batch, topk, steps, write_slots
        )
        self.assertEqual(buf[:needed].tolist(), expected.tolist())
        self.assertEqual(
            [name for name, _shape, _offset, _length in segments],
            self._segment_names(tokens, parents, indices, write_slots),
        )

    def _segment_names(self, tokens, parents, indices, write_slots):
        names = []
        for name, tensor in (
            ("tokens", tokens),
            ("parents", parents),
            ("indices", indices),
        ):
            if int(tensor.numel()) > 0:
                names.append(name)
        if write_slots and int(indices.numel()) > 0:
            names.append("candidate_slots")
        return names

    def test_pack_without_slots_skips_empty_parents(self):
        tokens = torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int32)
        parents = torch.empty((2, 0), dtype=torch.int64)
        indices = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64)
        self._assert_packed(tokens, parents, indices, None, None, 2, 1, 1, False)

    def test_pack_matches_oracle_for_duplicates_and_isolation(self):
        batch, topk, steps = 2, 2, 2
        tokens = torch.arange(12, dtype=torch.int64).view(3, 4)[:batch, ::2]
        parents = torch.tensor([[-1, 0], [-1, 1]], dtype=torch.int64)
        indices = torch.tensor([[4, 9, 7, -1], [1, 1, 8, -1]], dtype=torch.int64)
        node_ids = torch.full((steps, batch * topk + 1), -1, dtype=torch.int64)
        node_ids[:, : batch * topk] = torch.tensor(
            [
                [4, 4, 1, 2],
                [4, 9, 1, 8],
            ],
            dtype=torch.int64,
        )
        # Padding column repeats an id and must not be selected.
        node_ids[:, -1] = 4
        base = torch.arange(batch * topk * (steps + 1), dtype=torch.int64).view(
            batch, topk, steps + 1
        )
        compact = base[:, :, ::2][:, :, :steps]
        self.assertEqual(tuple(compact.shape), (batch, topk, steps))
        self.assertFalse(compact.is_contiguous())
        self._assert_packed(
            tokens, parents, indices, node_ids, compact, batch, topk, steps, True
        )

    def test_pack_reads_strided_sources_in_row_major_order(self):
        tokens = torch.arange(12, dtype=torch.int64).view(3, 4)[1:, 1::2]
        parents = torch.arange(12, dtype=torch.int64).view(3, 4)[:, ::2]
        indices = torch.arange(6, dtype=torch.int32).view(2, 3)[:, [2, 0]]
        self.assertFalse(tokens.is_contiguous())
        self.assertFalse(parents.is_contiguous())
        self._assert_packed(tokens, parents, indices, None, None, 2, 1, 1, False)

    def test_cpu_warmup_does_not_launch_or_count_packs(self):
        import sys
        from pathlib import Path

        kernel = (
            "sglang.srt.speculative.standalone_remote.drafter."
            "sr_tree_reply_pack_kernels"
        )
        self.assertNotIn(kernel, sys.modules)
        with patch(
            "sglang.srt.speculative.standalone_remote.drafter."
            "sr_tree_reply_pack.pack_tree_reply"
        ) as packed:
            held = warm_tree_reply_pack(torch.device("cpu"), 3, 5, 15)
        packed.assert_not_called()
        self.assertEqual(held, [])
        self.assertNotIn(kernel, sys.modules)
        drafter = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_tree_drafter.py"
        )
        text = drafter.read_text(encoding="utf-8")
        start = text.index("def _sr_warm_tree_shapes")
        body = text[start : text.index("\n    def ", start + 1)]
        self.assertIn("warm_tree_reply_pack", body)
        self.assertIn("warmup_synchronize", body)
        self.assertNotIn("tree_device_pack_copies", body)
        tokens = torch.tensor([[10, 11]], dtype=torch.int64)
        parents = torch.tensor([[-1, 0]], dtype=torch.int64)
        indices = torch.tensor([[0, 1]], dtype=torch.int64)
        self._assert_packed(tokens, parents, indices, None, None, 1, 1, 1, False)


def _identity_gold(ids, parents, depth):
    out = ids.clone()
    rows = int(parents.numel())
    idx = parents.to(dtype=torch.int64).reshape(-1)
    for step in range(int(depth)):
        out[step, :rows] = out[step, :rows].index_select(0, idx)
    return out


class TestIdentityRemap(unittest.TestCase):
    def test_swap_cycle_and_duplicate_parents_match_snapshot(self):
        cases = [
            torch.tensor([1, 0]),
            torch.tensor([1, 2, 0]),
            torch.tensor([0, 0, 0]),
            torch.tensor([1, 0, 1]),
            torch.tensor([2, 2, 0, 1]),
        ]
        ids = torch.arange(3 * 6, dtype=torch.int64).reshape(3, 6)
        for parents in cases:
            rows = int(parents.numel())
            table = ids.clone()
            tail = table[:, rows:].clone()
            below = table[2:].clone()
            scratch = torch.empty((3, 6), dtype=torch.int64)
            remap_slot_node_ids(table, parents, 2, scratch)
            self.assertEqual(table[:, :rows].tolist(), _identity_gold(ids, parents, 2)[:, :rows].tolist())
            self.assertEqual(table[:, rows:].tolist(), tail.tolist())
            self.assertEqual(table[2:].tolist(), below.tolist())

    def test_one_index_select_covers_every_depth(self):
        ids = torch.arange(12, dtype=torch.int64).reshape(3, 4)
        scratch = torch.empty_like(ids)
        parents = torch.tensor([1, 0, 3, 2])
        with patch(
            "sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease.torch.index_select",
            wraps=torch.index_select,
        ) as spy:
            remap_slot_node_ids(ids, parents, 3, scratch)
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(ids.tolist(), _identity_gold(torch.arange(12).reshape(3, 4), parents, 3).tolist())

    def test_wide_stride_and_noncontiguous_int32_parents(self):
        backing = torch.arange(80, dtype=torch.int64)
        ids = backing.as_strided((3, 5), (10, 1))
        before_gap = backing.clone()
        parent_backing = torch.tensor([3, 9, 1, 9, 0, 9, 2], dtype=torch.int32)
        parents = parent_backing.as_strided((4,), (2,))
        scratch = torch.empty((4, 8), dtype=torch.int64)
        remap_slot_node_ids(ids, parents, 2, scratch)
        self.assertEqual(parents.tolist(), [3, 1, 0, 2])
        self.assertEqual(
            ids[:, :4].tolist(),
            _identity_gold(before_gap.as_strided((3, 5), (10, 1)), parents, 2)[:, :4].tolist(),
        )
        for step in range(3):
            self.assertEqual(backing[step * 10 + 4 : step * 10 + 10].tolist(), before_gap[step * 10 + 4 : step * 10 + 10].tolist())
        self.assertEqual(backing[30:].tolist(), before_gap[30:].tolist())

    def test_depth_zero_alias_and_bounds(self):
        ids = torch.arange(6, dtype=torch.int64).reshape(2, 3)
        scratch = torch.empty((2, 3), dtype=torch.int64)
        before = ids.clone()
        remap_slot_node_ids(ids, torch.tensor([2, 0, 1]), 0, scratch)
        self.assertEqual(ids.tolist(), before.tolist())
        with self.assertRaisesRegex(RuntimeError, "aliases"):
            remap_slot_node_ids(ids, torch.tensor([0, 1, 2]), 1, ids)
        with self.assertRaises(IndexError):
            remap_slot_node_ids(ids, torch.tensor([0, 3]), 1, scratch)
        self.assertEqual(ids.tolist(), before.tolist())
        with self.assertRaisesRegex(RuntimeError, "depth exceeds"):
            remap_slot_node_ids(ids, torch.tensor([0, 1]), 3, scratch)

    def test_workspace_domains_freeze_and_submitted_failure(self):
        eager = IdentityRemapWorkspace("cpu", 5, 2, graph=False, domain="tree_eager")
        graph = IdentityRemapWorkspace("cpu", 5, 6, graph=True, domain="tree_graph")
        self.assertNotEqual(eager.scratch.data_ptr(), graph.scratch.data_ptr())
        self.assertIsInstance(IdentityRemapSubmittedError("x"), KVMoveSubmittedError)
        eager.reserve_rows(3)
        self.assertEqual(tuple(eager.scratch.shape), (5, 4))
        graph.reserve_rows(6)
        with self.assertRaisesRegex(RuntimeError, "cannot grow"):
            graph.reserve_rows(7)
        eager.frozen = True
        with self.assertRaisesRegex(RuntimeError, "cannot grow"):
            eager.reserve_rows(8)
        with self.assertRaisesRegex(RuntimeError, "cannot bind a graph"):
            with eager.capture_scope():
                pass
        with graph.capture_scope():
            self.assertTrue(graph._inside_capture)
        self.assertTrue(graph.frozen)
        ids = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 0]], dtype=torch.int64)
        parents = torch.tensor([1, 0, 1])
        scratch = torch.empty((2, 4), dtype=torch.int64)
        ws = IdentityRemapWorkspace("cpu", 2, 4, graph=False, domain="tree_eager")
        ws.remap(ids, parents, 2)
        self.assertFalse(ws.unresolved)
        self.assertEqual(ids[:, :3].tolist(), [[2, 1, 2], [5, 4, 5]])
        with self.assertRaises(IndexError):
            ws.remap(ids, torch.tensor([0, 9]), 1)
        self.assertFalse(ws.unresolved)
        def fail():
            raise RuntimeError("launch failed")

        with self.assertRaises(IdentityRemapSubmittedError):
            ws.submitted((ids,), fail)
        self.assertTrue(ws.unresolved)
        with self.assertRaises(IdentityRemapSubmittedError):
            ws.remap(ids, parents, 1)

    def test_capture_skips_stream_check_and_foreign_stream_fails(self):
        ws = IdentityRemapWorkspace("cpu", 2, 2, graph=True, domain="tree_graph")
        ws.device = type("Dev", (), {"type": "cuda"})()
        ws._stream_key = 1
        ws._inside_capture = True
        ws.check()
        ws._inside_capture = False
        with patch(
            "sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease._identity_stream_key",
            return_value=2,
        ):
            with self.assertRaisesRegex(RuntimeError, "another execution stream"):
                ws.check()


class TestLeaseLifecycle(unittest.TestCase):
    def test_budget_does_not_double_count_live(self):
        self.assertTrue(
            lease_budget_ok(
                immediate_free=10,
                live_lease_pages=6,
                need=3,
                reserve=1,
                max_fraction=1.0,
            )
        )
        self.assertFalse(
            lease_budget_ok(
                immediate_free=3,
                live_lease_pages=6,
                need=3,
                reserve=1,
                max_fraction=1.0,
            )
        )

    def test_align_snapshot_ignores_new_revision(self):
        req = DummyReq([1, 2, 3], kv_len=3, revision=4)
        dreq = DummyDreq([7])
        result = snapshot_sr_align(req, dreq, prefix_len=3)
        self.assertEqual(result.kind, "append_one")
        self.assertEqual(result.old_prefix_revision, 4)
        self.assertEqual(result.old_kv_committed_len, 3)
        self.assertEqual(result.old_local_len, 3)
        self.assertEqual(result.old_prefix_window, (1, 2, 3))
        self.assertFalse(hasattr(result, "old_committed_tokens"))
        req.sr_prefix_revision = 5
        self.assertNotEqual(result.old_prefix_revision, req.sr_prefix_revision)

    def test_commit_uses_old_revision(self):
        lease = SRTreeKVLease(
            rid="r",
            version=2,
            revision=4,
            base_committed_len=3,
            prefix_tokens=(1, 2, 3),
            page_ids=[1],
            page_slots=torch.arange(128),
            candidate_slots=[10, 11, -1],
            parent_list=[-1, 0],
            top_scores_index=[0, 1],
            draft_tokens=[7, 8],
        )
        align = SRAlignResult(
            kind="append_n",
            old_kv_committed_len=3,
            old_prefix_revision=4,
            old_local_len=3,
            old_prefix_window=(1, 2, 3),
            fork=3,
        )
        miss = validate_lease_commit(
            lease,
            commit_tree_version=2,
            commit_tree_base_committed_len=3,
            commit_candidate_indices=[0, 1],
            align=align,
            path_tokens=[7, 8],
        )
        self.assertIsNone(miss)
        align_new = SRAlignResult(
            kind="append_n",
            old_kv_committed_len=3,
            old_prefix_revision=5,
            old_local_len=3,
            old_prefix_window=(1, 2, 3),
            fork=3,
        )
        self.assertEqual(
            validate_lease_commit(
                lease,
                commit_tree_version=2,
                commit_tree_base_committed_len=3,
                commit_candidate_indices=[0, 1],
                align=align_new,
                path_tokens=[7, 8],
            ),
            "revision",
        )

    def test_missing_fields_and_depth(self):
        lease = SRTreeKVLease(
            rid="r",
            version=1,
            revision=0,
            base_committed_len=1,
            prefix_tokens=(9,),
            page_ids=[1],
            page_slots=torch.arange(4),
            candidate_slots=[-1, 3],
            parent_list=[],
            top_scores_index=[0],
            draft_tokens=[4],
        )
        align = SRAlignResult("append_one", 1, 0, 1, (9,), 1)
        self.assertEqual(
            validate_lease_commit(
                lease,
                commit_tree_version=None,
                commit_tree_base_committed_len=1,
                commit_candidate_indices=[0],
                align=align,
                path_tokens=[4],
            ),
            "fields",
        )
        self.assertEqual(MIN_TREE_KV_REUSE_DEPTH, 2)

    def test_in_use_skips_reclaim(self):
        store = SRTreeLeaseStore()
        idle = SRTreeKVLease(
            rid="a",
            version=1,
            revision=0,
            base_committed_len=1,
            prefix_tokens=(1,),
            page_ids=[1, 2],
            page_slots=torch.arange(2),
            candidate_slots=[0],
            parent_list=[],
            top_scores_index=[],
            draft_tokens=[],
        )
        busy = SRTreeKVLease(
            rid="b",
            version=2,
            revision=0,
            base_committed_len=1,
            prefix_tokens=(1,),
            page_ids=[3],
            page_slots=torch.arange(1),
            candidate_slots=[0],
            parent_list=[],
            top_scores_index=[],
            draft_tokens=[],
            in_use=True,
        )
        store.register(idle)
        store.register(busy)
        class Alloc:
            def __init__(self):
                self.freed = []

            def free(self, slots):
                self.freed.append(int(slots.numel()))

        alloc = Alloc()
        store.reclaim_idle(alloc)
        self.assertIsNone(store.get("a"))
        self.assertIsNotNone(store.get("b"))
        self.assertEqual(alloc.freed, [2])

    def test_lease_survives_later_graph_buffer_replay(self):
        store = SRTreeLeaseStore()
        graph_buf = torch.tensor([[0, 1, 2], [3, 4, 5]])
        lease = SRTreeKVLease(
            rid="a",
            version=9,
            revision=1,
            base_committed_len=2,
            prefix_tokens=(1, 2),
            page_ids=[4],
            page_slots=torch.arange(4),
            candidate_slots=[11, 12, -1],
            parent_list=[-1, 0],
            top_scores_index=[0, 1],
            draft_tokens=[7, 8],
        )
        store.register(lease)
        graph_buf.fill_(-1)
        graph_buf[0] = torch.tensor([9, 9, 9])
        held = store.get("a")
        self.assertEqual(held.candidate_slots, [11, 12, -1])
        self.assertEqual(held.draft_tokens, [7, 8])
        self.assertEqual(int(held.page_slots[0]), 0)

    def test_live_prefix_stops_at_minus_one(self):
        self.assertEqual(live_accept_prefix([4, 5, -1, 7], [0, 1, 2, 3]), [4, 5])

    def test_event_free_waits_for_query(self):
        store = SRTreeLeaseStore()
        lease = SRTreeKVLease(
            rid="e",
            version=1,
            revision=0,
            base_committed_len=1,
            prefix_tokens=(1,),
            page_ids=[1],
            page_slots=torch.arange(4),
            candidate_slots=[0],
            parent_list=[],
            top_scores_index=[],
            draft_tokens=[],
        )
        store.register(lease)

        class Event:
            def __init__(self):
                self.ready = False

            def query(self):
                return self.ready

        class Alloc:
            def __init__(self):
                self.freed = []

            def free(self, slots):
                self.freed.append(int(slots.numel()))

        event = Event()
        alloc = Alloc()
        store.release(lease, allocator=alloc, event=event)
        self.assertEqual(alloc.freed, [])
        store.poll_pending_frees(alloc)
        self.assertEqual(alloc.freed, [])
        event.ready = True
        store.poll_pending_frees(alloc)
        self.assertEqual(alloc.freed, [4])

    def test_query_failure_keeps_all_pages_and_marks_unknown_completion(self):
        from types import SimpleNamespace as NS
        from unittest.mock import Mock
        from sglang.srt.speculative.standalone_remote.sr_transfer_staging import (
            SRTransferUnresolved,
        )

        store = SRTreeLeaseStore()
        alloc = NS(free=Mock())
        leases = []
        for rid in ("ready", "unknown"):
            lease = SRTreeKVLease(
                rid, 1, 0, 1, (1,), [1], torch.arange(4), [0], [], [], []
            )
            store.register(lease)
            event = NS(query=Mock(return_value=True))
            if rid == "unknown":
                event.query.side_effect = RuntimeError("event query failed")
            store.release(lease, allocator=alloc, event=event)
            leases.append(lease)
        with self.assertRaises(SRTransferUnresolved):
            store.poll_pending_frees(alloc)
        alloc.free.assert_not_called()
        self.assertEqual(len(store._pending_frees(alloc)), 2)
        self.assertTrue(leases[1].copy_unresolved)
        leases[1].pending_free_event = None
        store.poll_pending_frees(alloc)
        alloc.free.assert_called_once()
        self.assertEqual(store._pending_frees(alloc), [leases[1]])
        store.poll_pending_frees(alloc)
        alloc.free.assert_called_once()

    def test_pin_lease_requires_same_object_and_version(self):
        store = SRTreeLeaseStore()
        first = SRTreeKVLease(
            rid="p",
            version=1,
            revision=0,
            base_committed_len=1,
            prefix_tokens=(1,),
            page_ids=[1],
            page_slots=torch.arange(2),
            candidate_slots=[0],
            parent_list=[],
            top_scores_index=[],
            draft_tokens=[],
        )
        store.register(first)
        self.assertIs(store.pin_lease(first), first)
        store.unpin_lease(first)
        second = SRTreeKVLease(
            rid="p",
            version=2,
            revision=0,
            base_committed_len=1,
            prefix_tokens=(1,),
            page_ids=[2],
            page_slots=torch.arange(2),
            candidate_slots=[0],
            parent_list=[],
            top_scores_index=[],
            draft_tokens=[],
        )
        store.register(second)
        self.assertIsNone(store.pin_lease(first))
        self.assertIs(store.pin_lease(second), second)
        store.unpin_lease(first)
        self.assertTrue(second.in_use)
        store.unpin_lease(second)
        self.assertFalse(second.in_use)

    def test_event_without_query_stays_pending(self):
        store = SRTreeLeaseStore()
        lease = SRTreeKVLease(
            rid="q",
            version=1,
            revision=0,
            base_committed_len=1,
            prefix_tokens=(1,),
            page_ids=[1],
            page_slots=torch.arange(4),
            candidate_slots=[0],
            parent_list=[],
            top_scores_index=[],
            draft_tokens=[],
        )
        store.register(lease)

        class Alloc:
            def __init__(self):
                self.freed = []

            def free(self, slots):
                self.freed.append(int(slots.numel()))

        alloc = Alloc()
        store.release(lease, allocator=alloc, event=object())
        store.poll_pending_frees(alloc)
        self.assertEqual(alloc.freed, [])

    def test_record_device_event_required_propagates(self):
        from unittest.mock import patch
        from types import SimpleNamespace as NS

        class Mod:
            def Event(self):
                raise RuntimeError("no event")

        with patch("torch.get_device_module", return_value=Mod()):
            with self.assertRaisesRegex(RuntimeError, "no event"):
                record_device_event(NS(type="npu"), required=True)
            self.assertIsNone(record_device_event(NS(type="npu")))
        self.assertIsNone(record_device_event(NS(type="cpu"), required=True))
        self.assertIsNone(record_device_event(None, required=True))

    def test_release_is_idempotent_and_wipe_recycles(self):
        store = SRTreeLeaseStore()
        lease = SRTreeKVLease(
            rid="w",
            version=1,
            revision=0,
            base_committed_len=1,
            prefix_tokens=(1,),
            page_ids=[1, 2],
            page_slots=torch.arange(2),
            candidate_slots=[0],
            parent_list=[],
            top_scores_index=[],
            draft_tokens=[],
            in_use=True,
        )
        store.register(lease)

        class Alloc:
            def __init__(self):
                self.freed = []

            def free(self, slots):
                self.freed.append(int(slots.numel()))

        alloc = Alloc()
        store.release_rid("w", allocator=alloc)
        store.release(lease, allocator=alloc)
        self.assertEqual(alloc.freed, [2])
        idle = SRTreeKVLease(
            rid="z",
            version=2,
            revision=0,
            base_committed_len=1,
            prefix_tokens=(1,),
            page_ids=[3],
            page_slots=torch.arange(1),
            candidate_slots=[],
            parent_list=[],
            top_scores_index=[],
            draft_tokens=[],
        )
        store.register(idle)
        store.release_all(alloc)
        self.assertIsNone(store.get("z"))
        self.assertEqual(alloc.freed, [2, 1])

    def test_path_mismatch_and_missing_lease_has_no_version(self):
        lease = SRTreeKVLease(
            rid="r",
            version=2,
            revision=0,
            base_committed_len=1,
            prefix_tokens=(9,),
            page_ids=[1],
            page_slots=torch.arange(4),
            candidate_slots=[10, 11],
            parent_list=[],
            top_scores_index=[0, 1],
            draft_tokens=[4, 5],
        )
        align = SRAlignResult("append_n", 1, 0, 1, (9,), 1)
        self.assertEqual(
            validate_lease_commit(
                lease,
                commit_tree_version=2,
                commit_tree_base_committed_len=1,
                commit_candidate_indices=[0, 1],
                align=align,
                path_tokens=[4, 9],
            ),
            "token",
        )
        store = SRTreeLeaseStore()
        self.assertIsNone(store.get("r"))

    def test_page_count_falls_back_to_page_ids(self):
        lease = SRTreeKVLease(
            rid="r",
            version=1,
            revision=0,
            base_committed_len=1,
            prefix_tokens=(1,),
            page_ids=[1, 2],
            page_slots=torch.arange(2),
            candidate_slots=[],
            parent_list=[],
            top_scores_index=[],
            draft_tokens=[],
        )
        self.assertEqual(lease.page_count, 2)
        self.assertEqual(lease_page_count(lease), 2)
        empty = SRTreeKVLease(
            rid="e",
            version=1,
            revision=0,
            base_committed_len=1,
            prefix_tokens=(1,),
            page_ids=[],
            page_slots=torch.arange(0),
            candidate_slots=[],
            parent_list=[],
            top_scores_index=[],
            draft_tokens=[],
            page_count=6,
        )
        self.assertEqual(lease_page_count(empty), 6)
        store = SRTreeLeaseStore()
        store.register(empty)
        self.assertEqual(store.live_pages(), 6)

    def test_prefix_window_bounded_match(self):
        window = LEASE_PREFIX_WINDOW
        base = 100
        tail = tuple(range(1000, 1000 + window))
        committed = tuple(range(base - window)) + tail
        lease = SRTreeKVLease(
            rid="r",
            version=2,
            revision=0,
            base_committed_len=base,
            prefix_tokens=tail,
            page_ids=[],
            page_slots=torch.arange(1),
            candidate_slots=[10, 11],
            parent_list=[-1, 0],
            top_scores_index=[0, 1],
            draft_tokens=[7, 8],
            page_count=1,
        )
        hit = SRAlignResult(
            "append_n", base, 0, base, prefix_window_from_committed(committed, base), base
        )
        self.assertIsNone(
            validate_lease_commit(
                lease,
                commit_tree_version=2,
                commit_tree_base_committed_len=base,
                commit_candidate_indices=[0, 1],
                align=hit,
                path_tokens=[7, 8],
            )
        )
        outside = (999,) + committed[1:]
        self.assertIsNone(
            validate_lease_commit(
                lease,
                commit_tree_version=2,
                commit_tree_base_committed_len=base,
                commit_candidate_indices=[0, 1],
                align=SRAlignResult(
                    "append_n",
                    base,
                    0,
                    base,
                    prefix_window_from_committed(outside, base),
                    base,
                ),
                path_tokens=[7, 8],
            )
        )
        inside = committed[:-1] + (0,)
        self.assertEqual(
            validate_lease_commit(
                lease,
                commit_tree_version=2,
                commit_tree_base_committed_len=base,
                commit_candidate_indices=[0, 1],
                align=SRAlignResult(
                    "append_n",
                    base,
                    0,
                    base,
                    prefix_window_from_committed(inside, base),
                    base,
                ),
                path_tokens=[7, 8],
            ),
            "token",
        )
        origin = list(range(80))
        output = list(range(80, 100))
        self.assertEqual(
            prefix_window_tokens(origin, output, base=base),
            tuple(range(base - window, base)),
        )


class TestProtocolAndExport(unittest.TestCase):
    def test_optional_fields_roundtrip(self):
        req = SRDraftRequest(
            rid="r",
            commit_tree_version=3,
            commit_tree_base_committed_len=10,
            commit_candidate_indices=[0, 2],
        )
        d = req.to_dict()
        back = SRDraftRequest.from_dict(d)
        self.assertEqual(back.commit_tree_version, 3)
        self.assertEqual(back.commit_candidate_indices, [0, 2])
        reply = SRDraftReply(rid="r", tree_version=3)
        self.assertEqual(SRDraftReply.from_dict(reply.to_dict()).tree_version, 3)
        self.assertNotIn("tree_version", SRDraftReply(rid="x").to_dict())

    def test_export_accept_path_excludes_root(self):
        accept = torch.tensor([[10, 12, 13, -1], [20, 21, -1, -1]])
        retrive = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
        paths = export_accepted_tree_candidate_indices(accept, retrive)
        self.assertEqual(paths, [[1, 2], [0]])
        cpu_rows = [[10, 12, 13, -1], [20, 21, -1, -1]]
        self.assertEqual(
            export_accepted_tree_candidate_indices(cpu_rows),
            [[1, 2], [0]],
        )
        eos = [[10, 12, -1, 13], [20, -1, -1, -1]]
        self.assertEqual(export_accepted_tree_candidate_indices(eos), [[1], []])
        bonus_only = [[4, -1], [8, -1]]
        self.assertEqual(export_accepted_tree_candidate_indices(bonus_only), [[], []])
        self.assertEqual(export_accepted_tree_candidate_indices([[-1]]), [[]])
        self.assertEqual(export_accepted_tree_candidate_indices([[]]), [[]])

    def test_eagle_verify_gates_sr_export(self):
        from pathlib import Path

        src = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/speculative/eagle_info.py"
        ).read_text(encoding="utf-8")
        self.assertLess(
            src.index("is_standalone_remote"),
            src.index("export_accepted_tree_candidate_indices"),
        )
        self.assertIn("sr_eos_cut", src)
        self.assertIn("sr_accepted_tree_candidate_indices = None", src)


if __name__ == "__main__":
    unittest.main()
