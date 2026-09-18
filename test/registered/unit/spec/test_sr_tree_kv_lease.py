"""CPU contracts for page-level SR tree KV leases."""

from __future__ import annotations

import unittest

import torch

from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
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
    remap_slot_node_ids,
    snapshot_sr_align,
    tree_raw_span_len,
    validate_lease_commit,
)
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
        tmp = torch.empty(bs * topk, dtype=torch.int64)
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
        tmp = torch.empty(topk, dtype=torch.int64)
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

    def test_identity_reset_clears_padding(self):
        ids = torch.arange(8, dtype=torch.int64).reshape(2, 4)
        ids.fill_(-1)
        self.assertTrue(bool((ids == -1).all()))


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
            old_committed_tokens=(1, 2, 3),
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
            old_committed_tokens=(1, 2, 3),
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
        align = SRAlignResult("append_one", 1, 0, (9,), 1)
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
        align = SRAlignResult("append_n", 1, 0, (9,), 1)
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
        hit = SRAlignResult("append_n", base, 0, committed, base)
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
                align=SRAlignResult("append_n", base, 0, outside, base),
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
                align=SRAlignResult("append_n", base, 0, inside, base),
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


if __name__ == "__main__":
    unittest.main()
