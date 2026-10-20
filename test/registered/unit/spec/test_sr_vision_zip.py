"""CPU tests for Draft prompt coordinates and VisionZip selection."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
    SRAlignResult,
    SRTreeKVLease,
    snapshot_sr_align,
    validate_lease_commit,
)
from sglang.srt.speculative.standalone_remote.drafter.sr_vision_zip import (
    append_visionzip_extra_key,
    rebuild_prompt,
    validate_draft_vision_config,
    visionzip_compact_rows,
    visionzip_radix_salt,
)
from sglang.srt.speculative.standalone_remote.sr_commit import (
    CommitOutcome,
    CommitVerdict,
    delta_fast_allowed,
    grammar_committed_ids,
    make_prefix_stamp,
)
from sglang.srt.speculative.standalone_remote.sr_prompt_coords import (
    SRPromptCoordError,
    lease_base_pair,
    logical_ids,
    logical_len,
    physical_ids_for_target,
    restore_compacted_mrope,
    to_local,
)
from sglang.srt.speculative.standalone_remote.sr_protocol import (
    SRDraftReply,
    SRDraftRequest,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-test-cpu")


def _req(**kwargs):
    base = dict(
        origin_input_ids=[],
        output_ids=[],
        sr_padded_ids=None,
        sr_local_prompt_ids=None,
        sr_prompt_delta=0,
        sr_prefix_revision=0,
        kv_committed_len=0,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


class TestPromptCoords(unittest.TestCase):
    def test_delta_zero_is_identity(self):
        req = _req(origin_input_ids=[1, 2, 3], output_ids=[4], sr_padded_ids=[1, 2])
        self.assertEqual(logical_ids(req), [1, 2, 3, 4])
        self.assertEqual(logical_len(req), 4)
        self.assertEqual(to_local(req, 2), 2)
        self.assertEqual(physical_ids_for_target(req, [1, 2, 9]), [1, 2, 9])

    def test_post_prompt_shift_and_inside_prompt_rejected(self):
        req = _req(
            origin_input_ids=[10, 11, 7],
            output_ids=[8],
            sr_padded_ids=[1, 2, 3, 4],
            sr_local_prompt_ids=[10, 11],
            sr_prompt_delta=2,
        )
        self.assertEqual(logical_ids(req), [1, 2, 3, 4, 7, 8])
        self.assertEqual(logical_len(req), 6)
        self.assertEqual(to_local(req, 4), 2)
        self.assertEqual(to_local(req, 6), 4)
        with self.assertRaises(SRPromptCoordError):
            to_local(req, 3)
        self.assertEqual(
            physical_ids_for_target(req, [1, 2, 3, 4, 7, 9]),
            [10, 11, 7, 9],
        )
        with self.assertRaises(SRPromptCoordError):
            physical_ids_for_target(req, [9, 2, 3, 4])

    def test_align_equal_fork_and_prompt_overflow(self):
        req = _req(
            origin_input_ids=[10, 11, 7],
            output_ids=[8],
            sr_padded_ids=[1, 2, 3, 4],
            sr_local_prompt_ids=[10, 11],
            sr_prompt_delta=2,
            kv_committed_len=3,
            sr_prefix_revision=4,
        )
        equal = snapshot_sr_align(req, SimpleNamespace(committed_ids=[7, 8]), 0)
        self.assertEqual(equal.kind, "equal")
        self.assertEqual(equal.old_local_len, 4)
        self.assertEqual(equal.fork, 6)
        self.assertEqual(to_local(req, equal.fork), 4)

        fork = snapshot_sr_align(req, SimpleNamespace(committed_ids=[7, 9]), 0)
        self.assertEqual(fork.kind, "replace_tail")
        self.assertEqual(fork.fork, 5)
        self.assertEqual(to_local(req, fork.fork), 3)
        self.assertEqual(fork.old_kv_committed_len, 3)

        broken = _req(
            origin_input_ids=[1],
            sr_padded_ids=[1, 2, 3, 4],
            sr_local_prompt_ids=[1, 2, 3],
            sr_prompt_delta=1,
        )
        overflow = snapshot_sr_align(broken, SimpleNamespace(committed_ids=[9]), 0)
        self.assertEqual(overflow.kind, "reprefill")

    def test_delta_fast_and_grammar_use_local_prompt(self):
        req = _req(
            origin_input_ids=[10, 11, 7],
            output_ids=[],
            sr_padded_ids=[1, 2, 3, 4],
            sr_local_prompt_ids=[10, 11],
            sr_prompt_delta=2,
        )
        stamp = make_prefix_stamp(req, 3)
        state = SimpleNamespace(
            degraded=False,
            completion_unknown=False,
            commit_trusted=True,
            pending_commit=None,
            acked_version=3,
            local_prefix_stamp=stamp,
            prompt_len=4,
        )
        verdict = CommitVerdict(CommitOutcome.APPLY, delta_ids=(9,))
        self.assertTrue(
            delta_fast_allowed(
                verdict, req=req, state=state, base_output_len=1, poisoned=False
            )
        )
        state.prompt_len = 3
        self.assertFalse(
            delta_fast_allowed(
                verdict, req=req, state=state, base_output_len=1, poisoned=False
            )
        )
        folded = grammar_committed_ids(
            None,
            output_ids=[],
            origin_ids=[10, 11, 7, 8],
            padded_ids=[1, 2, 3, 4],
            tree_mode=True,
            local_prompt_len=2,
        )
        self.assertEqual(folded, [7, 8])

    def test_lease_splits_logical_base_from_kv(self):
        req = _req(
            origin_input_ids=[10, 11, 7],
            sr_local_prompt_ids=[10, 11],
            sr_prompt_delta=2,
            sr_padded_ids=[1, 2, 3, 4],
        )
        self.assertEqual(lease_base_pair(req, 3), (5, 3))
        self.assertEqual(lease_base_pair(req, 1), (1, 1))
        lease = SRTreeKVLease(
            rid="r",
            version=2,
            revision=1,
            base_committed_len=5,
            kv_committed_len=3,
            prefix_tokens=(10, 11, 7),
            page_ids=[1],
            page_slots=torch.arange(4),
            candidate_slots=[0, 1],
            parent_list=[-1, 0],
            top_scores_index=[0, 1],
            draft_tokens=[8, 9],
        )
        align = SRAlignResult(
            kind="append_one",
            old_kv_committed_len=3,
            old_prefix_revision=1,
            old_local_len=3,
            old_prefix_window=(10, 11, 7),
            fork=5,
        )
        self.assertIsNone(
            validate_lease_commit(
                lease,
                commit_tree_version=2,
                commit_tree_base_committed_len=5,
                commit_candidate_indices=[0, 1],
                align=align,
                path_tokens=[8, 9],
            )
        )
        untouched = SRTreeKVLease(
            rid="r",
            version=1,
            revision=0,
            base_committed_len=4,
            prefix_tokens=(),
            page_ids=[],
            page_slots=torch.arange(1),
            candidate_slots=[],
            parent_list=[],
            top_scores_index=[],
            draft_tokens=[],
        )
        self.assertEqual(untouched.kv_committed_len, 4)

    def test_gathered_mrope_keeps_original_positions(self):
        full = torch.tensor(
            [
                [0, 1, 5, 8, 9],
                [0, 2, 6, 8, 9],
                [0, 3, 7, 8, 9],
            ],
            dtype=torch.int64,
        )
        mm = SimpleNamespace(
            mrope_positions=None,
            mrope_position_delta=None,
            mrope_position_delta_repeated_cache=torch.zeros(1),
        )
        req = _req(
            origin_input_ids=[10, 12, 7, 8],
            sr_padded_ids=[1, 2, 3],
            sr_local_prompt_ids=[10, 12],
            sr_prompt_delta=1,
            sr_prompt_keep_index=[0, 2],
            sr_logical_mrope=full,
            multimodal_inputs=mm,
        )
        self.assertTrue(restore_compacted_mrope(req))
        self.assertTrue(
            torch.equal(
                mm.mrope_positions,
                torch.tensor(
                    [
                        [0, 5, 8, 9],
                        [0, 6, 8, 9],
                        [0, 7, 8, 9],
                    ]
                ),
            )
        )
        self.assertEqual(int(mm.mrope_position_delta.item()), 9 + 1 - 4)
        self.assertIsNone(mm.mrope_position_delta_repeated_cache)

    def test_dropped_axis_max_still_sets_decode_delta(self):
        full = torch.tensor(
            [
                [0, 20, 5, 8],
                [0, 4, 6, 8],
                [0, 3, 7, 8],
            ],
            dtype=torch.int64,
        )
        mm = SimpleNamespace(
            mrope_positions=None,
            mrope_position_delta=None,
            mrope_position_delta_repeated_cache=torch.zeros(1),
        )
        req = _req(
            origin_input_ids=[10, 12, 8],
            sr_padded_ids=[1, 2, 3],
            sr_local_prompt_ids=[10, 12],
            sr_prompt_delta=1,
            sr_prompt_keep_index=[0, 2],
            sr_logical_mrope=full,
            multimodal_inputs=mm,
        )
        self.assertTrue(restore_compacted_mrope(req))
        self.assertEqual(int(mm.mrope_position_delta.item()), 20 + 1 - 3)

    def test_incoming_prompt_fork_reprefills(self):
        req = _req(
            origin_input_ids=[10, 11, 7],
            output_ids=[8],
            sr_padded_ids=[1, 2, 3, 4],
            sr_local_prompt_ids=[10, 11],
            sr_prompt_delta=2,
        )
        changed = snapshot_sr_align(
            req,
            SimpleNamespace(
                committed_ids=[7, 8],
                padded_input_ids=[9, 2, 3, 4],
            ),
            0,
        )
        self.assertEqual(changed.kind, "reprefill")
        self.assertLess(changed.fork, 4)
        same = snapshot_sr_align(
            req,
            SimpleNamespace(
                committed_ids=[7, 8],
                padded_input_ids=[1, 2, 3, 4],
            ),
            0,
        )
        self.assertEqual(same.kind, "equal")


class TestVisionZipRows(unittest.TestCase):
    def test_keep_order_merge_and_repeatability(self):
        embeddings = torch.tensor(
            [
                [1.0, 0.0, 0.0, 4.0],
                [0.0, 5.0, 0.0, 6.0],
                [1.0, 0.0, 3.0, 8.0],
                [1.0, 0.0, 0.0, 10.0],
            ]
        )
        scores = torch.tensor([0.0, 10.0, 0.0, 0.0])
        first, index = visionzip_compact_rows(
            embeddings, scores, keep_ratio=0.5, dominant_ratio=0.5, hidden_size=2
        )
        second, index_again = visionzip_compact_rows(
            embeddings, scores, keep_ratio=0.5, dominant_ratio=0.5, hidden_size=2
        )
        self.assertEqual(index, index_again)
        self.assertEqual(index, sorted(index))
        self.assertEqual(len(index), 2)
        self.assertEqual(tuple(first.shape), (2, 4))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first[0], embeddings[1]))
        self.assertEqual(first.shape[-1], embeddings.shape[-1])
        self.assertGreater(float(first[1, -1]), 0.0)

    def test_full_keep_is_uncompacted(self):
        embeddings = torch.arange(8, dtype=torch.float32).view(4, 2)
        rows, index = visionzip_compact_rows(
            embeddings, torch.ones(4), keep_ratio=1.0, dominant_ratio=0.85, hidden_size=2
        )
        self.assertIs(rows, embeddings)
        self.assertEqual(index, [0, 1, 2, 3])

    def test_rebuild_prompt_preserves_text_order(self):
        ids, keep, offsets = rebuild_prompt(
            [0, 9, 9, 9, 9, 1],
            [(1, 4, [0, 3])],
        )
        self.assertEqual(ids, [0, 9, 9, 1])
        self.assertEqual(keep, [0, 1, 4, 5])
        self.assertEqual(offsets, [(1, 2)])

    def test_radix_salt_tracks_the_keep_set(self):
        bare = SimpleNamespace(extra_key=None)
        salted = append_visionzip_extra_key(bare, [0, 2, 5], 0.25, 0.85, -2)
        self.assertNotEqual(salted, None)
        self.assertTrue(salted.startswith("sr-visionzip:"))
        again = visionzip_radix_salt([0, 2, 5], 0.25, 0.85, -2)
        self.assertEqual(salted, again)
        other = visionzip_radix_salt([0, 1, 5], 0.25, 0.85, -2)
        self.assertNotEqual(salted, other)
        kept = SimpleNamespace(extra_key="cache-salt")
        combined = append_visionzip_extra_key(kept, [0, 2, 5], 0.25, 0.85, -2)
        self.assertTrue(combined.startswith("cache-salt|"))
        self.assertTrue(combined.endswith(again))


class TestVisionProtocol(unittest.TestCase):
    def test_optional_fields_roundtrip_and_absence(self):
        bare = SRDraftRequest(rid="a")
        self.assertNotIn("draft_vision_keep_ratio", bare.to_dict())
        self.assertIsNone(SRDraftRequest.from_dict({"rid": "a"}).draft_vision_keep_ratio)
        sent = SRDraftRequest(rid="a", draft_vision_keep_ratio=0.5)
        loaded = SRDraftRequest.from_dict(sent.to_dict())
        self.assertEqual(loaded.draft_vision_keep_ratio, 0.5)
        loaded = SRDraftRequest.from_dict({"rid": "a", "future_field": 1})
        self.assertEqual(loaded.rid, "a")

        reply = SRDraftReply(rid="a")
        self.assertNotIn("draft_prompt_len", reply.to_dict())
        self.assertNotIn("prompt_compaction", reply.to_dict())
        marked = SRDraftReply(rid="a", draft_prompt_len=3, prompt_compaction=True)
        back = SRDraftReply.from_dict(marked.to_dict())
        self.assertEqual(back.draft_prompt_len, 3)
        self.assertTrue(back.prompt_compaction)
        old = SRDraftReply.from_dict({"rid": "a", "step_id": 1})
        self.assertIsNone(old.draft_prompt_len)
        self.assertIsNone(old.prompt_compaction)

    def test_target_cannot_enable_visionzip(self):
        with self.assertRaises(ValueError):
            validate_draft_vision_config("target", "visionzip", 0.25, 0.85, -2)
        with self.assertRaises(ValueError):
            validate_draft_vision_config("draft", "visionzip", 1.5, 0.85, -2)
        validate_draft_vision_config("target", "none", 0.25, 0.85, -2)
        validate_draft_vision_config("draft", "visionzip", 0.25, 0.85, -2)


if __name__ == "__main__":
    unittest.main()
