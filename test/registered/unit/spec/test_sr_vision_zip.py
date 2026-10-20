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
    QUERY_CHUNK,
    _attention_received,
    _item_token_counts,
    _merge_context,
    _sample_queries,
    _slice_encoded_items,
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
        # Dominant row is index 1. The one context target is index 2, and both
        # remaining rows (0 and 3) match it, so the kept row is their mean.
        expected = (embeddings[0] + embeddings[2] + embeddings[3]) / 3
        self.assertTrue(torch.allclose(first[1], expected))

    def test_metric_moves_donor_without_changing_the_mean(self):
        embeddings = torch.tensor(
            [
                [1.0, 0.0, 9.0],
                [0.0, 1.0, 3.0],
                [1.0, 0.0, 5.0],
            ]
        )
        metric = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        )
        remaining = [0, 1, 2]
        targets = [0, 1]
        by_embed = _merge_context(embeddings, remaining, targets, hidden_size=2)
        by_metric = _merge_context(
            embeddings, remaining, targets, hidden_size=2, metric=metric
        )
        self.assertFalse(torch.allclose(by_embed, by_metric))
        self.assertTrue(torch.allclose(by_embed[0], (embeddings[0] + embeddings[2]) / 2))
        self.assertTrue(torch.allclose(by_embed[1], embeddings[1]))
        self.assertTrue(torch.allclose(by_metric[0], embeddings[0]))
        self.assertTrue(
            torch.allclose(by_metric[1], (embeddings[1] + embeddings[2]) / 2)
        )

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
        self.assertNotIn("0,2,5", salted)
        self.assertEqual(len(salted.split(":")[-1]), 32)
        other = visionzip_radix_salt([0, 1, 5], 0.25, 0.85, -2)
        self.assertNotEqual(salted, other)
        self.assertNotIn("0,1,5", other)
        kept = SimpleNamespace(extra_key="cache-salt")
        combined = append_visionzip_extra_key(kept, [0, 2, 5], 0.25, 0.85, -2)
        self.assertTrue(combined.startswith("cache-salt|"))
        self.assertTrue(combined.endswith(again))


class TestVisionProtocol(unittest.TestCase):
    def test_optional_fields_roundtrip_and_absence(self):
        bare = SRDraftRequest(rid="a")
        self.assertNotIn("draft_vision_keep_ratio", bare.to_dict())
        loaded = SRDraftRequest.from_dict(
            {"rid": "a", "draft_vision_keep_ratio": 0.5, "future_field": 1}
        )
        self.assertEqual(loaded.rid, "a")
        self.assertFalse(hasattr(loaded, "draft_vision_keep_ratio"))

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


class TestVisionZipScoreAndSlice(unittest.TestCase):
    def test_einsum_matches_per_head_matmul(self):
        torch.manual_seed(0)
        tokens, heads, dim = 96, 3, 5
        q = torch.randn(tokens, heads, dim)
        k = torch.randn(tokens, heads, dim)
        got = _attention_received(q, k)

        queries = _sample_queries(tokens, q.device)
        qs = q.index_select(0, queries).float()
        kk = k.float()
        self.assertGreater(int(qs.shape[0]), QUERY_CHUNK)
        scale = dim ** -0.5
        acc = torch.zeros(tokens, dtype=torch.float32)
        q_count = int(qs.shape[0])
        for head in range(heads):
            key = kk[:, head, :]
            for q0 in range(0, q_count, QUERY_CHUNK):
                qq = qs[q0 : q0 + QUERY_CHUNK, head, :]
                logits = torch.matmul(qq, key.transpose(0, 1)) * scale
                acc += torch.softmax(logits, dim=-1).sum(dim=0)
        acc /= float(q_count * heads)
        self.assertTrue(torch.allclose(got, acc))

    def test_slice_encoded_items_covers_two_requests(self):
        first = SimpleNamespace(
            image_grid_thw=torch.tensor([[1, 2, 2]]),
            offsets=[(0, 0)],
        )
        second = SimpleNamespace(
            image_grid_thw=torch.tensor([[1, 4, 4]]),
            offsets=[(10, 13)],
        )
        merge_unit = 4
        counts = [
            sum(_item_token_counts(item, merge_unit)) for item in (first, second)
        ]
        self.assertEqual(counts, [1, 4])
        total = sum(counts)
        rows = torch.arange(total * 2, dtype=torch.float32).view(total, 2)
        scores = torch.arange(total, dtype=torch.float32)
        keys = torch.arange(total * 3, dtype=torch.float32).view(total, 3)
        sliced = _slice_encoded_items(rows, scores, keys, [first, second], merge_unit)
        cursor = 0
        for item, count in ((first, counts[0]), (second, counts[1])):
            emb, score, key = sliced[id(item)]
            self.assertTrue(torch.equal(emb, rows[cursor : cursor + count]))
            self.assertTrue(torch.equal(score, scores[cursor : cursor + count]))
            self.assertTrue(torch.equal(key, keys[cursor : cursor + count]))
            cursor += count
        self.assertEqual(cursor, total)


if __name__ == "__main__":
    unittest.main()
