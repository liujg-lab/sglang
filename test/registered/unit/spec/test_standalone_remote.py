"""Unit tests for STANDALONE_REMOTE protocol, alignment, mm, and stale replies."""

import threading
import time
import unittest
from unittest.mock import MagicMock

try:
    import torch
except ImportError:
    torch = None

from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.standalone_remote.sr_align import (
    DraftDecision,
    classify_prefix_alignment,
    committed_tail_not_in_kv,
    decide_draft_action,
    draft_needed_max_new_tokens,
    draft_token_budget,
    drop_duplicate_root_draft,
    find_fork_point,
    ingest_active_indices,
    shift_overlapped_prefill_drafts,
)
from sglang.srt.speculative.standalone_remote.sr_protocol import (
    SRAction,
    SRBatchReply,
    SRBatchRequest,
    SRDraftReply,
    SRDraftRequest,
    SRPendingEntry,
    SRReplyStatus,
)
from sglang.srt.speculative.standalone_remote.drafter.sr_draft_state import (
    SRDraftState,
    SRDraftStateManager,
)
from sglang.srt.speculative.standalone_remote.sr_transport import (
    SRDraftServer,
    SRTargetClient,
    _pack_request,
    _unpack_request,
    stale_drop_counts,
)

try:
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase

    register_cpu_ci(est_time=15, suite="stage-a-test-cpu")
except ImportError:
    CustomTestCase = unittest.TestCase


class TestSpecAlgorithmIsolation(CustomTestCase):
    def test_standalone_remote_does_not_pollute_existing_predicates(self):
        sr = SpeculativeAlgorithm.STANDALONE_REMOTE
        self.assertTrue(sr.is_standalone_remote())
        self.assertFalse(sr.is_standalone())
        self.assertFalse(sr.is_spectre())
        self.assertFalse(sr.is_eagle())
        self.assertFalse(sr.is_ngram())
        self.assertFalse(sr.supports_spec_v2())
        self.assertFalse(sr.is_none())

        self.assertTrue(SpeculativeAlgorithm.STANDALONE.is_standalone())
        self.assertFalse(SpeculativeAlgorithm.STANDALONE.is_standalone_remote())
        self.assertTrue(SpeculativeAlgorithm.SPECTRE.is_spectre())
        self.assertFalse(SpeculativeAlgorithm.SPECTRE.is_standalone_remote())
        self.assertTrue(SpeculativeAlgorithm.EAGLE.supports_spec_v2())
        self.assertTrue(SpeculativeAlgorithm.STANDALONE.supports_spec_v2())
        self.assertFalse(SpeculativeAlgorithm.SPECTRE.supports_spec_v2())

    def test_from_string(self):
        self.assertEqual(
            SpeculativeAlgorithm.from_string("STANDALONE_REMOTE"),
            SpeculativeAlgorithm.STANDALONE_REMOTE,
        )


class TestSRProtocolRoundTrip(CustomTestCase):
    def test_batch_request_roundtrip(self):
        req = SRDraftRequest(
            rid="r1",
            step_id=3,
            base_committed_len=12,
            committed_ids=[1, 2, 3],
            num_draft_tokens=5,
            padded_input_ids=[9, 8, 7],
            has_mm=True,
        )
        batch = SRBatchRequest(
            session_id="sess",
            rpc_seq=7,
            action=SRAction.PREFILL,
            reqs=[req],
        )
        restored = SRBatchRequest.from_dict(batch.to_dict())
        self.assertEqual(restored.session_id, "sess")
        self.assertEqual(restored.rpc_seq, 7)
        self.assertEqual(restored.action, SRAction.PREFILL)
        self.assertEqual(restored.reqs[0].rid, "r1")
        self.assertEqual(restored.reqs[0].step_id, 3)
        self.assertEqual(restored.reqs[0].base_committed_len, 12)
        self.assertEqual(restored.reqs[0].committed_ids, [1, 2, 3])
        self.assertEqual(restored.reqs[0].padded_input_ids, [9, 8, 7])
        self.assertTrue(restored.reqs[0].has_mm)

    def test_reply_tree_fields_roundtrip(self):
        reply = SRDraftReply(
            rid="r1",
            step_id=1,
            base_committed_len=5,
            draft_tokens=[10, 11, 12, 13, 14, 15, 16],
            parent_list=[-1, 0, 1, 2],
            top_scores_index=[0, 1, 2, 3, 4, 5, 6],
            status=SRReplyStatus.OK,
        )
        batch = SRBatchReply(session_id="s", rpc_seq=2, reqs=[reply])
        restored = SRBatchReply.from_dict(batch.to_dict())
        got = restored.reqs[0]
        self.assertEqual(got.draft_tokens, [10, 11, 12, 13, 14, 15, 16])
        self.assertEqual(got.parent_list, [-1, 0, 1, 2])
        self.assertEqual(got.top_scores_index, [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(len(got.draft_tokens), 7)

    def test_reply_matches_pending(self):
        pending = SRPendingEntry(step_id=2, base_committed_len=10)
        ok = SRDraftReply(rid="r", step_id=2, base_committed_len=10)
        bad_step = SRDraftReply(rid="r", step_id=3, base_committed_len=10)
        bad_len = SRDraftReply(rid="r", step_id=2, base_committed_len=11)
        self.assertEqual(ok.matches(pending), (True, None))
        self.assertEqual(bad_step.matches(pending)[0], False)
        self.assertEqual(bad_step.matches(pending)[1], "step")
        self.assertEqual(bad_len.matches(pending)[1], "base_len")


class TestForkPointAlign(CustomTestCase):
    def test_identical(self):
        ident, fork = find_fork_point([1, 2, 3], [1, 2, 3])
        self.assertTrue(ident)
        self.assertEqual(fork, 3)
        self.assertEqual(classify_prefix_alignment([1, 2, 3], [1, 2, 3], 1), "equal")

    def test_replace_tail(self):
        self.assertEqual(
            classify_prefix_alignment([1, 2, 9], [1, 2, 8], 1), "replace_tail"
        )

    def test_append_one(self):
        self.assertEqual(
            classify_prefix_alignment([1, 2, 3], [1, 2, 3, 4], 1), "append_one"
        )

    def test_append_n(self):
        # Target accepted several tokens in one verify step.
        self.assertEqual(
            classify_prefix_alignment([1, 2, 3], [1, 2, 3, 4, 5, 6], 1),
            "append_n",
        )
        self.assertEqual(
            classify_prefix_alignment([1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8], 3),
            "append_n",
        )
        # A mid-sequence fork is still reprefill, not a blind append.
        self.assertEqual(
            classify_prefix_alignment([1, 2, 3], [1, 2, 9, 10, 11], 1),
            "reprefill",
        )

    def test_committed_tail_not_in_kv(self):
        self.assertEqual(committed_tail_not_in_kv(10, [7, 8], 10), [7, 8])
        self.assertEqual(committed_tail_not_in_kv(10, [7, 8], 11), [8])
        self.assertEqual(committed_tail_not_in_kv(10, [7, 8], 12), [])
        self.assertEqual(committed_tail_not_in_kv(10, [], 10), [])
        self.assertEqual(committed_tail_not_in_kv(10, None, 10), [])

    def test_fused_ingest_groups_by_tail_offset(self):
        tails = [
            committed_tail_not_in_kv(10, [7, 8], 10),
            committed_tail_not_in_kv(10, [9], 10),
        ]
        self.assertEqual(tails, [[7, 8], [9]])
        self.assertEqual(ingest_active_indices([len(t) for t in tails]), [[0, 1], [0]])
        self.assertEqual(ingest_active_indices([1, 1, 1]), [[0, 1, 2]])
        self.assertEqual(ingest_active_indices([]), [])
        self.assertEqual(ingest_active_indices([0, 0]), [])

    def test_local_rollback(self):
        # Target is a prefix of local: trim extras.
        self.assertEqual(
            classify_prefix_alignment([1, 2, 3, 4, 5], [1, 2, 3], 1),
            "local_rollback",
        )

    def test_diverge_then_reprefill(self):
        # Fork before the end of target: trimming would drop committed tokens.
        self.assertEqual(
            classify_prefix_alignment([1, 2, 3, 4, 5], [1, 2, 9], 1),
            "reprefill",
        )

    def test_reprefill_into_prefix(self):
        self.assertEqual(
            classify_prefix_alignment([1, 2, 3], [9, 8, 7], 2), "reprefill"
        )


class TestDraftDecision(CustomTestCase):
    def test_old_session_dropped(self):
        d = decide_draft_action(
            SRAction.STEP,
            session_id="0001_a",
            rpc_seq=2,
            last_session_id="0002_b",
            last_rpc_seq=1,
            last_step_id=0,
            last_base_committed_len=4,
            step_id=1,
            base_committed_len=4,
            has_state=True,
        )
        self.assertEqual(d, DraftDecision.DROP_OLD_SESSION)

    def test_new_session_wipes(self):
        d = decide_draft_action(
            SRAction.STEP,
            session_id="0003_c",
            rpc_seq=1,
            last_session_id="0002_b",
            last_rpc_seq=9,
            last_step_id=4,
            last_base_committed_len=8,
            step_id=0,
            base_committed_len=8,
            has_state=True,
        )
        self.assertEqual(d, DraftDecision.WIPE_NEW_SESSION)

    def test_prefill_hard_resets(self):
        d = decide_draft_action(
            SRAction.PREFILL,
            session_id="s",
            rpc_seq=3,
            last_session_id="s",
            last_rpc_seq=2,
            last_step_id=4,
            last_base_committed_len=8,
            step_id=0,
            base_committed_len=8,
            has_state=True,
        )
        self.assertEqual(d, DraftDecision.HARD_RESET)

    def test_prefill_after_flush_new_session_is_hard_reset(self):
        """flush_cache: Target new session_id + rpc_seq=1 vs Draft last_rpc from
        the previous session. Passing the old session_id (and/or last_rpc=-1)
        must HARD_RESET, not DROP_STALE_SEQ — otherwise the next generate is AR.
        """
        kwargs = dict(
            last_step_id=4,
            last_base_committed_len=8,
            step_id=0,
            base_committed_len=8,
            has_state=False,
        )
        d = decide_draft_action(
            SRAction.PREFILL,
            session_id="0002_new",
            rpc_seq=1,
            last_session_id="0001_old",
            last_rpc_seq=5,
            **kwargs,
        )
        self.assertEqual(d, DraftDecision.HARD_RESET)

        d_after_wipe = decide_draft_action(
            SRAction.PREFILL,
            session_id="0002_new",
            rpc_seq=1,
            last_session_id="0001_old",
            last_rpc_seq=-1,
            **kwargs,
        )
        self.assertEqual(d_after_wipe, DraftDecision.HARD_RESET)

        d_same_session_reset_seq = decide_draft_action(
            SRAction.PREFILL,
            session_id="0002_new",
            rpc_seq=1,
            last_session_id="0002_new",
            last_rpc_seq=-1,
            **kwargs,
        )
        self.assertEqual(d_same_session_reset_seq, DraftDecision.HARD_RESET)

        # Buggy mixin wiring: new session as last_session_id + old last_rpc.
        d_bug = decide_draft_action(
            SRAction.PREFILL,
            session_id="0002_new",
            rpc_seq=1,
            last_session_id="0002_new",
            last_rpc_seq=5,
            **kwargs,
        )
        self.assertEqual(d_bug, DraftDecision.DROP_STALE_SEQ)

    def test_stale_rpc_seq_dropped(self):
        d = decide_draft_action(
            SRAction.STEP,
            session_id="s",
            rpc_seq=2,
            last_session_id="s",
            last_rpc_seq=2,
            last_step_id=1,
            last_base_committed_len=4,
            step_id=2,
            base_committed_len=5,
            has_state=True,
        )
        self.assertEqual(d, DraftDecision.DROP_STALE_SEQ)

    def test_idempotent_replay(self):
        d = decide_draft_action(
            SRAction.STEP,
            session_id="s",
            rpc_seq=5,
            last_session_id="s",
            last_rpc_seq=4,
            last_step_id=2,
            last_base_committed_len=10,
            step_id=2,
            base_committed_len=10,
            has_state=True,
        )
        self.assertEqual(d, DraftDecision.IDEMPOTENT)

    def test_next_step_full_realign(self):
        d = decide_draft_action(
            SRAction.STEP,
            session_id="s",
            rpc_seq=5,
            last_session_id="s",
            last_rpc_seq=4,
            last_step_id=2,
            last_base_committed_len=10,
            step_id=3,
            base_committed_len=14,
            has_state=True,
        )
        self.assertEqual(d, DraftDecision.FULL_REALIGN)

    def test_base_len_mismatch_full_realign(self):
        d = decide_draft_action(
            SRAction.STEP,
            session_id="s",
            rpc_seq=5,
            last_session_id="s",
            last_rpc_seq=4,
            last_step_id=2,
            last_base_committed_len=10,
            step_id=3,
            base_committed_len=11,
            has_state=True,
        )
        self.assertEqual(d, DraftDecision.FULL_REALIGN)

    def test_finish(self):
        d = decide_draft_action(
            SRAction.FINISH,
            session_id="s",
            rpc_seq=9,
            last_session_id="s",
            last_rpc_seq=8,
            last_step_id=2,
            last_base_committed_len=10,
            step_id=3,
            base_committed_len=11,
            has_state=True,
        )
        self.assertEqual(d, DraftDecision.FINISH)


class TestMMPayloadMultipart(CustomTestCase):
    def test_serialize_roundtrip_and_multipart(self):
        if torch is None:
            self.skipTest("torch not available")
        try:
            from sglang.srt.managers.schedule_batch import (
                Modality,
                MultimodalDataItem,
                MultimodalInputFormat,
            )
            from sglang.srt.speculative.standalone_remote.sr_mm_payload import (
                SRMMPayload,
                deserialize_mm_item,
                serialize_mm_item,
            )
        except ImportError as e:
            self.skipTest(str(e))

        feature = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            hash=42,
            pad_value=1_000_042,
            offsets=[0, 4],
            format=MultimodalInputFormat.NORMAL,
            feature=feature,
            model_specific_data={"image_grid_thw": torch.tensor([[1, 2, 2]])},
        )
        restored_item = deserialize_mm_item(serialize_mm_item(item))
        self.assertEqual(restored_item.pad_value, 1_000_042)
        self.assertTrue(torch.equal(restored_item.feature, feature))

        payload = SRMMPayload(
            rid="vl1",
            padded_input_ids=[1, 1_000_042, 1_000_042, 2],
            mm_items=[serialize_mm_item(item)],
            im_token_id=151655,
        )
        meta, bufs = payload.to_multipart()
        again = SRMMPayload.from_multipart(meta, [bytes(b) for b in bufs])
        self.assertEqual(again.rid, "vl1")
        self.assertEqual(again.padded_input_ids, [1, 1_000_042, 1_000_042, 2])
        mm = again.to_multimodal_inputs()
        self.assertEqual(mm.mm_items[0].pad_value, 1_000_042)
        self.assertTrue(torch.equal(mm.mm_items[0].feature, feature))
        grid = mm.mm_items[0].model_specific_data["image_grid_thw"]
        self.assertIsInstance(grid, torch.Tensor)
        self.assertTrue(torch.equal(grid, torch.tensor([[1, 2, 2]])))

    def test_pack_request_with_mm_frames(self):
        if torch is None:
            self.skipTest("torch not available")
        try:
            from sglang.srt.managers.schedule_batch import (
                Modality,
                MultimodalDataItem,
                MultimodalInputFormat,
            )
            from sglang.srt.speculative.standalone_remote.sr_mm_payload import (
                SRMMPayload,
                serialize_mm_item,
            )
        except ImportError as e:
            self.skipTest(str(e))

        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            hash=1,
            pad_value=7,
            offsets=[0, 1],
            format=MultimodalInputFormat.NORMAL,
            feature=torch.ones(2, 2),
        )
        payload = SRMMPayload(
            rid="r1",
            padded_input_ids=[7, 7],
            mm_items=[serialize_mm_item(item)],
        )
        batch = SRBatchRequest(
            session_id="s",
            rpc_seq=1,
            action=SRAction.PREFILL,
            reqs=[
                SRDraftRequest(
                    rid="r1",
                    padded_input_ids=[7, 7],
                    has_mm=True,
                    num_draft_tokens=2,
                )
            ],
        )
        frames = _pack_request(batch, {"r1": payload})
        restored, mm = _unpack_request(frames)
        self.assertEqual(restored.reqs[0].rid, "r1")
        self.assertIn("r1", mm)
        self.assertEqual(mm["r1"].padded_input_ids, [7, 7])

    def test_walk_treats_tensor_placeholder_as_leaf(self):
        try:
            from sglang.srt.speculative.standalone_remote.sr_mm_payload import (
                _TENSOR_PLACEHOLDER_PREFIX,
                _walk_and_transform,
            )
        except ImportError as e:
            self.skipTest(str(e))
        placeholder = {
            _TENSOR_PLACEHOLDER_PREFIX: 0,
            "shape": [1, 3],
            "dtype": "int64",
        }
        nested = {"model_specific_data": {"image_grid_thw": placeholder}}
        seen = []

        def restore(value):
            seen.append(value)
            return "TENSOR"

        out = _walk_and_transform(nested, restore)
        self.assertEqual(out["model_specific_data"]["image_grid_thw"], "TENSOR")
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0], placeholder)


class TestZmqStaleReply(CustomTestCase):
    def test_late_reply_dropped_by_rpc_seq(self):
        port = f"srtest{int(time.time() * 1000) % 100000}"
        server = SRDraftServer("127.0.0.1", port)
        client = SRTargetClient("127.0.0.1", port, timeout_ms=2000)
        time.sleep(0.05)

        before = stale_drop_counts.get("rpc_seq", 0)

        def serve_once():
            got = server.recv_batch(timeout_ms=2000)
            self.assertIsNotNone(got)
            batch, _ = got
            # Late reply for the previous seq, then the matching one.
            late = SRBatchReply(
                session_id=batch.session_id,
                rpc_seq=batch.rpc_seq - 1,
                reqs=[
                    SRDraftReply(
                        rid="r",
                        step_id=0,
                        base_committed_len=0,
                        draft_tokens=[999],
                        status=SRReplyStatus.OK,
                    )
                ],
            )
            ok = SRBatchReply(
                session_id=batch.session_id,
                rpc_seq=batch.rpc_seq,
                reqs=[
                    SRDraftReply(
                        rid="r",
                        step_id=0,
                        base_committed_len=0,
                        draft_tokens=[1, 2, 3],
                        status=SRReplyStatus.OK,
                    )
                ],
            )
            server.send_batch(late)
            server.send_batch(ok)

        t = threading.Thread(target=serve_once)
        t.start()
        req = SRBatchRequest(
            session_id="00000000000000000001_abcd",
            rpc_seq=5,
            action=SRAction.STEP,
            reqs=[SRDraftRequest(rid="r", step_id=0, base_committed_len=0)],
        )
        client.send_batch(req)
        reply = client.recv_batch(req.session_id, req.rpc_seq)
        t.join(timeout=3)
        self.assertIsNotNone(reply)
        self.assertEqual(reply.rpc_seq, 5)
        self.assertEqual(reply.reqs[0].draft_tokens, [1, 2, 3])
        self.assertGreaterEqual(stale_drop_counts.get("rpc_seq", 0), before + 1)
        client.close()
        server.close()

    def test_session_mismatch_dropped(self):
        port = f"srtest{int(time.time() * 1000) % 100000 + 1}"
        server = SRDraftServer("127.0.0.1", port)
        client = SRTargetClient("127.0.0.1", port, timeout_ms=1500)
        time.sleep(0.05)
        before = stale_drop_counts.get("session", 0)

        def serve_once():
            got = server.recv_batch(timeout_ms=2000)
            self.assertIsNotNone(got)
            batch, _ = got
            server.send_batch(
                SRBatchReply(session_id="old_session", rpc_seq=batch.rpc_seq, reqs=[])
            )
            server.send_batch(
                SRBatchReply(session_id=batch.session_id, rpc_seq=batch.rpc_seq, reqs=[])
            )

        t = threading.Thread(target=serve_once)
        t.start()
        req = SRBatchRequest(
            session_id="new_session",
            rpc_seq=1,
            action=SRAction.STEP,
            reqs=[],
        )
        client.send_batch(req)
        reply = client.recv_batch(req.session_id, req.rpc_seq)
        t.join(timeout=3)
        self.assertIsNotNone(reply)
        self.assertEqual(reply.session_id, "new_session")
        self.assertGreaterEqual(stale_drop_counts.get("session", 0), before + 1)
        client.close()
        server.close()


class TestIdempotentWindowCache(CustomTestCase):
    def test_cached_window_not_advanced(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_draft_state import (
            SRDraftState,
            SRDraftStateManager,
        )

        mgr = SRDraftStateManager()
        window = ([7, 8, 9], None, None)
        st = SRDraftState(
            req_id="r",
            session_id="s",
            last_step_id=2,
            last_base_committed_len=10,
            last_window=window,
        )
        mock_req = MagicMock()
        mock_req.origin_input_ids = [1, 2]
        mock_req.output_ids = [3, 4, 7, 8, 9]
        mock_req.req_pool_idx = 0
        mock_req.kv_committed_len = 7
        st.req_object = mock_req
        mgr.set("r", st)

        kv_before = mock_req.kv_committed_len
        decision = decide_draft_action(
            SRAction.STEP,
            session_id="s",
            rpc_seq=4,
            last_session_id="s",
            last_rpc_seq=3,
            last_step_id=st.last_step_id,
            last_base_committed_len=st.last_base_committed_len,
            step_id=2,
            base_committed_len=10,
            has_state=True,
        )
        self.assertEqual(decision, DraftDecision.IDEMPOTENT)
        self.assertEqual(st.last_window[0], [7, 8, 9])
        self.assertEqual(mock_req.kv_committed_len, kv_before)


class TestDraftSessionWipe(CustomTestCase):
    def test_wipe_clears_scheduler_caches_and_waiting_queue(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                StandaloneRemoteDraftSchedulerMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))

        mixin = StandaloneRemoteDraftSchedulerMixin()
        mixin.sr_state = SRDraftStateManager()
        mixin.sr_waiting = []
        mixin.draft_paused_reqs = []
        mixin.waiting_queue = []
        mixin.running_batch = MagicMock()
        mixin.running_batch.is_empty.return_value = True
        mixin.cur_batch = object()
        mixin.last_batch = object()
        mixin.chunked_req = object()
        mixin.tree_cache = MagicMock()
        mixin.req_to_token_pool = MagicMock()
        mixin.token_to_kv_pool_allocator = MagicMock()
        mixin.grammar_manager = MagicMock()
        mixin.sr_server = MagicMock()
        mixin.sr_server.last_rpc_seq = 9

        leftover = MagicMock()
        leftover.req_pool_idx = 3
        leftover.multimodal_inputs = None
        leftover.finished.return_value = False
        leftover.is_sr_draft = True
        mixin.waiting_queue.append(leftover)
        mixin.sr_waiting.append(leftover)
        mixin.sr_state.set(
            "old",
            SRDraftState(req_id="old", session_id="s1", req_object=leftover),
        )

        mixin._sr_wipe_all()

        self.assertEqual(mixin.waiting_queue, [])
        self.assertEqual(mixin.sr_waiting, [])
        self.assertIsNone(mixin.last_batch)
        self.assertIsNone(mixin.cur_batch)
        self.assertIsNone(mixin.chunked_req)
        mixin.tree_cache.reset.assert_called_once()
        mixin.req_to_token_pool.clear.assert_called_once()
        mixin.token_to_kv_pool_allocator.clear.assert_called_once()
        mixin.grammar_manager.clear.assert_called_once()
        self.assertEqual(mixin.sr_server.last_rpc_seq, -1)
        mixin.sr_server.drain.assert_called_once()
        self.assertIsNone(leftover.req_pool_idx)

    def test_remove_req_drops_waiting_queue(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                StandaloneRemoteDraftSchedulerMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))

        mixin = StandaloneRemoteDraftSchedulerMixin()
        mixin.sr_waiting = []
        mixin.draft_paused_reqs = []
        req = MagicMock()
        mixin.waiting_queue = [req]
        mixin.running_batch = MagicMock()
        mixin.running_batch.is_empty.return_value = True
        mixin._sr_remove_req(req)
        self.assertEqual(mixin.waiting_queue, [])

    def test_isolate_need_pauses_other_running_reqs(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                StandaloneRemoteDraftSchedulerMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))

        mixin = StandaloneRemoteDraftSchedulerMixin()
        mixin.draft_paused_reqs = []
        mixin.waiting_queue = []
        keep = MagicMock()
        keep.rid = "need"
        other = MagicMock()
        other.rid = "other"
        other.draft_is_paused = False
        running = MagicMock()
        running.is_empty.return_value = False
        running.reqs = [keep, other]
        mixin.running_batch = running
        mixin.last_batch = MagicMock()
        mixin.last_batch.is_empty.return_value = False
        mixin.last_batch.reqs = [other]

        mixin._sr_isolate_need({"need"})

        running.filter_batch.assert_called()
        _args, kwargs = running.filter_batch.call_args
        self.assertEqual(kwargs.get("keep_indices"), [0])
        self.assertTrue(other.draft_is_paused)
        self.assertIn(other, mixin.draft_paused_reqs)
        self.assertIsNone(mixin.last_batch)

    def test_isolate_need_keeps_all_rpc_rids(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                StandaloneRemoteDraftSchedulerMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))

        mixin = StandaloneRemoteDraftSchedulerMixin()
        mixin.draft_paused_reqs = []
        mixin.waiting_queue = []
        a = MagicMock()
        a.rid = "a"
        b = MagicMock()
        b.rid = "b"
        other = MagicMock()
        other.rid = "other"
        other.draft_is_paused = False
        running = MagicMock()
        running.is_empty.return_value = False
        running.reqs = [a, b, other]
        mixin.running_batch = running
        mixin.last_batch = None

        mixin._sr_isolate_need({"a", "b"})

        running.filter_batch.assert_called()
        _args, kwargs = running.filter_batch.call_args
        self.assertEqual(kwargs.get("keep_indices"), [0, 1])
        self.assertTrue(other.draft_is_paused)
        self.assertFalse(getattr(a, "draft_is_paused", False))
        self.assertFalse(getattr(b, "draft_is_paused", False))

    def test_wipe_preserves_http_waiting_queue(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                StandaloneRemoteDraftSchedulerMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))

        mixin = StandaloneRemoteDraftSchedulerMixin()
        mixin.sr_state = SRDraftStateManager()
        mixin.sr_waiting = []
        mixin.draft_paused_reqs = []
        mixin.waiting_queue = []
        mixin.running_batch = MagicMock()
        mixin.running_batch.is_empty.return_value = True
        mixin.cur_batch = object()
        mixin.last_batch = object()
        mixin.chunked_req = None
        mixin.tree_cache = MagicMock()
        mixin.req_to_token_pool = MagicMock()
        mixin.token_to_kv_pool_allocator = MagicMock()
        mixin.grammar_manager = MagicMock()
        mixin.sr_server = MagicMock()
        mixin.sr_server.last_rpc_seq = 9

        sr_req = MagicMock()
        sr_req.req_pool_idx = 3
        sr_req.multimodal_inputs = None
        sr_req.finished.return_value = False
        sr_req.is_sr_draft = True
        http_req = MagicMock()
        http_req.req_pool_idx = None
        http_req.multimodal_inputs = None
        http_req.finished.return_value = False
        http_req.is_sr_draft = False
        mixin.waiting_queue.extend([sr_req, http_req])
        mixin.sr_waiting.append(sr_req)
        mixin.sr_state.set(
            "old",
            SRDraftState(req_id="old", session_id="s1", req_object=sr_req),
        )

        mixin._sr_wipe_all()

        self.assertEqual(mixin.waiting_queue, [http_req])
        self.assertEqual(mixin.sr_waiting, [])
        self.assertIsNone(mixin.last_batch)
        self.assertIsNone(mixin.cur_batch)
        mixin.tree_cache.reset.assert_not_called()
        mixin.req_to_token_pool.clear.assert_not_called()
        mixin.token_to_kv_pool_allocator.clear.assert_not_called()
        self.assertEqual(mixin.sr_server.last_rpc_seq, -1)
        mixin.sr_server.drain.assert_called_once()

    def test_resume_http_reqs_after_isolate(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                StandaloneRemoteDraftSchedulerMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))

        mixin = StandaloneRemoteDraftSchedulerMixin()
        mixin.draft_paused_reqs = []
        mixin.waiting_queue = []
        mixin.sr_waiting = []
        http = MagicMock()
        http.rid = "http"
        http.is_sr_draft = False
        http.draft_is_paused = False
        http.req_pool_idx = 7
        sr = MagicMock()
        sr.rid = "sr"
        sr.is_sr_draft = True
        sr.draft_is_paused = False
        sr.req_pool_idx = 8
        running = MagicMock()
        running.is_empty.return_value = False
        running.reqs = [http, sr]
        mixin.running_batch = running
        mixin.last_batch = None
        parked = []
        mixin._sr_park_in_running_many = lambda reqs: parked.extend(reqs)

        mixin._sr_isolate_need({"sr"})
        self.assertTrue(http.draft_is_paused)
        self.assertIn(http, mixin.draft_paused_reqs)

        mixin._sr_resume_http_reqs()

        self.assertFalse(http.draft_is_paused)
        self.assertNotIn(http, mixin.draft_paused_reqs)
        self.assertEqual(parked, [http])
        self.assertFalse(getattr(sr, "draft_is_paused", False))

    def test_park_sr_reqs_and_run_http_batch_filters_sr(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                StandaloneRemoteDraftSchedulerMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))

        mixin = StandaloneRemoteDraftSchedulerMixin()
        mixin.draft_paused_reqs = []
        mixin.waiting_queue = []
        http = MagicMock()
        http.rid = "http"
        http.is_sr_draft = False
        sr = MagicMock()
        sr.rid = "sr"
        sr.is_sr_draft = True
        sr.draft_is_paused = False
        running = MagicMock()
        running.is_empty.return_value = True
        mixin.running_batch = running
        mixin.last_batch = MagicMock()
        mixin.last_batch.is_empty.return_value = False
        mixin.last_batch.reqs = [sr]

        mixin._sr_park_sr_reqs()
        self.assertIsNone(mixin.last_batch)

        batch = MagicMock()
        batch.reqs = [http, sr]

        def _filter(keep_indices):
            batch.reqs = [batch.reqs[i] for i in keep_indices]

        batch.filter_batch.side_effect = _filter
        mixin.get_next_batch_to_run = lambda: batch
        mixin.run_batch = MagicMock(return_value="res")
        mixin.process_batch_result = MagicMock()
        mixin.self_check_during_idle = MagicMock()
        mixin.running_batch.is_empty.return_value = True

        mixin._sr_run_http_batch()

        mixin.run_batch.assert_called_once_with(batch)
        mixin.process_batch_result.assert_called_once()
        self.assertEqual(batch.reqs, [http])
        self.assertIs(mixin.last_batch, batch)
        mixin.self_check_during_idle.assert_not_called()

    def test_ensure_window_budget_extends_length_finish(self):
        try:
            from sglang.srt.managers.schedule_batch import FINISH_LENGTH
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                StandaloneRemoteDraftSchedulerMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))

        mixin = StandaloneRemoteDraftSchedulerMixin()
        mixin.server_args = MagicMock()
        mixin.server_args.speculative_num_steps = 4
        req = MagicMock()
        req.output_ids = [1, 2, 3, 4, 5]
        sp = MagicMock()
        sp.max_new_tokens = 9
        req.sampling_params = sp
        req.finished_reason = FINISH_LENGTH(9)
        req.to_abort = False
        mixin._sr_ensure_window_budget(req, 5)
        self.assertEqual(sp.max_new_tokens, 14)
        self.assertIsNone(req.finished_reason)
        self.assertFalse(req.to_abort)


class TestDraftWindowBudget(CustomTestCase):
    def test_needed_max_new_tokens_grows_with_output(self):
        self.assertEqual(draft_token_budget(5, 4), 9)
        self.assertEqual(draft_needed_max_new_tokens(0, 5, 4, None), 9)
        self.assertEqual(draft_needed_max_new_tokens(5, 5, 4, 9), 14)
        self.assertGreater(draft_needed_max_new_tokens(28, 5, 4, 32), 32)


class TestTargetPrefillOverlap(CustomTestCase):
    """PREFILL send happens before Target GPU; STEP remains send+recv together."""

    def _import_mixin(self):
        try:
            from sglang.srt.speculative.standalone_remote.verifier.sr_target_scheduler_mixin import (
                SchedulerStandaloneRemoteTargetMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))
        return SchedulerStandaloneRemoteTargetMixin

    def _make_mixin(self, order, req, *, is_extend):
        mixin_cls = self._import_mixin()
        mixin = mixin_cls()
        mixin.tp_size = 1
        mixin.tp_rank = 0
        mixin.sr_session_id = "sess"
        mixin.sr_rpc_seq = 0
        mixin.sr_pending = {}
        mixin._sr_inflight = None
        mixin._engine_paused = False
        mixin.current_scheduler_metrics_enabled = False
        mixin.last_batch = None
        mixin.cur_batch = None

        server_args = MagicMock()
        server_args.speculative_num_steps = 4
        server_args.speculative_num_draft_tokens = 5
        server_args.speculative_eagle_topk = 1
        server_args.standalone_remote_max_batch_size = 32
        mixin.server_args = server_args

        running = MagicMock()
        running.batch_size.return_value = 1
        mixin.running_batch = running

        client = MagicMock()

        def send_batch(batch, mm=None):
            order.append("send")

        def recv_batch(session_id, rpc_seq):
            order.append("recv")
            return SRBatchReply(
                session_id=session_id,
                rpc_seq=rpc_seq,
                reqs=[
                    SRDraftReply(
                        rid=req.rid,
                        step_id=int(getattr(req, "sr_step_id", 0) or 0),
                        base_committed_len=len(req.origin_input_ids)
                        + len(req.output_ids or []),
                        draft_tokens=[11, 12],
                    )
                ],
            )

        client.send_batch.side_effect = send_batch
        client.recv_batch.side_effect = recv_batch
        mixin.sr_client = client

        batch = MagicMock()
        batch.reqs = [req]
        batch.is_extend_in_batch = False
        fm = MagicMock()
        fm.is_extend.return_value = is_extend
        batch.forward_mode = fm
        batch.batch_size.return_value = 1

        mixin._init_sr_target = lambda: None
        mixin.recv_requests = self._stop_after_one_recv()
        mixin.process_input_requests = lambda _reqs: None
        mixin.get_next_batch_to_run = lambda: batch
        mixin.process_batch_result = lambda _b, _r: None
        mixin.self_check_during_idle = lambda: None
        mixin.self_check_during_busy = lambda: None

        def run_batch(_batch):
            order.append("gpu")
            return MagicMock()

        mixin.run_batch = run_batch
        return mixin, client

    @staticmethod
    def _stop_after_one_recv():
        class _StopLoop(Exception):
            pass

        state = {"n": 0}

        def recv_requests():
            state["n"] += 1
            if state["n"] > 1:
                raise _StopLoop()
            return []

        recv_requests.StopLoop = _StopLoop
        return recv_requests

    @staticmethod
    def _make_req():
        req = MagicMock()
        req.rid = "r1"
        req.origin_input_ids = [1, 2, 3]
        req.output_ids = []
        req.sampling_params = None
        req.multimodal_inputs = None
        req.sr_step_id = 0
        req.finished.return_value = False
        req.cur_drafts = []
        return req

    def test_extend_sends_prefill_before_gpu(self):
        order = []
        req = self._make_req()
        mixin, _client = self._make_mixin(order, req, is_extend=True)
        with self.assertRaises(mixin.recv_requests.StopLoop):
            mixin.event_loop_normal_standalone_remote_target()
        self.assertEqual(order, ["send", "gpu", "recv"])
        self.assertIsNone(mixin._sr_inflight)

    def test_step_stays_blocking_send_recv(self):
        order = []
        req = self._make_req()
        mixin, _client = self._make_mixin(order, req, is_extend=False)
        with self.assertRaises(mixin.recv_requests.StopLoop):
            mixin.event_loop_normal_standalone_remote_target()
        self.assertEqual(order, ["send", "recv", "gpu", "send", "recv"])
        self.assertIsNone(mixin._sr_inflight)

    def test_reset_clears_inflight(self):
        mixin_cls = self._import_mixin()
        mixin = mixin_cls()
        mixin.sr_session_id = "old"
        mixin.sr_rpc_seq = 3
        mixin.sr_pending = {"r1": SRPendingEntry(0, 3)}
        mixin._sr_inflight = object()
        client = MagicMock()
        mixin.sr_client = client
        mixin.reset_standalone_remote_target_state()
        self.assertIsNone(mixin._sr_inflight)
        self.assertEqual(mixin.sr_rpc_seq, 0)
        self.assertEqual(mixin.sr_pending, {})
        client._drain.assert_called_once()


class TestOverlappedPrefillShift(CustomTestCase):
    """PREFILL drafts start at the prompt; Target already sampled T0 during extend."""

    def test_drop_matched_first_token(self):
        # Same-model greedy: D0 == T0, verify [D1, D2, D3, D4] → accept len ~5.
        self.assertEqual(
            shift_overlapped_prefill_drafts([10], [10, 11, 12, 13, 14]),
            [11, 12, 13, 14],
        )

    def test_drop_common_prefix(self):
        self.assertEqual(
            shift_overlapped_prefill_drafts([10, 11], [10, 11, 12, 13]),
            [12, 13],
        )

    def test_mismatch_discards_window(self):
        self.assertIsNone(shift_overlapped_prefill_drafts([99], [10, 11, 12, 13, 14]))

    def test_empty_output_keeps_window(self):
        self.assertEqual(
            shift_overlapped_prefill_drafts([], [10, 11, 12, 13, 14]),
            [10, 11, 12, 13, 14],
        )

    def test_empty_drafts(self):
        self.assertEqual(shift_overlapped_prefill_drafts([10], []), [])

    def test_drop_duplicate_verify_root(self):
        self.assertEqual(drop_duplicate_root_draft(10, [10, 11, 12, 13, 14]), [11, 12, 13, 14])
        self.assertEqual(drop_duplicate_root_draft(10, [11, 12, 13]), [11, 12, 13])
        self.assertEqual(drop_duplicate_root_draft(None, [10, 11]), [10, 11])


class TestPrefillHardResetRidReuse(CustomTestCase):
    def test_prefill_always_hard_reset(self):
        d = decide_draft_action(
            SRAction.PREFILL,
            session_id="s",
            rpc_seq=10,
            last_session_id="s",
            last_rpc_seq=9,
            last_step_id=8,
            last_base_committed_len=20,
            step_id=0,
            base_committed_len=4,
            has_state=True,
        )
        self.assertEqual(d, DraftDecision.HARD_RESET)


class TestStandaloneRemoteTree(CustomTestCase):
    def _import_worker(self):
        try:
            from sglang.srt.speculative.standalone_remote.verifier.sr_worker import (
                StandaloneRemoteWorker,
            )
        except ImportError as e:
            self.skipTest(str(e))
        return StandaloneRemoteWorker

    def _import_mixin(self):
        try:
            from sglang.srt.speculative.standalone_remote.verifier.sr_target_scheduler_mixin import (
                SchedulerStandaloneRemoteTargetMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))
        return SchedulerStandaloneRemoteTargetMixin

    def test_no_fake_bush_helper(self):
        worker_cls = self._import_worker()
        self.assertFalse(hasattr(worker_cls, "_construct_tree_structure_general"))

    def test_advance_tree_draft_positions_increments_mrope(self):
        if torch is None:
            self.skipTest("torch not available")
        from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
            advance_tree_draft_positions,
        )

        positions = torch.tensor([10, 10], dtype=torch.int64)
        mrope = torch.tensor([[100, 100], [101, 101], [102, 102]], dtype=torch.int64)
        advance_tree_draft_positions(positions, mrope)
        self.assertEqual(positions.tolist(), [11, 11])
        self.assertEqual(mrope.tolist(), [[101, 101], [102, 102], [103, 103]])
        advance_tree_draft_positions(positions, None)
        self.assertEqual(positions.tolist(), [12, 12])
        advance_tree_draft_positions(None, mrope)
        self.assertEqual(mrope[0].tolist(), [102, 102])

    def test_slice_decode_batch_row(self):
        if torch is None:
            self.skipTest("torch not available")
        from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
            slice_decode_batch_row,
        )

        logits = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.assertEqual(slice_decode_batch_row(logits, 0, 3).tolist(), [[1.0, 2.0]])
        self.assertEqual(slice_decode_batch_row(logits, 1, 3).tolist(), [[3.0, 4.0]])
        self.assertEqual(slice_decode_batch_row(logits, 2, 3).tolist(), [[5.0, 6.0]])
        # Multi-req must not take the last row of the whole batch.
        self.assertNotEqual(
            slice_decode_batch_row(logits, 0, 3).tolist(), [[5.0, 6.0]]
        )
        # Single-req EXTEND: longer leading dim → last row.
        extend = torch.tensor([[1.0], [2.0], [9.0]])
        self.assertEqual(slice_decode_batch_row(extend, 0, 1).tolist(), [[9.0]])
        # Multi-req packed EXTEND hidden: last token of each seq via token_lens.
        # Matches Draft logs: logits [4, vocab], hidden [sum_tokens, hidden].
        packed = torch.arange(10, dtype=torch.float32).unsqueeze(-1)
        lens = [2, 3, 1, 4]
        self.assertEqual(sum(lens), packed.shape[0])
        self.assertEqual(slice_decode_batch_row(packed, 0, 4, lens).tolist(), [[1.0]])
        self.assertEqual(slice_decode_batch_row(packed, 1, 4, lens).tolist(), [[4.0]])
        self.assertEqual(slice_decode_batch_row(packed, 2, 4, lens).tolist(), [[5.0]])
        self.assertEqual(slice_decode_batch_row(packed, 3, 4, lens).tolist(), [[9.0]])
        # Logits stay [bs, vocab] even when hidden is packed.
        logits_bs = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
        self.assertEqual(
            slice_decode_batch_row(logits_bs, 1, 4, lens).tolist(), [[20.0]]
        )
        # Multi-req EXTEND without lens / mismatched sum cannot be mapped.
        self.assertIsNone(slice_decode_batch_row(extend, 0, 2))
        self.assertIsNone(slice_decode_batch_row(packed, 0, 4, [1, 1, 1, 1]))
        self.assertIsNone(slice_decode_batch_row(logits, 3, 3))
        self.assertIsNone(slice_decode_batch_row(None, 0, 1))

    def test_expand_batch_splits_rows(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_tree_drafter import (
                SRTreeDrafter,
            )
        except ImportError as e:
            self.skipTest(str(e))
        if torch is None:
            self.skipTest("torch not available")

        drafter = SRTreeDrafter.__new__(SRTreeDrafter)
        parent = torch.tensor([[-1, 0], [-1, 1]], dtype=torch.int64)
        index = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64)
        tokens = torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int64)
        drafter._expand_tree = lambda reqs, *seed: (parent, index, tokens)

        def _seed(token_id):
            return (
                torch.ones(1, 2),
                torch.ones(1, 2, dtype=torch.int64),
                torch.ones(1, 4),
                torch.tensor([token_id], dtype=torch.int64),
            )

        r0 = MagicMock()
        r0.rid = "a"
        r0.req_pool_idx = 0
        r0.sr_tree_seed = _seed(1)
        r1 = MagicMock()
        r1.rid = "b"
        r1.req_pool_idx = 1
        r1.sr_tree_seed = _seed(2)
        windows = drafter.expand_batch([r0, r1])
        self.assertEqual(windows[0][0], [10, 11, 12])
        self.assertEqual(windows[0][1], [-1, 0])
        self.assertEqual(windows[1][0], [20, 21, 22])
        self.assertEqual(windows[1][1], [-1, 1])

    def test_assemble_keeps_organized_token_width(self):
        if torch is None:
            self.skipTest("torch not available")
        from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
            assemble_draft_rows,
        )

        tokens = list(range(100, 107))
        parents, indices, dt = assemble_draft_rows(
            [tokens],
            [[-1, 0, 1, 2]],
            [list(range(7))],
            topk=2,
            spec_steps=3,
            num_draft_tokens=8,
        )
        self.assertEqual(list(dt.shape), [1, 7])
        self.assertEqual(dt[0].tolist(), tokens)
        self.assertEqual(parents[0, :4].tolist(), [-1, 0, 1, 2])
        self.assertEqual(indices[0, :7].tolist(), list(range(7)))

    def test_missing_topology_uses_chain_not_repeated_tokens(self):
        if torch is None:
            self.skipTest("torch not available")
        from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
            assemble_draft_rows,
            chain_tree_structure,
        )

        parents, indices, dt = assemble_draft_rows(
            [[11, 12, 13]],
            [None],
            [None],
            topk=2,
            spec_steps=3,
            num_draft_tokens=4,
        )
        self.assertEqual(dt[0].tolist(), [11, 12, 13])
        chain_p, chain_i = chain_tree_structure(4, 3)
        self.assertEqual(parents[0].tolist(), chain_p.tolist())
        self.assertEqual(indices[0].tolist(), chain_i.tolist())
        self.assertNotEqual(dt[0, 0].item(), dt[0, 1].item())

    def test_build_request_sends_speculative_num_draft_tokens(self):
        mixin_cls = self._import_mixin()
        mixin = mixin_cls()
        mixin.server_args = MagicMock()
        mixin.server_args.speculative_num_steps = 4
        mixin.server_args.speculative_num_draft_tokens = 8
        req = MagicMock()
        req.rid = "r1"
        req.origin_input_ids = [1, 2, 3]
        req.output_ids = [9]
        req.sr_step_id = 2
        req.sampling_params = None
        req.multimodal_inputs = None
        dreq, _mm = mixin._build_sr_request(req, SRAction.STEP, include_full_context=False)
        self.assertEqual(dreq.num_draft_tokens, 8)
        dreq_f, _ = mixin._build_sr_request(req, SRAction.FINISH, include_full_context=False)
        self.assertEqual(dreq_f.num_draft_tokens, 0)

    def test_ingest_committed_runs_one_chain_decode_per_missing_token(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                StandaloneRemoteDraftSchedulerMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))
        mixin = StandaloneRemoteDraftSchedulerMixin()
        req = MagicMock()
        req.origin_input_ids = [1, 2, 3]
        req.output_ids = [10]
        req.kv_committed_len = 3
        req.kv_allocated_len = 3
        calls = []

        def run_ready(reqs, capture_tree_seed=False):
            calls.append(
                (
                    list(reqs[0].output_ids),
                    int(reqs[0].draft_tokens_target),
                    bool(capture_tree_seed),
                )
            )
            reqs[0].kv_committed_len = 3 + len(reqs[0].output_ids)
            reqs[0].output_ids = list(reqs[0].output_ids) + [99]

        mixin._sr_ensure_window_budget = lambda *_a, **_k: None
        mixin._sr_run_until_ready = run_ready
        mixin._sr_ingest_committed(req)
        self.assertEqual(calls, [([10], 1, True)])
        self.assertEqual(req.output_ids, [10])
        self.assertEqual(req.draft_generation_start_len, 1)

    def test_ingest_committed_batch_fuses_same_tail_len(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                StandaloneRemoteDraftSchedulerMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))
        mixin = StandaloneRemoteDraftSchedulerMixin()
        calls = []

        def run_ready(reqs, capture_tree_seed=False):
            calls.append(([r.rid for r in reqs], bool(capture_tree_seed)))
            for r in reqs:
                r.kv_committed_len = len(r.origin_input_ids) + len(r.output_ids)

        mixin._sr_ensure_window_budget = lambda *_a, **_k: None
        mixin._sr_run_until_ready = run_ready

        a = MagicMock()
        a.rid = "a"
        a.origin_input_ids = [1, 2, 3]
        a.output_ids = [10]
        a.kv_committed_len = 3
        b = MagicMock()
        b.rid = "b"
        b.origin_input_ids = [4, 5, 6]
        b.output_ids = [20]
        b.kv_committed_len = 3
        mixin._sr_ingest_committed_batch([a, b])
        self.assertEqual(calls, [(["a", "b"], True)])
        self.assertEqual(a.output_ids, [10])
        self.assertEqual(b.output_ids, [20])

    def test_ingest_committed_batch_uneven_tails(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                StandaloneRemoteDraftSchedulerMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))
        mixin = StandaloneRemoteDraftSchedulerMixin()
        calls = []

        def run_ready(reqs, capture_tree_seed=False):
            calls.append([r.rid for r in reqs])
            for r in reqs:
                r.kv_committed_len = len(r.origin_input_ids) + len(r.output_ids)

        mixin._sr_ensure_window_budget = lambda *_a, **_k: None
        mixin._sr_run_until_ready = run_ready

        a = MagicMock()
        a.rid = "a"
        a.origin_input_ids = [1, 2, 3]
        a.output_ids = [10, 11]
        a.kv_committed_len = 3
        b = MagicMock()
        b.rid = "b"
        b.origin_input_ids = [4, 5, 6]
        b.output_ids = [20]
        b.kv_committed_len = 3
        mixin._sr_ingest_committed_batch([a, b])
        self.assertEqual(calls, [["a", "b"], ["a"]])
        self.assertEqual(a.output_ids, [10, 11])
        self.assertEqual(b.output_ids, [20])

    def test_align_append_n_does_not_reprefill(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                StandaloneRemoteDraftSchedulerMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))
        mixin = StandaloneRemoteDraftSchedulerMixin()
        mixin.sr_kv = MagicMock()
        mixin.sr_kv.get_prefix_len.return_value = 3
        reprefill = []
        mixin._sr_reprefill = lambda *_a, **_k: reprefill.append(1)
        req = MagicMock()
        req.origin_input_ids = [1, 2, 3]
        req.output_ids = [4]
        req.sr_padded_ids = [1, 2, 3]
        req.kv_allocated_len = 4
        dreq = SRDraftRequest(
            rid="r",
            step_id=2,
            base_committed_len=7,
            committed_ids=[4, 5, 6, 7],
            num_draft_tokens=8,
        )
        mixin._sr_align(req, dreq, MagicMock())
        self.assertEqual(reprefill, [])
        self.assertEqual(req.output_ids, [4, 5, 6, 7])
        self.assertEqual(req.draft_generation_start_len, 4)
        self.assertEqual(req.draft_tokens_target, 8)

    def test_handle_batch_chain_fuses_gpu(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                StandaloneRemoteDraftSchedulerMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))
        mixin = StandaloneRemoteDraftSchedulerMixin()
        mixin.sr_state = SRDraftStateManager()
        mixin.sr_server = MagicMock()
        mixin.sr_server.last_rpc_seq = -1
        mixin.sr_tree_drafter = None
        produce_calls = []

        def fake_prepare(dreq, *_a, **_k):
            req = MagicMock()
            req.rid = dreq.rid
            return None, req

        def fake_produce(action, pairs):
            produce_calls.append((action, [req.rid for _, req in pairs]))
            return [([10, 11], None, None)] * len(pairs)

        mixin._sr_prepare_one = fake_prepare
        mixin._sr_produce_windows = fake_produce
        batch = SRBatchRequest(
            session_id="s",
            rpc_seq=1,
            action=SRAction.STEP,
            reqs=[
                SRDraftRequest(
                    rid="a", step_id=1, base_committed_len=4, num_draft_tokens=5
                ),
                SRDraftRequest(
                    rid="b", step_id=1, base_committed_len=4, num_draft_tokens=5
                ),
            ],
        )
        reply = mixin._sr_handle_batch(batch, {})
        self.assertEqual(produce_calls, [(SRAction.STEP, ["a", "b"])])
        self.assertEqual([r.rid for r in reply.reqs], ["a", "b"])
        self.assertEqual(reply.reqs[0].draft_tokens, [10, 11])
        self.assertEqual(reply.reqs[1].draft_tokens, [10, 11])

    def test_new_session_batch_wipes_once(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                StandaloneRemoteDraftSchedulerMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))
        mixin = StandaloneRemoteDraftSchedulerMixin()
        mixin.sr_state = SRDraftStateManager()
        mixin.sr_state.session_id = "old"
        mixin.sr_server = MagicMock()
        mixin.sr_server.last_rpc_seq = 9
        wipes = []
        mixin._sr_wipe_all = lambda: wipes.append("wipe")
        mixin._sr_prepare_one = lambda dreq, *_a, **_k: (None, MagicMock(rid=dreq.rid))
        mixin._sr_produce_windows = lambda action, pairs: [
            ([1], None, None)
        ] * len(pairs)
        batch = SRBatchRequest(
            session_id="new",
            rpc_seq=1,
            action=SRAction.STEP,
            reqs=[
                SRDraftRequest(rid="a", step_id=1, base_committed_len=0),
                SRDraftRequest(rid="b", step_id=1, base_committed_len=0),
            ],
        )
        mixin._sr_handle_batch(batch, {})
        self.assertEqual(wipes, ["wipe"])

    def test_tree_mode_skips_chain_align(self):
        mixin_cls = self._import_mixin()
        mixin = mixin_cls()
        mixin.server_args = MagicMock()
        mixin.server_args.speculative_eagle_topk = 2
        called = {"n": 0}

        def boom(*_a, **_k):
            called["n"] += 1

        mixin._sr_shift_prefill_replies = boom
        mixin._sr_drop_root_duplicate_replies = boom
        mixin._sr_maybe_align_chain_replies([], {}, prefill=True)
        mixin._sr_maybe_align_chain_replies([], {}, prefill=False)
        self.assertEqual(called["n"], 0)
        mixin.server_args.speculative_eagle_topk = 1
        mixin._sr_maybe_align_chain_replies([], {}, prefill=True)
        self.assertEqual(called["n"], 2)


if __name__ == "__main__":
    unittest.main()
