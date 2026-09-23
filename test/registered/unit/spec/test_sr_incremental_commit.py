import unittest

from sglang.srt.speculative.standalone_remote.sr_commit import (
    SR_PROTOCOL_VERSION,
    CommitOutcome,
    RecoveryRoute,
    ack_matches,
    commit_fingerprint,
    grammar_committed_ids,
    inspect_commit,
    mm_items_complete,
    reply_stops_speculation,
    route_snapshot_recovery,
    supported_protocol,
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


def _snap(output, prompt, version, step=1):
    return SRDraftRequest(
        rid="r",
        step_id=step,
        base_committed_len=len(prompt) + len(output),
        committed_ids=list(output),
        padded_input_ids=list(prompt),
        sampling_params={"temperature": 0},
        commit_mode="snapshot",
        commit_version=version,
        requires_mm=False,
    )


def _delta(history, new, prompt_len, base_version, version, step=2):
    return SRDraftRequest(
        rid="r",
        step_id=step,
        base_committed_len=prompt_len + len(history) + len(new),
        commit_mode="delta",
        commit_version=version,
        base_commit_version=base_version,
        base_output_len=len(history),
        delta_ids=list(new),
    )


class TestSRIncrementalCommit(unittest.TestCase):
    def test_missing_protocol_version_is_not_v2(self):
        raw = SRBatchRequest(
            session_id="s", rpc_seq=1, action=SRAction.STEP, reqs=[]
        ).to_dict()
        self.assertNotIn("protocol_version", raw)
        loaded = SRBatchRequest.from_dict(raw)
        self.assertIsNone(loaded.protocol_version)
        self.assertFalse(supported_protocol(loaded.protocol_version))
        raw["protocol_version"] = SR_PROTOCOL_VERSION
        self.assertTrue(
            supported_protocol(SRBatchRequest.from_dict(raw).protocol_version)
        )

    def test_snapshot_and_delta_roundtrip_and_exclusion(self):
        prompt, output = [1, 2], [9]
        snap = _snap(output, prompt, 1)
        again = SRDraftRequest.from_dict(snap.to_dict())
        self.assertEqual(again.committed_ids, output)
        self.assertIsNone(again.delta_ids)
        verdict = inspect_commit(
            snap,
            action=SRAction.PREFILL,
            current_version=None,
            current_output=None,
            prompt_len=len(prompt),
            cached_fingerprint=None,
            has_state=False,
            state_trusted=True,
        )
        self.assertEqual(verdict.outcome, CommitOutcome.APPLY)
        self.assertEqual(verdict.output_ids, output)

        bad = _snap(output, prompt, 1)
        bad.base_committed_len = 1
        rejected = inspect_commit(
            bad,
            action=SRAction.PREFILL,
            current_version=None,
            current_output=None,
            prompt_len=2,
            cached_fingerprint=None,
            has_state=False,
            state_trusted=True,
        )
        self.assertEqual(rejected.outcome, CommitOutcome.REJECT)
        self.assertEqual(rejected.reason, "snapshot_length")

        delta = _delta([9], [3, 4], prompt_len=2, base_version=1, version=2)
        self.assertNotIn("committed_ids", delta.to_dict())
        applied = inspect_commit(
            delta,
            action=SRAction.STEP,
            current_version=1,
            current_output=[9],
            prompt_len=2,
            cached_fingerprint=None,
            has_state=True,
            state_trusted=True,
        )
        self.assertEqual(applied.outcome, CommitOutcome.APPLY)
        self.assertEqual(applied.output_ids, [9, 3, 4])

    def test_resend_with_old_base_hits_cache(self):
        prompt_len = 4
        first = _delta([], [7], prompt_len, base_version=7, version=8, step=3)
        # After a successful apply the cursor is version 8. The retry still
        # carries base=7 because that was the base of version 8.
        cached = commit_fingerprint(first)
        retry = _delta([], [7], prompt_len, base_version=7, version=8, step=3)
        verdict = inspect_commit(
            retry,
            action=SRAction.STEP,
            current_version=8,
            current_output=[7],
            prompt_len=prompt_len,
            cached_fingerprint=cached,
            has_state=True,
            state_trusted=True,
        )
        self.assertEqual(verdict.outcome, CommitOutcome.CACHE)
        self.assertNotEqual(verdict.outcome, CommitOutcome.NEED_SNAPSHOT)

        other = _delta([], [8], prompt_len, base_version=7, version=8, step=3)
        mismatch = inspect_commit(
            other,
            action=SRAction.STEP,
            current_version=8,
            current_output=[7],
            prompt_len=prompt_len,
            cached_fingerprint=cached,
            has_state=True,
            state_trusted=True,
        )
        self.assertEqual(mismatch.outcome, CommitOutcome.REJECT)
        self.assertEqual(mismatch.reason, "commit_mismatch")

    def test_zero_delta_is_a_new_apply(self):
        req = _delta([1, 2], [], prompt_len=3, base_version=4, version=5, step=9)
        verdict = inspect_commit(
            req,
            action=SRAction.STEP,
            current_version=4,
            current_output=[1, 2],
            prompt_len=3,
            cached_fingerprint=None,
            has_state=True,
            state_trusted=True,
        )
        self.assertEqual(verdict.outcome, CommitOutcome.APPLY)
        self.assertEqual(verdict.output_ids, [1, 2])

    def test_delta_base_mismatch_needs_snapshot(self):
        req = _delta([1], [2], prompt_len=1, base_version=3, version=9)
        verdict = inspect_commit(
            req,
            action=SRAction.STEP,
            current_version=4,
            current_output=[1],
            prompt_len=1,
            cached_fingerprint=None,
            has_state=True,
            state_trusted=True,
        )
        self.assertEqual(verdict.outcome, CommitOutcome.NEED_SNAPSHOT)
        self.assertEqual(verdict.reason, "base_version")

    def test_older_version_rejected(self):
        req = _delta([1], [2], prompt_len=1, base_version=1, version=2)
        verdict = inspect_commit(
            req,
            action=SRAction.STEP,
            current_version=5,
            current_output=[1, 2, 3],
            prompt_len=1,
            cached_fingerprint=None,
            has_state=True,
            state_trusted=True,
        )
        self.assertEqual(verdict.outcome, CommitOutcome.REJECT)
        self.assertEqual(verdict.reason, "stale_version")

    def test_untrusted_kv_snapshot_forces_reprefill(self):
        snap = _snap([4, 5], [1], version=3)
        verdict = inspect_commit(
            snap,
            action=SRAction.STEP,
            current_version=2,
            current_output=[4],
            prompt_len=1,
            cached_fingerprint=None,
            has_state=True,
            state_trusted=False,
        )
        self.assertEqual(verdict.outcome, CommitOutcome.FORCE_RESET)
        self.assertEqual(verdict.output_ids, [4, 5])
        self.assertEqual(
            route_snapshot_recovery(kv_trusted=False), RecoveryRoute.REPREFILL
        )
        self.assertEqual(
            route_snapshot_recovery(kv_trusted=True), RecoveryRoute.ALIGN
        )
        self.assertEqual(
            route_snapshot_recovery(kv_trusted=False, degraded=True),
            RecoveryRoute.BLOCKED,
        )
        self.assertEqual(
            route_snapshot_recovery(kv_trusted=True, poisoned=True),
            RecoveryRoute.BLOCKED,
        )
        self.assertEqual(
            route_snapshot_recovery(kv_trusted=True, completion_unknown=True),
            RecoveryRoute.BLOCKED,
        )

    def test_ack_and_old_reply_policy(self):
        pending = SRPendingEntry(
            step_id=1,
            base_committed_len=6,
            commit_version=4,
            sent_output_len=2,
            protocol_version=SR_PROTOCOL_VERSION,
        )
        good = SRDraftReply(
            rid="r",
            step_id=1,
            base_committed_len=6,
            status=SRReplyStatus.EMPTY,
            ack_commit_version=4,
            ack_output_len=2,
        )
        self.assertEqual(ack_matches(good, pending), (True, None))
        good.ack_output_len = 3
        self.assertEqual(ack_matches(good, pending)[1], "ack_len")
        self.assertTrue(
            reply_stops_speculation(None, session_rpc_matched=True)
        )
        self.assertFalse(
            reply_stops_speculation(None, session_rpc_matched=False)
        )
        self.assertFalse(
            reply_stops_speculation(SR_PROTOCOL_VERSION, session_rpc_matched=True)
        )
        reply = SRBatchReply(session_id="s", rpc_seq=1, reqs=[])
        self.assertIsNone(SRBatchReply.from_dict(reply.to_dict()).protocol_version)

    def test_chain_grammar_uses_authoritative_history(self):
        # Snapshot folded the committed output into origin and cleared output.
        tokens = grammar_committed_ids(
            [7, 8, 9],
            output_ids=[],
            origin_ids=[1, 2, 7, 8, 9],
            padded_ids=[1, 2],
            tree_mode=False,
        )
        self.assertEqual(tokens, [7, 8, 9])
        folded = grammar_committed_ids(
            None,
            output_ids=[],
            origin_ids=[1, 2, 7, 8, 9],
            padded_ids=[1, 2],
            tree_mode=True,
        )
        self.assertEqual(folded, [7, 8, 9])
        self.assertFalse(mm_items_complete([{"feature": None}]))
        self.assertTrue(mm_items_complete([{"feature": object(), "precomputed_embeddings": None}]))


if __name__ == "__main__":
    unittest.main()
