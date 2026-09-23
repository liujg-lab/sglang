import time
import unittest
from types import SimpleNamespace

from sglang.srt.speculative.standalone_remote.sr_commit import (
    SR_PROTOCOL_VERSION,
    CommitOutcome,
    CommittedPrefixView,
    PendingCommit,
    RecoveryRoute,
    ack_matches,
    commit_fingerprint,
    grammar_committed_ids,
    inspect_commit,
    make_prefix_stamp,
    mm_items_complete,
    rebuild_committed_output,
    retarget_stamp_version,
    reply_stops_speculation,
    route_snapshot_recovery,
    stamp_matches,
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
        self.assertIsNone(applied.output_ids)
        self.assertEqual(applied.delta_ids, (3, 4))

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
        self.assertIsNone(verdict.output_ids)
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
        self.assertEqual(verdict.delta_ids, ())
        self.assertIsNone(verdict.output_ids)

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

    def test_fingerprint_skipped_until_same_version(self):
        import sglang.srt.speculative.standalone_remote.sr_commit as mod

        calls = []
        real = mod.commit_fingerprint

        def wrapped(req):
            calls.append(int(req.commit_version))
            return real(req)

        mod.commit_fingerprint = wrapped
        try:
            stale = _delta([1], [2], prompt_len=1, base_version=1, version=2)
            inspect_commit(
                stale,
                action=SRAction.STEP,
                current_version=5,
                current_output=[1, 2, 3],
                prompt_len=1,
                cached_fingerprint=None,
                has_state=True,
                state_trusted=True,
            )
            mismatch = _delta([1], [2], prompt_len=1, base_version=3, version=9)
            inspect_commit(
                mismatch,
                action=SRAction.STEP,
                current_version=4,
                current_output=[1],
                prompt_len=1,
                cached_fingerprint=None,
                has_state=True,
                state_trusted=True,
            )
            self.assertEqual(calls, [])
            fresh = _delta([9], [3], prompt_len=2, base_version=1, version=2)
            applied = inspect_commit(
                fresh,
                action=SRAction.STEP,
                current_version=1,
                current_output=[9],
                prompt_len=2,
                cached_fingerprint=None,
                has_state=True,
                state_trusted=True,
            )
            self.assertEqual(calls, [])
            self.assertEqual(applied.delta_ids, (3,))
            cached = real(fresh)
            inspect_commit(
                fresh,
                action=SRAction.STEP,
                current_version=2,
                current_output=[9, 3],
                prompt_len=2,
                cached_fingerprint=cached,
                has_state=True,
                state_trusted=True,
            )
            self.assertEqual(calls, [2])
        finally:
            mod.commit_fingerprint = real

    def test_view_prefix_survives_append_and_snapshot_replace(self):
        base = [7, 8]
        view = CommittedPrefixView(base, len(base), (9,))
        self.assertEqual(list(view), [7, 8, 9])
        base.append(9)
        self.assertEqual(len(view), 3)
        self.assertEqual(view[0], 7)
        self.assertEqual(view[2], 9)
        replacement = [1, 2, 3]
        self.assertIsNot(replacement, base)
        self.assertEqual(list(view), [7, 8, 9])
        tokens = grammar_committed_ids(
            view,
            output_ids=[7, 8, 9, 100],
            origin_ids=[1],
            padded_ids=[1],
            tree_mode=True,
        )
        self.assertEqual(tokens, [7, 8, 9])

    def test_stamp_uses_identity_and_empty_delta_does_not_create_one(self):
        req = SimpleNamespace(
            origin_input_ids=[1, 2],
            output_ids=[9],
            sr_prefix_revision=4,
        )
        stamp = make_prefix_stamp(req, 3)
        self.assertTrue(
            stamp_matches(
                stamp,
                version=3,
                req=req,
                origin=req.origin_input_ids,
                output=req.output_ids,
                revision=4,
            )
        )
        copy = [9]
        req.output_ids = copy
        self.assertFalse(
            stamp_matches(
                stamp,
                version=3,
                req=req,
                origin=req.origin_input_ids,
                output=req.output_ids,
                revision=4,
            )
        )
        self.assertIsNone(
            retarget_stamp_version(
                None,
                4,
                req=req,
                origin=req.origin_input_ids,
                output=req.output_ids,
                revision=4,
            )
        )

    def test_fast_path_touches_only_the_window(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
                prefix_window_tokens,
            )
        except ImportError as exc:
            self.skipTest(str(exc))

        class CountingList(list):
            def __init__(self, data):
                super().__init__(data)
                self.touches = 0

            def __getitem__(self, item):
                if isinstance(item, slice):
                    start, stop, step = item.indices(len(self))
                    self.touches += len(range(start, stop, step))
                else:
                    self.touches += 1
                return super().__getitem__(item)

            def __iter__(self):
                self.touches += len(self)
                return super().__iter__()

        lengths = (256, 4096, 16384, 32768)
        deltas = (1, 4, 8)
        for n in lengths:
            for width in deltas:
                origin = CountingList(range(n))
                output = CountingList([])
                acked = CountingList(range(n))
                req = SimpleNamespace(
                    rid="r",
                    origin_input_ids=origin,
                    output_ids=output,
                    sr_prefix_revision=1,
                    step_id=2,
                    base_committed_len=n + width,
                    num_draft_tokens=4,
                    has_mm=False,
                    requires_mm=False,
                    commit_mode="delta",
                    commit_version=2,
                    base_commit_version=1,
                    base_output_len=n,
                    delta_ids=list(range(n, n + width)),
                    committed_ids=None,
                    padded_input_ids=None,
                    sampling_params=None,
                    commit_tree_version=None,
                    commit_tree_base_committed_len=None,
                    commit_candidate_indices=None,
                )
                stamp = make_prefix_stamp(req, 1)
                started = time.perf_counter()
                self.assertTrue(
                    stamp_matches(
                        stamp,
                        version=1,
                        req=req,
                        origin=origin,
                        output=output,
                        revision=1,
                    )
                )
                fingerprint = commit_fingerprint(req)
                window = prefix_window_tokens(origin, output, base=n)
                output.extend(req.delta_ids)
                pending = PendingCommit(
                    old_version=1,
                    old_output_len=n,
                    delta_ids=tuple(req.delta_ids),
                    expected_len=n + width,
                    fingerprint=fingerprint,
                    candidate_stamp=make_prefix_stamp(req, 2),
                )
                acked.extend(pending.delta_ids)
                elapsed_us = (time.perf_counter() - started) * 1e6
                print(
                    f"fast commit n={n} delta={width} us={elapsed_us:.1f} "
                    f"origin_touches={origin.touches}"
                )
                self.assertEqual(len(window), min(32, n))
                self.assertLessEqual(origin.touches, 32)
                self.assertEqual(acked.touches, 0)
                self.assertEqual(pending.fingerprint, fingerprint)
                slow = CountingList(range(n))
                started = time.perf_counter()
                rebuilt = rebuild_committed_output(slow, req.delta_ids)
                slow_us = (time.perf_counter() - started) * 1e6
                print(f"slow rebuild n={n} delta={width} us={slow_us:.1f}")
                self.assertGreaterEqual(slow.touches, n)
                self.assertEqual(rebuilt[:n], list(range(n)))
                self.assertEqual(rebuilt[n:], list(req.delta_ids))


class TestSRDeltaFastPath(unittest.TestCase):
    def test_fast_prepare_source_does_not_rebuild_history(self):
        from pathlib import Path

        src = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_draft_scheduler_mixin.py"
        ).read_text(encoding="utf-8")
        body = src[
            src.index("def _sr_prepare_delta_fast") : src.index(
                "def _sr_stage_slow_commit"
            )
        ]
        self.assertNotIn("snapshot_sr_align", body)
        self.assertNotIn("find_fork_point", body)
        self.assertNotIn("rebuild_committed_output", body)
        fin = src[
            src.index("def _sr_finalize_v2_commit") : src.index(
                "def _sr_protocol_mismatch_reply"
            )
        ]
        self.assertNotIn('list(getattr(req, "sr_commit_output_ids"', fin)
        self.assertIn("state.pending_commit = None", fin)
    def _mixin(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_draft_state import (
            SRDraftState,
            SRDraftStateManager,
        )

        try:
            import torch.nn  # noqa: F401
        except ImportError:
            mixin = self._bind_mixin_methods()
        else:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                StandaloneRemoteDraftSchedulerMixin,
            )

            mixin = StandaloneRemoteDraftSchedulerMixin()
        mixin.sr_state = SRDraftStateManager()
        mixin.sr_state.session_id = "s"
        mixin.sr_kv = SimpleNamespace(
            get_prefix_len=lambda req: len(req.origin_input_ids),
            rollback=lambda *args, **kwargs: True,
        )
        mixin._sr_device_poisoned = False
        mixin._sr_reprefill = lambda *args, **kwargs: None
        return mixin, SRDraftState

    def _bind_mixin_methods(self):
        import ast
        import importlib.util
        import sys
        import types
        from pathlib import Path

        root = Path(__file__).resolve().parents[4] / "python"
        if "torch" not in sys.modules:
            torch_mod = types.ModuleType("torch")
            torch_mod.__path__ = []
            torch_mod.Tensor = type("Tensor", (), {})
            torch_mod.int64 = object()
            torch_mod.int32 = object()
            torch_mod.float32 = object()
            torch_mod.bool = object()
            torch_mod.arange = lambda *args, **kwargs: []
            dist = types.ModuleType("torch.distributed")
            torch_mod.distributed = dist
            sys.modules["torch"] = torch_mod
            sys.modules["torch.distributed"] = dist

        def ensure(name, path):
            mod = sys.modules.get(name)
            if mod is None:
                mod = types.ModuleType(name)
                sys.modules[name] = mod
            mod.__path__ = [str(path)]
            mod.__package__ = name
            return mod

        ensure("sglang", root / "sglang")
        ensure("sglang.srt", root / "sglang/srt")
        ensure("sglang.srt.sampling", root / "sglang/srt/sampling")
        ensure("sglang.srt.speculative", root / "sglang/srt/speculative")
        base = root / "sglang/srt/speculative/standalone_remote"
        ensure("sglang.srt.speculative.standalone_remote", base)
        ensure("sglang.srt.speculative.standalone_remote.drafter", base / "drafter")

        def load(name, file):
            if name in sys.modules and hasattr(sys.modules[name], "__file__"):
                return sys.modules[name]
            spec = importlib.util.spec_from_file_location(name, file)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod

        load(
            "sglang.srt.sampling.sampling_params",
            root / "sglang/srt/sampling/sampling_params.py",
        )
        protocol = load(
            "sglang.srt.speculative.standalone_remote.sr_protocol",
            base / "sr_protocol.py",
        )
        commit = load(
            "sglang.srt.speculative.standalone_remote.sr_commit",
            base / "sr_commit.py",
        )
        align = load(
            "sglang.srt.speculative.standalone_remote.sr_align",
            base / "sr_align.py",
        )
        lease = load(
            "sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease",
            base / "drafter/sr_tree_kv_lease.py",
        )
        metrics = load(
            "sglang.srt.speculative.standalone_remote.sr_round_metrics",
            base / "sr_round_metrics.py",
        )
        src_path = base / "drafter/sr_draft_scheduler_mixin.py"
        tree = ast.parse(src_path.read_text(encoding="utf-8"))
        class_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "StandaloneRemoteDraftSchedulerMixin"
        )
        wanted = {
            "_sr_prepare_v2",
            "_sr_prepare_delta_fast",
            "_sr_finalize_v2_commit",
            "_sr_reuse_zero_delta",
            "_sr_confirm_zero_delta",
            "_sr_stage_slow_commit",
            "_sr_drop_unconfirmed_commit",
            "_sr_clear_prefix_stamp",
            "_sr_align",
            "_sr_v2_reply",
            "_sr_remember_commit",
            "_sr_mm_can_restore",
        }
        nodes = [
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name in wanted
        ]
        module = ast.Module(
            [ast.parse("from __future__ import annotations").body[0], *nodes],
            [],
        )
        ast.fix_missing_locations(module)
        ns = {
            "CommitOutcome": commit.CommitOutcome,
            "CommittedPrefixView": commit.CommittedPrefixView,
            "PendingCommit": commit.PendingCommit,
            "RecoveryRoute": commit.RecoveryRoute,
            "candidate_stamp_current": commit.candidate_stamp_current,
            "commit_fingerprint": commit.commit_fingerprint,
            "delta_fast_allowed": commit.delta_fast_allowed,
            "inspect_commit": commit.inspect_commit,
            "make_prefix_stamp": commit.make_prefix_stamp,
            "note_commit_result": commit.note_commit_result,
            "mm_items_complete": commit.mm_items_complete,
            "rebuild_committed_output": commit.rebuild_committed_output,
            "retarget_stamp_version": commit.retarget_stamp_version,
            "route_snapshot_recovery": commit.route_snapshot_recovery,
            "get_sr_round_metrics": metrics.get_sr_round_metrics,
            "SRAlignResult": lease.SRAlignResult,
            "prefix_window_tokens": lease.prefix_window_tokens,
            "snapshot_sr_align": lease.snapshot_sr_align,
            "invalidate_tree_seed": lambda req: (
                setattr(
                    req,
                    "sr_prefix_revision",
                    int(getattr(req, "sr_prefix_revision", 0)) + 1,
                ),
                setattr(req, "sr_tree_seed", None),
                setattr(req, "sr_tree_seed_boundary", None),
            ),
            "last_token_in_kv": align.last_token_in_kv,
            "replace": __import__("dataclasses").replace,
            "SRReplyStatus": protocol.SRReplyStatus,
            "SRDraftReply": protocol.SRDraftReply,
            "Optional": __import__("typing").Optional,
            "Tuple": __import__("typing").Tuple,
            "List": __import__("typing").List,
        }
        exec(compile(module, str(src_path), "exec"), ns)
        mixin = SimpleNamespace()
        import types as types_mod

        for name in wanted:
            setattr(mixin, name, types_mod.MethodType(ns[name], mixin))
        return mixin

    def _install(self, mixin, state_cls, output):
        req = SimpleNamespace(
            rid="r",
            origin_input_ids=[1, 2],
            output_ids=list(output),
            sr_padded_ids=[1, 2],
            kv_committed_len=2 + len(output),
            kv_allocated_len=2 + len(output),
            sr_prefix_revision=0,
            multimodal_inputs=None,
            sr_commit_incomplete=False,
        )
        state = state_cls(req_id="r", session_id="s", req_object=req)
        state.acked_output_ids = list(output)
        state.acked_version = 1
        state.prompt_len = 2
        state.commit_trusted = True
        state.local_prefix_stamp = make_prefix_stamp(req, 1)
        state.last_window = ([4], None, None)
        state.last_step_id = 3
        state.last_base_committed_len = 2 + len(output)
        state.last_num_draft_tokens = 4
        mixin.sr_state.set("r", state)
        return req, state

    def test_zero_delta_keeps_fast_path_for_next_delta(self):
        mixin, state_cls = self._mixin()
        req, state = self._install(mixin, state_cls, [9])
        zero = _delta([9], [], prompt_len=2, base_version=1, version=2, step=3)
        zero.num_draft_tokens = 4
        zero.base_committed_len = 3
        reply, live, _local = mixin._sr_prepare_v2(zero, SRAction.STEP, "s", None)
        self.assertIsNotNone(reply)
        self.assertIsNone(live)
        self.assertEqual(state.acked_version, 2)
        self.assertEqual(state.acked_output_ids, [9])
        self.assertEqual(state.local_prefix_stamp.acked_version, 2)
        self.assertIsNone(state.pending_commit)
        nxt = _delta([9], [3, 4], prompt_len=2, base_version=2, version=3, step=4)
        nxt.num_draft_tokens = 4
        reply, live, wire = mixin._sr_prepare_v2(nxt, SRAction.STEP, "s", None)
        self.assertIsNone(reply)
        self.assertIs(live, req)
        self.assertIs(wire, nxt)
        self.assertEqual(req.output_ids, [9, 3, 4])
        self.assertEqual(state.acked_output_ids, [9])
        self.assertEqual(mixin._sr_round_metrics.counts["commit_fast_apply"], 1)
        self.assertEqual(req.sr_align_result.kind, "append_n")
        self.assertEqual(req.sr_align_result.old_prefix_revision, 0)
        self.assertEqual(req.sr_prefix_revision, 1)
        done = mixin._sr_finalize_v2_commit(nxt, req, reply or mixin._sr_v2_reply(nxt, SRReplyStatus.OK))
        self.assertEqual(done.ack_output_len, 3)
        self.assertEqual(state.acked_output_ids, [9, 3, 4])
        self.assertIsNone(state.pending_commit)
        self.assertEqual(state.local_prefix_stamp.acked_version, 3)

    def test_cpu_failure_retries_equal_without_second_append(self):
        mixin, state_cls = self._mixin()
        req, state = self._install(mixin, state_cls, [9])
        delta = _delta([9], [5], prompt_len=2, base_version=1, version=2, step=4)
        mixin._sr_prepare_v2(delta, SRAction.STEP, "s", None)
        self.assertEqual(req.output_ids, [9, 5])
        mixin._sr_drop_unconfirmed_commit(state)
        self.assertEqual(state.acked_output_ids, [9])
        self.assertTrue(state.commit_trusted)
        self.assertIsNone(state.pending_commit)
        self.assertIsNone(state.local_prefix_stamp)
        mixin._sr_prepare_v2(delta, SRAction.STEP, "s", None)
        self.assertEqual(req.output_ids, [9, 5])
        self.assertEqual(req.sr_align_result.kind, "equal")
        self.assertEqual(mixin._sr_round_metrics.counts["commit_slow_apply"], 1)
        reply = mixin._sr_finalize_v2_commit(
            delta, req, mixin._sr_v2_reply(delta, SRReplyStatus.OK)
        )
        self.assertEqual(state.acked_output_ids, [9, 5])
        self.assertEqual(reply.ack_commit_version, 2)
        self.assertIsNone(state.pending_commit)

    def test_unknown_completion_forbids_retry(self):
        mixin, state_cls = self._mixin()
        req, state = self._install(mixin, state_cls, [9])
        delta = _delta([9], [5], prompt_len=2, base_version=1, version=2, step=4)
        mixin._sr_prepare_v2(delta, SRAction.STEP, "s", None)
        req.sr_commit_incomplete = True
        reply = mixin._sr_finalize_v2_commit(
            delta, req, mixin._sr_v2_reply(delta, SRReplyStatus.OK)
        )
        self.assertEqual(reply.reason, "commit_incomplete")
        self.assertEqual(state.acked_output_ids, [9])
        self.assertFalse(state.commit_trusted)
        self.assertTrue(state.completion_unknown)
        self.assertIsNone(state.pending_commit)
        again, _live, _wire = mixin._sr_prepare_v2(delta, SRAction.STEP, "s", None)
        self.assertEqual(again.status, SRReplyStatus.REJECT)
        self.assertEqual(again.reason, "completion_unknown")
        self.assertEqual(req.output_ids, [9, 5])

    def test_missing_stamp_is_not_rebuilt_by_cache_or_empty_delta(self):
        mixin, state_cls = self._mixin()
        _req, state = self._install(mixin, state_cls, [9])
        state.local_prefix_stamp = None
        zero = _delta([9], [], prompt_len=2, base_version=1, version=2, step=3)
        zero.num_draft_tokens = 4
        zero.base_committed_len = 3
        mixin._sr_prepare_v2(zero, SRAction.STEP, "s", None)
        self.assertIsNone(state.local_prefix_stamp)
        self.assertEqual(state.acked_version, 2)


if __name__ == "__main__":
    unittest.main()
