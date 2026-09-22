"""Unit tests for STANDALONE_REMOTE protocol, alignment, mm, and stale replies."""

import inspect
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

try:
    import torch
except ImportError:
    torch = None

from sglang.srt.speculative.spec_info import (
    SpeculativeAlgorithm,
    cuda_graph_hidden_mode_can_run,
    cuda_graph_hidden_mode_needs_recapture,
    decode_cuda_graph_accepts_spec_info,
    resolve_cuda_graph_capture_hidden_mode,
)
from sglang.srt.speculative.standalone_remote.sr_align import (
    DraftDecision,
    apply_tree_seed_topk,
    broadcast_sr_obj,
    capture_tree_seed_topk,
    classify_prefix_alignment,
    committed_tail_not_in_kv,
    decide_draft_action,
    draft_needed_max_new_tokens,
    draft_token_budget,
    drop_duplicate_root_draft,
    find_fork_point,
    ingest_active_indices,
    kv_release_len,
    last_token_in_kv,
    plan_committed_ingest,
    plan_tree_seed_recovery,
    replay_grammar_from_committed,
    rollback_free_range,
    shift_overlapped_prefill_drafts,
    snapshot_reprefill_fill_ids,
    unwrap_tp_broadcast,
    wrap_tp_broadcast,
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
from sglang.srt.speculative.standalone_remote.sr_circuit_breaker import SRRpcBreaker
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

    def test_sr_target_captures_target_verify_cuda_graph(self):
        sr = SpeculativeAlgorithm.STANDALONE_REMOTE
        target = SimpleNamespace(
            standalone_remote_role="target",
            speculative_num_draft_tokens=5,
            spectre_role=None,
        )
        self.assertTrue(sr.captures_target_verify_cuda_graph(target))
        self.assertEqual(sr.target_verify_cuda_graph_num_tokens_per_bs(target), 5)
        self.assertFalse(
            sr.captures_target_verify_cuda_graph(target, is_draft_worker=True)
        )
        self.assertEqual(
            sr.target_verify_cuda_graph_num_tokens_per_bs(
                target, is_draft_worker=True
            ),
            1,
        )
        self.assertTrue(sr.uses_spec_topk_cuda_graph_layout())
        self.assertTrue(sr.uses_dual_ntpb_cuda_graph(target))
        self.assertEqual(sr.dual_ntpb_cuda_graph_options(target), [5, 1])
        self.assertFalse(
            sr.uses_dual_ntpb_cuda_graph(target, is_draft_worker=True)
        )

    def test_sr_draft_stays_on_decode_cuda_graph(self):
        sr = SpeculativeAlgorithm.STANDALONE_REMOTE
        draft = SimpleNamespace(
            standalone_remote_role="draft",
            speculative_num_draft_tokens=5,
            spectre_role=None,
        )
        self.assertFalse(sr.captures_target_verify_cuda_graph(draft))
        self.assertEqual(sr.target_verify_cuda_graph_num_tokens_per_bs(draft), 1)
        self.assertTrue(sr.uses_spec_topk_cuda_graph_layout())
        self.assertFalse(sr.uses_dual_ntpb_cuda_graph(draft))
        self.assertIsNone(sr.dual_ntpb_cuda_graph_options(draft))
        self.assertEqual(sr.decode_cuda_graph_hidden_mode(draft), 0)  # NULL
        self.assertEqual(
            sr.decode_cuda_graph_hidden_mode(draft, is_draft_worker=True),
            0,
        )

    def test_cuda_graph_hidden_mode_can_run_table(self):
        null, last, full = 0, 1, 2  # CaptureHiddenMode NULL / LAST / FULL
        # Emulation: weaker request on a stronger graph.
        self.assertTrue(cuda_graph_hidden_mode_can_run(null, last))
        self.assertTrue(cuda_graph_hidden_mode_can_run(last, last))
        self.assertTrue(cuda_graph_hidden_mode_can_run(last, full))
        self.assertFalse(cuda_graph_hidden_mode_needs_recapture(null, last))
        self.assertFalse(cuda_graph_hidden_mode_needs_recapture(last, last))
        self.assertFalse(cuda_graph_hidden_mode_needs_recapture(last, full))
        # Stronger request must not replay DECODE graphs (tree capture would
        # hit unset raw_num_token). Recapture stays on the replay path only.
        self.assertFalse(cuda_graph_hidden_mode_can_run(full, last))
        self.assertFalse(cuda_graph_hidden_mode_can_run(last, null))
        self.assertTrue(cuda_graph_hidden_mode_needs_recapture(full, last))
        self.assertTrue(cuda_graph_hidden_mode_needs_recapture(last, null))

    def test_decode_cuda_graph_rejects_tree_draft_spec(self):
        self.assertTrue(decode_cuda_graph_accepts_spec_info(None))
        self.assertTrue(
            decode_cuda_graph_accepts_spec_info(
                SimpleNamespace(capture_hidden_mode=1, num_tokens_per_req=1)
            )
        )
        self.assertFalse(
            decode_cuda_graph_accepts_spec_info(
                SimpleNamespace(is_draft_input=lambda: True, num_tokens_per_req=2)
            )
        )
        self.assertFalse(
            decode_cuda_graph_accepts_spec_info(
                SimpleNamespace(num_tokens_per_req=2)
            )
        )

    def test_capture_keeps_last_when_draft_spec_info_is_none(self):
        null, last, full = 0, 1, 2
        # Missing spec_info must not reset a captured mode.
        self.assertEqual(
            resolve_cuda_graph_capture_hidden_mode(last, None), last
        )
        self.assertEqual(
            resolve_cuda_graph_capture_hidden_mode(null, None), null
        )
        self.assertEqual(
            resolve_cuda_graph_capture_hidden_mode(full, None), full
        )
        # Target verify spec_info can raise NULL → FULL.
        self.assertEqual(
            resolve_cuda_graph_capture_hidden_mode(
                null, SimpleNamespace(capture_hidden_mode=full)
            ),
            full,
        )
        # FULL is sticky even if spec_info is weaker.
        self.assertEqual(
            resolve_cuda_graph_capture_hidden_mode(
                full, SimpleNamespace(capture_hidden_mode=null)
            ),
            full,
        )
        # Ingest LAST on LAST-captured graphs must still can_run.
        captured = resolve_cuda_graph_capture_hidden_mode(last, None)
        self.assertTrue(cuda_graph_hidden_mode_can_run(last, captured))
        self.assertTrue(cuda_graph_hidden_mode_can_run(null, captured))

    def test_existing_algorithms_keep_target_verify_capture(self):
        args = SimpleNamespace(
            standalone_remote_role=None,
            speculative_num_draft_tokens=7,
            spectre_role=None,
        )
        self.assertTrue(
            SpeculativeAlgorithm.EAGLE.captures_target_verify_cuda_graph(args)
        )
        self.assertTrue(
            SpeculativeAlgorithm.STANDALONE.captures_target_verify_cuda_graph(args)
        )
        self.assertTrue(
            SpeculativeAlgorithm.NGRAM.captures_target_verify_cuda_graph(args)
        )
        self.assertEqual(
            SpeculativeAlgorithm.EAGLE.target_verify_cuda_graph_num_tokens_per_bs(args),
            7,
        )
        self.assertFalse(
            SpeculativeAlgorithm.NONE.captures_target_verify_cuda_graph(args)
        )
        spectre_target = SimpleNamespace(
            spectre_role="target",
            standalone_remote_role=None,
            speculative_num_draft_tokens=4,
        )
        spectre_draft = SimpleNamespace(
            spectre_role="draft",
            standalone_remote_role=None,
            speculative_num_draft_tokens=4,
        )
        self.assertTrue(
            SpeculativeAlgorithm.SPECTRE.captures_target_verify_cuda_graph(
                spectre_target
            )
        )
        self.assertFalse(
            SpeculativeAlgorithm.SPECTRE.captures_target_verify_cuda_graph(
                spectre_draft
            )
        )
        self.assertFalse(SpeculativeAlgorithm.EAGLE.uses_dual_ntpb_cuda_graph(args))
        self.assertFalse(
            SpeculativeAlgorithm.STANDALONE.uses_dual_ntpb_cuda_graph(args)
        )
        self.assertTrue(
            SpeculativeAlgorithm.SPECTRE.uses_dual_ntpb_cuda_graph(spectre_target)
        )
        self.assertEqual(
            SpeculativeAlgorithm.SPECTRE.dual_ntpb_cuda_graph_options(spectre_target),
            [4, 1],
        )
        self.assertFalse(
            SpeculativeAlgorithm.SPECTRE.uses_dual_ntpb_cuda_graph(spectre_draft)
        )
        self.assertEqual(
            SpeculativeAlgorithm.EAGLE.decode_cuda_graph_hidden_mode(args),
            0,
        )
        self.assertEqual(
            SpeculativeAlgorithm.NONE.decode_cuda_graph_hidden_mode(args),
            0,
        )
        self.assertEqual(
            SpeculativeAlgorithm.STANDALONE_REMOTE.decode_cuda_graph_hidden_mode(
                SimpleNamespace(standalone_remote_role="target")
            ),
            0,
        )

    def test_ar_fallback_source_forces_eager(self):
        from pathlib import Path

        worker_path = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/speculative/standalone_remote/verifier/sr_worker.py"
        )
        if not worker_path.is_file():
            self.skipTest(f"missing {worker_path}")
        src = worker_path.read_text()
        self.assertIn("def _forward_target_eager", src)
        self.assertIn("runner.graph_runner = None", src)
        self.assertIn("can_run_cuda_graph=False", src)


class TestSRTreeSeedHiddenCapture(CustomTestCase):
    def test_enable_tree_seed_hidden_requests_null_not_full(self):
        try:
            from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                StandaloneRemoteDraftSchedulerMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))
        mixin = SimpleNamespace(
            server_args=SimpleNamespace(speculative_eagle_topk=4)
        )
        batch = SimpleNamespace(return_hidden_states=True, capture_hidden_mode=None)
        StandaloneRemoteDraftSchedulerMixin._sr_enable_tree_seed_hidden(mixin, batch)
        self.assertFalse(batch.return_hidden_states)
        self.assertEqual(batch.capture_hidden_mode, CaptureHiddenMode.NULL)
        self.assertEqual(batch.tree_seed_topk, 4)

    def test_schedule_batch_resolves_last_unless_http_full(self):
        try:
            from sglang.srt.managers.schedule_batch import ScheduleBatch
            from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
        except ImportError as e:
            self.skipTest(str(e))
        batch = ScheduleBatch()
        batch.return_hidden_states = False
        batch.capture_hidden_mode = CaptureHiddenMode.LAST
        self.assertEqual(batch.resolve_capture_hidden_mode(), CaptureHiddenMode.LAST)
        batch.return_hidden_states = True
        self.assertEqual(batch.resolve_capture_hidden_mode(), CaptureHiddenMode.FULL)
        batch.return_hidden_states = False
        batch.capture_hidden_mode = None
        batch.spec_info = SimpleNamespace(capture_hidden_mode=CaptureHiddenMode.LAST)
        self.assertEqual(batch.resolve_capture_hidden_mode(), CaptureHiddenMode.LAST)
        batch.spec_info = None
        self.assertEqual(batch.resolve_capture_hidden_mode(), CaptureHiddenMode.NULL)

    def test_explicit_full_stays_full_when_sr_decode_defaults_null(self):
        try:
            from sglang.srt.managers.schedule_batch import ScheduleBatch
            from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
            from sglang.srt.speculative.spec_info import (
                SpeculativeAlgorithm,
                resolve_cuda_graph_capture_hidden_mode,
            )
        except ImportError as e:
            self.skipTest(str(e))
        draft = SimpleNamespace(
            standalone_remote_role="draft",
            speculative_num_draft_tokens=5,
            spectre_role=None,
        )
        captured = SpeculativeAlgorithm.STANDALONE_REMOTE.decode_cuda_graph_hidden_mode(
            draft
        )
        self.assertEqual(captured, int(CaptureHiddenMode.NULL))
        self.assertEqual(
            resolve_cuda_graph_capture_hidden_mode(
                captured, SimpleNamespace(capture_hidden_mode=CaptureHiddenMode.FULL)
            ),
            CaptureHiddenMode.FULL,
        )
        batch = ScheduleBatch()
        batch.return_hidden_states = True
        batch.capture_hidden_mode = CaptureHiddenMode.NULL
        self.assertEqual(batch.resolve_capture_hidden_mode(), CaptureHiddenMode.FULL)


class TestSRTargetHiddenSkip(CustomTestCase):
    def test_select_hidden_states_for_draft_skip_and_gather(self):
        if torch is None:
            self.skipTest("torch not available")
        import ast
        from pathlib import Path

        path = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/speculative/eagle_info.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "select_hidden_states_for_draft"
        )
        ns = {}
        exec(compile(ast.Module([node], []), str(path), "exec"), ns)
        hidden = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        index = torch.tensor([0, 2])
        gathered = ns["select_hidden_states_for_draft"](hidden, index, True)
        self.assertEqual(gathered.tolist(), [[0.0, 1.0], [4.0, 5.0]])
        self.assertIsNone(ns["select_hidden_states_for_draft"](hidden, index, False))
        self.assertIsNone(ns["select_hidden_states_for_draft"](None, index, True))

    def test_verify_defaults_to_preparing_local_draft_hidden(self):
        from pathlib import Path

        eagle = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/speculative/eagle_info.py"
        )
        src = eagle.read_text(encoding="utf-8")
        self.assertIn("prepare_local_draft_hidden: bool = True", src)
        spectre = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/speculative/spectre/verifier/spectre_worker.py"
        )
        eagle_worker = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/speculative/eagle_worker.py"
        )
        for path in (spectre, eagle_worker):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("prepare_local_draft_hidden=", text)

    def _load_hidden_helpers(self):
        import ast
        from pathlib import Path
        from typing import Optional

        path = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/speculative/standalone_remote/verifier/sr_worker.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        class_node = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "StandaloneRemoteWorker"
        )
        wanted = {"_need_target_hidden", "_target_capture_hidden_mode"}
        capture = SimpleNamespace(NULL=0, LAST=1, FULL=2)
        ns = {
            "CaptureHiddenMode": capture,
            "Optional": Optional,
            "ScheduleBatch": object,
        }
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                exec(compile(ast.Module([node], []), str(path), "exec"), ns)
        return ns["_need_target_hidden"], ns["_target_capture_hidden_mode"], capture

    def test_need_target_hidden_skip_and_overrides(self):
        need, capture_fn, CaptureHiddenMode = self._load_hidden_helpers()
        worker = SimpleNamespace(
            server_args=SimpleNamespace(enable_return_hidden_states=False),
            _hybrid_needs_hidden=False,
        )
        worker._need_target_hidden = lambda batch=None: need(worker, batch)
        self.assertFalse(need(worker, None))
        self.assertEqual(capture_fn(worker, None), CaptureHiddenMode.NULL)
        batch = SimpleNamespace(return_hidden_states=False, reqs=[])
        self.assertFalse(need(worker, batch))
        self.assertEqual(capture_fn(worker, batch), CaptureHiddenMode.NULL)
        batch.reqs = [SimpleNamespace(return_hidden_states=True)]
        self.assertTrue(need(worker, batch))
        self.assertEqual(capture_fn(worker, batch), CaptureHiddenMode.FULL)
        worker.server_args.enable_return_hidden_states = True
        self.assertTrue(need(worker, SimpleNamespace(return_hidden_states=False, reqs=[])))
        worker.server_args.enable_return_hidden_states = False
        worker._hybrid_needs_hidden = True
        self.assertTrue(need(worker, None))
        self.assertEqual(capture_fn(worker, None), CaptureHiddenMode.FULL)

    def test_sr_worker_guards_none_hidden_states(self):
        from pathlib import Path

        src = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/speculative/standalone_remote/verifier/sr_worker.py"
        ).read_text(encoding="utf-8")
        self.assertIn("prepare_local_draft_hidden=prepare_hidden", src)
        self.assertIn("if logits_output.hidden_states is not None:", src)
        self.assertIn("CaptureHiddenMode.NULL", src)
        self.assertIn("_need_target_hidden", src)


class TestSRTpBroadcast(CustomTestCase):
    def test_wrap_unwrap_bool_none_dict_tuple(self):
        for obj in (True, False, None, {"k": 1}, (1, 2)):
            wrapped = wrap_tp_broadcast(obj)
            self.assertEqual(len(wrapped), 1)
            self.assertIs(unwrap_tp_broadcast(wrapped), obj)
        with self.assertRaises(TypeError):
            len(True)
        with self.assertRaises(TypeError):
            len(None)

    def test_broadcast_sr_obj_tp1_is_identity(self):
        group = SimpleNamespace(rank=0, ranks=[0])
        self.assertIs(
            broadcast_sr_obj(True, 1, 0, group, None),
            True,
        )
        self.assertIsNone(broadcast_sr_obj(None, 1, 0, group, None))

    @patch(
        "sglang.srt.speculative.standalone_remote.sr_align._default_broadcast_pyobj"
    )
    def test_rank0_sends_singleton_list(self, mock_bcast):
        mock_bcast.side_effect = lambda data, rank, group, src=0: data
        group = SimpleNamespace(rank=0, ranks=[0])
        self.assertIs(
            broadcast_sr_obj(True, 2, 0, group, "cpu"),
            True,
        )
        self.assertEqual(mock_bcast.call_args[0][0], [True])
        mock_bcast.reset_mock()
        self.assertIsNone(broadcast_sr_obj(None, 2, 0, group, "cpu"))
        self.assertEqual(mock_bcast.call_args[0][0], [None])

    @patch(
        "sglang.srt.speculative.standalone_remote.sr_align._default_broadcast_pyobj"
    )
    def test_nonsrc_dummy_is_empty_list(self, mock_bcast):
        mock_bcast.side_effect = lambda data, rank, group, src=0: [True]
        group = SimpleNamespace(rank=1, ranks=[0, 1])
        self.assertIs(
            broadcast_sr_obj(False, 2, 1, group, "cpu"),
            True,
        )
        self.assertEqual(mock_bcast.call_args[0][0], [])

    def test_init_syncs_session_id_to_rank1(self):
        try:
            from sglang.srt.speculative.standalone_remote.verifier.sr_target_scheduler_mixin import (
                SchedulerStandaloneRemoteTargetMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))

        shared = {}

        def fake_bcast(obj, tp_size, tp_rank, tp_group, tp_cpu_group):
            if tp_rank == 0:
                shared["id"] = obj
                return obj
            return shared.get("id")

        def _make(rank):
            mixin = SchedulerStandaloneRemoteTargetMixin()
            mixin.tp_size = 2
            mixin.tp_rank = rank
            mixin.tp_group = SimpleNamespace(rank=rank, ranks=[0, 1])
            mixin.tp_cpu_group = None
            mixin.server_args = SimpleNamespace(
                standalone_remote_breaker_failures=3,
                standalone_remote_breaker_cooldown=32,
            )
            return mixin

        with patch(
            "sglang.srt.speculative.standalone_remote.verifier."
            "sr_target_scheduler_mixin.broadcast_sr_obj",
            side_effect=fake_bcast,
        ), patch(
            "sglang.srt.speculative.standalone_remote.verifier."
            "sr_target_scheduler_mixin.make_transport_from_server_args",
        ):
            rank0 = _make(0)
            rank0._init_sr_target()
            rank1 = _make(1)
            rank1._init_sr_target()
        self.assertIsNotNone(rank0.sr_session_id)
        self.assertEqual(rank1.sr_session_id, rank0.sr_session_id)
        self.assertIsNone(rank1.sr_client)


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

    def test_last_token_in_kv_uses_committed_not_allocated(self):
        self.assertTrue(last_token_in_kv(127, 128))
        self.assertFalse(last_token_in_kv(127, 127))
        self.assertFalse(last_token_in_kv(127, 126))

    def test_capture_tree_seed_topk_matches_softmax_topk(self):
        if torch is None:
            self.skipTest("torch is required")
        logits = torch.tensor(
            [[1.0, 2.0, 0.5, -1.0], [0.1, 0.2, 3.0, 0.0]],
            dtype=torch.float32,
        )
        vals, idx = capture_tree_seed_topk(logits, 2)
        expect_vals, expect_idx = torch.topk(
            torch.softmax(logits.float(), dim=-1), 2, dim=-1
        )
        self.assertTrue(torch.allclose(vals, expect_vals))
        self.assertTrue(torch.equal(idx, expect_idx))

    def test_plan_tree_seed_recovery(self):
        self.assertEqual(
            plan_tree_seed_recovery(
                10, [7, 8], 10, seed_ok=True, can_rollback_last_slot=True
            ),
            "ingest",
        )
        self.assertEqual(
            plan_tree_seed_recovery(
                10, [7, 8], 12, seed_ok=True, can_rollback_last_slot=False
            ),
            "ok",
        )
        self.assertEqual(
            plan_tree_seed_recovery(
                10, [7], 11, seed_ok=False, can_rollback_last_slot=True
            ),
            "recapture_last",
        )
        # Origin last token: P=K=L even when the last slot is page-aligned.
        self.assertEqual(
            plan_tree_seed_recovery(
                128, [], 128, seed_ok=False, can_rollback_last_slot=True
            ),
            "reprefill",
        )
        self.assertEqual(
            plan_tree_seed_recovery(
                257, [], 256, seed_ok=False, can_rollback_last_slot=True
            ),
            "reprefill",
        )
        # page_size=128 cannot drop 127.
        self.assertEqual(
            plan_tree_seed_recovery(
                10, [7], 11, seed_ok=False, can_rollback_last_slot=False
            ),
            "reprefill",
        )

    def test_apply_tree_seed_topk_writes_live_sampling_info(self):
        info = SimpleNamespace(tree_seed_topk=0)
        batch = SimpleNamespace(tree_seed_topk=0, sampling_info=info)
        self.assertEqual(apply_tree_seed_topk(batch, 3), 3)
        self.assertEqual(batch.tree_seed_topk, 3)
        self.assertEqual(info.tree_seed_topk, 3)
        bare = SimpleNamespace()
        apply_tree_seed_topk(bare, 0)
        self.assertEqual(bare.tree_seed_topk, 1)

    def test_snapshot_reprefill_does_not_shrink_to_padded(self):
        origin = list(range(78))
        first = snapshot_reprefill_fill_ids(origin, [99])
        self.assertEqual(len(first), 79)
        second = snapshot_reprefill_fill_ids(first, [])
        self.assertEqual(second, first)
        padded = list(range(78))
        self.assertNotEqual(second, padded)

    def test_rollback_free_range_does_not_page_floor(self):
        self.assertEqual(rollback_free_range(256, 384, 4096), (256, 384))
        self.assertEqual(rollback_free_range(256, 257, 4096), (256, 257))
        self.assertEqual(kv_release_len(257, 384), 384)
        self.assertEqual(kv_release_len(128, 0), 128)


class TestSRTreeSeedSourceGuard(CustomTestCase):
    def test_sampler_captures_topk_before_inplace_softmax(self):
        from pathlib import Path

        sampler_path = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/layers/sampler.py"
        )
        if not sampler_path.is_file():
            self.skipTest(f"missing {sampler_path}")
        src = sampler_path.read_text()
        capture = src.find("capture_tree_seed_topk")
        softmax_assign = src.find("logits[:] = torch.softmax")
        self.assertGreater(capture, 0)
        self.assertGreater(softmax_assign, capture)

    def test_cache_tree_seeds_uses_precomputed_topk(self):
        from pathlib import Path

        mixin_path = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/speculative/standalone_remote/drafter/"
            "sr_draft_scheduler_mixin.py"
        )
        if not mixin_path.is_file():
            self.skipTest(f"missing {mixin_path}")
        src = mixin_path.read_text()
        start = src.find("def _sr_cache_tree_seeds")
        self.assertGreater(start, 0)
        nxt = src.find("\n    def ", start + 1)
        body = src[start : nxt if nxt > start else None]
        self.assertIn("tree_seed_topk_p", body)
        self.assertNotIn("softmax", body)
        self.assertIn("None,\n                    verified_id", body)
        self.assertNotIn("hidden.shape", body)
        self.assertNotIn("row_hidden", body)
        self.assertNotIn("kv_committed_len =", body)

    def test_enable_and_reprefill_use_sr_align_helpers(self):
        from pathlib import Path

        mixin_path = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/speculative/standalone_remote/drafter/"
            "sr_draft_scheduler_mixin.py"
        )
        if not mixin_path.is_file():
            self.skipTest(f"missing {mixin_path}")
        src = mixin_path.read_text()
        enable = src[src.find("def _sr_enable_tree_seed_hidden") :]
        enable = enable[: enable.find("\n    def ")]
        self.assertIn("apply_tree_seed_topk", enable)
        self.assertIn("CaptureHiddenMode.NULL", enable)
        self.assertNotIn("CaptureHiddenMode.LAST", enable)
        reprefill = src[src.find("def _sr_reprefill_committed") :]
        reprefill = reprefill[: reprefill.find("\n    def ")]
        self.assertIn("snapshot_reprefill_fill_ids", reprefill)
        ensure = src[src.find("def _sr_ensure_tree_seeds") :]
        ensure = ensure[: ensure.find("\n    def ")]
        self.assertIn('action == "ingest"', ensure)
        self.assertIn("tree seed recovery failed", ensure)

    def test_local_rollback_does_not_page_floor_end(self):
        from pathlib import Path

        path = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/speculative/standalone_remote/sr_kv_rollbacker.py"
        )
        if not path.is_file():
            self.skipTest(f"missing {path}")
        src = path.read_text()
        start = src.find("def local_rollback")
        body = src[start : src.find("\n    def ", start + 1)]
        self.assertIn("rollback_free_range", body)
        self.assertNotIn("end // self.page_size", body)
        self.assertIn("kv_release_len", src)


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

    def test_sr_mm_moves_fake_npu_tensor_to_cpu(self):
        if torch is None:
            self.skipTest("torch not available")
        try:
            from sglang.srt.speculative.standalone_remote.sr_mm_payload import (
                _to_cpu_contiguous_tensor,
            )
        except ImportError as e:
            self.skipTest(str(e))
        real = torch.arange(4, dtype=torch.float32)
        fake = MagicMock(spec=torch.Tensor)
        fake.device = SimpleNamespace(type="npu")
        fake.detach.return_value = fake
        fake.cpu.return_value = real
        out = _to_cpu_contiguous_tensor(fake)
        self.assertTrue(torch.equal(out, real))

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

    def test_ensure_window_budget_extends_max_new_tokens(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                StandaloneRemoteDraftSchedulerMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))

        mixin = StandaloneRemoteDraftSchedulerMixin()
        mixin.server_args = MagicMock()
        mixin.server_args.speculative_num_steps = 4
        mixin.sr_state = SRDraftStateManager()
        req = MagicMock()
        req.rid = "a"
        req.origin_input_ids = [1, 2, 3]
        req.output_ids = [1, 2, 3, 4, 5]
        sp = MagicMock()
        sp.max_new_tokens = 9
        req.sampling_params = sp
        mixin._sr_ensure_window_budget(req, 5)
        self.assertEqual(sp.max_new_tokens, 14)
        self.assertFalse(mixin._sr_is_degraded("a"))

    def test_ensure_window_budget_degrades_on_horizon(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                StandaloneRemoteDraftSchedulerMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))

        mixin = StandaloneRemoteDraftSchedulerMixin()
        mixin.server_args = MagicMock()
        mixin.server_args.speculative_num_steps = 4
        mixin.max_req_input_len = 10
        mixin.sr_state = SRDraftStateManager()
        req = MagicMock()
        req.rid = "a"
        req.origin_input_ids = [1] * 8
        req.output_ids = [1, 2, 3, 4, 5]
        sp = MagicMock()
        sp.max_new_tokens = 9
        req.sampling_params = sp
        mixin.sr_state.set("a", SRDraftState(req_id="a", session_id="s", req_object=req))
        mixin._sr_ensure_window_budget(req, 5)
        self.assertEqual(sp.max_new_tokens, 14)
        self.assertTrue(mixin._sr_is_degraded("a"))


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

    def test_reset_resets_breaker(self):
        mixin_cls = self._import_mixin()
        mixin = mixin_cls()
        mixin.sr_session_id = "old"
        mixin.sr_rpc_seq = 1
        mixin.sr_pending = {}
        mixin._sr_inflight = None
        mixin.sr_client = None
        mixin.sr_breaker = SRRpcBreaker(failure_threshold=3, cooldown_steps=4)
        for _ in range(3):
            mixin.sr_breaker.record_failure()
        self.assertFalse(mixin.sr_breaker.should_send())
        mixin.reset_standalone_remote_target_state()
        self.assertTrue(mixin.sr_breaker.should_send())
        self.assertEqual(mixin.sr_breaker.state, SRRpcBreaker.CLOSED)

    def test_open_breaker_skips_step_rpc(self):
        order = []
        req = self._make_req()
        mixin, _client = self._make_mixin(order, req, is_extend=False)
        mixin.sr_breaker = SRRpcBreaker(failure_threshold=3, cooldown_steps=32)
        mixin.sr_breaker.state = SRRpcBreaker.OPEN
        with self.assertRaises(mixin.recv_requests.StopLoop):
            mixin.event_loop_normal_standalone_remote_target()
        self.assertEqual(order, ["gpu"])
        self.assertEqual(mixin.sr_breaker.steps_in_open, 1)

    def test_open_breaker_still_sends_prefill(self):
        order = []
        req = self._make_req()
        mixin, _client = self._make_mixin(order, req, is_extend=True)
        mixin.sr_breaker = SRRpcBreaker(failure_threshold=3, cooldown_steps=32)
        mixin.sr_breaker.state = SRRpcBreaker.OPEN
        with self.assertRaises(mixin.recv_requests.StopLoop):
            mixin.event_loop_normal_standalone_remote_target()
        self.assertEqual(order, ["send", "gpu", "recv"])
        self.assertTrue(mixin.sr_breaker.should_send())
        self.assertEqual(mixin.sr_breaker.state, SRRpcBreaker.CLOSED)


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
        self.assertEqual(
            drop_duplicate_root_draft(10, [10, 11, 12, 13, 14]),
            [10, 11, 12, 13, 14],
        )
        self.assertEqual(drop_duplicate_root_draft(10, [11, 12, 13]), [11, 12, 13])
        self.assertEqual(drop_duplicate_root_draft(None, [10, 11]), [10, 11])
        self.assertEqual(
            drop_duplicate_root_draft(10, [10, 11, 12, 13, 14], include_root=True),
            [11, 12, 13, 14],
        )
        self.assertEqual(
            drop_duplicate_root_draft(10, [11, 12, 13], include_root=True),
            [11, 12, 13],
        )


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

    def test_ar_fallback_forces_eager(self):
        worker_cls = self._import_worker()
        self.assertTrue(hasattr(worker_cls, "_forward_target_eager"))
        src = inspect.getsource(worker_cls._forward_normal_decode)
        self.assertIn("_forward_target_eager", src)
        self.assertIn("can_run_cuda_graph=False", src)
        eager_src = inspect.getsource(worker_cls._forward_target_eager)
        self.assertIn("graph_runner = None", eager_src)

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

        drafter._tree_slot_trace = []
        drafter._tree_device_pack_copies = 0
        drafter._tree_host_payload_submits = 0
        drafter._tree_cross_device_d2h = 0
        copies = {"cpu": 0, "cross": 0}
        orig_copy = torch.Tensor.copy_
        order = []
        from sglang.srt.speculative.standalone_remote.sr_transfer_staging import (
            submit_copy,
            wait_event,
        )

        def counting_copy(self, src, *args, **kwargs):
            if src.device.type != "cpu" and self.device.type == "cpu":
                copies["cross"] += 1
            else:
                copies["cpu"] += 1
            return orig_copy(self, src, *args, **kwargs)

        def traced_submit(dst, src):
            order.append("submit")
            return submit_copy(dst, src)

        def traced_wait(event):
            order.append("wait")
            return wait_event(event)

        with patch.object(torch.Tensor, "copy_", counting_copy), patch(
            "sglang.srt.speculative.standalone_remote.sr_transfer_staging.submit_copy",
            traced_submit,
        ), patch(
            "sglang.srt.speculative.standalone_remote.sr_transfer_staging.wait_event",
            traced_wait,
        ):
            drafter.expand_batch([r0, r1])
        self.assertEqual(order, ["submit", "wait"])
        self.assertEqual(drafter._tree_host_payload_submits, 1)
        self.assertEqual(drafter._tree_device_pack_copies, 3)
        self.assertEqual(drafter._tree_cross_device_d2h, 0)
        self.assertEqual(copies["cross"], 0)
        self.assertEqual(copies["cpu"], 4)
        self.assertEqual(
            drafter._tree_slot_trace, ["free", "in_flight", "consuming", "free"]
        )

    def test_unresolved_tree_transfer_keeps_lease(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_tree_drafter import (
                SRTreeDrafter,
            )
            from sglang.srt.speculative.standalone_remote.sr_transfer_staging import (
                SRTransferUnresolved,
            )
        except ImportError as e:
            self.skipTest(str(e))
        if torch is None:
            self.skipTest("torch not available")

        parent = torch.tensor([[-1, 0]], dtype=torch.int64)
        index = torch.tensor([[0, 1]], dtype=torch.int64)
        tokens = torch.tensor([[10, 11]], dtype=torch.int64)
        req = MagicMock()
        req.rid = "a"
        req.req_pool_idx = 0
        req.sr_tree_seed = (None, None, None, None)

        def call(drafter, name):
            if name == "expand_batch":
                drafter.expand_batch([req])
            else:
                drafter._expand_one(req)

        for name in ("expand_batch", "_expand_one"):
            drafter = SRTreeDrafter.__new__(SRTreeDrafter)
            drafter._stack_seeds = lambda reqs: (None, None, None, None)
            drafter._expand_tree = lambda reqs, *seed: (parent, index, tokens)
            freed = {"n": 0}
            drafter._free_lease_alloc = lambda state: freed.__setitem__(
                "n", freed["n"] + 1
            )
            drafter._pending_lease_state = {"page_slots": [torch.tensor([7])]}
            with patch(
                "sglang.srt.speculative.standalone_remote.sr_transfer_staging.submit_copy",
                side_effect=SRTransferUnresolved("d2h"),
            ):
                with self.assertRaises(SRTransferUnresolved):
                    call(drafter, name)
            self.assertEqual(freed["n"], 0)
            self.assertIsNotNone(drafter._pending_lease_state)
            slot = drafter._active_tree_slot
            self.assertEqual(slot.state, "unresolved")
            self.assertTrue(slot.src_hold)

    def test_tree_drafter_exposes_eagle_draft_graph_aliases(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_tree_drafter import (
                SRTreeDrafter,
            )
        except ImportError as e:
            self.skipTest(str(e))
        self.assertIsInstance(
            inspect.getattr_static(SRTreeDrafter, "model_runner"), property
        )
        self.assertTrue(callable(getattr(SRTreeDrafter, "draft_forward")))
        src = inspect.getsource(SRTreeDrafter._expand_tree)
        self.assertIn("cuda_graph_runner", src)
        self.assertIn("replay", src)
        init_src = inspect.getsource(SRTreeDrafter._init_cuda_graphs)
        self.assertIn("EAGLEDraftCudaGraphRunner", init_src)
        self.assertIn("EAGLEDraftNpuGraphRunner", init_src)
        self.assertNotIn("EAGLEDraftExtendCudaGraphRunner", init_src)
        self.assertLess(
            init_src.find("self.need_draft_hidden = False"),
            init_src.find("runner_cls(self)"),
        )

    def test_stack_seeds_ignores_mixed_old_hidden(self):
        if torch is None:
            self.skipTest("torch not available")
        import ast
        from pathlib import Path

        src_path = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_tree_drafter.py"
        )
        tree = ast.parse(src_path.read_text(encoding="utf-8"))
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "SRTreeDrafter"
        )
        fn = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_stack_seeds"
        )
        dummy = ast.ClassDef(
            name="LoadedDrafter",
            bases=[],
            keywords=[],
            body=[fn],
            decorator_list=[],
        )
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        mod = ast.fix_missing_locations(
            ast.Module(body=[future, dummy], type_ignores=[])
        )
        ns = {"torch": torch, "List": list, "Optional": type(None), "Tuple": tuple}
        exec(compile(mod, str(src_path), "exec"), ns)
        drafter = ns["LoadedDrafter"]()
        hidden = torch.ones(1, 4)
        reqs = [
            SimpleNamespace(
                sr_tree_seed=(
                    torch.ones(1, 2),
                    torch.ones(1, 2, dtype=torch.int64),
                    None,
                    torch.tensor([1]),
                )
            ),
            SimpleNamespace(
                sr_tree_seed=(
                    torch.ones(1, 2) * 2,
                    torch.ones(1, 2, dtype=torch.int64) * 3,
                    hidden,
                    torch.tensor([2]),
                )
            ),
        ]
        p, ix, hs, vid = drafter._stack_seeds(reqs)
        self.assertIsNone(hs)
        self.assertEqual(tuple(p.shape), (2, 2))
        self.assertEqual(tuple(ix.shape), (2, 2))
        self.assertEqual(vid.tolist(), [1, 2])

    def test_tree_drafter_skips_cuda_graph_when_disabled(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_tree_drafter import (
                SRTreeDrafter,
            )
        except ImportError as e:
            self.skipTest(str(e))
        drafter = SRTreeDrafter.__new__(SRTreeDrafter)
        drafter.server_args = SimpleNamespace(disable_cuda_graph=True)
        drafter.speculative_num_steps = 4
        drafter.draft_attn_backend = object()
        drafter.cuda_graph_runner = "sentinel"
        drafter._init_cuda_graphs()
        self.assertIsNone(drafter.cuda_graph_runner)

        drafter.server_args = SimpleNamespace(disable_cuda_graph=False)
        drafter.draft_attn_backend = None
        drafter.cuda_graph_runner = "sentinel"
        drafter._init_cuda_graphs()
        self.assertIsNone(drafter.cuda_graph_runner)

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

    def test_assemble_writes_out_buffers(self):
        if torch is None:
            self.skipTest("torch not available")
        from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
            assemble_draft_rows,
        )

        out_tokens = torch.full((2, 4), 7, dtype=torch.int64)
        out_parents = torch.full((2, 4), 9, dtype=torch.int64)
        out_indices = torch.full((2, 4), 8, dtype=torch.int64)
        parents, indices, dt = assemble_draft_rows(
            [[11, 12, 13], [21, 22, 23]],
            [[-1, 0], [-1, 1]],
            [[0, 1, 2], [3, 4, 5]],
            topk=2,
            spec_steps=3,
            num_draft_tokens=4,
            device="cpu",
            out_tokens=out_tokens,
            out_parents=out_parents,
            out_indices=out_indices,
        )
        self.assertEqual(dt[0].tolist(), [11, 12, 13])
        self.assertEqual(dt.shape, (2, 3))
        self.assertEqual(dt.data_ptr(), out_tokens.data_ptr())
        self.assertEqual(parents[0, :2].tolist(), [-1, 0])
        self.assertEqual(indices[1, :3].tolist(), [3, 4, 5])

    def test_sync_kv_from_cpu_lengths_sets_bonus_and_finished(self):
        if torch is None:
            self.skipTest("torch not available")
        import ast
        from pathlib import Path

        from sglang.srt.speculative.standalone_remote.sr_protocol import (
            is_health_check_req,
        )

        path = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/speculative/standalone_remote/verifier/sr_worker.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name in {"_req_committed_len", "_sync_kv_from_cpu_lengths"}
        ]
        ns = {
            "torch": torch,
            "_is_health_check": is_health_check_req,
        }
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(
                        body=[
                            ast.parse("from __future__ import annotations").body[0],
                            *nodes,
                        ],
                        type_ignores=[],
                    )
                ),
                str(path),
                "exec",
            ),
            ns,
        )
        _sync_kv_from_cpu_lengths = ns["_sync_kv_from_cpu_lengths"]

        live = SimpleNamespace(
            rid="live",
            origin_input_ids=[1, 2],
            output_ids=[3],
            kv_committed_len=3,
            kv_allocated_len=3,
        )
        finished = SimpleNamespace(
            rid="finished",
            origin_input_ids=[1, 2],
            output_ids=[3, 4],
            kv_committed_len=3,
            kv_allocated_len=3,
        )
        health = SimpleNamespace(
            rid="HEALTH_CHECK_0",
            origin_input_ids=[1],
            output_ids=[],
            kv_committed_len=1,
            kv_allocated_len=1,
        )
        batch = SimpleNamespace(
            reqs=[live, finished, health],
            seq_lens=torch.tensor([3, 3, 1]),
        )
        _sync_kv_from_cpu_lengths(
            batch,
            torch.tensor([3, 3, 1], dtype=torch.int64),
            [2, 0, 9],
        )
        self.assertEqual(live.kv_committed_len, 6)
        self.assertEqual(live.kv_allocated_len, 6)
        self.assertEqual(finished.kv_committed_len, 4)
        self.assertEqual(finished.kv_allocated_len, 4)
        self.assertEqual(health.kv_committed_len, 1)
        self.assertEqual(health.kv_allocated_len, 1)

        paged = SimpleNamespace(
            rid="paged",
            origin_input_ids=[1, 2, 3],
            output_ids=[4],
            kv_committed_len=99,
            kv_allocated_len=99,
        )
        stale_device = SimpleNamespace(
            reqs=[paged],
            seq_lens=torch.tensor([4]),
        )
        _sync_kv_from_cpu_lengths(stale_device, torch.tensor([4]), [1])
        self.assertEqual(paged.kv_committed_len, 6)
        self.assertEqual(paged.kv_allocated_len, 6)

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

        def run_ready(reqs, capture_tree_seed=False, teacher_forcing=False):
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
        mixin._sr_ingest_committed_batch([req])
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

        def run_ready(reqs, capture_tree_seed=False, teacher_forcing=False):
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

        def run_ready(reqs, capture_tree_seed=False, teacher_forcing=False):
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


class TestReplayGrammarFromCommitted(CustomTestCase):
    def test_none_template(self):
        self.assertIsNone(replay_grammar_from_committed(None, [1, 2]))

    def test_replays_committed_ids_in_order(self):
        accepted = []

        class FakeGrammar:
            def copy(self):
                child = FakeGrammar()
                child._accepted = accepted
                return child

            def accept_token(self, token):
                self._accepted.append(int(token))

        template = FakeGrammar()
        out = replay_grammar_from_committed(template, [7, 8, 9])
        self.assertIsNot(out, template)
        self.assertEqual(accepted, [7, 8, 9])

    def test_empty_committed(self):
        class FakeGrammar:
            def __init__(self):
                self.accepted = []

            def copy(self):
                return FakeGrammar()

            def accept_token(self, token):
                self.accepted.append(token)

        out = replay_grammar_from_committed(FakeGrammar(), [])
        self.assertEqual(out.accepted, [])
        out = replay_grammar_from_committed(FakeGrammar(), None)
        self.assertEqual(out.accepted, [])


class TestSRDraftStateTTL(CustomTestCase):
    def test_cleanup_pops_idle_keeps_active_and_keep_rids(self):
        mgr = SRDraftStateManager(timeout_threshold=10.0)
        stale = SRDraftState(req_id="old", session_id="s")
        mgr.set("old", stale)
        stale.last_updated_time = 0.0
        fresh = SRDraftState(req_id="keep", session_id="s")
        mgr.set("keep", fresh)
        protected = SRDraftState(req_id="live", session_id="s")
        mgr.set("live", protected)
        protected.last_updated_time = 0.0

        popped = mgr.cleanup_stale_states(now=100.0, keep_rids={"live"})
        self.assertEqual([s.req_id for s in popped], ["old"])
        self.assertFalse(mgr.exists("old"))
        self.assertTrue(mgr.exists("keep"))
        self.assertTrue(mgr.exists("live"))

    def test_set_refreshes_last_updated_time(self):
        mgr = SRDraftStateManager(timeout_threshold=10.0)
        state = SRDraftState(req_id="r", session_id="s")
        state.last_updated_time = 0.0
        mgr.set("r", state)
        self.assertGreater(state.last_updated_time, 0.0)

    def test_nonpositive_timeout_disables(self):
        mgr = SRDraftStateManager(timeout_threshold=0.0)
        state = SRDraftState(req_id="r", session_id="s")
        mgr.set("r", state)
        state.last_updated_time = 0.0
        self.assertEqual(mgr.cleanup_stale_states(now=1e9), [])
        self.assertTrue(mgr.exists("r"))


class TestSRRpcBreaker(CustomTestCase):
    def test_three_failures_open_then_cooldown_half_open_success_closes(self):
        b = SRRpcBreaker(failure_threshold=3, cooldown_steps=4)
        self.assertTrue(b.should_send())
        b.record_failure()
        b.record_failure()
        self.assertTrue(b.should_send())
        b.record_failure()
        self.assertFalse(b.should_send())
        self.assertEqual(b.state, SRRpcBreaker.OPEN)

        for _ in range(3):
            b.note_skipped_step()
            self.assertFalse(b.should_send())
        b.note_skipped_step()
        self.assertTrue(b.should_send())
        self.assertEqual(b.state, SRRpcBreaker.HALF_OPEN)

        b.record_success()
        self.assertTrue(b.should_send())
        self.assertEqual(b.state, SRRpcBreaker.CLOSED)
        self.assertEqual(b.consecutive_failures, 0)

    def test_reject_must_not_record_failure(self):
        """Draft REJECT is a live reply; only timeouts call record_failure."""
        b = SRRpcBreaker(failure_threshold=3, cooldown_steps=4)
        b.record_success()
        self.assertTrue(b.should_send())
        b.record_success()
        b.record_success()
        self.assertEqual(b.state, SRRpcBreaker.CLOSED)
        self.assertEqual(b.consecutive_failures, 0)

    def test_open_timeout_does_not_reset_cooldown(self):
        b = SRRpcBreaker(failure_threshold=3, cooldown_steps=4)
        for _ in range(3):
            b.record_failure()
        b.note_skipped_step()
        b.note_skipped_step()
        b.record_failure()
        b.note_skipped_step()
        b.note_skipped_step()
        self.assertTrue(b.should_send())
        self.assertEqual(b.state, SRRpcBreaker.HALF_OPEN)

    def test_observe_reject_packet_is_success(self):
        try:
            from sglang.srt.speculative.standalone_remote.verifier.sr_target_scheduler_mixin import (
                SRInflight,
                SchedulerStandaloneRemoteTargetMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))

        mixin = SchedulerStandaloneRemoteTargetMixin()
        mixin.sr_breaker = SRRpcBreaker(failure_threshold=3, cooldown_steps=4)
        for _ in range(3):
            mixin.sr_breaker.record_failure()
        self.assertFalse(mixin.sr_breaker.should_send())
        inflight = SRInflight(
            session_id="s",
            rpc_seq=1,
            pending={},
            wait_reply=True,
            send_ok=True,
        )
        mixin._sr_observe_rpc(inflight, got_packet=True)
        self.assertTrue(mixin.sr_breaker.should_send())
        self.assertEqual(mixin.sr_breaker.state, SRRpcBreaker.CLOSED)

    def test_observe_timeout_is_failure(self):
        try:
            from sglang.srt.speculative.standalone_remote.verifier.sr_target_scheduler_mixin import (
                SRInflight,
                SchedulerStandaloneRemoteTargetMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))

        mixin = SchedulerStandaloneRemoteTargetMixin()
        mixin.sr_breaker = SRRpcBreaker(failure_threshold=3, cooldown_steps=4)
        inflight = SRInflight(
            session_id="s",
            rpc_seq=1,
            pending={},
            wait_reply=True,
            send_ok=True,
        )
        mixin._sr_observe_rpc(inflight, got_packet=False)
        mixin._sr_observe_rpc(inflight, got_packet=False)
        mixin._sr_observe_rpc(inflight, got_packet=False)
        self.assertFalse(mixin.sr_breaker.should_send())


class TestSRDraftBusyReject(CustomTestCase):
    def test_busy_rejects_without_producing_windows(self):
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
        mixin._sr_draft_busy = lambda: True
        produce_calls = []
        mixin._sr_produce_windows = lambda *_a, **_k: produce_calls.append(1) or []
        mixin._sr_prepare_one = lambda *_a, **_k: (None, MagicMock())
        batch = SRBatchRequest(
            session_id="s",
            rpc_seq=1,
            action=SRAction.STEP,
            reqs=[
                SRDraftRequest(
                    rid="a", step_id=1, base_committed_len=4, num_draft_tokens=5
                )
            ],
        )
        reply = mixin._sr_handle_batch(batch, {})
        self.assertEqual(produce_calls, [])
        self.assertEqual(len(reply.reqs), 1)
        self.assertEqual(reply.reqs[0].status, SRReplyStatus.REJECT)
        self.assertEqual(reply.reqs[0].draft_tokens, [])

    def test_busy_still_handles_finish(self):
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
        mixin._sr_draft_busy = lambda: True
        prepare_calls = []

        def fake_prepare(dreq, *_a, **_k):
            prepare_calls.append(dreq.rid)
            return mixin._sr_empty_reply(dreq), None

        mixin._sr_prepare_one = fake_prepare
        mixin._sr_produce_windows = lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("produce must not run")
        )
        batch = SRBatchRequest(
            session_id="s",
            rpc_seq=1,
            action=SRAction.FINISH,
            reqs=[SRDraftRequest(rid="a", step_id=1, base_committed_len=4)],
        )
        reply = mixin._sr_handle_batch(batch, {})
        self.assertEqual(prepare_calls, ["a"])
        self.assertEqual(reply.reqs[0].status, SRReplyStatus.EMPTY)

    def _make_draft_mixin(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
            StandaloneRemoteDraftSchedulerMixin,
        )

        mixin = StandaloneRemoteDraftSchedulerMixin()
        mixin.sr_waiting = []
        mixin.draft_paused_reqs = []
        mixin.waiting_queue = []
        mixin.running_batch = MagicMock()
        mixin.running_batch.is_empty.return_value = True
        mixin.sr_kv = MagicMock()
        mixin._maybe_compute_mrope_positions = MagicMock()
        mixin.server_args = SimpleNamespace(speculative_eagle_topk=2)
        return mixin

    def test_sr_retract_skips_waiting_queue_and_resets_mm(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                StandaloneRemoteDraftSchedulerMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))
        mixin = self._make_draft_mixin()
        req = SimpleNamespace(
            rid="sr-1",
            is_sr_draft=True,
            origin_input_ids=[1, 2, 3],
            output_ids=[4, 5],
            req_pool_idx=None,
            sr_tree_seed=object(),
            sr_padded_ids=[1, 2, 3],
            multimodal_inputs=MagicMock(),
            is_retracted=True,
            retracted_stain=True,
            kv_committed_len=5,
        )
        handled = mixin._sr_intercept_retract_enqueue(req)
        self.assertTrue(handled)
        self.assertEqual(mixin.waiting_queue, [])
        self.assertEqual(req.output_ids, [])
        self.assertEqual(req.origin_input_ids, [1, 2, 3, 4, 5])
        self.assertEqual(req.fill_ids, [1, 2, 3, 4, 5])
        self.assertIsNone(req.sr_tree_seed)
        self.assertEqual(req.kv_committed_len, 0)
        self.assertIn(req, mixin.draft_paused_reqs)
        mixin._maybe_compute_mrope_positions.assert_called_once_with(req)

    def test_http_retract_is_not_intercepted(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                StandaloneRemoteDraftSchedulerMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))
        mixin = self._make_draft_mixin()
        req = SimpleNamespace(is_sr_draft=False, rid="http")
        self.assertFalse(mixin._sr_intercept_retract_enqueue(req))
        self.assertEqual(mixin.draft_paused_reqs, [])

    def test_materialize_repairs_retracted_output_tail(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                StandaloneRemoteDraftSchedulerMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))
        mixin = self._make_draft_mixin()
        mixin.sr_state = SRDraftStateManager()
        req = SimpleNamespace(
            rid="sr-2",
            is_sr_draft=True,
            origin_input_ids=[1, 2],
            output_ids=[3],
            req_pool_idx=None,
            sr_tree_seed=object(),
            multimodal_inputs=MagicMock(),
            is_retracted=True,
            retracted_stain=True,
            kv_committed_len=3,
            draft_is_paused=False,
            finished=lambda: False,
        )
        mixin.get_next_batch_to_run = lambda: None
        mixin._sr_isolate_need = lambda *_a, **_k: None
        mixin._sr_materialize_prefix_batch([req])
        self.assertEqual(req.origin_input_ids, [1, 2, 3])
        self.assertEqual(req.output_ids, [])
        self.assertIsNone(req.sr_tree_seed)
        mixin._maybe_compute_mrope_positions.assert_called_once_with(req)

    def test_materialize_rebuilds_mrope_when_pool_slot_lost(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                StandaloneRemoteDraftSchedulerMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))
        if torch is None:
            self.skipTest("torch is required")
        mixin = self._make_draft_mixin()
        mixin.sr_state = SRDraftStateManager()
        mm = SimpleNamespace(mrope_positions=torch.zeros(3, 2, dtype=torch.int64))
        req = SimpleNamespace(
            rid="sr-lost-slot",
            is_sr_draft=True,
            origin_input_ids=[1, 2],
            output_ids=[3, 4, 5],
            req_pool_idx=None,
            sr_tree_seed=object(),
            multimodal_inputs=mm,
            is_retracted=False,
            retracted_stain=False,
            kv_committed_len=5,
            draft_is_paused=False,
            finished=lambda: False,
        )

        def recompute(r):
            r.multimodal_inputs.mrope_positions = torch.zeros(
                3, len(r.origin_input_ids), dtype=torch.int64
            )

        mixin._maybe_compute_mrope_positions = recompute
        mixin.get_next_batch_to_run = lambda: None
        mixin._sr_isolate_need = lambda *_a, **_k: None
        mixin._sr_materialize_prefix_batch([req])
        self.assertEqual(req.origin_input_ids, [1, 2, 3, 4, 5])
        self.assertEqual(req.output_ids, [])
        self.assertEqual(tuple(req.multimodal_inputs.mrope_positions.shape), (3, 5))
        self.assertIn(req, mixin.draft_paused_reqs)

    def test_cache_tree_seed_reraises_cuda_ima(self):
        if torch is None:
            self.skipTest("torch not available")
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                StandaloneRemoteDraftSchedulerMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))
        mixin = self._make_draft_mixin()
        req = SimpleNamespace(rid="r", output_ids=[7], origin_input_ids=[1])
        result = MagicMock()
        result.logits_output.hidden_states = torch.ones(1, 4)
        result.logits_output.tree_seed_topk_p = torch.ones(1, 2)
        result.logits_output.tree_seed_topk_index = torch.ones(1, 2, dtype=torch.int64)
        with patch(
            "sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin.slice_decode_batch_row",
            side_effect=RuntimeError(
                "CUDA error: an illegal memory access was encountered"
            ),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                mixin._sr_cache_tree_seeds([req], result, None)
        self.assertIn("illegal memory access", str(ctx.exception).lower())

    def test_cache_tree_seed_shape_error_stays_warning(self):
        if torch is None:
            self.skipTest("torch not available")
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                StandaloneRemoteDraftSchedulerMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))
        mixin = self._make_draft_mixin()
        req = SimpleNamespace(rid="r", output_ids=[7], origin_input_ids=[1])
        result = MagicMock()
        result.logits_output.hidden_states = torch.ones(1, 4)
        result.logits_output.tree_seed_topk_p = torch.ones(1, 2)
        result.logits_output.tree_seed_topk_index = torch.ones(1, 2, dtype=torch.int64)
        with patch(
            "sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin.slice_decode_batch_row",
            side_effect=ValueError("shape mismatch"),
        ):
            mixin._sr_cache_tree_seeds([req], result, None)

    def _load_cache_tree_seeds(self):
        import ast
        from pathlib import Path

        def stamp_tree_seed(req, boundary):
            req.sr_tree_seed_boundary = boundary
            req.sr_tree_seed_revision = int(getattr(req, "sr_prefix_revision", 0))

        def slice_decode_batch_row(tensor, i, n, token_lens=None):
            del n, token_lens
            return tensor[i : i + 1]

        src_path = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/speculative/standalone_remote/drafter/"
            "sr_draft_scheduler_mixin.py"
        )
        tree = ast.parse(src_path.read_text(encoding="utf-8"))
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef)
            and n.name == "StandaloneRemoteDraftSchedulerMixin"
        )
        fn = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_sr_cache_tree_seeds"
        )
        dummy = ast.ClassDef(
            name="LoadedMixin",
            bases=[],
            keywords=[],
            body=[fn],
            decorator_list=[],
        )
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        mod = ast.fix_missing_locations(
            ast.Module(body=[future, dummy], type_ignores=[])
        )
        ns = {
            "torch": torch,
            "List": list,
            "Optional": type(None),
            "logger": SimpleNamespace(
                warning=lambda *_a, **_k: None, error=lambda *_a, **_k: None
            ),
            "slice_decode_batch_row": slice_decode_batch_row,
            "stamp_tree_seed": stamp_tree_seed,
            "_sr_is_device_context_error": lambda _e: False,
        }
        exec(compile(mod, str(src_path), "exec"), ns)
        mixin = ns["LoadedMixin"]()
        mixin.server_args = SimpleNamespace(speculative_eagle_topk=2)
        return mixin

    def test_cache_tree_seed_writes_none_hidden_without_advancing_kv(self):
        if torch is None:
            self.skipTest("torch not available")
        try:
            mixin = self._load_cache_tree_seeds()
        except ImportError as e:
            self.skipTest(str(e))
        req = SimpleNamespace(
            rid="r",
            output_ids=[7],
            origin_input_ids=[1],
            kv_committed_len=3,
        )
        hidden = torch.ones(1, 4)
        result = SimpleNamespace(
            logits_output=SimpleNamespace(
                hidden_states=hidden,
                tree_seed_topk_p=torch.tensor([[0.5, 0.5]]),
                tree_seed_topk_index=torch.tensor([[3, 4]], dtype=torch.int64),
            )
        )
        mixin._sr_cache_tree_seeds([req], result, None)
        self.assertIsNone(req.sr_tree_seed[2])
        self.assertEqual(req.sr_tree_seed[3].tolist(), [7])
        self.assertEqual(req.sr_tree_seed[3].device, req.sr_tree_seed[1].device)
        self.assertEqual(req.kv_committed_len, 3)
        result.logits_output.tree_seed_topk_p.zero_()
        self.assertEqual(req.sr_tree_seed[0].tolist(), [[0.5, 0.5]])

    def test_cache_tree_seed_rejects_width_inferred_from_table(self):
        if torch is None:
            self.skipTest("torch not available")
        try:
            mixin = self._load_cache_tree_seeds()
        except ImportError as e:
            self.skipTest(str(e))
        req = SimpleNamespace(
            rid="r",
            output_ids=[7],
            origin_input_ids=[1],
            sr_tree_seed=None,
        )
        result = SimpleNamespace(
            logits_output=SimpleNamespace(
                hidden_states=torch.ones(1, 4),
                tree_seed_topk_p=torch.ones(1, 3),
                tree_seed_topk_index=torch.ones(1, 3, dtype=torch.int64),
            )
        )
        mixin._sr_cache_tree_seeds([req], result, None)
        self.assertIsNone(req.sr_tree_seed)

    def test_align_replace_tail_written_last_token_rollbacks_allocated(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                StandaloneRemoteDraftSchedulerMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))
        mixin = self._make_draft_mixin()
        mixin._sr_reprefill = MagicMock()
        mixin.sr_kv.get_prefix_len.return_value = 2
        mixin.sr_kv.rollback.return_value = True
        req = SimpleNamespace(
            origin_input_ids=[1, 2],
            output_ids=[9],
            sr_padded_ids=[1, 2],
            sr_tree_seed=object(),
            kv_committed_len=3,
            kv_allocated_len=5,
            draft_generation_start_len=0,
            draft_tokens_target=0,
        )
        dreq = SimpleNamespace(committed_ids=[8], num_draft_tokens=4)
        mixin._sr_align(req, dreq, SimpleNamespace())
        self.assertEqual(req.output_ids, [8])
        self.assertIsNone(req.sr_tree_seed)
        mixin.sr_kv.rollback.assert_called_once_with(req, 2, 5)
        mixin._sr_reprefill.assert_not_called()

    def test_align_replace_tail_unwritten_does_not_rollback(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                StandaloneRemoteDraftSchedulerMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))
        mixin = self._make_draft_mixin()
        mixin._sr_reprefill = MagicMock()
        mixin.sr_kv.get_prefix_len.return_value = 2
        req = SimpleNamespace(
            origin_input_ids=[1, 2],
            output_ids=[9],
            sr_padded_ids=[1, 2],
            sr_tree_seed=object(),
            kv_committed_len=2,
            kv_allocated_len=5,
            draft_generation_start_len=0,
            draft_tokens_target=0,
        )
        dreq = SimpleNamespace(committed_ids=[8], num_draft_tokens=4)
        mixin._sr_align(req, dreq, SimpleNamespace())
        self.assertEqual(req.output_ids, [8])
        self.assertIsNone(req.sr_tree_seed)
        mixin.sr_kv.rollback.assert_not_called()
        mixin._sr_reprefill.assert_not_called()

    def test_cuda_context_error_helper(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                _sr_is_cuda_context_error,
            )
        except ImportError as e:
            self.skipTest(str(e))
        self.assertTrue(
            _sr_is_cuda_context_error(
                RuntimeError("CUDA error: an illegal memory access was encountered")
            )
        )
        self.assertFalse(_sr_is_cuda_context_error(ValueError("shape mismatch")))
        if torch is not None and hasattr(torch, "AcceleratorError"):
            try:
                err = torch.AcceleratorError("cuda boom")
            except TypeError:
                err = None
            if err is not None:
                self.assertTrue(_sr_is_cuda_context_error(err))


class TestPlanCommittedIngest(CustomTestCase):
    def test_noop_when_tail_already_in_kv(self):
        mode, tail = plan_committed_ingest(3, [10, 11], 5)
        self.assertEqual(mode, "noop")
        self.assertEqual(tail, [])

    def test_decode_for_short_tail(self):
        mode, tail = plan_committed_ingest(3, [10, 11, 12], 3)
        self.assertEqual(mode, "decode")
        self.assertEqual(tail, [10, 11, 12])

    def test_reprefill_for_long_tail(self):
        committed = list(range(100, 340))
        mode, tail = plan_committed_ingest(3, committed, 3, max_decode_steps=16)
        self.assertEqual(mode, "reprefill")
        self.assertEqual(len(tail), 240)

    def test_threshold_boundary(self):
        committed = list(range(16))
        self.assertEqual(
            plan_committed_ingest(0, committed, 0, max_decode_steps=16)[0], "decode"
        )
        self.assertEqual(
            plan_committed_ingest(0, committed + [99], 0, max_decode_steps=16)[0],
            "reprefill",
        )


class TestSRTeacherForcedIngest(CustomTestCase):
    def _mixin(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                StandaloneRemoteDraftSchedulerMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))
        return StandaloneRemoteDraftSchedulerMixin()

    def test_suppress_local_finish_keeps_to_finish(self):
        try:
            from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
        except ImportError as e:
            self.skipTest(str(e))
        req = Req.__new__(Req)
        req.finished_reason = None
        req.to_finish = None
        req.output_ids = [151645]
        req.sampling_params = SimpleNamespace(
            max_new_tokens=1,
            ignore_eos=False,
            stop_token_ids=None,
            stop_strs=[],
            stop_regex_strs=[],
        )
        req.eos_token_ids = {151645}
        req.grammar = None
        req.tokenizer = None
        req.vocab_size = 200000
        req.spec_type = None
        req.suppress_local_finish = True
        req.check_finished()
        self.assertIsNone(req.finished_reason)

        abort = FINISH_ABORT("Target request finished")
        req.to_finish = abort
        req.check_finished()
        self.assertIs(req.finished_reason, abort)

    def test_draft_eos_does_not_truncate_ingest(self):
        try:
            from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN
        except ImportError as e:
            self.skipTest(str(e))
        mixin = self._mixin()
        req = MagicMock()
        req.rid = "a"
        req.origin_input_ids = [1, 2, 3]
        req.output_ids = [10, 11, 12]
        req.kv_committed_len = 3
        req.finished_reason = None
        steps = []

        def run_ready(reqs, capture_tree_seed=False):
            r = reqs[0]
            steps.append(list(r.output_ids))
            r.kv_committed_len = 3 + len(r.output_ids)
            # Draft samples EOS on the first ingest step.
            r.finished_reason = FINISH_MATCHED_TOKEN(matched=151645)

        mixin._sr_ensure_window_budget = lambda *_a, **_k: None
        mixin._sr_run_until_ready = run_ready
        mixin._sr_ingest_committed_batch([req])

        self.assertEqual(steps, [[10], [10, 11], [10, 11, 12]])
        self.assertEqual(req.output_ids, [10, 11, 12])
        self.assertEqual(
            committed_tail_not_in_kv(3, req.output_ids, req.kv_committed_len), []
        )

    def test_long_tail_uses_single_reprefill(self):
        mixin = self._mixin()
        mixin.sr_state = SRDraftStateManager()
        committed = list(range(100, 340))
        req = MagicMock()
        req.rid = "a"
        req.origin_input_ids = [1, 2, 3]
        req.output_ids = list(committed)
        req.kv_committed_len = 3
        req.sr_padded_ids = [1, 2, 3]
        req.draft_tokens_target = 4
        resets = []
        materialized = []

        mixin._sr_ensure_window_budget = lambda *_a, **_k: None
        mixin._sr_enqueue_for_reprefill = lambda r, ids: resets.append(
            (r.rid, len(ids))
        ) or True
        mixin._sr_materialize_prefix_batch = lambda reqs: materialized.append(
            [r.rid for r in reqs]
        )
        mixin._sr_run_until_ready = MagicMock()
        mixin._sr_ingest_committed_batch([req])

        self.assertEqual(resets, [("a", 243)])
        self.assertEqual(materialized, [["a"]])
        mixin._sr_run_until_ready.assert_not_called()


class TestSRDegradedRequests(CustomTestCase):
    def _mixin(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                StandaloneRemoteDraftSchedulerMixin,
            )
        except ImportError as e:
            self.skipTest(str(e))
        return StandaloneRemoteDraftSchedulerMixin()

    def test_leftover_marks_degraded(self):
        mixin = self._mixin()
        mixin.sr_state = SRDraftStateManager()
        req = MagicMock()
        req.rid = "a"
        req.origin_input_ids = [1, 2, 3]
        req.output_ids = [10, 11]
        req.kv_committed_len = 4
        req.req_pool_idx = None
        mixin.sr_state.set("a", SRDraftState(req_id="a", session_id="s", req_object=req))
        mixin.sr_tree_drafter = None
        mixin._sr_materialize_prefix_batch = lambda reqs: None
        mixin._sr_run_tree_ingest = lambda reqs: True
        mixin._sr_ensure_tree_seeds = lambda reqs: None
        mixin._sr_replay_grammars = lambda reqs: None

        windows = mixin._sr_tree_expand_batch([req])

        self.assertEqual(windows, [([], None, None)])
        self.assertTrue(mixin.sr_state.get("a").degraded)

    def test_degraded_rid_replies_empty_without_align(self):
        mixin = self._mixin()
        mixin.sr_state = SRDraftStateManager()
        req = MagicMock()
        req.rid = "a"
        state = SRDraftState(req_id="a", session_id="s", req_object=req)
        state.degraded = True
        state.last_rpc_seq = 1
        state.last_step_id = 0
        state.last_base_committed_len = 3
        mixin.sr_state.set("a", state)
        mixin._sr_align = MagicMock()

        dreq = SRDraftRequest(
            rid="a",
            step_id=1,
            base_committed_len=4,
            committed_ids=[10],
            num_draft_tokens=4,
        )
        reply, live = mixin._sr_prepare_one(
            dreq,
            SRAction.STEP,
            session_id="s",
            rpc_seq=2,
            mm=None,
            last_session_id="s",
            last_rpc_seq=1,
        )
        self.assertIsNone(live)
        self.assertEqual(reply.status, SRReplyStatus.EMPTY)
        self.assertEqual(reply.draft_tokens, [])
        mixin._sr_align.assert_not_called()

    def test_mm_embed_error_detects_released_features(self):
        mixin = self._mixin()
        item = SimpleNamespace(
            is_image=lambda: True,
            is_video=lambda: False,
            feature=None,
            precomputed_embeddings=None,
        )
        req = SimpleNamespace(
            rid="a", multimodal_inputs=SimpleNamespace(mm_items=[item])
        )
        self.assertIsNotNone(mixin._sr_mm_embed_error(req))
        item.feature = object()
        self.assertIsNone(mixin._sr_mm_embed_error(req))
        self.assertIsNone(
            mixin._sr_mm_embed_error(SimpleNamespace(multimodal_inputs=None))
        )


class TestSchedulerSkipsDraftMmClear(CustomTestCase):
    def test_sr_draft_req_keeps_mm_inputs(self):
        try:
            from sglang.srt.managers.scheduler import Scheduler
        except ImportError as e:
            self.skipTest(str(e))

        def make_req(is_sr_draft):
            req = MagicMock()
            req.finished.return_value = True
            req.session = None
            req.is_sr_draft = is_sr_draft
            req.multimodal_inputs = MagicMock()
            return req

        draft_req = make_req(True)
        normal_req = make_req(False)
        batch = SimpleNamespace(reqs=[draft_req, normal_req])

        Scheduler._maybe_clear_mm_inputs(MagicMock(), batch)

        self.assertIsNotNone(draft_req.multimodal_inputs)
        self.assertIsNone(normal_req.multimodal_inputs)


class TestComputeMropeWidthFallback(CustomTestCase):
    def test_extend_falls_back_when_cached_mrope_is_short(self):
        if torch is None:
            self.skipTest("torch is required")
        try:
            from sglang.srt.model_executor.forward_batch_info import (
                ForwardBatch,
                ForwardMode,
            )
        except ImportError as e:
            self.skipTest(str(e))

        fb = ForwardBatch.__new__(ForwardBatch)
        fb.seq_lens_cpu = torch.tensor([10])
        fb.forward_mode = ForwardMode.EXTEND
        fb.input_ids = torch.arange(8, dtype=torch.int64)
        mm = SimpleNamespace(
            mrope_positions=torch.arange(5).unsqueeze(0).repeat(3, 1),
            mrope_position_delta=torch.tensor([0]),
            mrope_position_delta_repeated_cache=None,
        )
        batch = SimpleNamespace(
            multimodal_inputs=[mm],
            extend_seq_lens=[8],
            extend_prefix_lens=[0],
        )
        fallback = torch.arange(8).unsqueeze(0).repeat(3, 1)
        fb._expand_mrope_from_input = MagicMock(return_value=fallback)
        with patch(
            "sglang.srt.model_executor.forward_batch_info.get_global_server_args",
            return_value=SimpleNamespace(rl_on_policy_target=None),
        ):
            fb._compute_mrope_positions(SimpleNamespace(device="cpu"), batch)
        fb._expand_mrope_from_input.assert_called_once()
        self.assertEqual(tuple(fb.mrope_positions.shape), (3, 8))

    def test_extend_keeps_cached_mrope_when_width_matches(self):
        if torch is None:
            self.skipTest("torch is required")
        try:
            from sglang.srt.model_executor.forward_batch_info import (
                ForwardBatch,
                ForwardMode,
            )
        except ImportError as e:
            self.skipTest(str(e))

        fb = ForwardBatch.__new__(ForwardBatch)
        fb.seq_lens_cpu = torch.tensor([8])
        fb.forward_mode = ForwardMode.EXTEND
        fb.input_ids = torch.arange(5, dtype=torch.int64)
        cached = torch.arange(8).unsqueeze(0).repeat(3, 1)
        mm = SimpleNamespace(mrope_positions=cached)
        batch = SimpleNamespace(
            multimodal_inputs=[mm],
            extend_seq_lens=[5],
            extend_prefix_lens=[3],
        )
        fb._expand_mrope_from_input = MagicMock(
            side_effect=AssertionError("must not fall back")
        )
        with patch(
            "sglang.srt.model_executor.forward_batch_info.get_global_server_args",
            return_value=SimpleNamespace(rl_on_policy_target=None),
        ):
            fb._compute_mrope_positions(SimpleNamespace(device="cpu"), batch)
        fb._expand_mrope_from_input.assert_not_called()
        self.assertEqual(tuple(fb.mrope_positions.shape), (3, 5))
        self.assertTrue(torch.equal(fb.mrope_positions.cpu(), cached[:, 3:8].cpu()))


if __name__ == "__main__":
    unittest.main()
