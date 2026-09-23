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

    def test_assemble_chain_tree_and_empty_list_expected(self):
        if torch is None:
            self.skipTest("torch not available")
        from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
            assemble_draft_rows,
            cached_chain_template,
        )

        parents, indices, tokens = assemble_draft_rows(
            [[11, 12, 13]],
            [None],
            [None],
            topk=2,
            spec_steps=3,
            num_draft_tokens=4,
        )
        self.assertEqual(tokens.tolist(), [[11, 12, 13]])
        self.assertEqual(parents.tolist(), [[-1, 0, 1]])
        self.assertEqual(indices.tolist(), [[0, 1, 2]])

        parents, indices, tokens = assemble_draft_rows(
            [[11, 12, 13]],
            [[9, 9, 9, 9]],
            [[7, 7, 7]],
            topk=1,
            spec_steps=3,
            num_draft_tokens=4,
        )
        self.assertEqual(tokens.tolist(), [[11, 12, 13]])
        self.assertEqual(parents.tolist(), [[-1, 0, 1]])
        self.assertEqual(indices.tolist(), [[0, 1, 2]])

        parents, indices, tokens = assemble_draft_rows(
            [[11, 12]],
            [[]],
            [[]],
            topk=2,
            spec_steps=3,
            num_draft_tokens=4,
        )
        self.assertEqual(tokens.tolist(), [[11, 12, 0]])
        self.assertEqual(parents.tolist(), [[-1, -1, -1]])
        self.assertEqual(indices.tolist(), [[0, 0, 0]])

        parents, indices, tokens = assemble_draft_rows(
            [[11, 12, 13], [21]],
            [[-1, 4, 5, 6], None],
            [[8, 7, 6, 5], [1]],
            topk=2,
            spec_steps=3,
            num_draft_tokens=4,
        )
        self.assertEqual(tokens.tolist(), [[11, 12, 13], [21, 0, 0]])
        self.assertEqual(parents.tolist(), [[-1, 4, 5, 6], [-1, 0, 1, -1]])
        self.assertEqual(indices.tolist(), [[8, 7, 6, 5], [0, 1, 2, 0]])

        first_parents, first_indices = cached_chain_template(4, 3)
        parent_ptr = first_parents.data_ptr()
        index_ptr = first_indices.data_ptr()
        parent_snapshot = first_parents.clone()
        index_snapshot = first_indices.clone()
        again_parents, again_indices = cached_chain_template(4, 3)
        self.assertEqual(again_parents.data_ptr(), parent_ptr)
        self.assertEqual(again_indices.data_ptr(), index_ptr)
        self.assertEqual(again_parents.tolist(), parent_snapshot.tolist())
        self.assertEqual(again_indices.tolist(), index_snapshot.tolist())

    def test_packet_width_changes_do_not_reuse_old_stride(self):
        if torch is None:
            self.skipTest("torch not available")
        from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
            packet_segment_views,
            plan_draft_rows,
            write_draft_regions,
        )

        storage = torch.full((64,), 99, dtype=torch.int64)

        def fill(tokens, parents, indices):
            plan = plan_draft_rows([tokens], [parents], [indices], 2, 3, 4)
            verified, tok, par, idx = packet_segment_views(storage, plan)
            verified[0] = 5
            write_draft_regions(tok, par, idx, plan)
            return plan, verified, tok, par, idx

        wide, verified, tokens, parents, indices = fill(
            [11, 12, 13],
            [-1, 0, 1, 2, 3, 4],
            [9, 8, 7, 6, 5, 4, 3, 2],
        )
        self.assertEqual(wide.used, 18)
        self.assertEqual(wide.parent_w, 6)
        self.assertEqual(wide.index_w, 8)
        self.assertEqual(verified.tolist(), [5])
        self.assertEqual(tokens.tolist(), [[11, 12, 13]])
        self.assertEqual(parents.tolist(), [[-1, 0, 1, 2, 3, 4]])
        self.assertEqual(indices.tolist(), [[9, 8, 7, 6, 5, 4, 3, 2]])

        chain, verified, tokens, parents, indices = fill([21, 22], None, None)
        self.assertEqual(chain.used, 10)
        self.assertEqual(chain.parent_w, 3)
        self.assertEqual(chain.index_w, 3)
        self.assertEqual(list(parents.shape), [1, 3])
        self.assertEqual(parents.tolist(), [[-1, 0, 1]])
        self.assertEqual(indices.tolist(), [[0, 1, 2]])
        self.assertEqual(tokens.tolist(), [[21, 22, 0]])
        self.assertNotIn(4, parents.reshape(-1).tolist())
        self.assertNotIn(9, indices.reshape(-1).tolist())

        wide, verified, tokens, parents, indices = fill(
            [31, 32, 33],
            [-1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        )
        self.assertEqual(wide.used, 18)
        self.assertEqual(storage.numel(), 64)
        self.assertEqual(parents.tolist(), [[-1, 1, 1, 1, 1, 1]])
        self.assertEqual(indices.tolist(), [[1, 1, 1, 1, 1, 1, 1, 1]])
        self.assertEqual(tokens.tolist(), [[31, 32, 33]])

    def _packet_worker(self):
        from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
            VerifyInputPacket,
        )

        return VerifyInputPacket()

    def _packet_batch(self, rows, metrics=None):
        reqs = []
        for verified, tokens, parents, indices in rows:
            req = SimpleNamespace(
                origin_input_ids=[verified],
                output_ids=[],
                draft_tokens_and_logits={
                    "draft_tokens": tokens,
                    "parent_list": parents,
                    "top_scores_index": indices,
                },
            )
            reqs.append(req)
        return SimpleNamespace(
            reqs=reqs,
            device=torch.device("cpu"),
            sr_round_metrics=metrics,
        )

    def _assemble(self, packet, batch, topk=2):
        verified_ids = []
        token_rows = []
        parent_rows = []
        index_rows = []
        for req in batch.reqs:
            verified_ids.append(
                req.output_ids[-1]
                if len(req.output_ids) > 0
                else req.origin_input_ids[-1]
            )
            draft = req.draft_tokens_and_logits
            token_rows.append(None if draft is None else draft.get("draft_tokens"))
            parent_rows.append(None if draft is None else draft.get("parent_list"))
            index_rows.append(None if draft is None else draft.get("top_scores_index"))
        return packet.load(
            verified_ids,
            token_rows,
            parent_rows,
            index_rows,
            topk,
            3,
            4,
            batch.device,
            batch.sr_round_metrics,
        )

    def test_cpu_packet_skips_submit_and_drops_stale_rows(self):
        if torch is None:
            self.skipTest("torch not available")
        from collections import Counter

        packet = self._packet_worker()
        from sglang.srt.speculative.standalone_remote import sr_verify_layout as layout_mod

        def fail_submit(*_args, **_kwargs):
            raise AssertionError("cpu path called submit_copy")

        metrics = SimpleNamespace(host=Counter(), counts=Counter())
        with patch.object(layout_mod, "submit_copy", fail_submit):
            verified, parents, indices, tokens = self._assemble(
                packet,
                self._packet_batch(
                    [
                        (10, [11, 12, 13], [-1, 4, 5, 6], [8, 7, 6, 5]),
                        (20, [21, 22], None, None),
                    ],
                    metrics,
                ),
            )
            self.assertEqual(verified.tolist(), [10, 20])
            self.assertEqual(tokens.tolist(), [[11, 12, 13], [21, 22, 0]])
            self.assertEqual(parents.tolist(), [[-1, 4, 5, 6], [-1, 0, 1, -1]])
            self.assertEqual(indices.tolist(), [[8, 7, 6, 5], [0, 1, 2, 0]])
            host_ptr = packet.host_packet.data_ptr()
            capacity = packet.capacity

            verified, parents, indices, tokens = self._assemble(
                packet,
                self._packet_batch([(30, [7], None, None)], metrics),
            )
            self.assertEqual(verified.tolist(), [30])
            self.assertEqual(tokens.tolist(), [[7, 0, 0]])
            self.assertEqual(parents.tolist(), [[-1, 0, 1]])
            self.assertNotIn(20, verified.tolist())
            self.assertEqual(packet.host_packet.data_ptr(), host_ptr)
            self.assertEqual(packet.capacity, capacity)

            verified, parents, indices, tokens = self._assemble(
                packet,
                self._packet_batch(
                    [
                        (40, [8, 9], None, None),
                        (50, [6], None, None),
                    ],
                    metrics,
                ),
            )
        self.assertEqual(verified.tolist(), [40, 50])
        self.assertEqual(tokens.tolist(), [[8, 9, 0], [6, 0, 0]])
        self.assertEqual(parents.tolist(), [[-1, 0, 1], [-1, 0, 1]])
        self.assertEqual(metrics.counts["verify_packet_upload"], 0)
        self.assertGreaterEqual(metrics.counts["verify_packet_grow"], 1)
        self.assertGreater(metrics.host["verify_packet_fill"], 0)
        self.assertEqual(metrics.host["verify_packet_wait"], 0)

    def test_packet_upload_once_and_unresolved_blocks_reuse(self):
        if torch is None:
            self.skipTest("torch not available")
        from collections import Counter

        packet = self._packet_worker()
        from sglang.srt.speculative.standalone_remote.sr_transfer_staging import (
            SRTransferUnresolved,
        )
        from sglang.srt.speculative.standalone_remote import sr_verify_layout as layout_mod

        packet.on_accelerator = lambda device: True
        uploads = []
        waits = []

        def fake_submit(dst, src):
            uploads.append((int(dst.numel()), int(src[0].item())))
            dst.copy_(src)
            return SimpleNamespace(name="event")

        def fake_wait(event):
            waits.append(int(packet.host_packet[0].item()))
            self.assertEqual(event.name, "event")

        metrics = SimpleNamespace(host=Counter(), counts=Counter())
        batch = self._packet_batch([(11, [1, 2, 3], None, None)], metrics)
        with patch.object(layout_mod, "submit_copy", fake_submit), patch.object(
            layout_mod, "wait_event", fake_wait
        ):
            verified, parents, indices, tokens = self._assemble(packet, batch)
            self.assertEqual(uploads, [(10, 11)])
            self.assertEqual(metrics.counts["verify_packet_upload"], 1)
            self.assertEqual(verified.tolist(), [11])
            self.assertEqual(tokens.tolist(), [[1, 2, 3]])
            self.assertEqual(parents.tolist(), [[-1, 0, 1]])
            self.assertEqual(indices.tolist(), [[0, 1, 2]])
            self.assertEqual(waits, [])

            empty = self._packet_batch([], metrics)
            empty_views = self._assemble(packet, empty)
            self.assertEqual([view.shape[0] for view in empty_views], [0, 0, 0, 0])
            self.assertEqual(len(uploads), 1)

            second = self._packet_batch([(22, [4], None, None)], metrics)
            verified, _parents, _indices, tokens = self._assemble(packet, second)
            self.assertEqual(waits, [11])
            self.assertEqual(verified.tolist(), [22])
            self.assertEqual(tokens.tolist(), [[4, 0, 0]])
            self.assertEqual(len(uploads), 2)
            self.assertGreater(metrics.host["verify_packet_wait"], 0)

        host = packet.host_packet
        device = packet.device_packet
        capacity = packet.capacity
        host_ptr = host.data_ptr()
        device_ptr = device.data_ptr()

        def wait_ok(_event):
            return None

        def record_fails(dst, src):
            dst.copy_(src)
            raise SRTransferUnresolved("record failed")

        with patch.object(layout_mod, "wait_event", wait_ok), patch.object(
            layout_mod, "submit_copy", record_fails
        ):
            with self.assertRaises(SRTransferUnresolved):
                self._assemble(
                    packet, self._packet_batch([(33, [5, 5, 5], None, None)], metrics)
                )
        self.assertTrue(packet.unresolved)
        self.assertEqual(int(host[0].item()), 33)
        filled = host.clone()
        self.assertEqual(packet.host_packet.data_ptr(), host_ptr)
        self.assertEqual(packet.device_packet.data_ptr(), device_ptr)
        self.assertEqual(packet.capacity, capacity)
        with self.assertRaises(RuntimeError):
            self._assemble(
                packet,
                self._packet_batch(
                    [
                        (44, [1], [-1, 0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7, 8]),
                        (45, [2], None, None),
                    ],
                    metrics,
                ),
            )
        self.assertEqual(packet.host_packet.data_ptr(), host_ptr)
        self.assertEqual(packet.capacity, capacity)
        self.assertEqual(host.tolist(), filled.tolist())

        packet = self._packet_worker()
        packet.on_accelerator = lambda device: True
        packet.event = SimpleNamespace(name="event")
        packet.host_packet = torch.full((16,), 7, dtype=torch.int64)
        packet.device_packet = torch.full((16,), 7, dtype=torch.int64)
        packet.capacity = 16
        host_ptr = packet.host_packet.data_ptr()
        snapshot = packet.host_packet.clone()

        def wait_fails(_event):
            raise SRTransferUnresolved("wait failed")

        def fail_submit(*_args, **_kwargs):
            raise AssertionError("wait failure still uploaded")

        with patch.object(layout_mod, "wait_event", wait_fails), patch.object(
            layout_mod, "submit_copy", fail_submit
        ):
            with self.assertRaises(SRTransferUnresolved):
                self._assemble(
                    packet, self._packet_batch([(99, [8, 8, 8], None, None)], metrics)
                )
        self.assertTrue(packet.unresolved)
        self.assertEqual(packet.host_packet.tolist(), snapshot.tolist())
        self.assertEqual(packet.capacity, 16)
        with self.assertRaises(RuntimeError):
            packet.ensure(128, torch.device("cpu"), metrics)
        self.assertEqual(packet.capacity, 16)
        self.assertEqual(packet.host_packet.data_ptr(), host_ptr)

    def test_unindexed_device_matches_indexed_buffer(self):
        if torch is None:
            self.skipTest("torch not available")
        from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
            devices_compatible,
        )

        self.assertTrue(
            devices_compatible(torch.device("cuda:0"), torch.device("cuda"))
        )
        self.assertFalse(
            devices_compatible(torch.device("cuda:0"), torch.device("cuda:1"))
        )
        self.assertTrue(devices_compatible(torch.device("cpu"), torch.device("cpu")))
        self.assertFalse(
            devices_compatible(torch.device("cpu"), torch.device("cuda"))
        )
        self.assertFalse(
            devices_compatible(torch.device("cuda"), torch.device("cuda:0"))
        )

    def test_ensure_reuses_packet_for_unindexed_device(self):
        if torch is None:
            self.skipTest("torch not available")
        npu = getattr(torch, "npu", None)
        has_npu = bool(npu is not None and npu.is_available())
        if not (torch.cuda.is_available() or has_npu):
            self.skipTest("no accelerator for packet allocation reuse")
        from collections import Counter

        from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
            VerifyInputPacket,
        )

        device_type = "npu" if has_npu else "cuda"
        packet = VerifyInputPacket()
        metrics = SimpleNamespace(host=Counter(), counts=Counter())
        packet.ensure(8, device_type, metrics)
        self.assertEqual(metrics.counts["verify_packet_grow"], 1)
        pointer = packet.device_packet.data_ptr()
        packet.ensure(8, device_type, metrics)
        self.assertEqual(metrics.counts["verify_packet_grow"], 1)
        self.assertEqual(packet.device_packet.data_ptr(), pointer)

    def test_packet_build_tree_matches_and_keeps_graph_buffers(self):
        if torch is None:
            self.skipTest("torch not available")
        npu = getattr(torch, "npu", None)
        has_npu = bool(npu is not None and npu.is_available())
        if not (torch.cuda.is_available() or has_npu):
            self.skipTest("no accelerator for build_tree")
        try:
            from sglang.srt.speculative.eagle_utils import build_tree_kernel_efficient
            from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
                packet_segment_views,
                plan_draft_rows,
                write_draft_regions,
            )
        except ImportError as exc:
            self.skipTest(str(exc))

        device = torch.device("npu") if has_npu else torch.device("cuda")
        plan = plan_draft_rows(
            [[11, 12, 13]],
            [None],
            [None],
            1,
            3,
            4,
        )
        host = torch.empty((plan.used,), dtype=torch.int64)
        verified, tokens, parents, indices = packet_segment_views(host, plan)
        verified[0] = 5
        write_draft_regions(tokens, parents, indices, plan)
        seq_lens = torch.tensor([4], dtype=torch.int64, device=device)
        mask = torch.empty((4 * 4 + 4 * 4,), dtype=torch.bool, device=device)
        positions = torch.empty((4,), dtype=torch.int64, device=device)
        mask_ptr = mask.data_ptr()
        pos_ptr = positions.data_ptr()
        try:
            packet_out = build_tree_kernel_efficient(
                verified.to(device),
                parents.to(device),
                indices.to(device),
                tokens.to(device),
                seq_lens,
                4,
                1,
                3,
                4,
                tree_mask_buf=mask,
                position_buf=positions,
            )
            separate_out = build_tree_kernel_efficient(
                verified.to(device),
                parents.to(device).clone(),
                indices.to(device).clone(),
                tokens.to(device).clone(),
                seq_lens,
                4,
                1,
                3,
                4,
                tree_mask_buf=torch.empty_like(mask),
                position_buf=torch.empty_like(positions),
            )
        except Exception as exc:
            self.skipTest(f"build_tree unavailable: {exc}")
        self.assertEqual(packet_out[0].data_ptr(), mask_ptr)
        self.assertEqual(packet_out[1].data_ptr(), pos_ptr)
        for left, right in zip(packet_out, separate_out):
            self.assertEqual(left.detach().cpu().tolist(), right.detach().cpu().tolist())

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


class _ForwardModeStub:
    def __init__(self, *, idle=False, extend=False):
        self._idle = idle
        self._extend = extend

    def is_idle(self):
        return self._idle

    def is_extend(self, include_draft_extend_v2=False):
        return self._extend


class _VerifyForwardMode:
    """Names used by ``verify()`` without importing the model-executor module."""

    IDLE = _ForwardModeStub(idle=True)
    DECODE = _ForwardModeStub()
    TARGET_VERIFY = _ForwardModeStub()


class _LogitsProbe:
    def __init__(self, data, *, forbid_get=False, forbid_meta=False):
        self.data = data
        self.forbid_get = forbid_get
        self.forbid_meta = forbid_meta
        self.gets = []

    def __getitem__(self, index):
        if self.forbid_get:
            raise AssertionError("accepted-row gather ran")
        self.gets.append(index)
        return self.data[index]

    @property
    def shape(self):
        if self.forbid_meta:
            raise AssertionError("logits shape was read")
        return self.data.shape

    def element_size(self):
        if self.forbid_meta:
            raise AssertionError("logits element size was read")
        return self.data.element_size()


class TestSRDiscardUnusedVerifyLogits(CustomTestCase):
    def _load(self):
        import ast
        import importlib.util
        import sys
        from contextlib import nullcontext
        from pathlib import Path

        from sglang.srt.speculative.standalone_remote.verifier.sr_fixed_accept import (
            conservative_mode_reason,
        )

        logprob_path = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/layers/utils/logprob.py"
        )
        logprob_spec = importlib.util.spec_from_file_location(
            "sr_test_logprob", logprob_path
        )
        logprob_mod = importlib.util.module_from_spec(logprob_spec)
        sys.modules[logprob_spec.name] = logprob_mod
        logprob_spec.loader.exec_module(logprob_mod)

        path = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/speculative/standalone_remote/verifier/sr_worker.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        class_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "StandaloneRemoteWorker"
        )
        wanted = {
            "verify",
            "_can_discard_verify_logits",
            "_need_target_hidden",
            "forward_batch_generation",
        }
        nodes = [
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name in wanted
        ]
        self.assertEqual({node.name for node in nodes}, wanted)

        class GenerationBatchResult:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        ns = {
            "time": time,
            "nullcontext": nullcontext,
            "ForwardMode": _VerifyForwardMode,
            "GenerationBatchResult": GenerationBatchResult,
            "conservative_mode_reason": conservative_mode_reason,
            "bind_graph_host_metrics": lambda runner, metrics: None,
            "restore_graph_host_metrics": lambda token: None,
            "generate_token_bitmask": lambda *args, **kwargs: None,
            "maybe_detect_nan": lambda *args, **kwargs: None,
            "_snapshot_seq_lens_cpu": lambda batch: batch.seq_lens,
            "_sync_kv_from_cpu_lengths": lambda batch, seq_lens_cpu, lengths: None,
            "_is_health_check": lambda req: False,
            "add_output_logprobs_for_spec_v1": logprob_mod.add_output_logprobs_for_spec_v1,
        }
        module = ast.Module(
            body=[ast.parse("from __future__ import annotations").body[0], *nodes],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
        return ns

    def _blank_logits(self, next_token_logits, **overrides):
        fields = {
            "next_token_logits": next_token_logits,
            "hidden_states": None,
            "next_token_logprobs": None,
            "next_token_top_logprobs_val": None,
            "next_token_top_logprobs_idx": None,
            "next_token_token_ids_logprobs_val": None,
            "next_token_token_ids_logprobs_idx": None,
            "input_token_logprobs": None,
            "input_top_logprobs_val": None,
            "input_top_logprobs_idx": None,
            "input_token_ids_logprobs_val": None,
            "input_token_ids_logprobs_idx": None,
            "customized_info": None,
            "full_logits": None,
            "mm_input_embeds": None,
            "tree_seed_topk_p": None,
            "tree_seed_topk_index": None,
        }
        fields.update(overrides)
        return SimpleNamespace(**fields)

    def _eligible_batch(self, **overrides):
        batch = SimpleNamespace(
            forward_mode=_ForwardModeStub(),
            return_logprob=False,
            has_grammar=False,
            sampling_info=SimpleNamespace(
                is_all_greedy=True,
                has_custom_logit_processor=False,
            ),
        )
        for key, value in overrides.items():
            setattr(batch, key, value)
        return batch

    def test_admission_uses_real_mode_function_and_none_checks(self):
        if torch is None:
            self.skipTest("torch not available")
        ns = self._load()
        seen = []
        real = ns["conservative_mode_reason"]

        def wrapped(mode, is_all_greedy):
            seen.append((mode, is_all_greedy))
            return real(mode, is_all_greedy)

        ns["conservative_mode_reason"] = wrapped
        discard = ns["_can_discard_verify_logits"]
        worker = SimpleNamespace(
            server_args=SimpleNamespace(speculative_verify_mode="greedy")
        )
        logits = self._blank_logits(torch.zeros(2, 3))
        batch = self._eligible_batch()
        batch.sampling_info.is_all_greedy = False
        self.assertTrue(discard(worker, batch, logits, prepare_hidden=False))
        self.assertEqual(seen, [("greedy", False)])
        with self.assertRaises(TypeError):
            discard(worker, batch, logits, False)

        ns["conservative_mode_reason"] = real
        cases = [
            ("auto", True, True),
            ("auto", False, False),
            (None, True, True),
            ("", True, True),
            (None, False, False),
            ("target_only", True, False),
            ("rpd", True, False),
            ("other", True, False),
        ]
        for mode, greedy, expected in cases:
            worker.server_args.speculative_verify_mode = mode
            batch.sampling_info.is_all_greedy = greedy
            self.assertEqual(
                discard(worker, batch, logits, prepare_hidden=False),
                expected,
                msg=f"mode={mode!r} greedy={greedy}",
            )

        worker.server_args.speculative_verify_mode = "greedy"
        batch.sampling_info.is_all_greedy = True
        batch.return_logprob = True
        self.assertFalse(discard(worker, batch, logits, prepare_hidden=False))
        batch.return_logprob = False
        batch.has_grammar = True
        self.assertFalse(discard(worker, batch, logits, prepare_hidden=False))
        batch.has_grammar = False
        batch.sampling_info.has_custom_logit_processor = True
        self.assertFalse(discard(worker, batch, logits, prepare_hidden=False))
        batch.sampling_info.has_custom_logit_processor = False
        self.assertFalse(discard(worker, batch, logits, prepare_hidden=True))
        self.assertFalse(
            discard(
                worker,
                batch,
                self._blank_logits(logits.next_token_logits, hidden_states=torch.zeros(1)),
                prepare_hidden=False,
            )
        )
        self.assertFalse(
            discard(worker, self._eligible_batch(forward_mode=_ForwardModeStub(idle=True)), logits, prepare_hidden=False)
        )
        for name in (
            "next_token_logprobs",
            "next_token_top_logprobs_val",
            "next_token_top_logprobs_idx",
            "next_token_token_ids_logprobs_val",
            "next_token_token_ids_logprobs_idx",
            "input_token_logprobs",
            "input_top_logprobs_val",
            "input_top_logprobs_idx",
            "input_token_ids_logprobs_val",
            "input_token_ids_logprobs_idx",
            "full_logits",
            "mm_input_embeds",
            "tree_seed_topk_p",
            "tree_seed_topk_index",
        ):
            self.assertFalse(
                discard(
                    worker,
                    batch,
                    self._blank_logits(logits.next_token_logits, **{name: []}),
                    prepare_hidden=False,
                ),
                msg=name,
            )
        self.assertFalse(
            discard(
                worker,
                batch,
                self._blank_logits(logits.next_token_logits, customized_info={}),
                prepare_hidden=False,
            )
        )
        self.assertFalse(
            discard(
                worker,
                batch,
                self._blank_logits(logits.next_token_logits, customized_info=[]),
                prepare_hidden=False,
            )
        )

    def _run_verify(
        self,
        ns,
        *,
        logits_data,
        accepted_indices,
        accept_lengths,
        reqs,
        verify_mode="greedy",
        is_all_greedy=True,
        prepare_overrides=None,
        logits_overrides=None,
        return_logprob=False,
        has_grammar=False,
        has_custom=False,
        idle=False,
        metrics=None,
        fixed_state=None,
        hybrid=False,
        forbid_get=False,
        forbid_meta=False,
        hidden=None,
        mamba=None,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        temperatures=None,
        verified_id=None,
    ):
        probe = _LogitsProbe(
            logits_data, forbid_get=forbid_get, forbid_meta=forbid_meta
        )
        logits_output = self._blank_logits(probe, **(logits_overrides or {}))
        if hidden is not None:
            logits_output.hidden_states = hidden
        seen = {}
        seq_lens = torch.tensor(
            [4 + index for index in range(len(reqs))], dtype=torch.int64
        )
        if not reqs:
            seq_lens = torch.empty(0, dtype=torch.int64)

        def stub_verify(batch, logits, allocator, page_size, vocab_mask, **kwargs):
            seen["logits"] = logits.next_token_logits
            seen["prepare_hidden"] = kwargs["prepare_local_draft_hidden"]
            seen["state"] = kwargs["sr_accept_state"]
            if verified_id is None:
                ids = [
                    req.output_ids[-1] if req.output_ids else 0 for req in reqs
                ]
                verified = torch.tensor(ids, dtype=torch.int64)
            else:
                verified = verified_id
            return SimpleNamespace(
                accepted_indices=accepted_indices,
                accept_length_per_req_cpu=list(accept_lengths),
                verified_id=verified,
                draft_input="kept-draft",
            )

        spec_info = SimpleNamespace(
            draft_token_num=4,
            seq_lens_cpu=seq_lens,
            capture_hidden_mode="capture",
            hidden_states=None,
            prepare_for_verify=lambda batch, page_size: None,
            verify=stub_verify,
        )
        batch = SimpleNamespace(
            forward_mode=_ForwardModeStub(idle=idle),
            return_hidden_states=False,
            reqs=reqs,
            has_grammar=has_grammar,
            return_logprob=return_logprob,
            sampling_info=SimpleNamespace(
                is_all_greedy=is_all_greedy,
                has_custom_logit_processor=has_custom,
                vocab_size=int(logits_data.shape[-1]),
                temperatures=temperatures
                if temperatures is not None
                else torch.ones(max(len(reqs), 1)),
                device=torch.device("cpu"),
            ),
            seq_lens=seq_lens.clone(),
            seq_lens_cpu=seq_lens.clone(),
            sr_round_metrics=metrics,
            spec_info=None,
            top_logprobs_nums=top_logprobs_nums or [0 for _ in reqs],
            token_ids_logprobs=token_ids_logprobs or [None for _ in reqs],
            get_model_worker_batch=lambda seq_lens_cpu_cache=None: SimpleNamespace(
                capture_hidden_mode=spec_info.capture_hidden_mode
            ),
        )
        runner = SimpleNamespace(
            graph_runner=None,
            hybrid_gdn_config=object() if hybrid else None,
            mamba2_config=None,
            hybrid_lightning_config=None,
        )
        worker = SimpleNamespace(
            page_size=1,
            enable_nan_detection=False,
            _fixed_accept_state=fixed_state,
            _hybrid_needs_hidden=hybrid,
            server_args=SimpleNamespace(
                speculative_verify_mode=verify_mode,
                enable_return_hidden_states=False,
            ),
            target_worker=SimpleNamespace(
                model_runner=runner,
                forward_batch_generation=lambda model_batch, is_verify=False: SimpleNamespace(
                    logits_output=logits_output,
                    can_run_cuda_graph=False,
                ),
            ),
            token_to_kv_pool_allocator=SimpleNamespace(),
            _mamba_verify_update=mamba
            if mamba is not None
            else lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("mamba update ran")
            ),
        )
        if prepare_overrides:
            if "enable_return_hidden_states" in prepare_overrides:
                worker.server_args.enable_return_hidden_states = prepare_overrides[
                    "enable_return_hidden_states"
                ]
            if prepare_overrides.get("return_hidden_states"):
                batch.return_hidden_states = True
            if prepare_overrides.get("req_hidden"):
                for req in reqs:
                    req.return_hidden_states = True
        worker._need_target_hidden = lambda batch=None: ns["_need_target_hidden"](
            worker, batch
        )
        worker._can_discard_verify_logits = (
            lambda *args, **kwargs: ns["_can_discard_verify_logits"](
                worker, *args, **kwargs
            )
        )
        output_before = [list(req.output_ids) for req in reqs]
        finish_before = [req.finished_reason for req in reqs]
        seq_before = batch.seq_lens.clone()
        logits_out, res, _, _ = ns["verify"](worker, batch, spec_info)
        self.assertEqual([list(req.output_ids) for req in reqs], output_before)
        self.assertEqual([req.finished_reason for req in reqs], finish_before)
        self.assertTrue(torch.equal(batch.seq_lens, seq_before))
        self.assertEqual(res.accept_length_per_req_cpu, list(accept_lengths))
        self.assertIs(res.draft_input, "kept-draft")
        self.assertIs(batch.spec_info, res.draft_input)
        return logits_out, res, batch, seen, probe

    def _req(self, output_ids, finished_reason=None):
        return SimpleNamespace(
            output_ids=list(output_ids),
            finished_reason=finished_reason,
            return_hidden_states=False,
            return_logprob=False,
            top_logprobs_num=0,
            token_ids_logprob=None,
            spec_cnt=0,
            len_output_ids=None,
            sr_step_id=0,
            output_token_logprobs_val=[],
            output_token_logprobs_idx=[],
            output_top_logprobs_val=[],
            output_top_logprobs_idx=[],
            output_token_ids_logprobs_val=[],
            output_token_ids_logprobs_idx=[],
        )

    def test_fast_path_leaves_tokens_and_skips_gather(self):
        if torch is None:
            self.skipTest("torch not available")
        ns = self._load()
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            SRRoundMetrics,
        )

        data = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        cases = [
            (
                "partial",
                [self._req([7, 8], None), self._req([9], "stop")],
                torch.tensor([1, 0, 3]),
                [2, 0],
                None,
            ),
            (
                "all-finished",
                [self._req([1], "length"), self._req([2], "stop")],
                torch.tensor([0, 2]),
                [0, 0],
                None,
            ),
            (
                "bonus-only",
                [self._req([4]), self._req([5])],
                torch.tensor([2, 3]),
                [0, 0],
                None,
            ),
            ("empty", [], torch.empty(0, dtype=torch.int64), [], None),
            (
                "fixed-accept",
                [self._req([7], None), self._req([8], "stop")],
                torch.tensor([0, 1]),
                [1, 0],
                SimpleNamespace(metrics=None),
            ),
        ]
        for name, reqs, indices, lengths, fixed_state in cases:
            metrics = SRRoundMetrics("Target")
            logits_out, _, _, seen, probe = self._run_verify(
                ns,
                logits_data=data,
                accepted_indices=indices,
                accept_lengths=lengths,
                reqs=reqs,
                metrics=metrics,
                fixed_state=fixed_state,
                forbid_get=True,
            )
            self.assertIsNone(logits_out.next_token_logits, msg=name)
            self.assertIs(seen["logits"], probe)
            self.assertEqual(probe.gets, [], msg=name)
            self.assertEqual(seen["state"], fixed_state, msg=name)
            self.assertFalse(seen["prepare_hidden"], msg=name)
            self.assertEqual(metrics.counts["verify_logits_discard_batches"], 1, msg=name)
            self.assertEqual(metrics.counts["verify_logits_gather_batches"], 0, msg=name)
            expected_bytes = int(indices.numel()) * int(data.shape[-1]) * data.element_size()
            self.assertEqual(
                metrics.counts["verify_logits_skipped_output_bytes"],
                expected_bytes,
                msg=name,
            )

    def test_unused_replay_hidden_still_discards_logits(self):
        if torch is None:
            self.skipTest("torch not available")
        ns = self._load()
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            SRRoundMetrics,
        )

        class UnusedHidden:
            def __getitem__(self, index):
                raise AssertionError("unused hidden was indexed")

        data = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        indices = torch.tensor([1, 0, 3])
        metrics = SRRoundMetrics("Target")
        logits_out, _, _, seen, probe = self._run_verify(
            ns,
            logits_data=data,
            accepted_indices=indices,
            accept_lengths=[2, 0],
            reqs=[self._req([7, 8], None), self._req([9], "stop")],
            metrics=metrics,
            forbid_get=True,
            hidden=UnusedHidden(),
        )
        self.assertFalse(seen["prepare_hidden"])
        self.assertIsNone(logits_out.next_token_logits)
        self.assertIsNone(logits_out.hidden_states)
        self.assertEqual(probe.gets, [])
        self.assertEqual(metrics.counts["verify_logits_discard_batches"], 1)
        self.assertEqual(metrics.counts["verify_logits_gather_batches"], 0)
        self.assertEqual(
            metrics.counts["verify_logits_skipped_output_bytes"],
            int(indices.numel()) * int(data.shape[-1]) * data.element_size(),
        )

    def test_metrics_off_does_not_read_logits_metadata(self):
        if torch is None:
            self.skipTest("torch not available")
        ns = self._load()
        data = torch.zeros(2, 4)
        logits_out, _, _, _, probe = self._run_verify(
            ns,
            logits_data=data,
            accepted_indices=torch.tensor([0]),
            accept_lengths=[0],
            reqs=[self._req([3])],
            metrics=None,
            forbid_get=True,
            forbid_meta=True,
        )
        self.assertIsNone(logits_out.next_token_logits)
        self.assertEqual(probe.gets, [])

    def test_saved_hidden_decision_keeps_gather(self):
        if torch is None:
            self.skipTest("torch not available")
        ns = self._load()
        data = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        indices = torch.tensor([3, 1])
        hidden = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        order = []

        class HiddenProbe:
            def __getitem__(self, index):
                order.append("hidden")
                return hidden[index]

        def mamba(batch, res, logits_output, spec_info, seq_lens_pre):
            order.append("mamba")
            self.assertIsInstance(logits_output.next_token_logits, torch.Tensor)

        logits_out, _, batch, seen, probe = self._run_verify(
            ns,
            logits_data=data,
            accepted_indices=indices,
            accept_lengths=[1],
            reqs=[self._req([6])],
            hybrid=True,
            hidden=HiddenProbe(),
            mamba=mamba,
        )
        self.assertTrue(seen["prepare_hidden"])
        self.assertFalse(batch.return_hidden_states)
        self.assertTrue(torch.equal(logits_out.next_token_logits, data[indices]))
        self.assertTrue(torch.equal(logits_out.hidden_states, hidden[indices]))
        self.assertEqual(order, ["hidden", "mamba"])
        self.assertEqual(len(probe.gets), 1)

        for label, overrides in (
            ("server", {"enable_return_hidden_states": True}),
            ("batch", {"return_hidden_states": True}),
            ("req", {"req_hidden": True}),
        ):
            logits_out, _, batch, seen, _ = self._run_verify(
                ns,
                logits_data=data,
                accepted_indices=indices,
                accept_lengths=[0],
                reqs=[self._req([6])],
                prepare_overrides=overrides,
                mamba=lambda *args, **kwargs: None,
            )
            self.assertTrue(seen["prepare_hidden"], msg=label)
            self.assertFalse(batch.return_hidden_states, msg=label)
            self.assertTrue(
                torch.equal(logits_out.next_token_logits, data[indices]), msg=label
            )

    def test_keep_path_aligns_real_logprobs(self):
        if torch is None:
            self.skipTest("torch not available")
        ns = self._load()
        data = torch.tensor(
            [
                [0.0, 8.0, 0.0],
                [1.0, 1.0, 1.0],
                [9.0, 0.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        )
        base = torch.tensor([2, 9, 0, 9])
        indices = base[::2]
        self.assertFalse(indices.is_contiguous())
        req = self._req([4, 5])
        req.return_logprob = True
        req.top_logprobs_num = 1
        req.token_ids_logprob = [0]
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            SRRoundMetrics,
        )

        metrics = SRRoundMetrics("Target")
        logits_out, _, _, _, probe = self._run_verify(
            ns,
            logits_data=data,
            accepted_indices=indices,
            accept_lengths=[1],
            reqs=[req],
            verify_mode="auto",
            is_all_greedy=False,
            return_logprob=True,
            metrics=metrics,
            top_logprobs_nums=[1],
            token_ids_logprobs=[[0]],
            temperatures=torch.ones(1, 1),
            verified_id=torch.tensor([0, 1]),
        )
        self.assertTrue(torch.equal(logits_out.next_token_logits, data[indices]))
        self.assertEqual(len(probe.gets), 1)
        self.assertEqual(req.output_token_logprobs_idx, [0, 1])
        self.assertEqual(len(req.output_token_logprobs_val), 2)
        self.assertEqual(req.output_top_logprobs_idx[0][0], 0)
        self.assertEqual(req.output_top_logprobs_idx[1][0], 1)
        self.assertEqual(req.output_token_ids_logprobs_idx, [[0], [0]])
        self.assertEqual(len(req.output_token_ids_logprobs_val), 2)
        self.assertEqual(metrics.counts["verify_logits_gather_batches"], 1)
        self.assertEqual(metrics.counts["verify_logits_discard_batches"], 0)
        self.assertEqual(metrics.counts["verify_logits_skipped_output_bytes"], 0)

    def test_replay_clear_does_not_change_persistent_buffer(self):
        if torch is None:
            self.skipTest("torch not available")
        ns = self._load()
        persistent = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        original = persistent.clone()
        wrappers = []

        def forward(model_batch, is_verify=False):
            wrapper = self._blank_logits(persistent)
            wrappers.append(wrapper)
            return SimpleNamespace(logits_output=wrapper, can_run_cuda_graph=True)

        indices = torch.tensor([1, 2])
        for _ in range(2):
            req = self._req([1])
            seen = {}

            def stub_verify(batch, logits, allocator, page_size, vocab_mask, **kwargs):
                seen["logits"] = logits.next_token_logits
                return SimpleNamespace(
                    accepted_indices=indices,
                    accept_length_per_req_cpu=[1],
                    verified_id=torch.tensor([1]),
                    draft_input="draft",
                )

            spec_info = SimpleNamespace(
                draft_token_num=4,
                seq_lens_cpu=torch.tensor([3]),
                capture_hidden_mode="capture",
                prepare_for_verify=lambda batch, page_size: None,
                verify=stub_verify,
            )
            batch = SimpleNamespace(
                forward_mode=_ForwardModeStub(),
                return_hidden_states=False,
                reqs=[req],
                has_grammar=False,
                return_logprob=False,
                sampling_info=SimpleNamespace(
                    is_all_greedy=True,
                    has_custom_logit_processor=False,
                ),
                seq_lens=torch.tensor([3]),
                sr_round_metrics=None,
                top_logprobs_nums=[0],
                token_ids_logprobs=[None],
                get_model_worker_batch=lambda seq_lens_cpu_cache=None: SimpleNamespace(
                    capture_hidden_mode="capture"
                ),
            )
            worker = SimpleNamespace(
                page_size=1,
                enable_nan_detection=False,
                _fixed_accept_state=None,
                _hybrid_needs_hidden=False,
                server_args=SimpleNamespace(
                    speculative_verify_mode="greedy",
                    enable_return_hidden_states=False,
                ),
                target_worker=SimpleNamespace(
                    model_runner=SimpleNamespace(
                        graph_runner=None,
                        hybrid_gdn_config=None,
                        mamba2_config=None,
                        hybrid_lightning_config=None,
                    ),
                    forward_batch_generation=forward,
                ),
                token_to_kv_pool_allocator=SimpleNamespace(),
                _mamba_verify_update=lambda *args, **kwargs: None,
            )
            worker._need_target_hidden = lambda batch=None: ns["_need_target_hidden"](
                worker, batch
            )
            worker._can_discard_verify_logits = (
                lambda *args, **kwargs: ns["_can_discard_verify_logits"](
                    worker, *args, **kwargs
                )
            )
            ns["verify"](worker, batch, spec_info)
            self.assertIs(seen["logits"], persistent)

        self.assertIsNone(wrappers[0].next_token_logits)
        self.assertIsNone(wrappers[1].next_token_logits)
        self.assertTrue(torch.equal(persistent, original))

    def test_prefill_ar_and_idle_do_not_discard(self):
        if torch is None:
            self.skipTest("torch not available")
        ns = self._load()
        calls = []
        worker = SimpleNamespace(
            speculative_num_draft_tokens=4,
            speculative_num_steps=2,
            forward_target_extend=lambda batch: ("extend-logits", torch.tensor([3]), None),
            _forward_normal_decode=lambda batch: calls.append("ar") or "ar-result",
            verify=lambda *args, **kwargs: calls.append("verify"),
            construct_draft_input=lambda *args, **kwargs: calls.append("tree"),
        )
        extend_batch = SimpleNamespace(
            forward_mode=_ForwardModeStub(extend=True),
            is_extend_in_batch=False,
        )
        result = ns["forward_batch_generation"](worker, extend_batch)
        self.assertEqual(calls, [])
        self.assertEqual(result.logits_output, "extend-logits")

        ar_batch = SimpleNamespace(
            forward_mode=_ForwardModeStub(),
            is_extend_in_batch=False,
            draft_num_tokens=1,
        )
        self.assertEqual(ns["forward_batch_generation"](worker, ar_batch), "ar-result")
        self.assertEqual(calls, ["ar"])

        data = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        indices = torch.tensor([2, 0])
        logits_out, _, batch, _, probe = self._run_verify(
            ns,
            logits_data=data,
            accepted_indices=indices,
            accept_lengths=[1],
            reqs=[self._req([8])],
            idle=True,
        )
        self.assertTrue(batch.forward_mode.is_idle())
        self.assertTrue(torch.equal(logits_out.next_token_logits, data[indices]))
        self.assertEqual(len(probe.gets), 1)

    def _load_methods(self, relative_path, class_name, names, extra_ns):
        import ast
        from pathlib import Path

        path = Path(__file__).resolve().parents[4] / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"))
        class_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        nodes = [
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        self.assertEqual({node.name for node in nodes}, set(names))
        ns = dict(extra_ns)
        module = ast.Module(
            body=[ast.parse("from __future__ import annotations").body[0], *nodes],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
        return ns

    def test_decode_spec_v1_consumes_none_logits(self):
        if torch is None:
            self.skipTest("torch not available")
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        class ReqStub:
            def __init__(self, output_ids, finished, grammar=None):
                self.output_ids = list(output_ids)
                self.origin_input_ids = [1, 2]
                self._finished = finished
                self.grammar = grammar
                self.return_hidden_states = False
                self.mamba_ping_pong_track_buffer = None
                self.multimodal_inputs = None
                self.session = None
                self.time_stats = SimpleNamespace(
                    set_last_decode_finish_time=lambda ts=None: events.append("decode"),
                    set_completion_time=lambda ts=None: events.append("complete"),
                )
                self.routed_experts = "unset"
                self.req_pool_idx = 0
                self.seqlen = 4

            def finished(self):
                return self._finished

        events = []
        released = []
        sent = []
        stats = []
        grammar = SimpleNamespace(finished=False)
        running = ReqStub([7, 8], False)
        finished = ReqStub([9], True, grammar)
        batch = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.STANDALONE_REMOTE,
            is_spec_v2=False,
            draft_num_tokens=15,
            reqs=[running, finished],
            return_logprob=False,
            batch_size=lambda: 2,
            dp_cooperation_info=None,
        )

        class TokenBoom:
            def tolist(self):
                raise AssertionError("spec v1 read next_token_ids")

        result = SimpleNamespace(
            copy_done=None,
            logits_output=self._blank_logits(None),
            next_token_ids=TokenBoom(),
            num_accepted_tokens=3,
            accept_length_per_req_cpu=[2, 0],
            can_run_cuda_graph=True,
        )
        scheduler = SimpleNamespace(
            is_remote_spec_draft=False,
            enable_metrics=False,
            enable_overlap=False,
            enable_hisparse=False,
            num_generated_tokens=0,
            spec_num_accepted_tokens=0,
            spec_num_forward_ct=0,
            spec_num_draft_tokens=0,
            token_to_kv_pool_allocator=SimpleNamespace(
                free_group_begin=lambda: events.append("free-begin"),
                free_group_end=lambda: events.append("free-end"),
            ),
            tree_cache=SimpleNamespace(),
            server_args=SimpleNamespace(
                disaggregation_decode_enable_offload_kvcache=False,
                decode_log_interval=50,
            ),
            current_scheduler_metrics_enabled=True,
            enable_mfu_metrics=False,
            scheduler_status_logger=None,
            metrics_collector=SimpleNamespace(
                increment_realtime_tokens=lambda **kwargs: stats.append(kwargs)
            ),
            forward_ct_decode=0,
            req_to_token_pool=SimpleNamespace(),
            stream_output=lambda reqs, return_logprob: sent.append((reqs, return_logprob)),
            maybe_notify_remote_draft_finished=lambda req: events.append("notify"),
        )
        class _Capturer:
            def get_routed_experts(self, req_pool_idx, seqlen, req_to_token_pool):
                return None

        output_ns = self._load_methods(
            "python/sglang/srt/managers/scheduler_output_processor_mixin.py",
            "SchedulerOutputProcessorMixin",
            {
                "process_batch_result_decode",
                "_mamba_prefix_cache_update",
                "_handle_finished_req",
                "_is_spectre_draft_ar",
                "maybe_collect_customized_info",
                "maybe_collect_routed_experts",
            },
            {
                "release_kv_cache": lambda req, tree_cache, is_insert=True: released.append(
                    req
                ),
                "get_global_experts_capturer": lambda: _Capturer(),
            },
        )
        metrics_ns = self._load_methods(
            "python/sglang/srt/observability/scheduler_metrics_mixin.py",
            "SchedulerMetricsMixin",
            {"update_spec_metrics", "report_decode_stats"},
            {},
        )
        for name, fn in {**output_ns, **metrics_ns}.items():
            if name in {
                "process_batch_result_decode",
                "_mamba_prefix_cache_update",
                "_handle_finished_req",
                "_is_spectre_draft_ar",
                "maybe_collect_customized_info",
                "maybe_collect_routed_experts",
                "update_spec_metrics",
                "report_decode_stats",
            }:
                setattr(scheduler, name, fn.__get__(scheduler))
        scheduler.process_batch_result_decode(batch, result)

        self.assertEqual(running.output_ids, [7, 8])
        self.assertEqual(finished.output_ids, [9])
        self.assertIs(result.logits_output.next_token_logits, None)
        self.assertTrue(grammar.finished)
        self.assertEqual(released, [finished])
        self.assertIsNone(finished.routed_experts)
        self.assertEqual(running.routed_experts, "unset")
        self.assertEqual(sent, [([running, finished], False)])
        self.assertIn("free-begin", events)
        self.assertIn("free-end", events)
        self.assertIn("complete", events)
        self.assertEqual(scheduler.spec_num_accepted_tokens, 5)
        self.assertEqual(scheduler.spec_num_forward_ct, 2)
        self.assertEqual(scheduler.spec_num_draft_tokens, 30)
        self.assertEqual(scheduler.num_generated_tokens, 5)
        self.assertEqual(stats, [{"decode_tokens": 5, "dp_cooperation_info": None}])


if __name__ == "__main__":
    unittest.main()
