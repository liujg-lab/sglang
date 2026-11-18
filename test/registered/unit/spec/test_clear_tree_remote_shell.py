"""CPU checks for the CLEAR_TREE_REMOTE shell.

The shell is a renamed copy of fixed-width OPT. It must not be recognized as
OPT or STANDALONE_REMOTE, and OPT must not pick up ClearTree fields.
"""

import ast
import os
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock, call, patch

import zmq

from sglang.srt.speculative.clear_tree_remote import clear_tree_transport as transport
from sglang.srt.speculative.clear_tree_remote.clear_tree_args import (
    apply_clear_tree_settings,
)
from sglang.srt.speculative.clear_tree_remote.clear_tree_protocol import (
    ClearAction,
    ClearBatchRequest,
    ClearDraftRequest,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm, remote_spec_role
from sglang.srt.speculative.standalone_remote.sr_protocol import local_spec_shape
from sglang.srt.speculative.standalone_remote.verifier.sr_fixed_accept import (
    static_disable_reason,
)
from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


def _args(**overrides):
    values = dict(
        speculative_algorithm="CLEAR_TREE_REMOTE",
        clear_tree_role="target",
        clear_tree_width=4,
        clear_tree_tau=0.5,
        clear_tree_max_depth=8,
        clear_tree_addr="127.0.0.1",
        clear_tree_port="30019",
        opt_tree_role=None,
        opt_tree_width=None,
        opt_tree_max_depth=None,
        standalone_remote_role=None,
        speculative_num_draft_tokens=None,
        speculative_num_steps=None,
        speculative_verify_mode=None,
        speculative_eagle_topk=None,
        max_running_requests=1,
        page_size=1,
        disable_overlap_schedule=False,
        enable_mixed_chunk=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class ClearTreeRemoteShellTest(CustomTestCase):
    def test_algorithm_is_distinct(self):
        clear = SpeculativeAlgorithm.from_string("CLEAR_TREE_REMOTE")
        opt = SpeculativeAlgorithm.from_string("OPT_TREE_REMOTE")
        sr = SpeculativeAlgorithm.from_string("STANDALONE_REMOTE")

        self.assertTrue(clear.is_clear_tree_remote())
        self.assertFalse(clear.is_opt_tree_remote())
        self.assertFalse(clear.is_standalone_remote())
        self.assertTrue(opt.is_opt_tree_remote())
        self.assertFalse(opt.is_clear_tree_remote())
        self.assertFalse(sr.is_clear_tree_remote())

    def test_role_and_draft_hooks(self):
        clear = SpeculativeAlgorithm.from_string("CLEAR_TREE_REMOTE")
        draft = _args(clear_tree_role="draft")
        target = _args(clear_tree_role="target")

        self.assertTrue(clear.skips_colocated_draft_worker(draft))
        self.assertFalse(clear.skips_colocated_draft_worker(target))
        self.assertTrue(clear.uses_draft_paused_reqs(draft))
        self.assertTrue(clear.uses_ordinary_decode_prepare(draft))
        self.assertFalse(clear.uses_draft_paused_reqs(target))
        self.assertEqual(clear.guess_tokens_per_step(draft), 4)
        self.assertTrue(clear.reports_remote_spec_metrics())
        self.assertEqual(remote_spec_role(draft), "draft")

        sr_wins = _args(standalone_remote_role="target", clear_tree_role="draft")
        self.assertEqual(remote_spec_role(sr_wins), "target")
        opt_wins = _args(opt_tree_role="draft", clear_tree_role="target")
        self.assertEqual(remote_spec_role(opt_wins), "draft")

    def test_omitted_draft_tokens_follow_role(self):
        target = _args(clear_tree_role="target")
        apply_clear_tree_settings(target, on_npu=False)
        self.assertEqual(target.speculative_num_draft_tokens, 65)
        self.assertEqual(target.clear_tree_widths, [2, 4, 8, 16, 32, 64])
        self.assertEqual(target.clear_tree_ks, [2, 4, 8, 16, 32, 64])
        self.assertEqual(target.speculative_num_steps, 8)
        self.assertTrue(target.disable_overlap_schedule)
        self.assertFalse(target.enable_mixed_chunk)

        draft = _args(clear_tree_role="draft")
        apply_clear_tree_settings(draft, on_npu=False)
        self.assertEqual(draft.speculative_num_draft_tokens, 64)

    def test_capture_widths_follow_role(self):
        from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
            bonus_ntpbs,
            install_candidates,
        )

        install_candidates(None, None)
        target = _args(clear_tree_role="target", speculative_num_draft_tokens=65)
        draft = _args(clear_tree_role="draft", speculative_num_draft_tokens=64)
        opt = _args(
            speculative_algorithm="OPT_TREE_REMOTE",
            opt_tree_role="target",
            speculative_num_draft_tokens=17,
        )
        self.assertEqual(
            SpeculativeAlgorithm.CLEAR_TREE_REMOTE.dual_ntpb_cuda_graph_options(target),
            [65, 33, 17, 9, 5, 3, 1],
        )
        self.assertEqual(
            SpeculativeAlgorithm.CLEAR_TREE_REMOTE.dual_ntpb_cuda_graph_options(draft),
            [64, 32, 16, 8, 4, 2, 1],
        )
        self.assertEqual(bonus_ntpbs(), (65, 33, 17, 9, 5, 3))
        self.assertEqual(
            SpeculativeAlgorithm.OPT_TREE_REMOTE.dual_ntpb_cuda_graph_options(opt),
            [17, 1],
        )

    def test_custom_candidates_set_capacity_and_capture(self):
        from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
            bonus_ntpbs,
            install_candidates,
        )

        target = _args(
            clear_tree_role="target",
            clear_tree_widths=[8, 4, 4],
            clear_tree_ks=[16, 8],
        )
        draft = _args(
            clear_tree_role="draft",
            clear_tree_widths=[8, 4],
            clear_tree_ks=[16, 8],
        )
        try:
            apply_clear_tree_settings(target, on_npu=False)
            self.assertEqual(target.clear_tree_widths, [4, 8])
            self.assertEqual(target.clear_tree_ks, [8, 16])
            self.assertEqual(target.speculative_num_draft_tokens, 17)
            self.assertEqual(
                SpeculativeAlgorithm.CLEAR_TREE_REMOTE.dual_ntpb_cuda_graph_options(
                    target
                ),
                [17, 9, 1],
            )
            apply_clear_tree_settings(draft, on_npu=False)
            self.assertEqual(draft.speculative_num_draft_tokens, 8)
            self.assertEqual(
                SpeculativeAlgorithm.CLEAR_TREE_REMOTE.dual_ntpb_cuda_graph_options(
                    draft
                ),
                [8, 4, 1],
            )
            self.assertEqual(bonus_ntpbs(), (9, 5))
            rejected = _args(clear_tree_role="draft", clear_tree_widths=[1, 4])
            with self.assertRaises(ValueError):
                apply_clear_tree_settings(rejected, on_npu=False)
        finally:
            install_candidates(None, None)

    def test_flat_table_tiers_are_required(self):
        from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
            install_candidates,
        )

        draft = _args(
            clear_tree_role="draft",
            clear_tree_widths=[16, 32],
            clear_tree_ks=[32, 64],
        )
        try:
            with self.assertRaises(ValueError):
                apply_clear_tree_settings(draft, on_npu=False)
        finally:
            install_candidates(None, None)

    def test_explicit_draft_tokens_must_match(self):
        draft = _args(clear_tree_role="draft", speculative_num_draft_tokens=5)
        with self.assertRaises(ValueError):
            apply_clear_tree_settings(draft, on_npu=False)

    def test_wire_shape_uses_remote_spec_shape(self):
        clear = _args(
            speculative_eagle_topk=3,
            opt_tree_width=7,
            opt_tree_max_depth=3,
        )
        apply_clear_tree_settings(clear, on_npu=False)
        clear.speculative_num_draft_tokens = 99
        clear.speculative_num_steps = 2
        self.assertEqual(clear.remote_spec_shape, (4, 8, 4))
        self.assertEqual(local_spec_shape(clear), (4, 8, 4))

        bare = _args(
            speculative_num_draft_tokens=99,
            speculative_num_steps=2,
            speculative_eagle_topk=3,
        )
        self.assertEqual(local_spec_shape(bare), (3, 2, 99))

        opt = _args(
            speculative_algorithm="OPT_TREE_REMOTE",
            opt_tree_width=7,
            opt_tree_max_depth=3,
            clear_tree_width=4,
            clear_tree_max_depth=8,
        )
        self.assertEqual(local_spec_shape(opt), (7, 3, 7))

        import sglang.srt.speculative.standalone_remote.sr_protocol as proto
        from pathlib import Path

        source = Path(proto.__file__).read_text(encoding="utf-8").lower()
        self.assertNotIn("clear_tree", source)

    def test_non_clear_send_uses_super(self):
        import ast
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/speculative/clear_tree_remote/clear_tree_scheduler.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        mixin = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ClearTreeSchedulerMixin"
        )
        send = next(
            node
            for node in mixin.body
            if isinstance(node, ast.FunctionDef) and node.name == "_sr_send"
        )
        gate = next(node for node in send.body if isinstance(node, ast.If))
        self.assertEqual(gate.test.func.attr, "_clear_step_rtt_enabled")
        self.assertIn(
            "_clear_send", ast.dump(ast.Module(body=gate.body, type_ignores=[]))
        )
        self.assertNotIn(
            "_clear_send", ast.dump(ast.Module(body=gate.orelse, type_ignores=[]))
        )
        orelse = ast.dump(ast.Module(body=gate.orelse, type_ignores=[]))
        self.assertIn("_sr_send", orelse)
        self.assertIn("super", orelse)
        enabled = next(
            node
            for node in mixin.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_clear_step_rtt_enabled"
        )
        enabled_src = ast.get_source_segment(source, enabled)
        self.assertIn("is_clear_tree_remote", enabled_src)

    def test_draft_loop_post_runs_only_after_valid_reply_and_drain(self):
        import ast
        from collections import defaultdict
        from contextlib import nullcontext
        from pathlib import Path

        from sglang.srt.speculative.clear_tree_remote import clear_tree_ledger as ledger
        from sglang.srt.speculative.clear_tree_remote import clear_tree_replay as replay
        from sglang.srt.speculative.clear_tree_remote.clear_tree_protocol import (
            supported_clear_protocol,
        )

        source = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/speculative/clear_tree_remote/clear_tree_scheduler.py"
        ).read_text()
        mixin = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.ClassDef) and node.name == "ClearTreeSchedulerMixin"
        )
        method = next(
            node
            for node in mixin.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "event_loop_normal_clear_tree_draft"
        )
        method.decorator_list = []
        for status, send_error, identity, handler_error, reason in (
            ("ok", False, b"peer", False, None),
            ("empty", False, b"peer", False, None),
            ("empty", False, b"peer", False, "commit_incomplete"),
            ("reject", False, b"peer", False, None),
            ("ok", True, b"peer", False, None),
            ("ok", False, None, False, None),
            ("ok", False, b"peer", True, None),
        ):
            with self.subTest(
                status=status,
                send_error=send_error,
                identity=identity,
                handler_error=handler_error,
            ):
                events = []
                batch = SimpleNamespace(
                    action=ClearAction.STEP,
                    session_id="sess",
                    rpc_seq=1,
                    protocol_version=1,
                    reqs=[SimpleNamespace(rid="r")],
                )
                reply = SimpleNamespace(
                    session_id="sess",
                    rpc_seq=1,
                    protocol_version=1,
                    reqs=[SimpleNamespace(status=status, reason=reason)],
                )

                def send(_reply):
                    events.append("send")
                    if send_error:
                        raise RuntimeError("send failed")

                def handle(*_args):
                    if handler_error:
                        raise RuntimeError("handler failed")
                    return reply

                metrics = SimpleNamespace(
                    active=False, phase=lambda _name: nullcontext()
                )
                scheduler = SimpleNamespace(
                    tp_rank=0,
                    tp_size=1,
                    _opt_deferred=[],
                    sr_state=SimpleNamespace(session_id="sess"),
                    sr_server=SimpleNamespace(send_batch=send, _last_identity=identity),
                    _init_clear_tree_draft=lambda: None,
                    recv_requests=Mock(side_effect=[[], StopIteration()]),
                    process_input_requests=lambda _reqs: None,
                    _sr_recv_packet=lambda: (batch, {}),
                    _sr_batch_is_stale=lambda _batch: False,
                    _clear_spec_shape_rejected=lambda _batch: False,
                    _clear_handle_batch_impl=handle,
                    _sr_resume_http_reqs=lambda: None,
                    _sr_poll_deferred_finishes=lambda: None,
                )
                namespace = {
                    "time": __import__("time"),
                    "sys": __import__("sys"),
                    "ClearAction": ClearAction,
                    "get_sr_round_metrics": lambda *_args: metrics,
                    "supported_clear_protocol": supported_clear_protocol,
                    "run_opt_draft_round": lambda _m, _measure, work: work(
                        defaultdict(int)
                    ),
                    "dump_tree_reply": lambda _reply: None,
                    "drain_deferred_layer_events": lambda *_a, **_kw: events.append(
                        "drain"
                    ),
                }
                exec(
                    compile(
                        ast.Module(body=[method], type_ignores=[]),
                        "<draft-loop>",
                        "exec",
                    ),
                    namespace,
                )
                with patch.object(ledger, "note_packet"), patch.object(
                    ledger, "mark_send"
                ), patch.object(
                    ledger, "mark_post", side_effect=lambda *_a: events.append("mark")
                ), patch.object(
                    replay,
                    "run_post_reply",
                    side_effect=lambda *_a: events.append("post"),
                ):
                    expected = (
                        RuntimeError if send_error or handler_error else StopIteration
                    )
                    with self.assertRaises(expected):
                        namespace[method.name](scheduler)
                self.assertIn("drain", events)
                should_post = (
                    status in {"ok", "empty"}
                    and reason != "commit_incomplete"
                    and not send_error
                    and identity is not None
                    and not handler_error
                )
                self.assertEqual("post" in events, should_post)
                if should_post:
                    self.assertEqual(events, ["send", "drain", "post", "mark"])

    def test_recv_packet_passes_local_queue_flag_before_tp1_return(self):
        import ast
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/speculative/clear_tree_remote/clear_tree_scheduler.py"
        ).read_text()
        mixin = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.ClassDef) and node.name == "ClearTreeSchedulerMixin"
        )
        method = next(
            node
            for node in mixin.body
            if isinstance(node, ast.FunctionDef) and node.name == "_sr_recv_packet"
        )
        namespace = {}
        exec(
            compile(
                ast.Module(body=[method], type_ignores=[]), "<recv-packet>", "exec"
            ),
            namespace,
        )
        for queued in (False, True):
            with self.subTest(queued=queued):
                packet = (object(), {})
                scheduler = SimpleNamespace(
                    spec_algorithm=SimpleNamespace(is_clear_tree_remote=lambda: True),
                    tp_size=1,
                    tp_rank=0,
                    sr_server=SimpleNamespace(
                        recv_batch=Mock(return_value=packet), last_recv_queued=queued
                    ),
                )
                self.assertIs(namespace[method.name](scheduler), packet)
                self.assertEqual(scheduler._clear_recv_queued, queued)
                scheduler.sr_server.recv_batch.assert_called_once_with(timeout_ms=50)

    def test_fixed_accept_stays_off(self):
        clear = SimpleNamespace(fixed_accept_skip="clear_tree")
        named = SimpleNamespace(speculative_algorithm="CLEAR_TREE_REMOTE")
        opt = SimpleNamespace(speculative_algorithm="OPT_TREE_REMOTE")
        self.assertEqual(static_disable_reason(clear), "clear_tree")
        self.assertIsNone(getattr(named, "fixed_accept_skip", None))
        self.assertNotEqual(static_disable_reason(named), "clear_tree")
        self.assertEqual(static_disable_reason(opt), "opt_tree")


class _DraftSocket:
    def __init__(self, *responses):
        self.recv_multipart = Mock(side_effect=responses)
        self.setsockopt = Mock()


class ClearTreeDraftTransportTest(CustomTestCase):
    def _server(self, *responses):
        # Exercise recv_batch without binding a real ROUTER or opening a context.
        server = transport.ClearDraftServer.__new__(transport.ClearDraftServer)
        server._socket = _DraftSocket(*responses)
        server._comm_pending = None
        server.comm_metrics = Mock()
        server._last_identity = None
        server.last_recv_queued = False
        return server

    def _frames(self):
        batch = ClearBatchRequest(
            session_id="session",
            rpc_seq=3,
            action=ClearAction.STEP,
            reqs=[ClearDraftRequest(rid="r", committed_ids=[1, 2])],
        )
        return batch, [b"target", *transport._pack_request(batch)]

    def test_immediate_request_is_queued(self):
        batch, frames = self._frames()
        server = self._server(frames)

        result = server.recv_batch(timeout_ms=17)

        self.assertEqual(result, (batch, {}))
        self.assertTrue(server.last_recv_queued)
        server._socket.recv_multipart.assert_called_once_with(
            flags=zmq.NOBLOCK, copy=False
        )
        server._socket.setsockopt.assert_not_called()
        self.assertEqual(server._last_identity, b"target")
        self.assertEqual(server._comm_pending.key, ("session", 3))

    def test_again_then_request_is_not_queued(self):
        batch, frames = self._frames()
        for timeout, expected_timeout in ((None, -1), (17, 17), (0, 1)):
            with self.subTest(timeout=timeout):
                server = self._server(zmq.Again(), frames)
                server.last_recv_queued = True

                self.assertEqual(server.recv_batch(timeout), (batch, {}))

                self.assertFalse(server.last_recv_queued)
                self.assertEqual(
                    server._socket.recv_multipart.call_args_list,
                    [
                        call(flags=zmq.NOBLOCK, copy=False),
                        call(copy=False),
                    ],
                )
                server._socket.setsockopt.assert_called_once_with(
                    zmq.RCVTIMEO, expected_timeout
                )

    def test_full_timeout_clears_previous_queued_result(self):
        _, frames = self._frames()
        server = self._server(frames, zmq.Again(), zmq.Again())
        self.assertIsNotNone(server.recv_batch(17))
        self.assertTrue(server.last_recv_queued)

        self.assertIsNone(server.recv_batch(17))

        self.assertFalse(server.last_recv_queued)
        self.assertIsNone(server._comm_pending)
        server._socket.setsockopt.assert_called_once_with(zmq.RCVTIMEO, 17)
        server.comm_metrics.record.assert_called_once_with(
            ClearAction.STEP, "abandoned", ANY
        )

    def test_receive_errors_clear_previous_queued_result(self):
        _, frames = self._frames()
        for responses in (
            (RuntimeError("probe failed"),),
            (zmq.Again(), RuntimeError("blocking receive failed")),
        ):
            with self.subTest(responses=responses):
                server = self._server(frames, *responses)
                self.assertIsNotNone(server.recv_batch(17))
                self.assertTrue(server.last_recv_queued)

                self.assertIsNone(server.recv_batch(17))

                self.assertFalse(server.last_recv_queued)
                self.assertIsNone(server._comm_pending)
                server.comm_metrics.record.assert_any_call("unknown", "recv_error")

    def test_timeout_option_error_is_not_queued(self):
        server = self._server(zmq.Again())
        server.last_recv_queued = True
        server._socket.setsockopt.side_effect = RuntimeError("setsockopt failed")

        self.assertIsNone(server.recv_batch(17))

        self.assertFalse(server.last_recv_queued)
        server._socket.recv_multipart.assert_called_once_with(
            flags=zmq.NOBLOCK, copy=False
        )
        server.comm_metrics.record.assert_called_once_with("unknown", "recv_error")

    def test_empty_or_malformed_request_is_not_queued(self):
        _, frames = self._frames()
        for bad_frames in (
            [],
            [b"target", b"wrong magic"],
            [b"target", b"CTCTRL", b"invalid pickle"],
        ):
            for queued in (True, False):
                with self.subTest(frames=bad_frames, queued=queued):
                    responses = [bad_frames] if queued else [zmq.Again(), bad_frames]
                    server = self._server(frames, *responses)
                    self.assertIsNotNone(server.recv_batch(17))
                    self.assertTrue(server.last_recv_queued)

                    self.assertIsNone(server.recv_batch(17))

                    self.assertFalse(server.last_recv_queued)
                    self.assertIsNone(server._comm_pending)
                    if bad_frames:
                        server.comm_metrics.record.assert_any_call(
                            "unknown", "unpack_error"
                        )

    def test_unpack_preserves_multimodal_return(self):
        batch, frames = self._frames()
        mm = {"r": object()}
        server = self._server(frames)
        with patch.object(transport, "_unpack_request", return_value=(batch, mm)):
            result = server.recv_batch(17)

        self.assertEqual(result, (batch, mm))
        self.assertIs(result[1], mm)
        self.assertTrue(server.last_recv_queued)


@contextmanager
def _bonus_prediction(value):
    # Restore the caller's environment, including an originally absent flag.
    with patch.dict(os.environ):
        os.environ.pop("SGLANG_CLEAR_BONUS_PREDICTION", None)
        if value is not None:
            os.environ["SGLANG_CLEAR_BONUS_PREDICTION"] = value
        yield


class ClearTreeBonusPredictionTest(CustomTestCase):
    def setUp(self):
        super().setUp()
        import torch

        from sglang.srt.speculative.clear_tree_remote import (
            clear_tree_config,
            clear_tree_forward,
            clear_tree_lease,
        )

        self.torch = torch
        self.config = clear_tree_config
        self.forward = clear_tree_forward
        self.lease = clear_tree_lease

    def _host(self, *, finished=False):
        torch = self.torch
        pool = torch.full((1, 16), -1, dtype=torch.int32)
        freed = []

        def free(slots):
            freed.extend(int(slot) for slot in slots.tolist())

        def rollback(req, keep):
            end = int(req.kv_allocated_len)
            free(pool[0, keep:end][pool[0, keep:end] >= 0])
            pool[0, keep:end] = -1
            req.kv_allocated_len = req.kv_committed_len = keep

        host = SimpleNamespace(
            server_args=_args(clear_tree_role="draft", clear_tree_width=2),
            tree_cache=SimpleNamespace(page_size=1),
            req_to_token_pool=SimpleNamespace(req_to_token=pool, device=pool.device),
            token_to_kv_pool_allocator=SimpleNamespace(free=free),
            _sr_device_poisoned=False,
            _sr_is_finished=lambda _req: finished,
            _sr_pause_req=Mock(),
            _clear_rollback=Mock(side_effect=rollback),
            last_batch="stale",
            cur_batch="stale",
        )
        return host, pool, freed

    def _req(self, allocated=5):
        return SimpleNamespace(
            rid="r",
            req_pool_idx=0,
            origin_input_ids=[1, 2, 3],
            output_ids=[],
            kv_committed_len=3,
            kv_allocated_len=allocated,
            clear_tree_lease=None,
        )

    def _state(self):
        from sglang.srt.speculative.clear_tree_remote.clear_tree_ops import (
            ClearTreeState,
            expand_from_topk,
            seed_from_topk,
        )

        torch = self.torch
        state = ClearTreeState.create(
            self.config.ClearTreeConfig(width=2, tau=-1.0, max_depth=4)
        )
        seed_from_topk(state, torch.tensor([0.6, 0.4]), torch.tensor([20, 30]))
        expand_from_topk(
            state,
            torch.tensor([[0.5, 0.2], [0.4, 0.1]]),
            torch.tensor([[21, 31], [22, 32]]),
        )
        state.finished = True
        return state

    def _job(self, host, *, complete=False):
        req = self._req(allocated=7 if complete else 5)
        runner = SimpleNamespace(
            scheduler=host,
            req=req,
            prefix_len=3,
            cached_len=req.kv_allocated_len,
            layer_slots=[self.torch.tensor([100, 101])],
            bonus_ids=[self.torch.tensor([1, 2, 3])],
        )
        if complete:
            runner.layer_slots.append(self.torch.tensor([200, 201]))
        return SimpleNamespace(
            req=req,
            runner=runner,
            keep=3,
            window=([20, 21], [-1, 0], [0, 0]),
            state=self._state(),
        )

    def _batch_fixture(self, count=2):
        torch = self.torch
        host, _pool, freed = self._host()
        pool = torch.full((count, 128), -1, dtype=torch.int64)
        host.req_to_token_pool.req_to_token = pool
        jobs = []
        for i in range(count):
            job = self._job(host, complete=True)
            job.req.rid = str(i)
            job.req.req_pool_idx = i
            job.runner.prefix_len += i
            job.runner.cached_len += i
            job.req.kv_allocated_len = job.runner.cached_len
            job.runner._alloc_layer_slots = Mock()
            jobs.append(job)
        allocator = host.token_to_kv_pool_allocator
        allocator.available_size = Mock(return_value=1000)
        allocator.alloc = Mock(side_effect=lambda n: torch.arange(100, 100 + n))
        host.tree_cache.token_to_kv_pool_allocator = allocator
        host.tree_cache.evictable_size = Mock(return_value=10000)
        host.tree_cache.evict = Mock()
        return host, jobs, pool, freed

    def _candidate(self, token=9, rank=0, parent=-1, chain=()):
        return self.forward._BonusCandidate(parent, token, 3, chain, end_rank=rank)

    @contextmanager
    def _batch_runtime(self, host, *, fail=None):
        # Scoped stubs keep this CPU check independent of Triton/sgl_kernel.
        from types import ModuleType

        torch = self.torch
        common = ModuleType("sglang.srt.mem_cache.common")
        utils = ModuleType("sglang.srt.speculative.spec_utils")

        def alloc(cache, n):
            if cache.token_to_kv_pool_allocator.available_size() < n:
                cache.evict(n)
            return cache.token_to_kv_pool_allocator.alloc(n)

        def assign(rows, pool, starts, ends, loc, count):
            offset = 0
            for i in range(count):
                n = int(ends[i] - starts[i])
                pool[int(rows[i]), int(starts[i]) : int(ends[i])] = loc[
                    offset : offset + n
                ]
                offset += n
            if fail == "assign":
                raise RuntimeError("assign failed after writing")

        common.alloc_token_slots = Mock(side_effect=alloc)
        utils.assign_req_to_token_pool_func = Mock(side_effect=assign)
        output = {}

        def verify(_host, runners, tokens, positions, mask, loc, cached, **kwargs):
            if fail == "batch":
                raise RuntimeError("batch construction failed")
            output.update(
                runners=runners,
                tokens=tokens,
                positions=positions,
                mask=mask,
                loc=loc,
                cached=cached,
                kwargs=kwargs,
            )
            return SimpleNamespace()

        def run(_batch):
            if fail == "forward":
                raise RuntimeError("forward failed")
            n = output["loc"].numel()
            output["logits"] = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3)
            return SimpleNamespace(
                logits_output=SimpleNamespace(next_token_logits=output["logits"]),
                can_run_cuda_graph=True,
            )

        host.run_batch = Mock(side_effect=run)
        with patch.dict(
            "sys.modules",
            {
                common.__name__: common,
                utils.__name__: utils,
            },
        ), patch.object(
            self.forward, "_verify_batch_many", side_effect=verify
        ) as batch:
            yield common.alloc_token_slots, utils.assign_req_to_token_pool_func, batch, output

    def test_bonus_batch_one_allocation_forward_copy_and_owned_logits(self):
        for count in (2, 3, 4):
            with self.subTest(count=count):
                host, jobs, pool, freed = self._batch_fixture(count)
                width = 4
                entries = [
                    (
                        job,
                        self.forward.bonus_verify_chunks(
                            [
                                self._candidate(20 + i),
                                self._candidate(30 + i, 1, 0, (3 + i,)),
                            ]
                            + ([self._candidate(40 + i, 2)] if i % 2 else []),
                            width,
                        )[0],
                    )
                    for i, job in enumerate(jobs)
                ]
                with self._batch_runtime(host) as (
                    alloc,
                    assign,
                    batch,
                    output,
                ), patch.object(
                    self.forward,
                    "_read_bonus_slot_ids",
                    wraps=self.forward._read_bonus_slot_ids,
                ) as read:
                    self.forward._launch_bonus_batch(host, entries, width)
                alloc.assert_called_once_with(host.tree_cache, count * width)
                assign.assert_called_once()
                batch.assert_called_once()
                read.assert_called_once()
                host.run_batch.assert_called_once()
                host.tree_cache.evict.assert_not_called()
                self.assertEqual(output["kwargs"], {"is_bonus": False})
                self.assertEqual(output["positions"].tolist(), [3] * count * width)
                self.assertEqual(output["cached"], [7 + i for i in range(count)])
                offset = 0
                for i, job in enumerate(jobs):
                    end = 7 + i
                    mask = output["mask"][
                        offset : offset + width * (end + width)
                    ].reshape(width, end + width)
                    offset += width * (end + width)
                    self.assertTrue(mask[0, : 3 + i].all())
                    self.assertFalse(mask[0, 3 + i : end].any())
                    self.assertTrue(mask[1, 3 + i])
                    self.assertEqual(
                        mask[:, end:].tolist(),
                        self.torch.eye(width, dtype=self.torch.bool).tolist(),
                    )
                    self.assertEqual(pool[i, end : end + width].tolist(), [-1] * width)
                    self.assertEqual(job.req.kv_allocated_len, end)
                    self.assertEqual(job.runner.cached_len, end)
                    self.assertEqual(len(job.runner.layer_slots), 2)
                    self.assertEqual(
                        [g.slot for g in job.req.opt_bonus_guesses],
                        list(range(100 + i * width, 102 + i * width + i % 2)),
                    )
                    self.assertEqual(
                        job.req.opt_bonus_guesses[0].logits.tolist(),
                        [float(i * width * 3 + j) for j in range(3)],
                    )
                output["logits"].fill_(-100)
                self.assertEqual(
                    jobs[0].req.opt_bonus_guesses[0].logits.tolist(), [0.0, 1.0, 2.0]
                )
                self.assertCountEqual(
                    freed,
                    [
                        100 + i * width + j
                        for i in range(count)
                        for j in range(2 + i % 2, width)
                    ],
                )
                for job in jobs:
                    self.lease.release_opt_bonus_guesses(host, job.req)
                    self.lease.release_opt_bonus_guesses(host, job.req)
                self.assertCountEqual(freed, range(100, 100 + count * width))

    def test_bonus_batch_free_budget_excludes_evictable_and_checks_map(self):
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        for reason in ("kv", "map", "duplicate", "swa"):
            with self.subTest(reason=reason):
                host, jobs, pool, freed = self._batch_fixture()
                if reason == "kv":
                    host.token_to_kv_pool_allocator.available_size.return_value = 7
                elif reason == "map":
                    jobs[0].runner.cached_len = 127
                elif reason == "duplicate":
                    jobs[1].req.req_pool_idx = 0
                else:
                    host.token_to_kv_pool_allocator.full_attn_allocator = object()
                    host.token_to_kv_pool_allocator.swa_attn_allocator = object()
                entries = [
                    (job, self.forward.bonus_verify_chunks([self._candidate()], 4)[0])
                    for job in jobs
                ]
                with self._batch_runtime(host) as (alloc, assign, batch, _output):
                    self.forward._launch_bonus_batch(host, entries, 4)
                alloc.assert_not_called()
                assign.assert_not_called()
                batch.assert_not_called()
                host.tree_cache.evict.assert_not_called()
                if reason != "swa":
                    host.tree_cache.evictable_size.assert_not_called()
                self.assertEqual(freed, [])
                self.assertTrue((pool == -1).all())
                if reason == "swa":
                    self.assertEqual(
                        get_sr_round_metrics(host, "Draft").counts[
                            "opt_bonus_batch_skipped_allocator"
                        ],
                        1,
                    )

    def test_bonus_batch_failure_restores_all_requests_without_double_free(self):
        for fail in ("assign", "batch", "forward", "d2h", "clone", "shape"):
            with self.subTest(fail=fail):
                host, jobs, pool, freed = self._batch_fixture()
                entries = [
                    (job, self.forward.bonus_verify_chunks([self._candidate()], 4)[0])
                    for job in jobs
                ]
                with self._batch_runtime(host, fail=fail), ExitStack() as stack:
                    if fail == "d2h":
                        stack.enter_context(
                            patch.object(
                                self.forward,
                                "_read_bonus_slot_ids",
                                side_effect=RuntimeError("copy failed"),
                            )
                        )
                    elif fail == "clone":
                        stack.enter_context(
                            patch.object(
                                self.lease,
                                "OptBonusGuess",
                                side_effect=[object(), RuntimeError("clone failed")],
                            )
                        )
                    elif fail == "shape":
                        stack.enter_context(
                            patch.object(
                                self.forward,
                                "_as_layer_rows",
                                side_effect=RuntimeError("bad shape"),
                            )
                        )
                    with self.assertRaises(RuntimeError):
                        self.forward._launch_bonus_batch(host, entries, 4)
                self.assertCountEqual(freed, range(100, 108))
                self.assertTrue((pool == -1).all())
                for job in jobs:
                    self.assertEqual(job.req.kv_allocated_len, job.runner.cached_len)
                    self.assertIsNone(getattr(job.req, "opt_bonus_guesses", None))
                self.assertFalse(host._sr_device_poisoned)

    def test_bonus_batch_allocation_failure_has_no_map_or_slots(self):
        host, jobs, pool, freed = self._batch_fixture()
        host.token_to_kv_pool_allocator.alloc.side_effect = RuntimeError("alloc failed")
        entries = [
            (job, self.forward.bonus_verify_chunks([self._candidate()], 4)[0])
            for job in jobs
        ]
        with self._batch_runtime(host) as (_alloc, assign, batch, _output):
            with self.assertRaisesRegex(RuntimeError, "alloc failed"):
                self.forward._launch_bonus_batch(host, entries, 4)
        assign.assert_not_called()
        batch.assert_not_called()
        self.assertTrue((pool == -1).all())
        self.assertEqual(freed, [])

    def test_bonus_batch_unknown_free_is_quarantined_not_retried(self):
        host, jobs, _pool, freed = self._batch_fixture()
        entries = [
            (job, self.forward.bonus_verify_chunks([self._candidate()], 4)[0])
            for job in jobs
        ]
        host.token_to_kv_pool_allocator.free = Mock(
            side_effect=RuntimeError("free uncertain")
        )
        with self._batch_runtime(host), self.assertRaises(RuntimeError):
            self.forward._launch_bonus_batch(host, entries, 4)
        host.token_to_kv_pool_allocator.free.assert_called_once()
        self.assertTrue(host._sr_device_poisoned)
        record = host._sr_kv_quarantine.records[0]
        self.assertEqual(record.owner.loc.numel(), 8)
        host._sr_kv_quarantine.reclaim(host.token_to_kv_pool_allocator)
        host.token_to_kv_pool_allocator.free.assert_called_once()
        self.assertEqual(freed, [])

    def test_bonus_batch_poison_retains_slots_without_free(self):
        host, jobs, _pool, freed = self._batch_fixture()
        entries = [
            (job, self.forward.bonus_verify_chunks([self._candidate()], 4)[0])
            for job in jobs
        ]

        def poison(_batch):
            host._sr_device_poisoned = True
            raise RuntimeError("device lost")

        with self._batch_runtime(host):
            host.run_batch.side_effect = poison
            with self.assertRaises(RuntimeError):
                self.forward._launch_bonus_batch(host, entries, 4)
        self.assertEqual(freed, [])
        self.assertEqual(host._sr_kv_quarantine.records[0].owner.loc.numel(), 8)

    def test_batch_guesses_use_existing_accept_and_seed_logits(self):
        host, jobs, pool, freed = self._batch_fixture()
        entries = [
            (
                job,
                self.forward.bonus_verify_chunks(
                    [self._candidate(30 + i, parent=0, chain=(3 + i,))], 4
                )[0],
            )
            for i, job in enumerate(jobs)
        ]
        with self._batch_runtime(host):
            self.forward._launch_bonus_batch(host, entries, 4)
        for i, job in enumerate(jobs):
            req = job.req
            req.origin_input_ids = list(range(3 + i))
            req.output_ids = [20, 30 + i if i == 0 else 99]
            lease = self.lease.ClearTreeKVLease(
                keep=3 + i,
                layer_slots=((50 + i, 60 + i),),
                tokens=(20,),
                parents=(-1,),
                cols=(0,),
                depths=(0,),
            )
            self.lease.accept_opt_bonus_guess(host, req, lease, [50 + i])
            self.assertIsNone(req.opt_bonus_guesses)
            if i == 0:
                self.assertEqual(pool[i, 4 + i].item(), 100 + i * 4)
                self.assertEqual(req.opt_prefill_tail_logits.tolist(), [0.0, 1.0, 2.0])
            else:
                self.assertFalse(hasattr(req, "opt_prefill_tail_logits"))
        self.assertCountEqual(freed, [101, 102, 103, 104, 105, 106, 107])

    def test_bonus_batch_cleanup_wait_failure_quarantines_owner(self):
        host, jobs, _pool, freed = self._batch_fixture()
        loc = SimpleNamespace(device=self.torch.device("cuda"))
        owner = self.forward._BonusBatchOwner(
            [(job, []) for job in jobs],
            4,
            loc=loc,
            owned=self.torch.arange(100, 108),
            mapped=True,
        )
        stream = SimpleNamespace(
            synchronize=Mock(side_effect=RuntimeError("sync unknown"))
        )
        with patch.object(
            self.torch,
            "get_device_module",
            return_value=SimpleNamespace(current_stream=lambda: stream),
        ):
            with self.assertRaisesRegex(RuntimeError, "sync unknown"):
                self.forward._cleanup_bonus_batch(host, owner)
        self.assertTrue(host._sr_device_poisoned)
        self.assertTrue(owner.mapped)
        self.assertEqual(freed, [])
        self.assertIs(host._sr_kv_quarantine.records[0].owner, owner)

    def test_bonus_batch_path_metrics_and_no_serial_retry(self):
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        for flag in (True, False, None):
            with self.subTest(flag=flag):
                host, jobs, _pool, _freed = self._batch_fixture()
                entries = [
                    (job, self.forward.bonus_verify_chunks([self._candidate()], 4)[0])
                    for job in jobs
                ]
                with self._batch_runtime(host):
                    run = host.run_batch.side_effect

                    def result(batch):
                        out = run(batch)
                        out.can_run_cuda_graph = flag
                        return out

                    host.run_batch.side_effect = result
                    self.forward._launch_bonus_batch(host, entries, 4)
                path = "unknown" if flag is None else ("graph" if flag else "eager")
                self.assertEqual(
                    get_sr_round_metrics(host, "Draft").counts[
                        "opt_bonus_batch_" + path
                    ],
                    1,
                )
        host, jobs, _pool, _freed = self._batch_fixture()
        with _bonus_prediction("1"), patch.object(
            self.forward, "select_bonus_candidates", return_value=[self._candidate()]
        ), patch.object(
            self.forward, "_bonus_batch_width", return_value=4
        ), patch.object(
            self.forward,
            "_launch_bonus_batch",
            side_effect=RuntimeError("optional failure"),
        ) as launch, patch.object(
            self.forward, "_forward_bonus_candidates"
        ) as serial:
            self.forward.maybe_attach_bonus_guesses_many(host, jobs)
        launch.assert_called_once()
        serial.assert_not_called()
        self.assertTrue(all(not job.runner.bonus_ids for job in jobs))

    def test_bonus_batch_width_checks_real_padded_layer_keys(self):
        from sglang.srt.speculative.clear_tree_remote import clear_tree_width as widths

        host, _jobs, _pool, _freed = self._batch_fixture()
        graph = SimpleNamespace(
            capture_bs=[1, 2, 4],
            disable_padding=False,
            graphs={"r8_4": object(), "r16_4": object(), "r5_1_s0": object()},
            _make_graph_key=lambda b, s, w: f"r{w}_{b}",
        )
        host.req_to_token_pool.device = "cuda"
        host.model_worker = SimpleNamespace(
            model_runner=SimpleNamespace(graph_runner=graph)
        )
        with patch.object(widths, "LAYER_WIDTHS", [4, 5, 8, 16]):
            self.assertEqual(self.forward._bonus_batch_width(host, 3, 5), 8)
            self.assertEqual(self.forward._bonus_batch_width(host, 3, 9), 16)
            self.assertEqual(self.forward._bonus_batch_width(host, 3, 65), 16)
            self.assertIsNone(self.forward._bonus_batch_width(host, 5, 9))
            graph.graphs.clear()
            self.assertIsNone(self.forward._bonus_batch_width(host, 3, 9))
            graph.disable_padding = True
            self.assertEqual(self.forward._bonus_batch_width(host, 3, 9), 16)
            self.assertIsNone(self.forward._bonus_batch_width(host, 2, 9))
            for device in ("cpu", "npu"):
                # NPU torch.device is only available when torch_npu is loaded.
                with patch.object(
                    self.forward.torch,
                    "device",
                    return_value=SimpleNamespace(type=device),
                ):
                    self.assertEqual(self.forward._bonus_batch_width(host, 3, 9), 16)
            host.model_worker.model_runner.graph_runner = None
            self.assertEqual(self.forward._bonus_batch_width(host, 3, 5), 5)

    def test_bonus_many_truncates_by_rank_but_keeps_original_order(self):
        host, jobs, _pool, _freed = self._batch_fixture()
        plans = [self._candidate(i, rank) for i, rank in enumerate((4, 1, 0, 3, 2))]
        with _bonus_prediction("1"), patch.object(
            self.forward, "select_bonus_candidates", return_value=plans
        ), patch.object(
            self.forward, "_bonus_batch_width", return_value=4
        ), patch.object(
            self.forward, "_launch_bonus_batch"
        ) as launch, patch.object(
            self.forward, "_forward_bonus_candidates"
        ) as serial:
            self.forward.maybe_attach_bonus_guesses_many(host, jobs)
        launch.assert_called_once()
        self.assertEqual(
            [[p.token for p in chunk] for _, chunk in launch.call_args.args[1]],
            [[1, 2, 3, 4]] * 2,
        )
        serial.assert_not_called()
        self.assertTrue(all(not job.runner.bonus_ids for job in jobs))

    def test_bonus_many_filters_incomplete_and_preserves_single_lookahead(self):
        for mode in (
            "off",
            "single",
            "lookahead",
            "incomplete",
            "finished",
            "page",
            "no_graph",
        ):
            with self.subTest(mode=mode):
                host, jobs, _pool, _freed = self._batch_fixture()
                if mode == "single":
                    jobs = jobs[:1]
                elif mode == "lookahead":
                    jobs[0].runner._lookahead_satisfied = True
                elif mode == "incomplete":
                    for job in jobs:
                        job.runner.cached_len -= 2
                elif mode == "finished":
                    host._sr_is_finished = lambda _req: True
                elif mode == "page":
                    host.tree_cache.page_size = 128
                with _bonus_prediction("0" if mode == "off" else "1"), patch.object(
                    self.forward,
                    "select_bonus_candidates",
                    return_value=[self._candidate()],
                ), patch.object(
                    self.forward,
                    "_bonus_batch_width",
                    return_value=None if mode == "no_graph" else 4,
                ), patch.object(
                    self.forward, "_launch_bonus_batch"
                ) as launch, patch.object(
                    self.forward, "_forward_bonus_candidates"
                ) as serial:
                    self.forward.maybe_attach_bonus_guesses_many(host, jobs)
                launch.assert_not_called()
                self.assertEqual(
                    serial.call_count, {"single": 1, "lookahead": 2}.get(mode, 0)
                )
                self.assertTrue(all(not job.runner.bonus_ids for job in jobs))

    def test_deferred_flush_batches_before_publishing(self):
        host, jobs, _pool, _freed = self._batch_fixture()
        events = []

        def fitting(_host, _runners, _steps, **_kwargs):
            events.append("layer")
            return [], 1

        def launch(_host, _entries, _width):
            events.append("bonus")

        def publish(*_args):
            self.assertTrue(all(not j.runner.bonus_ids for j in jobs))
            events.append("publish")

        with _bonus_prediction("1"), patch.object(
            self.forward, "_forward_verify_fitting", side_effect=fitting
        ), patch.object(
            self.forward, "select_bonus_candidates", return_value=[self._candidate()]
        ), patch.object(
            self.forward, "_bonus_batch_width", return_value=4
        ), patch.object(
            self.forward, "_launch_bonus_batch", side_effect=launch
        ), patch.object(
            self.forward, "_publish_or_rollback", side_effect=publish
        ):
            self.forward.forward_deferred_final_layers(host, jobs)
        self.assertEqual(events, ["layer", "bonus", "publish", "publish"])

    def test_bonus_prediction_flag_defaults_off_and_accepts_explicit_values(self):
        for value, expected in ((None, False), ("0", False), ("1", True)):
            with self.subTest(value=value), _bonus_prediction(value):
                self.assertIs(self.config.bonus_prediction_enabled(), expected)

    def test_disabled_attach_drops_ids_without_selection_or_forward(self):
        for value in (None, "0"):
            with self.subTest(value=value), _bonus_prediction(value):
                host, _pool, _freed = self._host()
                job = self._job(host)
                with patch.object(
                    self.forward, "select_bonus_candidates"
                ) as select, patch.object(
                    self.forward, "_forward_bonus_candidates"
                ) as predict:
                    self.forward.maybe_attach_bonus_guesses(host, job)
                select.assert_not_called()
                predict.assert_not_called()
                self.assertEqual(job.runner.bonus_ids, [])
                self.assertIsNone(getattr(job.req, "opt_bonus_guesses", None))

    def test_disabled_append_does_not_run_dedicated_topk(self):
        torch = self.torch
        for value, expected_calls in ((None, 0), ("0", 0), ("1", 1)):
            with self.subTest(value=value), _bonus_prediction(value):
                host, _pool, _freed = self._host()
                runner = SimpleNamespace(scheduler=host, bonus_ids=[])
                with patch.object(torch, "topk", wraps=torch.topk) as topk:
                    self.forward._append_layer_bonus_ids(
                        [runner], [torch.tensor([[0.1, 0.9, 0.3, 0.2]])]
                    )
                self.assertEqual(topk.call_count, expected_calls)
                self.assertEqual(len(runner.bonus_ids), expected_calls)
                if expected_calls:
                    self.assertEqual(runner.bonus_ids[0].tolist(), [[1, 2, 3]])

    def test_deferred_final_layer_publishes_lease_and_leaves_only_real_bonus(self):
        torch = self.torch
        for value in (None, "0"):
            with self.subTest(value=value), _bonus_prediction(value):
                host, pool, freed = self._host()
                job = self._job(host)

                def fitting(_host, runners, steps, *, consume):
                    runner = runners[0]
                    self.assertEqual(steps[0].positions.tolist(), [2, 2])
                    runner.layer_slots.append(torch.tensor([200, 201]))
                    runner.cached_len += 2
                    runner.req.kv_allocated_len = runner.cached_len
                    consume(runners, [torch.zeros(2, 4)])
                    return [], 1

                with patch.object(
                    self.forward, "_forward_verify_fitting", side_effect=fitting
                ) as fit, patch.object(
                    self.forward, "_forward_bonus_candidates"
                ) as predict, patch.object(
                    self.forward, "_host_bonus_ids"
                ) as topk:
                    self.forward.forward_deferred_final_layers(host, [job])
                fit.assert_called_once()
                predict.assert_not_called()
                topk.assert_not_called()
                self.assertEqual(job.runner.bonus_ids, [])
                self.assertEqual(
                    job.req.clear_tree_lease.layer_slots, ((100, 101), (200, 201))
                )
                self.assertEqual(job.req.kv_allocated_len, 3)
                self.assertEqual(pool[0, 3:7].tolist(), [-1] * 4)
                self.assertIsNone(host.last_batch)
                self.assertIsNone(host.cur_batch)
                host._sr_pause_req.assert_called_once_with(job.req)
                job.req.output_ids = [20, 21, 99]
                job.req.sr_align_result = SimpleNamespace(
                    kind="append_n", old_local_len=3
                )
                self.lease.consume_clear_tree_lease(host, job.req)
                self.assertEqual(pool[0, 3:5].tolist(), [100, 200])
                self.assertEqual(job.req.kv_committed_len, 5)
                self.assertEqual(
                    len(job.req.origin_input_ids)
                    + len(job.req.output_ids)
                    - job.req.kv_committed_len,
                    1,
                )
                self.assertCountEqual(freed, [101, 201])

    def test_deferred_final_layers_group_by_depth(self):
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        for depths, expected in (((2, 2), [["a", "b"]]), ((1, 2), [["a"], ["b"]])):
            with self.subTest(depths=depths), _bonus_prediction(None):
                host, _pool, _freed = self._host()
                jobs = [self._job(host), self._job(host)]
                for job, rid, depth in zip(jobs, ("a", "b"), depths):
                    job.req.rid = rid
                    job.state.depth = depth
                groups = []

                def fitting(_host, runners, _steps, **_kwargs):
                    groups.append([runner.req.rid for runner in runners])
                    return [], 1

                with patch.object(
                    self.forward, "_forward_verify_fitting", side_effect=fitting
                ), patch.object(self.forward, "_publish_or_rollback"):
                    self.forward.forward_deferred_final_layers(host, jobs)
                self.assertEqual(groups, expected)
                self.assertEqual(
                    get_sr_round_metrics(host, "Draft").counts[
                        "opt_layer_forwards_deferred"
                    ],
                    len(expected),
                )

    def test_deferred_failure_drops_partial_layer_and_finished_releases_tree(self):
        torch = self.torch
        for finished in (False, True):
            with self.subTest(finished=finished), _bonus_prediction(None):
                host, pool, freed = self._host(finished=finished)
                job = self._job(host)
                pool[0, 3:5] = torch.tensor([100, 101])

                def fail(_host, runners, _steps, **_kwargs):
                    runner = runners[0]
                    runner.layer_slots.append(torch.tensor([200, 201]))
                    runner.req.kv_allocated_len = 7
                    pool[0, 5:7] = torch.tensor([200, 201])
                    raise RuntimeError("last layer failed")

                with patch.object(
                    self.forward, "_forward_verify_fitting", side_effect=fail
                ) as fit:
                    self.forward.forward_deferred_final_layers(host, [job])
                self.assertEqual(fit.call_count, 0 if finished else 1)
                self.assertEqual(job.runner.bonus_ids, [])
                self.assertEqual(pool[0, 5:7].tolist(), [-1, -1])
                if finished:
                    self.assertIsNone(job.req.clear_tree_lease)
                    self.assertCountEqual(freed, [100, 101])
                    self.assertEqual(job.req.kv_allocated_len, 3)
                    host._sr_pause_req.assert_not_called()
                else:
                    self.assertEqual(
                        job.req.clear_tree_lease.layer_slots, ((100, 101),)
                    )
                    self.assertCountEqual(freed, [200, 201])
                    self.assertEqual(job.req.kv_committed_len, 3)

    def test_enabled_prediction_precedes_publish_and_always_clears_ids(self):
        for fail in (False, True):
            with self.subTest(fail=fail), _bonus_prediction("1"):
                host, pool, _freed = self._host()
                job = self._job(host, complete=True)
                job.runner._lookahead_satisfied = True
                job.runner._alloc_layer_slots = Mock()
                pool[0, 3:7] = self.torch.tensor([100, 101, 200, 201])
                events = []
                plans = [object()]
                publish = self.forward._publish_or_rollback

                def predict(_host, runner, selected):
                    self.assertIs(selected, plans)
                    self.assertIsNone(job.req.clear_tree_lease)
                    self.assertEqual(pool[0, 3:7].tolist(), [100, 101, 200, 201])
                    self.assertTrue(runner.bonus_ids)
                    events.append("predict")
                    if fail:
                        raise RuntimeError("bonus failed")

                def published(*args):
                    self.assertEqual(job.runner.bonus_ids, [])
                    events.append("publish")
                    return publish(*args)

                with patch.object(
                    self.forward, "select_bonus_candidates", return_value=plans
                ), patch.object(
                    self.forward, "_forward_bonus_candidates", side_effect=predict
                ), patch.object(
                    self.forward, "_publish_or_rollback", side_effect=published
                ), patch.object(
                    self.forward, "_forward_verify_fitting"
                ) as fit:
                    self.forward.forward_deferred_final_layers(host, [job])
                self.assertEqual(events, ["predict", "publish"])
                self.assertIsNotNone(job.req.clear_tree_lease)
                fit.assert_not_called()

    def test_enabled_bonus_hit_and_miss_free_unused_guesses(self):
        torch = self.torch
        for hit in (False, True):
            with self.subTest(hit=hit), _bonus_prediction("1"):
                host, pool, freed = self._host()
                req = self._req(allocated=3)
                req.output_ids = [20, 30 if hit else 99]
                req.sr_align_result = SimpleNamespace(kind="append_n", old_local_len=3)
                req.clear_tree_lease = self.lease.ClearTreeKVLease(
                    keep=3,
                    layer_slots=((100, 101),),
                    tokens=(20,),
                    parents=(-1,),
                    cols=(0,),
                    depths=(0,),
                )
                seed = torch.tensor([1.0, 2.0, 3.0])
                req.opt_bonus_guesses = [
                    self.lease.OptBonusGuess(parent=0, token=30, slot=200, logits=seed),
                    self.lease.OptBonusGuess(
                        parent=-1, token=8, slot=201, logits=torch.zeros(3)
                    ),
                ]
                self.lease.consume_clear_tree_lease(host, req)
                self.assertEqual(req.kv_committed_len, 5 if hit else 4)
                self.assertEqual(pool[0, 3].item(), 100)
                self.assertIsNone(req.opt_bonus_guesses)
                self.assertCountEqual(freed, [101, 201] if hit else [101, 200, 201])
                if hit:
                    self.assertEqual(pool[0, 4].item(), 200)
                    self.assertTrue(
                        torch.equal(self.forward.take_cached_tail_logits(req, 5), seed)
                    )
                else:
                    self.assertIsNone(getattr(req, "opt_prefill_tail_logits", None))
                self.lease.release_opt_bonus_guesses(host, req)
                self.assertEqual(len(freed), 2 if hit else 3)

    def test_lookahead_runtime_bonus_buffer_is_opt_in(self):
        torch = self.torch
        for value, enabled in ((None, False), ("0", False), ("1", True)):
            with self.subTest(value=value), _bonus_prediction(value):
                host, _pool, _freed = self._host()
                runner = SimpleNamespace(req=self._req())
                batch = SimpleNamespace(
                    seq_lens=object(),
                    seq_lens_cpu=object(),
                    orig_seq_lens=object(),
                    req_pool_indices=object(),
                )
                with patch.object(
                    self.forward, "_lookahead_batch", return_value=batch
                ), patch.object(
                    self.forward, "_round_width", return_value=2
                ), patch.object(
                    self.forward, "_PinnedLayer"
                ), patch.object(
                    self.forward, "_PinnedBonus"
                ) as bonus, patch.object(
                    torch.cuda, "Event"
                ):
                    runtime = self.forward.prepare_lookahead_runtime(host, runner)
                self.assertIs(runtime[0], batch)
                self.assertEqual(bonus.call_count, int(enabled))
                self.assertIs(runtime[2], bonus.return_value if enabled else None)
                self.assertEqual(runner.bonus_ids, [])

    def test_lookahead_disabled_skips_final_bonus_topk(self):
        torch = self.torch
        # A one-layer tree still launches its final layer, but must not score
        # that layer solely to predict bonus ids when prediction is disabled.
        config = self.config.ClearTreeConfig(width=2, tau=-1.0, max_depth=2)
        for value, enabled in ((None, False), ("0", False), ("1", True)):
            with self.subTest(value=value), _bonus_prediction(value):
                host, _pool, _freed = self._host()
                runner = SimpleNamespace(
                    bonus_ids=[], _last_logits=torch.zeros(2, 4), launch_layer=Mock()
                )
                slot = Mock()
                slot.clone_cpu.return_value = (
                    torch.tensor([0.6, 0.4]),
                    torch.tensor([20, 30]),
                    torch.zeros(2, dtype=torch.long),
                    torch.tensor([[20, 30, 40], [0, 0, 0]]),
                    "seed",
                )
                bonus = Mock() if enabled else None
                with patch.object(
                    self.forward, "_vocab_topk", wraps=self.forward._vocab_topk
                ) as topk:
                    windows, state, layers = self.forward.grow_lookahead(
                        host,
                        runner,
                        config,
                        torch.zeros(4),
                        (object(), (slot, slot), bonus),
                    )
                self.assertEqual(topk.call_count, 2 if enabled else 1)
                self.assertEqual(layers, 1)
                self.assertEqual(state.depth, 1)
                self.assertTrue(windows)
                runner.launch_layer.assert_called_once()
                self.assertTrue(runner.lookahead_inflight)
                self.assertEqual(len(runner.bonus_ids), int(enabled))
                self.assertIs(getattr(runner, "_lookahead_bonus", None), bonus)
                if enabled:
                    bonus.copy.assert_called_once()

    def test_lookahead_without_bonus_waits_before_publish_and_never_reforwards(self):
        with _bonus_prediction(None):
            host, _pool, _freed = self._host()
            job = self._job(host, complete=True)
            runner = job.runner
            runner.lookahead_inflight = True
            runner._lookahead_bonus = None
            runner._lookahead_satisfied = False
            events = []

            def wait():
                self.assertTrue(runner.lookahead_inflight)
                self.assertFalse(runner._lookahead_satisfied)
                self.assertIsNone(job.req.clear_tree_lease)
                events.append("launch_wait")

            runner._lookahead_launch_event = SimpleNamespace(
                synchronize=Mock(side_effect=wait)
            )
            publish = self.forward._publish_or_rollback

            def published(*args):
                self.assertFalse(runner.lookahead_inflight)
                self.assertTrue(runner._lookahead_satisfied)
                events.append("publish")
                return publish(*args)

            with patch.object(
                self.forward, "_forward_verify_fitting"
            ) as fit, patch.object(
                self.forward, "_forward_bonus_candidates"
            ) as predict, patch.object(
                self.forward, "_publish_or_rollback", side_effect=published
            ):
                self.forward.forward_deferred_final_layers(host, [job])
                self.forward._settle_lookahead_job(host, job)
            self.assertEqual(events, ["launch_wait", "publish"])
            runner._lookahead_launch_event.synchronize.assert_called_once()
            fit.assert_not_called()
            predict.assert_not_called()
            self.assertEqual(
                job.req.clear_tree_lease.layer_slots, ((100, 101), (200, 201))
            )
            self.assertEqual(runner.bonus_ids, [])

    def test_lookahead_disabled_omits_candidate_copies_and_bonus_kv_budget(self):
        torch = self.torch
        for value, enabled in ((None, False), ("0", False), ("1", True)):
            with self.subTest(value=value), _bonus_prediction(value):
                host, _pool, _freed = self._host()
                host.server_args.clear_tree_max_depth = 4
                self.assertEqual(
                    self.forward._lookahead_kv_need(host, 1),
                    7 + (3 if enabled else 0),
                )

                def cpu_empty(*args, **kwargs):
                    kwargs.pop("pin_memory", None)
                    return torch.zeros(*args, **kwargs)

                with patch.object(torch, "empty", side_effect=cpu_empty), patch.object(
                    torch.cuda, "Event"
                ):
                    slot = self.forward._PinnedLayer(2)
                self.assertEqual(slot.ids is not None, enabled)
                slot.copy_seed(torch.ones(3), torch.arange(3))
                self.assertEqual(slot.clone_cpu()[3] is not None, enabled)
                slot.copy_expand(
                    torch.ones(2),
                    torch.arange(2),
                    torch.zeros(2, dtype=torch.long),
                    torch.zeros(2, 3, dtype=torch.long),
                )
                self.assertEqual(slot.clone_cpu()[3] is not None, enabled)
                self.assertEqual(slot.event.synchronize.call_count, 2)

    def test_lookahead_failed_wait_requires_completion_before_lease_publish(self):
        for sync_fails in (False, True):
            with self.subTest(sync_fails=sync_fails), _bonus_prediction(None):
                host, _pool, freed = self._host()
                host.device = "cpu"
                host._sr_clear_stamps_after_poison = Mock()
                job = self._job(host, complete=True)
                runner = job.runner
                runner.lookahead_inflight = True
                runner._lookahead_bonus = None
                runner._lookahead_launch_event = SimpleNamespace(
                    synchronize=Mock(side_effect=RuntimeError("event failed"))
                )
                sync = Mock(
                    side_effect=RuntimeError("sync failed") if sync_fails else None
                )
                with patch.object(
                    self.torch,
                    "get_device_module",
                    return_value=SimpleNamespace(synchronize=sync),
                ), patch.object(self.forward, "_forward_verify_fitting") as fit:
                    self.forward.forward_deferred_final_layers(host, [job])
                sync.assert_called_once()
                fit.assert_not_called()
                self.assertFalse(runner.lookahead_inflight)
                self.assertTrue(runner._lookahead_satisfied)
                self.assertEqual(host._sr_device_poisoned, sync_fails)
                self.assertEqual(
                    bool(getattr(runner, "lookahead_dropped", False)), sync_fails
                )
                self.assertEqual(
                    host._sr_clear_stamps_after_poison.call_count, int(sync_fails)
                )
                if sync_fails:
                    self.assertIsNone(job.req.clear_tree_lease)
                    host._clear_rollback.assert_not_called()
                    self.assertEqual(freed, [])
                else:
                    self.assertIsNotNone(job.req.clear_tree_lease)

    def test_bonus_capture_graphs_are_opt_in_without_changing_layer_or_target(self):
        from sglang.srt.speculative.clear_tree_remote.clear_tree_graph import (
            ClearTreeGraphCapture,
        )
        from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
            install_candidates,
        )

        install_candidates(None, None)
        self.addCleanup(install_candidates, None, None)
        draft = _args(clear_tree_role="draft", speculative_num_draft_tokens=64)
        target = _args(clear_tree_role="target", speculative_num_draft_tokens=65)
        for value, enabled in ((None, False), ("0", False), ("1", True)):
            with self.subTest(value=value), _bonus_prediction(value):
                expected = [65, 33, 17, 9, 5, 3] if enabled else []
                self.assertEqual(self.forward.opt_draft_bonus_ntpbs(draft), expected)
                self.assertEqual(
                    ClearTreeGraphCapture.bonus_ntpbs_for_capture(draft), expected
                )
                self.assertEqual(self.forward.opt_draft_bonus_ntpbs(target), [])
                self.assertEqual(
                    self.forward.opt_draft_capture_extra_keys(draft, 3),
                    [0] if enabled else [None],
                )
                for width in (1, 2, 4, 8, 16, 32, 64):
                    self.assertEqual(
                        self.forward.opt_draft_capture_extra_keys(draft, width), [None]
                    )
                self.assertEqual(
                    self.forward.opt_draft_capture_extra_keys(target, 65), [None]
                )
                self.assertEqual(
                    SpeculativeAlgorithm.CLEAR_TREE_REMOTE.dual_ntpb_cuda_graph_options(
                        draft
                    ),
                    [64, 32, 16, 8, 4, 2, 1],
                )
                self.assertEqual(
                    SpeculativeAlgorithm.CLEAR_TREE_REMOTE.dual_ntpb_cuda_graph_options(
                        target
                    ),
                    [65, 33, 17, 9, 5, 3, 1],
                )

    def test_single_and_batch_growth_only_collect_ids_when_enabled(self):
        from sglang.srt.speculative.clear_tree_remote import clear_tree_policy as policy
        from sglang.srt.speculative.clear_tree_remote import clear_tree_replay as replay
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        # Importing the scheduler pulls GPU/model dependencies (and dill) into
        # CPU CI. Execute only this unchanged method from its parsed source;
        # all forward, config, and lease logic uses the real ClearTree modules.
        path = Path(self.forward.__file__).with_name("clear_tree_scheduler.py")
        mixin = next(
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.ClassDef) and node.name == "ClearTreeSchedulerMixin"
        )
        method = next(
            node
            for node in mixin.body
            if isinstance(node, ast.FunctionDef) and node.name == "_clear_expand_one"
        )
        for value, enabled in ((None, False), ("0", False), ("1", True)):
            for batch in (False, True):
                with self.subTest(value=value, batch=batch), _bonus_prediction(
                    value
                ), ExitStack() as stack:
                    host, _pool, _freed = self._host()
                    reqs = [self._req(allocated=3) for _ in range(2 if batch else 1)]
                    for index, req in enumerate(reqs):
                        req.rid = str(index)
                    host._sr_resume_req = Mock()
                    host._sr_isolate_need = Mock()
                    host._clear_release_uncommitted = Mock()
                    runners = []

                    def make_runner(_host, req, keep):
                        runner = SimpleNamespace(
                            req=req, prefix_len=keep, bonus_ids=[], forward_layer=Mock()
                        )
                        runners.append(runner)
                        return runner

                    def grow(_config, _tails, _forward, _host, *, states_out, topk_out):
                        self.assertEqual(
                            topk_out, [[] for _ in reqs] if enabled else None
                        )
                        states_out.extend([self._state() for _ in reqs])
                        if enabled:
                            for rows in topk_out:
                                rows.append(self.torch.tensor([1, 2, 3]))
                        return [([20, 21], [-1, 0], [0, 0]) for _ in reqs]

                    def finish(_host, _req, runner, *_args):
                        self.assertEqual(len(runner.bonus_ids), int(enabled))
                        return None

                    stack.enter_context(
                        patch.object(
                            policy,
                            "attach_round",
                            return_value=SimpleNamespace(width=2),
                        )
                    )
                    stack.enter_context(patch.object(policy, "remember_choice"))
                    stack.enter_context(patch.object(replay, "queue_hold_replay"))
                    stack.enter_context(
                        patch.object(
                            self.forward, "OptLayerForward", side_effect=make_runner
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            self.forward,
                            "ingest_tail_logits",
                            return_value=[self.torch.zeros(4) for _ in reqs],
                        )
                    )
                    grow_mock = stack.enter_context(
                        patch.object(
                            self.forward, "grow_active_layers", side_effect=grow
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            self.forward, "finish_opt_expand_req", side_effect=finish
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            self.forward, "commit_deferred_expands", return_value=None
                        )
                    )
                    stack.enter_context(
                        patch.object(self.forward, "drain_opt_layer_gpu_us")
                    )
                    stack.enter_context(
                        patch.object(
                            self.forward, "lookahead_supported", return_value=False
                        )
                    )
                    if batch:
                        result = self.forward.grow_batch(host, reqs)
                        self.assertEqual(len(result), 2)
                    else:
                        namespace = dict(
                            vars(self.forward),
                            get_sr_round_metrics=get_sr_round_metrics,
                        )
                        exec(
                            compile(
                                ast.Module(body=[method], type_ignores=[]),
                                str(path),
                                "exec",
                            ),
                            namespace,
                        )
                        result = namespace[method.name](host, reqs[0])
                        self.assertEqual(result[0], [20, 21])
                    grow_mock.assert_called_once()
                    self.assertEqual(len(runners), len(reqs))


if __name__ == "__main__":
    unittest.main()
