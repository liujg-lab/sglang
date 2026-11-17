"""CPU checks for the CLEAR_TREE_REMOTE shell.

The shell is a renamed copy of fixed-width OPT. It must not be recognized as
OPT or STANDALONE_REMOTE, and OPT must not pick up ClearTree fields.
"""

import unittest
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


if __name__ == "__main__":
    unittest.main()
