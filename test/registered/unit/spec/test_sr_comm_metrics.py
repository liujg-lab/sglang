"""Deterministic CPU coverage of the real SR protocol and transport paths."""

import ast
import logging
import pickle
import sys
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.speculative.standalone_remote import sr_transport as transport
from sglang.srt.speculative.standalone_remote.sr_protocol import (
    SRAction,
    SRBatchReply,
    SRBatchRequest,
    SRDraftRequest,
)
from sglang.srt.speculative.standalone_remote import sr_round_metrics as round_metrics
from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
    SRCommMetrics,
    SRRoundMetrics,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class Clock:
    def __init__(self):
        self.now = 0

    def advance(self, ms):
        self.now += int(ms * 1e6)

    def ns(self):
        return self.now

    def seconds(self):
        return self.now / 1e9


class Again(Exception):
    pass


class Socket:
    def __init__(self, clock):
        self.clock = clock
        self.incoming = []
        self.sent = []
        self.send_error = False
        self.closed = False
        self.send_ms = 0

    def setsockopt(self, *args):
        pass

    def bind(self, url):
        pass

    def connect(self, url):
        pass

    def close(self, **kwargs):
        self.closed = True

    def send_multipart(self, frames, **kwargs):
        if self.send_error:
            raise RuntimeError("send failed")
        self.clock.advance(self.send_ms)
        self.sent.append(frames)

    def recv_multipart(self, flags=None, **kwargs):
        if self.incoming:
            value = self.incoming.pop(0)
            if isinstance(value, Exception):
                raise value
            return value
        if flags is None:
            self.clock.advance(10)
        raise Again()


class TestCommTransport(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.sockets = []

        def make_socket(kind):
            sock = Socket(self.clock)
            self.sockets.append(sock)
            return sock

        zmq = SimpleNamespace(
            Again=Again,
            NOBLOCK=1,
            DEALER=2,
            ROUTER=3,
            LINGER=4,
            RCVHWM=5,
            SNDHWM=6,
            IDENTITY=7,
            RCVTIMEO=8,
            Context=SimpleNamespace(
                instance=lambda: SimpleNamespace(socket=make_socket)
            ),
        )
        for p in (
            patch.dict(sys.modules, {"zmq": zmq}),
            patch.object(transport.time, "perf_counter_ns", self.clock.ns),
            patch.object(transport.time, "perf_counter", self.clock.seconds),
        ):
            p.start()
            self.addCleanup(p.stop)
        self.client = transport.SRTargetClient("draft.example", "30019", timeout_ms=25)
        self.server = transport.SRDraftServer("draft.example", "30019")
        self.addCleanup(self.client.close)
        self.addCleanup(self.server.close)

    def request(self, seq=1, action=SRAction.STEP):
        return SRBatchRequest("session", seq, action)

    def reply_frames(self, seq=1, residence=110_000_000, session="session"):
        data = {"session_id": session, "rpc_seq": seq, "reqs": []}
        if residence is not None:
            data["draft_residence_ns"] = residence
        return [transport._CTRL_FRAME, pickle.dumps(data)]

    def window(self, endpoint, action="step"):
        return endpoint.comm_metrics.windows[action]

    def test_round_trip_uses_durations_despite_remote_clock_offset(self):
        self.client.send_batch(self.request())
        request_frames = self.client._socket.sent[-1]
        self.clock.advance(5)
        self.clock.advance(9_000_000)  # Remote monotonic clock has unrelated epoch.
        self.server._socket.incoming.append([b"identity", *request_frames])
        self.server.recv_batch()
        self.clock.advance(110)
        self.server.send_batch(SRBatchReply("session", 1))
        reply_frames = self.server._socket.sent[-1][1:]
        self.clock.advance(-9_000_000)
        self.clock.advance(5)
        self.client._socket.incoming.append(reply_frames)
        self.assertIsNotNone(self.client.recv_batch("session", 1))
        values = self.window(self.client)["values"]
        self.assertEqual(values["rpc_elapsed_ms"], [120])
        self.assertEqual(values["draft_residence_ms"], [110])
        self.assertEqual(values["non_draft_elapsed_ms"], [10])
        self.assertEqual(values["request_bytes"], [sum(map(len, request_frames))])
        self.assertEqual(values["reply_bytes"], [sum(map(len, reply_frames))])
        self.assertEqual(
            self.window(self.server)["values"]["draft_residence_ms"], [110]
        )
        self.assertIsNone(self.client._comm_pending)
        self.assertIsNone(self.server._comm_pending)

    def test_prefill_gap_is_separate_from_step_and_send_call(self):
        self.client._socket.send_ms = 2
        for action, delay in ((SRAction.PREFILL, 80), (SRAction.STEP, 0)):
            self.client.send_batch(self.request(action=action))
            self.clock.advance(delay)
            self.client._socket.incoming.append(self.reply_frames(residence=0))
            self.client.recv_batch("session", 1)
            values = self.window(self.client, action.value)["values"]
            self.assertEqual(values["recv_entry_gap_ms"], [delay])
            self.assertEqual(values["request_send_host_ms"], [2])
            self.assertEqual(values["rpc_elapsed_ms"], [delay + 2])

    def test_missing_and_invalid_timing_never_reject_reply_or_report_zero(self):
        for value in (None, -1, True, "110", 1.5, 121_000_000, float("nan")):
            self.client.send_batch(self.request())
            self.clock.advance(120)
            self.client._socket.incoming.append(self.reply_frames(residence=value))
            self.assertIsNotNone(self.client.recv_batch("session", 1))
        window = self.window(self.client)
        self.assertEqual(window["counts"]["success"], 7)
        self.assertEqual(window["counts"]["missing_timing"], 1)
        self.assertEqual(window["counts"]["invalid_timing"], 6)
        self.assertNotIn("non_draft_elapsed_ms", window["values"])

    def test_stale_replies_and_bad_payload_do_not_finish_sample(self):
        self.client.send_batch(self.request())
        self.clock.advance(120)
        self.client._socket.incoming.extend(
            [
                self.reply_frames(session="old"),
                self.reply_frames(seq=0),
                [b"invalid pickle"],
                self.reply_frames(),
            ]
        )
        self.assertEqual(self.client.recv_batch("session", 1).rpc_seq, 1)
        window = self.window(self.client)
        self.assertEqual(window["calls"], 1)
        self.assertEqual(window["counts"]["stale_reply"], 2)
        self.assertEqual(window["counts"]["unpack_error"], 1)
        self.assertEqual(window["values"]["non_draft_elapsed_ms"], [10])

    def test_timeout_reconnect_and_close_clear_pending(self):
        self.client.send_batch(self.request())
        old_socket = self.client._socket
        self.assertIsNone(self.client.recv_batch("session", 1))
        self.assertTrue(old_socket.closed)
        self.assertIsNone(self.client._comm_pending)
        self.client.send_batch(self.request(seq=2))
        self.clock.advance(120)
        self.client._socket.incoming.extend(
            [self.reply_frames(), self.reply_frames(seq=2)]
        )
        self.client.recv_batch("session", 2)
        window = self.window(self.client)
        self.assertEqual(window["counts"]["timeout"], 1)
        self.assertEqual(window["values"]["rpc_elapsed_ms"], [120])
        self.client.send_batch(self.request(seq=3))
        self.client.close()
        self.assertIsNone(self.client._comm_pending)
        self.assertFalse(self.client.comm_metrics.windows)

    def test_send_only_commands_and_io_failures_clear_state(self):
        for action in (SRAction.FINISH, SRAction.ABORT):
            self.client.send_batch(self.request(action=action))
            self.server._socket.incoming.append([b"id", *self.client._socket.sent[-1]])
            self.server.recv_batch()
            self.assertIsNone(self.client._comm_pending)
            self.assertIsNone(self.server._comm_pending)
            self.assertNotIn(
                "rpc_elapsed_ms", self.window(self.client, action.value)["values"]
            )
        self.client._socket.send_error = True
        with self.assertRaisesRegex(RuntimeError, "send failed"):
            self.client.send_batch(self.request())
        self.assertIsNone(self.client._comm_pending)
        self.client._socket.send_error = False
        self.client.send_batch(self.request())
        self.client._socket.incoming.append(RuntimeError("recv failed"))
        self.assertIsNone(self.client.recv_batch("session", 1))
        self.assertIsNone(self.client._comm_pending)
        self.server._socket.incoming.append(
            [b"id", *transport._pack_request(self.request())]
        )
        self.server.recv_batch()
        self.server._socket.send_error = True
        with self.assertRaisesRegex(RuntimeError, "send failed"):
            self.server.send_batch(SRBatchReply("session", 1))
        self.assertIsNone(self.server._comm_pending)

    def test_server_identity_stale_and_close(self):
        frames = [b"id", *transport._pack_request(self.request())]
        self.server._socket.incoming.append(frames)
        self.server.recv_batch()
        reply = SRBatchReply("session", 1, draft_residence_ns=999)
        self.server.send_batch(reply, identity=b"other")
        self.assertIsNone(
            transport._unpack_reply(self.server._socket.sent[-1][1:]).draft_residence_ns
        )
        self.assertIsNotNone(self.server._comm_pending)
        self.clock.advance(10)
        self.server.send_batch(SRBatchReply("session", 1))
        self.assertEqual(
            transport._unpack_reply(
                self.server._socket.sent[-1][1:]
            ).draft_residence_ns,
            10_000_000,
        )
        self.assertIsNone(self.server._comm_pending)
        self.server._socket.incoming.append(frames)
        batch, _ = self.server.recv_batch()
        self.server.remember(batch)
        self.assertTrue(self.server.is_stale(batch))
        self.assertIsNone(self.server._comm_pending)
        self.server._socket.incoming.append(frames)
        self.server.recv_batch()
        self.server.close()
        self.assertIsNone(self.server._comm_pending)
        self.assertFalse(self.server.comm_metrics.windows)

    def test_multimodal_pack_time_and_bytes_without_extra_copy(self):
        buffer = memoryview(array("i", [1, 2, 3, 4]))

        def pack_mm():
            self.clock.advance(3)
            return {"rid": "r"}, [buffer]

        req = self.request()
        req.reqs = [SRDraftRequest(rid="r", step_id=0, base_committed_len=0)]
        self.client.send_batch(req, {"r": SimpleNamespace(to_multipart=pack_mm)})
        self.assertIs(self.client._socket.sent[-1][-1], buffer)
        self.assertEqual(self.client._comm_pending.values["request_pack_ms"], 3)
        self.assertEqual(
            self.client._comm_pending.values["request_bytes"],
            sum(memoryview(f).nbytes for f in self.client._socket.sent[-1]),
        )

    def test_pack_and_draft_receive_failures_do_not_leave_state(self):
        with patch.object(transport, "_pack_request", side_effect=ValueError("pack")):
            with self.assertRaises(ValueError):
                self.client.send_batch(self.request())
        self.assertIsNone(self.client._comm_pending)
        self.assertEqual(self.window(self.client)["counts"]["send_error"], 1)
        self.server._socket.incoming.append([b"id", b"bad pickle"])
        self.assertIsNone(self.server.recv_batch())
        self.server._socket.incoming.append(RuntimeError("recv failed"))
        self.assertIsNone(self.server.recv_batch())
        self.assertIsNone(self.server._comm_pending)
        counts = self.window(self.server, "unknown")["counts"]
        self.assertEqual(counts["unpack_error"], 1)
        self.assertEqual(counts["recv_error"], 1)
        self.assertIsNone(self.server.recv_batch(timeout_ms=1))
        self.assertEqual(self.window(self.server, "unknown")["calls"], 2)

    def test_drain_counts_stale_without_starting_rpc_early(self):
        self.client._socket.incoming.extend([self.reply_frames(), self.reply_frames()])
        self.client.send_batch(self.request())
        self.assertEqual(self.window(self.client)["counts"]["stale_reply"], 2)
        self.assertIsNotNone(self.client._comm_pending)
        self.server._socket.incoming.append(
            [b"id", *transport._pack_request(self.request())]
        )
        self.server.recv_batch()
        pending = self.server._comm_pending
        self.server._socket.incoming.append(
            [b"id", *transport._pack_request(self.request(seq=2))]
        )
        self.assertEqual(self.server.drain(), 1)
        self.assertIs(self.server._comm_pending, pending)
        self.assertEqual(self.window(self.server, "unknown")["counts"]["stale_request"], 1)

    def test_new_session_prefill_reset_preserves_received_request_timing(self):
        # Execute the production reset method without importing GPU scheduler
        # dependencies. Only cache/model state is replaced; transport is real.
        path = Path(transport.__file__).parent / "drafter/sr_draft_scheduler_mixin.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(
            n for n in tree.body
            if isinstance(n, ast.ClassDef)
            and n.name == "StandaloneRemoteDraftSchedulerMixin"
        )
        method = next(
            n for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_sr_wipe_all"
        )
        namespace = {"logger": logging.getLogger(__name__)}
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        reset = namespace["_sr_wipe_all"]
        for queued in (False, True):
            with self.subTest(queued=queued):
                request = SRBatchRequest("new-session", int(queued), SRAction.PREFILL)
                self.client.send_batch(request)
                self.clock.advance(1)
                self.server._socket.incoming.append(
                    [b"current", *self.client._socket.sent[-1]]
                )
                self.server.recv_batch()
                pending = self.server._comm_pending
                start = pending.start_ns
                if queued:
                    old = SRBatchRequest("old-session", 9, SRAction.STEP)
                    self.server._socket.incoming.append(
                        [b"old-identity", *transport._pack_request(old)]
                    )
                scheduler = SimpleNamespace(
                    sr_state=SimpleNamespace(clear=lambda: []),
                    sr_waiting=[],
                    sr_server=self.server,
                    _sr_http_alive=lambda: False,
                    _sr_reset_scheduler_caches=lambda: self.clock.advance(20),
                )
                reset(scheduler)
                self.assertIs(self.server._comm_pending, pending)
                self.assertEqual(pending.start_ns, start)
                self.assertEqual(pending.key, (request.session_id, request.rpc_seq))
                self.assertEqual(pending.identity, b"current")
                self.assertEqual(self.server.last_rpc_seq, -1)
                self.assertFalse(self.server._socket.incoming)
                self.clock.advance(170)
                self.server.send_batch(SRBatchReply(request.session_id, request.rpc_seq))
                frames = self.server._socket.sent[-1]
                self.assertEqual(frames[0], b"current")
                self.assertEqual(
                    transport._unpack_reply(frames[1:]).draft_residence_ns,
                    190_000_000,
                )
                self.assertIsNone(self.server._comm_pending)
                self.clock.advance(3)  # Target local prefill overlaps Draft work.
                self.client._socket.incoming.append(frames[1:])
                self.client.recv_batch(request.session_id, request.rpc_seq)
                values = self.window(self.client, "prefill")["values"]
                self.assertEqual(values["recv_entry_gap_ms"][-1], 194)
                self.assertEqual(values["rpc_elapsed_ms"][-1], 194)
                self.assertEqual(values["non_draft_elapsed_ms"][-1], 4)
        self.assertEqual(self.window(self.server, "prefill")["counts"], {"success": 2})
        self.assertEqual(self.window(self.client, "prefill")["counts"], {"success": 2})
        self.assertEqual(self.window(self.server, "unknown")["counts"]["stale_request"], 1)

    def test_next_receive_abandons_unanswered_request_after_drain(self):
        for seq in (1, 2):
            self.server._socket.incoming.append(
                [b"id", *transport._pack_request(self.request(seq=seq))]
            )
            self.server.recv_batch()
            self.assertEqual(self.server._comm_pending.key, ("session", seq))
            self.assertEqual(self.server.drain(), 0)
        self.assertEqual(self.window(self.server)["counts"]["abandoned"], 1)
        self.server.send_batch(SRBatchReply("session", 2))
        self.assertIsNone(self.server._comm_pending)

    def test_stale_traffic_does_not_extend_receive_deadline(self):
        self.client.send_batch(self.request())
        socket = self.client._socket

        def receive(**kwargs):
            self.clock.advance(10)
            return self.reply_frames(seq=0)

        with patch.object(socket, "recv_multipart", side_effect=receive):
            self.assertIsNone(self.client.recv_batch("session", 1))
        self.assertEqual(self.clock.now, 30_000_000)
        self.assertEqual(self.window(self.client)["counts"]["stale_reply"], 3)
        self.assertEqual(self.window(self.client)["counts"]["timeout"], 1)
        self.assertIsNone(self.client._comm_pending)


class TestCommMetrics(unittest.TestCase):
    def test_protocol_compatibility(self):
        old = {"session_id": "s", "rpc_seq": 1, "reqs": []}
        self.assertEqual(SRBatchReply.from_dict(old).to_dict(), old)
        new = dict(old, draft_residence_ns=123, future_field="ignored")
        self.assertEqual(SRBatchReply.from_dict(new).draft_residence_ns, 123)
        for invalid in (-1, True, "bad"):
            reply = SRBatchReply.from_dict(dict(old, draft_residence_ns=invalid))
            self.assertEqual(reply.draft_residence_ns, invalid)
            self.assertNotIn("draft_residence_ns", reply.to_dict())

    def test_byte_count_does_not_materialize_buffers(self):
        class Frame:
            buffer = memoryview(array("i", [1, 2, 3]))

            def __bytes__(self):
                raise AssertionError("must not copy frame")

        self.assertEqual(transport._multipart_nbytes([b"abc", Frame()]), 15)
        self.assertEqual(transport.sr_endpoint("127.0.0.1", "123", bind=True)[1], "ipc")

    def test_window_counts_percentiles_and_valid_sample_counts(self):
        metrics = SRCommMetrics("Target", "tcp")
        with patch.object(metrics, "_emit", wraps=metrics._emit) as emit:
            for i in range(1, 33):
                if i <= 16:
                    metrics.record(
                        SRAction.STEP,
                        "success",
                        {"rpc_elapsed_ms": i, "non_draft_elapsed_ms": 1},
                    )
                elif i == 32:
                    metrics.record(SRAction.STEP, "timeout")
                else:
                    metrics.count(SRAction.STEP, "missing_timing")
                    metrics.record(SRAction.STEP, "success", {"rpc_elapsed_ms": i})
            self.assertEqual(emit.call_count, 2)  # First sample plus 32-call window.
            action, kind, calls, counts, values = emit.call_args.args
            self.assertEqual((action, kind, calls), ("step", "window", 32))
            self.assertEqual(counts["timeout"], 1)
            self.assertEqual(counts["missing_timing"], 15)
            self.assertEqual(
                metrics.summarize(values["rpc_elapsed_ms"]),
                {"n": 31, "mean": 16, "p50": 16, "p95": 30, "max": 31},
            )
            self.assertEqual(len(values["non_draft_elapsed_ms"]), 16)
            self.assertFalse(metrics.windows)
            metrics.record(SRAction.STEP, "success", {"rpc_elapsed_ms": 1})
            self.assertEqual(emit.call_count, 2)  # No repeated first-success log.
            metrics.flush()
            self.assertEqual(emit.call_count, 3)
            self.assertFalse(metrics.windows)


class TestSRRoundMetrics(unittest.TestCase):
    def test_host_max_is_single_round_not_mean_and_clears(self):
        metrics = SRRoundMetrics("Draft")
        for _ in range(31):
            with metrics.round():
                metrics.add_host("tree_expand_pack", 0.001)
        self.assertAlmostEqual(metrics.host_max["tree_expand_pack"], 0.001)
        self.assertAlmostEqual(metrics.host["tree_expand_pack"], 0.031)

        metrics.add_host("tree_expand_pack", 9.0)
        self.assertAlmostEqual(metrics.host_max["tree_expand_pack"], 0.001)
        self.assertAlmostEqual(metrics.host["tree_expand_pack"], 0.031)

        logged = {}

        def capture(fmt, *args, **kwargs):
            logged["mean"] = args[1]
            logged["max"] = args[2]

        with patch.object(round_metrics.logger, "info", side_effect=capture):
            with metrics.round():
                metrics.add_host("tree_expand_pack", 0.100)

        self.assertAlmostEqual(logged["max"]["tree_expand_pack"], 100.0)
        self.assertLess(logged["mean"]["tree_expand_pack"], 10.0)
        self.assertFalse(metrics.host_max)
        self.assertFalse(metrics.host)

    def test_add_host_ignores_inactive_and_phase_updates_max(self):
        metrics = SRRoundMetrics("Draft")
        metrics.add_host("tree_prepare_meta", 0.5)
        self.assertFalse(metrics.host)
        self.assertFalse(metrics.host_max)
        with metrics.round():
            with metrics.phase("tree_alloc_kv"):
                pass
            metrics.add_host("tree_alloc_kv", 0.02)
        self.assertAlmostEqual(metrics.host_max["tree_alloc_kv"], 0.02)
        self.assertGreater(metrics.host["tree_alloc_kv"], 0.02)


if __name__ == "__main__":
    unittest.main()
