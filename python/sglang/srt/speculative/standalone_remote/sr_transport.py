"""pyzmq DEALER/ROUTER transport for STANDALONE_REMOTE sync RPC.

Draft binds ROUTER (server); Target connects DEALER (client). Control is a
pickled SRBatchRequest/Reply. Vision tensors, when present, follow as
multipart frames on the same RPC so the two channels cannot race.
"""

from __future__ import annotations

import logging
import pickle
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from sglang.srt.speculative.standalone_remote.sr_protocol import (
    SRAction,
    SRBatchReply,
    SRBatchRequest,
)
from sglang.srt.speculative.standalone_remote.sr_round_metrics import SRCommMetrics

logger = logging.getLogger(__name__)

_CTRL_FRAME = b"SRCTRL"
_MM_FRAME = b"SRMM"


def _multipart_nbytes(frames) -> int:
    """Application frame bytes, without materializing tensor/buffer contents."""
    return sum(memoryview(getattr(f, "buffer", f)).nbytes for f in frames)


@dataclass
class _CommCall:
    key: tuple
    action: SRAction
    start_ns: int
    sent_ns: int = 0
    identity: Optional[bytes] = None
    values: dict = field(default_factory=dict)


class _CommEndpoint:
    def _finish_comm(self, outcome):
        call, self._comm_pending = self._comm_pending, None
        if call is not None:
            self.comm_metrics.record(call.action, outcome, call.values)

    def _count_comm(self, event, n=1):
        call = self._comm_pending
        self.comm_metrics.count(call.action if call else "unknown", event, n)


stale_drop_counts = {"session": 0, "rpc_seq": 0, "step": 0, "base_len": 0}


def note_stale_drop(reason: str) -> None:
    stale_drop_counts[reason] = stale_drop_counts.get(reason, 0) + 1


def _as_bytes(frame) -> bytes:
    if isinstance(frame, (bytes, bytearray)):
        return bytes(frame)
    if isinstance(frame, memoryview):
        return frame.tobytes()
    buf = getattr(frame, "buffer", None)
    if buf is not None:
        return bytes(buf)
    return bytes(frame)


def _load_mm_payload():
    from sglang.srt.speculative.standalone_remote.sr_mm_payload import SRMMPayload

    return SRMMPayload


def sr_endpoint(addr: str, port: str, *, bind: bool) -> Tuple[str, str]:
    """Return (zmq_url, transport). transport is 'ipc' or 'tcp'."""
    addr = addr or "127.0.0.1"
    if addr in ("127.0.0.1", "0.0.0.0"):
        safe = "".join(ch if ch.isalnum() else "_" for ch in addr)
        return f"ipc:///tmp/sr_{safe}_{port}", "ipc"
    if bind:
        return f"tcp://*:{port}", "tcp"
    return f"tcp://{addr}:{port}", "tcp"


def _pack_request(
    batch: SRBatchRequest, mm_by_rid: Optional[Dict[str, Any]] = None
) -> List[bytes]:
    frames: List[Any] = [_CTRL_FRAME, pickle.dumps(batch.to_dict())]
    mm_by_rid = mm_by_rid or {}
    if not mm_by_rid:
        return frames
    metas: List[Dict[str, Any]] = []
    buffers: List[Any] = []
    for req in batch.reqs:
        payload = mm_by_rid.get(req.rid)
        if payload is None:
            metas.append(None)
            continue
        meta, bufs = payload.to_multipart()
        offset = len(buffers)
        meta["_sr_buf_offset"] = offset
        meta["_sr_buf_count"] = len(bufs)
        metas.append(meta)
        buffers.extend(bufs)
    frames.append(_MM_FRAME)
    frames.append(pickle.dumps(metas))
    frames.extend(buffers)
    return frames


def _unpack_request(frames: List[Any]) -> Tuple[SRBatchRequest, Dict[str, Any]]:
    if not frames:
        raise ValueError("empty STANDALONE_REMOTE frames")
    raw = [_as_bytes(f) for f in frames]
    offset = 0
    if raw[0] != _CTRL_FRAME:
        if len(raw) >= 2 and raw[1] == _CTRL_FRAME:
            offset = 1
        else:
            return SRBatchRequest.from_dict(pickle.loads(raw[0])), {}
    batch = SRBatchRequest.from_dict(pickle.loads(raw[offset + 1]))
    mm_by_rid: Dict[str, Any] = {}
    rest = raw[offset + 2 :]
    if rest and rest[0] == _MM_FRAME:
        metas = pickle.loads(rest[1])
        bufs = rest[2:]
        for meta in metas:
            if not meta:
                continue
            off = int(meta.get("_sr_buf_offset", 0))
            n = int(meta.get("_sr_buf_count", 0))
            payload = _load_mm_payload().from_multipart(meta, bufs[off : off + n])
            mm_by_rid[payload.rid] = payload
    return batch, mm_by_rid


def _pack_reply(batch: SRBatchReply) -> List[bytes]:
    return [_CTRL_FRAME, pickle.dumps(batch.to_dict())]


def _unpack_reply(frames: List[Any]) -> SRBatchReply:
    if not frames:
        raise ValueError("empty STANDALONE_REMOTE reply")
    raw = [_as_bytes(f) for f in frames]
    idx = 0
    if raw[0] != _CTRL_FRAME:
        if len(raw) >= 2 and raw[1] == _CTRL_FRAME:
            idx = 1
        else:
            return SRBatchReply.from_dict(pickle.loads(raw[0]))
    return SRBatchReply.from_dict(pickle.loads(raw[idx + 1]))


class SRTargetClient(_CommEndpoint):
    """Target-side DEALER. Connects to Draft ROUTER. Blocking RPC with
    session_id/rpc_seq filtering and queue drain before each send."""

    def __init__(
        self,
        addr: str,
        port: str,
        timeout_ms: int = 5000,
        drop_callback: Optional[Callable[[str], None]] = None,
        rebuild_on_timeout: bool = True,
    ):
        import zmq

        self.addr = addr
        self.port = port
        self.timeout_ms = max(1, int(timeout_ms))
        self.drop_callback = drop_callback
        self.rebuild_on_timeout = rebuild_on_timeout
        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._identity = b"sr-target-" + uuid.uuid4().hex[:8].encode()
        self._socket = None
        self._url, transport = sr_endpoint(addr, port, bind=False)
        self.comm_metrics = SRCommMetrics("Target", transport)
        self._comm_pending = None
        self._open()

    def _open(self) -> None:
        import zmq

        self._finish_comm("connection_reset")
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
        sock = self._ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVHWM, 256)
        sock.setsockopt(zmq.SNDHWM, 256)
        sock.setsockopt(zmq.IDENTITY, self._identity)
        sock.connect(self._url)
        self._socket = sock
        logger.info("[SR] Target DEALER connected to %s", self._url)

    def close(self) -> None:
        self._finish_comm("closed")
        self.comm_metrics.flush()
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
            self._socket = None

    def _drain(self) -> int:
        import zmq

        n = 0
        while True:
            try:
                self._socket.recv_multipart(flags=zmq.NOBLOCK, copy=False)
                n += 1
            except zmq.Again:
                break
            except Exception:
                break
        return n

    def _note_drop(self, reason: str) -> None:
        if self.drop_callback is not None:
            try:
                self.drop_callback(reason)
            except Exception:
                pass

    def send_batch(
        self,
        batch: SRBatchRequest,
        mm_by_rid: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._finish_comm("abandoned")
        drained = self._drain()
        if drained:
            self.comm_metrics.count(batch.action, "stale_reply", drained)
            self._note_drop("rpc_seq")
            note_stale_drop("rpc_seq")
            logger.info("[SR] Target drained %d stale frames before send", drained)
        call = _CommCall(
            (batch.session_id, batch.rpc_seq), batch.action, time.perf_counter_ns()
        )
        self._comm_pending = call
        try:
            frames = _pack_request(batch, mm_by_rid)
            packed = time.perf_counter_ns()
            call.values["request_pack_ms"] = (packed - call.start_ns) / 1e6
            call.values["request_bytes"] = _multipart_nbytes(frames)
            send_start = time.perf_counter_ns()
            self._socket.send_multipart(frames, copy=False)
            call.sent_ns = time.perf_counter_ns()
            call.values["request_send_host_ms"] = (call.sent_ns - send_start) / 1e6
        except Exception:
            self._finish_comm("send_error")
            raise
        if batch.action in (SRAction.FINISH, SRAction.ABORT):
            self._finish_comm("send_only")

    def recv_batch(self, session_id: str, rpc_seq: int) -> Optional[SRBatchReply]:
        import zmq

        recv_entry = time.perf_counter_ns()
        call = self._comm_pending
        if call is not None and call.key != (session_id, rpc_seq):
            self._finish_comm("abandoned")
            call = None
        if call is not None:
            call.values["recv_entry_gap_ms"] = (recv_entry - call.sent_ns) / 1e6
        deadline = time.perf_counter() + self.timeout_ms / 1000.0
        while True:
            remaining_ms = int((deadline - time.perf_counter()) * 1000)
            if remaining_ms <= 0:
                waited_ms = (time.perf_counter_ns() - recv_entry) / 1e6
                self._finish_comm("timeout")
                if self.rebuild_on_timeout:
                    self._open()
                logger.warning(
                    "[SR] Target recv timeout session=%s rpc_seq=%s waited_ms=%.3f",
                    session_id,
                    rpc_seq,
                    waited_ms,
                )
                return None
            try:
                self._socket.setsockopt(zmq.RCVTIMEO, max(1, remaining_ms))
                frames = self._socket.recv_multipart(copy=False)
            except zmq.Again:
                continue
            except Exception as e:
                self._finish_comm("recv_error")
                logger.warning("[SR] Target recv error: %s", e)
                return None
            try:
                unpack_start = time.perf_counter_ns()
                reply = _unpack_reply(frames)
                unpack_end = time.perf_counter_ns()
            except Exception as e:
                self._count_comm("unpack_error")
                logger.warning("[SR] Target failed to unpack reply: %s", e)
                self._note_drop("rpc_seq")
                continue
            if reply.session_id != session_id:
                self._count_comm("stale_reply")
                self._note_drop("session")
                note_stale_drop("session")
                logger.info(
                    "[SR] drop stale reply session=%s expected=%s",
                    reply.session_id,
                    session_id,
                )
                continue
            if reply.rpc_seq != rpc_seq:
                self._count_comm("stale_reply")
                self._note_drop("rpc_seq")
                note_stale_drop("rpc_seq")
                logger.info(
                    "[SR] drop stale reply rpc_seq=%s expected=%s",
                    reply.rpc_seq,
                    rpc_seq,
                )
                continue
            if call is not None:
                elapsed = unpack_end - call.start_ns
                call.values.update(
                    rpc_elapsed_ms=elapsed / 1e6,
                    reply_unpack_ms=(unpack_end - unpack_start) / 1e6,
                    reply_bytes=_multipart_nbytes(frames),
                )
                residence = reply.draft_residence_ns
                if residence is None:
                    self._count_comm("missing_timing")
                elif type(residence) is not int or not 0 <= residence <= elapsed:
                    self._count_comm("invalid_timing")
                else:
                    call.values["draft_residence_ms"] = residence / 1e6
                    call.values["non_draft_elapsed_ms"] = (elapsed - residence) / 1e6
                self._finish_comm("success")
            return reply


class SRDraftServer(_CommEndpoint):
    """Draft-side ROUTER. Binds and serves blocking recv/send."""

    def __init__(self, addr: str, port: str):
        import zmq

        self.addr = addr
        self.port = port
        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._url, transport = sr_endpoint(addr, port, bind=True)
        self.comm_metrics = SRCommMetrics("Draft", transport)
        self._comm_pending = None
        sock = self._ctx.socket(zmq.ROUTER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVHWM, 256)
        sock.setsockopt(zmq.SNDHWM, 256)
        sock.bind(self._url)
        self._socket = sock
        self._last_identity = None
        self.last_rpc_seq = -1
        self.last_session_id: Optional[str] = None
        logger.info("[SR] Draft ROUTER bound on %s", self._url)

    def close(self) -> None:
        self._finish_comm("closed")
        self.comm_metrics.flush()
        try:
            self._socket.close(linger=0)
        except Exception:
            pass

    def drain(self) -> int:
        """Discard queued messages, preserving the received request's timing.

        Session reset calls this while processing the current request. Its
        identity and residence start must survive until the matching reply.
        """
        import zmq

        n = 0
        while True:
            try:
                self._socket.recv_multipart(flags=zmq.NOBLOCK, copy=False)
                n += 1
            except zmq.Again:
                break
            except Exception:
                break
        if n:
            self.comm_metrics.count("unknown", "stale_request", n)
        return n

    def recv_batch(
        self, timeout_ms: Optional[int] = None
    ) -> Optional[Tuple[SRBatchRequest, Dict[str, Any]]]:
        import zmq

        self._finish_comm("abandoned")
        try:
            if timeout_ms is None:
                self._socket.setsockopt(zmq.RCVTIMEO, -1)
            else:
                self._socket.setsockopt(zmq.RCVTIMEO, max(1, int(timeout_ms)))
            frames = self._socket.recv_multipart(copy=False)
            received = time.perf_counter_ns()
        except zmq.Again:
            return None
        except Exception as e:
            self.comm_metrics.record("unknown", "recv_error")
            logger.warning("[SR] Draft recv error: %s", e)
            return None
        if not frames:
            return None
        self._last_identity = frames[0]
        try:
            unpack_start = time.perf_counter_ns()
            batch, mm = _unpack_request(frames)
            unpack_end = time.perf_counter_ns()
        except Exception as e:
            self.comm_metrics.record("unknown", "unpack_error")
            logger.warning("[SR] Draft failed to unpack request: %s", e)
            return None
        self._comm_pending = _CommCall(
            (batch.session_id, batch.rpc_seq),
            batch.action,
            received,
            identity=_as_bytes(frames[0]),
            values={
                "request_unpack_ms": (unpack_end - unpack_start) / 1e6,
                "request_bytes": _multipart_nbytes(frames[1:]),
            },
        )
        if batch.action in (SRAction.FINISH, SRAction.ABORT):
            self._finish_comm("receive_only")
        return batch, mm

    def is_stale(self, batch: SRBatchRequest) -> bool:
        stale = self.last_session_id is not None and (
            batch.session_id < self.last_session_id
            or (
                batch.session_id == self.last_session_id
                and batch.rpc_seq <= self.last_rpc_seq
            )
        )
        if (
            stale
            and self._comm_pending is not None
            and self._comm_pending.key == (batch.session_id, batch.rpc_seq)
        ):
            self._finish_comm("stale_request")
        return stale

    def remember(self, batch: SRBatchRequest) -> None:
        self.last_session_id = batch.session_id
        self.last_rpc_seq = batch.rpc_seq

    def send_batch(self, batch: SRBatchReply, identity: Optional[Any] = None) -> None:
        ident = identity if identity is not None else self._last_identity
        if ident is None:
            self._finish_comm("send_error")
            logger.warning("[SR] Draft send with no ROUTER identity")
            return
        call = self._comm_pending
        if call is not None and (
            call.key != (batch.session_id, batch.rpc_seq)
            or call.identity != _as_bytes(ident)
        ):
            # A late/unrelated reply must not consume the current request's
            # timing, just as Target filtering keeps its current RPC alive.
            self._count_comm("unmatched_reply")
            call = None
        batch.draft_residence_ns = None
        try:
            pack_start = time.perf_counter_ns()
            if call is not None:
                batch.draft_residence_ns = pack_start - call.start_ns
                call.values["draft_residence_ms"] = batch.draft_residence_ns / 1e6
            payload = _pack_reply(batch)
            pack_end = time.perf_counter_ns()
            if call is not None:
                call.values["reply_pack_ms"] = (pack_end - pack_start) / 1e6
                call.values["reply_bytes"] = _multipart_nbytes(payload)
            frames = [ident, *payload]
            send_start = time.perf_counter_ns()
            self._socket.send_multipart(frames, copy=False)
            send_end = time.perf_counter_ns()
            if call is not None:
                call.values["reply_send_host_ms"] = (send_end - send_start) / 1e6
        except Exception:
            self._finish_comm("send_error")
            raise
        if call is not None:
            self._finish_comm("success")


def make_transport_from_server_args(server_args):
    role = server_args.standalone_remote_role
    addr = server_args.standalone_remote_addr or "127.0.0.1"
    port = server_args.standalone_remote_port or "30019"
    if role == "draft":
        return SRDraftServer(addr, port)
    return SRTargetClient(
        addr,
        port,
        timeout_ms=server_args.standalone_remote_rpc_timeout_ms,
    )
