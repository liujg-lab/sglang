"""DEALER/ROUTER transport for CLEAR_TREE_REMOTE.

Control frames are pickled ClearTree batches. The magic is ``CTCTRL``, so a
standalone-remote peer cannot unpack them. Vision tensors still follow as
multipart frames through the existing payload helper.
"""

from __future__ import annotations

import logging
import pickle
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from sglang.srt.speculative.clear_tree_remote.clear_tree_protocol import (
    ClearAction,
    ClearBatchReply,
    ClearBatchRequest,
)
from sglang.srt.speculative.standalone_remote.sr_round_metrics import SRCommMetrics

logger = logging.getLogger(__name__)

_CTRL_FRAME = b"CTCTRL"
_MM_FRAME = b"CTMM"


def _multipart_nbytes(frames) -> int:
    return sum(memoryview(getattr(frame, "buffer", frame)).nbytes for frame in frames)


@dataclass
class _CommCall:
    key: tuple
    action: ClearAction
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


def _as_bytes(frame) -> bytes:
    if isinstance(frame, (bytes, bytearray)):
        return bytes(frame)
    if isinstance(frame, memoryview):
        return frame.tobytes()
    buf = getattr(frame, "buffer", None)
    if buf is not None:
        return bytes(buf)
    return bytes(frame)


def _frame_view(frame) -> memoryview:
    if isinstance(frame, memoryview):
        return frame
    buf = getattr(frame, "buffer", None)
    if buf is not None:
        return memoryview(buf)
    return memoryview(frame)


def _is_magic(frame, magic: bytes) -> bool:
    view = _frame_view(frame)
    return view.nbytes == len(magic) and view.tobytes() == magic


def _ctrl_offset(frames: List[Any]) -> int:
    if not frames:
        raise ValueError("empty CLEAR_TREE_REMOTE frames")
    if _is_magic(frames[0], _CTRL_FRAME):
        return 0
    if len(frames) >= 2 and _is_magic(frames[1], _CTRL_FRAME):
        return 1
    raise ValueError("CLEAR_TREE_REMOTE frame missing CTCTRL")


def _load_mm_payload():
    from sglang.srt.speculative.standalone_remote.sr_mm_payload import SRMMPayload

    return SRMMPayload


def clear_endpoint(addr: str, port: str, *, bind: bool) -> Tuple[str, str]:
    """Return (zmq_url, transport). A loopback address uses a ClearTree IPC path."""
    addr = addr or "127.0.0.1"
    if addr in ("127.0.0.1", "0.0.0.0"):
        safe = "".join(ch if ch.isalnum() else "_" for ch in addr)
        return f"ipc:///tmp/ct_{safe}_{port}", "ipc"
    if bind:
        return f"tcp://*:{port}", "tcp"
    return f"tcp://{addr}:{port}", "tcp"


def _pack_request(
    batch: ClearBatchRequest, mm_by_rid: Optional[Dict[str, Any]] = None
) -> List[Any]:
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


def _unpack_request(frames: List[Any]) -> Tuple[ClearBatchRequest, Dict[str, Any]]:
    offset = _ctrl_offset(frames)
    batch = ClearBatchRequest.from_dict(pickle.loads(_frame_view(frames[offset + 1])))
    mm_by_rid: Dict[str, Any] = {}
    rest_at = offset + 2
    if rest_at < len(frames) and _is_magic(frames[rest_at], _MM_FRAME):
        metas = pickle.loads(_frame_view(frames[rest_at + 1]))
        bufs = frames[rest_at + 2 :]
        for meta in metas:
            if not meta:
                continue
            off = int(meta.get("_sr_buf_offset", 0))
            count = int(meta.get("_sr_buf_count", 0))
            payload = _load_mm_payload().from_multipart(meta, bufs[off : off + count])
            mm_by_rid[payload.rid] = payload
    return batch, mm_by_rid


def _pack_reply(batch: ClearBatchReply) -> List[Any]:
    return [_CTRL_FRAME, pickle.dumps(batch.to_dict())]


def _unpack_reply(frames: List[Any]) -> ClearBatchReply:
    idx = _ctrl_offset(frames)
    return ClearBatchReply.from_dict(pickle.loads(_frame_view(frames[idx + 1])))


class ClearTargetClient(_CommEndpoint):
    """Target-side DEALER. Connects to the Draft ROUTER."""

    def __init__(
        self,
        addr: str,
        port: str,
        timeout_ms: int = 5000,
        drop_callback: Optional[Callable[[str, int], None]] = None,
        rebuild_on_timeout: bool = True,
    ):
        import zmq

        self.addr = addr
        self.port = port
        self.timeout_ms = max(1, int(timeout_ms))
        self.drop_callback = drop_callback
        self.rebuild_on_timeout = rebuild_on_timeout
        self._ctx = zmq.Context.instance()
        self._identity = b"ct-target-" + uuid.uuid4().hex[:8].encode()
        self._socket = None
        self._url, transport = clear_endpoint(addr, port, bind=False)
        self.comm_metrics = SRCommMetrics("Target", transport)
        self._comm_pending = None
        self.connection_generation = 0
        self._open()

    def _open(self) -> None:
        import zmq

        self.connection_generation = int(self.connection_generation) + 1
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
        logger.info("[CLEAR] Target DEALER connected to %s", self._url)

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

        count = 0
        while True:
            try:
                self._socket.recv_multipart(flags=zmq.NOBLOCK, copy=False)
                count += 1
            except zmq.Again:
                break
            except Exception:
                break
        return count

    def _note_drop(self, reason: str, count: int = 1) -> None:
        if count <= 0 or self.drop_callback is None:
            return
        try:
            self.drop_callback(reason, count)
        except Exception:
            pass

    def send_batch(
        self,
        batch: ClearBatchRequest,
        mm_by_rid: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._finish_comm("abandoned")
        drained = self._drain()
        if drained:
            self.comm_metrics.count(batch.action, "stale_reply", drained)
            self._note_drop("rpc_seq", drained)
            logger.info("[CLEAR] Target drained %d stale frames before send", drained)
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
        if batch.action in (ClearAction.FINISH, ClearAction.ABORT):
            self._finish_comm("send_only")

    def recv_batch(self, session_id: str, rpc_seq: int) -> Optional[ClearBatchReply]:
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
                    "[CLEAR] Target recv timeout session=%s rpc_seq=%s waited_ms=%.3f",
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
            except Exception as exc:
                self._finish_comm("recv_error")
                logger.warning("[CLEAR] Target recv error: %s", exc)
                return None
            try:
                unpack_start = time.perf_counter_ns()
                reply = _unpack_reply(frames)
                unpack_end = time.perf_counter_ns()
            except Exception as exc:
                self._count_comm("unpack_error")
                logger.warning("[CLEAR] Target failed to unpack reply: %s", exc)
                self._note_drop("rpc_seq")
                continue
            if reply.session_id != session_id or reply.rpc_seq != rpc_seq:
                self._count_comm("stale_reply")
                self._note_drop(
                    "session" if reply.session_id != session_id else "rpc_seq"
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


class ClearDraftServer(_CommEndpoint):
    """Draft-side ROUTER."""

    def __init__(self, addr: str, port: str):
        import zmq

        self.addr = addr
        self.port = port
        self._ctx = zmq.Context.instance()
        self._url, transport = clear_endpoint(addr, port, bind=True)
        self.comm_metrics = SRCommMetrics("Draft", transport)
        self._comm_pending = None
        sock = self._ctx.socket(zmq.ROUTER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVHWM, 256)
        sock.setsockopt(zmq.SNDHWM, 256)
        sock.bind(self._url)
        self._socket = sock
        self._last_identity = None
        self.last_recv_queued = False
        logger.info("[CLEAR] Draft ROUTER bound on %s", self._url)

    def close(self) -> None:
        self._finish_comm("closed")
        self.comm_metrics.flush()
        try:
            self._socket.close(linger=0)
        except Exception:
            pass

    def drain(self) -> int:
        import zmq

        count = 0
        while True:
            try:
                self._socket.recv_multipart(flags=zmq.NOBLOCK, copy=False)
                count += 1
            except zmq.Again:
                break
            except Exception:
                break
        if count:
            self.comm_metrics.count("unknown", "stale_request", count)
        return count

    def recv_batch(
        self, timeout_ms: Optional[int] = None
    ) -> Optional[Tuple[ClearBatchRequest, Dict[str, Any]]]:
        import zmq

        self.last_recv_queued = False
        self._finish_comm("abandoned")
        queued = False
        try:
            try:
                frames = self._socket.recv_multipart(flags=zmq.NOBLOCK, copy=False)
                queued = True
            except zmq.Again:
                if timeout_ms is None:
                    self._socket.setsockopt(zmq.RCVTIMEO, -1)
                else:
                    self._socket.setsockopt(zmq.RCVTIMEO, max(1, int(timeout_ms)))
                frames = self._socket.recv_multipart(copy=False)
            received = time.perf_counter_ns()
        except zmq.Again:
            return None
        except Exception as exc:
            self.comm_metrics.record("unknown", "recv_error")
            logger.warning("[CLEAR] Draft recv error: %s", exc)
            return None
        if not frames:
            return None
        self._last_identity = frames[0]
        try:
            unpack_start = time.perf_counter_ns()
            batch, mm = _unpack_request(frames)
            unpack_end = time.perf_counter_ns()
        except Exception as exc:
            self.comm_metrics.record("unknown", "unpack_error")
            logger.warning("[CLEAR] Draft failed to unpack request: %s", exc)
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
        if batch.action in (ClearAction.FINISH, ClearAction.ABORT):
            self._finish_comm("receive_only")
        self.last_recv_queued = queued
        return batch, mm

    def send_batch(self, batch: ClearBatchReply, identity: Optional[Any] = None) -> None:
        ident = identity if identity is not None else self._last_identity
        if ident is None:
            self._finish_comm("send_error")
            logger.warning("[CLEAR] Draft send with no ROUTER identity")
            return
        call = self._comm_pending
        if call is not None and (
            call.key != (batch.session_id, batch.rpc_seq)
            or call.identity != _as_bytes(ident)
        ):
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
