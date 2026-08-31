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
from typing import Any, Callable, Dict, List, Optional, Tuple

from sglang.srt.speculative.standalone_remote.sr_protocol import (
    SRBatchReply,
    SRBatchRequest,
)

logger = logging.getLogger(__name__)

_CTRL_FRAME = b"SRCTRL"
_MM_FRAME = b"SRMM"

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


class SRTargetClient:
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
        self._url, _ = sr_endpoint(addr, port, bind=False)
        self._open()

    def _open(self) -> None:
        import zmq

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
        drained = self._drain()
        if drained:
            self._note_drop("rpc_seq")
            note_stale_drop("rpc_seq")
            logger.info("[SR] Target drained %d stale frames before send", drained)
        frames = _pack_request(batch, mm_by_rid)
        self._socket.send_multipart(frames, copy=False)

    def recv_batch(
        self, session_id: str, rpc_seq: int
    ) -> Optional[SRBatchReply]:
        import zmq

        deadline = time.perf_counter() + self.timeout_ms / 1000.0
        while True:
            remaining_ms = int((deadline - time.perf_counter()) * 1000)
            if remaining_ms <= 0:
                if self.rebuild_on_timeout:
                    self._open()
                logger.warning(
                    "[SR] Target recv timeout session=%s rpc_seq=%s",
                    session_id,
                    rpc_seq,
                )
                return None
            self._socket.setsockopt(zmq.RCVTIMEO, max(1, remaining_ms))
            try:
                frames = self._socket.recv_multipart(copy=False)
            except zmq.Again:
                continue
            except Exception as e:
                logger.warning("[SR] Target recv error: %s", e)
                return None
            try:
                reply = _unpack_reply(frames)
            except Exception as e:
                logger.warning("[SR] Target failed to unpack reply: %s", e)
                self._note_drop("rpc_seq")
                continue
            if reply.session_id != session_id:
                self._note_drop("session")
                note_stale_drop("session")
                logger.info(
                    "[SR] drop stale reply session=%s expected=%s",
                    reply.session_id,
                    session_id,
                )
                continue
            if reply.rpc_seq != rpc_seq:
                self._note_drop("rpc_seq")
                note_stale_drop("rpc_seq")
                logger.info(
                    "[SR] drop stale reply rpc_seq=%s expected=%s",
                    reply.rpc_seq,
                    rpc_seq,
                )
                continue
            return reply


class SRDraftServer:
    """Draft-side ROUTER. Binds and serves blocking recv/send."""

    def __init__(self, addr: str, port: str):
        import zmq

        self.addr = addr
        self.port = port
        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._url, _ = sr_endpoint(addr, port, bind=True)
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
        try:
            self._socket.close(linger=0)
        except Exception:
            pass

    def drain(self) -> int:
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

    def recv_batch(
        self, timeout_ms: Optional[int] = None
    ) -> Optional[Tuple[SRBatchRequest, Dict[str, Any]]]:
        import zmq

        if timeout_ms is None:
            self._socket.setsockopt(zmq.RCVTIMEO, -1)
        else:
            self._socket.setsockopt(zmq.RCVTIMEO, max(1, int(timeout_ms)))
        try:
            frames = self._socket.recv_multipart(copy=False)
        except zmq.Again:
            return None
        except Exception as e:
            logger.warning("[SR] Draft recv error: %s", e)
            return None
        if not frames:
            return None
        self._last_identity = frames[0]
        try:
            batch, mm = _unpack_request(frames)
        except Exception as e:
            logger.warning("[SR] Draft failed to unpack request: %s", e)
            return None
        return batch, mm

    def is_stale(self, batch: SRBatchRequest) -> bool:
        if self.last_session_id is None:
            return False
        if batch.session_id < self.last_session_id:
            return True
        if (
            batch.session_id == self.last_session_id
            and batch.rpc_seq <= self.last_rpc_seq
        ):
            return True
        return False

    def remember(self, batch: SRBatchRequest) -> None:
        self.last_session_id = batch.session_id
        self.last_rpc_seq = batch.rpc_seq

    def send_batch(self, batch: SRBatchReply, identity: Optional[Any] = None) -> None:
        ident = identity if identity is not None else self._last_identity
        if ident is None:
            logger.warning("[SR] Draft send with no ROUTER identity")
            return
        frames = [ident, *_pack_reply(batch)]
        self._socket.send_multipart(frames, copy=False)


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
