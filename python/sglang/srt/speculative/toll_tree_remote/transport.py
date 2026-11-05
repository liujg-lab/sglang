"""ROUTER/DEALER control channel for TOLL_TREE_REMOTE.

Draft binds ROUTER. Target connects DEALER. Frames are msgspec, not pickle.
This module does not import another speculative algorithm.
"""

from __future__ import annotations

import logging
import time
import uuid

from sglang.srt.speculative.toll_tree_remote.protocol import (
    TollReply,
    TollRequest,
    decode_reply,
    decode_request,
    encode_reply,
    encode_request,
)

logger = logging.getLogger(__name__)

_MAGIC = b"TOLLCTRL"


def _nbytes(frames) -> int:
    total = 0
    for frame in frames:
        if isinstance(frame, (bytes, bytearray)):
            total += len(frame)
        else:
            total += memoryview(frame).nbytes
    return total


class TollDraftServer:
    """Draft-side ROUTER."""

    def __init__(self, url: str):
        import zmq

        self.url = url
        self.bytes_sent = 0
        self.bytes_recv = 0
        self._ctx = zmq.Context.instance()
        sock = self._ctx.socket(zmq.ROUTER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(url)
        self._socket = sock
        self._identity = None

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None

    def recv_request(self, timeout_ms: int = 5000) -> TollRequest | None:
        import zmq

        self._socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        try:
            frames = self._socket.recv_multipart()
        except zmq.Again:
            return None
        self.bytes_recv += _nbytes(frames)
        if len(frames) < 3 or frames[1] != _MAGIC:
            return None
        self._identity = frames[0]
        return decode_request(frames[2])

    def send_reply(self, reply: TollReply) -> None:
        if self._identity is None or self._socket is None:
            raise RuntimeError("draft has no router identity")
        payload = encode_reply(reply)
        frames = [self._identity, _MAGIC, payload]
        self._socket.send_multipart(frames)
        self.bytes_sent += _nbytes(frames)


class TollTargetClient:
    """Target-side DEALER. Stale frames are dropped. Timeout rebuilds the socket."""

    def __init__(self, url: str, cfg, timeout_ms: int = 5000):
        import zmq

        self.url = url
        self.cfg = cfg
        self.timeout_ms = max(1, int(timeout_ms))
        self.bytes_sent = 0
        self.bytes_recv = 0
        self.timeouts = 0
        self.stale_replies = 0
        self._ctx = zmq.Context.instance()
        self._identity = b"toll-target-" + uuid.uuid4().hex[:8].encode()
        self._socket = None
        self._open()

    def _open(self) -> None:
        import zmq

        if self._socket is not None:
            self._socket.close(linger=0)
        sock = self._ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.IDENTITY, self._identity)
        sock.connect(self.url)
        self._socket = sock

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None

    def _drain(self) -> int:
        import zmq

        dropped = 0
        while True:
            try:
                frames = self._socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            dropped += 1
            self.bytes_recv += _nbytes(frames)
        self.stale_replies += dropped
        return dropped

    def send_request(self, request: TollRequest) -> None:
        self._drain()
        payload = encode_request(request)
        frames = [_MAGIC, payload]
        self._socket.send_multipart(frames)
        self.bytes_sent += _nbytes(frames)

    def recv_reply(self, session: str, seq: int) -> TollReply | None:
        import zmq

        deadline = time.perf_counter() + self.timeout_ms / 1000.0
        while True:
            remaining_ms = int((deadline - time.perf_counter()) * 1000)
            if remaining_ms <= 0:
                self.timeouts += 1
                self._open()
                return None
            self._socket.setsockopt(zmq.RCVTIMEO, max(1, remaining_ms))
            try:
                frames = self._socket.recv_multipart()
            except zmq.Again:
                continue
            self.bytes_recv += _nbytes(frames)
            if len(frames) < 2 or frames[0] != _MAGIC:
                self.stale_replies += 1
                continue
            reply = decode_reply(frames[1], self.cfg)
            if reply.session != session or int(reply.seq) != int(seq):
                self.stale_replies += 1
                continue
            return reply
