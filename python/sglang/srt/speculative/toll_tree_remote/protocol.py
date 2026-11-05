"""Msgspec control frames for TOLL_TREE_REMOTE. No pickle."""

from __future__ import annotations

import msgspec

PROTO_VERSION = 1


class TollAction:
    PREFILL = "PREFILL"
    STEP = "STEP"
    FINISH = "FINISH"
    ABORT = "ABORT"


class TollStatus:
    OK = "OK"
    EMPTY = "EMPTY"
    REJECT = "REJECT"
    NEED_SNAPSHOT = "NEED_SNAPSHOT"


class TollReqItem(msgspec.Struct, kw_only=True):
    rid: str
    commit_len: int
    new_tokens: list[int]
    input_ids: list[int]
    sampling: dict[str, float]


class TollRequest(msgspec.Struct, kw_only=True):
    proto: int
    session: str
    seq: int
    action: str
    spec_shape: list
    t_target_ns: int
    reqs: list[TollReqItem]


class TollReplyItem(msgspec.Struct, kw_only=True):
    rid: str
    base_len: int
    tokens: list[int]
    parents: list[int]
    status: str


class TollReply(msgspec.Struct, kw_only=True):
    proto: int
    session: str
    seq: int
    verify_bucket: int
    residence_ns: int
    reqs: list[TollReplyItem]


def spec_shape_of(cfg) -> list:
    """``(width buckets, verify buckets, max depth)`` as one JSON-like list."""
    return [list(cfg.width_buckets), list(cfg.verify_buckets), int(cfg.max_depth)]


def shapes_match(spec_shape, cfg) -> bool:
    if not isinstance(spec_shape, (list, tuple)) or len(spec_shape) != 3:
        return False
    widths, verifies, depth = spec_shape
    return (
        [int(v) for v in widths] == list(cfg.width_buckets)
        and [int(v) for v in verifies] == list(cfg.verify_buckets)
        and int(depth) == int(cfg.max_depth)
    )


def _empty_reply(session: str, seq: int, bucket: int, reqs=None) -> TollReply:
    items = []
    for item in reqs or []:
        items.append(
            TollReplyItem(
                rid=str(getattr(item, "rid", "")),
                base_len=int(getattr(item, "base_len", 0)),
                tokens=[],
                parents=[],
                status=TollStatus.EMPTY,
            )
        )
    return TollReply(
        proto=PROTO_VERSION,
        session=session,
        seq=int(seq),
        verify_bucket=int(bucket),
        residence_ns=0,
        reqs=items,
    )


def encode_request(request: TollRequest) -> bytes:
    return msgspec.msgpack.encode(request)


def encode_reply(reply: TollReply) -> bytes:
    return msgspec.msgpack.encode(reply)


def decode_request(payload: bytes) -> TollRequest:
    return msgspec.msgpack.decode(payload, type=TollRequest)


def request_ok(request: TollRequest, cfg) -> bool:
    if int(request.proto) != PROTO_VERSION:
        return False
    if request.action not in (
        TollAction.PREFILL,
        TollAction.STEP,
        TollAction.FINISH,
        TollAction.ABORT,
    ):
        return False
    return shapes_match(request.spec_shape, cfg)


def decode_reply(payload: bytes, cfg) -> TollReply:
    """Decode one reply. A bad frame becomes an empty draft."""
    bucket = int(cfg.verify_buckets[0])
    try:
        reply = msgspec.msgpack.decode(payload, type=TollReply)
    except (msgspec.DecodeError, msgspec.ValidationError):
        return _empty_reply("", -1, bucket)
    if int(reply.proto) != PROTO_VERSION or int(reply.verify_bucket) not in cfg.verify_buckets:
        return _empty_reply(reply.session, reply.seq, bucket, reply.reqs)
    for item in reply.reqs:
        if len(item.tokens) != len(item.parents) or len(item.tokens) > int(reply.verify_bucket):
            return _empty_reply(reply.session, reply.seq, bucket, reply.reqs)
        for index, parent in enumerate(item.parents):
            if int(parent) != -1 and not (0 <= int(parent) < index):
                return _empty_reply(reply.session, reply.seq, bucket, reply.reqs)
        if item.status not in (
            TollStatus.OK,
            TollStatus.EMPTY,
            TollStatus.REJECT,
            TollStatus.NEED_SNAPSHOT,
        ):
            return _empty_reply(reply.session, reply.seq, bucket, reply.reqs)
    return reply
