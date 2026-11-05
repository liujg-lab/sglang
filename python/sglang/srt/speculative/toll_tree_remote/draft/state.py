"""Per-request draft state. A new session drops every rid."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class DraftReqState:
    rid: str
    session: str
    prefix_len: int
    commit_len: int
    last_seen: float = field(default_factory=time.time)


class DraftStateStore:
    def __init__(self, ttl_s: float = 60.0) -> None:
        self.ttl_s = float(ttl_s)
        self.session: str | None = None
        self.reqs: dict[str, DraftReqState] = {}

    def begin_session(self, session: str) -> None:
        if self.session is not None and session != self.session:
            self.reqs.clear()
        self.session = session

    def get(self, rid: str) -> DraftReqState | None:
        state = self.reqs.get(rid)
        if state is not None:
            state.last_seen = time.time()
        return state

    def upsert(self, rid: str, prefix_len: int, commit_len: int) -> DraftReqState:
        state = DraftReqState(
            rid=rid,
            session=self.session or "",
            prefix_len=int(prefix_len),
            commit_len=int(commit_len),
        )
        self.reqs[rid] = state
        return state

    def delete(self, rid: str) -> None:
        self.reqs.pop(rid, None)

    def expire(self, now: float | None = None, keep: set[str] | None = None) -> list[str]:
        if self.ttl_s <= 0:
            return []
        stamp = time.time() if now is None else float(now)
        kept = keep or set()
        stale = [
            rid
            for rid, state in self.reqs.items()
            if rid not in kept and stamp - state.last_seen > self.ttl_s
        ]
        for rid in stale:
            self.reqs.pop(rid, None)
        return stale
