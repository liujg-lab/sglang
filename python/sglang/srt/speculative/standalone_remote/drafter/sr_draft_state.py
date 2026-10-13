import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from sglang.srt.speculative.standalone_remote.sr_commit import (
    LocalPrefixStamp,
    PendingCommit,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

SRWindow = Tuple[
    List[int],
    Optional[List[int]],
    Optional[List[int]],
]


def freeze_ids(seq: Optional[List[int]]) -> Optional[Tuple[int, ...]]:
    if seq is None:
        return None
    return tuple(int(x) for x in seq)


def freeze_tokens(seq: Optional[List[int]]) -> Tuple[int, ...]:
    if not seq:
        return ()
    return tuple(int(x) for x in seq)


def thaw_ids(seq: Optional[Tuple[int, ...]]) -> Optional[List[int]]:
    if seq is None:
        return None
    return list(seq)


@dataclass(frozen=True)
class SRCachedReply:
    """One confirmed draft window. ``None`` on the state means no cache.

    An empty ``draft_tokens`` is a cached empty window, not a missing cache.
    """

    draft_tokens: Tuple[int, ...] = ()
    parent_list: Optional[Tuple[int, ...]] = None
    top_scores_index: Optional[Tuple[int, ...]] = None
    tree_version: Optional[int] = None
    ack_commit_version: Optional[int] = None
    ack_output_len: Optional[int] = None


@dataclass
class SRDraftState:
    req_id: str
    session_id: str
    last_step_id: int = -1
    last_base_committed_len: int = -1
    cached_reply: Optional[SRCachedReply] = None
    req_object: Optional["Req"] = None
    # Sticky: this rid hit unrecoverable Draft state (half-ingested KV, broken
    # VL tensors). Reply REJECT/degraded with no ACK until Target FINISHes.
    degraded: bool = False
    last_updated_time: float = field(default_factory=time.time)
    # Authoritative committed output, without prompt and without draft tokens.
    acked_output_ids: List[int] = field(default_factory=list)
    acked_version: Optional[int] = None
    prompt_len: int = 0
    commit_trusted: bool = True
    completion_unknown: bool = False
    last_commit_fingerprint: Optional[Tuple] = None
    last_num_draft_tokens: int = 0
    local_prefix_stamp: Optional[LocalPrefixStamp] = None
    pending_commit: Optional[PendingCommit] = None


class SRDraftStateManager:
    def __init__(self, timeout_threshold: float = 60.0) -> None:
        self.active: Dict[str, SRDraftState] = {}
        self._lock = threading.Lock()
        self.session_id: Optional[str] = None
        self.timeout_threshold = float(timeout_threshold)

    def get(self, req_id: str) -> Optional[SRDraftState]:
        with self._lock:
            return self.active.get(req_id)

    def set(self, req_id: str, state: SRDraftState) -> None:
        with self._lock:
            state.last_updated_time = time.time()
            self.active[req_id] = state

    def delete(self, req_id: str) -> Optional[SRDraftState]:
        with self._lock:
            return self.active.pop(req_id, None)

    def exists(self, req_id: str) -> bool:
        with self._lock:
            return req_id in self.active

    def cleanup_stale_states(
        self,
        timeout: Optional[float] = None,
        keep_rids: Optional[set] = None,
        now: Optional[float] = None,
    ) -> List[SRDraftState]:
        """Pop RPC states idle longer than ``timeout``. ``timeout<=0`` disables."""
        limit = self.timeout_threshold if timeout is None else float(timeout)
        if limit <= 0:
            return []
        ts = time.time() if now is None else float(now)
        keep = keep_rids or set()
        popped: List[SRDraftState] = []
        with self._lock:
            for rid, state in list(self.active.items()):
                if rid in keep:
                    continue
                if ts - state.last_updated_time > limit:
                    popped.append(self.active.pop(rid))
        return popped

    def clear(self) -> List[SRDraftState]:
        with self._lock:
            states = list(self.active.values())
            self.active.clear()
            return states
