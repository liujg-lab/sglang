import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

SRWindow = Tuple[
    List[int],
    Optional[List[int]],
    Optional[List[int]],
]


@dataclass
class SRDraftState:
    req_id: str
    session_id: str
    last_step_id: int = -1
    last_base_committed_len: int = -1
    last_rpc_seq: int = -1
    last_window: Optional[SRWindow] = None
    req_object: Optional["Req"] = None
    last_updated_time: float = field(default_factory=time.time)
    created_time: float = field(default_factory=time.time)


class SRDraftStateManager:
    def __init__(self) -> None:
        self.active: Dict[str, SRDraftState] = {}
        self._lock = threading.Lock()
        self.session_id: Optional[str] = None

    def get(self, req_id: str) -> Optional[SRDraftState]:
        with self._lock:
            return self.active.get(req_id)

    def set(self, req_id: str, state: SRDraftState) -> None:
        with self._lock:
            self.active[req_id] = state

    def delete(self, req_id: str) -> Optional[SRDraftState]:
        with self._lock:
            return self.active.pop(req_id, None)

    def exists(self, req_id: str) -> bool:
        with self._lock:
            return req_id in self.active

    def clear(self) -> List[SRDraftState]:
        with self._lock:
            states = list(self.active.values())
            self.active.clear()
            return states
