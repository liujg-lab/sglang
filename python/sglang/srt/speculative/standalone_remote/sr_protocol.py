"""Protocol types for STANDALONE_REMOTE sync RPC.

Identity fields (session_id / rpc_seq / step_id / base_committed_len) are
first-class members. Sync mutual-wait does not remove the need to reject
late replies from a previous RPC after a timeout.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from sglang.srt.sampling.sampling_params import SamplingParams


class SRAction(Enum):
    PREFILL = "prefill"
    STEP = "step"
    FINISH = "finish"
    ABORT = "abort"


class SRReplyStatus(Enum):
    OK = "ok"
    EMPTY = "empty"
    IDEMPOTENT = "idempotent"
    REJECT = "reject"


def _sampling_params_to_dict(sampling_params: SamplingParams) -> Dict[str, Any]:
    stop_token_ids = (
        list(sampling_params.stop_token_ids) if sampling_params.stop_token_ids else None
    )
    return {
        "max_new_tokens": sampling_params.max_new_tokens,
        "stop_strs": sampling_params.stop_strs,
        "stop_token_ids": stop_token_ids,
        "stop_regex_strs": sampling_params.stop_regex_strs,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "top_k": sampling_params.top_k,
        "min_p": sampling_params.min_p,
        "frequency_penalty": sampling_params.frequency_penalty,
        "presence_penalty": sampling_params.presence_penalty,
        "repetition_penalty": sampling_params.repetition_penalty,
        "min_new_tokens": sampling_params.min_new_tokens,
        "n": sampling_params.n,
        "json_schema": sampling_params.json_schema,
        "regex": sampling_params.regex,
        "ebnf": sampling_params.ebnf,
        "structural_tag": sampling_params.structural_tag,
        "ignore_eos": sampling_params.ignore_eos,
        "skip_special_tokens": sampling_params.skip_special_tokens,
        "spaces_between_special_tokens": sampling_params.spaces_between_special_tokens,
        "no_stop_trim": sampling_params.no_stop_trim,
        "custom_params": sampling_params.custom_params,
        "stream_interval": sampling_params.stream_interval,
        "logit_bias": sampling_params.logit_bias,
        "sampling_seed": sampling_params.sampling_seed,
    }


def _sampling_params_from_dict(val: Dict[str, Any]) -> SamplingParams:
    d = dict(val)
    stop_strs = d.pop("stop_strs", None)
    stop_regex_strs = d.pop("stop_regex_strs", None)
    sp = SamplingParams(**{k: v for k, v in d.items() if v is not None or k == "n"})
    if stop_strs:
        sp.stop_strs = stop_strs
    if stop_regex_strs:
        sp.stop_regex_strs = stop_regex_strs
    return sp


@dataclass
class SRPendingEntry:
    step_id: int
    base_committed_len: int


@dataclass
class SRDraftRequest:
    rid: str
    step_id: int = 0
    base_committed_len: int = 0
    committed_ids: Optional[List[int]] = None
    num_draft_tokens: int = 1
    padded_input_ids: Optional[List[int]] = None
    sampling_params: Optional[SamplingParams] = None
    has_mm: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if val is None:
                continue
            if isinstance(val, Enum):
                result[f.name] = val.value
            elif isinstance(val, SamplingParams):
                result[f.name] = {
                    k: v
                    for k, v in _sampling_params_to_dict(val).items()
                    if v is not None
                }
            else:
                result[f.name] = val
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SRDraftRequest":
        kwargs: Dict[str, Any] = {}
        names = {f.name for f in fields(cls)}
        for k, val in d.items():
            if k not in names:
                continue
            if k == "sampling_params" and isinstance(val, dict):
                kwargs[k] = _sampling_params_from_dict(val)
            else:
                kwargs[k] = val
        return cls(**kwargs)


@dataclass
class SRDraftReply:
    rid: str
    step_id: int = 0
    base_committed_len: int = 0
    draft_tokens: Optional[List[int]] = None
    status: SRReplyStatus = SRReplyStatus.OK
    parent_list: Optional[List[int]] = None
    top_scores_index: Optional[List[int]] = None

    def matches(self, pending: SRPendingEntry) -> Tuple[bool, Optional[str]]:
        if self.step_id != pending.step_id:
            return False, "step"
        if self.base_committed_len != pending.base_committed_len:
            return False, "base_len"
        return True, None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if val is None:
                continue
            if isinstance(val, Enum):
                result[f.name] = val.value
            else:
                result[f.name] = val
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SRDraftReply":
        kwargs: Dict[str, Any] = {}
        names = {f.name for f in fields(cls)}
        for k, val in d.items():
            if k not in names:
                continue
            if k == "status" and not isinstance(val, SRReplyStatus):
                kwargs[k] = SRReplyStatus(val)
            else:
                kwargs[k] = val
        return cls(**kwargs)


@dataclass
class SRBatchRequest:
    session_id: str
    rpc_seq: int
    action: SRAction
    reqs: List[SRDraftRequest] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "rpc_seq": self.rpc_seq,
            "action": self.action.value,
            "reqs": [r.to_dict() for r in self.reqs],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SRBatchRequest":
        action = d.get("action", SRAction.STEP)
        if not isinstance(action, SRAction):
            action = SRAction(action)
        reqs = [
            SRDraftRequest.from_dict(x) if isinstance(x, dict) else x
            for x in (d.get("reqs") or [])
        ]
        return cls(
            session_id=d["session_id"],
            rpc_seq=int(d["rpc_seq"]),
            action=action,
            reqs=reqs,
        )


@dataclass
class SRBatchReply:
    session_id: str
    rpc_seq: int
    reqs: List[SRDraftReply] = field(default_factory=list)
    draft_residence_ns: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "session_id": self.session_id,
            "rpc_seq": self.rpc_seq,
            "reqs": [r.to_dict() for r in self.reqs],
        }
        if type(self.draft_residence_ns) is int and self.draft_residence_ns >= 0:
            result["draft_residence_ns"] = self.draft_residence_ns
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SRBatchReply":
        reqs = [
            SRDraftReply.from_dict(x) if isinstance(x, dict) else x
            for x in (d.get("reqs") or [])
        ]
        return cls(
            session_id=d["session_id"],
            rpc_seq=int(d["rpc_seq"]),
            reqs=reqs,
            # Preserve malformed optional telemetry for transport validation;
            # it must never prevent an otherwise valid reply from being used.
            draft_residence_ns=d.get("draft_residence_ns"),
        )


def is_health_check_req(req) -> bool:
    return getattr(req, "rid", "").startswith("HEALTH_CHECK")
