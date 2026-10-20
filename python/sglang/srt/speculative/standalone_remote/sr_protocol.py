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

# Only a real int 2 is the v2 protocol. Missing, bool, float, and str stay unsupported.
SR_PROTOCOL_VERSION = 2


def parse_protocol_version(value: Any) -> Optional[int]:
    """Keep a wire integer unchanged. Do not coerce other types into a version."""
    if type(value) is int:
        return value
    return None


def local_spec_shape(server_args) -> Tuple[int, int, int]:
    """``(topk, num_steps, num_draft_tokens)`` from one process's server args.

    Matches the widths the draft tree, verify kernel, and packet stride use.
    A missing draft width becomes ``steps + 1``.
    """
    topk = max(1, int(getattr(server_args, "speculative_eagle_topk", 1) or 1))
    steps = max(1, int(getattr(server_args, "speculative_num_steps", 1) or 1))
    width = int(getattr(server_args, "speculative_num_draft_tokens", 0) or 0)
    if width <= 0:
        width = steps + 1
    return (topk, steps, width)


def make_draft_tokens_and_logits(
    tokens: Optional[List[int]],
    parent_list: Optional[List[int]] = None,
    top_scores_index: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Host draft row stored on a request.

    Tokens stay a list. ``cur_drafts`` already holds the same ids, so this
    does not build a tensor. An empty row is a fresh ``[0]`` sentinel, not a
    shared list.
    """
    row = list(tokens) if tokens else [0]
    return {
        "draft_tokens": row,
        "parent_list": parent_list,
        "top_scores_index": top_scores_index,
    }


def parse_spec_shape(value: Any) -> Optional[Tuple[int, ...]]:
    """Parse a wire spec shape.

    ``None`` means the peer omitted the field. A present value that is not
    three real ints becomes an empty tuple so it cannot match a local shape.
    """
    if value is None:
        return None
    if (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(type(item) is int for item in value)
    ):
        return (value[0], value[1], value[2])
    return ()


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
    NEED_SNAPSHOT = "need_snapshot"


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
    commit_version: Optional[int] = None
    sent_output_len: Optional[int] = None


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
    commit_tree_version: Optional[int] = None
    commit_tree_base_committed_len: Optional[int] = None
    commit_candidate_indices: Optional[List[int]] = None
    # Incremental commit. ``commit_mode`` is "snapshot" or "delta".
    commit_mode: Optional[str] = None
    commit_version: Optional[int] = None
    base_commit_version: Optional[int] = None
    base_output_len: Optional[int] = None
    delta_ids: Optional[List[int]] = None
    requires_mm: bool = False
    # Optional per-request Draft vision keep ratio. None uses the Draft flag.
    draft_vision_keep_ratio: Optional[float] = None

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
    tree_version: Optional[int] = None
    ack_commit_version: Optional[int] = None
    ack_output_len: Optional[int] = None
    reason: Optional[str] = None
    # Draft local (topk, num_steps, num_draft_tokens), set on spec_shape_mismatch.
    spec_shape: Optional[Tuple[int, ...]] = None
    # Set only on a PREFILL reply when Draft shortened its own vision prompt.
    draft_prompt_len: Optional[int] = None
    prompt_compaction: Optional[bool] = None

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
            elif k == "spec_shape":
                kwargs[k] = parse_spec_shape(val)
            else:
                kwargs[k] = val
        return cls(**kwargs)


@dataclass
class SRBatchRequest:
    session_id: str
    rpc_seq: int
    action: SRAction
    reqs: List[SRDraftRequest] = field(default_factory=list)
    # Local batches default to v2. from_dict passes an explicit value so a
    # missing or non-int wire field stays unsupported instead of becoming 2.
    protocol_version: Optional[int] = SR_PROTOCOL_VERSION
    # (topk, num_steps, num_draft_tokens). v2 Target always sends this.
    # Draft rejects PREFILL/STEP when it is missing.
    spec_shape: Optional[Tuple[int, ...]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "session_id": self.session_id,
            "rpc_seq": self.rpc_seq,
            "action": self.action.value,
            "reqs": [r.to_dict() for r in self.reqs],
            "protocol_version": self.protocol_version,
        }
        if self.spec_shape is not None:
            result["spec_shape"] = [int(x) for x in self.spec_shape]
        return result

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
            protocol_version=parse_protocol_version(d.get("protocol_version", None)),
            spec_shape=(
                parse_spec_shape(d["spec_shape"]) if "spec_shape" in d else None
            ),
        )


@dataclass
class SRBatchReply:
    session_id: str
    rpc_seq: int
    reqs: List[SRDraftReply] = field(default_factory=list)
    draft_residence_ns: Optional[int] = None
    protocol_version: Optional[int] = SR_PROTOCOL_VERSION

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "session_id": self.session_id,
            "rpc_seq": self.rpc_seq,
            "reqs": [r.to_dict() for r in self.reqs],
            "protocol_version": self.protocol_version,
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
            protocol_version=parse_protocol_version(d.get("protocol_version", None)),
        )


def is_health_check_req(req) -> bool:
    return getattr(req, "rid", "").startswith("HEALTH_CHECK")


def account_sr_verify_step(batch) -> None:
    """Count one verify step. A second call in the same round does nothing.

    ``len_output_ids`` is the accepted output length. Callers that send STEP
    early must account before the send; the late path accounts only if this
    round has not already done so.
    """
    if getattr(batch, "sr_step_accounted", False):
        return
    for req in getattr(batch, "reqs", None) or ():
        if is_health_check_req(req):
            continue
        req.spec_cnt = int(getattr(req, "spec_cnt", 0) or 0) + 1
        req.len_output_ids = len(req.output_ids)
        req.sr_step_id = int(getattr(req, "sr_step_id", 0) or 0) + 1
    batch.sr_step_accounted = True
