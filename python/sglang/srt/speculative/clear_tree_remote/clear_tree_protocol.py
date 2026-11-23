"""Wire messages for CLEAR_TREE_REMOTE.

These types are not the standalone-remote messages. A peer that still speaks
that protocol cannot decode a ClearTree batch. ``verify_samples`` rides on
the batch request: each item is ``[batch, k, kv_bucket, device_us]``.
``target_verify_*`` names one finished verify by the reply rpc frozen at its start.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from sglang.srt.sampling.sampling_params import SamplingParams

# Only a real int equal to the version is this protocol. Missing, bool, float, and str stay unsupported.
# 2: PREFILL/snapshot may carry the original image bytes instead of pixels.
CLEAR_PROTOCOL_VERSION = 2


def supported_clear_protocol(version: Optional[int]) -> bool:
    """Only a real integer equal to the version. Float and bool must not compare equal."""
    return type(version) is int and version == CLEAR_PROTOCOL_VERSION


def parse_protocol_version(value: Any) -> Optional[int]:
    """Keep a wire integer unchanged. Do not coerce other types into a version."""
    if type(value) is int:
        return value
    return None


def parse_spec_shape(value: Any) -> Optional[Tuple[int, ...]]:
    """Parse a wire spec shape. A present value that is not three ints matches nothing."""
    if value is None:
        return None
    if (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(type(item) is int for item in value)
    ):
        return (value[0], value[1], value[2])
    return ()


_TARGET_INFO_KEYS = (
    "backend",
    "device",
    "tp",
    "model",
    "served_model_name",
    "dtype",
    "quantization",
    "verify_capture",
    "kv_buckets",
    "disable_cuda_graph",
    "attention_backend",
    "page_size",
    "cuda_graph_bs",
    "prefit_batches",
    "rpc_batch_limit",
)


def _target_info_value(value: Any) -> bool:
    """A header value is a string, a real int, a bool, None, or a list of real ints."""
    if value is None or type(value) is str or type(value) is int or type(value) is bool:
        return True
    return isinstance(value, list) and all(type(item) is int for item in value)


def parse_target_info(value: Any) -> Optional[dict]:
    """Keep the Target header keys whose values are strings, ints, bools, None, or int lists."""
    if not isinstance(value, dict):
        return None
    kept = {}
    for key in _TARGET_INFO_KEYS:
        if key not in value:
            continue
        item = value[key]
        if _target_info_value(item):
            kept[key] = item
    return kept


def parse_verify_samples(value: Any) -> List[List[int]]:
    """Keep only four-int samples. Anything else is dropped, not coerced."""
    if not isinstance(value, list):
        return []
    samples: List[List[int]] = []
    for item in value:
        if (
            isinstance(item, (list, tuple))
            and len(item) == 4
            and all(type(part) is int for part in item)
        ):
            samples.append([int(part) for part in item])
    return samples


class ClearAction(Enum):
    PREFILL = "prefill"
    STEP = "step"
    FINISH = "finish"
    ABORT = "abort"
    FIT = "fit"
    CONTROL = "control"


_CONTROL_OPS = ("begin", "commit", "abandon", "status")


def parse_control(value: Any) -> Optional[dict]:
    """Keep one ClearTree control op. Other shapes are dropped, not coerced."""
    if not isinstance(value, dict):
        return None
    op = value.get("op")
    run = value.get("run")
    seq = value.get("seq")
    fingerprint = value.get("fingerprint")
    if op not in _CONTROL_OPS or not isinstance(run, str) or not run:
        return None
    if type(seq) is not int or not isinstance(fingerprint, str):
        return None
    return {"op": op, "run": run, "seq": seq, "fingerprint": fingerprint}


def parse_fit_status(value: Any) -> Optional[dict]:
    """Keep a ClearTree fit status. Other shapes are dropped, not coerced."""
    if not isinstance(value, dict):
        return None
    kept: Dict[str, Any] = {}
    run = value.get("run")
    if isinstance(run, str):
        kept["run"] = run
    seq = value.get("seq")
    if type(seq) is int:
        kept["seq"] = seq
    ready = value.get("ready")
    if type(ready) is bool:
        kept["ready"] = ready
    reason = value.get("reason")
    if isinstance(reason, str):
        kept["reason"] = reason
    state = value.get("state")
    if isinstance(state, str):
        kept["state"] = state
    written = value.get("written")
    if type(written) is bool:
        kept["written"] = written
    for key in ("missing_h", "missing_layers", "skipped"):
        item = value.get(key)
        if isinstance(item, list):
            kept[key] = item
    pairs = value.get("b_to_g")
    if (
        isinstance(pairs, list)
        and all(
            isinstance(pair, (list, tuple))
            and len(pair) == 2
            and type(pair[0]) is int
            and type(pair[1]) is int
            for pair in pairs
        )
    ):
        kept["b_to_g"] = [[int(pair[0]), int(pair[1])] for pair in pairs]
    groups = value.get("refill_groups")
    if isinstance(groups, list) and all(_refill_group_ok(item) for item in groups):
        kept["refill_groups"] = [_refill_group_row(item) for item in groups]
    batches = value.get("prefit_batches")
    if (
        isinstance(batches, list)
        and batches
        and all(type(item) is int and item > 0 for item in batches)
    ):
        kept["prefit_batches"] = [int(item) for item in batches]
    return kept or None


def _refill_group_ok(item) -> bool:
    """A stable scan-group id: warm, v, or w."""
    if not isinstance(item, (list, tuple)) or not item or not isinstance(item[0], str):
        return False
    kind = item[0]
    if kind == "v":
        return len(item) == 4 and all(type(part) is int for part in item[1:])
    if kind == "w":
        return len(item) == 3 and all(type(part) is int for part in item[1:])
    if kind == "warm":
        return len(item) == 2 and type(item[1]) is int
    return False


def _refill_group_row(item) -> list:
    return [item[0], *[int(part) for part in item[1:]]]


_VERIFY_PATHS = ("graph", "eager", "fallback")


def attach_verify_report(batch, report, session_id) -> bool:
    """Copy a frozen verify onto this STEP. A missing rpc is not a report."""
    if not isinstance(report, dict) or type(report.get("rpc_seq")) is not int:
        return False
    session = report.get("session")
    if session not in (None, session_id):
        return False
    batch.target_verify_rpc_seq = int(report["rpc_seq"])
    batch.target_verify_b = report.get("b") if type(report.get("b")) is int else None
    batch.target_verify_k = report.get("k") if type(report.get("k")) is int else None
    bucket = report.get("bucket")
    batch.target_verify_bucket = bucket if type(bucket) is int and bucket > 0 else None
    fallback = report.get("fallback")
    batch.target_verify_fallback = fallback if type(fallback) is bool else None
    path = report.get("path")
    batch.target_verify_path = path if path in _VERIFY_PATHS else None
    return True


class ClearReplyStatus(Enum):
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
    data = dict(val)
    stop_strs = data.pop("stop_strs", None)
    stop_regex_strs = data.pop("stop_regex_strs", None)
    params = SamplingParams(
        **{key: value for key, value in data.items() if value is not None or key == "n"}
    )
    if stop_strs:
        params.stop_strs = stop_strs
    if stop_regex_strs:
        params.stop_regex_strs = stop_regex_strs
    return params


@dataclass
class ClearDraftRequest:
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
    commit_mode: Optional[str] = None
    commit_version: Optional[int] = None
    base_commit_version: Optional[int] = None
    base_output_len: Optional[int] = None
    delta_ids: Optional[List[int]] = None
    requires_mm: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for item in fields(self):
            val = getattr(self, item.name)
            if val is None:
                continue
            if isinstance(val, Enum):
                result[item.name] = val.value
            elif isinstance(val, SamplingParams):
                result[item.name] = {
                    key: value
                    for key, value in _sampling_params_to_dict(val).items()
                    if value is not None
                }
            else:
                result[item.name] = val
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClearDraftRequest":
        kwargs: Dict[str, Any] = {}
        names = {item.name for item in fields(cls)}
        for key, val in data.items():
            if key not in names:
                continue
            if key == "sampling_params" and isinstance(val, dict):
                kwargs[key] = _sampling_params_from_dict(val)
            else:
                kwargs[key] = val
        return cls(**kwargs)


@dataclass
class ClearDraftReply:
    rid: str
    step_id: int = 0
    base_committed_len: int = 0
    draft_tokens: Optional[List[int]] = None
    status: ClearReplyStatus = ClearReplyStatus.OK
    parent_list: Optional[List[int]] = None
    top_scores_index: Optional[List[int]] = None
    tree_version: Optional[int] = None
    ack_commit_version: Optional[int] = None
    ack_output_len: Optional[int] = None
    reason: Optional[str] = None
    spec_shape: Optional[Tuple[int, ...]] = None
    draft_prompt_len: Optional[int] = None
    prompt_compaction: Optional[bool] = None

    def matches(self, pending) -> Tuple[bool, Optional[str]]:
        if self.step_id != pending.step_id:
            return False, "step"
        if self.base_committed_len != pending.base_committed_len:
            return False, "base_len"
        return True, None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for item in fields(self):
            val = getattr(self, item.name)
            if val is None:
                continue
            if isinstance(val, Enum):
                result[item.name] = val.value
            else:
                result[item.name] = val
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClearDraftReply":
        kwargs: Dict[str, Any] = {}
        names = {item.name for item in fields(cls)}
        for key, val in data.items():
            if key not in names:
                continue
            if key == "status" and not isinstance(val, ClearReplyStatus):
                kwargs[key] = ClearReplyStatus(val)
            elif key == "spec_shape":
                kwargs[key] = parse_spec_shape(val)
            else:
                kwargs[key] = val
        return cls(**kwargs)


@dataclass
class ClearBatchRequest:
    session_id: str
    rpc_seq: int
    action: ClearAction
    reqs: List[ClearDraftRequest] = field(default_factory=list)
    protocol_version: Optional[int] = CLEAR_PROTOCOL_VERSION
    spec_shape: Optional[Tuple[int, ...]] = None
    verify_samples: Optional[List[List[int]]] = None
    target_residence_ns: Optional[int] = None
    target_residence_rpc_seq: Optional[int] = None
    target_queued: Optional[bool] = None
    target_info: Optional[dict] = None
    control: Optional[dict] = None
    target_verify_rpc_seq: Optional[int] = None
    target_verify_b: Optional[int] = None
    target_verify_k: Optional[int] = None
    target_verify_bucket: Optional[int] = None
    target_verify_fallback: Optional[bool] = None
    target_verify_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "session_id": self.session_id,
            "rpc_seq": self.rpc_seq,
            "action": self.action.value,
            "reqs": [req.to_dict() for req in self.reqs],
            "protocol_version": self.protocol_version,
        }
        if self.spec_shape is not None:
            result["spec_shape"] = [int(item) for item in self.spec_shape]
        if self.verify_samples:
            result["verify_samples"] = [list(sample) for sample in self.verify_samples]
        if type(self.target_residence_ns) is int and self.target_residence_ns >= 0:
            result["target_residence_ns"] = self.target_residence_ns
        if type(self.target_residence_rpc_seq) is int:
            result["target_residence_rpc_seq"] = self.target_residence_rpc_seq
        if type(self.target_queued) is bool:
            result["target_queued"] = self.target_queued
        info = parse_target_info(self.target_info)
        if info:
            result["target_info"] = info
        control = parse_control(self.control)
        if control:
            result["control"] = control
        if type(self.target_verify_rpc_seq) is int:
            result["target_verify_rpc_seq"] = self.target_verify_rpc_seq
        if type(self.target_verify_b) is int and self.target_verify_b > 0:
            result["target_verify_b"] = self.target_verify_b
        if type(self.target_verify_k) is int and self.target_verify_k > 0:
            result["target_verify_k"] = self.target_verify_k
        if type(self.target_verify_bucket) is int and self.target_verify_bucket > 0:
            result["target_verify_bucket"] = self.target_verify_bucket
        if type(self.target_verify_fallback) is bool:
            result["target_verify_fallback"] = self.target_verify_fallback
        if self.target_verify_path in ("graph", "eager", "fallback"):
            result["target_verify_path"] = self.target_verify_path
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClearBatchRequest":
        action = data.get("action", ClearAction.STEP)
        if not isinstance(action, ClearAction):
            action = ClearAction(action)
        reqs = [
            ClearDraftRequest.from_dict(item) if isinstance(item, dict) else item
            for item in (data.get("reqs") or [])
        ]
        samples = (
            parse_verify_samples(data["verify_samples"])
            if "verify_samples" in data
            else None
        )
        residence = data.get("target_residence_ns")
        residence_seq = data.get("target_residence_rpc_seq")
        queued = data.get("target_queued")
        verify_rpc = data.get("target_verify_rpc_seq")
        verify_b = data.get("target_verify_b")
        verify_k = data.get("target_verify_k")
        verify_bucket = data.get("target_verify_bucket")
        verify_fallback = data.get("target_verify_fallback")
        verify_path = data.get("target_verify_path")
        return cls(
            session_id=data["session_id"],
            rpc_seq=int(data["rpc_seq"]),
            action=action,
            reqs=reqs,
            protocol_version=parse_protocol_version(data.get("protocol_version", None)),
            spec_shape=(
                parse_spec_shape(data["spec_shape"]) if "spec_shape" in data else None
            ),
            verify_samples=samples,
            target_residence_ns=(
                residence if type(residence) is int and residence >= 0 else None
            ),
            target_residence_rpc_seq=(
                residence_seq if type(residence_seq) is int else None
            ),
            target_queued=queued if type(queued) is bool else None,
            target_info=parse_target_info(data.get("target_info")),
            control=parse_control(data.get("control")),
            target_verify_rpc_seq=verify_rpc if type(verify_rpc) is int else None,
            target_verify_b=verify_b if type(verify_b) is int and verify_b > 0 else None,
            target_verify_k=verify_k if type(verify_k) is int and verify_k > 0 else None,
            target_verify_bucket=(
                verify_bucket if type(verify_bucket) is int and verify_bucket > 0 else None
            ),
            target_verify_fallback=verify_fallback if type(verify_fallback) is bool else None,
            target_verify_path=(
                verify_path if verify_path in ("graph", "eager", "fallback") else None
            ),
        )


@dataclass
class ClearBatchReply:
    session_id: str
    rpc_seq: int
    reqs: List[ClearDraftReply] = field(default_factory=list)
    draft_residence_ns: Optional[int] = None
    protocol_version: Optional[int] = CLEAR_PROTOCOL_VERSION
    fit_ready: Optional[bool] = None
    fit_status: Optional[dict] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "session_id": self.session_id,
            "rpc_seq": self.rpc_seq,
            "reqs": [req.to_dict() for req in self.reqs],
            "protocol_version": self.protocol_version,
        }
        if type(self.draft_residence_ns) is int and self.draft_residence_ns >= 0:
            result["draft_residence_ns"] = self.draft_residence_ns
        if type(self.fit_ready) is bool:
            result["fit_ready"] = self.fit_ready
        status = parse_fit_status(self.fit_status)
        if status:
            result["fit_status"] = status
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClearBatchReply":
        reqs = [
            ClearDraftReply.from_dict(item) if isinstance(item, dict) else item
            for item in (data.get("reqs") or [])
        ]
        ready = data.get("fit_ready")
        return cls(
            session_id=data["session_id"],
            rpc_seq=int(data["rpc_seq"]),
            reqs=reqs,
            draft_residence_ns=data.get("draft_residence_ns"),
            protocol_version=parse_protocol_version(data.get("protocol_version", None)),
            fit_ready=ready if type(ready) is bool else None,
            fit_status=parse_fit_status(data.get("fit_status")),
        )


def clear_batch_reply(reply) -> ClearBatchReply:
    """Copy a draft reply onto the ClearTree wire. The version is always 1."""
    if isinstance(reply, ClearBatchReply):
        reply.protocol_version = CLEAR_PROTOCOL_VERSION
        return reply
    reqs = []
    for item in list(getattr(reply, "reqs", None) or []):
        status = getattr(item, "status", ClearReplyStatus.OK)
        if not isinstance(status, ClearReplyStatus):
            status = ClearReplyStatus(getattr(status, "value", status))
        reqs.append(
            ClearDraftReply(
                rid=item.rid,
                step_id=item.step_id,
                base_committed_len=item.base_committed_len,
                draft_tokens=item.draft_tokens,
                status=status,
                parent_list=item.parent_list,
                top_scores_index=item.top_scores_index,
                tree_version=item.tree_version,
                ack_commit_version=item.ack_commit_version,
                ack_output_len=item.ack_output_len,
                reason=item.reason,
                spec_shape=item.spec_shape,
                draft_prompt_len=item.draft_prompt_len,
                prompt_compaction=item.prompt_compaction,
            )
        )
    return ClearBatchReply(
        session_id=reply.session_id,
        rpc_seq=reply.rpc_seq,
        reqs=reqs,
        protocol_version=CLEAR_PROTOCOL_VERSION,
    )


def protocol_mismatch_reply(batch: ClearBatchRequest) -> ClearBatchReply:
    return ClearBatchReply(
        session_id=batch.session_id,
        rpc_seq=batch.rpc_seq,
        protocol_version=CLEAR_PROTOCOL_VERSION,
        reqs=[
            ClearDraftReply(
                rid=req.rid,
                step_id=req.step_id,
                base_committed_len=req.base_committed_len,
                draft_tokens=[],
                status=ClearReplyStatus.REJECT,
                reason="protocol_mismatch",
            )
            for req in batch.reqs
        ],
    )


def as_clear_action(action) -> ClearAction:
    if isinstance(action, ClearAction):
        return action
    return ClearAction(getattr(action, "value", action))


def action_value(action) -> str:
    return str(getattr(action, "value", action))
