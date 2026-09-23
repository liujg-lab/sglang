"""CPU commit cursor for STANDALONE_REMOTE incremental STEP.

The wire delta is the suffix appended since the last ACK. Draft rebuilds the
full output history in memory and then uses the existing align path. A missing
protocol version is the historical protocol and must not be read as version 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Sequence, Tuple

from sglang.srt.speculative.standalone_remote.sr_protocol import (
    SRAction,
    SRReplyStatus,
)

SR_PROTOCOL_VERSION = 2
PROTOCOL_MISMATCH = "protocol_mismatch"

_CONTROL = (SRAction.FINISH, SRAction.ABORT)
_ADVANCE = (SRReplyStatus.OK, SRReplyStatus.EMPTY, SRReplyStatus.IDEMPOTENT)


class CommitOutcome(Enum):
    APPLY = "apply"
    CACHE = "cache"
    REJECT = "reject"
    NEED_SNAPSHOT = "need_snapshot"
    FORCE_RESET = "force_reset"
    CONTROL = "control"


class RecoveryRoute(Enum):
    ALIGN = "align"
    REPREFILL = "reprefill"
    BLOCKED = "blocked"


@dataclass
class CommitVerdict:
    outcome: CommitOutcome
    reason: Optional[str] = None
    output_ids: Optional[List[int]] = None


def supported_protocol(version: Optional[int]) -> bool:
    return version == SR_PROTOCOL_VERSION


def reply_stops_speculation(
    protocol_version: Optional[int], *, session_rpc_matched: bool
) -> bool:
    """Old or missing versions stop the session only after session/rpc match."""
    if not session_rpc_matched:
        return False
    return not supported_protocol(protocol_version)


def status_advances_cursor(status: SRReplyStatus) -> bool:
    return status in _ADVANCE


def ack_matches(reply, pending) -> Tuple[bool, Optional[str]]:
    if getattr(reply, "ack_commit_version", None) != getattr(
        pending, "commit_version", None
    ):
        return False, "ack_version"
    if getattr(reply, "ack_output_len", None) != getattr(
        pending, "sent_output_len", None
    ):
        return False, "ack_len"
    return True, None


def _as_tuple(value: Optional[Sequence[int]]) -> Optional[Tuple[int, ...]]:
    if value is None:
        return None
    return tuple(int(x) for x in value)


def _sampling_key(sampling_params: Any) -> Optional[Tuple]:
    if sampling_params is None:
        return None
    if isinstance(sampling_params, dict):
        items = sampling_params.items()
    else:
        from sglang.srt.speculative.standalone_remote.sr_protocol import (
            _sampling_params_to_dict,
        )

        items = _sampling_params_to_dict(sampling_params).items()
    return tuple(sorted((str(k), repr(v)) for k, v in items))


def commit_fingerprint(req) -> Tuple:
    """Raw commit content. Excludes rpc_seq and any locally rebuilt history."""
    return (
        req.rid,
        int(req.step_id),
        int(req.base_committed_len),
        int(getattr(req, "num_draft_tokens", 0) or 0),
        bool(getattr(req, "has_mm", False)),
        bool(getattr(req, "requires_mm", False)),
        getattr(req, "commit_mode", None),
        getattr(req, "commit_version", None),
        getattr(req, "base_commit_version", None),
        getattr(req, "base_output_len", None),
        _as_tuple(getattr(req, "delta_ids", None)),
        _as_tuple(getattr(req, "committed_ids", None)),
        _as_tuple(getattr(req, "padded_input_ids", None)),
        _sampling_key(getattr(req, "sampling_params", None)),
        getattr(req, "commit_tree_version", None),
        getattr(req, "commit_tree_base_committed_len", None),
        _as_tuple(getattr(req, "commit_candidate_indices", None)),
    )


def _delta_fields_present(req) -> bool:
    return any(
        getattr(req, name, None) is not None
        for name in ("delta_ids", "base_commit_version", "base_output_len")
    )


def structural_error(req, action: SRAction) -> Optional[str]:
    if action in _CONTROL:
        if (
            getattr(req, "commit_mode", None) is not None
            or req.committed_ids is not None
            or req.padded_input_ids is not None
            or req.sampling_params is not None
            or _delta_fields_present(req)
        ):
            return "commit_on_control"
        return None
    mode = getattr(req, "commit_mode", None)
    if mode not in ("snapshot", "delta"):
        return "commit_mode"
    if getattr(req, "commit_version", None) is None:
        return "commit_version"
    if mode == "snapshot":
        if req.committed_ids is None or req.padded_input_ids is None:
            return "snapshot_context"
        if req.sampling_params is None:
            return "snapshot_sampling"
        if _delta_fields_present(req):
            return "snapshot_has_delta"
        if int(req.base_committed_len) != len(req.padded_input_ids) + len(
            req.committed_ids
        ):
            return "snapshot_length"
        return None
    if req.delta_ids is None or not isinstance(req.delta_ids, list):
        return "delta_ids"
    if getattr(req, "base_commit_version", None) is None:
        return "base_commit_version"
    if getattr(req, "base_output_len", None) is None:
        return "base_output_len"
    if (
        req.committed_ids is not None
        or req.padded_input_ids is not None
        or req.sampling_params is not None
    ):
        return "delta_has_snapshot"
    return None


def route_snapshot_recovery(
    *,
    kv_trusted: bool,
    degraded: bool = False,
    poisoned: bool = False,
    completion_unknown: bool = False,
) -> RecoveryRoute:
    """Cursor mismatch may align. Untrusted KV must not stop on an equal prefix."""
    if degraded or poisoned or completion_unknown:
        return RecoveryRoute.BLOCKED
    if not kv_trusted:
        return RecoveryRoute.REPREFILL
    return RecoveryRoute.ALIGN


def inspect_commit(
    req,
    *,
    action: SRAction,
    current_version: Optional[int],
    current_output: Optional[Sequence[int]],
    prompt_len: int,
    cached_fingerprint: Optional[Tuple],
    has_state: bool,
    state_trusted: bool,
) -> CommitVerdict:
    """Same-version comparison happens before the delta base-version check."""
    error = structural_error(req, action)
    if error is not None:
        return CommitVerdict(CommitOutcome.REJECT, error)
    if action in _CONTROL:
        return CommitVerdict(CommitOutcome.CONTROL)

    version = int(req.commit_version)
    fingerprint = commit_fingerprint(req)
    if current_version is not None and version == int(current_version):
        if cached_fingerprint is not None and fingerprint == cached_fingerprint:
            if not state_trusted:
                if getattr(req, "commit_mode", None) == "snapshot":
                    return CommitVerdict(
                        CommitOutcome.FORCE_RESET,
                        "untrusted_resend",
                        list(req.committed_ids or []),
                    )
                return CommitVerdict(CommitOutcome.NEED_SNAPSHOT, "untrusted_resend")
            return CommitVerdict(
                CommitOutcome.CACHE,
                output_ids=list(current_output or []),
            )
        return CommitVerdict(CommitOutcome.REJECT, "commit_mismatch")
    if current_version is not None and version < int(current_version):
        return CommitVerdict(CommitOutcome.REJECT, "stale_version")

    mode = req.commit_mode
    if mode == "snapshot":
        output = list(req.committed_ids or [])
        if has_state and not state_trusted:
            return CommitVerdict(CommitOutcome.FORCE_RESET, "untrusted_kv", output)
        return CommitVerdict(CommitOutcome.APPLY, output_ids=output)

    if not has_state or current_version is None or current_output is None:
        return CommitVerdict(CommitOutcome.NEED_SNAPSHOT, "no_cursor")
    if not state_trusted:
        return CommitVerdict(CommitOutcome.NEED_SNAPSHOT, "untrusted_kv")
    if int(req.base_commit_version) != int(current_version):
        return CommitVerdict(CommitOutcome.NEED_SNAPSHOT, "base_version")
    if int(req.base_output_len) != len(current_output):
        return CommitVerdict(CommitOutcome.NEED_SNAPSHOT, "base_output_len")
    expected = int(prompt_len) + int(req.base_output_len) + len(req.delta_ids)
    if int(req.base_committed_len) != expected:
        return CommitVerdict(CommitOutcome.REJECT, "delta_length")
    output = list(current_output) + [int(x) for x in req.delta_ids]
    return CommitVerdict(CommitOutcome.APPLY, output_ids=output)


def note_commit_send(metrics, mode: str, n_tokens: int) -> None:
    if metrics is None:
        return
    counts = metrics.counts
    if mode == "snapshot":
        counts["commit_snapshot_reqs"] += 1
    elif mode == "delta":
        counts["commit_delta_reqs"] += 1
        counts["commit_delta_tokens"] += int(n_tokens)


def grammar_committed_ids(
    commit_output: Optional[Sequence[int]],
    output_ids: Optional[Sequence[int]],
    origin_ids: Optional[Sequence[int]],
    padded_ids: Optional[Sequence[int]],
    *,
    tree_mode: bool,
) -> List[int]:
    """Committed output for grammar. Excludes prompt and unaccepted drafts."""
    if commit_output is not None:
        return list(commit_output)
    if tree_mode:
        prompt = list(padded_ids if padded_ids is not None else origin_ids or [])
        folded = list(origin_ids or []) + list(output_ids or [])
        return folded[len(prompt) :]
    return list(output_ids or [])


def mm_items_complete(items) -> bool:
    """Every item needs a feature or embedding. One live feature is not enough."""
    if not items:
        return False
    for item in items:
        if isinstance(item, dict):
            feature = item.get("feature")
            embedding = item.get("precomputed_embeddings")
        else:
            feature = getattr(item, "feature", None)
            embedding = getattr(item, "precomputed_embeddings", None)
        if feature is None and embedding is None:
            return False
    return True


def note_commit_result(metrics, name: str, n: int = 1) -> None:
    if metrics is None:
        return
    metrics.counts[name] += int(n)
