"""Alignment and identity decisions for STANDALONE_REMOTE Draft."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch


class SRWarmupFatalError(RuntimeError):
    """Abort initialization: sync failed, ownership unknown, or restore failed."""


def find_fork_point(draft_ids: Sequence[int], target_ids: Sequence[int]) -> Tuple[bool, int]:
    min_len = min(len(draft_ids), len(target_ids))
    for i in range(min_len):
        if draft_ids[i] != target_ids[i]:
            return False, i
    return len(draft_ids) == len(target_ids), min_len


def classify_prefix_alignment(
    local: Sequence[int],
    target: Sequence[int],
    prefix_len: int,
) -> str:
    """Return one of: equal, replace_tail, append_one, append_n, local_rollback, reprefill."""
    identical, fork = find_fork_point(local, target)
    if identical:
        return "equal"
    if len(local) == len(target) and fork == len(local) - 1:
        return "replace_tail"
    extra = len(target) - len(local)
    if extra >= 1 and fork == len(local):
        # Target accepted one or more tokens on top of Draft's prefix.
        # append_one is the n=1 case (common after AR fallback).
        return "append_one" if extra == 1 else "append_n"
    # Trim only when Target's sequence is a prefix of local. A mid-sequence
    # fork still needs the rest of `target` (reprefill), not a blind trim.
    if fork == len(target) and fork < len(local) and fork >= prefix_len:
        return "local_rollback"
    return "reprefill"


def draft_token_budget(num_draft_tokens: Optional[int], spec_steps: Optional[int]) -> int:
    """Slack for one Draft window (not a lifetime cap). Pause stops the GPU."""
    n = int(num_draft_tokens or 0)
    if n <= 0:
        n = int(spec_steps or 1) + 1
    return max(n + 4, 8)


def sr_decode_seq_len(req) -> int:
    """Materialized decode length used to build the Draft tree/decode batch."""
    committed = int(getattr(req, "kv_committed_len", 0) or 0)
    if committed > 0:
        return committed
    origin = getattr(req, "origin_input_ids", None) or []
    output = getattr(req, "output_ids", None) or []
    return max(0, len(origin) + len(output) - 1)


def seq_lens_cpu_for_host(batch):
    """CPU sequence lengths. Never reads device seq_lens values."""
    cpu = getattr(batch, "seq_lens_cpu", None)
    reqs = getattr(batch, "reqs", None) or ()
    n_req = len(reqs)
    seq = getattr(batch, "seq_lens", None)
    n_dev = int(seq.shape[0]) if seq is not None and hasattr(seq, "shape") else None
    if cpu is not None:
        device = getattr(cpu, "device", None)
        if device is not None and getattr(device, "type", "cpu") != "cpu":
            raise RuntimeError("seq_lens_cpu must stay on CPU")
        n_cpu = int(cpu.shape[0]) if hasattr(cpu, "shape") else len(cpu)
        if n_req and n_cpu != n_req:
            raise RuntimeError(
                f"seq_lens_cpu rows {n_cpu} != request count {n_req}"
            )
        if n_dev is not None and n_cpu != n_dev:
            raise RuntimeError(
                f"seq_lens_cpu rows {n_cpu} != seq_lens rows {n_dev}"
            )
        return cpu
    if not n_req:
        return []
    return [sr_decode_seq_len(req) for req in reqs]


def seq_lens_sum_from_batch(batch) -> int:
    """Host sum of current batch lengths. Never reads device seq_lens values."""
    values = seq_lens_cpu_for_host(batch)
    if hasattr(values, "tolist"):
        values = values.tolist()
    else:
        values = list(values)
    return int(sum(int(x) for x in values))


def draft_needed_max_new_tokens(
    already_generated: int,
    num_draft_tokens: Optional[int],
    spec_steps: Optional[int],
    current_max: Optional[int] = None,
) -> int:
    """Lifetime max_new_tokens so the next window can finish without FINISH_LENGTH."""
    need = int(already_generated) + draft_token_budget(num_draft_tokens, spec_steps)
    return max(int(current_max or 0), need)


def committed_tail_not_in_kv(
    origin_len: int,
    output_ids: Optional[Sequence[int]],
    kv_len: int,
) -> List[int]:
    """Committed output tokens that are not yet in linear KV.

    ``kv_len`` is ``kv_committed_len`` (prefix + ingested outputs). Align only
    mutates ``output_ids``; tree ingest must decode this tail before expand.
    """
    already = max(0, int(kv_len) - int(origin_len))
    return list(output_ids or [])[already:]


def last_token_in_kv(fork: int, kv_committed_len: int) -> bool:
    """True when the token at ``fork`` already has valid written KV."""
    return int(kv_committed_len) > int(fork)


def apply_tree_seed_topk(batch, topk: int) -> int:
    """Write top-k onto the batch and a live sampling_info, if present."""
    k = max(1, int(topk or 1))
    batch.tree_seed_topk = k
    info = getattr(batch, "sampling_info", None)
    if info is not None:
        info.tree_seed_topk = k
    return k


def snapshot_reprefill_fill_ids(
    origin_ids: Optional[Sequence[int]],
    output_ids: Optional[Sequence[int]],
) -> List[int]:
    """Committed prefix for one reprefill: current origin plus output tail."""
    return list(origin_ids or []) + list(output_ids or [])


def rollback_free_range(
    fork_point: int,
    allocated_len: int,
    max_len: int,
) -> Tuple[int, int]:
    """Slots to free on rollback. Do not page-floor ``allocated_len``.

    The start is ``fork_point`` itself. A mid-page fork is refused by the
    rollbacker before this range is used.
    """
    start = min(max(0, int(fork_point)), int(max_len))
    end = min(max(0, int(allocated_len)), int(max_len))
    return start, end


def kv_release_len(committed_len: int, allocated_len: int) -> int:
    """FINISH must cover allocated pages, not only written tokens."""
    return max(int(committed_len or 0), int(allocated_len or 0))


def topk_probs_from_logits(logits: torch.Tensor, k: int, *, upcast: bool = False):
    """Top-k probabilities without a ``[rows, vocab]`` softmax tensor.

    Softmax is strictly increasing in the logit, so the top-k indices match
    softmax-then-topk. ``exp(vals - logsumexp(logits))`` is the softmax of
    those entries. ``upcast`` scores in fp32, which is the tree-seed path.
    """
    work = logits.float() if upcast else logits
    width = int(work.shape[-1])
    k = min(max(int(k), 1), width)
    if k == 1:
        vals, idx = torch.max(work, dim=-1, keepdim=True)
    else:
        vals, idx = torch.topk(work, k, dim=-1)
    lse = torch.logsumexp(work, dim=-1, keepdim=True)
    return torch.exp(vals - lse), idx


def capture_tree_seed_topk(logits: torch.Tensor, topk: int):
    """Tree-seed top-k. Scored in fp32; caller must not keep a vocab tensor."""
    return topk_probs_from_logits(logits, topk, upcast=True)


def tree_seed_matches_prefix(seed, origin_ids: Sequence[int], output_ids: Sequence[int]) -> bool:
    """True when cached ``verified_id`` is the current last committed token."""
    if seed is None or not isinstance(seed, (tuple, list)) or len(seed) < 4:
        return False
    verified = seed[3]
    if verified is None:
        return False
    try:
        if isinstance(verified, torch.Tensor):
            token = int(verified.reshape(-1)[0].item())
        else:
            token = int(verified)
    except (TypeError, ValueError, IndexError):
        return False
    if output_ids:
        last = int(output_ids[-1])
    elif origin_ids:
        last = int(origin_ids[-1])
    else:
        return False
    return token == last


def plan_tree_seed_recovery(
    origin_len: int,
    output_ids: Optional[Sequence[int]],
    kv_committed_len: int,
    seed_ok: bool,
    can_rollback_last_slot: bool,
) -> str:
    """How to get a tree seed after align/ingest.

    ``P`` is origin length, ``L = P + len(output)``, ``K`` is committed KV.
    Returns ``ok``, ``ingest``, ``recapture_last``, or ``reprefill``.
    """
    outputs = list(output_ids or [])
    prefix_len = int(origin_len)
    committed = int(kv_committed_len)
    if committed < prefix_len:
        return "reprefill"
    leftover = committed_tail_not_in_kv(prefix_len, outputs, committed)
    if leftover:
        return "ingest"
    if seed_ok:
        return "ok"
    last_in_output = bool(outputs)
    if last_in_output and can_rollback_last_slot:
        return "recapture_last"
    return "reprefill"


DEFAULT_MAX_INGEST_DECODE_STEPS = 16


def plan_committed_ingest(
    origin_len: int,
    output_ids: Optional[Sequence[int]],
    kv_len: int,
    max_decode_steps: int = DEFAULT_MAX_INGEST_DECODE_STEPS,
) -> Tuple[str, List[int]]:
    """Decide how to get the committed tail into linear KV.

    Returns one of ``("noop", [])``, ``("decode", tail)``, ``("reprefill", tail)``.

    Per-token teacher forcing costs one decode per missing token. After a
    Target AR stretch the tail can be hundreds of tokens, which blows the RPC
    timeout and keeps the tail growing. One extend over the whole sequence is
    orders of magnitude cheaper, so switch to reprefill past the threshold.
    """
    tail = committed_tail_not_in_kv(origin_len, output_ids, kv_len)
    if not tail:
        return "noop", []
    if len(tail) > max(1, int(max_decode_steps)):
        return "reprefill", tail
    return "decode", tail


def ingest_active_indices(tail_lens: Sequence[int]) -> List[List[int]]:
    """For fused tree ingest: at offset t, which reqs still have a tail token."""
    max_t = max(tail_lens) if tail_lens else 0
    return [
        [i for i, n in enumerate(tail_lens) if t < int(n)]
        for t in range(max_t)
    ]


def shift_overlapped_prefill_drafts(
    output_ids: Sequence[int],
    draft_tokens: Sequence[int],
) -> Optional[List[int]]:
    """Drop PREFILL draft prefix that Target already sampled during overlapped extend.

    PREFILL is sent before GPU, so Draft's window is the prompt continuation
    ``[D0, D1, ...]``. After extend, Target has ``T0`` in ``output_ids``. Same-model
    greedy means ``D0 == T0``; verify must use ``[D1, ...]`` with root ``T0``.

    Returns remaining draft tokens. ``None`` means the window diverged (discard;
    the next STEP realigns). Empty ``output_ids`` keeps the full window.
    """
    committed = list(output_ids or [])
    drafts = list(draft_tokens or [])
    if not drafts:
        return []
    n = 0
    while n < len(committed) and n < len(drafts) and committed[n] == drafts[n]:
        n += 1
    if committed and n == 0:
        return None
    return drafts[n:]


def replay_grammar_from_committed(template, committed_ids: Optional[Sequence[int]]):
    """Copy ``template`` and accept Target's committed tokens in order.

    Draft ingest/decode can desync grammar; replay from a pristine template
    before each window so the constraint state matches the committed prefix.
    Returns None when there is no template.
    """
    if template is None:
        return None
    grammar = template.copy()
    for tok in committed_ids or []:
        grammar.accept_token(int(tok))
    return grammar


def wrap_tp_broadcast(obj: Any) -> List[Any]:
    """Box an object for ``broadcast_pyobj`` (needs ``len``). ``[None]`` is valid."""
    return [obj]


def unwrap_tp_broadcast(wrapped: Sequence[Any]) -> Any:
    """Inverse of ``wrap_tp_broadcast``."""
    return wrapped[0]


def is_device_context_error(exc: BaseException) -> bool:
    """True when the accelerator context is already poisoned (do not keep running)."""
    if any(cls.__name__ == "SRTransferUnresolved" for cls in type(exc).__mro__):
        return True
    name = type(exc).__name__
    if name in ("AcceleratorError", "CUDAError", "NPUError", "XPUError"):
        return True
    try:
        import torch

        accel = getattr(torch, "AcceleratorError", None)
    except Exception:
        accel = None
    if accel is not None and isinstance(exc, accel):
        return True
    msg = str(exc).lower()
    return (
        "illegal memory access" in msg
        or "device-side assert" in msg
        or "cudaerrorillegaladdress" in msg
        or "npu error" in msg
        or ("ascend" in msg and "illegal" in msg)
        or "acl stream synchronize failed" in msg
        or "model execution failed" in msg
        or "rtstreamsynchronize" in msg
        or "rtdevicesynchronize" in msg
        or "error code:5070" in msg
    )


def _default_broadcast_pyobj(data, rank, dist_group, src=0):
    from sglang.srt.utils import broadcast_pyobj

    return broadcast_pyobj(data, rank, dist_group, src=src)


def broadcast_sr_obj(
    obj: Any,
    tp_size: int,
    tp_rank: int,
    tp_group,
    tp_cpu_group,
):
    """Broadcast a pickleable object from TP rank 0.

    ``broadcast_pyobj`` requires a sequence. Wrap as ``[obj]`` so bool / None /
    dict all work. Empty list is only the non-src dummy; never the None sentinel.
    """
    if tp_size <= 1:
        return obj
    wrapped = wrap_tp_broadcast(obj)
    out = _default_broadcast_pyobj(
        wrapped if tp_rank == 0 else [],
        tp_group.rank,
        tp_cpu_group,
        src=tp_group.ranks[0],
    )
    return unwrap_tp_broadcast(out)
