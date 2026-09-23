from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Optional, Sequence

import torch
import triton
import triton.language as tl
from huggingface_hub import snapshot_download

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.distributed.parallel_state import (
    GroupCoordinator,
    patch_tensor_parallel_group,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.common import get_last_loc
from sglang.srt.server_args import ServerArgs, get_global_server_args
from sglang.srt.utils import is_cuda, is_hip, is_npu, next_power_of_2

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_info import EagleVerifyInput


if _is_cuda:
    from sgl_kernel import fast_topk
elif _is_hip:
    from sgl_kernel import fast_topk
else:
    from sglang.srt.utils.common import fast_topk


logger = logging.getLogger(__name__)


# Simulate acceptance length for benchmarking purposes
SIMULATE_ACC_LEN = envs.SGLANG_SIMULATE_ACC_LEN.get()  # turn off if < 0
SIMULATE_ACC_METHOD = envs.SGLANG_SIMULATE_ACC_METHOD.get()

TREE_TRAVERSE_TIME_THRESHOLD = 1  # TODO: set this properly
# Backward-compatible alias used by ngram / HIP EAGLE fallback.
TREE_SPEC_KERNEL_AVAILABLE = _is_cuda

_REMOTE_SPEC_ALGOS = {"SPECTRE", "STANDALONE_REMOTE"}


def tree_verify_backend() -> str:
    if _is_cuda:
        return "cuda"
    if _is_npu:
        return "npu"
    if _is_hip:
        return "hip"
    return "cpu"


def device_backend_key(device) -> str:
    """Map a torch device / device string to a backend key (cuda/npu/cpu/...)."""
    if device is None:
        return tree_verify_backend()
    if isinstance(device, torch.device):
        return device.type
    text = str(device).split(":", 1)[0].lower()
    if text.startswith("npu"):
        return "npu"
    if text.startswith("cuda"):
        return "cuda"
    if text.startswith("hip"):
        return "hip"
    return text or tree_verify_backend()


def expand_seq_lens_for_spec_topk(seq_lens, num_tokens: int):
    """Repeat per-seq KV lengths so FIA batch dim matches Q tokens (bs * topk)."""
    if seq_lens is None:
        return seq_lens
    values = list(seq_lens)
    n = len(values)
    if n == 0 or n == num_tokens or n == 1:
        return values
    if num_tokens > 0 and num_tokens % n == 0:
        rpt = num_tokens // n
        return [s for s in values for _ in range(rpt)]
    return values


def normalize_tree_draft_kv_lens(seq_lens, num_q: int, topk: int):
    """Normalize prefix/step KV lengths to one FIA batch row per tree branch.

    Tree draft Q is ``bs * topk``. Each branch of a sequence shares
    ``kv_len = prefix + step_id + 1``.

    - ``len(seq_lens) == num_q``: already per-branch, return as-is.
    - ``num_q == len(seq_lens) * topk``: repeat each length ``topk`` times.
    - otherwise: raise. Never silently truncate or broadcast a singleton.
    """
    if seq_lens is None:
        raise ValueError("tree draft KV lengths must not be None")
    values = list(seq_lens)
    n = len(values)
    num_q = int(num_q)
    topk = max(int(topk), 1)
    if num_q < 0:
        raise ValueError(f"tree draft num_q must be >= 0, got {num_q}")
    if n == num_q:
        return values
    if topk > 1 and n * topk == num_q:
        return [s for s in values for _ in range(topk)]
    raise ValueError(
        f"tree draft KV lengths length {n} incompatible with num_q={num_q} topk={topk}"
    )


class NpuGraphPreparationError(RuntimeError):
    """Raised before NPU graph.replay(); callers may disable the graph and fall back to eager.

    ``scope="format"``: dispatch-record API/fields unreadable (disable all tree graphs).
    ``scope="graph"``: this captured graph or this request's KV lengths are invalid.
    """

    def __init__(self, message, scope="graph"):
        super().__init__(message)
        self.scope = scope


class NpuGraphReplaySubmittedError(RuntimeError):
    """Raised after NPU graph.replay() has been invoked; do not retry the same graph."""


TREE_DRAFT_FIA_OP_NAMES = frozenset(
    {
        "npu_fused_infer_attention_score",
        "npu_fused_infer_attention_score.out",
    }
)

_dumped_unreadable_dispatch_record = False


def build_draft_graph_step_kv_lens(prefix_lens, capture_bs, topk, step_id):
    """Build per-branch KV lengths for one captured tree-draft graph step.

    ``prefix_lens`` is the unpadded raw batch. Pads with 0 to ``capture_bs``,
    then expands to ``capture_bs * topk`` branch rows. Parent replay already
    copies ``seq_lens_cpu`` to capture ``bs``, so callers must pass ``[:raw_bs]``.
    """
    if prefix_lens is None:
        raise ValueError("draft graph prefix_lens must not be None")
    values = [int(s) for s in list(prefix_lens)]
    raw_bs = len(values)
    capture_bs = int(capture_bs)
    topk = max(int(topk), 1)
    step_id = int(step_id)
    if capture_bs < raw_bs:
        raise ValueError(f"capture_bs={capture_bs} smaller than raw_bs={raw_bs}")
    seq_lens = [s + step_id + 1 for s in values] + [0] * (capture_bs - raw_bs)
    num_tokens = capture_bs * topk
    if topk > 1:
        return normalize_tree_draft_kv_lens(seq_lens, num_tokens, topk)
    return expand_seq_lens_for_spec_topk(seq_lens, num_tokens)


def expand_fia_cpu_update_inputs(step_lens_list, num_layers, attr_name):
    """Expand per-step KV lengths to one FIA cpu_update_input per captured layer.

    Capture order is step-major: step0 x L, step1 x L, ...
    Do not use ``list * num_layers`` (that repeats the whole step list).
    """
    num_layers = int(num_layers)
    if num_layers < 0:
        raise ValueError(f"num_layers must be >= 0, got {num_layers}")
    if not attr_name:
        raise ValueError("FIA update attr_name must not be empty")
    return [
        {attr_name: list(step_lens)}
        for step_lens in step_lens_list
        for _ in range(num_layers)
    ]


def fill_fia_cpu_update_payload(payload, step_lens_list, step_ids, attr_name):
    """Copy this round's per-step lengths into a captured cpu_update_input.

    Mutates existing list objects in ``payload``. Callers must not rewrite
    ``payload`` until ``graph.update`` has returned.
    """
    if not attr_name:
        raise ValueError("FIA update attr_name must not be empty")
    if payload is None:
        raise ValueError("FIA update payload must not be None")
    n = len(payload)
    if n != len(step_ids):
        raise ValueError(
            f"FIA payload length {n} != step_ids length {len(step_ids)}"
        )
    n_steps = len(step_lens_list)
    for i, rec in enumerate(payload):
        step = int(step_ids[i])
        if step < 0 or step >= n_steps:
            raise ValueError(
                f"FIA step_ids[{i}]={step} out of range n_steps={n_steps}"
            )
        dest = rec[attr_name]
        src = list(step_lens_list[step])
        dest[:] = src
    return payload


def run_npu_graph_update_and_replay(update_fn, replay_fn, overlap=False):
    """Run graph.update then graph.replay.

    Default is serial. Plain AR DECODE enables overlap explicitly;
    compact-FIA tree keeps the existing env-var policy. Draft paged tree,
    Target ``tree_paged_fia``, and SR tail EXTEND overlap by default. Set
    ``SGLANG_NPU_SR_TREE_UPDATE_OVERLAP=0``,
    ``SGLANG_NPU_SR_TARGET_UPDATE_OVERLAP=0``, or
    ``SGLANG_NPU_SR_TAIL_UPDATE_OVERLAP=0`` for the corresponding serial path.
    Callers pass ``overlap=True`` only after that runner's own gate.

    After replay has been invoked, or if update fails, wrap ordinary
    ``Exception`` as ``NpuGraphReplaySubmittedError`` so callers do not free
    in-flight graph slots. Replay ``BaseException`` (for example
    ``KeyboardInterrupt``) joins the update thread and then propagates the
    original interrupt. Overlap uses a non-daemon thread that must be joined.
    Thread construct/start failure does not call replay and is not wrapped as
    submitted; callers keep their in-flight confirmation path.
    """
    if not overlap:
        try:
            update_fn()
            replay_fn()
        except Exception as exc:
            raise NpuGraphReplaySubmittedError(
                "NPU graph update/replay failed"
            ) from exc
        return

    errors = []

    def _run_update():
        try:
            update_fn()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=_run_update, daemon=False)
    thread.start()
    replay_error = None
    try:
        replay_fn()
    except Exception as exc:
        replay_error = exc
    finally:
        thread.join()
    if replay_error is not None:
        raise NpuGraphReplaySubmittedError(
            "NPU graph update/replay failed"
        ) from replay_error
    if errors:
        raise NpuGraphReplaySubmittedError(
            "NPU graph update/replay failed"
        ) from errors[0]


def normalize_fia_op_name(name):
    """Strip ``npu::`` and ``.default``; keep ``.out``."""
    if name is None:
        return None
    text = str(name).strip()
    if not text:
        return None
    if "::" in text:
        text = text.split("::", 1)[1]
    if text.endswith(".default"):
        text = text[: -len(".default")]
    return text


def _structured_op_name(obj):
    if obj is None:
        return None
    dunder = getattr(obj, "__name__", None)
    if isinstance(dunder, str) and dunder:
        return dunder
    for key in ("op_name", "name", "op"):
        val = getattr(obj, key, None)
        if isinstance(val, str) and val:
            return val
        if isinstance(obj, dict):
            val = obj.get(key)
            if isinstance(val, str) and val:
                return val
    return None


def _structured_kwargs(obj):
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj
    kwargs = getattr(obj, "kwargs", None)
    if isinstance(kwargs, Mapping):
        return kwargs
    update_info = getattr(obj, "update_info", None)
    if isinstance(update_info, Mapping):
        return update_info
    return None


def _schema_has_kv_attr(obj, kv_attr):
    if obj is None:
        return False
    schema = getattr(obj, "_schema", None)
    if schema is None:
        return False
    if isinstance(schema, str):
        return kv_attr in schema
    if isinstance(schema, (list, tuple)):
        return kv_attr in {str(x) for x in schema}
    arguments = getattr(schema, "arguments", None)
    if arguments is not None:
        names = []
        for arg in arguments:
            name = getattr(arg, "name", None)
            names.append(str(name if name is not None else arg))
        return kv_attr in names
    return False


def _dump_unreadable_dispatch_record(rec):
    global _dumped_unreadable_dispatch_record
    if _dumped_unreadable_dispatch_record:
        return
    _dumped_unreadable_dispatch_record = True
    entry = getattr(rec, "op_cache_entry", None)
    logger.warning(
        "Unreadable NPU graph dispatch record: type=%s dir=%s "
        "entry_type=%s entry.__name__=%s",
        type(rec),
        dir(rec),
        type(entry),
        getattr(entry, "__name__", None) if entry is not None else None,
    )


def inspect_dispatch_record(rec, kv_attr):
    """Return ``(normalized_op_name, has_kv_attr)`` from structured fields only.

    Never uses ``str(rec)``. Unreadable records raise
    ``NpuGraphPreparationError(scope="format")``.
    """
    if rec is None:
        raise NpuGraphPreparationError(
            "NPU graph dispatch record is None; disable tree graph",
            scope="format",
        )
    entry = getattr(rec, "op_cache_entry", None)
    raw_name = _structured_op_name(entry) or _structured_op_name(rec)
    kwargs = _structured_kwargs(entry) or _structured_kwargs(rec)
    if raw_name is None:
        _dump_unreadable_dispatch_record(rec)
        raise NpuGraphPreparationError(
            "NPU graph dispatch record missing structured op name "
            "(__name__/op_name); disable tree graph",
            scope="format",
        )
    if kwargs is None and not (
        _schema_has_kv_attr(entry, kv_attr) or _schema_has_kv_attr(rec, kv_attr)
    ):
        _dump_unreadable_dispatch_record(rec)
        raise NpuGraphPreparationError(
            "NPU graph dispatch record missing structured kwargs/_schema; "
            "disable tree graph",
            scope="format",
        )
    op_name = normalize_fia_op_name(raw_name)
    has_kv = False
    if kwargs is not None:
        has_kv = kv_attr in kwargs
    if not has_kv:
        has_kv = _schema_has_kv_attr(entry, kv_attr) or _schema_has_kv_attr(
            rec, kv_attr
        )
    return op_name, has_kv


def validate_tree_draft_fia_records(records, n_steps, num_layers, kv_attr):
    """Require every captured record to be expected FIA with ``kv_attr``.

    Mixed ops or a missing record API refuse replay; do not filter a subset.
    Returns ``(n_records, step_ids)`` with step-major ``step_ids[i] = i // L``.
    """
    if records is None:
        raise NpuGraphPreparationError(
            "NPU graph dispatch records unavailable; disable tree graph",
            scope="format",
        )
    n_steps = int(n_steps)
    num_layers = int(num_layers)
    expected = n_steps * num_layers
    n_records = len(records)
    if n_records != expected:
        raise NpuGraphPreparationError(
            f"FIA records={n_records} != steps*num_layers="
            f"{n_steps}*{num_layers}={expected}; disable tree graph",
            scope="graph",
        )
    for i, rec in enumerate(records):
        op_name, has_kv = inspect_dispatch_record(rec, kv_attr)
        if op_name not in TREE_DRAFT_FIA_OP_NAMES:
            raise NpuGraphPreparationError(
                f"dispatch record[{i}] op={op_name!r} is not expected FIA; "
                "disable tree graph",
                scope="graph",
            )
        if not has_kv:
            raise NpuGraphPreparationError(
                f"dispatch record[{i}] op={op_name!r} missing {kv_attr}; "
                "disable tree graph",
                scope="graph",
            )
    step_ids = [i // num_layers for i in range(n_records)] if num_layers else []
    return n_records, step_ids


def validate_draft_graph_step_kv_lens(
    seq_lens, capture_bs, topk, raw_bs, prefix_lens, step_id
):
    """Check one step's branch KV lengths: real rows prefix+step+1, padding 0."""
    capture_bs = int(capture_bs)
    topk = max(int(topk), 1)
    raw_bs = int(raw_bs)
    step_id = int(step_id)
    values = list(seq_lens)
    expected_len = capture_bs * topk
    if len(values) != expected_len:
        raise NpuGraphPreparationError(
            f"step KV lengths length {len(values)} != capture_bs*topk={expected_len}"
        )
    if prefix_lens is None:
        raise NpuGraphPreparationError("draft graph prefix_lens must not be None")
    prefixes = [int(s) for s in list(prefix_lens)]
    if len(prefixes) != raw_bs:
        raise NpuGraphPreparationError(
            f"prefix_lens length {len(prefixes)} != raw_bs={raw_bs}"
        )
    for b, prefix in enumerate(prefixes):
        want = prefix + step_id + 1
        start = b * topk
        row = values[start : start + topk]
        if any(int(x) != want for x in row):
            raise NpuGraphPreparationError(
                f"branch KV lengths {row} != prefix+step_id+1={want} at seq {b}"
            )
    pad = values[raw_bs * topk :]
    if any(int(x) != 0 for x in pad):
        raise NpuGraphPreparationError(f"padding KV lengths {pad} must be 0")
    return values


def resolve_fia_update_count(n_records, n_steps, num_layers):
    """Return how many cpu_update_input dicts the captured graph needs.

    Expected ``n_steps * num_layers``. Missing record counts refuse replay;
    never guess ``steps * layers``.
    """
    if n_records is None:
        raise NpuGraphPreparationError(
            "FIA record count unavailable; disable tree graph",
            scope="format",
        )
    n_steps = int(n_steps)
    num_layers = int(num_layers)
    expected = n_steps * num_layers
    n_records = int(n_records)
    if n_records != expected:
        raise NpuGraphPreparationError(
            f"FIA records={n_records} != steps*num_layers="
            f"{n_steps}*{num_layers}={expected}; disable tree graph"
        )
    return expected


def build_tree_draft_block_tables(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
    topk: int,
    step_id: int,
    num_steps: int,
    max_pages: Optional[int] = None,
    index_mapping: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build FIA block tables with one row per (seq, topk) draft branch.

    Tree draft Q is ``bs * topk``. Each branch shares prefix pages, then has
    its own last/new pages in ``req_to_token`` (see ``assign_draft_cache_locs``
    and ``generate_draft_decode_kv_indices``). KV length per row is
    ``seq_len + step_id + 1``.

    ``index_mapping``, if set, is applied to gathered token ids before
    converting them to page ids (hybrid SWA).
    """
    device = req_to_token.device
    bs = int(req_pool_indices.shape[0])
    page_size = int(page_size)
    topk = max(int(topk), 1)
    step_id = int(step_id)
    num_steps = max(int(num_steps), 0)
    kv_extra = step_id + 1
    ctx_len = int(req_to_token.shape[1]) if req_to_token.ndim >= 2 else 0

    if bs == 0:
        n_cols = 0 if max_pages is None else int(max_pages)
        return torch.zeros((0, n_cols), dtype=torch.int32, device=device)

    pool_idx = req_pool_indices[:bs].to(device=device, dtype=torch.int64)
    seq = seq_lens[:bs].to(device=device, dtype=torch.int64)
    kv_len = seq + kv_extra
    n_pages = (kv_len + page_size - 1) // page_size
    n_cols = int(n_pages.max().item()) if kv_len.numel() else 0
    if max_pages is not None:
        n_cols = int(max_pages)
    if n_cols <= 0:
        return torch.zeros((bs * topk, 0), dtype=torch.int32, device=device)

    page_idx = torch.arange(n_cols, device=device, dtype=torch.int64)

    if topk == 1:
        token_pos = page_idx.view(1, n_cols) * page_size
        valid = token_pos < kv_len.view(bs, 1)
        token_pos = token_pos.expand(bs, n_cols)
        gather_idx = pool_idx.view(bs, 1).expand(bs, n_cols)
    elif page_size == 1:
        page_idx_3d = page_idx.view(1, 1, n_cols)
        seq_3d = seq.view(bs, 1, 1)
        k_ids = torch.arange(topk, device=device, dtype=torch.int64).view(1, topk, 1)
        token_pos = torch.where(
            page_idx_3d < seq_3d,
            page_idx_3d,
            seq_3d + k_ids * num_steps + (page_idx_3d - seq_3d),
        )
        valid = page_idx_3d < kv_len.view(bs, 1, 1)
        gather_idx = pool_idx.view(bs, 1, 1).expand(bs, topk, n_cols)
        token_pos = token_pos.expand(bs, topk, n_cols)
    else:
        last_page_len = seq % page_size
        prefix_base = seq - last_page_len
        num_new_pages = (last_page_len + num_steps + page_size - 1) // page_size
        n_shared = prefix_base // page_size
        page_idx_3d = page_idx.view(1, 1, n_cols)
        k_ids = torch.arange(topk, device=device, dtype=torch.int64).view(1, topk, 1)
        token_pos_shared = page_idx_3d * page_size
        branch_page = page_idx_3d - n_shared.view(bs, 1, 1)
        token_pos_branch = (
            prefix_base.view(bs, 1, 1)
            + k_ids * num_new_pages.view(bs, 1, 1) * page_size
            + branch_page * page_size
        )
        token_pos = torch.where(
            page_idx_3d < n_shared.view(bs, 1, 1),
            token_pos_shared,
            token_pos_branch,
        )
        valid = page_idx_3d < n_pages.view(bs, 1, 1)
        gather_idx = pool_idx.view(bs, 1, 1).expand(bs, topk, n_cols)

    if ctx_len <= 0:
        pages = torch.zeros(gather_idx.shape, dtype=torch.int32, device=device)
    else:
        valid = valid.expand_as(token_pos)
        overflow = valid & ((token_pos < 0) | (token_pos >= ctx_len))
        if bool(overflow.any().item()):
            max_pos = int(token_pos[overflow].max().item())
            raise RuntimeError(
                "tree draft block table token_pos out of req_to_token range: "
                f"max_valid_pos={max_pos} ctx_len={ctx_len}"
            )
        token_pos_safe = torch.where(valid, token_pos, torch.zeros_like(token_pos))
        token_ids = req_to_token[gather_idx, token_pos_safe]
        if index_mapping is not None:
            token_ids = index_mapping.to(device=device)[token_ids]
        pages = (token_ids // page_size).to(torch.int32)
        pages = torch.where(valid, pages, torch.zeros_like(pages))

    if pages.ndim == 2:
        return pages
    return pages.reshape(bs * topk, n_cols)


def _raise_tree_draft_pos_overflow(token_pos, overflow, ctx_len: int, what: str) -> None:
    max_pos = int(token_pos[overflow].max().item())
    raise RuntimeError(
        f"{what} token_pos out of req_to_token range: "
        f"max_valid_pos={max_pos} ctx_len={ctx_len}"
    )


def build_tree_draft_kv_slots(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
    topk: int,
    step_id: int,
    num_steps: int,
    index_mapping: Optional[torch.Tensor] = None,
    max_kv: Optional[int] = None,
):
    """Token-level KV slots per tree-draft branch, matching CUDA decode indices.

    Returns ``(kv_slots[bs * topk, max_kv], kv_lens[bs * topk])``.

    Row ``i = b * topk + k`` attends ``kv_len = seq_b + step_id + 1`` slots:

    - ``j < seq_b``: ``req_to_token[pool_b, j]`` (true prefix, not a duplicated page)
    - otherwise the compact draft slot of branch ``k`` at offset ``j - seq_b``,
      using the same three-way layout as ``generate_draft_decode_kv_indices``.
    """
    device = req_to_token.device
    bs = int(req_pool_indices.shape[0])
    page_size = int(page_size)
    topk = max(int(topk), 1)
    step_id = int(step_id)
    num_steps = max(int(num_steps), 0)
    kv_extra = step_id + 1
    ctx_len = int(req_to_token.shape[1]) if req_to_token.ndim >= 2 else 0

    if bs == 0:
        n_cols = 0 if max_kv is None else int(max_kv)
        return (
            torch.zeros((0, n_cols), dtype=torch.int64, device=device),
            torch.zeros((0,), dtype=torch.int32, device=device),
        )

    pool_idx = req_pool_indices[:bs].to(device=device, dtype=torch.int64)
    seq = seq_lens[:bs].to(device=device, dtype=torch.int64)
    kv_len = seq + kv_extra
    n_cols = int(kv_len.max().item()) if kv_len.numel() else 0
    if max_kv is not None:
        n_cols = int(max_kv)
    kv_lens = kv_len.repeat_interleave(topk).to(torch.int32)
    if n_cols <= 0:
        return (
            torch.zeros((bs * topk, 0), dtype=torch.int64, device=device),
            kv_lens,
        )

    col = torch.arange(n_cols, device=device, dtype=torch.int64)
    seq_3d = seq.view(bs, 1, 1)
    col_3d = col.view(1, 1, n_cols)
    k_ids = torch.arange(topk, device=device, dtype=torch.int64).view(1, topk, 1)
    draft_i = col_3d - seq_3d
    valid = col_3d < kv_len.view(bs, 1, 1)

    if topk == 1 or page_size == 1:
        draft_pos = seq_3d + k_ids * num_steps + draft_i
    else:
        last_page_len = (seq % page_size).view(bs, 1, 1)
        prefix_base = seq_3d - last_page_len
        num_new_pages = (seq % page_size + num_steps + page_size - 1) // page_size
        draft_pos = (
            prefix_base
            + k_ids * num_new_pages.view(bs, 1, 1) * page_size
            + last_page_len
            + draft_i
        )

    token_pos = torch.where(col_3d < seq_3d, col_3d.expand_as(draft_pos), draft_pos)
    gather_idx = pool_idx.view(bs, 1, 1).expand(bs, topk, n_cols)
    valid = valid.expand_as(token_pos)

    if ctx_len <= 0:
        slots = torch.zeros(gather_idx.shape, dtype=torch.int64, device=device)
    else:
        overflow = valid & ((token_pos < 0) | (token_pos >= ctx_len))
        if bool(overflow.any().item()):
            _raise_tree_draft_pos_overflow(
                token_pos, overflow, ctx_len, "tree draft kv slots"
            )
        token_pos_safe = torch.where(valid, token_pos, torch.zeros_like(token_pos))
        token_ids = req_to_token[gather_idx, token_pos_safe]
        if index_mapping is not None:
            token_ids = index_mapping.to(device=device)[token_ids]
        slots = torch.where(
            valid, token_ids.to(torch.int64), torch.zeros_like(token_ids, dtype=torch.int64)
        )

    return slots.reshape(bs * topk, n_cols), kv_lens


def _tree_draft_cpu_max_token_pos(
    prefix_lens, page_size, topk, speculative_num_steps, step_id
):
    """Max req_to_token column a valid cell of this step can touch."""
    page_size = int(page_size)
    topk = max(int(topk), 1)
    num_steps = max(int(speculative_num_steps), 0)
    kv_extra = int(step_id) + 1
    max_pos = -1
    for seq in prefix_lens:
        seq = int(seq)
        prefix_max = seq - 1 if seq > 0 else -1
        if kv_extra <= 0:
            max_pos = max(max_pos, prefix_max)
            continue
        draft_i_max = kv_extra - 1
        if topk == 1 or page_size == 1:
            pos = seq + (topk - 1) * num_steps + draft_i_max
        else:
            last = seq % page_size
            base = seq - last
            nnp = (last + num_steps + page_size - 1) // page_size
            pos = base + (topk - 1) * nnp * page_size + last + draft_i_max
        max_pos = max(max_pos, pos, prefix_max)
    return max_pos


def _as_cpu_prefix_lens(prefix_lens, raw_bs):
    if prefix_lens is None:
        raise ValueError("tree draft prefix_lens must not be None")
    if isinstance(prefix_lens, torch.Tensor):
        if prefix_lens.device.type != "cpu":
            raise ValueError("tree draft prefix_lens must be a CPU tensor or sequence")
        values = [int(x) for x in prefix_lens.reshape(-1).tolist()]
    else:
        values = [int(x) for x in list(prefix_lens)]
    raw_bs = int(raw_bs)
    if raw_bs < 0:
        raise ValueError(f"raw_bs must be >= 0, got {raw_bs}")
    if len(values) < raw_bs:
        raise ValueError(
            f"prefix_lens length {len(values)} smaller than raw_bs={raw_bs}"
        )
    return values[:raw_bs]


def fill_tree_draft_metadata_(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    prefix_lens,
    slots_out,
    lens_out,
    *,
    raw_bs,
    capture_bs,
    page_size,
    topk,
    speculative_num_steps,
    kv_bucket,
):
    """Fill preallocated tree-draft slot/lens buffers for all forward steps.

    Writes ``speculative_num_steps - 1`` steps. Branch reservation still uses
    the full ``speculative_num_steps``. CPU ``prefix_lens`` drive lengths and
    bounds; device tensors are only used to index ``req_to_token``.
    Returns a list of CPU KV-length vectors (padding 0, no FIA placeholder).
    """
    raw_bs = int(raw_bs)
    capture_bs = int(capture_bs)
    page_size = int(page_size)
    topk = max(int(topk), 1)
    num_steps = max(int(speculative_num_steps), 0)
    n_forward = max(num_steps - 1, 0)
    n_cols = int(kv_bucket)
    rows = capture_bs * topk
    if capture_bs < raw_bs:
        raise ValueError(f"capture_bs={capture_bs} smaller than raw_bs={raw_bs}")
    if n_forward <= 0:
        return []
    if len(slots_out) < n_forward or len(lens_out) < n_forward:
        raise ValueError(
            f"slots/lens buffers {len(slots_out)}/{len(lens_out)} "
            f"smaller than n_forward={n_forward}"
        )

    prefix = _as_cpu_prefix_lens(prefix_lens, raw_bs)
    device = req_to_token.device
    ctx_len = int(req_to_token.shape[1]) if req_to_token.ndim >= 2 else 0

    last_step = n_forward - 1
    max_pos = _tree_draft_cpu_max_token_pos(
        prefix, page_size, topk, num_steps, last_step
    )
    if raw_bs > 0 and max_pos >= ctx_len:
        raise RuntimeError(
            "tree draft kv slots token_pos out of req_to_token range: "
            f"max_valid_pos={max_pos} ctx_len={ctx_len}"
        )
    need_cols = 0
    if raw_bs > 0 and ctx_len > 0 and max_pos >= 0:
        need_cols = min(ctx_len, max_pos + 1)

    pool = req_pool_indices.reshape(-1)
    if int(pool.numel()) < raw_bs:
        raise ValueError(
            f"req_pool_indices too short: numel={int(pool.numel())} need={raw_bs}"
        )
    table = None
    if raw_bs > 0 and need_cols > 0:
        table = req_to_token[pool[:raw_bs].to(device=device, dtype=torch.int64), :need_cols]

    seq = None
    if raw_bs > 0:
        seq = torch.tensor(prefix, dtype=torch.int64, device=device)

    cpu_lens_by_step = []
    for step_id in range(n_forward):
        dest_slots = slots_out[step_id]
        dest_lens = lens_out[step_id]
        if dest_slots.shape[0] < rows or dest_slots.shape[1] < n_cols:
            raise ValueError(
                f"step {step_id} slots shape {tuple(dest_slots.shape)} "
                f"smaller than ({rows}, {n_cols})"
            )
        if dest_lens.shape[0] < rows:
            raise ValueError(
                f"step {step_id} lens shape {tuple(dest_lens.shape)} "
                f"smaller than ({rows},)"
            )
        dest_slots[:rows].fill_(0)
        dest_lens[:rows].fill_(0)
        if dest_slots.shape[0] > rows:
            dest_slots[rows:].fill_(0)
        if dest_lens.shape[0] > rows:
            dest_lens[rows:].fill_(0)

        cpu_kv = []
        for b in range(capture_bs):
            seq_b = prefix[b] if b < raw_bs else 0
            kv_len = seq_b + step_id + 1 if b < raw_bs else 0
            cpu_kv.extend([kv_len] * topk)
        cpu_lens_by_step.append(cpu_kv)

        if raw_bs == 0 or n_cols <= 0:
            continue

        kv_extra = step_id + 1
        kv_len = seq + kv_extra
        raw_rows = raw_bs * topk
        dest_lens[:raw_rows].copy_(kv_len.repeat_interleave(topk).to(dtype=dest_lens.dtype))

        col = torch.arange(n_cols, device=device, dtype=torch.int64)
        seq_3d = seq.view(raw_bs, 1, 1)
        col_3d = col.view(1, 1, n_cols)
        k_ids = torch.arange(topk, device=device, dtype=torch.int64).view(1, topk, 1)
        draft_i = col_3d - seq_3d
        valid = col_3d < kv_len.view(raw_bs, 1, 1)

        if topk == 1 or page_size == 1:
            draft_pos = seq_3d + k_ids * num_steps + draft_i
        else:
            last_page_len = (seq % page_size).view(raw_bs, 1, 1)
            prefix_base = seq_3d - last_page_len
            num_new_pages = (seq % page_size + num_steps + page_size - 1) // page_size
            draft_pos = (
                prefix_base
                + k_ids * num_new_pages.view(raw_bs, 1, 1) * page_size
                + last_page_len
                + draft_i
            )

        token_pos = torch.where(col_3d < seq_3d, col_3d.expand_as(draft_pos), draft_pos)
        valid = valid.expand_as(token_pos)
        if table is None or need_cols <= 0:
            continue
        token_pos_safe = torch.where(
            valid, token_pos.clamp(min=0, max=need_cols - 1), torch.zeros_like(token_pos)
        )
        row_idx = torch.arange(raw_bs, device=device, dtype=torch.int64)
        gather_idx = row_idx.view(raw_bs, 1, 1).expand(raw_bs, topk, n_cols)
        token_ids = table[gather_idx, token_pos_safe]
        slots = torch.where(
            valid,
            token_ids.to(torch.int64),
            torch.zeros_like(token_ids, dtype=torch.int64),
        )
        dest_slots[:raw_rows, :n_cols].copy_(slots.reshape(raw_rows, n_cols)[:, :n_cols])

    return cpu_lens_by_step


def is_remote_spec_algorithm(server_args: Optional[ServerArgs] = None) -> bool:
    if server_args is None:
        try:
            server_args = get_global_server_args()
        except Exception:
            return False
    algo = getattr(server_args, "speculative_algorithm", None) or ""
    return str(algo).upper() in _REMOTE_SPEC_ALGOS


def tree_verify_method_available(
    method: str, backend: Optional[str] = None
) -> bool:
    """Capability table: backend × verify method.

    greedy: CUDA/HIP kernels; NPU uses tree_verify_npu sibling-walk, with
    the portable CPU reference as fallback before device submit.
    target_only: CUDA kernel or portable NPU/CPU reference. Not HIP.
    rpd: CUDA and NPU compact readback with CPU path selection, or the CPU
    reference. Available on every backend.
    """
    backend = backend or tree_verify_backend()
    method = (method or "").lower()
    if method == "greedy":
        return backend in ("cuda", "hip", "npu", "cpu")
    if method == "target_only":
        return backend in ("cuda", "npu", "cpu")
    if method == "rpd":
        return True
    return False


def tensor_on_accelerator(value) -> bool:
    return isinstance(value, torch.Tensor) and value.device.type != "cpu"


def check_tree_verify_tensors(**kwargs):
    from sglang.srt.speculative.tree_verify import check_tree_verify_tensors as impl

    return impl(**kwargs)


def spec_need_hidden_states(server_args: Optional[ServerArgs] = None) -> bool:
    if server_args is None:
        server_args = get_global_server_args()

    # TODO(lsyin): also skip when 1) step = 1 or 2) standalone draft model
    return not server_args.enable_multi_layer_eagle


@triton.jit
def create_extend_after_decode_spec_info(
    verified_id,
    seq_lens,
    accept_lens,
    positions,
    new_verified_id,
    bs_upper: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, bs_upper)
    seq_length = tl.load(seq_lens + pid)
    accept_length = tl.load(accept_lens + pid)

    accept_len_cumsum = tl.sum(
        tl.load(accept_lens + offsets, mask=offsets < pid, other=0)
    )
    positions_ptr = positions + accept_len_cumsum
    mask = offsets < accept_length
    tl.store(positions_ptr + offsets, seq_length - accept_length + offsets, mask)

    accept_len_cumsum += accept_length - 1
    verified_id_data = tl.load(verified_id + accept_len_cumsum)
    tl.store(new_verified_id + pid, verified_id_data)


@triton.jit
def assign_req_to_token_pool(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0)
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    save_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    load_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = save_offset < kv_end
        data = tl.load(out_cache_ptr + load_offset, mask=mask)
        tl.store(token_pool + save_offset, data, mask=mask)
        save_offset += BLOCK_SIZE
        load_offset += BLOCK_SIZE


def assign_req_to_token_pool_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    batch_size: int,
):
    assign_req_to_token_pool[(batch_size,)](
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc,
        req_to_token.shape[1],
        next_power_of_2(batch_size),
    )


def paged_tree_mapping_end(
    seq_len: int, page_size: int, topk: int, num_steps: int
) -> int:
    """Exclusive end column written into ``req_to_token`` for one paged tree."""
    page = max(int(page_size), 1)
    k = max(int(topk), 1)
    steps = max(int(num_steps), 0)
    length = max(int(seq_len), 0)
    remainder = length % page
    branch_pages = (remainder + steps + page - 1) // page
    return length - remainder + k * branch_pages * page


def paged_tree_mapping_extra(page_size: int, topk: int, num_steps: int) -> int:
    """Conservative extra columns beyond context_len for branch expansion."""
    page = int(page_size or 1)
    k = int(topk or 1)
    steps = int(num_steps or 0)
    if page <= 1 or k <= 1:
        return 0
    branch_pages = (page - 1 + steps + page - 1) // page
    return k * branch_pages * page


def req_to_token_extra_context_len(
    draft_tokens, page_size: int = 1, topk: int = 1, num_steps: int = 1
) -> int:
    base = 4 + int(draft_tokens or 0)
    return max(base, paged_tree_mapping_extra(page_size, topk, num_steps))


def paged_tree_mapping_fits(
    seq_lens, page_size: int, topk: int, num_steps: int, pool_len: int
) -> bool:
    """True when every seq's branch mapping stays inside ``req_to_token``."""
    if int(page_size) <= 1 or int(topk) <= 1:
        return True
    if hasattr(seq_lens, "tolist"):
        lens = seq_lens.tolist()
    else:
        lens = list(seq_lens or [])
    pool = int(pool_len)
    return all(
        paged_tree_mapping_end(int(length), page_size, topk, num_steps) <= pool
        for length in lens
    )


def split_draft_cache_locs(raw, num_seqs, topk, num_steps, page_size):
    """Separate paged expand slots from compact draft slots.

    ``topk==1`` or ``page_size==1`` keep a single buffer.
    The compact buffer is filled with ``-1`` so a skipped compact write
    cannot silently reuse uninitialized slots.
    """
    if int(page_size) > 1 and int(topk) > 1:
        draft = torch.full(
            (int(num_seqs) * int(topk) * int(num_steps),),
            -1,
            dtype=raw.dtype,
            device=raw.device,
        )
        return raw, draft
    return raw, raw


def build_paged_draft_cache_locs(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    num_new_pages_per_topk: torch.Tensor,
    topk: int,
    num_steps: int,
    page_size: int,
) -> torch.Tensor:
    """Compact per-branch draft slots from paged ``req_to_token`` (Triton Part 3).

    After Part 1 copies the expanded pages into ``req_to_token`` starting at
    ``seq_len``, each branch's draft tokens live at

    ``prefix_base + k * num_new_pages_per_topk * page_size + last_page_len + i``

    for ``i in [0, num_steps)``. That is the same set Part 3 used to gather
    with ``iter_offset in [L, L + num_steps)`` then subtract ``L``.
    """
    device = req_to_token.device
    dtype = req_to_token.dtype
    num_seqs = int(req_pool_indices.shape[0])
    topk = int(topk)
    num_steps = int(num_steps)
    page_size = int(page_size)
    pool_len = int(req_to_token.shape[1]) if req_to_token.ndim >= 2 else 0

    if num_seqs == 0 or topk <= 0 or num_steps <= 0:
        return torch.empty(0, dtype=dtype, device=device)

    pool = req_pool_indices[:num_seqs].to(device=device, dtype=torch.int64)
    seq = seq_lens[:num_seqs].to(device=device, dtype=torch.int64)
    nnp = num_new_pages_per_topk[:num_seqs].to(device=device, dtype=torch.int64)
    last_page_len = seq % page_size
    prefix_base = seq - last_page_len

    k_ids = torch.arange(topk, device=device, dtype=torch.int64).view(1, topk, 1)
    step_ids = torch.arange(num_steps, device=device, dtype=torch.int64).view(
        1, 1, num_steps
    )
    pos = (
        prefix_base.view(num_seqs, 1, 1)
        + k_ids * nnp.view(num_seqs, 1, 1) * page_size
        + last_page_len.view(num_seqs, 1, 1)
        + step_ids
    )
    overflow = (pos < 0) | (pos >= pool_len)
    if bool(overflow.any().item()):
        _raise_tree_draft_pos_overflow(
            pos, overflow, pool_len, "paged draft cache loc"
        )
    gather_idx = pool.view(num_seqs, 1, 1).expand(num_seqs, topk, num_steps)
    draft = req_to_token[gather_idx, pos]
    if bool((draft < 0).any().item()):
        raise RuntimeError(
            "paged draft cache loc contains unfilled (-1) slots; "
            "req_to_token Part 1 copy may have failed"
        )
    return draft.reshape(num_seqs * topk * num_steps)


def build_last_page_dup_locs(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    num_new_pages_per_topk: torch.Tensor,
    topk: int,
    page_size: int,
):
    """Source/target slots for last-page KV duplication (former Triton Part 2).

    Page-level attention needs the prefix tail copied onto each extra branch
    page. Token-level slot gather reads the original prefix slots and must
    not depend on this copy.
    """
    device = req_to_token.device
    dtype = req_to_token.dtype
    num_seqs = int(req_pool_indices.shape[0])
    topk = int(topk)
    page_size = int(page_size)
    extra = max(topk - 1, 0)
    empty = torch.empty(0, dtype=dtype, device=device)
    if num_seqs == 0 or extra == 0 or page_size <= 1:
        return empty, empty

    pool = req_pool_indices[:num_seqs].to(device=device, dtype=torch.int64)
    seq = seq_lens[:num_seqs].to(device=device, dtype=torch.int64)
    nnp = num_new_pages_per_topk[:num_seqs].to(device=device, dtype=torch.int64)
    last_page_len = seq % page_size
    prefix_base = seq - last_page_len
    n_copy = int(last_page_len.sum().item()) * extra
    if n_copy <= 0:
        return empty, empty

    src_parts = []
    tgt_parts = []
    offsets = torch.arange(page_size, device=device, dtype=torch.int64)
    for b in range(num_seqs):
        length = int(last_page_len[b].item())
        if length <= 0:
            continue
        mask = offsets < length
        src = req_to_token[pool[b], prefix_base[b] + offsets]
        src = src[mask]
        for topk_id in range(1, topk):
            src_parts.append(src)
            tgt = req_to_token[
                pool[b],
                prefix_base[b] + topk_id * nnp[b] * page_size + offsets,
            ]
            tgt_parts.append(tgt[mask])
    return torch.cat(src_parts), torch.cat(tgt_parts)


@triton.jit
def assign_draft_cache_locs(
    req_pool_indices,
    req_to_token,
    seq_lens,
    extend_lens,
    raw_cache_loc,
    pool_len: tl.constexpr,
    topk: tl.constexpr,
    speculative_num_steps: tl.constexpr,
    page_size: tl.constexpr,
    bs_upper: tl.constexpr,
):
    """Part 1 only: copy expanded paged slots into ``req_to_token``.

    Compact draft slots and last-page duplication indices are built in
    PyTorch so ``page_size>1 && topk>1`` does not JIT-specialize on a
    batch-varying duplication length.
    """
    BLOCK_SIZE: tl.constexpr = 128
    pid = tl.program_id(axis=0)

    if page_size == 1 or topk == 1:
        copy_len = topk * speculative_num_steps
        out_cache_ptr = raw_cache_loc + pid * topk * speculative_num_steps
    else:
        bs_offset = tl.arange(0, bs_upper)
        copy_len = tl.load(extend_lens + pid)
        cum_copy_len = tl.sum(
            tl.load(extend_lens + bs_offset, mask=bs_offset < pid, other=0)
        )
        out_cache_ptr = raw_cache_loc + cum_copy_len

    kv_start = tl.load(seq_lens + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    num_loop = tl.cdiv(copy_len, BLOCK_SIZE)
    for i in range(num_loop):
        copy_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = copy_offset < copy_len
        data = tl.load(out_cache_ptr + copy_offset, mask=mask)
        tl.store(token_pool + kv_start + copy_offset, data, mask=mask)


@triton.jit
def generate_draft_decode_kv_indices(
    req_pool_indices,
    req_to_token,
    paged_kernel_lens,
    kv_indices,
    kv_indptr,
    positions,
    pool_len: tl.constexpr,
    kv_indices_stride: tl.constexpr,
    kv_indptr_stride: tl.constexpr,
    bs_upper: tl.constexpr,
    iter_upper: tl.constexpr,
    num_tokens_upper: tl.constexpr,
    page_size: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128
    iters = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    topk_id = tl.program_id(axis=2)

    num_steps = tl.num_programs(axis=0)
    num_seqs = tl.num_programs(axis=1)
    topk = tl.num_programs(axis=2)

    kv_indices += kv_indices_stride * iters
    kv_indptr += kv_indptr_stride * iters
    iters += 1

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(paged_kernel_lens + load_offset, mask=load_offset < bid, other=0)
    seq_len = tl.load(paged_kernel_lens + bid)
    cum_seq_len = tl.sum(seq_lens)

    # Update kv_indices
    kv_offset = cum_seq_len * topk + bid * iters * topk + topk_id * (seq_len + iters)
    kv_ptr = kv_indices + kv_offset
    token_pool_ptr = req_to_token + tl.load(req_pool_indices + bid) * pool_len

    kv_offset = tl.arange(0, BLOCK_SIZE)
    num_loop = tl.cdiv(seq_len, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = kv_offset < seq_len
        data = tl.load(token_pool_ptr + kv_offset, mask=mask)
        tl.store(kv_ptr + kv_offset, data, mask=mask)
        kv_offset += BLOCK_SIZE

    extend_offset = tl.arange(0, iter_upper)
    if page_size == 1 or topk == 1:
        extend_data = tl.load(
            token_pool_ptr + seq_len + topk_id * num_steps + tl.arange(0, iter_upper),
            mask=extend_offset < iters,
        )
    else:
        prefix_len = seq_len
        last_page_len = prefix_len % page_size
        num_new_pages_per_topk = (
            last_page_len + num_steps + page_size - 1
        ) // page_size
        prefix_base = seq_len // page_size * page_size
        start = (
            prefix_base + topk_id * num_new_pages_per_topk * page_size + last_page_len
        )
        extend_data = tl.load(
            token_pool_ptr + start + extend_offset,
            mask=extend_offset < iters,
        )

    tl.store(kv_ptr + seq_len + extend_offset, extend_data, mask=extend_offset < iters)

    # Update kv_indptr
    bs_offset = tl.arange(0, num_tokens_upper)

    zid = bid * topk + topk_id
    if zid == 0:
        zid = num_seqs * topk
    positions = tl.load(positions + bs_offset, mask=bs_offset < zid, other=0)
    base = tl.sum(positions)
    tl.store(kv_indptr + zid, base + zid * iters)


@triton.jit
def align_evict_mask_to_page_size(
    seq_lens,
    evict_mask,
    page_size: tl.constexpr,
    num_draft_tokens: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    t_range = tl.arange(0, BLOCK_SIZE)

    bid = tl.program_id(axis=0)
    seq_len = tl.load(seq_lens + bid)
    io_mask = t_range < num_draft_tokens
    mask_row = tl.load(
        evict_mask + bid * num_draft_tokens + t_range, mask=io_mask, other=0
    )

    num_trues = tl.sum(mask_row)
    num_false = num_draft_tokens - num_trues

    start = (seq_len + num_false - 1) // page_size * page_size - seq_len
    for i in range(max(start, 0), min(start + page_size, num_draft_tokens)):
        tl.store(evict_mask + bid * num_draft_tokens + i, False)


@triton.jit
def get_target_cache_loc(
    tgt_cache_loc,
    to_free_slots,
    accept_length,
    to_free_num_slots,
    out_cache_loc,
    num_verify_tokens: tl.constexpr,
    num_verify_tokens_upper: tl.constexpr,
    bs_upper: tl.constexpr,
):
    bid = tl.program_id(axis=0)
    offset = tl.arange(0, num_verify_tokens_upper)
    bs_offset = tl.arange(0, bs_upper)

    # write the first part to tgt_cache_loc
    accept_len_all = tl.load(accept_length + bs_offset, mask=bs_offset < bid, other=0)
    tgt_cache_loc_start = tl.sum(accept_len_all) + bid
    copy_len = tl.load(accept_length + bid) + 1
    out_cache_loc_row = tl.load(
        out_cache_loc + bid * num_verify_tokens + offset, mask=offset < copy_len
    )
    tl.store(
        tgt_cache_loc + tgt_cache_loc_start + offset,
        out_cache_loc_row,
        mask=offset < copy_len,
    )

    # write the second part to to_free_num_pages
    to_free_num_slots_all = tl.load(
        to_free_num_slots + bs_offset, mask=bs_offset < bid, other=0
    )
    to_free_num_slots_cur = tl.load(to_free_num_slots + bid)
    out_cache_loc_start = num_verify_tokens - to_free_num_slots_cur
    to_free_slots_start = tl.sum(to_free_num_slots_all)

    copy_len = to_free_num_slots_cur
    out_cache_loc_row = tl.load(
        out_cache_loc + bid * num_verify_tokens + out_cache_loc_start + offset,
        mask=offset < copy_len,
    )
    tl.store(
        to_free_slots + to_free_slots_start + offset,
        out_cache_loc_row,
        mask=offset < copy_len,
    )


@torch.compile(dynamic=True, disable=_is_npu)
def get_src_tgt_cache_loc(
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    accept_index: torch.Tensor,
    accept_length: torch.Tensor,
    draft_token_num: int,
    page_size: int,
):
    src_cache_loc = out_cache_loc[accept_index]
    tgt_cache_loc = torch.empty_like(src_cache_loc)
    extended_len = seq_lens + draft_token_num
    keep_len = torch.minimum(
        (seq_lens + accept_length + 1 + page_size - 1) // page_size * page_size,
        extended_len,
    )
    to_free_num_slots = extended_len - keep_len
    return src_cache_loc, tgt_cache_loc, to_free_num_slots


@triton.jit
def filter_finished_cache_loc_kernel(
    out_cache_loc,
    tgt_cache_loc,
    accept_length,
    accept_length_filter,
    bs_upper: tl.constexpr,
    num_verify_tokens_upper: tl.constexpr,
):
    bid = tl.program_id(0)
    bs_offset = tl.arange(0, bs_upper)

    accept_length_all = tl.load(accept_length + bs_offset, mask=bs_offset < bid, other=0)
    old_start = tl.sum(accept_length_all) + bid

    accept_length_filter_all = tl.load(
        accept_length_filter + bs_offset, mask=bs_offset < bid, other=0
    )
    new_start = tl.sum(accept_length_filter_all)

    copy_len = tl.load(accept_length_filter + bid)
    copy_offset = tl.arange(0, num_verify_tokens_upper)
    value = tl.load(
        tgt_cache_loc + old_start + copy_offset, mask=copy_offset < copy_len
    )
    tl.store(
        out_cache_loc + new_start + copy_offset, value, mask=copy_offset < copy_len
    )


@torch.compile(dynamic=True, disable=_is_npu)
def create_accept_length_filter(
    accept_length: torch.Tensor,
    unfinished_index_device: torch.Tensor,
    seq_lens: torch.Tensor,
):
    accept_length_filter = torch.zeros_like(accept_length)
    accept_length_filter[unfinished_index_device] = (
        accept_length[unfinished_index_device] + 1
    )
    seq_lens.add_(accept_length + 1)
    return accept_length_filter


def tree_reselect_parent_rows(
    topk_cs_index: torch.Tensor, num_hidden_rows: int, topk: int
) -> torch.Tensor:
    """Map reselected tree rows to the parent hidden/KV row.

    Same index math as ``select_top_k_tokens`` for ``i > 0``.
    """
    topk = max(int(topk), 1)
    device = topk_cs_index.device
    return topk_cs_index.flatten() // topk + torch.arange(
        0, num_hidden_rows, step=topk, device=device
    ).repeat_interleave(topk)


@torch.compile(dynamic=True, disable=_is_npu)
def select_top_k_tokens(
    i: int,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
):
    parent_rows = None
    if i == 0:
        # The first step after extend
        input_ids = topk_index.flatten()
        if hidden_states is not None:
            hidden_states = hidden_states.repeat_interleave(topk, dim=0)
        scores = topk_p  # shape: (b, topk)

        tree_info = (
            topk_p.unsqueeze(1),  # shape: (b, 1, topk)
            topk_index,  # shape: (b, topk)
            torch.arange(-1, topk, dtype=torch.long, device=input_ids.device)
            .unsqueeze(0)
            .repeat(topk_p.shape[0], 1),  # shape: (b, topk + 1)
        )
    else:
        # The later decode steps
        expand_scores = torch.mul(
            scores.unsqueeze(2), topk_p.reshape(-1, topk, topk)
        )  # (b, topk, 1) x (b, topk ,topk) -> (b, topk, topk)
        topk_cs_p, topk_cs_index = fast_topk(
            expand_scores.flatten(start_dim=1), topk, dim=-1
        )  # (b, topk)
        scores = topk_cs_p  # shape: (b, topk)

        topk_index = topk_index.reshape(-1, topk**2)
        input_ids = torch.gather(topk_index, index=topk_cs_index, dim=1).flatten()

        # Later-step topk_p is (B * K, K). Parent rows come from that layout so
        # KV remap still runs when hidden_states is None.
        parent_rows = tree_reselect_parent_rows(topk_cs_index, topk_p.shape[0], topk)
        if hidden_states is not None and hidden_states.shape[0] > 0:
            hidden_states = hidden_states[parent_rows, :]

        tree_info = (
            expand_scores,  # shape: (b, topk, topk)
            topk_index,  # shape: (b, topk * topk)
            topk_cs_index + (topk**2 * (i - 1) + topk),  # shape: (b, topk)
        )

    return input_ids, hidden_states, scores, tree_info, parent_rows


def generate_simulated_accept_index(
    accept_index,
    predict,
    accept_length,
    bs,
    spec_steps,
    simulate_acc_len: float = SIMULATE_ACC_LEN,
    simulate_acc_method: str = SIMULATE_ACC_METHOD,
):
    assert simulate_acc_len > 0.0

    if simulate_acc_method == "multinomial":
        simulated_values = torch.normal(
            mean=simulate_acc_len,
            std=1.0,
            size=(1,),
            device="cpu",
        )
        # clamp simulated values to be between 1 and self.spec_steps
        simulated_values = torch.clamp(simulated_values, min=1.0, max=spec_steps + 1)
        simulate_acc_len = int(simulated_values.round().item())
    elif simulate_acc_method == "match-expected":
        # multinomial sampling does not match the expected length
        # we keep it for the sake of compatibility of existing tests
        # but it's better to use "match-expected" for the cases that need to
        # match the expected length, One caveat is that this will only sample
        # either round down or round up of the expected length
        simulate_acc_len = max(1.0, min(spec_steps + 1, simulate_acc_len))
        lower = int(simulate_acc_len // 1)
        upper = lower + 1 if lower < spec_steps + 1 else lower
        if lower == upper:
            simulate_acc_len = lower
        else:
            weight_upper = simulate_acc_len - lower
            weight_lower = 1.0 - weight_upper
            probs = torch.tensor([weight_lower, weight_upper], device="cpu")
            sampled_index = torch.multinomial(probs, num_samples=1)
            simulate_acc_len = lower if sampled_index == 0 else upper
    else:
        raise ValueError(f"Invalid simulate_acc_method: {SIMULATE_ACC_METHOD}")

    accept_indx_first_col = accept_index[:, 0].view(-1, 1)
    sim_accept_index = torch.full(
        (bs, spec_steps + 1), -1, dtype=torch.int32, device=accept_index.device
    )
    sim_accept_index[:, :simulate_acc_len] = accept_indx_first_col + torch.arange(
        simulate_acc_len, device=accept_index.device
    )
    accept_length.fill_(simulate_acc_len - 1)
    predict.fill_(100)  # some legit token id
    return sim_accept_index


def traverse_tree(
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    draft_tokens: torch.Tensor,
    grammar: BaseGrammarObject,
    allocate_token_bitmask: torch.Tensor,
    vocab_size: Optional[int] = None,
):
    """
    Traverse the tree constructed by the draft model to generate the logits mask.
    """
    assert (
        retrieve_next_token.shape == retrieve_next_sibling.shape == draft_tokens.shape
    )

    def dfs(
        curr: int,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        parent_pos: int,
    ):
        if curr == 0:
            # the first token generated by the target model, and thus it is always
            # accepted from the previous iteration
            accepted = True
        else:
            parent_bitmask = allocate_token_bitmask[parent_pos]
            curr_token_id = draft_tokens[curr]
            if vocab_size and curr_token_id >= vocab_size:
                accepted = False
            else:
                # 32 boolean bitmask values are packed into 32-bit integers
                accepted = (
                    parent_bitmask[curr_token_id // 32] & (1 << (curr_token_id % 32))
                ) != 0

        if accepted:
            if curr != 0:
                # Accept the current token
                grammar.accept_token(draft_tokens[curr])
            if not grammar.is_terminated():
                # Generate the bitmask for the current token
                grammar.fill_vocab_mask(allocate_token_bitmask, curr)
                if retrieve_next_token[curr] != -1:
                    # Visit the child node
                    dfs(
                        retrieve_next_token[curr],
                        retrieve_next_token,
                        retrieve_next_sibling,
                        curr,
                    )

            if curr != 0:
                # Rollback the current token
                grammar.rollback(1)

        if retrieve_next_sibling[curr] != -1:
            # Visit the sibling node
            dfs(
                retrieve_next_sibling[curr],
                retrieve_next_token,
                retrieve_next_sibling,
                parent_pos,
            )

    dfs(0, retrieve_next_token, retrieve_next_sibling, -1)


def generate_token_bitmask(
    reqs: List[Req],
    verify_input: EagleVerifyInput,
    retrieve_next_token_cpu: torch.Tensor,
    retrieve_next_sibling_cpu: torch.Tensor,
    draft_tokens_cpu: torch.Tensor,
    vocab_size: int,
):
    """
    Generate the logit mask for structured output.
    Draft model's token can be either valid or invalid with respect to the grammar.
    We need to perform DFS to
    1. figure out which tokens are accepted by the grammar.
    2. if so, what is the corresponding logit mask.
    """

    num_draft_tokens = draft_tokens_cpu.shape[-1]

    allocate_token_bitmask = None
    assert len(reqs) == retrieve_next_token_cpu.shape[0]
    grammar = None
    for i, req in enumerate(reqs):
        if req.grammar is not None:
            if allocate_token_bitmask is None:
                allocate_token_bitmask = req.grammar.allocate_vocab_mask(
                    vocab_size=vocab_size,
                    batch_size=draft_tokens_cpu.numel(),
                    device="cpu",
                )
            grammar = req.grammar
            s = time.perf_counter()
            traverse_tree(
                retrieve_next_token_cpu[i],
                retrieve_next_sibling_cpu[i],
                draft_tokens_cpu[i],
                req.grammar,
                allocate_token_bitmask[
                    i * num_draft_tokens : (i + 1) * num_draft_tokens
                ],
                vocab_size=vocab_size,
            )
            tree_traverse_time = time.perf_counter() - s
            if tree_traverse_time > TREE_TRAVERSE_TIME_THRESHOLD:
                logger.warning(
                    f"Bit mask generation took {tree_traverse_time} seconds with "
                    f"grammar: {req.grammar}"
                )

    verify_input.grammar = grammar
    return allocate_token_bitmask


def load_token_map(token_map_path: str) -> List[int]:
    if not os.path.exists(token_map_path):
        repo_id = os.path.dirname(token_map_path)
        file_name = os.path.basename(token_map_path)

        cache_dir = None
        if envs.SGLANG_USE_MODELSCOPE.get():
            from modelscope.utils.file_utils import get_model_cache_root

            cached_repo_path = os.path.join(get_model_cache_root(), repo_id)
            if os.path.exists(cached_repo_path):
                cache_dir = cached_repo_path

        if cache_dir is None:
            if envs.SGLANG_USE_MODELSCOPE.get():
                from modelscope.hub.snapshot_download import (
                    snapshot_download as download_func,
                )
            else:
                download_func = snapshot_download
            cache_dir = download_func(
                repo_id,
                ignore_patterns=["*.bin", "*.safetensors"],
            )

        token_map_path = os.path.join(cache_dir, file_name)
    hot_token_id = torch.load(token_map_path, weights_only=True)
    return torch.tensor(hot_token_id, dtype=torch.int64)


@contextmanager
def draft_tp_context(tp_group: GroupCoordinator):
    # Draft model doesn't use dp and has its own tp group.
    # We disable mscclpp now because it doesn't support 2 comm groups.
    with patch_tensor_parallel_group(tp_group):
        yield


def maybe_detect_nan(tensor: torch.Tensor, msg: str = ""):
    """Async NaN check — no GPU-CPU sync, error surfaces at next sync point."""
    if not envs.SGLANG_SPEC_NAN_DETECTION.get():
        return
    torch._assert_async(~torch.any(torch.isnan(tensor)), f"NaN detected! {msg}")


def maybe_detect_oob(indices: torch.Tensor, low: int, high: int, msg: str):
    """Async OOB check — no GPU-CPU sync, error surfaces at next sync point."""
    if not envs.SGLANG_SPEC_OOB_DETECTION.get():
        return
    if indices.numel() == 0:
        return
    torch._assert_async(
        (indices.min() >= low) & (indices.max() < high),
        f"OOB indices not in [{low}, {high}): {msg}",
    )


# Disable torch.compile for this function because it will be
# even slower.
# @torch.compile(dynamic=True)
def get_last_loc_large_page_size_large_top_k(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    speculative_num_steps: int,
    topk: int,
    page_size: int,
):
    prefix_lens = seq_lens
    last_page_lens = prefix_lens % page_size
    num_new_pages_per_topk = (
        last_page_lens + speculative_num_steps + page_size - 1
    ) // page_size
    seq_lens = prefix_lens // page_size * page_size + num_new_pages_per_topk * (
        page_size * topk
    )
    extend_lens = seq_lens - prefix_lens
    last_loc = get_last_loc(
        req_to_token,
        req_pool_indices,
        prefix_lens,
    )

    return (
        prefix_lens,
        seq_lens,
        last_loc,
        num_new_pages_per_topk,
        extend_lens,
        last_page_lens,
    )
