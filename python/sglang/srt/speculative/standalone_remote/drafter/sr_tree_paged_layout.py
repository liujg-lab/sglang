"""Paged tree attention layout for STANDALONE_REMOTE NPU Draft.

CPU-importable. CPU helpers plan logical indices; device helpers gather
physical slots. Do not ``.cpu()`` / ``.tolist()`` device mappings, and do not
use device boolean indexing that yields a dynamic shape.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch

SeqLens = Union[torch.Tensor, Sequence[int]]

SR_TREE_PAGED_ENV = "SGLANG_NPU_SR_TREE_PAGED"
SR_TREE_WARMUP_ENV = "SGLANG_NPU_SR_TREE_WARMUP"
SR_TREE_UPDATE_OVERLAP_ENV = "SGLANG_NPU_SR_TREE_UPDATE_OVERLAP"
SR_TAIL_UPDATE_OVERLAP_ENV = "SGLANG_NPU_SR_TAIL_UPDATE_OVERLAP"
ALLOC_ORDINARY = "ordinary"
ALLOC_LEASE = "lease"
IMPL_PAGED_ATB = "paged_atb"
IMPL_PAGED_FIA = "paged_fia"
IMPL_COMPACT_FIA = "compact_fia"

PAGED_FIA_OP_NAMES = frozenset(
    {
        "npu_fused_infer_attention_score",
        "npu_fused_infer_attention_score.out",
    }
)
PAGED_ATB_OP_NAMES = frozenset(
    {
        "npu_paged_attention",
        "_npu_paged_attention",
        "npu_paged_attention.out",
    }
)

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def read_sr_tree_paged_env(env=None) -> bool:
    """Read once during initialization; default eligible Drafts to paged."""
    environ = os.environ if env is None else env
    raw = environ.get(SR_TREE_PAGED_ENV, "1")
    return str(raw).strip().lower() in _TRUTHY


def read_sr_tree_warmup_env(env=None) -> bool:
    """Read once during SR Draft/Target init; default on.

    Gates extra-graph warmup on both ends (layout, alloc/mapping, Target
    kernels). Set 0 to skip. Not a runtime toggle.
    """
    environ = os.environ if env is None else env
    raw = environ.get(SR_TREE_WARMUP_ENV, "1")
    return str(raw).strip().lower() in _TRUTHY


def read_sr_tree_update_overlap_env(env=None) -> bool:
    """Read once during SR Draft init; default on.

    Gates paged-tree ``graph.update`` / ``graph.replay`` overlap on NPU Draft.
    Set 0 to keep serial. Not a runtime toggle. Compact-FIA keeps
    ``SGLANG_NPU_TREE_FIA_SERIAL_UPDATE``.
    """
    environ = os.environ if env is None else env
    raw = environ.get(SR_TREE_UPDATE_OVERLAP_ENV, "1")
    return str(raw).strip().lower() in _TRUTHY


def read_sr_tail_update_overlap_env(env=None) -> bool:
    """Read once during SR tail graph init; default off.

    Requests overlap of tail ``graph.update`` and ``graph.replay``. Set 1 to
    enable after capture succeeds. Not a runtime toggle.
    """
    environ = os.environ if env is None else env
    raw = environ.get(SR_TAIL_UPDATE_OVERLAP_ENV, "0")
    return str(raw).strip().lower() in _TRUTHY


def _as_int_list(values: SeqLens) -> list[int]:
    if values is None:
        return []
    if isinstance(values, torch.Tensor):
        return [int(x) for x in values.reshape(-1).tolist()]
    return [int(x) for x in values]


def remainder(prefix_len: int, page_size: int) -> int:
    page = max(int(page_size), 1)
    return int(prefix_len) % page


def pages_per_branch(prefix_remainder: int, steps: int, page_size: int) -> int:
    page = max(int(page_size), 1)
    return int(math.ceil((int(prefix_remainder) + int(steps)) / page))


def max_query_context_len(prefix_len: int, num_steps: int) -> int:
    """Last tree-forward context: ``P + (S-2) + 1 == P + S - 1`` for S>=1."""
    steps = max(int(num_steps), 1)
    return max(int(prefix_len), 0) + steps - 1


def query_page_count(prefix_len: int, step_id: int, page_size: int) -> int:
    page = max(int(page_size), 1)
    ctx = max(int(prefix_len), 0) + int(step_id) + 1
    return max(int(math.ceil(ctx / page)), 1) if ctx > 0 else 1


def shared_page_count(prefix_len: int, page_size: int) -> int:
    page = max(int(page_size), 1)
    return max(int(prefix_len), 0) // page


def branch_page_step_index(page_j: int, prefix_remainder: int, page_size: int) -> int:
    page = max(int(page_size), 1)
    return max(int(page_j) * page - int(prefix_remainder), 0)


@dataclass
class PrefixTailCopyIndices:
    req_index: torch.Tensor
    branch: torch.Tensor
    tail_off: torch.Tensor
    prefix_col: torch.Tensor

    def __len__(self) -> int:
        return int(self.req_index.numel())


def plan_prefix_tail_copy_indices(
    prefix_lens_cpu: SeqLens,
    allocation_kind: str,
    topk: int,
    page_size: int,
) -> PrefixTailCopyIndices:
    """CPU logical copy plan. Shape comes only from host prefix lengths."""
    prefixes = _as_int_list(prefix_lens_cpu)
    page = max(int(page_size), 1)
    k = max(int(topk), 1)
    kind = str(allocation_kind)
    reqs: list[int] = []
    branches: list[int] = []
    tails: list[int] = []
    cols: list[int] = []
    start_k = 0 if kind == ALLOC_LEASE else 1
    for b, prefix in enumerate(prefixes):
        rem = remainder(prefix, page)
        if rem == 0:
            continue
        for br in range(start_k, k):
            base = prefix - rem
            for t in range(rem):
                reqs.append(b)
                branches.append(br)
                tails.append(t)
                cols.append(base + t)
    empty = not reqs
    dtype = torch.int64
    return PrefixTailCopyIndices(
        req_index=torch.tensor(reqs, dtype=dtype) if not empty else torch.empty(0, dtype=dtype),
        branch=torch.tensor(branches, dtype=dtype) if not empty else torch.empty(0, dtype=dtype),
        tail_off=torch.tensor(tails, dtype=dtype) if not empty else torch.empty(0, dtype=dtype),
        prefix_col=torch.tensor(cols, dtype=dtype) if not empty else torch.empty(0, dtype=dtype),
    )


def materialize_prefix_tail_copy_slots(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    branch_pages: torch.Tensor,
    copy_indices: PrefixTailCopyIndices,
    page_size: int,
):
    """Device gather of physical src/dst. Indices stay dense; no device bool mask."""
    device = req_to_token.device
    n = int(copy_indices.req_index.numel())
    if n == 0:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return empty, empty
    req_idx = copy_indices.req_index.to(device=device, dtype=torch.int64)
    branch = copy_indices.branch.to(device=device, dtype=torch.int64)
    tail = copy_indices.tail_off.to(device=device, dtype=torch.int64)
    col = copy_indices.prefix_col.to(device=device, dtype=torch.int64)
    pool = req_pool_indices.to(device=device, dtype=torch.int64)[req_idx]
    src = req_to_token.to(device=device)[pool, col]
    page0 = branch_pages.to(device=device, dtype=torch.int64)[req_idx, branch, 0]
    dst = page0 * int(page_size) + tail
    return src, dst


def materialize_branch_pages(
    draft_slots: torch.Tensor,
    prefix_lens_cpu: SeqLens,
    page_size: int,
    topk: int,
    num_steps: int,
    max_nnp: Optional[int] = None,
    dummy_page: int = 0,
) -> torch.Tensor:
    """Physical first-token page of each branch page, from compact ``(B,K,S)`` slots."""
    slots = draft_slots
    if slots.dim() != 3:
        raise ValueError(f"draft_slots must be (B,K,S), got {tuple(slots.shape)}")
    batch, k_dim, steps_dim = (int(slots.shape[0]), int(slots.shape[1]), int(slots.shape[2]))
    prefixes = _as_int_list(prefix_lens_cpu)
    if len(prefixes) != batch:
        raise ValueError(
            f"prefix_lens_cpu length {len(prefixes)} != draft batch {batch}"
        )
    if k_dim != max(int(topk), 1):
        raise ValueError(f"draft_slots topk {k_dim} != topk {topk}")
    page = max(int(page_size), 1)
    nnp = [pages_per_branch(remainder(p, page), num_steps, page) for p in prefixes]
    width = int(max_nnp) if max_nnp is not None else max(nnp, default=0)
    device = slots.device
    if width <= 0:
        return torch.zeros((batch, k_dim, 0), dtype=torch.int64, device=device)
    step_idx = torch.zeros((batch, width), dtype=torch.int64)
    valid = torch.zeros((batch, width), dtype=torch.bool)
    for b, prefix in enumerate(prefixes):
        rem = remainder(prefix, page)
        for j in range(nnp[b]):
            step_idx[b, j] = min(branch_page_step_index(j, rem, page), steps_dim - 1)
            valid[b, j] = True
    step_idx = step_idx.to(device=device)
    valid = valid.to(device=device)
    gather_idx = step_idx.unsqueeze(1).expand(batch, k_dim, width)
    pages = slots.to(dtype=torch.int64).gather(2, gather_idx) // page
    dummy = torch.full((), int(dummy_page), dtype=torch.int64, device=device)
    return torch.where(valid.unsqueeze(1), pages, dummy)


def materialize_shared_prefix_pages(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    prefix_lens_cpu: SeqLens,
    page_size: int,
    max_shared: Optional[int] = None,
    dummy_page: int = 0,
) -> torch.Tensor:
    """Complete prefix pages ``P // PS``. Partial last page is not shared."""
    prefixes = _as_int_list(prefix_lens_cpu)
    batch = len(prefixes)
    page = max(int(page_size), 1)
    n_shared = [shared_page_count(p, page) for p in prefixes]
    width = int(max_shared) if max_shared is not None else max(n_shared, default=0)
    device = req_to_token.device
    if batch == 0 or width <= 0:
        return torch.zeros((batch, width), dtype=torch.int64, device=device)
    cols = (torch.arange(width, dtype=torch.int64) * page).unsqueeze(0).expand(batch, width)
    valid = torch.zeros((batch, width), dtype=torch.bool)
    for b, n in enumerate(n_shared):
        if n:
            valid[b, :n] = True
    cols = cols.to(device=device)
    valid = valid.to(device=device)
    pool = req_pool_indices.to(device=device, dtype=torch.int64).reshape(batch, 1).expand(
        batch, width
    )
    token = req_to_token.to(device=device)[pool, cols]
    pages = token.to(dtype=torch.int64) // page
    dummy = torch.full((), int(dummy_page), dtype=torch.int64, device=device)
    return torch.where(valid, pages, dummy)


def assemble_block_tables(
    shared_pages: torch.Tensor,
    branch_pages: torch.Tensor,
    n_shared_cpu: SeqLens,
    n_query_pages_cpu: SeqLens,
    max_pages: int,
    dummy_page: int = 0,
) -> torch.Tensor:
    """One row per (seq, branch). Query page counts, not reserved alloc pages."""
    if branch_pages.dim() != 3:
        raise ValueError(f"branch_pages must be (B,K,nnp), got {tuple(branch_pages.shape)}")
    batch, topk, _nnp = (
        int(branch_pages.shape[0]),
        int(branch_pages.shape[1]),
        int(branch_pages.shape[2]),
    )
    n_shared = _as_int_list(n_shared_cpu)
    n_query = _as_int_list(n_query_pages_cpu)
    if len(n_shared) != batch or len(n_query) != batch:
        raise ValueError("n_shared/n_query length must match batch")
    cols = max(int(max_pages), 0)
    device = branch_pages.device
    dummy = int(dummy_page)
    if cols <= 0:
        return torch.zeros((batch * topk, 0), dtype=torch.int32, device=device)
    shared_w = int(shared_pages.shape[1]) if shared_pages.dim() == 2 else 0
    col = torch.arange(cols, device=device, dtype=torch.int64).view(1, 1, cols)
    n_sh = torch.tensor(n_shared, dtype=torch.int64, device=device).view(batch, 1, 1)
    n_q = torch.tensor(n_query, dtype=torch.int64, device=device).view(batch, 1, 1)
    is_shared = col < n_sh
    branch_j = col - n_sh
    is_branch = (~is_shared) & (branch_j < n_q)
    if shared_w <= 0:
        shared_val = torch.full(
            (batch, topk, cols), dummy, dtype=torch.int64, device=device
        )
    else:
        idx = col.reshape(cols).clamp(min=0, max=shared_w - 1)
        gathered = shared_pages.to(device=device, dtype=torch.int64)[:, idx]
        shared_val = gathered.unsqueeze(1).expand(batch, topk, cols)
    branch_idx = branch_j.clamp(min=0, max=max(_nnp - 1, 0))
    if _nnp <= 0:
        branch_val = torch.full(
            (batch, topk, cols), dummy, dtype=torch.int64, device=device
        )
    else:
        k_ids = torch.arange(topk, device=device, dtype=torch.int64).view(1, topk, 1)
        b_ids = torch.arange(batch, device=device, dtype=torch.int64).view(batch, 1, 1)
        branch_val = branch_pages.to(device=device, dtype=torch.int64)[
            b_ids.expand(batch, topk, cols),
            k_ids.expand(batch, topk, cols),
            branch_idx.expand(batch, topk, cols),
        ]
    dummy_t = torch.full((), dummy, dtype=torch.int64, device=device)
    pages = torch.where(
        is_shared.expand(batch, topk, cols),
        shared_val,
        torch.where(is_branch.expand(batch, topk, cols), branch_val, dummy_t),
    )
    return pages.reshape(batch * topk, cols).to(torch.int32)


def fill_active_rows(raw_bs: int, topk: int, capture_rows: int, device=None) -> torch.Tensor:
    rows = max(int(capture_rows), 0)
    live = max(int(raw_bs), 0) * max(int(topk), 1)
    active = torch.zeros((rows,), dtype=torch.bool, device=device)
    if live and rows:
        active[: min(live, rows)] = True
    return active


def build_step_context_lens(
    prefix_lens_cpu: SeqLens,
    topk: int,
    step_id: int,
    capture_rows: int,
    pad_len: int = 1,
) -> torch.Tensor:
    """Independent CPU int32 payload for one tree-forward step."""
    prefixes = _as_int_list(prefix_lens_cpu)
    k = max(int(topk), 1)
    rows = max(int(capture_rows), 0)
    values = []
    for prefix in prefixes:
        values.extend([max(int(prefix), 0) + int(step_id) + 1] * k)
    if len(values) > rows:
        values = values[:rows]
    else:
        values.extend([int(pad_len)] * (rows - len(values)))
    return torch.tensor(values, dtype=torch.int32)


def context_lens_list(context_lens_cpu: torch.Tensor) -> list[int]:
    return [int(x) for x in context_lens_cpu.reshape(-1).tolist()]


def kv_buckets_to_page_buckets(kv_buckets: Sequence[int], page_size: int) -> list[int]:
    page = max(int(page_size), 1)
    return sorted(
        {max((int(b) + page - 1) // page, 1) for b in kv_buckets},
        reverse=True,
    )


def select_page_bucket(needed_pages: int, page_buckets: Sequence[int]) -> Optional[int]:
    """Smallest captured page bucket that can hold ``needed_pages``."""
    need = max(int(needed_pages), 1)
    fitted = sorted(int(b) for b in page_buckets if int(b) >= need)
    return fitted[0] if fitted else None


def quantize_page_width(needed_pages, page_buckets=None) -> int:
    """Eager table width snapped to graph buckets, then to powers of two.

    Always returns a value ``>= needed_pages`` so tables are only widened.
    Empty or undersized buckets fall back to the next power of two.
    """
    need = max(int(needed_pages), 1)
    chosen = select_page_bucket(need, page_buckets or [])
    if chosen is not None:
        return int(chosen)
    width = 1
    while width < need:
        width *= 2
    return width


def resolve_eager_page_buckets(kv_buckets, page_size: int, max_pages=None):
    """Same bucket source as draft ``can_run``: kv buckets, else max pages."""
    buckets = list(kv_buckets) if kv_buckets else None
    if buckets:
        return kv_buckets_to_page_buckets(buckets, page_size)
    if max_pages is not None:
        return [max(int(max_pages), 1)]
    return None


def tree_paged_shape_key(
    prefix_lens_cpu: SeqLens,
    page_size: int,
    topk: int,
    num_steps: int,
    page_buckets=None,
    max_pages: Optional[int] = None,
):
    """Shapes the eager builders allocate: raw shared/branch, quantized width.

    Only the query width is bucketed, so shared and branch stay data driven.
    Widening them cost host time every round and bought nothing once warmup
    enumerates raw shared page counts.
    """
    del topk
    prefixes = _as_int_list(prefix_lens_cpu)
    n_shared = [shared_page_count(p, page_size) for p in prefixes]
    n_query_total = max_query_pages_for_tree(prefixes, num_steps, page_size)
    nnp_list = [
        pages_per_branch(remainder(p, page_size), num_steps, page_size)
        for p in prefixes
    ]
    if max_pages is not None:
        width = int(max_pages)
    else:
        width = quantize_page_width(max(n_query_total, default=1), page_buckets)
    return (
        len(prefixes),
        max(n_shared, default=0),
        max(nnp_list, default=0),
        width,
    )


def fill_paged_cpu_update_payload(payload, step_lens_list, step_ids, attr_name):
    """Copy independent per-step CPU lengths into captured update records.

    Two-pass: validate every record, then write. A late error leaves dests
    unchanged. Source tensors are CPU and cached only for this call; captured
    dest objects are mutated in place and never replaced.
    """
    if not attr_name:
        raise ValueError("paged update attr_name must not be empty")
    if payload is None:
        raise ValueError("paged update payload must not be None")
    n = len(payload)
    if n != len(step_ids):
        raise ValueError(
            f"paged payload length {n} != step_ids length {len(step_ids)}"
        )
    n_steps = len(step_lens_list)
    src_lists = [[int(x) for x in list(step_lens)] for step_lens in step_lens_list]

    dests = []
    steps = []
    for i, rec in enumerate(payload):
        step = int(step_ids[i])
        if step < 0 or step >= n_steps:
            raise ValueError(
                f"paged step_ids[{i}]={step} out of range n_steps={n_steps}"
            )
        dest = rec[attr_name]
        src = src_lists[step]
        if torch.is_tensor(dest) and int(dest.numel()) != len(src):
            raise ValueError(
                f"paged payload[{i}] {attr_name} size {int(dest.numel())} "
                f"!= step lens {len(src)}"
            )
        dests.append(dest)
        steps.append(step)

    src_tensors = {}
    for dest, step in zip(dests, steps):
        src = src_lists[step]
        if torch.is_tensor(dest):
            key = (step, dest.dtype, int(dest.numel()))
            buf = src_tensors.get(key)
            if buf is None:
                buf = torch.tensor(src, dtype=dest.dtype, device="cpu")
                src_tensors[key] = buf
            dest.copy_(buf)
        else:
            dest[:] = src
    return payload


def make_dummy_block_tables(
    rows: int, max_pages: int, dummy_page: int, device=None
) -> torch.Tensor:
    return torch.full(
        (max(int(rows), 0), max(int(max_pages), 0)),
        int(dummy_page),
        dtype=torch.int32,
        device=device,
    )


@dataclass
class SRTreePagedMetadata:
    block_tables: torch.Tensor
    active_rows: torch.Tensor
    context_lens_cpu: torch.Tensor
    context_lens_list: list
    dummy_page: int
    max_pages: int
    impl: str


def max_query_pages_for_tree(prefix_lens_cpu: SeqLens, num_steps: int, page_size: int) -> list[int]:
    """Pages needed by the last real forward, not reserved alloc slots."""
    prefixes = _as_int_list(prefix_lens_cpu)
    last_step = max(int(num_steps) - 2, 0)
    return [query_page_count(p, last_step, page_size) for p in prefixes]


def prepare_tree_paged_view(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    draft_slots: torch.Tensor,
    prefix_lens_cpu: SeqLens,
    page_size: int,
    topk: int,
    num_steps: int,
    dummy_page: int = 0,
    max_pages: Optional[int] = None,
    capture_rows: Optional[int] = None,
):
    """Build shared/branch pages and a query-width block table once per round."""
    prefixes = _as_int_list(prefix_lens_cpu)
    n_shared = [shared_page_count(p, page_size) for p in prefixes]
    n_query_total = max_query_pages_for_tree(prefixes, num_steps, page_size)
    n_query_branch = [max(q - s, 0) for q, s in zip(n_query_total, n_shared)]
    width = int(max_pages) if max_pages is not None else max(n_query_total, default=1)
    shared = materialize_shared_prefix_pages(
        req_to_token,
        req_pool_indices,
        prefixes,
        page_size,
        dummy_page=dummy_page,
    )
    branch = materialize_branch_pages(
        draft_slots,
        prefixes,
        page_size,
        topk,
        num_steps,
        dummy_page=dummy_page,
    )
    tables = assemble_block_tables(
        shared, branch, n_shared, n_query_branch, width, dummy_page=dummy_page
    )
    raw_bs = len(prefixes)
    rows = int(capture_rows) if capture_rows is not None else raw_bs * max(int(topk), 1)
    if tables.shape[0] < rows:
        pad = make_dummy_block_tables(
            rows - tables.shape[0], width, dummy_page, device=tables.device
        )
        tables = torch.cat([tables, pad], dim=0)
    elif tables.shape[0] > rows:
        tables = tables[:rows]
    active = fill_active_rows(raw_bs, topk, tables.shape[0], device=tables.device)
    tables = torch.where(
        active.view(-1, 1),
        tables,
        torch.full_like(tables, int(dummy_page)),
    )
    return tables, shared, branch, active, n_shared, n_query_branch


class SRTreeExpandTxn:
    """Allocation/copy/compute lifetime for one tree expand."""

    def __init__(self):
        self.allocation_owned = False
        self.prefix_copy_submitted = False
        self.tree_compute_submitted = False
        self.completion_confirmed = False
        self.rolled_back = False
        self.lease_state = None
        self.allocator_backup = None

    def mark_copy_begin(self) -> None:
        self.prefix_copy_submitted = True

    def mark_compute_begin(self) -> None:
        self.tree_compute_submitted = True

    def in_flight(self) -> bool:
        return (
            self.prefix_copy_submitted or self.tree_compute_submitted
        ) and not self.completion_confirmed

    def may_rollback(self) -> bool:
        return (not self.in_flight()) or self.completion_confirmed


def visible_token_slots_from_pages(
    shared_pages: torch.Tensor,
    branch_pages: torch.Tensor,
    prefix_lens_cpu: SeqLens,
    step_id: int,
    page_size: int,
    dummy_page: int = 0,
) -> list[list[int]]:
    """Oracle visible slots for one step. Host-side, for CPU tests."""
    prefixes = _as_int_list(prefix_lens_cpu)
    page = max(int(page_size), 1)
    topk = int(branch_pages.shape[1]) if branch_pages.dim() == 3 else 1
    shared = shared_pages.detach().cpu()
    branch = branch_pages.detach().cpu()
    rows = []
    for b, prefix in enumerate(prefixes):
        ctx = prefix + int(step_id) + 1
        n_sh = shared_page_count(prefix, page)
        for k in range(topk):
            slots = []
            for t in range(ctx):
                if t < n_sh * page:
                    page_id = int(shared[b, t // page])
                else:
                    bj = (t // page) - n_sh
                    page_id = int(branch[b, k, bj]) if bj < int(branch.shape[2]) else int(dummy_page)
                slots.append(page_id * page + (t % page))
            rows.append(slots)
    return rows


def validate_tree_draft_paged_records(
    records,
    n_steps,
    num_layers,
    kv_attr,
    impl: str,
):
    """Every record must match the captured paged impl. Do not guess order."""
    from sglang.srt.speculative.spec_utils import (
        NpuGraphPreparationError,
        inspect_dispatch_record,
        normalize_fia_op_name,
    )

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
            f"paged records={n_records} != steps*num_layers="
            f"{n_steps}*{num_layers}={expected}; disable tree graph",
            scope="graph",
        )
    if impl == IMPL_PAGED_FIA:
        allowed = PAGED_FIA_OP_NAMES
    elif impl == IMPL_PAGED_ATB:
        allowed = PAGED_ATB_OP_NAMES
    else:
        raise NpuGraphPreparationError(
            f"unknown paged impl {impl!r}; disable tree graph",
            scope="graph",
        )
    for i, rec in enumerate(records):
        op_name, has_kv = inspect_dispatch_record(rec, kv_attr)
        op_name = normalize_fia_op_name(op_name)
        if op_name not in allowed:
            raise NpuGraphPreparationError(
                f"dispatch record[{i}] op={op_name!r} is not {impl}; "
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
