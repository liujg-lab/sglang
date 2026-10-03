"""Page-level tree KV lease helpers for STANDALONE_REMOTE Draft.

CPU-importable. Lease reuses still-live accepted-path KV after parent remap.
These constants are experimental policy, not performance conclusions.
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.speculative.standalone_remote.sr_align import (
    classify_prefix_alignment,
    find_fork_point,
)
from sglang.srt.speculative.standalone_remote.sr_kv_copy import KVMoveSubmittedError

logger = logging.getLogger(__name__)

MIN_TREE_KV_REUSE_DEPTH = 2
LEASE_PREFIX_WINDOW = 32
TREE_LEASE_RESERVE_PAGES = 8
MAX_TREE_LEASE_PAGES = 256
MAX_TREE_LEASE_FRACTION = 0.25

MISS_VERSION = "version"
MISS_BASE = "base"
MISS_REVISION = "revision"
MISS_PATH = "path"
MISS_TOKEN = "token"
MISS_DEPTH = "depth"
MISS_FIELDS = "fields"


@dataclass(frozen=True)
class SRAlignResult:
    kind: str
    old_kv_committed_len: int
    old_prefix_revision: int
    old_local_len: int
    old_prefix_window: Tuple[int, ...]
    fork: int


@dataclass
class PagedTreeLayout:
    remainders: List[int]
    pages_per_branch: List[int]
    pages_per_req: List[int]
    extend_lens: List[int]
    logical_seq_lens: List[int]
    prefix_lens: List[int]


@dataclass
class SRTreeKVLease:
    rid: str
    version: int
    revision: int
    base_committed_len: int
    prefix_tokens: Tuple[int, ...]
    page_ids: List[int]
    page_slots: torch.Tensor
    candidate_slots: List[int]
    parent_list: List[int]
    top_scores_index: List[int]
    draft_tokens: List[int]
    page_count: int = 0
    in_use: bool = False
    pending_free_event: Any = None
    released: bool = False
    copy_unresolved: bool = False

    def __post_init__(self) -> None:
        if not self.page_count:
            self.page_count = len(self.page_ids or ())


def pages_per_branch(remainder: int, steps: int, page_size: int) -> int:
    page_size = max(int(page_size), 1)
    return int(math.ceil((int(remainder) + int(steps)) / page_size))


def tree_raw_span_len(remainder: int, topk: int, steps: int, page_size: int) -> int:
    nnp = pages_per_branch(remainder, steps, page_size)
    return int(topk) * nnp * int(page_size) - int(remainder)


def physical_tree_slot(
    page_ids: Sequence[int],
    branch: int,
    step: int,
    remainder: int,
    nnp: int,
    page_size: int,
) -> int:
    offset = int(remainder) + int(step)
    page_index = int(branch) * int(nnp) + offset // int(page_size)
    return int(page_ids[page_index]) * int(page_size) + offset % int(page_size)


def build_tree_raw_slots(
    page_ids: Sequence[int],
    remainder: int,
    topk: int,
    steps: int,
    page_size: int,
    dtype=torch.int64,
    device=None,
) -> torch.Tensor:
    """Full logical expand span from r to the last exclusive page end."""
    nnp = pages_per_branch(remainder, steps, page_size)
    span = tree_raw_span_len(remainder, topk, steps, page_size)
    r = int(remainder)
    ps = int(page_size)
    col = torch.arange(span, dtype=torch.int64)
    abs_off = r + col
    branch_span = nnp * ps
    branch = torch.div(abs_off, branch_span, rounding_mode="floor")
    within = abs_off % branch_span
    page_index = branch * nnp + torch.div(within, ps, rounding_mode="floor")
    ids = torch.as_tensor(list(page_ids), dtype=torch.int64)
    slots = ids.index_select(0, page_index) * ps + (within % ps)
    return slots.to(dtype=dtype, device=device)


def build_lease_page_slots(
    page_ids: Sequence[int], page_size: int, dtype=torch.int64, device=None
) -> torch.Tensor:
    ps = int(page_size)
    ids = torch.as_tensor(list(page_ids), dtype=dtype, device=device)
    return (ids.unsqueeze(1) * ps + torch.arange(ps, dtype=dtype, device=device)).reshape(
        -1
    )


def plan_paged_tree_layout(
    seq_lens: Sequence[int],
    page_size: int,
    topk: int,
    steps: int,
) -> PagedTreeLayout:
    ps = max(int(page_size), 1)
    remainders = [int(s) % ps for s in seq_lens]
    nnp = [pages_per_branch(r, steps, ps) for r in remainders]
    per_req = [int(topk) * n for n in nnp]
    prefix = [int(s) for s in seq_lens]
    logical = [
        (p // ps) * ps + n * ps * int(topk) for p, n in zip(prefix, nnp)
    ]
    extend = [logi - p for logi, p in zip(logical, prefix)]
    return PagedTreeLayout(
        remainders=remainders,
        pages_per_branch=nnp,
        pages_per_req=per_req,
        extend_lens=extend,
        logical_seq_lens=logical,
        prefix_lens=prefix,
    )


def lease_budget_ok(
    immediate_free: int,
    live_lease_pages: int,
    need: int,
    reserve: int = TREE_LEASE_RESERVE_PAGES,
    max_pages: int = MAX_TREE_LEASE_PAGES,
    max_fraction: float = MAX_TREE_LEASE_FRACTION,
) -> bool:
    if need <= 0:
        return True
    if need + int(reserve) > int(immediate_free):
        return False
    live_need = int(live_lease_pages) + int(need)
    if live_need > int(max_pages):
        return False
    denom = max(int(live_lease_pages) + int(immediate_free), 1)
    return live_need / denom <= float(max_fraction)


def snapshot_sr_align(req, dreq, prefix_len: int) -> SRAlignResult:
    origin = req.origin_input_ids or []
    output = req.output_ids or []
    kv_len = int(getattr(req, "kv_committed_len", 0) or 0)
    window = prefix_window_tokens(origin, output, base=kv_len)
    padded = list(getattr(req, "sr_padded_ids", None) or origin)
    local = list(origin) + list(output)
    target = list(padded) + list(dreq.committed_ids or [])
    _, fork = find_fork_point(local, target)
    kind = "equal" if local == target else classify_prefix_alignment(
        local, target, prefix_len
    )
    return SRAlignResult(
        kind=kind,
        old_kv_committed_len=kv_len,
        old_prefix_revision=int(getattr(req, "sr_prefix_revision", 0) or 0),
        old_local_len=len(local),
        old_prefix_window=window,
        fork=int(fork),
    )


def read_token_span(origin, output, start: int, count: int) -> List[int]:
    """Read ``count`` logical tokens starting at ``start``. Do not concat the lists."""
    origin = origin or ()
    output = output or ()
    n_origin = len(origin)
    out: List[int] = []
    begin = int(start)
    for i in range(begin, begin + int(count)):
        if i < n_origin:
            out.append(int(origin[i]))
            continue
        j = i - n_origin
        if j < 0 or j >= len(output):
            break
        out.append(int(output[j]))
    return out


def first_forward_node_ids(topk: int, batch_size: int, device=None) -> torch.Tensor:
    return torch.arange(int(topk), dtype=torch.int64, device=device).repeat(
        int(batch_size)
    )


def later_forward_node_ids(tree_info_parents: torch.Tensor) -> torch.Tensor:
    return tree_info_parents.reshape(-1).to(dtype=torch.int64)


class IdentityRemapSubmittedError(KVMoveSubmittedError):
    """Identity remap was submitted; retain scratch and do not retry the move."""


def _identity_stream(device):
    if device.type == "cpu":
        return None
    return torch.get_device_module(device.type).current_stream(device)


def _identity_stream_key(device):
    stream = _identity_stream(device)
    if stream is None:
        return None
    for name in ("stream_id", "cuda_stream", "npu_stream"):
        value = getattr(stream, name, None)
        if value is not None:
            return int(value)
    raise RuntimeError("backend stream does not expose a stable identity")


def _identity_capturing(device):
    if device.type == "cpu":
        return False
    fn = getattr(
        torch.get_device_module(device.type), "is_current_stream_capturing", None
    )
    return bool(fn and fn())


def _same_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.numel() == 0 or right.numel() == 0:
        return False
    return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()


def _validate_identity_remap(slot_node_ids, parent_rows, depth, scratch) -> int:
    if slot_node_ids.ndim != 2 or scratch.ndim != 2:
        raise RuntimeError("identity table and scratch must be rank 2")
    if slot_node_ids.dtype != torch.int64 or scratch.dtype != torch.int64:
        raise RuntimeError("identity table and scratch must be int64")
    if slot_node_ids.device != scratch.device:
        raise RuntimeError("identity scratch must be device-local")
    if (
        parent_rows.ndim != 1
        or parent_rows.dtype not in (torch.int32, torch.int64)
        or parent_rows.device != slot_node_ids.device
    ):
        raise RuntimeError("identity parents must be device-local integers")
    rows = int(parent_rows.numel())
    if depth > int(slot_node_ids.shape[0]) or depth > int(scratch.shape[0]):
        raise RuntimeError("identity remap depth exceeds prepared capacity")
    if rows > int(slot_node_ids.shape[1]) or rows > int(scratch.shape[1]):
        raise RuntimeError("identity remap rows exceed prepared capacity")
    if rows == 0:
        return 0
    if _same_storage(slot_node_ids, scratch):
        raise RuntimeError("identity scratch aliases the identity table")
    if slot_node_ids.device.type == "cpu":
        idx = parent_rows.to(dtype=torch.int64)
        if bool(((idx < 0) | (idx >= rows)).any()):
            raise IndexError("identity parent outside rows")
    return rows


def _execute_identity_remap(slot_node_ids, parent_rows, depth, scratch) -> None:
    """One gather of every historical row, then one scatter back into the table."""
    rows = int(parent_rows.numel())
    depth = int(depth)
    if rows == 0 or depth <= 0:
        return
    if slot_node_ids.device.type == "cpu":
        idx = parent_rows.to(dtype=torch.int64).reshape(-1).contiguous()
        src = slot_node_ids.narrow(0, 0, depth).narrow(1, 0, rows)
        out = scratch.narrow(0, 0, depth).narrow(1, 0, rows)
        if out.is_contiguous():
            torch.index_select(src, 1, idx, out=out)
        else:
            out.copy_(torch.index_select(src, 1, idx))
        src.copy_(out)
        return
    from sglang.srt.speculative.standalone_remote.drafter.sr_identity_remap_kernels import (
        remap_identity,
    )

    remap_identity(slot_node_ids, scratch, parent_rows, depth, rows)


def remap_slot_node_ids(
    slot_node_ids: torch.Tensor,
    parent_rows: torch.Tensor,
    n_prev_steps: int,
    scratch: torch.Tensor,
) -> None:
    """Gather every historical identity, then scatter it back.

    ``scratch`` is ``[depth_capacity, row_capacity]``. Only ``[:depth, :rows]``
    is written. Padding columns past ``rows`` stay untouched, and there is no
    active-row mask.
    """
    depth = int(n_prev_steps)
    if depth <= 0:
        return
    _validate_identity_remap(slot_node_ids, parent_rows, depth, scratch)
    _execute_identity_remap(slot_node_ids, parent_rows, depth, scratch)


class IdentityRemapWorkspace:
    """Draft-owned identity scratch. Graph and eager domains do not share it."""

    def __init__(
        self, device, depth_capacity, row_capacity, *, graph=False, domain="tree_eager"
    ):
        depth_capacity = int(depth_capacity)
        row_capacity = int(row_capacity)
        if depth_capacity < 1 or row_capacity < 1:
            raise ValueError("identity remap capacity must be positive")
        self.device = torch.device(device)
        self.depth_capacity = depth_capacity
        self.row_capacity = row_capacity
        self.graph = bool(graph)
        self.domain = domain
        self.frozen = False
        self.unresolved = False
        self._inside_capture = False
        self._used = False
        self._warmed = False
        self._warmup_hold = None
        self.hold = None
        self._retired = []
        self.scratch = self._empty_scratch(row_capacity)
        self._stream_key = _identity_stream_key(self.device)

    def _empty_scratch(self, rows: int) -> torch.Tensor:
        return torch.empty(
            (self.depth_capacity, int(rows)),
            dtype=torch.int64,
            device=self.device,
        )

    def poison(self) -> None:
        self.unresolved = True

    def check(self, *, check_stream=True) -> None:
        if self.unresolved:
            raise IdentityRemapSubmittedError(
                "identity remap completion is unresolved"
            )
        if (
            check_stream
            and not self._inside_capture
            and self.device.type != "cpu"
            and _identity_stream_key(self.device) != self._stream_key
        ):
            raise RuntimeError(
                "identity workspace belongs to another execution stream"
            )

    def _record_event(self):
        try:
            event = torch.get_device_module(self.device.type).Event()
            event.record(_identity_stream(self.device))
            return event
        except BaseException as exc:
            self.poison()
            raise IdentityRemapSubmittedError(
                "cannot confirm identity scratch consumer completion"
            ) from exc

    def _collect_retired(self) -> None:
        kept = []
        try:
            for item in self._retired:
                if not item[0].query():
                    kept.append(item)
        except BaseException as exc:
            self.poison()
            raise IdentityRemapSubmittedError(
                "identity scratch retirement query failed"
            ) from exc
        self._retired = kept

    def reserve_rows(self, rows: int) -> "IdentityRemapWorkspace":
        self.check()
        rows = int(rows)
        if rows < 0:
            raise ValueError("negative identity row capacity")
        if not self._inside_capture and not _identity_capturing(self.device):
            self._collect_retired()
        if rows <= self.row_capacity:
            return self
        if (
            self.graph
            or self.frozen
            or self._inside_capture
            or _identity_capturing(self.device)
        ):
            raise RuntimeError(
                "identity scratch cannot grow during capture or after graph binding"
            )
        cap = 1 << (rows - 1).bit_length()
        new = self._empty_scratch(cap)
        if self._used and self.device.type != "cpu":
            event = self._record_event()
            self._retired.append((event, self.scratch, self.hold))
        self.scratch = new
        self.row_capacity = cap
        return self

    @contextmanager
    def capture_scope(self):
        if not self.graph:
            raise RuntimeError("eager identity workspace cannot bind a graph")
        self.check(check_stream=False)
        self._inside_capture = True
        try:
            yield self
        finally:
            self._inside_capture = False
            self.frozen = True

    def warm(self) -> None:
        """Compile the device kernel on private storage before capture."""
        self.check()
        if self._warmed or self.device.type == "cpu":
            self._warmed = True
            return
        private_ids = torch.empty((2, 2), dtype=torch.int64, device=self.device)
        private_scratch = torch.empty((2, 2), dtype=torch.int64, device=self.device)
        parents = torch.tensor([1, 0], dtype=torch.int64, device=self.device)
        private_ids.copy_(
            torch.tensor([[0, 1], [2, 3]], dtype=torch.int64, device=self.device)
        )
        self._warmup_hold = (private_ids, private_scratch, parents)
        try:
            _execute_identity_remap(private_ids, parents, 2, private_scratch)
            torch.get_device_module(self.device.type).synchronize()
        except BaseException as exc:
            self.poison()
            raise IdentityRemapSubmittedError(
                "identity remap warmup failed"
            ) from exc
        self._warmup_hold = None
        self._warmed = True

    def submitted(self, inputs, run) -> None:
        self.hold = (inputs, self.scratch, self._warmup_hold)
        self._used = True
        try:
            stream = _identity_stream(self.device)
            if stream is not None and not self._inside_capture:
                for tensor in inputs:
                    if torch.is_tensor(tensor):
                        tensor.record_stream(stream)
                self.scratch.record_stream(stream)
            run()
        except BaseException as exc:
            self.poison()
            raise IdentityRemapSubmittedError(
                "identity gather/scatter failed after submission; refuse retry"
            ) from exc

    def remap(self, slot_node_ids, parent_rows, depth) -> None:
        self.check()
        depth = int(depth)
        if depth <= 0:
            return
        _validate_identity_remap(slot_node_ids, parent_rows, depth, self.scratch)
        self.submitted(
            (slot_node_ids, parent_rows),
            lambda: _execute_identity_remap(
                slot_node_ids, parent_rows, depth, self.scratch
            ),
        )


def lookup_candidate_slots(
    slot_node_ids: torch.Tensor,
    physical_slots: torch.Tensor,
    top_scores_index: torch.Tensor,
    batch_size: int,
    topk: int,
) -> torch.Tensor:
    """Map reply indices to still-live physical slots; missing -> -1.

    Lookup is confined to each request's own ``topk`` columns.
    """
    steps = int(slot_node_ids.shape[0])
    rows = int(batch_size) * int(topk)
    ids = slot_node_ids[:, :rows].reshape(steps, int(batch_size), int(topk))
    phys = physical_slots[:, :rows].reshape(steps, int(batch_size), int(topk))
    cands = top_scores_index.to(dtype=torch.int64)
    if cands.dim() == 1:
        cands = cands.unsqueeze(0)
    ids_f = ids.permute(1, 0, 2).reshape(int(batch_size), steps * int(topk))
    phys_f = phys.permute(1, 0, 2).reshape(int(batch_size), steps * int(topk))
    match = ids_f.unsqueeze(-1) == cands.unsqueeze(1)
    any_match = match.any(dim=1)
    first = match.to(dtype=torch.int64).argmax(dim=1)
    gathered = phys_f.gather(1, first)
    return torch.where(any_match, gathered, gathered.new_full(gathered.shape, -1))


def live_accept_prefix(
    candidate_slots: Sequence[int], accept_indices: Sequence[int]
) -> List[int]:
    """Consecutive live prefix of the accepted path; stop at the first -1."""
    out: List[int] = []
    for idx in accept_indices:
        i = int(idx)
        if i < 0 or i >= len(candidate_slots):
            break
        slot = int(candidate_slots[i])
        if slot < 0:
            break
        out.append(slot)
    return out


def lease_page_count(lease: SRTreeKVLease) -> int:
    n = int(getattr(lease, "page_count", 0) or 0)
    if n > 0:
        return n
    return len(lease.page_ids or ())


def prefix_window_tokens(
    origin: Sequence[int],
    output: Sequence[int] = (),
    base: int = 0,
    window: int = LEASE_PREFIX_WINDOW,
) -> Tuple[int, ...]:
    """Copy at most ``window`` tokens ending at ``base``. Do not concat full lists."""
    base = max(int(base), 0)
    start = max(0, base - int(window))
    origin = origin or ()
    output = output or ()
    n_origin = len(origin)
    out: List[int] = []
    for i in range(start, base):
        if i < n_origin:
            out.append(int(origin[i]))
            continue
        j = i - n_origin
        if j < 0 or j >= len(output):
            break
        out.append(int(output[j]))
    return tuple(out)


def prefix_window_from_committed(
    committed: Sequence[int],
    base: int,
    window: int = LEASE_PREFIX_WINDOW,
) -> Tuple[int, ...]:
    base = max(int(base), 0)
    start = max(0, base - int(window))
    return tuple(int(x) for x in committed[start:base])


def validate_lease_commit(
    lease: SRTreeKVLease,
    *,
    commit_tree_version,
    commit_tree_base_committed_len,
    commit_candidate_indices,
    align: SRAlignResult,
    path_tokens: Sequence[int],
) -> Optional[str]:
    if (
        commit_tree_version is None
        or commit_tree_base_committed_len is None
        or commit_candidate_indices is None
    ):
        return MISS_FIELDS
    if int(lease.version) != int(commit_tree_version):
        return MISS_VERSION
    if int(lease.base_committed_len) != int(commit_tree_base_committed_len):
        return MISS_BASE
    if int(align.old_kv_committed_len) != int(lease.base_committed_len):
        return MISS_BASE
    if int(align.old_prefix_revision) != int(lease.revision):
        return MISS_REVISION
    actual = tuple(align.old_prefix_window)
    stored = tuple(int(x) for x in lease.prefix_tokens)
    stored_window = stored[max(0, len(stored) - LEASE_PREFIX_WINDOW) :]
    if stored_window != actual:
        return MISS_TOKEN
    indices = [int(x) for x in commit_candidate_indices]
    if any(i < 0 or i >= len(lease.draft_tokens) for i in indices):
        return MISS_PATH
    expected = [int(lease.draft_tokens[i]) for i in indices]
    if list(path_tokens) != expected:
        return MISS_TOKEN
    slots = live_accept_prefix(lease.candidate_slots, indices)
    if len(slots) < MIN_TREE_KV_REUSE_DEPTH:
        return MISS_DEPTH
    return None


class SRTreeLeaseStore:
    def __init__(self) -> None:
        self._leases: Dict[str, SRTreeKVLease] = {}
        self._next_version = 1
        self.counts: Dict[str, int] = {
            "tree_lease_created": 0,
            "tree_lease_released": 0,
            "tree_lease_evicted": 0,
            "tree_lease_pages": 0,
            "tree_lease_skip_budget": 0,
            "tree_kv_commit_hit": 0,
            "tree_kv_reuse_skip_depth": 0,
            "tree_kv_reused_tokens": 0,
            "tree_kv_unmaterialized_leaf_tokens": 0,
            "tree_kv_overwritten_candidates": 0,
            "tree_kv_copy_submit_ms": 0,
            "tree_kv_copy_device_ms": 0,
        }
        for key in (
            MISS_VERSION,
            MISS_BASE,
            MISS_REVISION,
            MISS_PATH,
            MISS_TOKEN,
            MISS_DEPTH,
            MISS_FIELDS,
        ):
            self.counts[f"tree_kv_commit_miss_{key}"] = 0

    def next_version(self) -> int:
        ver = self._next_version
        self._next_version += 1
        return ver

    def get(self, rid: str) -> Optional[SRTreeKVLease]:
        return self._leases.get(rid)

    def live_pages(self) -> int:
        return sum(lease_page_count(lease) for lease in self._leases.values())

    def register(self, lease: SRTreeKVLease) -> None:
        previous = self._leases.get(lease.rid)
        if previous is not None and not previous.in_use:
            self.release(previous, evicted=True)
        self._leases[lease.rid] = lease
        self.counts["tree_lease_created"] += 1
        self.counts["tree_lease_pages"] += lease_page_count(lease)

    def pin(self, rid: str) -> Optional[SRTreeKVLease]:
        return self.pin_lease(self._leases.get(rid))

    def pin_lease(self, lease: Optional[SRTreeKVLease]) -> Optional[SRTreeKVLease]:
        """Pin this exact object if it is still the stored version."""
        if lease is None or lease.released:
            return None
        stored = self._leases.get(lease.rid)
        if stored is not lease or int(stored.version) != int(lease.version):
            return None
        lease.in_use = True
        return lease

    def unpin(self, rid: str) -> None:
        stored = self._leases.get(rid)
        if stored is not None:
            stored.in_use = False

    def unpin_lease(self, lease: Optional[SRTreeKVLease]) -> None:
        if lease is None:
            return
        stored = self._leases.get(lease.rid)
        if stored is lease:
            lease.in_use = False

    def pop(self, rid: str) -> Optional[SRTreeKVLease]:
        return self._leases.pop(rid, None)

    def release(
        self,
        lease: Optional[SRTreeKVLease],
        *,
        allocator=None,
        evicted: bool = False,
        event=None,
    ) -> None:
        if lease is None or lease.released:
            return
        stored = self._leases.get(lease.rid)
        if stored is lease:
            self._leases.pop(lease.rid, None)
        lease.in_use = False
        lease.released = True
        if evicted:
            self.counts["tree_lease_evicted"] += 1
        self.counts["tree_lease_released"] += 1
        self.counts["tree_lease_pages"] = max(
            0, self.counts["tree_lease_pages"] - lease_page_count(lease)
        )
        if allocator is None:
            return
        if event is None or lease.copy_unresolved:
            event = lease.pending_free_event
        if event is not None or lease.copy_unresolved:
            lease.pending_free_event = event
            self._pending_frees(allocator).append(lease)
            return
        _free_lease_pages(allocator, lease)

    def reclaim_idle(
        self, allocator, need_pages: int = 0, skip_rids: Optional[set] = None
    ) -> int:
        """Free leases that are not in_use. Returns pages freed."""
        skip = skip_rids or set()
        self.poll_pending_frees(allocator)
        freed = 0
        for rid, lease in list(self._leases.items()):
            if rid in skip or lease.in_use:
                continue
            n = lease_page_count(lease)
            self.release(lease, allocator=allocator, evicted=True)
            freed += n
            if need_pages > 0 and freed >= need_pages:
                break
        return freed

    def release_rid(self, rid: str, allocator=None, event=None) -> None:
        self.release(self.pop(rid), allocator=allocator, event=event)

    def release_all(self, allocator=None) -> None:
        for lease in list(self._leases.values()):
            if lease.in_use:
                continue
            self.release(lease, allocator=allocator, evicted=True)

    def poll_pending_frees(self, allocator) -> None:
        pending = self._pending_frees(allocator)
        keep = []
        ready = []
        for lease in pending:
            if lease.copy_unresolved:
                keep.append(lease)
                continue
            event = lease.pending_free_event
            if event is None:
                ready.append(lease)
                continue
            query = getattr(event, "query", None)
            try:
                done = callable(query) and query()
            except Exception as exc:
                from sglang.srt.speculative.standalone_remote.sr_transfer_staging import (
                    SRTransferUnresolved,
                )

                lease.copy_unresolved = True
                raise SRTransferUnresolved(
                    "tree lease completion query failed; retain pages"
                ) from exc
            if not done:
                keep.append(lease)
                continue
            ready.append(lease)
        # Query the complete set before freeing anything. A failed query must
        # not leave an already freed lease in the pending list for a retry.
        for lease in ready:
            _free_lease_pages(allocator, lease)
        pending[:] = keep

    def _pending_frees(self, allocator) -> List[SRTreeKVLease]:
        bag = getattr(allocator, "_sr_tree_lease_pending_frees", None)
        if bag is None:
            bag = []
            setattr(allocator, "_sr_tree_lease_pending_frees", bag)
        return bag


def _free_lease_pages(allocator, lease: SRTreeKVLease) -> None:
    slots = lease.page_slots
    if slots is None or (torch.is_tensor(slots) and slots.numel() == 0):
        return
    try:
        allocator.free(slots)
    except Exception:
        logger.warning("[SR] tree lease free failed for %s", lease.rid, exc_info=True)


def page_ids_from_alloc(allocated: torch.Tensor, page_size: int) -> List[int]:
    ps = int(page_size)
    if allocated is None or allocated.numel() == 0:
        return []
    pages = allocated.view(-1, ps)[:, 0] // ps
    return [int(x) for x in pages.detach().to("cpu").tolist()]


def immediate_free_pages(allocator) -> int:
    merge = getattr(allocator, "merge_and_sort_free", None)
    if callable(merge):
        merge()
    free_pages = getattr(allocator, "free_pages", None)
    if free_pages is None:
        return 0
    return int(len(free_pages))


def record_device_event(device, stream=None, *, required=False):
    """Record a completion event on stream or the current stream. CPU returns None."""
    if device is None:
        return None
    dev_type = getattr(device, "type", None) or str(device)
    if dev_type == "cpu" or "cpu" in str(dev_type):
        return None
    try:
        module = torch.get_device_module(dev_type)
        event = module.Event()
        if stream is not None:
            try:
                event.record(stream)
            except TypeError:
                event.record()
        else:
            event.record()
        return event
    except Exception:
        if required:
            raise
        return None
