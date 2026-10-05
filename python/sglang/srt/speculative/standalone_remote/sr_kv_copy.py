"""Pool-owned, overlap-safe KV movement with reusable device staging.

This module is CPU-importable. Accelerator kernels are imported only after
layout admission. A move gathers *all* buffers before writing any destination.
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from sglang.srt.speculative.standalone_remote.sr_transfer_staging import (
    SRTransferUnresolved,
)

logger = logging.getLogger(__name__)


class UnsupportedKVMoveLayout(RuntimeError):
    """The complete pool cannot be described without copying its storage."""


_NPU_PAGED_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.uint8,
)


def require_npu_paged_dtype(dtype):
    """NPU's two-launch kernel copies these storage dtypes. Others are errors."""
    if dtype not in _NPU_PAGED_DTYPES:
        raise UnsupportedKVMoveLayout(
            f"NPU paged KV dtype {dtype} has no scratch kernel"
        )


def _npu_format(tensor):
    if tensor.device.type != "npu":
        return None
    import torch_npu

    return torch_npu.get_npu_format(tensor)


def _require_nd(tensor, role):
    actual = _npu_format(tensor)
    if actual != 2 or not tensor.is_contiguous():
        raise UnsupportedKVMoveLayout(
            f"NPU paged KV role={role} requires contiguous ND storage: "
            f"actual_format={actual} expected_format=2 shape={tuple(tensor.shape)} "
            f"stride={tuple(tensor.stride())} dtype={tensor.dtype} device={tensor.device}"
        )


def _empty_nd(shape, *, dtype, device, role):
    """Allocate ND backing before forming the logical four-dimensional view.

    Direct four-dimensional allocation can infer NCHW even when ND is requested.
    Keep the backing explicitly with its consumer; never cast live KV storage.
    """
    import torch_npu

    shape = tuple(shape)
    backing = torch_npu.empty_with_format(
        (math.prod(shape),), dtype=dtype, device=device, acl_format=2
    )
    _require_nd(backing, role + ".backing")
    tensor = backing.view(shape)
    _require_nd(tensor, role)
    stride = []
    size = 1
    for dim in reversed(shape):
        stride.append(size)
        size *= dim
    if (
        tuple(backing.shape) != (size,)
        or tuple(tensor.stride()) != tuple(reversed(stride))
        or tensor.data_ptr() != backing.data_ptr()
        or tensor.storage_offset() != 0
        or tensor.dtype != dtype
        or backing.dtype != dtype
        or tensor.device != backing.device
        or tensor.device.type != device.type
        or (device.index is not None and tensor.device.index != device.index)
    ):
        raise UnsupportedKVMoveLayout(
            f"NPU paged KV role={role} ND allocation metadata mismatch: "
            f"actual_format={_npu_format(tensor)} expected_format=2 "
            f"shape={tuple(tensor.shape)} stride={tuple(tensor.stride())} "
            f"dtype={tensor.dtype} device={tensor.device}; "
            f"expected_shape={shape} expected_dtype={dtype} expected_device={device} "
            f"backing_ptr={backing.data_ptr()} view_ptr={tensor.data_ptr()}"
        )
    return tensor, backing


# Triton-Ascend rejects a grid whose program product is 65536 or larger.
_PAGED_GRID_LIMIT = 65535


def paged_launch_chunks(n, groups, width, block=256):
    """Slot and column ranges whose grids stay under the Triton-Ascend limit.

    A range is ``(slot_base, slot_count, tile_base, tile_count)``. When the
    whole move fits, the result is one range. Otherwise slots are split
    first and tiles second, and the ranges together cover every slot and
    tile once. ``groups`` occupies axis 0 of a single slot, so it cannot
    itself reach the limit.
    """
    n = int(n)
    groups = int(groups)
    width = int(width)
    block = int(block)
    if n <= 0 or width <= 0:
        return []
    if groups <= 0 or groups > _PAGED_GRID_LIMIT:
        raise RuntimeError("KV move groups do not fit one Triton program axis")
    tiles = (width + block - 1) // block
    if n * groups * tiles <= _PAGED_GRID_LIMIT:
        return [(0, n, 0, tiles)]
    tile_chunk = min(tiles, _PAGED_GRID_LIMIT // groups)
    slot_chunk = max(1, _PAGED_GRID_LIMIT // (groups * tile_chunk))
    chunks = []
    for slot_base in range(0, n, slot_chunk):
        slot_count = min(slot_chunk, n - slot_base)
        for tile_base in range(0, tiles, tile_chunk):
            tile_count = min(tile_chunk, tiles - tile_base)
            chunks.append((slot_base, slot_count, tile_base, tile_count))
    return chunks


class KVMoveSubmittedError(SRTransferUnresolved):
    """Movement was submitted; retain its storage and never replay the move."""


@dataclass(frozen=True)
class KVMoveBuffer:
    tensor: torch.Tensor
    axis: int
    backing: torch.Tensor | None = None

    @property
    def signature(self):
        t = self.tensor
        return (
            t.data_ptr(),
            tuple(t.shape),
            tuple(t.stride()),
            t.dtype,
            t.device,
            self.axis,
        )


def _source_signature(value):
    if torch.is_tensor(value):
        return (
            type(value),
            id(value),
            value.data_ptr(),
            tuple(value.shape),
            tuple(value.stride()),
            value.dtype,
            value.device,
            value.requires_grad,
            _npu_format(value),
        )
    if isinstance(value, (list, tuple)):
        return type(value), tuple(_source_signature(t) for t in value)
    return type(value), id(value)


@dataclass(frozen=True)
class KVMoveLayout:
    buffers: tuple[KVMoveBuffer, ...]
    kind: str
    sources: tuple = ()
    role: str = "live_pool"

    @property
    def device(self):
        return self.buffers[0].tensor.device

    @property
    def signature(self):
        return self.kind, tuple(b.signature for b in self.buffers)

    def check_pool(self, pool):
        """Check live storage metadata without views or alias analysis."""
        if self.device.type == "npu" and self.kind == "paged6":
            source = getattr(pool, "kv_buffer", None)
            if torch.is_tensor(source):
                _require_nd(source, self.role)
            _require_nd(self.buffers[0].tensor, self.role + ".view")
        for name, signature in self.sources:
            if _source_signature(getattr(pool, name, None)) != signature:
                raise UnsupportedKVMoveLayout("KV backing storage changed")

    @classmethod
    def from_pool(cls, pool):
        kv = getattr(pool, "kv_buffer", None)
        k = getattr(pool, "k_buffer", None)
        v = getattr(pool, "v_buffer", None)
        index = getattr(pool, "index_k_buffer", None)

        def one(t, *, paged=False):
            if not torch.is_tensor(t) or t.ndim < 1 or t.requires_grad:
                raise UnsupportedKVMoveLayout("KV buffers must be non-grad tensors")
            if str(t.dtype).startswith("torch.float8"):
                # KV movement copies storage bits, including FP8 encodings.
                t = t.view(torch.uint8)
            if paged:
                if t.ndim != 6 or t.shape[0] != 2 or not t.is_contiguous():
                    raise UnsupportedKVMoveLayout(
                        "paged KV must be contiguous [2,L,P,S,H,D]"
                    )
                if t.device.type == "npu":
                    _require_nd(t, "live_pool")
                result = KVMoveBuffer(
                    t.view(2, t.shape[1], -1, t.shape[4] * t.shape[5]), 2
                )
                if t.device.type == "npu":
                    _require_nd(result.tensor, "live_pool.view")
                return result
            if t.ndim >= 5:
                try:
                    t = t.view(t.shape[0], -1, *t.shape[3:])
                except RuntimeError as exc:
                    raise UnsupportedKVMoveLayout(
                        "paged token axes cannot be viewed"
                    ) from exc
                return KVMoveBuffer(t, 1)
            return KVMoveBuffer(t, 0)

        def collection(value):
            if torch.is_tensor(value):
                return [one(value)]
            if isinstance(value, (list, tuple)) and value:
                return [one(t) for t in value]
            raise UnsupportedKVMoveLayout("KV collection must contain tensors")

        if torch.is_tensor(kv) and kv.ndim == 6:
            if index is not None:
                raise UnsupportedKVMoveLayout(
                    "six-dimensional KV with extra index storage"
                )
            buffers, kind = [one(kv, paged=True)], "paged6"
        elif k is not None:
            kb, vb = collection(k), collection(v)
            if len(kb) != len(vb):
                raise UnsupportedKVMoveLayout("K/V list lengths do not match")
            if torch.is_tensor(k) != torch.is_tensor(v):
                raise UnsupportedKVMoveLayout("K/V collection kinds do not match")
            if torch.is_tensor(k) and k.ndim >= 5:
                if v.ndim != k.ndim or tuple(k.shape[:3]) != tuple(v.shape[:3]):
                    raise UnsupportedKVMoveLayout(
                        "stacked K/V layer or page axes do not match"
                    )
            buffers, kind = kb + vb, "token"
            if index is not None:
                ib = collection(index)
                if not torch.is_tensor(index) and len(ib) != len(kb):
                    raise UnsupportedKVMoveLayout("index list length does not match K")
                if torch.is_tensor(k) and (len(ib) != 1 or ib[0].axis != kb[0].axis):
                    raise UnsupportedKVMoveLayout(
                        "index storage does not match stacked K"
                    )
                if (
                    torch.is_tensor(k)
                    and k.ndim >= 5
                    and (
                        not torch.is_tensor(index)
                        or index.ndim != k.ndim
                        or tuple(index.shape[:3]) != tuple(k.shape[:3])
                    )
                ):
                    raise UnsupportedKVMoveLayout(
                        "stacked index layer or page axes do not match K"
                    )
                buffers += ib
        elif isinstance(kv, (list, tuple)):
            buffers, kind = collection(kv), "token"
        else:
            raise UnsupportedKVMoveLayout("unrecognized KV pool layout")
        first = buffers[0]
        slots = first.tensor.shape[first.axis]
        for b in buffers:
            if (
                b.tensor.device != first.tensor.device
                or b.tensor.shape[b.axis] != slots
            ):
                raise UnsupportedKVMoveLayout(
                    "KV token capacities or devices do not match"
                )
            if any(s <= 0 for s in b.tensor.shape):
                raise UnsupportedKVMoveLayout("KV storage has an empty dimension")
        # Exact aliases need only one snapshot. Reject partial aliases rather
        # than giving different buffers conflicting scatter semantics.
        unique = []
        for b in buffers:
            if any(b.signature == old.signature for old in unique):
                continue
            for old in unique:
                if (
                    b.tensor.untyped_storage().data_ptr()
                    == old.tensor.untyped_storage().data_ptr()
                ):

                    def span(t):
                        return (
                            t.data_ptr(),
                            t.data_ptr()
                            + (
                                1
                                + sum(
                                    (n - 1) * st for n, st in zip(t.shape, t.stride())
                                )
                            )
                            * t.element_size(),
                        )

                    lo, hi = span(b.tensor)
                    old_lo, old_hi = span(old.tensor)
                    if lo < old_hi and old_lo < hi:
                        raise UnsupportedKVMoveLayout("partially aliased KV buffers")
            unique.append(b)
        sources = tuple(
            (name, _source_signature(getattr(pool, name, None)))
            for name in ("kv_buffer", "k_buffer", "v_buffer", "index_k_buffer")
        )
        return cls(tuple(unique), kind, sources)


def _stream(device):
    if device.type == "cpu":
        return None
    return torch.get_device_module(device.type).current_stream(device)


def _stream_key(stream):
    if stream is None:
        return None
    for name in ("stream_id", "cuda_stream", "npu_stream"):
        value = getattr(stream, name, None)
        if value is not None:
            return int(value)
    raise RuntimeError("backend stream does not expose a stable identity")


def _capturing(device):
    if device.type == "cpu":
        return False
    fn = getattr(
        torch.get_device_module(device.type), "is_current_stream_capturing", None
    )
    return bool(fn and fn())


class KVMoveWorkspace:
    def __init__(self, pool, layout, *, graph=False, metrics=None):
        self.pool = pool
        self.layout = layout
        self.graph = bool(graph)
        self.metrics = metrics
        self.stream = _stream(layout.device)
        self.capacity = 0
        self.scratch = []
        self.scratch_backings = []
        self.index_scratch = None
        self.retired = []
        self.hold = None
        self.unresolved = False
        self.frozen = False
        self._inside_capture = False
        self._used = False
        self.backend = "torch_out"
        self.kernels = None
        self.counts = {"grow": 0, "calls": 0, "slots": 0, "bytes": 0}
        self.scratch_bytes = 0
        self.retired_bytes = 0
        self.peak_bytes = 0
        self._retired_sizes = {}
        if layout.device.type == "npu" and layout.kind == "paged6":
            b = layout.buffers[0].tensor
            require_npu_paged_dtype(b.dtype)
            layout.check_pool(pool)
            _require_nd(b, layout.role)
            from sglang.srt.speculative.standalone_remote import sr_kv_copy_kernels

            self.kernels = sr_kv_copy_kernels
            self.backend = "npu_paged6"

    def _count(self, key, value=1):
        self.counts[key] += value
        counts = getattr(self.metrics, "counts", None)
        if counts is not None:
            name = "kv_move_" + key
            counts[name] = counts.get(name, 0) + value

    def check(self, *, check_stream=True):
        if self.unresolved or getattr(self.pool, "_kv_move_unresolved", False):
            raise KVMoveSubmittedError("KV movement completion is unresolved")
        self.layout.check_pool(self.pool)
        if check_stream and not self._inside_capture:
            if _stream_key(_stream(self.layout.device)) != _stream_key(self.stream):
                raise RuntimeError("KV workspace belongs to another execution stream")

    def reserve(self, count):
        self.check()
        count = int(count)
        if count < 0:
            raise ValueError("negative KV move capacity")
        if not self._inside_capture and not _capturing(self.layout.device):
            self._collect_retired()
        if count <= self.capacity:
            return self
        if self.frozen or _capturing(self.layout.device):
            raise RuntimeError(
                "KV scratch cannot grow during capture or after graph binding"
            )
        cap = 1 << (count - 1).bit_length()
        new = []
        new_backings = []
        for b in self.layout.buffers:
            shape = list(b.tensor.shape)
            shape[b.axis] = cap
            if self.backend == "npu_paged6":
                tensor, backing = _empty_nd(
                    shape, dtype=b.tensor.dtype, device=b.tensor.device, role="scratch"
                )
                new.append(tensor)
                new_backings.append(backing)
            else:
                new.append(
                    torch.empty(shape, dtype=b.tensor.dtype, device=b.tensor.device)
                )
        index = self.index_scratch
        if self.kernels is None:
            index = torch.empty((2, cap), dtype=torch.int64, device=self.layout.device)
        tree_index = None
        if hasattr(self, "tree_indices"):
            tree_index = torch.empty(
                (2, cap), dtype=self.tree_indices.dtype, device=self.layout.device
            )
        # Commit all allocations together. A failed integer-buffer allocation
        # must not leave a larger capacity paired with the old small indices.
        if self.scratch and self._used and self.layout.device.type != "cpu":
            event = self._record_event()
            self.retired.append(
                (
                    event,
                    self.scratch,
                    self.index_scratch,
                    getattr(self, "tree_indices", None),
                    self.hold,
                    self.scratch_backings,
                )
            )
            self._retired_sizes[id(event)] = self.scratch_bytes
            self.retired_bytes += self.scratch_bytes
        self.scratch = new
        self.scratch_backings = new_backings
        self.index_scratch = index
        if tree_index is not None:
            self.tree_indices = tree_index
        self.capacity = cap
        self._refresh_memory()
        self.peak_bytes = max(self.peak_bytes, self.scratch_bytes + self.retired_bytes)
        self._count("grow")
        self._note_memory()
        logger.info(
            "[SR] KV move backend=%s graph=%s capacity=%s scratch_bytes=%s retired_bytes=%s peak_bytes=%s",
            self.backend,
            self.graph,
            cap,
            self.scratch_bytes,
            self.retired_bytes,
            self.peak_bytes,
        )
        return self

    def _poison(self):
        self.unresolved = True
        self.pool._kv_move_unresolved = True

    def _record_event(self):
        try:
            ev = torch.get_device_module(self.layout.device.type).Event()
            ev.record(_stream(self.layout.device))
            return ev
        except BaseException as exc:
            self._poison()
            raise KVMoveSubmittedError(
                "cannot confirm KV scratch consumer completion"
            ) from exc

    def _collect_retired(self):
        kept = []
        completed = []
        try:
            for item in self.retired:
                if not item[0].query():
                    kept.append(item)
                else:
                    completed.append(id(item[0]))
        except BaseException as exc:
            self._poison()
            raise KVMoveSubmittedError("KV scratch retirement query failed") from exc
        for key in completed:
            self.retired_bytes -= self._retired_sizes.pop(key, 0)
        self.retired = kept
        if completed:
            self._note_memory()

    def _note_memory(self):
        workspaces = getattr(self.pool, "_kv_move_workspaces", {}).values()
        workspaces = list(workspaces) or [self]
        current = sum(w.scratch_bytes for w in workspaces)
        retired = sum(w.retired_bytes for w in workspaces)
        peak = max(getattr(self.pool, "_kv_move_peak_bytes", 0), current + retired)
        self.pool._kv_move_peak_bytes = peak
        counts = getattr(self.metrics, "counts", None)
        if counts is None:
            return
        counts["kv_move_scratch_bytes"] = current
        counts["kv_move_retired_bytes"] = retired
        counts["kv_move_peak_bytes"] = peak

    def _refresh_memory(self):
        tensors = self.scratch + [
            self.index_scratch,
            getattr(self, "tree_indices", None),
        ]
        self.scratch_bytes = sum(
            t.numel() * t.element_size() for t in tensors if t is not None
        )
        self.peak_bytes = max(self.peak_bytes, self.scratch_bytes + self.retired_bytes)
        self._note_memory()

    @contextmanager
    def capture_scope(self):
        if not self.graph:
            raise RuntimeError("eager workspace cannot bind a graph")
        self.check(check_stream=False)
        self._inside_capture = True
        try:
            yield self
        finally:
            self._inside_capture = False
            self.frozen = True

    def account(self, count, calls=1):
        # Captured Python calls are not replay counts. The graph runner calls
        # account explicitly after a successful replay submission.
        if self._inside_capture:
            return
        self._count("calls", calls)
        self._count("slots", int(count))
        per_slot = sum(
            b.tensor.numel() // b.tensor.shape[b.axis] * b.tensor.element_size()
            for b in self.layout.buffers
        )
        self._count("bytes", int(count) * per_slot)
        self._note_memory()
        counts = getattr(self.metrics, "counts", None)
        if counts is not None:
            name = "kv_move_backend_" + self.backend
            counts[name] = counts.get(name, 0) + calls
            scope = "graph" if self.graph else "eager"
            counts["kv_move_" + scope + "_calls"] = (
                counts.get("kv_move_" + scope + "_calls", 0) + calls
            )

    def submitted(self, inputs, run):
        self.hold = (inputs, self.scratch, self.scratch_backings)
        self._used = True
        try:
            stream = _stream(self.layout.device)
            if stream is not None and not self._inside_capture:
                for tensor in inputs:
                    tensor.record_stream(stream)
                for tensor in self.scratch:
                    tensor.record_stream(stream)
            phase = getattr(self.metrics, "phase", None)
            scope = (
                phase("kv_move_eager", device=True, stream=stream)
                if callable(phase) and not self._inside_capture
                else nullcontext()
            )
            with scope:
                run()
        except BaseException as exc:
            self._poison()
            raise KVMoveSubmittedError(
                "KV gather/scatter failed after submission; refuse retry"
            ) from exc


_TREE_EXPERIMENT_DOMAINS = frozenset({"tree_eager", "tree_graph"})


def cuda_tree_experiment_eligible(pool, layout, domain):
    """Contiguous CUDA MHA/GQA tree remaps measured faster than per-buffer ATen."""
    if domain not in _TREE_EXPERIMENT_DOMAINS:
        return False
    if layout.device.type != "cuda" or layout.kind != "token":
        return False
    if (
        not isinstance(getattr(pool, "k_buffer", None), (list, tuple))
        or not isinstance(getattr(pool, "v_buffer", None), (list, tuple))
        or getattr(pool, "index_k_buffer", None) is not None
    ):
        return False
    from sglang.srt.speculative.standalone_remote.sr_kv_copy_cuda_experiment import (
        continuous_groups,
    )

    try:
        continuous_groups(layout)
    except UnsupportedKVMoveLayout:
        return False
    return True


def _workspace_key(domain, graph, stream_key, *, experiment):
    # Tree experiments own their pointer tables. Other eager users still share
    # one torch_out workspace; captured graphs stay named by runner domain.
    if experiment:
        return (("cuda_tree", domain, bool(graph)), stream_key)
    return (("graph", domain) if graph else "eager", stream_key)


def _collect_parked_tree_experiments(pool):
    parked = getattr(pool, "_kv_tree_experiment_retired", None)
    if not parked:
        return
    kept = []
    try:
        for event, old in parked:
            if not event.query():
                kept.append((event, old))
    except BaseException as exc:
        pool._kv_move_unresolved = True
        raise KVMoveSubmittedError("KV scratch retirement query failed") from exc
    pool._kv_tree_experiment_retired = kept


def _park_tree_experiment(pool, workspace):
    event = workspace._record_event()
    parked = getattr(pool, "_kv_tree_experiment_retired", None)
    if parked is None:
        parked = pool._kv_tree_experiment_retired = []
    parked.append((event, workspace))


def prepare_kv_move(pool, capacity, *, domain="eager", graph=False, metrics=None):
    layout = getattr(pool, "_kv_move_layout", None)
    if layout is None:
        layout = KVMoveLayout.from_pool(pool)
        pool._kv_move_layout = layout
    else:
        layout.check_pool(pool)
    registry = getattr(pool, "_kv_move_workspaces", None)
    if registry is None:
        registry = pool._kv_move_workspaces = {}
    capacity = int(capacity)
    admit = capacity > 0 and cuda_tree_experiment_eligible(pool, layout, domain)
    key = _workspace_key(
        domain, graph, _stream_key(_stream(layout.device)), experiment=admit
    )
    if not _capturing(layout.device):
        _collect_parked_tree_experiments(pool)
    ws = registry.get(key)
    if (
        ws is not None
        and admit
        and ws.backend == "cuda_contiguous_experiment"
        and capacity > ws.capacity
    ):
        if ws.graph or ws._inside_capture or _capturing(layout.device):
            raise RuntimeError(
                "KV scratch cannot grow during capture or after graph binding"
            )
        # Build the replacement first. A failed allocation leaves the captured
        # or in-flight pointer tables registered.
        from sglang.srt.speculative.standalone_remote.sr_kv_copy_cuda_experiment import (
            CudaKVExperiment,
        )

        replacement = CudaKVExperiment(pool, capacity, graph=graph)
        _park_tree_experiment(pool, ws)
        ws = replacement
        registry[key] = ws
    if ws is None:
        if _capturing(layout.device):
            raise RuntimeError("KV workspace must be prepared before graph capture")
        if admit:
            from sglang.srt.speculative.standalone_remote.sr_kv_copy_cuda_experiment import (
                CudaKVExperiment,
            )

            ws = CudaKVExperiment(pool, capacity, graph=graph)
        else:
            ws = KVMoveWorkspace(pool, layout, graph=graph, metrics=metrics)
        registry[key] = ws
    if ws.graph != bool(graph):
        raise RuntimeError("KV graph/eager workspace domain mismatch")
    ws.metrics = metrics
    return ws.reserve(capacity)


def warm_private_slot_move(
    pool,
    count=2,
    *,
    index_dtype=torch.int64,
    dst_index_dtype=None,
    src_stride=1,
    dst_stride=1,
):
    """Warm production geometry using small private storage, once per signature.

    PoolSlots, Capacity and chunk lengths are runtime kernel parameters. Only
    device, storage types, non-token dimensions and index strides specialize.
    """
    count = int(count)
    if count <= 0:
        return
    layout = getattr(pool, "_kv_move_layout", None)
    if layout is None:
        layout = KVMoveLayout.from_pool(pool)
        pool._kv_move_layout = layout
    layout.check_pool(pool)
    dst_index_dtype = index_dtype if dst_index_dtype is None else dst_index_dtype
    if (
        index_dtype not in (torch.int32, torch.int64)
        or dst_index_dtype not in (torch.int32, torch.int64)
        or min(src_stride, dst_stride) < 1
    ):
        raise ValueError("invalid KV warmup index geometry")
    key = (
        layout.kind,
        layout.device,
        index_dtype,
        dst_index_dtype,
        src_stride,
        dst_stride,
        tuple(
            (
                b.axis,
                b.tensor.dtype,
                tuple(b.tensor.shape[: b.axis]),
                tuple(b.tensor.shape[b.axis + 1 :]),
            )
            for b in layout.buffers
        ),
    )
    warmed = getattr(pool, "_kv_move_warmups", None)
    if warmed is None:
        warmed = pool._kv_move_warmups = {}
    if key in warmed:
        if not warmed[key].complete:
            raise KVMoveSubmittedError("previous KV warmup completion is unresolved")
        return
    if _capturing(layout.device):
        raise RuntimeError("KV slot warmup must precede graph capture")
    # Keep both length-one and length-two launches on the same specialization.
    private = []
    for b in layout.buffers:
        shape = list(b.tensor.shape)
        shape[b.axis] = 3
        if layout.device.type == "npu" and layout.kind == "paged6":
            tensor, backing = _empty_nd(
                shape, dtype=b.tensor.dtype, device=layout.device, role="private_warmup"
            )
            # Initialize the one-dimensional allocation, not a four-dimensional
            # operand that a backend operator could re-infer as NCHW.
            backing.zero_()
            private.append(KVMoveBuffer(tensor, b.axis, backing))
        else:
            private.append(
                KVMoveBuffer(
                    torch.zeros(shape, dtype=b.tensor.dtype, device=layout.device),
                    b.axis,
                )
            )
    owner = SimpleNamespace()
    ws = KVMoveWorkspace(
        owner, KVMoveLayout(tuple(private), layout.kind, role="private_warmup")
    )
    ws.reserve(2)
    src = torch.zeros(2 * src_stride, dtype=index_dtype, device=layout.device)[
        ::src_stride
    ]
    dst = torch.zeros(2 * dst_stride, dtype=dst_index_dtype, device=layout.device)[
        ::dst_stride
    ]
    src.copy_(torch.arange(2, dtype=index_dtype, device=layout.device))
    dst.copy_(torch.arange(1, -1, -1, dtype=dst_index_dtype, device=layout.device))
    # Retain the private tensors even if completion or compilation fails.
    entry = SimpleNamespace(workspace=ws, src=src, dst=dst, complete=False)
    warmed[key] = entry
    try:
        move_kv_slots_(ws, src[:1], dst[:1])
        move_kv_slots_(ws, src, dst)
        if layout.device.type != "cpu":
            _stream(layout.device).synchronize()
        entry.complete = True
    except BaseException:
        ws._poison()
        raise
    pool._kv_move_warmup_count = getattr(pool, "_kv_move_warmup_count", 0) + 1
    logger.info(
        "[SR] KV slot warmup compiled signature=%s private_bytes=%s",
        key,
        ws.scratch_bytes,
    )
    # Successful startup synchronization allows these private tensors to retire.
    # Incomplete entries retain all references and block retry instead.
    entry.workspace = entry.src = entry.dst = None


def bind_kernel_pointers(src, dst, slots, parents, active):
    """Fill unused Triton pointer arguments with a tensor from this launch."""
    tensors = (src, dst, slots, parents, active)
    base = next((tensor for tensor in tensors if tensor is not None), None)
    if base is None:
        raise RuntimeError("KV kernel launch has no tensor pointer")
    return tuple(tensor if tensor is not None else base for tensor in tensors)


def kernel_pointer_launch(src, dst, slots, parents, active):
    """Return mode flags, original strides, then non-None pointer arguments.

    ``Tree`` and ``HasActive`` follow the caller's tensors. Strides are read
    before empty arguments are replaced, so a stand-in cannot change them.
    """
    tree = slots is not None
    has_active = active is not None
    strides = (
        src.stride(0) if src is not None else 0,
        dst.stride(0) if dst is not None else 0,
        slots.stride(0) if slots is not None else 0,
        slots.stride(1) if slots is not None else 0,
        parents.stride(0) if parents is not None else 0,
        active.stride(0) if active is not None else 0,
    )
    return (
        tree,
        has_active,
        strides,
        bind_kernel_pointers(src, dst, slots, parents, active),
    )


def _indices(ws, src, dst):
    nsrc = 0 if src is None else src.numel()
    ndst = 0 if dst is None else dst.numel()
    if nsrc != ndst:
        raise RuntimeError("KV move src/dst length mismatch")
    if not nsrc:
        return None, None
    ws.check()
    if nsrc > ws.capacity:
        raise RuntimeError("KV move exceeds prepared scratch capacity")
    for t in (src, dst):
        if t.device != ws.layout.device or t.dtype not in (torch.int32, torch.int64):
            raise RuntimeError("KV indices must be device-local integers")
    src, dst = src.reshape(-1), dst.reshape(-1)
    if ws.layout.device.type == "cpu":
        size = ws.layout.buffers[0].tensor.shape[ws.layout.buffers[0].axis]
        if bool(((src < 0) | (src >= size) | (dst < 0) | (dst >= size)).any()):
            raise IndexError("KV slot outside pool")
    return src, dst


def _portable_move(workspace, src, dst):
    n = src.numel()
    if src.dtype != torch.int64:
        workspace.index_scratch[0, :n].copy_(src)
        src = workspace.index_scratch[0, :n]
    if dst.dtype != torch.int64:
        workspace.index_scratch[1, :n].copy_(dst)
        dst = workspace.index_scratch[1, :n]
    views = [
        s.narrow(b.axis, 0, n)
        for b, s in zip(workspace.layout.buffers, workspace.scratch)
    ]
    for b, out in zip(workspace.layout.buffers, views):
        torch.index_select(b.tensor, b.axis, src, out=out)
    for b, out in zip(workspace.layout.buffers, views):
        b.tensor.index_copy_(b.axis, dst, out)


def move_kv_slots_(workspace, src, dst):
    original_indices = (src, dst)
    src, dst = _indices(workspace, src, dst)
    if src is None:
        return
    n = src.numel()

    def run():
        if workspace.kernels is not None:
            workspace.kernels.move(
                workspace.layout.buffers[0].tensor,
                workspace.scratch[0],
                src,
                dst,
            )
        else:
            _portable_move(workspace, src, dst)

    workspace.submitted(original_indices + (src, dst), run)
    workspace.account(n)


def remap_tree_kv_(workspace, slots, parents, depth, active_rows=None):
    depth = int(depth)
    if depth <= 0:
        return
    workspace.check()
    if slots.ndim != 2 or parents.ndim != 1 or slots.shape[1] != parents.numel():
        raise RuntimeError("tree KV parent width must match slot rows")
    rows = parents.numel()
    if rows == 0:
        return
    if depth > slots.shape[0] or depth * rows > workspace.capacity:
        raise RuntimeError("tree KV depth exceeds prepared capacity")
    for t in (slots, parents):
        if t.device != workspace.layout.device or t.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise RuntimeError("tree KV mappings must be device-local integers")
    if active_rows is not None and (
        active_rows.ndim != 1
        or active_rows.numel() != rows
        or active_rows.device != slots.device
        or active_rows.dtype not in (torch.bool, torch.int32, torch.int64)
    ):
        raise RuntimeError("tree KV active mask mismatch")
    if getattr(workspace, "backend", None) == "cuda_contiguous_experiment":
        if active_rows is not None:
            raise RuntimeError("CUDA tree remap does not accept an active mask")
        # Same bytes as the portable path: every group gathers before scatter.
        workspace.remap(slots, parents, depth)
        return
    if workspace.kernels is not None:
        workspace.submitted(
            (slots, parents) + (() if active_rows is None else (active_rows,)),
            lambda: workspace.kernels.remap(
                workspace.layout.buffers[0].tensor,
                workspace.scratch[0],
                slots,
                parents,
                depth,
                active_rows,
            ),
        )
        workspace.account(depth * rows)
        return
    # The portable backend stages index mappings, too. Fixed slices have no
    # boolean compression and replay reads the live parents each time.
    index = getattr(workspace, "tree_indices", None)
    if index is None or index.dtype != slots.dtype:
        if workspace.frozen or _capturing(slots.device):
            raise RuntimeError("tree index scratch must be prepared before capture")
        replacement = torch.empty(
            (2, workspace.capacity), dtype=slots.dtype, device=slots.device
        )
        if index is not None and workspace._used and slots.device.type != "cpu":
            event = workspace._record_event()
            workspace.retired.append((event, index, workspace.hold))
            size = index.numel() * index.element_size()
            workspace._retired_sizes[id(event)] = size
            workspace.retired_bytes += size
        workspace.tree_indices = index = replacement
        workspace._refresh_memory()
    if slots.device.type == "cpu" and bool(((parents < 0) | (parents >= rows)).any()):
        raise IndexError("tree parent outside rows")
    if active_rows is not None and slots.device.type != "cpu":
        raise UnsupportedKVMoveLayout("masked tree remap requires the NPU paged kernel")

    def run():
        for step in range(depth):
            src = index[0, step * rows : (step + 1) * rows]
            dst = index[1, step * rows : (step + 1) * rows]
            torch.index_select(slots[step], 0, parents, out=src)
            dst.copy_(slots[step])
        if active_rows is None:
            _portable_move(
                workspace, index[0, : depth * rows], index[1, : depth * rows]
            )
            return
        # CPU reference masking skips dummy writes. NPU paged graphs use
        # the runtime mask in both stages of the dedicated kernel.
        for b, scratch in zip(workspace.layout.buffers, workspace.scratch):
            out = scratch.narrow(b.axis, 0, depth * rows)
            torch.index_select(b.tensor, b.axis, index[0, : depth * rows], out=out)
        for b, scratch in zip(workspace.layout.buffers, workspace.scratch):
            for j in range(depth * rows):
                if bool(active_rows[j % rows]):
                    b.tensor.index_copy_(
                        b.axis, index[1, j : j + 1].long(), scratch.narrow(b.axis, j, 1)
                    )

    workspace.submitted(
        (slots, parents, index) + (() if active_rows is None else (active_rows,)), run
    )
    workspace.account(depth * rows)
