"""Owned, stream-ordered parameter staging for SR paged attention metadata.

CPU plans contain no device reads. NPU kernels are imported only after complete
host validation. Upload completion protects pinned memory; consumer completion
separately protects outputs and the tensors read by the metadata kernels.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from sglang.srt.speculative.standalone_remote.sr_kv_copy import KVMoveSubmittedError
from sglang.srt.speculative.standalone_remote.sr_transfer_staging import alloc_host


class SRPagedMetadataSubmittedError(KVMoveSubmittedError):
    """Metadata or its consumer may be in flight: never retry or release leases."""


def stream_key(device):
    device = torch.device(device)
    if device.type == "cpu":
        return ("cpu", None, None)
    stream = torch.get_device_module(device.type).current_stream(device)
    return (device.type, stream.device.index, stream.stream_id)


def empty_metadata(shape, dtype, device):
    """Allocate a base ND backing, including for four-dimensional bool masks."""
    device = torch.device(device)
    if device.type != "npu":
        return torch.empty(shape, dtype=dtype, device=device)
    import torch_npu

    backing = torch_npu.empty_with_format(
        (math.prod(shape),), dtype=dtype, device=device, acl_format=2
    )
    result = backing.view(shape)
    for role, tensor in (("backing", backing), ("view", result)):
        if torch_npu.get_npu_format(tensor) != 2 or not tensor.is_contiguous():
            raise RuntimeError(
                f"paged metadata {role}: expected ND, got "
                f"format={torch_npu.get_npu_format(tensor)} shape={tensor.shape} "
                f"stride={tensor.stride()} dtype={tensor.dtype} device={tensor.device}"
            )
    if result.data_ptr() != backing.data_ptr() or result.storage_offset() != 0:
        raise RuntimeError("paged metadata view does not own the ND backing")
    # A view retains its backing through _base, including while retired.
    return result


def _identity(t):
    storage_format = None
    if t.device.type == "npu":
        import torch_npu

        storage_format = torch_npu.get_npu_format(t)
    return (
        t.data_ptr(),
        tuple(t.shape),
        tuple(t.stride()),
        t.dtype,
        t.device,
        storage_format,
    )


def _integer(t, device, ndim, name):
    if (
        t.device != device
        or t.ndim != ndim
        or t.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            f"paged metadata {name}: expected rank-{ndim} integer on {device}"
        )
    if any(s < 0 for s in t.stride()):
        raise ValueError(f"paged metadata {name}: negative stride")
    if device.type == "npu":
        import torch_npu

        if torch_npu.get_npu_format(t) != 2:
            raise ValueError(f"paged metadata {name}: input must have ND storage")


def _prefix_tuple(prefixes):
    if isinstance(prefixes, torch.Tensor):
        if prefixes.device.type != "cpu" or prefixes.ndim != 1:
            raise ValueError("paged metadata lengths must be rank-1 CPU data")
        prefixes = prefixes.tolist()
    return tuple(int(p) for p in prefixes)


@dataclass(frozen=True)
class SRTargetPagedPlan:
    prefixes: tuple
    capacity: int
    queries: int
    pages: int
    page_size: int
    params: tuple
    kv_lens: tuple

    @classmethod
    def build(cls, prefixes, capacity, queries, pages, page_size):
        prefixes = _prefix_tuple(prefixes)
        if capacity < len(prefixes) or queries < 1 or pages < 1 or page_size < 1:
            raise ValueError("invalid target metadata capacity")
        if any(p < 0 or p + queries > pages * page_size for p in prefixes):
            raise ValueError("target prefix exceeds metadata capacity")
        offset, params = 0, []
        for p in prefixes:
            params.append((p, offset, 1))
            offset += queries * (p + queries)
        params.extend([(0, 0, 0)] * (capacity - len(prefixes)))
        return cls(
            prefixes,
            capacity,
            queries,
            pages,
            page_size,
            tuple(params),
            tuple(p + queries for p in prefixes) + (1,) * (capacity - len(prefixes)),
        )


@dataclass(frozen=True)
class SRDraftPagedPlan:
    prefixes: tuple
    capacity: int
    topk: int
    steps: int
    pages: int
    page_size: int
    branch_capacity: int
    params: tuple
    domain: str

    @classmethod
    def build(cls, prefixes, capacity, topk, steps, pages, page_size, domain="eager"):
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_paged_layout import (
            max_query_pages_for_tree,
            pages_per_branch,
            shared_page_count,
        )

        prefixes = _prefix_tuple(prefixes)
        if capacity < len(prefixes) or min(topk, steps, pages, page_size) < 1:
            raise ValueError("invalid draft metadata capacity")
        if any(p < 0 for p in prefixes):
            raise ValueError("negative draft prefix")
        totals = max_query_pages_for_tree(prefixes, steps, page_size)
        if max(totals, default=0) > pages:
            raise ValueError("draft query pages exceed bucket")
        params = []
        for p, total in zip(prefixes, totals):
            rem = p % page_size
            shared = shared_page_count(p, page_size)
            reserved = pages_per_branch(rem, steps, page_size)
            query = max(total - shared, 0)
            if query > reserved:
                raise ValueError("query branch pages exceed reserved pages")
            params.append((rem, shared, query, reserved, 1))
        params.extend([(0, 0, 0, 0, 0)] * (capacity - len(prefixes)))
        # Fixed across prefix changes within a graph key.
        branch_capacity = pages_per_branch(page_size - 1, steps, page_size)
        return cls(
            prefixes,
            capacity,
            topk,
            steps,
            pages,
            page_size,
            branch_capacity,
            tuple(params),
            domain,
        )


class SRPagedMetadataWorkspace:
    def __init__(
        self, device, capacity, fields, outputs, *, domain="eager", metrics=None
    ):
        self.device = torch.device(device)
        self.capacity, self.fields, self.domain = capacity, fields, domain
        self.owner_stream = None
        self.outputs = tuple(outputs)
        # ModelRunner passes "npu", whereas allocated tensors carry "npu:0".
        # Resolve the actual device without reading a device tensor or scalar.
        if (
            self.outputs
            and self.device.index is None
            and self.device.type == self.outputs[0].device.type
        ):
            self.device = self.outputs[0].device
        self.identities = tuple(_identity(t) for t in self.outputs)
        self.host, pinned = alloc_host((capacity, fields), torch.int64, self.device)
        if self.device.type == "npu" and not pinned:
            raise RuntimeError("paged metadata requires pinned parameter staging")
        self.params = empty_metadata((capacity, fields), torch.int64, self.device)
        self.parameter_identity = _identity(self.params)
        self.metrics = metrics
        self.upload_event = self.ready_event = self.consumer_event = None
        if self.device.type != "cpu":
            Event = torch.get_device_module(self.device.type).Event
            self.upload_event, self.ready_event, self.consumer_event = (
                Event(),
                Event(),
                Event(),
            )
        self.upload_pending = False
        self.pending_consumer = False
        self.unresolved = False
        self.generation = 0
        self.holds = ()
        self.lease_hold = None
        self.retired_inputs = []
        self.growth_reported = False

    @property
    def bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.params, *self.outputs))

    def count(self, key, value=1):
        if self.metrics is not None:
            self.metrics.counts[
                f"paged_metadata_{'target' if self.fields == 3 else 'draft'}_{key}"
            ] += value

    def check(self):
        if self.unresolved:
            raise SRPagedMetadataSubmittedError(
                "paged metadata completion is unresolved"
            )
        current = stream_key(self.device)
        if self.owner_stream is None:
            self.owner_stream = current
        if self.owner_stream != current:
            raise RuntimeError("paged metadata workspace used on a different stream")
        if self.identities != tuple(_identity(t) for t in self.outputs):
            raise RuntimeError("paged metadata output storage was replaced")
        if self.parameter_identity != _identity(self.params):
            raise RuntimeError("paged metadata parameter storage was replaced")

    def poison(self):
        self.unresolved = True

    def upload(self, rows, holds):
        self.check()
        if not self.growth_reported and self.metrics is not None:
            self.count("workspace_grow")
            self.growth_reported = True
        if self.pending_consumer:
            raise RuntimeError(
                "previous paged metadata has no consumer completion record"
            )
        if len(rows) != self.capacity or any(len(r) != self.fields for r in rows):
            raise ValueError("paged metadata parameter shape mismatch")
        # Validate all values before touching staging or submitting a copy.
        if any(not -(1 << 63) <= int(v) < (1 << 63) for row in rows for v in row):
            raise ValueError("paged metadata parameter exceeds int64")
        try:
            if self.upload_pending and not self.upload_event.query():
                start = time.perf_counter()
                self.upload_event.synchronize()
                self.count("staging_waits")
                if self.metrics is not None:
                    self.metrics.add_host(
                        "paged_metadata_staging_wait", time.perf_counter() - start
                    )
            if self.holds:
                # Previous input reads finish at ready_event, independently of
                # the later attention consumer. Retain them until confirmed.
                if self.ready_event is not None and not self.ready_event.query():
                    self.retired_inputs.append((self.ready_event, self.holds))
                    self.ready_event = torch.get_device_module(self.device.type).Event()
            self.retired_inputs = [
                (e, h) for e, h in self.retired_inputs if not e.query()
            ]
        except BaseException as exc:
            self.poison()
            raise SRPagedMetadataSubmittedError(
                "paged metadata reuse event failed"
            ) from exc
        start = time.perf_counter()
        # Assign through the CPU NumPy view: no per-element torch operations.
        if self.capacity:
            self.host.numpy()[:] = rows
        if self.metrics is not None:
            self.metrics.add_host(
                "paged_metadata_params_cpu", time.perf_counter() - start
            )
        self.holds = tuple(holds)
        self.pending_consumer = True
        try:
            if self.capacity:
                self.params.copy_(self.host, non_blocking=self.device.type != "cpu")
                if self.upload_event is not None:
                    self.upload_event.record()
                    self.upload_pending = True
                self.count("h2d_count", int(self.device.type != "cpu"))
                self.count(
                    "h2d_bytes",
                    self.params.numel() * 8 if self.device.type != "cpu" else 0,
                )
        except BaseException as exc:
            self.poison()
            raise SRPagedMetadataSubmittedError("paged metadata upload failed") from exc
        self.generation += 1
        self.count(self.domain + "_prepares")

    def ready(self, calls):
        if self.capacity and self.ready_event is not None:
            self.ready_event.record()
        self.count("kernel_calls", calls)

    def consumed(self):
        if not self.pending_consumer:
            return
        self.check()
        try:
            if self.capacity and self.consumer_event is not None:
                self.consumer_event.record()
        except BaseException as exc:
            self.poison()
            raise SRPagedMetadataSubmittedError(
                "paged metadata consumer event failed"
            ) from exc
        self.pending_consumer = False

    def completed(self):
        if self.unresolved or self.pending_consumer:
            return False
        try:
            return self.consumer_event is None or self.consumer_event.query()
        except BaseException as exc:
            self.poison()
            raise SRPagedMetadataSubmittedError(
                "paged metadata retirement query failed"
            ) from exc


class SRPagedMetadataCache:
    """One eager shape per stream; graph entries remain owned by their graphs."""

    def __init__(self):
        self.entries = {}
        self.retired = []

    def get(self, key, shape, factory):
        self.retired = [
            entry for entry in self.retired if not entry.workspace.completed()
        ]
        old = self.entries.get(key)
        if old is not None and old[0] == shape:
            return old[1]
        new = factory()  # Allocation failure preserves the old entry.
        if old is not None:
            self.retired.append(old[1])
        self.entries[key] = (shape, new)
        return new


@dataclass
class SRDraftPagedView:
    block_tables: torch.Tensor
    branch_pages: torch.Tensor
    active_rows: torch.Tensor
    workspace: SRPagedMetadataWorkspace
    plan: SRDraftPagedPlan
    generation: int = 0

    def check(self):
        self.workspace.check()
        if self.generation != self.workspace.generation:
            raise RuntimeError("stale draft paged metadata generation")
        if not self.workspace.pending_consumer:
            raise RuntimeError("draft paged metadata has already been consumed")


def allocate_draft_view(plan, device, *, tables=None, active=None, metrics=None):
    rows = plan.capacity * plan.topk
    if tables is None:
        tables = empty_metadata((rows, plan.pages), torch.int32, device)
    if active is None:
        active = empty_metadata((rows,), torch.bool, device)
    branch = empty_metadata(
        (plan.capacity, plan.topk, plan.branch_capacity), torch.int64, device
    )
    ws = SRPagedMetadataWorkspace(
        device,
        plan.capacity,
        5,
        (tables, branch, active),
        domain=plan.domain,
        metrics=metrics,
    )
    return SRDraftPagedView(tables, branch, active, ws, plan)


def _outputs(ws, expected):
    ws.check()
    if len(ws.outputs) != len(expected):
        raise ValueError("paged metadata output set mismatch")
    for tensor, (shape, dtype) in zip(ws.outputs, expected):
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.device != ws.device
            or not tensor.is_contiguous()
        ):
            raise ValueError("paged metadata output layout mismatch")
        if tensor.device.type == "npu":
            import torch_npu

            if torch_npu.get_npu_format(tensor) != 2:
                raise ValueError("paged metadata output storage is no longer ND")


def fill_target_paged_metadata_(md, mapping, pool, mask, plan, dummy=0):
    ws = md.workspace
    if any(
        a is not b
        for a, b in zip(ws.outputs, (md.block_tables, md.blocked_mask, md.active_rows))
    ):
        raise RuntimeError("target paged metadata output storage binding was replaced")
    _outputs(
        ws,
        (
            ((plan.capacity, plan.pages), torch.int32),
            ((plan.capacity, 1, plan.queries, plan.pages * plan.page_size), torch.bool),
            ((plan.capacity,), torch.bool),
        ),
    )
    _integer(mapping, ws.device, 2, "request mapping")
    _integer(pool, ws.device, 1, "request indices")
    if pool.numel() < len(plan.prefixes) or any(
        p + plan.queries > mapping.shape[1] for p in plan.prefixes
    ):
        raise ValueError("target paged metadata input capacity mismatch")
    expected = sum(plan.queries * (p + plan.queries) for p in plan.prefixes)
    if mask.device != ws.device or mask.numel() != expected or mask.dtype != torch.bool:
        raise ValueError("target FULL_MASK device/dtype/length mismatch")
    if mask.ndim == 1:
        flat = mask
    elif mask.is_contiguous():
        flat = mask.view(-1)
    else:
        raise ValueError("target FULL_MASK must flatten without copying")
    if ws.device.type == "npu":
        import torch_npu

        if torch_npu.get_npu_format(flat) != 2:
            raise ValueError("target FULL_MASK must have ND storage")
    from sglang.srt.speculative.standalone_remote import (
        sr_paged_metadata_kernels_npu as kernels,
    )

    ws.upload(plan.params, (mapping, pool, mask))
    try:
        if plan.capacity:
            kernels.target(md, mapping, pool, flat, plan, dummy)
        ws.ready(2 if plan.capacity else 0)
    except BaseException as exc:
        ws.poison()
        raise SRPagedMetadataSubmittedError(
            "target paged metadata kernel failed"
        ) from exc
    md.kv_lens_cpu = list(plan.kv_lens)
    md.q_lens_cpu = [plan.queries] * plan.capacity


def fill_draft_paged_metadata_(view, mapping, pool, slots, plan, dummy=0):
    ws = view.workspace
    if any(
        a is not b
        for a, b in zip(
            ws.outputs, (view.block_tables, view.branch_pages, view.active_rows)
        )
    ):
        raise RuntimeError("draft paged metadata output storage binding was replaced")
    _outputs(
        ws,
        (
            ((plan.capacity * plan.topk, plan.pages), torch.int32),
            ((plan.capacity, plan.topk, plan.branch_capacity), torch.int64),
            ((plan.capacity * plan.topk,), torch.bool),
        ),
    )
    _integer(mapping, ws.device, 2, "request mapping")
    _integer(pool, ws.device, 1, "request indices")
    _integer(slots, ws.device, 3, "draft slots")
    bs = len(plan.prefixes)
    if pool.numel() < bs or tuple(slots.shape) != (bs, plan.topk, plan.steps):
        raise ValueError("draft paged metadata input capacity mismatch")
    if any(p > mapping.shape[1] for p in plan.prefixes):
        raise ValueError("draft prefix exceeds request mapping")
    from sglang.srt.speculative.standalone_remote import (
        sr_paged_metadata_kernels_npu as kernels,
    )

    ws.upload(plan.params, (mapping, pool, slots))
    try:
        if plan.capacity:
            kernels.draft(view, mapping, pool, slots, plan, dummy)
        ws.ready(int(plan.capacity > 0))
    except BaseException as exc:
        ws.poison()
        raise SRPagedMetadataSubmittedError(
            "draft paged metadata kernel failed"
        ) from exc
    view.plan, view.generation = plan, ws.generation


def finish_paged_metadata(backend):
    """Called on the consumer stream after forward/replay, never after H2D alone."""
    if backend is None:
        return
    ws = getattr(backend, "_sr_target_paged_workspace", None)
    view = getattr(backend, "_sr_draft_paged_view", None)
    if view is not None:
        ws = view.workspace
    if ws is not None:
        ws.consumed()


def poison_paged_metadata(backend):
    if backend is None:
        return False
    ws = getattr(backend, "_sr_target_paged_workspace", None)
    view = getattr(backend, "_sr_draft_paged_view", None)
    if view is not None:
        ws = view.workspace
    if ws is not None and ws.pending_consumer:
        ws.poison()
        return True
    return False


def report_paged_metadata(owner, role, metrics):
    if metrics is None:
        return
    if role == "target":
        graph = getattr(owner, "_target_fia_graph_metadata", {}).values()
        eager = getattr(owner, "_target_paged_eager_cache", None)
    else:
        graph = getattr(owner, "_sr_paged_graph_views", {}).values()
        eager = getattr(owner, "_sr_paged_eager_cache", None)
    current = {id(x.workspace): x.workspace for x in graph if x.workspace is not None}
    retired = {}
    if eager is not None:
        current.update(
            {id(v.workspace): v.workspace for _, v in eager.entries.values()}
        )
        retired.update({id(v.workspace): v.workspace for v in eager.retired})
    live_bytes = sum(w.bytes for w in current.values())
    retired_bytes = sum(w.bytes for w in retired.values())
    key = f"paged_metadata_{role}_"
    peak_attr = "_" + key + "peak_bytes"
    peak = max(getattr(owner, peak_attr, 0), live_bytes + retired_bytes)
    setattr(owner, peak_attr, peak)
    metrics.counts[key + "current_bytes"] = live_bytes
    metrics.counts[key + "retired_bytes"] = retired_bytes
    metrics.counts[key + "peak_bytes"] = peak


def _private_mapping(mapping, columns):
    # One row with production strides. Storage covers only that private row;
    # the row stride still participates in the production compile signature.
    span = max((columns - 1) * mapping.stride(1) + 1, 1)
    backing = empty_metadata((span,), mapping.dtype, mapping.device)
    result = backing.as_strided((1, columns), mapping.stride())
    result.zero_()
    return result


def warm_target_paged_metadata(md, mapping):
    """Compile on private inputs/outputs before capture; never touch live KV."""
    from sglang.srt.speculative.standalone_remote.verifier.sr_target_tree_fia import (
        SRTargetTreeFiaMetadata,
    )

    if md.workspace is None or getattr(md, "_paged_warm_complete", False):
        return
    b, _, q, _ = md.blocked_mask.shape
    private = SRTargetTreeFiaMetadata.allocate(
        b, q, md.block_tables.shape[1], md.page_size, mapping.device
    )
    md._paged_warm_holds = private
    req = _private_mapping(mapping, q)
    pool = empty_metadata((b,), torch.int64, mapping.device)
    pool.zero_()
    mask = empty_metadata((b * q * q,), torch.bool, mapping.device)
    mask.fill_(True)
    plan = SRTargetPagedPlan.build(
        [0] * b, b, q, md.block_tables.shape[1], md.page_size
    )
    try:
        fill_target_paged_metadata_(private, req, pool, mask, plan)
        private.workspace.consumed()
        private.workspace.consumer_event.synchronize()
    except BaseException as exc:
        private.workspace.poison()
        raise SRPagedMetadataSubmittedError(
            "target metadata warmup completion failed"
        ) from exc
    md._paged_warm_complete = True
    md._paged_warm_holds = None


def warm_draft_paged_metadata(view, mapping):
    if getattr(view, "_paged_warm_complete", False):
        return
    p = view.plan
    plan = SRDraftPagedPlan.build(
        [0] * p.capacity, p.capacity, p.topk, p.steps, p.pages, p.page_size, "warmup"
    )
    private = allocate_draft_view(plan, mapping.device)
    view._paged_warm_holds = private
    req = _private_mapping(mapping, max(p.steps, 1))
    pool = empty_metadata((p.capacity,), torch.int64, mapping.device)
    pool.zero_()
    slots = empty_metadata((p.capacity, p.topk, p.steps), torch.int64, mapping.device)
    slots.zero_()
    try:
        fill_draft_paged_metadata_(private, req, pool, slots, plan)
        private.workspace.consumed()
        private.workspace.consumer_event.synchronize()
    except BaseException as exc:
        private.workspace.poison()
        raise SRPagedMetadataSubmittedError(
            "draft metadata warmup completion failed"
        ) from exc
    view._paged_warm_complete = True
    view._paged_warm_holds = None
