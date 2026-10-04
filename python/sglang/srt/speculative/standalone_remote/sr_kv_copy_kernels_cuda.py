"""Grouped token-major CUDA KV copying, with a global gather/scatter boundary.

Pointer/stride tables belong to a workspace and are rebuilt only on reserve.
Tree parents, physical slots and active flags are read on every replay.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N", "Rows", "SlotBase"])
def _stage(
    Table,
    ScratchTable,
    Src,
    Dst,
    Slots,
    Parents,
    Active,
    N,
    Rows,
    SlotBase,
    SS: tl.constexpr,
    DS: tl.constexpr,
    TS0: tl.constexpr,
    TS1: tl.constexpr,
    PS: tl.constexpr,
    AS: tl.constexpr,
    TREE: tl.constexpr,
    HAS_ACTIVE: tl.constexpr,
    SCATTER: tl.constexpr,
    STORAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    group = tl.program_id(0)
    j = SlotBase + tl.program_id(1)
    c = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    meta = Table + group * 7
    ptr = tl.load(meta).to(tl.pointer_type(STORAGE))
    scratch = tl.load(ScratchTable + group).to(tl.pointer_type(STORAGE))
    slot_stride = tl.load(meta + 1)
    heads = tl.load(meta + 2)
    dim = tl.load(meta + 3)
    head_stride = tl.load(meta + 4)
    dim_stride = tl.load(meta + 5)
    pool_slots = tl.load(meta + 6)
    if TREE:
        row, step = j % Rows, j // Rows
        parent = tl.load(Parents + row * PS, j < N, other=0).to(tl.int64)
        active = (j < N) & (parent >= 0) & (parent < Rows)
        if HAS_ACTIVE:
            active = active & tl.load(Active + row * AS, j < N, other=0).to(tl.int1)
        src = tl.load(Slots + step * TS0 + parent * TS1, active, other=0).to(tl.int64)
        dst = tl.load(Slots + step * TS0 + row * TS1, active, other=0).to(tl.int64)
    else:
        src = tl.load(Src + j * SS, j < N, other=0).to(tl.int64)
        dst = tl.load(Dst + j * DS, j < N, other=0).to(tl.int64)
        active = j < N
    active = active & (src >= 0) & (src < pool_slots) & (dst >= 0) & (dst < pool_slots)
    mask = active & (src != dst) & (c < heads * dim)
    slot = dst if SCATTER else src
    # Inactive lanes must not form an out-of-range address. Slot 0 is in range
    # for every admitted pool, and masked loads still substitute zero.
    slot = tl.where(mask, slot, 0)
    offset = slot * slot_stride + (c // dim) * head_stride + (c % dim) * dim_stride
    scratch_offset = j * heads * dim + c
    if SCATTER:
        value = tl.load(scratch + scratch_offset, mask=mask, other=0)
        tl.store(ptr + offset, value, mask=mask)
    else:
        value = tl.load(ptr + offset, mask=mask, other=0)
        tl.store(scratch + scratch_offset, value, mask=mask)


class CudaLayeredCopy:
    def __init__(self, layout):
        groups = {}
        for i, b in enumerate(layout.buffers):
            # Copy raw bits, including NaN payloads and signed zero.
            groups.setdefault(b.tensor.element_size(), []).append(i)
        self.groups = tuple(tuple(indices) for indices in groups.values())

    def prepare(self, workspace, scratch):
        tables = []
        workspace._cuda_preparing = tables
        for indices in self.groups:
            rows = []
            for i in indices:
                t = workspace.layout.buffers[i].tensor
                rows.append(
                    [
                        t.data_ptr(),
                        t.stride(0),
                        t.shape[1],
                        t.shape[2],
                        t.stride(1),
                        t.stride(2),
                        t.shape[0],
                    ]
                )
            table = torch.tensor(
                rows, dtype=torch.int64, device=workspace.layout.device
            )
            # Retain a successful first upload if the following upload fails.
            workspace._cuda_preparing.append((table,))
            scratch_ptr = torch.tensor(
                [scratch[i].data_ptr() for i in indices],
                dtype=torch.int64,
                device=workspace.layout.device,
            )
            width = max(rows[j][2] * rows[j][3] for j in range(len(rows)))
            tables[-1] = (
                table,
                scratch_ptr,
                width,
                workspace.layout.buffers[indices[0]].tensor.element_size(),
            )
        return tables

    def launch(
        self,
        workspace,
        src=None,
        dst=None,
        *,
        slots=None,
        parents=None,
        depth=0,
        active=None,
    ):
        tree = slots is not None
        n = depth * parents.numel() if tree else src.numel()
        if not n:
            return
        # Valid dummy pointers are used only in compile-time-eliminated branches.
        dummy = slots if tree else src
        ss, ds = (0, 0) if tree else (src.stride(0), dst.stride(0))
        ts0, ts1 = slots.stride() if tree else (0, 0)
        ps = parents.stride(0) if tree else 0
        rows = parents.numel() if tree else 1
        args = (
            dummy if tree else src,
            dummy if tree else dst,
            dummy if not tree else slots,
            dummy if not tree else parents,
            active if active is not None else dummy,
        )
        types = {1: tl.uint8, 2: tl.uint16, 4: tl.uint32, 8: tl.uint64}
        # Slot grid is chunked for CUDA's y-axis limit. All gather chunks and
        # all dtype groups complete before any scatter chunk is submitted.
        for scatter in (False, True):
            for table, scratch_table, width, size in workspace.cuda_tables:
                for base in range(0, n, 65535):
                    _stage[
                        (table.shape[0], min(n - base, 65535), triton.cdiv(width, 256))
                    ](
                        table,
                        scratch_table,
                        *args,
                        n,
                        rows,
                        base,
                        ss,
                        ds,
                        ts0,
                        ts1,
                        ps,
                        active.stride(0) if active is not None else 0,
                        tree,
                        active is not None,
                        scatter,
                        types[size],
                        256,
                    )
