"""Chunked NPU paged KV snapshot/scatter. Imported after NPU admission."""

import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=["SlotBase", "TileBase", "ValidSlots", "PoolSlots", "Capacity"]
)
def _paged_move(
    KV,
    Scratch,
    Src,
    Dst,
    Slots,
    Parents,
    Active,
    SlotBase,
    TileBase,
    ValidSlots,
    Groups: tl.constexpr,
    PoolSlots,
    Capacity,
    Width: tl.constexpr,
    SrcStride: tl.constexpr,
    DstStride: tl.constexpr,
    Rows: tl.constexpr,
    StepStride: tl.constexpr,
    RowStride: tl.constexpr,
    ParentStride: tl.constexpr,
    ActiveStride: tl.constexpr,
    Tree: tl.constexpr,
    HasActive: tl.constexpr,
    Scatter: tl.constexpr,
    Block: tl.constexpr,
):
    task = tl.program_id(0)
    local = task // Groups
    group = task % Groups
    j = local + SlotBase
    col = (tl.program_id(1) + TileBase) * Block + tl.arange(0, Block)
    valid = local < ValidSlots
    if Tree:
        row = j % Rows
        step = j // Rows
        if HasActive:
            valid = valid & tl.load(Active + row * ActiveStride).to(tl.int1)
        parent = tl.load(Parents + row * ParentStride, valid, other=0)
        tl.device_assert(
            (~valid) | ((parent >= 0) & (parent < Rows)),
            "tree KV parent outside rows",
        )
        src_slot = tl.load(
            Slots + step * StepStride + parent * RowStride, valid, other=0
        )
        dst_slot = tl.load(Slots + step * StepStride + row * RowStride, valid, other=0)
    else:
        src_slot = tl.load(Src + j * SrcStride, valid, other=0)
        dst_slot = tl.load(Dst + j * DstStride, valid, other=0)
    tl.device_assert(
        (~valid)
        | (
            (src_slot >= 0)
            & (src_slot < PoolSlots)
            & (dst_slot >= 0)
            & (dst_slot < PoolSlots)
        ),
        "KV slot outside pool",
    )
    # Identity destinations remain unchanged; other gathers may still read
    # them. Both stages must use the same mask, with unique active targets.
    mask = valid & (src_slot != dst_slot) & (col < Width)
    if Scatter:
        slot = dst_slot
    else:
        slot = src_slot
    pool_offset = (group.to(tl.int64) * PoolSlots + slot.to(tl.int64)) * Width + col
    scratch_offset = (group.to(tl.int64) * Capacity + j.to(tl.int64)) * Width + col
    if Scatter:
        value = tl.load(Scratch + scratch_offset, mask, other=0)
        tl.store(KV + pool_offset, value, mask)
    else:
        value = tl.load(KV + pool_offset, mask, other=0)
        tl.store(Scratch + scratch_offset, value, mask)


def _launch(kv, scratch, src, dst, slots, parents, active, n, rows):
    from sglang.srt.speculative.standalone_remote.sr_kv_copy import (
        kernel_pointer_launch,
        paged_launch_chunks,
    )

    tree, has_active, strides, pointers = kernel_pointer_launch(
        src, dst, slots, parents, active
    )
    src_stride, dst_stride, step_stride, row_stride, parent_stride, active_stride = (
        strides
    )
    src, dst, slots, parents, active = pointers
    groups = kv.shape[0] * kv.shape[1]
    width = kv.shape[3]
    args = dict(
        Groups=groups,
        PoolSlots=kv.shape[2],
        Capacity=scratch.shape[2],
        Width=width,
        SrcStride=src_stride,
        DstStride=dst_stride,
        Rows=max(rows, 1),
        StepStride=step_stride,
        RowStride=row_stride,
        ParentStride=parent_stride,
        ActiveStride=active_stride,
        Tree=tree,
        HasActive=has_active,
        Block=256,
    )
    chunks = paged_launch_chunks(n, groups, width, block=256)
    # Every gather chunk finishes before the first scatter chunk. A later
    # chunk must not overwrite a source that another chunk has not read.
    for scatter in (False, True):
        for slot_base, slot_count, tile_base, tile_count in chunks:
            _paged_move[(slot_count * groups, tile_count)](
                kv,
                scratch,
                src,
                dst,
                slots,
                parents,
                active,
                slot_base,
                tile_base,
                slot_count,
                Scatter=scatter,
                **args,
            )


def move(kv, scratch, src, dst):
    if src.numel():
        _launch(kv, scratch, src, dst, None, None, None, src.numel(), 0)


def remap(kv, scratch, slots, parents, depth, active):
    if parents.numel():
        _launch(
            kv,
            scratch,
            None,
            None,
            slots,
            parents,
            active,
            depth * parents.numel(),
            parents.numel(),
        )
