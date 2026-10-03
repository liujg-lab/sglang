"""Three integer-only SR metadata kernels. No attention or KV writes."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["MapRows", "Dummy"])
def _target_pages(
    Params,
    Mapping,
    Pool,
    Tables,
    Active,
    B: tl.constexpr,
    Pages: tl.constexpr,
    Q: tl.constexpr,
    Page: tl.constexpr,
    MapRows,
    M0: tl.constexpr,
    M1: tl.constexpr,
    P0: tl.constexpr,
    Dummy,
    Block: tl.constexpr,
):
    tiles: tl.constexpr = triton.cdiv(Pages, Block)
    for task in range(tl.program_id(0), B * tiles, tl.num_programs(0)):
        b, tile = task // tiles, task % tiles
        j = tile * Block + tl.arange(0, Block)
        active = tl.load(Params + b * 3 + 2) != 0
        prefix = tl.load(Params + b * 3)
        req = tl.load(Pool + b * P0, active, other=0).to(tl.int64)
        tl.device_assert(
            (~active) | ((req >= 0) & (req < MapRows)), "request row outside mapping"
        )
        valid = active & (j < Pages) & (j * Page < prefix + Q)
        slot = tl.load(Mapping + req * M0 + j.to(tl.int64) * Page * M1, valid, other=0)
        tl.store(
            Tables + b * Pages + j, tl.where(valid, slot // Page, Dummy), j < Pages
        )
        if tile == 0:
            tl.store(Active + b, active)


@triton.jit
def _target_mask(
    Params,
    Mask,
    Out,
    B: tl.constexpr,
    Q: tl.constexpr,
    Width: tl.constexpr,
    MaskStride: tl.constexpr,
    Block: tl.constexpr,
):
    tiles: tl.constexpr = triton.cdiv(Width, Block)
    for task in range(tl.program_id(0), B * Q * tiles, tl.num_programs(0)):
        row, tile = task // tiles, task % tiles
        b, q = row // Q, row % Q
        s = tile * Block + tl.arange(0, Block)
        prefix = tl.load(Params + b * 3)
        start = tl.load(Params + b * 3 + 1)
        active = tl.load(Params + b * 3 + 2) != 0
        valid = active & (s < Width) & (s < prefix + Q)
        index = start + q.to(tl.int64) * (prefix + Q) + s
        attend = tl.load(Mask + index * MaskStride, valid, other=0).to(tl.int1)
        blocked = tl.where(active, ~attend, s != 0)
        tl.store(Out + row.to(tl.int64) * Width + s, blocked, s < Width)


@triton.jit(do_not_specialize=["MapRows", "Dummy"])
def _draft(
    Params,
    Mapping,
    Pool,
    Slots,
    Tables,
    Branch,
    Active,
    B: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    Pages: tl.constexpr,
    BranchCap: tl.constexpr,
    Page: tl.constexpr,
    MapRows,
    M0: tl.constexpr,
    M1: tl.constexpr,
    P0: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    Dummy,
    Block: tl.constexpr,
    Tiles: tl.constexpr,
):
    for task in range(tl.program_id(0), B * K * Tiles, tl.num_programs(0)):
        row, tile = task // Tiles, task % Tiles
        b, k = row // K, row % K
        j = tile * Block + tl.arange(0, Block)
        rem = tl.load(Params + b * 5)
        shared = tl.load(Params + b * 5 + 1)
        query = tl.load(Params + b * 5 + 2)
        reserved = tl.load(Params + b * 5 + 3)
        active = tl.load(Params + b * 5 + 4) != 0
        req = tl.load(Pool + b * P0, active, other=0).to(tl.int64)
        tl.device_assert(
            (~active) | ((req >= 0) & (req < MapRows)), "request row outside mapping"
        )
        shared_valid = active & (j < Pages) & (j < shared)
        shared_slot = tl.load(
            Mapping + req * M0 + j.to(tl.int64) * Page * M1, shared_valid, other=0
        )
        branch_j = j.to(tl.int64) - shared
        step = tl.minimum(tl.maximum(branch_j * Page - rem, 0), S - 1)
        query_valid = active & (j < Pages) & (j >= shared) & (branch_j < query)
        query_slot = tl.load(
            Slots + b.to(tl.int64) * S0 + k * S1 + step * S2, query_valid, other=0
        )
        page = tl.where(
            shared_valid,
            shared_slot // Page,
            tl.where(query_valid, query_slot // Page, Dummy),
        )
        tl.store(Tables + row.to(tl.int64) * Pages + j, page.to(tl.int32), j < Pages)
        step = tl.minimum(tl.maximum(j.to(tl.int64) * Page - rem, 0), S - 1)
        reserved_valid = active & (j < BranchCap) & (j < reserved)
        slot = tl.load(
            Slots + b.to(tl.int64) * S0 + k * S1 + step * S2, reserved_valid, other=0
        )
        tl.store(
            Branch + row.to(tl.int64) * BranchCap + j,
            tl.where(reserved_valid, slot // Page, Dummy),
            j < BranchCap,
        )
        if tile == 0:
            tl.store(Active + row, active)


def target(md, mapping, pool, mask, plan, dummy):
    b, q, pages, page = plan.capacity, plan.queries, plan.pages, plan.page_size
    _target_pages[(min(b * triton.cdiv(pages, 256), 65535),)](
        md.workspace.params,
        mapping,
        pool,
        md.block_tables,
        md.active_rows,
        b,
        pages,
        q,
        page,
        mapping.shape[0],
        *mapping.stride(),
        pool.stride(0),
        dummy,
        256,
    )
    width = pages * page
    _target_mask[(min(b * q * triton.cdiv(width, 256), 65535),)](
        md.workspace.params,
        mask,
        md.blocked_mask,
        b,
        q,
        width,
        mask.stride(0),
        256,
    )


def draft(view, mapping, pool, slots, plan, dummy):
    tiles = triton.cdiv(max(plan.pages, plan.branch_capacity), 256)
    tasks = plan.capacity * plan.topk * tiles
    _draft[(min(tasks, 65535),)](
        view.workspace.params,
        mapping,
        pool,
        slots,
        view.block_tables,
        view.branch_pages,
        view.active_rows,
        plan.capacity,
        plan.topk,
        plan.steps,
        plan.pages,
        plan.branch_capacity,
        plan.page_size,
        mapping.shape[0],
        *mapping.stride(),
        pool.stride(0),
        *slots.stride(),
        dummy,
        256,
        tiles,
    )
