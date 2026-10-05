"""Private CUDA experiment: static contiguous rows, bitwise two-stage copy."""

import triton
import triton.language as tl


@triton.jit
def _stage(
    Table,
    Src,
    Dst,
    Slots,
    Parents,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    BASE: tl.constexpr,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    POOL: tl.constexpr,
    SS: tl.constexpr,
    DS: tl.constexpr,
    TS0: tl.constexpr,
    TS1: tl.constexpr,
    PS: tl.constexpr,
    TREE: tl.constexpr,
    SCATTER: tl.constexpr,
    BITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    group = tl.program_id(0)
    j = BASE + tl.program_id(1)
    c = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    ptr = tl.load(Table + group * 2).to(tl.pointer_type(BITS))
    scratch = tl.load(Table + group * 2 + 1).to(tl.pointer_type(BITS))
    valid = j < N
    if TREE:
        row, step = j % ROWS, j // ROWS
        parent = tl.load(Parents + row * PS, valid, other=0).to(tl.int64)
        valid = valid & (parent >= 0) & (parent < ROWS)
        src = tl.load(Slots + step * TS0 + parent * TS1, valid, other=0).to(tl.int64)
        dst = tl.load(Slots + step * TS0 + row * TS1, valid, other=0).to(tl.int64)
    else:
        src = tl.load(Src + j * SS, valid, other=0).to(tl.int64)
        dst = tl.load(Dst + j * DS, valid, other=0).to(tl.int64)
    valid = valid & (src >= 0) & (src < POOL) & (dst >= 0) & (dst < POOL)
    mask = valid & (src != dst) & (c < WIDTH)
    slot = tl.where(valid, dst if SCATTER else src, 0)
    offset = slot * STRIDE + c
    tmp = j.to(tl.int64) * WIDTH + c
    if SCATTER:
        value = tl.load(scratch + tmp, mask, other=0)
        tl.store(ptr + offset, value, mask)
    else:
        value = tl.load(ptr + offset, mask, other=0)
        tl.store(scratch + tmp, value, mask)


def launch(workspace, src=None, dst=None, *, slots=None, parents=None, depth=0):
    tree = slots is not None
    rows = parents.numel() if tree else 1
    n = depth * rows if tree else src.numel()
    dummy = slots if tree else src
    ts0, ts1 = slots.stride() if tree else (0, 0)
    bits = {1: tl.uint8, 2: tl.uint16, 4: tl.uint32, 8: tl.uint64}
    # Every group/chunk is gathered before the first target write.
    for scatter in (False, True):
        for table, width, stride, size in workspace.tables:
            block = min(1024, triton.next_power_of_2(width))
            for base in range(0, n, 65535):
                _stage[
                    (table.shape[0], min(n - base, 65535), triton.cdiv(width, block))
                ](
                    table,
                    dummy if tree else src,
                    dummy if tree else dst,
                    dummy if not tree else slots,
                    dummy if not tree else parents,
                    n,
                    rows,
                    base,
                    width,
                    stride,
                    workspace.layout.buffers[0].tensor.shape[0],
                    0 if tree else src.stride(0),
                    0 if tree else dst.stride(0),
                    ts0,
                    ts1,
                    parents.stride(0) if tree else 0,
                    tree,
                    scatter,
                    bits[size],
                    block,
                    num_warps=4,
                )
