"""Two-launch integer identity remap. No KV writes and no active-row mask.

Imported only on the CUDA/NPU path. Every gather chunk finishes before the
first scatter chunk, so swaps, cycles, and repeated parents read the old row.
"""

import triton
import triton.language as tl

# Triton-Ascend rejects a grid whose program count is 65536 or larger.
_GRID_LIMIT = 65535


@triton.jit(
    do_not_specialize=[
        "SlotBase",
        "Valid",
        "Depth",
        "Rows",
        "IdStep",
        "IdRow",
        "ScratchStep",
        "ScratchRow",
        "ParentStride",
    ]
)
def _remap_identity(
    Ids,
    Scratch,
    Parents,
    SlotBase,
    Valid,
    Depth,
    Rows,
    IdStep,
    IdRow,
    ScratchStep,
    ScratchRow,
    ParentStride,
    Scatter: tl.constexpr,
):
    local = tl.program_id(0)
    token = (local + SlotBase).to(tl.int64)
    rows = Rows.to(tl.int64)
    row = token % rows
    step = token // rows
    valid = (local < Valid) & (step < Depth.to(tl.int64))
    id_step = IdStep.to(tl.int64)
    id_row = IdRow.to(tl.int64)
    scratch_step = ScratchStep.to(tl.int64)
    scratch_row = ScratchRow.to(tl.int64)
    parent_stride = ParentStride.to(tl.int64)
    parent = tl.load(
        Parents + row * parent_stride, mask=valid, other=0
    ).to(tl.int64)
    in_range = (parent >= 0) & (parent < rows)
    # device_assert is diagnostic only. Mask the copy so a release build
    # cannot publish column 0 in place of an out-of-range parent.
    tl.device_assert((~valid) | in_range, "identity parent outside rows")
    active = valid & in_range
    if Scatter:
        value = tl.load(
            Scratch + step * scratch_step + row * scratch_row, mask=active, other=0
        )
        tl.store(Ids + step * id_step + row * id_row, value, mask=active)
    else:
        value = tl.load(
            Ids + step * id_step + parent * id_row, mask=active, other=0
        )
        tl.store(
            Scratch + step * scratch_step + row * scratch_row, value, mask=active
        )


def remap_identity(ids, scratch, parents, depth, rows):
    """Gather ``ids[step, parent[row]]`` for every historical step, then scatter."""
    n = int(depth) * int(rows)
    if n <= 0:
        return
    id_step, id_row = int(ids.stride(0)), int(ids.stride(1))
    scratch_step, scratch_row = int(scratch.stride(0)), int(scratch.stride(1))
    parent_stride = int(parents.stride(0))
    table_depth = int(depth)
    width = int(rows)
    for scatter in (False, True):
        for base in range(0, n, _GRID_LIMIT):
            count = min(_GRID_LIMIT, n - base)
            _remap_identity[(count,)](
                ids,
                scratch,
                parents,
                base,
                count,
                table_depth,
                width,
                id_step,
                id_row,
                scratch_step,
                scratch_row,
                parent_stride,
                Scatter=scatter,
            )
