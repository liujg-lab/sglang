"""One-launch RPD edge gather. Imported only on the real NPU path.

Reads the already uploaded ``int64[2, E]`` edge index and writes the parent
max logit and the candidate logit into the reused stats buffer. Values stay
in the logits dtype. This kernel does not subtract, cast, or test ``tau``.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 128


@triton.jit(do_not_specialize=["E", "RowStride"])
def _edge_gather(
    Values,
    Logits,
    Parent,
    Token,
    OutMax,
    OutCand,
    E,
    RowStride,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < E
    parent = tl.load(Parent + i, mask, other=0).to(tl.int64)
    token = tl.load(Token + i, mask, other=0).to(tl.int64)
    z_star = tl.load(Values + parent, mask)
    offset = parent * tl.cast(RowStride, tl.int64) + token
    z_cand = tl.load(Logits + offset, mask)
    tl.store(OutMax + i, z_star, mask)
    tl.store(OutCand + i, z_cand, mask)


def gather_edge_stats(values, logits, edge_index, stats) -> bool:
    """Write ``stats[0]`` and ``stats[1]`` for a contiguous live ``[2, E]`` view.

    Returns whether a kernel was launched. ``E == 0`` launches nothing.
    Callers pass the flattened live region, not the capacity-strided rows of
    the workspace allocation.
    """
    if (
        values.ndim != 1
        or logits.ndim != 2
        or edge_index.ndim != 2
        or stats.ndim != 2
        or int(edge_index.shape[0]) != 2
        or int(stats.shape[0]) != 2
    ):
        raise RuntimeError(
            "RPD edge gather shapes must be values[R], logits[R, V], "
            "index[2, E], stats[2, E]"
        )
    edges = int(stats.shape[1])
    if edges == 0:
        return False
    if int(edge_index.shape[1]) != edges:
        raise RuntimeError("RPD edge index width does not match stats")
    if (
        values.dtype != logits.dtype
        or stats.dtype != logits.dtype
        or edge_index.dtype != torch.int64
    ):
        raise RuntimeError(
            "RPD edge gather requires int64 indices and the logits dtype"
        )
    rows = (
        values,
        logits,
        edge_index,
        edge_index[0],
        edge_index[1],
        stats,
        stats[0],
        stats[1],
    )
    if any(not tensor.is_contiguous() for tensor in rows):
        raise RuntimeError(
            "RPD edge gather requires contiguous values, logits, indices, and stats"
        )
    _edge_gather[(triton.cdiv(edges, _BLOCK),)](
        values,
        logits,
        edge_index[0],
        edge_index[1],
        stats[0],
        stats[1],
        edges,
        int(logits.stride(0)),
        BLOCK=_BLOCK,
    )
    return True
