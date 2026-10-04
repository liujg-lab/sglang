"""CUDA gather of original-dtype logits; all gap arithmetic stays on CPU."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["E", "RowStride"])
def _edge_gather(
    Values, Logits, Parent, Token, OutMax, OutCand, E, RowStride, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < E
    parent = tl.load(Parent + i, mask, other=0).to(tl.int64)
    token = tl.load(Token + i, mask, other=0).to(tl.int64)
    z_star = tl.load(Values + parent, mask)
    z_cand = tl.load(Logits + parent * tl.cast(RowStride, tl.int64) + token, mask)
    tl.store(OutMax + i, z_star, mask)
    tl.store(OutCand + i, z_cand, mask)


def gather_edge_stats(values, logits, edge_index, stats):
    if (
        logits.ndim != 2
        or values.ndim != 1
        or edge_index.ndim != 2
        or stats.ndim != 2
        or edge_index.shape[0] != 2
        or tuple(stats.shape) != tuple(edge_index.shape)
        or values.numel() != logits.shape[0]
        or edge_index.dtype != torch.int64
        or values.dtype != logits.dtype
        or stats.dtype != logits.dtype
        or any(
            t.device != logits.device or not t.is_contiguous()
            for t in (values, logits, edge_index, stats)
        )
    ):
        raise RuntimeError("CUDA RPD edge gather layout mismatch")
    edges = stats.shape[1]
    if not edges:
        return False
    _edge_gather[(triton.cdiv(edges, 128),)](
        values,
        logits,
        edge_index[0],
        edge_index[1],
        stats[0],
        stats[1],
        edges,
        logits.stride(0),
        128,
    )
    return True
