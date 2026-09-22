"""Fixed-shape accept pack and KV-slot gather.

Lengths come from the caller. These ops do not use boolean compression,
``nonzero``, or ``.item`` to discover an output length.
"""

from __future__ import annotations

import torch


def pack_accept(
    accept_index: torch.Tensor,
    predict: torch.Tensor,
    accept_length: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Write indices, masked tokens, pre-truncation length, and an error flag.

    ``out`` is ``[bs, 2L+2]`` int64: indices, tokens, length, error.
    An index of ``-1`` or an out-of-range index does not contribute a token.
    Out-of-range indexes set the error flag and are clamped so the gather
    itself stays in range.
    """
    bs, path_cap = accept_index.shape
    if out.shape[0] < bs or out.shape[1] < path_cap * 2 + 2:
        raise RuntimeError("fixed accept pack buffer is smaller than the batch")
    idx = accept_index.to(dtype=torch.int64)
    pred = predict.reshape(-1)
    npred = int(pred.shape[0])
    if npred == 0:
        tokens = torch.zeros((bs, path_cap), dtype=torch.int64, device=idx.device)
        err = idx >= 0
    else:
        in_range = (idx >= 0) & (idx < npred)
        safe = idx.clamp(0, npred - 1)
        gathered = pred.index_select(0, safe.reshape(-1)).reshape(bs, path_cap)
        tokens = torch.where(in_range, gathered.to(torch.int64), torch.zeros_like(idx))
        err = (idx < -1) | ((idx >= 0) & ~in_range)
    out[:bs, :path_cap] = idx
    out[:bs, path_cap : path_cap * 2] = tokens
    out[:bs, path_cap * 2] = accept_length.to(dtype=torch.int64).reshape(bs)
    out[:bs, path_cap * 2 + 1] = err.any(dim=1).to(dtype=torch.int64)


def gather_commit_slots(
    out_cache_loc: torch.Tensor,
    src_index: torch.Tensor,
    tgt_index: torch.Tensor,
    page_index: torch.Tensor,
    page_size: int,
    src_buf: torch.Tensor,
    tgt_buf: torch.Tensor,
    page_buf: torch.Tensor,
):
    """Gather source slots, destination slots, and page ids of known lengths.

    Indexes are already on ``out_cache_loc``'s device. An empty page list does
    not read cache slots. Returned tensors do not alias the reusable buffers.
    """
    cache = out_cache_loc.reshape(-1)
    n_out = int(src_index.numel())
    n_tgt = int(tgt_index.numel())
    n_free = int(page_index.numel())
    if n_out != n_tgt:
        raise RuntimeError("fixed accept source and destination lengths differ")
    if n_out > src_buf.numel() or n_free > page_buf.numel():
        raise RuntimeError("fixed accept commit buffer is smaller than the batch")
    for name, index in (
        ("src_index", src_index),
        ("tgt_index", tgt_index),
        ("page_index", page_index),
    ):
        if index.device != cache.device or index.dtype != torch.int64:
            raise RuntimeError(
                f"fixed accept {name} must be int64 on the cache device"
            )
    if n_out:
        src_buf[:n_out].copy_(cache.index_select(0, src_index).to(torch.int64))
        tgt_buf[:n_tgt].copy_(cache.index_select(0, tgt_index).to(torch.int64))
        src = src_buf[:n_out]
        tgt = tgt_buf[:n_tgt]
    else:
        src = src_buf[:0]
        tgt = tgt_buf[:0]
    if n_free:
        slots = cache.index_select(0, page_index).to(torch.int64)
        page_buf[:n_free].copy_(slots // int(page_size))
        pages = page_buf[:n_free]
    else:
        pages = page_buf[:0]
    return src.clone(), tgt.clone(), pages.clone()
