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
    if out.dtype != torch.int64 or any(
        t.device != out.device or t.dtype not in (torch.int32, torch.int64)
        for t in (accept_index, predict, accept_length)
    ):
        raise RuntimeError("fixed accept pack requires integer tensors on one device")
    if accept_length.numel() != bs:
        raise RuntimeError("fixed accept length batch mismatch")
    if out.device.type == "cuda":
        from sglang.srt.speculative.standalone_remote import sr_small_kernels_cuda

        if not predict.is_contiguous() or accept_length.ndim != 1:
            raise RuntimeError(
                "CUDA accept pack requires contiguous predict and flat lengths"
            )
        sr_small_kernels_cuda.pack_accept(accept_index, predict, accept_length, out)
        return
    if out.device.type == "npu":
        from sglang.srt.speculative.standalone_remote import sr_small_kernels_npu

        if not predict.is_contiguous() or accept_length.ndim != 1:
            raise RuntimeError(
                "NPU accept pack requires flat lengths and contiguous predict"
            )
        sr_small_kernels_npu.pack_accept(accept_index, predict, accept_length, out)
        return
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
):
    """Gather source slots, destination slots, and page ids of known lengths.

    Indexes are already on ``out_cache_loc``'s device. An empty page list does
    not read cache slots. Each output owns independent, zero-offset storage.
    Besides preserving results across rounds, this avoids passing a small tail
    view of a larger allocation to NPU sort during page release.
    """
    cache = out_cache_loc.reshape(-1)
    n_out = int(src_index.numel())
    n_tgt = int(tgt_index.numel())
    n_free = int(page_index.numel())
    if n_out != n_tgt:
        raise RuntimeError("fixed accept source and destination lengths differ")
    if page_size <= 0 or cache.dtype not in (torch.int32, torch.int64):
        raise RuntimeError("fixed accept requires positive page size and integer cache")
    for name, index in (
        ("src_index", src_index),
        ("tgt_index", tgt_index),
        ("page_index", page_index),
    ):
        if (
            index.device != cache.device
            or index.dtype != torch.int64
            or index.ndim != 1
        ):
            raise RuntimeError(f"fixed accept {name} must be int64 on the cache device")
    src = torch.empty(n_out, dtype=torch.int64, device=cache.device)
    tgt = torch.empty(n_tgt, dtype=torch.int64, device=cache.device)
    pages = torch.empty(n_free, dtype=torch.int64, device=cache.device)
    if cache.device.type == "npu":
        from sglang.srt.speculative.standalone_remote import sr_small_kernels_npu

        # Logical indexes are range-checked on CPU while constructing the
        # commit packet; the device kernel never discovers output lengths.
        sr_small_kernels_npu.gather_commit_slots(
            cache, src_index, tgt_index, page_index, int(page_size), src, tgt, pages
        )
    else:
        if n_out:
            src.copy_(cache.index_select(0, src_index))
            tgt.copy_(cache.index_select(0, tgt_index))
        if n_free:
            pages.copy_(
                cache.index_select(0, page_index).to(torch.int64) // int(page_size)
            )
    return src, tgt, pages


def gather_token_slots(out_cache_loc, kept_index, free_index):
    """Known CPU-planned lengths; independent outputs survive later rounds."""
    cache = out_cache_loc.reshape(-1)
    if cache.dtype not in (torch.int32, torch.int64):
        raise RuntimeError("token-slot cache must be integer")
    for index in (kept_index, free_index):
        if (
            index.ndim != 1
            or index.device != cache.device
            or index.dtype != torch.int64
        ):
            raise RuntimeError("token-slot indices must be local int64 vectors")
    kept = torch.empty(kept_index.numel(), dtype=torch.int64, device=cache.device)
    released = torch.empty(free_index.numel(), dtype=torch.int64, device=cache.device)
    if cache.device.type == "cuda":
        from sglang.srt.speculative.standalone_remote import sr_small_kernels_cuda

        sr_small_kernels_cuda.gather_token_slots(
            cache, kept_index, free_index, kept, released
        )
    else:
        torch.index_select(cache, 0, kept_index, out=kept)
        torch.index_select(cache, 0, free_index, out=released)
    return kept, released
