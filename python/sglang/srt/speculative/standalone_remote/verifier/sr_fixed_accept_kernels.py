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


def owned_span(tensor: torch.Tensor) -> torch.Tensor:
    """Return a tensor whose storage byte size matches its logical size.

    A 1-D slice is contiguous, so ``contiguous()`` keeps the parent allocation.
    NPU ``aclnnSort`` uses that parent element count as the source and the
    logical length as the destination.
    """
    needed = tensor.numel() * tensor.element_size()
    storage = tensor.untyped_storage()
    if (
        tensor.storage_offset() == 0
        and tensor.is_contiguous()
        and storage.nbytes() == needed
    ):
        return tensor
    return tensor.clone()


def gather_commit_slots(
    out_cache_loc: torch.Tensor,
    src_index: torch.Tensor,
    tgt_index: torch.Tensor,
    page_index: torch.Tensor,
    page_size: int,
    src: torch.Tensor,
    tgt: torch.Tensor,
    pages: torch.Tensor,
):
    """Gather source slots, destination slots, and page ids of known lengths.

    Indexes are already on ``out_cache_loc``'s device. An empty page list does
    not read cache slots. ``src``, ``tgt``, and ``pages`` are caller-owned
    int64 buffers; this function does not allocate them. Returned prefixes are
    the live lengths. ``pages`` must already have storage whose byte size
    equals ``page_index.numel() * 8`` so NPU sort sees an exact descriptor.
    ``src`` and ``tgt`` may be prefixes of a larger scratch and are invalid
    after the next gather into that scratch.
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
    for name, buf, need in (
        ("src", src, n_out),
        ("tgt", tgt, n_tgt),
        ("pages", pages, n_free),
    ):
        if (
            buf.device != cache.device
            or buf.dtype != torch.int64
            or buf.ndim != 1
            or not buf.is_contiguous()
            or int(buf.numel()) < need
        ):
            raise RuntimeError(
                f"fixed accept {name} output must be a contiguous int64 buffer"
            )
    src_out = src[:n_out]
    tgt_out = tgt[:n_tgt]
    page_out = pages[:n_free]
    if cache.device.type == "npu":
        from sglang.srt.speculative.standalone_remote import sr_small_kernels_npu

        # Logical indexes are range-checked on CPU while constructing the
        # commit packet; the device kernel never discovers output lengths.
        sr_small_kernels_npu.gather_commit_slots(
            cache,
            src_index,
            tgt_index,
            page_index,
            int(page_size),
            src_out,
            tgt_out,
            page_out,
        )
    else:
        if n_out:
            src_out.copy_(cache.index_select(0, src_index))
            tgt_out.copy_(cache.index_select(0, tgt_index))
        if n_free:
            page_out.copy_(
                cache.index_select(0, page_index).to(torch.int64) // int(page_size)
            )
    return src_out, tgt_out, page_out
