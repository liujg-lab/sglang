"""Pack one tree reply straight into the Draft staging buffer.

CPU writes the int64 payload in place. CUDA and NPU use one Triton launch
(chunked only to satisfy the 65535-program grid limit). The packed order is
tokens, parents, indices, then optional candidate slots. Empty regions are
omitted. Candidate lookup keeps the first ``step`` then ``branch`` match.
"""

from __future__ import annotations

import numpy as np
import torch

_INT_DTYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
)


def plan_tree_reply(tokens, parents, indices, write_slots: bool):
    """Return ``(segments, needed)`` for the current reply shapes.

    Each segment is ``(name, shape, offset, length)``. A tensor with no
    elements does not occupy a segment.
    """
    segments = []
    offset = 0
    for name, tensor in (
        ("tokens", tokens),
        ("parents", parents),
        ("indices", indices),
    ):
        length = int(tensor.numel())
        if length == 0:
            continue
        segments.append((name, tuple(int(v) for v in tensor.shape), offset, length))
        offset += length
    if write_slots and int(indices.numel()) > 0:
        length = int(indices.numel())
        segments.append(
            (
                "candidate_slots",
                tuple(int(v) for v in indices.shape),
                offset,
                length,
            )
        )
        offset += length
    return segments, offset


def prepare_tree_reply(
    tokens,
    parents,
    indices,
    node_ids,
    compact,
    *,
    batch: int,
    topk: int,
    steps: int,
    write_slots: bool,
):
    """Validate sources and return the segment plan. Does not write."""
    _validate_sources(
        tokens,
        parents,
        indices,
        node_ids,
        compact,
        batch=batch,
        topk=topk,
        steps=steps,
        write_slots=write_slots,
    )
    return plan_tree_reply(tokens, parents, indices, write_slots)


def pack_tree_reply(
    buf,
    tokens,
    parents,
    indices,
    node_ids,
    compact,
    *,
    batch: int,
    topk: int,
    steps: int,
    write_slots: bool,
) -> None:
    """Write one reply into ``buf`` (1-D int64). Raises before any launch."""
    segments, needed = prepare_tree_reply(
        tokens,
        parents,
        indices,
        node_ids,
        compact,
        batch=batch,
        topk=topk,
        steps=steps,
        write_slots=write_slots,
    )
    del segments
    if buf.ndim != 1 or buf.dtype != torch.int64 or not buf.is_contiguous():
        raise RuntimeError("tree reply buffer must be a contiguous int64 vector")
    if buf.device != tokens.device:
        raise RuntimeError("tree reply buffer must be on the source device")
    if int(buf.numel()) < needed:
        raise RuntimeError("tree reply buffer is shorter than the packed payload")
    if needed == 0:
        return
    if buf.device.type == "cpu":
        _pack_cpu(
            buf[:needed],
            tokens,
            parents,
            indices,
            node_ids,
            compact,
            batch=batch,
            topk=topk,
            steps=steps,
            write_slots=write_slots,
        )
        return
    _pack_device(
        buf[:needed],
        tokens,
        parents,
        indices,
        node_ids,
        compact,
        batch=batch,
        topk=topk,
        steps=steps,
        write_slots=write_slots,
        needed=needed,
    )


def warm_tree_reply_pack(device, topk: int, steps: int, num_draft_tokens: int):
    """Compile both reply-pack kernels before the first request.

    CPU returns immediately. CUDA and NPU each launch the production packer
    once with slots and once without. Pointer dtypes are specialized;
    sizes are not. The caller must keep the returned tensors alive until
    ``warmup_synchronize`` returns, because the launches are still queued.
    """
    dev = torch.device(device)
    if dev.type == "cpu":
        return []
    topk = max(int(topk), 1)
    steps = max(int(steps), 1)
    width = max(int(num_draft_tokens), 1)
    tokens = torch.zeros((1, width), dtype=torch.int64, device=dev)
    parents = torch.zeros((1, width), dtype=torch.int64, device=dev)
    scores = torch.zeros((1, width), dtype=torch.float32, device=dev)
    indices = torch.topk(scores, k=width, dim=-1).indices
    node_ids = torch.zeros((steps, topk), dtype=torch.int64, device=dev)
    compact = torch.zeros((topk * steps,), dtype=torch.int64, device=dev)
    needed = tokens.numel() + parents.numel() + indices.numel()
    buf = torch.empty(needed + indices.numel(), dtype=torch.int64, device=dev)
    pack_tree_reply(
        buf,
        tokens,
        parents,
        indices,
        None,
        None,
        batch=1,
        topk=topk,
        steps=steps,
        write_slots=False,
    )
    pack_tree_reply(
        buf,
        tokens,
        parents,
        indices,
        node_ids,
        compact,
        batch=1,
        topk=topk,
        steps=steps,
        write_slots=True,
    )
    return [buf, tokens, parents, indices, node_ids, compact]


def record_reply_inputs(tensors, device) -> None:
    """Keep source storage alive until the current stream finishes reading it."""
    if device is None:
        return
    dev = torch.device(device)
    if dev.type == "cpu":
        return
    stream = torch.get_device_module(dev.type).current_stream(dev)
    for tensor in tensors:
        if torch.is_tensor(tensor):
            tensor.record_stream(stream)


def _validate_sources(
    tokens,
    parents,
    indices,
    node_ids,
    compact,
    *,
    batch: int,
    topk: int,
    steps: int,
    write_slots: bool,
) -> None:
    sources = (tokens, parents, indices)
    if any(not torch.is_tensor(tensor) or tensor.ndim != 2 for tensor in sources):
        raise RuntimeError("tree reply sources must be rank-2 tensors")
    device = tokens.device
    if any(tensor.device != device for tensor in sources):
        raise RuntimeError("tree reply sources must share one device")
    if any(tensor.dtype not in _INT_DTYPES for tensor in sources):
        raise RuntimeError("tree reply sources must be integer tensors")
    if not write_slots:
        return
    if node_ids is None or compact is None:
        raise RuntimeError("candidate slot pack requires the node and slot tables")
    if (
        not torch.is_tensor(node_ids)
        or node_ids.ndim != 2
        or node_ids.dtype not in _INT_DTYPES
        or node_ids.device != device
    ):
        raise RuntimeError("node id table must be a rank-2 integer tensor")
    if (
        not torch.is_tensor(compact)
        or compact.dtype not in _INT_DTYPES
        or compact.device != device
    ):
        raise RuntimeError("physical slot table must be an integer tensor")
    batch = int(batch)
    topk = int(topk)
    steps = int(steps)
    if batch < 1 or topk < 1 or steps < 1:
        raise RuntimeError("candidate slot pack requires a positive batch, topk, and depth")
    if int(indices.shape[0]) != batch:
        raise RuntimeError("selected indices must have one row per request")
    if int(node_ids.shape[0]) != steps or int(node_ids.shape[1]) < batch * topk:
        raise RuntimeError("node id table does not cover this batch")
    if int(compact.numel()) != batch * topk * steps:
        raise RuntimeError("physical slot table does not cover this batch")
    try:
        compact.view(batch, topk, steps)
    except RuntimeError as exc:
        raise RuntimeError(
            "physical slot table cannot be viewed as [batch, topk, steps]"
        ) from exc


def _region_ends(tokens, parents, indices):
    tok = int(tokens.numel())
    par = tok + int(parents.numel())
    idx = par + int(indices.numel())
    return tok, par, idx


def _pack_cpu(
    buf,
    tokens,
    parents,
    indices,
    node_ids,
    compact,
    *,
    batch: int,
    topk: int,
    steps: int,
    write_slots: bool,
) -> None:
    destination = buf.numpy()
    offset = 0
    for tensor in (tokens, parents, indices):
        offset = _write_matrix(destination, offset, tensor)
    if write_slots and int(indices.numel()) > 0:
        slots = _scan_slots(node_ids, compact, indices, batch, topk, steps)
        destination[offset : offset + slots.size] = slots


def _write_matrix(destination, offset: int, tensor) -> int:
    length = int(tensor.numel())
    if length == 0:
        return offset
    flat = np.ascontiguousarray(tensor.detach().numpy(), dtype=np.int64).reshape(-1)
    destination[offset : offset + length] = flat
    return offset + length


def _scan_slots(node_ids, compact, indices, batch, topk, steps):
    """First match in step-then-branch order. Missing ids store -1."""
    view = compact.view(int(batch), int(topk), int(steps))
    ids = node_ids.detach().numpy()
    phys = view.detach().numpy()
    cands = indices.detach().numpy()
    width = int(cands.shape[1])
    out = np.empty(int(batch) * width, dtype=np.int64)
    pos = 0
    for request in range(int(batch)):
        for cand in range(width):
            target = int(cands[request, cand])
            found = -1
            for step in range(int(steps)):
                hit = False
                for branch in range(int(topk)):
                    col = request * int(topk) + branch
                    if int(ids[step, col]) == target:
                        found = int(phys[request, branch, step])
                        hit = True
                        break
                if hit:
                    break
            out[pos] = found
            pos += 1
    return out


def _pack_device(
    buf,
    tokens,
    parents,
    indices,
    node_ids,
    compact,
    *,
    batch: int,
    topk: int,
    steps: int,
    write_slots: bool,
    needed: int,
) -> None:
    from sglang.srt.speculative.standalone_remote.drafter.sr_tree_reply_pack_kernels import (
        launch_tree_reply_pack,
    )

    tok_end, par_end, idx_end = _region_ends(tokens, parents, indices)
    compact_view = compact
    if write_slots:
        compact_view = compact.view(int(batch), int(topk), int(steps))
    # An empty region is not loaded. Reuse a live source as its pointer so
    # the launch does not allocate a dummy tensor.
    fallback = next(
        tensor for tensor in (tokens, parents, indices) if int(tensor.numel()) > 0
    )
    launch_tree_reply_pack(
        buf,
        tokens if int(tokens.numel()) else fallback,
        parents if int(parents.numel()) else fallback,
        indices if int(indices.numel()) else fallback,
        node_ids,
        compact_view,
        tok_end,
        par_end,
        idx_end,
        int(needed),
        int(topk),
        int(steps),
        bool(write_slots) and int(indices.numel()) > 0,
    )
