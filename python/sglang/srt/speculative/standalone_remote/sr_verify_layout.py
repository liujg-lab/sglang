"""CPU-safe verify-layout helpers for STANDALONE_REMOTE trees.

Kept free of sglang.srt.utils so unit tests can import it without torchvision.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch


def advance_tree_draft_positions(
    positions: Optional[torch.Tensor],
    mrope_positions: Optional[torch.Tensor] = None,
) -> None:
    """Advance RoPE by one tree-decode step in-place.

    EAGLE ``draft_forward`` only does ``positions.add_(1)``. Qwen2/3-VL ignores
    that 1D tensor and reads ``mrope_positions`` (shape ``(3, n)``). After the
    vision prefix, all three M-RoPE axes increment together during text decode,
    so both tensors must move by +1 on every spec step. Leaving M-RoPE frozen
    makes layer-0 drafts match and deeper nodes garbage (accept len stuck at 2).
    """
    if positions is not None:
        positions.add_(1)
    if mrope_positions is not None:
        mrope_positions.add_(1)


def advance_tree_draft_positions_for_step(
    step_i: int,
    positions: Optional[torch.Tensor],
    mrope_positions: Optional[torch.Tensor] = None,
) -> None:
    """Advance RoPE only after the first tree forward.

    STANDALONE initializes ``positions`` / M-RoPE at ``seq_lens``, which is
    already the correct id for the first draft tokens. EAGLE CUDA still
    ``add_(1)`` on every step including ``i==0``; SR full models must not.
    """
    if int(step_i) <= 0:
        return
    advance_tree_draft_positions(positions, mrope_positions)


def export_accepted_tree_candidate_indices(
    accept_index,
    retrive_index=None,
) -> List[List[int]]:
    """Map 2D ``accept_index`` to reply candidate indices (exclude root).

    ``accept_index[b, j]`` stores ``retrive_index`` values. Column 0 is the
    tree root. Later columns that are not ``-1`` are accepted draft nodes.
    Bonus lives in ``predict[last_accepted]``, not an extra accept slot.

    When ``retrive_index`` is omitted, each CPU row uses ``row[0]`` as root.
    Empty rows or invalid roots yield an empty path. Never default root to 0.
    """
    if accept_index is None:
        return []
    if hasattr(accept_index, "detach"):
        rows = accept_index.detach().to("cpu").tolist()
    else:
        rows = list(accept_index)
    if not rows:
        return []
    roots = None
    if retrive_index is not None:
        if hasattr(retrive_index, "detach"):
            roots = retrive_index[:, 0].detach().to("cpu").reshape(-1).tolist()
        else:
            roots = [int(row[0]) for row in retrive_index]
    out: List[List[int]] = []
    for b, row in enumerate(rows):
        seq = row if isinstance(row, (list, tuple)) else [row]
        if roots is not None:
            root = int(roots[b]) if b < len(roots) else None
        elif not seq:
            root = None
        else:
            try:
                root = int(seq[0])
            except (TypeError, ValueError):
                root = None
        if root is None or root < 0:
            out.append([])
            continue
        path: List[int] = []
        for j, idx in enumerate(seq):
            if j == 0:
                continue
            if idx is None or int(idx) < 0:
                break
            local = int(idx) - root
            if local >= 1:
                path.append(local - 1)
        out.append(path)
    return out


def copy_paged_kv_buffer_by_slot(
    kv_buffer: torch.Tensor,
    src_loc: torch.Tensor,
    tgt_loc: torch.Tensor,
) -> None:
    """Copy token slots in a paged KV buffer without 6D advanced indexing.

    ``kv_buffer`` is ``[2, layer, num_pages, page_size, head, dim]``. ``src_loc``
    / ``tgt_loc`` are token-slot indices (``page * page_size + offset``).

    Uses device ``index_select`` / ``index_copy_`` so NPU/CUDA graph replay
    re-reads live indices instead of frozen Python ints from ``.tolist()``.
    Staging keeps overlapping src/tgt memmove-safe. Do not 6D-index pages.
    """
    if tgt_loc is None or src_loc is None:
        return
    if int(tgt_loc.numel()) == 0:
        return
    kv2, layer, _pages, _page_size, head, dim = kv_buffer.shape
    flat = kv_buffer.view(kv2, layer, -1, head, dim)
    src = src_loc.reshape(-1).to(dtype=torch.int64, device=kv_buffer.device)
    tgt = tgt_loc.reshape(-1).to(dtype=torch.int64, device=kv_buffer.device)
    staged = flat.index_select(2, src)
    flat.index_copy_(2, tgt, staged)


def _copy_token_slots(buf: torch.Tensor, src: torch.Tensor, tgt: torch.Tensor) -> None:
    """Copy slots along the token axis of one KV tensor."""
    src = src.to(device=buf.device, dtype=torch.int64)
    tgt = tgt.to(device=buf.device, dtype=torch.int64)
    if buf.dim() >= 5:
        # NPU MLA-style [layer, pages, page_size, ...]
        layer = int(buf.shape[0])
        rest = buf.shape[3:]
        flat = buf.view(layer, -1, *rest)
        staged = flat.index_select(1, src)
        flat.index_copy_(1, tgt, staged)
        return
    staged = buf.index_select(0, src)
    buf.index_copy_(0, tgt, staged)


class UnsupportedTreeKVLayout(RuntimeError):
    """KV pool layout is not one of the staged copy paths."""


def _flat_copy_locs(src_loc, tgt_loc):
    src_empty = src_loc is None or int(src_loc.numel()) == 0
    tgt_empty = tgt_loc is None or int(tgt_loc.numel()) == 0
    if src_empty and tgt_empty:
        return None, None
    if src_loc is None or tgt_loc is None or src_empty or tgt_empty:
        raise RuntimeError("tree KV copy src/dst length mismatch")
    src = src_loc.reshape(-1)
    tgt = tgt_loc.reshape(-1)
    if int(src.numel()) != int(tgt.numel()):
        raise RuntimeError("tree KV copy src/dst length mismatch")
    return src, tgt


def _is_layer_list(buf) -> bool:
    return isinstance(buf, (list, tuple)) and bool(buf) and all(
        torch.is_tensor(item) for item in buf
    )


def _validate_token_axis_buffer(buf: torch.Tensor, *, mla5: bool) -> None:
    if mla5:
        if buf.dim() != 5:
            raise UnsupportedTreeKVLayout(
                f"NPU MLA KV buffer must be 5D, got {tuple(buf.shape)}"
            )
        buf.view(int(buf.shape[0]), -1, *buf.shape[3:])
        return
    if buf.dim() < 1:
        raise UnsupportedTreeKVLayout("token-major KV buffer has no token axis")
    if buf.dim() >= 5:
        buf.view(int(buf.shape[0]), -1, *buf.shape[3:])


def _validate_paged6(kv_buffer: torch.Tensor) -> None:
    if kv_buffer.dim() != 6 or int(kv_buffer.shape[0]) != 2:
        raise UnsupportedTreeKVLayout(
            f"paged kv_buffer must be [2, layer, pages, page_size, head, dim], got {tuple(kv_buffer.shape)}"
        )
    kv_buffer.view(
        int(kv_buffer.shape[0]),
        int(kv_buffer.shape[1]),
        -1,
        int(kv_buffer.shape[4]),
        int(kv_buffer.shape[5]),
    )


def _validate_matching_pair(k_buffer, v_buffer, index_k_buffer, *, mla5: bool) -> None:
    if not torch.is_tensor(k_buffer) or not torch.is_tensor(v_buffer):
        raise UnsupportedTreeKVLayout("K/V buffers must both be tensors")
    _validate_token_axis_buffer(k_buffer, mla5=mla5)
    _validate_token_axis_buffer(v_buffer, mla5=mla5)
    if int(k_buffer.shape[0]) != int(v_buffer.shape[0]):
        raise UnsupportedTreeKVLayout("K/V layer or token axes do not match")
    if index_k_buffer is None:
        return
    if not torch.is_tensor(index_k_buffer):
        raise UnsupportedTreeKVLayout("index_k_buffer must match stacked K layout")
    _validate_token_axis_buffer(index_k_buffer, mla5=mla5)
    if int(index_k_buffer.shape[0]) != int(k_buffer.shape[0]):
        raise UnsupportedTreeKVLayout("index_k_buffer layer axis does not match K")


def _validate_layer_lists(k_buffer, v_buffer, index_k_buffer) -> None:
    if not _is_layer_list(k_buffer) or not _is_layer_list(v_buffer):
        raise UnsupportedTreeKVLayout("per-layer K/V must both be non-empty tensor lists")
    if len(k_buffer) != len(v_buffer):
        raise UnsupportedTreeKVLayout("per-layer K/V list lengths do not match")
    for buf in list(k_buffer) + list(v_buffer):
        _validate_token_axis_buffer(buf, mla5=False)
    if index_k_buffer is None:
        return
    if torch.is_tensor(index_k_buffer):
        _validate_token_axis_buffer(index_k_buffer, mla5=False)
        return
    if not _is_layer_list(index_k_buffer) or len(index_k_buffer) != len(k_buffer):
        raise UnsupportedTreeKVLayout("index_k_buffer list length does not match K")
    for buf in index_k_buffer:
        _validate_token_axis_buffer(buf, mla5=False)


def copy_kv_pool_by_slot(kv_pool, src_loc, tgt_loc) -> None:
    """Copy token slots using an explicit pool layout. Fail before any write."""
    src, tgt = _flat_copy_locs(src_loc, tgt_loc)
    if src is None:
        return
    kv_buffer = getattr(kv_pool, "kv_buffer", None)
    k_buffer = getattr(kv_pool, "k_buffer", None)
    v_buffer = getattr(kv_pool, "v_buffer", None)
    index_k_buffer = getattr(kv_pool, "index_k_buffer", None)

    if torch.is_tensor(kv_buffer) and kv_buffer.dim() == 6:
        _validate_paged6(kv_buffer)
        copy_paged_kv_buffer_by_slot(kv_buffer, src, tgt)
        return
    if _is_layer_list(k_buffer):
        _validate_layer_lists(k_buffer, v_buffer, index_k_buffer)
        copy_mha_kv_by_slot(k_buffer, v_buffer, src, tgt, index_k_buffer)
        return
    if torch.is_tensor(k_buffer) and k_buffer.dim() == 5:
        _validate_matching_pair(k_buffer, v_buffer, index_k_buffer, mla5=True)
        copy_mha_kv_by_slot(k_buffer, v_buffer, src, tgt, index_k_buffer)
        return
    if _is_layer_list(kv_buffer):
        for buf in kv_buffer:
            _validate_token_axis_buffer(buf, mla5=False)
        copy_mha_kv_by_slot(kv_buffer, None, src, tgt)
        return
    if torch.is_tensor(k_buffer):
        _validate_matching_pair(k_buffer, v_buffer, index_k_buffer, mla5=False)
        copy_mha_kv_by_slot(k_buffer, v_buffer, src, tgt, index_k_buffer)
        return
    raise UnsupportedTreeKVLayout("unrecognized tree KV pool layout")


def copy_mha_kv_by_slot(
    k_buffer: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]],
    v_buffer: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]],
    src_loc: Optional[torch.Tensor],
    tgt_loc: Optional[torch.Tensor],
    index_k_buffer: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
) -> None:
    """Copy token slots in per-layer or stacked K/V buffers.

    CUDA MHA/MLA store a list of ``[tokens, ...]`` tensors (index dim 0).
    NPU MLA stores ``[layer, pages, page_size, 1, dim]`` (flatten pages).
    """
    if src_loc is None or tgt_loc is None:
        return
    if int(tgt_loc.numel()) == 0:
        return
    src = src_loc.reshape(-1).to(dtype=torch.int64)
    tgt = tgt_loc.reshape(-1).to(dtype=torch.int64)

    def _copy_one(buf):
        if buf is None:
            return
        if isinstance(buf, torch.Tensor):
            _copy_token_slots(buf, src, tgt)
            return
        for layer_buf in buf:
            _copy_token_slots(layer_buf, src, tgt)

    _copy_one(k_buffer)
    _copy_one(v_buffer)
    _copy_one(index_k_buffer)


def chain_tree_structure(
    num_draft_tokens: int, spec_steps: int, device=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    dev = device if device is not None else torch.device("cpu")
    parent_list = torch.arange(-1, spec_steps - 1, dtype=torch.int64, device=dev)
    top_scores_index = torch.arange(
        max(num_draft_tokens - 1, 0), dtype=torch.int64, device=dev
    )
    return parent_list, top_scores_index


def slice_decode_batch_row(
    tensor: Optional[torch.Tensor],
    row: int,
    n_rows: int,
    token_lens: Optional[Sequence[int]] = None,
) -> Optional[torch.Tensor]:
    """Pick one request's last-token row from decode or packed-extend tensors.

    DECODE logits/hidden are ``[n_rows, ...]``. A packed EXTEND hidden is
    ``[sum(token_lens), ...]`` (FULL capture); take the last token of ``row``
    using ``token_lens`` (same as logits_processor ``cumsum(extend_seq_lens)-1``).
    A single-request EXTEND without lens takes the last row.
    """
    if tensor is None or n_rows <= 0 or row < 0 or row >= n_rows:
        return None
    if tensor.dim() == 0:
        return tensor
    leading = int(tensor.shape[0])
    if leading == n_rows:
        return tensor[row : row + 1]
    if token_lens is not None and len(token_lens) == n_rows:
        lens = [int(x) for x in token_lens]
        if lens[row] > 0 and sum(lens) == leading:
            end = sum(lens[: row + 1])
            return tensor[end - 1 : end]
    if n_rows == 1 and leading >= 1:
        return tensor[-1:]
    return None


def copy_tokens_into_row(
    dest: torch.Tensor, src, token_width: int
) -> None:
    if src is None or token_width <= 0:
        return
    if isinstance(src, torch.Tensor):
        flat = src.detach().reshape(-1)
        if flat.device != dest.device:
            flat = flat.to(dest.device)
        n = min(int(flat.numel()), token_width)
        if n:
            dest[:n] = flat[:n]
        return
    n = min(len(src), token_width)
    if n:
        dest[:n] = torch.as_tensor(src[:n], dtype=torch.int64)


def _export_assembled_rows(
    cpu_tensor: torch.Tensor,
    out: Optional[torch.Tensor],
    device,
) -> torch.Tensor:
    bs, width = int(cpu_tensor.shape[0]), int(cpu_tensor.shape[1])
    if out is not None and out.shape[0] >= bs and out.shape[1] >= width:
        view = out[:bs, :width]
        if width:
            view.copy_(cpu_tensor, non_blocking=view.device.type != "cpu")
        return view
    if device is None or torch.device(device).type == "cpu":
        return cpu_tensor
    return cpu_tensor.to(device=device, non_blocking=True)


def assemble_draft_rows(
    token_rows: Sequence,
    parent_rows: Sequence[Optional[Sequence[int]]],
    index_rows: Sequence[Optional[Sequence[int]]],
    topk: int,
    spec_steps: int,
    num_draft_tokens: int,
    device=None,
    out_tokens: Optional[torch.Tensor] = None,
    out_parents: Optional[torch.Tensor] = None,
    out_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (parent_list, top_scores_index, draft_tokens). Never fakes a bush."""
    dev = device if device is not None else torch.device("cpu")
    bs = len(token_rows)
    token_width = max(num_draft_tokens - 1, 1)
    draft_cpu = torch.zeros(bs, token_width, dtype=torch.int64)
    parsed_parents: List[Optional[torch.Tensor]] = [None] * bs
    parsed_indices: List[Optional[torch.Tensor]] = [None] * bs
    n_tree = 0
    for i in range(bs):
        copy_tokens_into_row(draft_cpu[i], token_rows[i], token_width)
        pl = parent_rows[i] if i < len(parent_rows) else None
        ix = index_rows[i] if i < len(index_rows) else None
        if pl is not None and ix is not None:
            n_tree += 1
            parsed_parents[i] = torch.as_tensor(pl, dtype=torch.int64)
            parsed_indices[i] = torch.as_tensor(ix, dtype=torch.int64)
    chain_p, chain_i = chain_tree_structure(num_draft_tokens, spec_steps, device="cpu")
    if topk <= 1 or n_tree == 0:
        parents_cpu = chain_p.unsqueeze(0).expand(bs, -1).contiguous()
        indices_cpu = chain_i.unsqueeze(0).expand(bs, -1).contiguous()
        return (
            _export_assembled_rows(parents_cpu, out_parents, dev),
            _export_assembled_rows(indices_cpu, out_indices, dev),
            _export_assembled_rows(draft_cpu, out_tokens, dev),
        )
    parent_w = int(chain_p.numel())
    index_w = int(chain_i.numel())
    for p, ix in zip(parsed_parents, parsed_indices):
        if p is not None:
            parent_w = max(parent_w, int(p.numel()))
        if ix is not None:
            index_w = max(index_w, int(ix.numel()))
    parents_cpu = torch.full((bs, parent_w), -1, dtype=torch.int64)
    indices_cpu = torch.zeros((bs, index_w), dtype=torch.int64)
    for i, (p, ix) in enumerate(zip(parsed_parents, parsed_indices)):
        if p is None or ix is None:
            n = min(parent_w, int(chain_p.numel()))
            parents_cpu[i, :n] = chain_p[:n]
            n = min(index_w, int(chain_i.numel()))
            indices_cpu[i, :n] = chain_i[:n]
            continue
        n = min(parent_w, int(p.numel()))
        parents_cpu[i, :n] = p.flatten()[:n]
        n = min(index_w, int(ix.numel()))
        indices_cpu[i, :n] = ix.flatten()[:n]
    return (
        _export_assembled_rows(parents_cpu, out_parents, dev),
        _export_assembled_rows(indices_cpu, out_indices, dev),
        _export_assembled_rows(draft_cpu, out_tokens, dev),
    )
