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
