"""Convert EAGLE FULL_MASK custom_mask into Ascend FIA polarity/layout.

CUDA ``custom_mask`` is flattened FULL_MASK with True = can attend.
Ascend ``atten_mask`` uses True = masked (see ``generate_mask_flag``).

FIA currently cannot consume a tree mask: CheckFAIMask requires
``sparse_mode=3`` whenever ``atten_mask`` is set, and mode 3 is linear
causal only. TARGET_VERIFY with ``topk>1`` uses slot-gather fallback in
``tree_attn_fallback`` instead of passing this mask to FIA. The conversion
helpers remain for polarity tests and a future FIA tree path.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

FIA_TREE_MASK_CONTRACT = {
    "cuda_true": "attend",
    "ascend_true": "masked",
    "layout": "TND",
    "sparse_mode": 0,
    "fia_consumes_mask": False,
    "shape": "[T, S] = [bs * num_draft, max_kv]",
}


def full_mask_row_starts(seq_lens: Sequence[int], num_draft: int) -> List[int]:
    starts = [0]
    for seq_len in seq_lens:
        starts.append(starts[-1] + (int(seq_len) + num_draft) * num_draft)
    return starts


def iter_full_mask_rows(
    custom_mask: torch.Tensor,
    seq_lens: Sequence[int],
    num_draft: int,
):
    """Yield (batch, draft_idx, attend_row) where attend_row is True=visible."""
    mask = custom_mask.detach()
    offset = 0
    for b, seq_len in enumerate(seq_lens):
        row_len = int(seq_len) + num_draft
        for t in range(num_draft):
            row = mask[offset : offset + row_len]
            yield b, t, row
            offset += row_len


def custom_mask_to_ascend_masked(
    custom_mask: torch.Tensor,
    seq_lens: torch.Tensor,
    num_draft: int,
    *,
    max_kv: Optional[int] = None,
    device=None,
) -> torch.Tensor:
    """Return [bs * num_draft, max_kv] bool mask, True = masked (Ascend)."""
    seq_list = [int(x) for x in seq_lens.tolist()]
    bs = len(seq_list)
    if max_kv is None:
        max_kv = max((s + num_draft for s in seq_list), default=num_draft)
    device = device or custom_mask.device
    out = torch.ones((bs * num_draft, max_kv), dtype=torch.bool, device=device)
    for b, t, row in iter_full_mask_rows(custom_mask, seq_list, num_draft):
        row_len = row.numel()
        # CUDA True=attend → Ascend True=masked
        out[b * num_draft + t, :row_len] = ~row.to(device=device, dtype=torch.bool)
        if row_len < max_kv:
            out[b * num_draft + t, row_len:] = True
    return out


def visible_token_indices(
    attend_row: torch.Tensor,
    prefix_locs: torch.Tensor,
    draft_locs: torch.Tensor,
) -> torch.Tensor:
    """Gather prefix+draft cache locations that this query may attend."""
    seq_len = prefix_locs.numel()
    prefix_vis = attend_row[:seq_len]
    draft_vis = attend_row[seq_len : seq_len + draft_locs.numel()]
    vis = []
    if prefix_vis.any():
        vis.append(prefix_locs[prefix_vis])
    if draft_vis.any():
        vis.append(draft_locs[draft_vis])
    if not vis:
        return prefix_locs.new_empty((0,), dtype=prefix_locs.dtype)
    return torch.cat(vis, dim=0)


def locs_to_page_ids(locs: torch.Tensor, page_size: int) -> torch.Tensor:
    if locs.numel() == 0:
        return locs.to(dtype=torch.int32)
    return (locs.to(dtype=torch.int64) // page_size).to(dtype=torch.int32)


def inplace_update_graph_tree_attn_mask(
    buf: torch.Tensor, converted: torch.Tensor
) -> torch.Tensor:
    """Copy a converted tree mask into the captured graph buffer in place.

    Padding stays True (masked). Returns ``buf`` itself so FIA keeps the
    capture-time storage, shape, and stride.
    """
    if converted.dim() != 2:
        raise ValueError(
            f"converted tree mask must be 2D, got shape {tuple(converted.shape)}"
        )
    if converted.shape[0] > buf.shape[0] or converted.shape[1] > buf.shape[1]:
        raise RuntimeError(
            f"Ascend tree mask {tuple(converted.shape)} exceeds graph buffer "
            f"{tuple(buf.shape)}"
        )
    buf.fill_(True)
    buf[: converted.shape[0], : converted.shape[1]].copy_(converted)
    return buf
