"""Convert EAGLE FULL_MASK custom_mask into Ascend FIA polarity/layout.

CUDA ``custom_mask`` is flattened FULL_MASK with True = can attend.
Ascend ``atten_mask`` uses True = masked (see ``generate_mask_flag``).

The original TND helpers keep ``FIA_TREE_MASK_CONTRACT``:
``sparse_mode=3`` is linear causal only, so that path still does not feed a
tree mask to FIA. Target ``tree_paged_fia`` uses a separate BSND contract
(``FIA_TREE_MASK_CONTRACT_BSND``): paged KV, per-request ``[B,1,Q,S]`` mask,
and ``sparse_mode=0``. Do not change the old contract; old TND tests depend
on ``fia_consumes_mask=False``.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple, Union

import torch

FIA_TREE_MASK_CONTRACT = {
    "cuda_true": "attend",
    "ascend_true": "masked",
    "layout": "TND",
    "sparse_mode": 0,
    "fia_consumes_mask": False,
    "shape": "[T, S] = [bs * num_draft, max_kv]",
}

FIA_TREE_MASK_CONTRACT_BSND = {
    "cuda_true": "attend",
    "ascend_true": "masked",
    "layout": "BSND",
    "sparse_mode": 0,
    "fia_consumes_mask": True,
    "shape": "[B, 1, Q, S] = [bs, 1, num_draft, pages * page_size]",
}

SeqLens = Union[torch.Tensor, Sequence[int]]

logger = logging.getLogger(__name__)
_LOGGED_MASK_SEQ_LENS_FALLBACK = False


def _as_seq_list(seq_lens: SeqLens) -> List[int]:
    if isinstance(seq_lens, torch.Tensor):
        return [int(x) for x in seq_lens.detach().reshape(-1).tolist()]
    return [int(x) for x in seq_lens]


def _seq_preview(seq_list: Sequence[int], n: int = 8) -> str:
    seq_list = list(seq_list)
    if len(seq_list) <= n:
        return str(seq_list)
    return f"{seq_list[:n]}... (len={len(seq_list)}, sum={sum(seq_list)})"


def full_mask_numel(seq_lens: SeqLens, num_draft: int) -> int:
    """Flattened FULL_MASK size: ``num_draft * (sum(seq_lens) + num_draft * bs)``."""
    seq_list = _as_seq_list(seq_lens)
    num_draft = int(num_draft)
    return num_draft * (sum(seq_list) + num_draft * len(seq_list))


def assert_full_mask_layout(
    custom_mask: torch.Tensor,
    seq_lens: SeqLens,
    num_draft: int,
    *,
    where: str = "tree verify",
) -> int:
    """Raise ``ValueError`` unless ``custom_mask`` matches the FULL_MASK layout.

    Returns the expected numel on success.
    """
    seq_list = _as_seq_list(seq_lens)
    num_draft = int(num_draft)
    expected = full_mask_numel(seq_list, num_draft)
    got = 0 if custom_mask is None else int(custom_mask.numel())
    if got != expected:
        raise ValueError(
            f"{where}: FULL_MASK layout mismatch: "
            f"mask numel={got} expected numel={expected} "
            f"bs={len(seq_list)} num_draft={num_draft} "
            f"seq_lens={_seq_preview(seq_list)} seq_lens_sum={sum(seq_list)}"
        )
    return expected


def resolve_tree_verify_mask_seq_lens(
    spec_info,
    fallback_seq_lens: SeqLens,
) -> Tuple[List[int], int, str]:
    """Pick the seq_lens that produced ``custom_mask``.

    Prefers ``spec_info.seq_lens_cpu`` (snapshotted when the mask was built)
    and cross-checks ``spec_info.seq_lens_sum``. Missing cpu lengths or a sum
    mismatch falls back to ``fallback_seq_lens``.

    Returns ``(seq_list, raw_bs, source)`` with ``source`` in
    ``{"spec_info", "fallback"}``.
    """
    global _LOGGED_MASK_SEQ_LENS_FALLBACK
    fallback_list = _as_seq_list(fallback_seq_lens)
    seq_cpu = getattr(spec_info, "seq_lens_cpu", None) if spec_info is not None else None
    seq_sum = getattr(spec_info, "seq_lens_sum", None) if spec_info is not None else None
    if seq_cpu is not None:
        cpu_list = _as_seq_list(seq_cpu)
        if cpu_list:
            computed = sum(cpu_list)
            if seq_sum is not None and int(seq_sum) != computed:
                reason = (
                    f"spec_info.seq_lens_sum={int(seq_sum)} != "
                    f"seq_lens_cpu sum={computed}"
                )
            else:
                return cpu_list, len(cpu_list), "spec_info"
        else:
            reason = "spec_info.seq_lens_cpu is empty"
    else:
        reason = "spec_info.seq_lens_cpu is missing"
    if not _LOGGED_MASK_SEQ_LENS_FALLBACK:
        _LOGGED_MASK_SEQ_LENS_FALLBACK = True
        logger.warning(
            "tree verify mask layout: %s; falling back to caller seq_lens "
            "bs=%s seq_lens_sum=%s",
            reason,
            len(fallback_list),
            sum(fallback_list),
        )
    return fallback_list, len(fallback_list), "fallback"


def full_mask_row_starts(seq_lens: Sequence[int], num_draft: int) -> List[int]:
    starts = [0]
    num_draft = int(num_draft)
    for seq_len in seq_lens:
        starts.append(starts[-1] + (int(seq_len) + num_draft) * num_draft)
    return starts


def iter_full_mask_rows(
    custom_mask: torch.Tensor,
    seq_lens: Sequence[int],
    num_draft: int,
    *,
    strict: bool = True,
):
    """Yield (batch, draft_idx, attend_row) where attend_row is True=visible."""
    mask = custom_mask.detach().reshape(-1)
    num_draft = int(num_draft)
    seq_list = _as_seq_list(seq_lens)
    offset = 0
    n = int(mask.numel())
    for b, seq_len in enumerate(seq_list):
        row_len = int(seq_len) + num_draft
        for t in range(num_draft):
            if strict and offset + row_len > n:
                raise ValueError(
                    f"FULL_MASK row overflow at batch={b} draft={t} "
                    f"offset={offset} row_len={row_len} mask.numel()={n} "
                    f"bs={len(seq_list)} num_draft={num_draft} "
                    f"seq_lens={_seq_preview(seq_list)} seq_lens_sum={sum(seq_list)}"
                )
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
    seq_list = _as_seq_list(seq_lens)
    num_draft = int(num_draft)
    assert_full_mask_layout(
        custom_mask, seq_list, num_draft, where="custom_mask_to_ascend_masked"
    )
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
    seq_len = int(prefix_locs.numel())
    n_draft = int(draft_locs.numel())
    n_row = int(attend_row.numel())
    if n_row != seq_len + n_draft:
        raise ValueError(
            "attend_row.numel() does not match prefix+draft: "
            f"attend_row={n_row} prefix={seq_len} draft={n_draft}"
        )
    prefix_vis = attend_row[:seq_len]
    draft_vis = attend_row[seq_len : seq_len + n_draft]
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
