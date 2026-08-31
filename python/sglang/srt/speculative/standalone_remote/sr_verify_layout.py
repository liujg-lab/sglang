"""CPU-safe verify-layout helpers for STANDALONE_REMOTE trees.

Kept free of sglang.srt.utils so unit tests can import it without torchvision.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

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
        flat = src.detach().to("cpu").flatten()
        n = min(int(flat.numel()), token_width)
        if n:
            dest[:n] = flat[:n]
        return
    n = min(len(src), token_width)
    if n:
        dest[:n] = torch.tensor(src[:n], dtype=torch.int64)


def assemble_draft_rows(
    token_rows: Sequence,
    parent_rows: Sequence[Optional[Sequence[int]]],
    index_rows: Sequence[Optional[Sequence[int]]],
    topk: int,
    spec_steps: int,
    num_draft_tokens: int,
    device=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (parent_list, top_scores_index, draft_tokens). Never fakes a bush."""
    dev = device if device is not None else torch.device("cpu")
    bs = len(token_rows)
    token_width = max(num_draft_tokens - 1, 1)
    draft_tokens = torch.zeros(bs, token_width, dtype=torch.int64, device=dev)
    parsed_parents: List[Optional[torch.Tensor]] = [None] * bs
    parsed_indices: List[Optional[torch.Tensor]] = [None] * bs
    n_tree = 0
    for i in range(bs):
        copy_tokens_into_row(draft_tokens[i], token_rows[i], token_width)
        pl = parent_rows[i] if i < len(parent_rows) else None
        ix = index_rows[i] if i < len(index_rows) else None
        if pl is not None and ix is not None:
            n_tree += 1
            parsed_parents[i] = torch.as_tensor(pl, dtype=torch.int64)
            parsed_indices[i] = torch.as_tensor(ix, dtype=torch.int64)
    chain_p, chain_i = chain_tree_structure(num_draft_tokens, spec_steps, device=dev)
    if topk <= 1 or n_tree == 0:
        parents = chain_p.unsqueeze(0).expand(bs, -1).contiguous()
        indices = chain_i.unsqueeze(0).expand(bs, -1).contiguous()
        return parents, indices, draft_tokens
    parent_w = int(chain_p.numel())
    index_w = int(chain_i.numel())
    for p, ix in zip(parsed_parents, parsed_indices):
        if p is not None:
            parent_w = max(parent_w, int(p.numel()))
        if ix is not None:
            index_w = max(index_w, int(ix.numel()))
    parents = torch.full((bs, parent_w), -1, dtype=torch.int64, device=dev)
    indices = torch.zeros((bs, index_w), dtype=torch.int64, device=dev)
    for i, (p, ix) in enumerate(zip(parsed_parents, parsed_indices)):
        if p is None or ix is None:
            n = min(parent_w, int(chain_p.numel()))
            parents[i, :n] = chain_p[:n]
            n = min(index_w, int(chain_i.numel()))
            indices[i, :n] = chain_i[:n]
            continue
        n = min(parent_w, int(p.numel()))
        parents[i, :n] = p.flatten()[:n].to(dev)
        n = min(index_w, int(ix.numel()))
        indices[i, :n] = ix.flatten()[:n].to(dev)
    return parents, indices, draft_tokens
