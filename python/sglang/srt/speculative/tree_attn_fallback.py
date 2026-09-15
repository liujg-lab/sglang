"""Eager tree-verify attention by gathering KV at token slots.

FIA rejects ``atten_mask`` unless ``sparse_mode=3``, and mode 3 is linear
causal only. Tree TARGET_VERIFY therefore gathers visible K/V by token slot
(not page id) from CUDA ``custom_mask`` (True = attend) and runs chunked
``QKᵀ → softmax → V``. Siblings that share a page stay distinct slots.

This module must stay free of ``sglang.srt.utils`` so CPU tests can import it
without torchvision.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

from sglang.srt.speculative.tree_attn_mask import (
    iter_full_mask_rows,
    visible_token_indices,
)

SeqLens = Union[torch.Tensor, Sequence[int]]

DEFAULT_ATTN_CHUNK_SIZE = 256


def use_tree_verify_fallback(
    is_target_verify: bool, draft_topk: int, custom_mask
) -> bool:
    """True when TARGET_VERIFY must skip FIA and gather by slot."""
    if not is_target_verify:
        return False
    if int(draft_topk) <= 1:
        return False
    if custom_mask is None:
        return False
    numel = getattr(custom_mask, "numel", None)
    if callable(numel) and int(numel()) == 0:
        return False
    return True


def should_skip_npu_target_verify_graph(device, draft_topk: int) -> bool:
    """NPU tree verify is eager-only; keep ntpb=1 AR graphs."""
    return str(device).startswith("npu") and int(draft_topk) > 1


def flatten_paged_kv(
    cache: torch.Tensor, n_heads: int, head_dim: int
) -> torch.Tensor:
    """Flatten a paged KV cache to ``[num_slots, n_heads, head_dim]``.

    Accepts ``[pages, page_size, H*D]``, ``[pages, page_size, H, D]``,
    ``[tokens, 1, H, D]``, or MLA-style ``[pages, H, page_size, D]``.
    Token slot ``i`` is ``page * page_size + offset``, never a page id.
    """
    if cache.dim() == 4:
        if cache.shape[-1] == head_dim and cache.shape[-2] == n_heads:
            return cache.reshape(-1, n_heads, head_dim)
        if cache.shape[1] == n_heads and cache.shape[-1] == head_dim:
            return cache.permute(0, 2, 1, 3).contiguous().reshape(-1, n_heads, head_dim)
    return cache.reshape(-1, n_heads, head_dim)


def gather_kv_by_slots(
    flat_k: torch.Tensor, flat_v: torch.Tensor, slots: torch.Tensor
):
    """Index-select token slots along axis 0 of flattened K/V."""
    slots = slots.reshape(-1).to(dtype=torch.int64, device=flat_k.device)
    if slots.numel() == 0:
        return (
            flat_k.new_empty((0, flat_k.shape[1], flat_k.shape[2])),
            flat_v.new_empty((0, flat_v.shape[1], flat_v.shape[2])),
        )
    return flat_k.index_select(0, slots), flat_v.index_select(0, slots)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)


def chunked_attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    chunk_size: int = DEFAULT_ATTN_CHUNK_SIZE,
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Online-softmax attention for one query token. ``q`` is ``[H, D]``.

    Visible KV is already gathered; no extra mask. Optional MLA rope terms
    add ``q_rope @ k_ropeᵀ`` to the scores before the shared scale.
    """
    q = q.reshape(-1, q.shape[-1])
    n_q = q.shape[0]
    n_kv = k.shape[1]
    n_rep = n_q // max(int(n_kv), 1)
    k = _repeat_kv(k, n_rep)
    v = _repeat_kv(v, n_rep)
    seq_len = int(k.shape[0])
    v_dim = v.shape[-1]
    if seq_len == 0:
        return q.new_zeros(n_q, v_dim)

    if k_rope is not None:
        k_rope = _repeat_kv(k_rope, n_rep)
        q_rope = q_rope.reshape(-1, q_rope.shape[-1])

    q_f = q.float()
    q_rope_f = q_rope.float() if q_rope is not None else None
    neg_inf = torch.finfo(torch.float32).min
    running_max = q_f.new_full((n_q,), neg_inf)
    running_sum = q_f.new_zeros(n_q)
    acc = q_f.new_zeros(n_q, v_dim)
    chunk = max(int(chunk_size), 1)

    for start in range(0, seq_len, chunk):
        end = min(start + chunk, seq_len)
        k_c = k[start:end].float()
        v_c = v[start:end].float()
        scores = torch.einsum("hd,shd->hs", q_f, k_c)
        if q_rope_f is not None:
            scores = scores + torch.einsum("hd,shd->hs", q_rope_f, k_rope[start:end].float())
        scores = scores * float(scale)
        chunk_max = scores.amax(dim=-1)
        new_max = torch.maximum(running_max, chunk_max)
        alpha = torch.exp(running_max - new_max)
        probs = torch.exp(scores - new_max.unsqueeze(-1))
        acc = acc * alpha.unsqueeze(-1) + torch.einsum("hs,shd->hd", probs, v_c)
        running_sum = running_sum * alpha + probs.sum(dim=-1)
        running_max = new_max

    return (acc / running_sum.clamp_min(1e-20).unsqueeze(-1)).to(dtype=q.dtype)


def _as_seq_list(seq_lens: SeqLens) -> list:
    if isinstance(seq_lens, torch.Tensor):
        return [int(x) for x in seq_lens.detach().reshape(-1).tolist()]
    return [int(x) for x in seq_lens]


def tree_verify_attention(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    custom_mask: torch.Tensor,
    seq_lens: SeqLens,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    out_cache_loc: torch.Tensor,
    num_draft: int,
    scale: float,
    n_q_heads: int,
    n_kv_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    chunk_size: int = DEFAULT_ATTN_CHUNK_SIZE,
    q_rope: Optional[torch.Tensor] = None,
    k_rope_cache: Optional[torch.Tensor] = None,
    rope_head_dim: Optional[int] = None,
) -> torch.Tensor:
    """Per-query slot-gather tree attention. Returns ``[T, n_q_heads * v_head_dim]``."""
    query = query.reshape(-1, n_q_heads, qk_head_dim)
    if q_rope is not None:
        q_rope = q_rope.reshape(-1, n_q_heads, int(rope_head_dim))
    flat_k = flatten_paged_kv(k_cache, n_kv_heads, qk_head_dim)
    flat_v = flatten_paged_kv(v_cache, n_kv_heads, v_head_dim)
    flat_k_rope = None
    if k_rope_cache is not None:
        flat_k_rope = flatten_paged_kv(k_rope_cache, n_kv_heads, int(rope_head_dim))

    seq_list = _as_seq_list(seq_lens)
    req_pool = req_pool_indices.reshape(-1).to(dtype=torch.int64)
    draft_locs_all = out_cache_loc.reshape(-1)
    num_draft = int(num_draft)
    outputs = []
    for b, t, attend_row in iter_full_mask_rows(custom_mask, seq_list, num_draft):
        q_idx = b * num_draft + t
        if q_idx >= query.shape[0]:
            break
        seq_len = seq_list[b]
        req = int(req_pool[b].item())
        prefix_locs = req_to_token[req, :seq_len]
        draft_locs = draft_locs_all[b * num_draft : (b + 1) * num_draft]
        slots = visible_token_indices(attend_row, prefix_locs, draft_locs)
        k_vis, v_vis = gather_kv_by_slots(flat_k, flat_v, slots)
        k_rope_vis = None
        q_rope_t = None
        if flat_k_rope is not None:
            k_rope_vis, _ = gather_kv_by_slots(flat_k_rope, flat_k_rope, slots)
            q_rope_t = q_rope[q_idx]
        out = chunked_attend(
            query[q_idx],
            k_vis,
            v_vis,
            scale,
            chunk_size=chunk_size,
            q_rope=q_rope_t,
            k_rope=k_rope_vis,
        )
        outputs.append(out)

    if not outputs:
        return query.new_zeros(query.shape[0], n_q_heads * v_head_dim)
    stacked = torch.stack(outputs, dim=0)
    if stacked.shape[0] < query.shape[0]:
        stacked = torch.cat(
            [
                stacked,
                stacked.new_zeros(
                    query.shape[0] - stacked.shape[0], n_q_heads, v_head_dim
                ),
            ],
            dim=0,
        )
    return stacked.reshape(stacked.shape[0], n_q_heads * v_head_dim)
