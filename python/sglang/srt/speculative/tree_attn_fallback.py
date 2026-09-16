"""Eager tree-verify attention by gathering KV at token slots.

FIA rejects ``atten_mask`` unless ``sparse_mode=3``, and mode 3 is linear
causal only. Tree TARGET_VERIFY therefore gathers visible K/V by token slot
(not page id) from CUDA ``custom_mask`` (True = attend) and runs chunked
``QKᵀ → softmax → V``. Siblings that share a page stay distinct slots.

This module must stay free of ``sglang.srt.utils`` so CPU tests can import it
without torchvision.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Union

import torch

from sglang.srt.speculative.tree_attn_mask import (
    iter_full_mask_rows,
    visible_token_indices,
)

SeqLens = Union[torch.Tensor, Sequence[int]]

DEFAULT_ATTN_CHUNK_SIZE = 256

# A device graph unrolls the Python chunk loop, so the node count must not grow
# with the slot-table width. Bound the number of chunks and derive the width
# from it, aligned to a page so a chunk covers whole pages.
MAX_ATTN_CHUNKS = 8
ATTN_CHUNK_ALIGN = 128
# The chunk width is what a gather actually materializes, so cap it too: an
# oversized slot table then costs extra chunks rather than extra memory.
MAX_ATTN_CHUNK_WIDTH = 512

logger = logging.getLogger(__name__)
_LOGGED_TREE_VERIFY_FALLBACK = False
_LOGGED_TREE_DRAFT_SLOT_GATHER = False


def verify_tree_topk_from_server_args(server_args) -> int:
    """Target verify tree width from launch args, not Draft ``draft_topk``."""
    return max(int(getattr(server_args, "speculative_eagle_topk", 1) or 1), 1)


def resolve_backend_topks(ctor_draft_topk, server_args):
    """Mirror AscendAttnBackend construction: Draft layout vs Target verify width.

    Target factory leaves ``ctor_draft_topk=1``. Verify fallback must use
    ``speculative_eagle_topk`` from ``server_args``.
    """
    draft_topk = max(int(ctor_draft_topk), 1)
    verify_tree_topk = verify_tree_topk_from_server_args(server_args)
    return draft_topk, verify_tree_topk


def use_tree_verify_fallback(
    is_target_verify: bool, verify_tree_topk: int, custom_mask
) -> bool:
    """True when TARGET_VERIFY must skip FIA and gather by slot.

    Missing ``custom_mask`` on a tree verify must not silently use linear FIA.
    """
    if not is_target_verify:
        return False
    if int(verify_tree_topk) <= 1:
        return False
    if custom_mask is None:
        raise RuntimeError(
            "TARGET_VERIFY tree attention requires custom_mask; "
            "refusing to fall back to linear FIA"
        )
    numel = getattr(custom_mask, "numel", None)
    if callable(numel) and int(numel()) == 0:
        raise RuntimeError(
            "TARGET_VERIFY tree attention requires a non-empty custom_mask; "
            "refusing to fall back to linear FIA"
        )
    return True


def log_tree_verify_fallback_once(verify_tree_topk: int) -> None:
    """Log once when slot-gather tree verify actually runs."""
    global _LOGGED_TREE_VERIFY_FALLBACK
    if _LOGGED_TREE_VERIFY_FALLBACK:
        return
    _LOGGED_TREE_VERIFY_FALLBACK = True
    logger.info(
        "tree verify slot-gather fallback enabled topk=%s",
        int(verify_tree_topk),
    )


def log_tree_draft_slot_gather_once(draft_topk: int, page_size: int) -> None:
    """Log once when token-level tree-draft attention actually runs."""
    global _LOGGED_TREE_DRAFT_SLOT_GATHER
    if _LOGGED_TREE_DRAFT_SLOT_GATHER:
        return
    _LOGGED_TREE_DRAFT_SLOT_GATHER = True
    logger.info(
        "tree draft token-level slot-gather enabled topk=%s page_size=%s",
        int(draft_topk),
        int(page_size),
    )


def should_skip_npu_target_verify_graph(device, draft_topk: int) -> bool:
    """Tree TARGET_VERIFY now captures NPU graphs over batched slot-gather."""
    return False


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


def tree_attn_chunk_width(max_kv: int, align: int = ATTN_CHUNK_ALIGN) -> int:
    """KV columns per gather chunk.

    Aims for ``ceil(max_kv / width) <= MAX_ATTN_CHUNKS`` so a captured graph
    holds a bounded number of nodes, while never exceeding
    ``MAX_ATTN_CHUNK_WIDTH`` so one gather stays small.
    """
    max_kv = int(max_kv)
    align = max(int(align), 1)
    if max_kv <= 0:
        return align
    width = -(-max_kv // MAX_ATTN_CHUNKS)
    width = -(-width // align) * align
    return max(min(width, MAX_ATTN_CHUNK_WIDTH), align)


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


def _kv_lens_tensor(kv_lens: SeqLens, rows: int, device) -> torch.Tensor:
    if isinstance(kv_lens, torch.Tensor):
        kv_len_t = kv_lens.reshape(-1).to(dtype=torch.int64, device=device)
    else:
        kv_len_t = torch.as_tensor(list(kv_lens), dtype=torch.int64, device=device)
    if int(kv_len_t.numel()) != rows:
        raise ValueError(
            f"tree draft kv_lens length {int(kv_len_t.numel())} != num query rows {rows}"
        )
    return kv_len_t


def build_tree_verify_kv_slots(
    custom_mask: torch.Tensor,
    seq_lens: SeqLens,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    out_cache_loc: torch.Tensor,
    num_draft: int,
    max_kv: Optional[int] = None,
):
    """Visible token slots per TARGET_VERIFY query.

    Host-side (before graph replay). Returns
    ``(kv_slots[T, max_kv], kv_lens[T])`` with ``T = bs * num_draft``.
    Padding columns are ignored via ``kv_lens``.
    """
    seq_list = _as_seq_list(seq_lens)
    bs = len(seq_list)
    num_draft = int(num_draft)
    rows = bs * num_draft
    if max_kv is None:
        max_kv = max((int(s) + num_draft for s in seq_list), default=num_draft)
    max_kv = int(max_kv)
    device = req_to_token.device
    slots_out = torch.zeros((rows, max_kv), dtype=torch.int64, device=device)
    lens_out = torch.zeros((rows,), dtype=torch.int32, device=device)
    if rows == 0 or max_kv == 0:
        return slots_out, lens_out

    req_pool = req_pool_indices.reshape(-1).to(dtype=torch.int64)
    draft_locs_all = out_cache_loc.reshape(-1)
    for b, t, attend_row in iter_full_mask_rows(custom_mask, seq_list, num_draft):
        q_idx = b * num_draft + t
        if q_idx >= rows:
            break
        seq_len = seq_list[b]
        req = int(req_pool[b].item())
        prefix_locs = req_to_token[req, :seq_len]
        draft_locs = draft_locs_all[b * num_draft : (b + 1) * num_draft]
        vis = visible_token_indices(attend_row, prefix_locs, draft_locs)
        n = int(vis.numel())
        if n > max_kv:
            raise RuntimeError(
                f"tree verify visible slots {n} exceed max_kv={max_kv}"
            )
        if n:
            slots_out[q_idx, :n] = vis[:n].to(dtype=torch.int64, device=device)
        lens_out[q_idx] = n
    return slots_out, lens_out


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
    kv_slots: Optional[torch.Tensor] = None,
    kv_lens: Optional[SeqLens] = None,
    kv_bound: Optional[int] = None,
) -> torch.Tensor:
    """Batched slot-gather tree attention. Returns ``[T, n_q_heads * v_head_dim]``.

    ``chunk_size`` is unused; kept so callers and tests do not need a split API.
    """
    del chunk_size
    if kv_slots is None or kv_lens is None:
        kv_slots, kv_lens = build_tree_verify_kv_slots(
            custom_mask,
            seq_lens,
            req_to_token,
            req_pool_indices,
            out_cache_loc,
            num_draft,
        )
    return tree_draft_attention(
        query,
        k_cache,
        v_cache,
        kv_slots=kv_slots,
        kv_lens=kv_lens,
        scale=scale,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
        qk_head_dim=qk_head_dim,
        v_head_dim=v_head_dim,
        q_rope=q_rope,
        k_rope_cache=k_rope_cache,
        rope_head_dim=rope_head_dim,
        kv_bound=kv_bound,
    )


def tree_draft_attention(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    kv_slots: torch.Tensor,
    kv_lens: SeqLens,
    scale: float,
    n_q_heads: int,
    n_kv_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    q_rope: Optional[torch.Tensor] = None,
    k_rope_cache: Optional[torch.Tensor] = None,
    rope_head_dim: Optional[int] = None,
    kv_bound: Optional[int] = None,
) -> torch.Tensor:
    """Batched token-level attention for tree-draft decode.

    ``kv_slots`` is ``[R, S_pad]`` with one row per ``(seq, topk)`` branch.
    Padding columns are ignored via ``kv_lens``.

    KV is gathered one static chunk at a time and combined with online
    softmax, so peak memory follows the chunk width instead of ``S_pad``.
    A dense ``[R, S_pad, H, D]`` gather would be ``S_pad``-sized even when
    every ``kv_lens`` entry is zero, which is what graph capture feeds in.

    ``kv_bound`` caps the columns actually visited. Callers outside a device
    graph may pass ``max(kv_lens)`` to skip all-padding tail chunks; inside a
    graph it must stay ``None`` so the trip count is capture-time constant.
    """
    query = query.reshape(-1, n_q_heads, qk_head_dim)
    rows = int(query.shape[0])
    if rows == 0:
        return query.new_zeros(0, n_q_heads * v_head_dim)

    kv_slots = kv_slots.reshape(rows, -1).to(dtype=torch.int64, device=query.device)
    max_kv = int(kv_slots.shape[1])
    kv_len_t = _kv_lens_tensor(kv_lens, rows, query.device)
    if kv_bound is not None:
        max_kv = min(max_kv, max(int(kv_bound), 0))
    if max_kv == 0:
        return query.new_zeros(rows, n_q_heads * v_head_dim)

    n_kv_heads = max(int(n_kv_heads), 1)
    n_rep = n_q_heads // n_kv_heads
    if n_rep < 1 or n_kv_heads * n_rep != n_q_heads:
        raise ValueError(
            f"tree attention needs n_q_heads ({n_q_heads}) to be a multiple of "
            f"n_kv_heads ({n_kv_heads})"
        )

    flat_k = flatten_paged_kv(k_cache, n_kv_heads, qk_head_dim)
    flat_v = flatten_paged_kv(v_cache, n_kv_heads, v_head_dim)
    flat_k_rope = None
    rd = 0
    if q_rope is not None and k_rope_cache is not None:
        rd = int(rope_head_dim)
        flat_k_rope = flatten_paged_kv(k_rope_cache, n_kv_heads, rd)
        # Grouped like q below: q head h attends kv head h // n_rep.
        q_rope = q_rope.reshape(rows, n_kv_heads, n_rep, rd).float()

    # [R, n_kv, n_rep, D]: q head h pairs with kv head h // n_rep, matching a
    # repeat_interleave of the KV heads, but without materializing the copies.
    q_g = query.view(rows, n_kv_heads, n_rep, qk_head_dim).float()
    scale = float(scale)
    neg_inf = torch.finfo(torch.float32).min
    running_max = q_g.new_full((rows, n_kv_heads, n_rep), neg_inf)
    running_sum = q_g.new_zeros(rows, n_kv_heads, n_rep)
    acc = q_g.new_zeros(rows, n_kv_heads, n_rep, v_head_dim)

    col_all = torch.arange(max_kv, device=query.device, dtype=torch.int64)
    chunk = tree_attn_chunk_width(max_kv)
    for start in range(0, max_kv, chunk):
        width = min(chunk, max_kv - start)
        slots_c = kv_slots[:, start : start + width].clamp(min=0)
        idx = slots_c.reshape(-1)
        k_c = flat_k.index_select(0, idx).view(rows, width, n_kv_heads, qk_head_dim)
        v_c = flat_v.index_select(0, idx).view(rows, width, n_kv_heads, v_head_dim)

        scores = torch.einsum("rkgd,rckd->rkgc", q_g, k_c.float()) * scale
        if flat_k_rope is not None:
            k_rope_c = flat_k_rope.index_select(0, idx).view(
                rows, width, n_kv_heads, rd
            )
            scores = scores + (
                torch.einsum("rkgd,rckd->rkgc", q_rope, k_rope_c.float()) * scale
            )

        pad = col_all[start : start + width].view(1, 1, 1, width) >= kv_len_t.view(
            rows, 1, 1, 1
        )
        scores = scores.masked_fill(pad, neg_inf)

        chunk_max = scores.amax(dim=-1)
        new_max = torch.maximum(running_max, chunk_max)
        alpha = torch.exp(running_max - new_max)
        probs = torch.exp(scores - new_max.unsqueeze(-1)).masked_fill(pad, 0.0)
        acc = acc * alpha.unsqueeze(-1) + torch.einsum(
            "rkgc,rckd->rkgd", probs, v_c.float()
        )
        running_sum = running_sum * alpha + probs.sum(dim=-1)
        running_max = new_max

    out = acc / running_sum.clamp_min(1e-20).unsqueeze(-1)
    return out.to(dtype=query.dtype).reshape(rows, n_q_heads * v_head_dim)
