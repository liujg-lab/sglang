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
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch

from sglang.srt.speculative.tree_attn_mask import (
    assert_full_mask_layout,
    iter_full_mask_rows,
    resolve_tree_verify_mask_seq_lens,
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

# Graph slot-table width cap. Attention walks kv_slots.shape[1]; this does not
# size custom_mask. Default 1024 → tree_attn_chunk_width 128 → 8 chunks.
TREE_GRAPH_MAX_KV = 1024
TREE_GRAPH_MAX_KV_ENV = "SGLANG_NPU_TREE_GRAPH_MAX_KV"
TREE_GRAPH_KV_BUCKETS = (256, 512, 1024, 2048)
TREE_GRAPH_KV_BUCKETS_ENV = "SGLANG_NPU_TREE_GRAPH_KV_BUCKETS"
TREE_DRAFT_CAPTURE_BS = (1, 2)
TREE_DRAFT_CAPTURE_BS_ENV = "SGLANG_NPU_TREE_DRAFT_CAPTURE_BS"


@dataclass(frozen=True)
class TreeReplayPlan:
    """Admission and replay share this key. ``graph_key`` is ``_make_graph_key`` as-is."""

    graph_key: int | str
    raw_bs: int
    capture_bs: int
    tokens_per_req: int
    kv_bucket: int | None


logger = logging.getLogger(__name__)
_LOGGED_TREE_VERIFY_FALLBACK = False
_LOGGED_TREE_DRAFT_SLOT_GATHER = False
_LOGGED_TREE_VERIFY_LAYOUT = False


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


def log_tree_verify_kv_slot_layout_once(
    *,
    mask_numel: int,
    expected_numel: int,
    bs: int,
    raw_bs: int,
    num_draft: int,
    seq_lens_sum: int,
    graph_rows: int,
    max_kv: int,
) -> None:
    """Log once when tree-verify slot-gather fills the KV slot table."""
    global _LOGGED_TREE_VERIFY_LAYOUT
    if _LOGGED_TREE_VERIFY_LAYOUT:
        return
    _LOGGED_TREE_VERIFY_LAYOUT = True
    logger.info(
        "tree verify slot-gather layout: mask_numel=%s expected_numel=%s "
        "bs=%s raw_bs=%s num_draft=%s seq_lens_sum=%s graph_rows=%s max_kv=%s",
        int(mask_numel),
        int(expected_numel),
        int(bs),
        int(raw_bs),
        int(num_draft),
        int(seq_lens_sum),
        int(graph_rows),
        int(max_kv),
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


def parse_tree_graph_max_kv(
    raw: Optional[str] = None,
    *,
    default: int = TREE_GRAPH_MAX_KV,
    env_name: str = TREE_GRAPH_MAX_KV_ENV,
) -> int:
    """Parse a frozen positive slot-table cap.

    ``raw is None`` (env unset) returns ``default``. Empty, zero, negative,
    and non-integer values raise ``ValueError``.
    """
    if raw is None:
        return int(default)
    text = str(raw).strip()
    if not text:
        raise ValueError(f"{env_name} is empty; expected a positive integer")
    try:
        value = int(text, 10)
    except ValueError as e:
        raise ValueError(f"{env_name}={raw!r} is not an integer") from e
    if value <= 0:
        raise ValueError(f"{env_name}={value} must be a positive integer")
    return value


def tree_graph_slot_max_kv(orig_max_kv: int, s_cap: int) -> int:
    """Slot-table width: min(original graph bound, configured S_cap)."""
    return min(int(orig_max_kv), int(s_cap))


def parse_tree_graph_kv_buckets(
    raw: Optional[str] = None,
    *,
    orig_max_kv: int,
    default: Sequence[int] = TREE_GRAPH_KV_BUCKETS,
    env_name: str = TREE_GRAPH_KV_BUCKETS_ENV,
) -> list:
    """Parse capacity buckets, clip to the legal upper bound, sort, and unique.

    ``raw is None`` uses ``default``. Empty tokens and non-positive integers
    raise ``ValueError``. Buckets above ``orig_max_kv`` are dropped; if none
    remain, the clipped original bound is the sole bucket.
    """
    orig_max_kv = int(orig_max_kv)
    if orig_max_kv <= 0:
        raise ValueError(f"orig_max_kv={orig_max_kv} must be a positive integer")
    if raw is None:
        values = [int(x) for x in default]
    else:
        text = str(raw).strip()
        if not text:
            raise ValueError(f"{env_name} is empty; expected comma-separated integers")
        values = []
        for tok in text.split(","):
            piece = tok.strip()
            if not piece:
                raise ValueError(f"{env_name}={raw!r} has an empty bucket")
            try:
                value = int(piece, 10)
            except ValueError as e:
                raise ValueError(
                    f"{env_name}={raw!r} contains a non-integer bucket {piece!r}"
                ) from e
            if value <= 0:
                raise ValueError(
                    f"{env_name}={raw!r} bucket {value} must be a positive integer"
                )
            values.append(value)
    clipped = sorted({min(v, orig_max_kv) for v in values if v > 0})
    if not clipped:
        return [orig_max_kv]
    return clipped


def parse_tree_draft_capture_bs(
    raw: Optional[str] = None,
    *,
    default: Sequence[int] = TREE_DRAFT_CAPTURE_BS,
    env_name: str = TREE_DRAFT_CAPTURE_BS_ENV,
) -> list:
    """Draft graph capture batch sizes. Default is the experiment set ``1,2``."""
    if raw is None:
        values = [int(x) for x in default]
    else:
        text = str(raw).strip()
        if not text:
            raise ValueError(f"{env_name} is empty; expected comma-separated integers")
        values = []
        for tok in text.split(","):
            piece = tok.strip()
            if not piece:
                raise ValueError(f"{env_name}={raw!r} has an empty batch size")
            try:
                value = int(piece, 10)
            except ValueError as e:
                raise ValueError(
                    f"{env_name}={raw!r} contains a non-integer batch {piece!r}"
                ) from e
            if value <= 0:
                raise ValueError(
                    f"{env_name}={raw!r} batch {value} must be a positive integer"
                )
            values.append(value)
    return sorted(set(values))


def select_tree_kv_bucket(needed_kv: int, buckets: Sequence[int]) -> Optional[int]:
    """Smallest bucket that can hold ``needed_kv``, or None if all are too small."""
    needed = int(needed_kv)
    for bucket in buckets:
        if needed <= int(bucket):
            return int(bucket)
    return None


def tree_compact_fia_layout_supported(*, use_mla: bool, has_rope_split: bool) -> bool:
    """First complete path: MHA/GQA dense BSND. MLA stays on chunked attention."""
    return (not bool(use_mla)) and (not bool(has_rope_split))


def tree_fia_actual_seq_lengths_kv(kv_lens: SeqLens, capture_rows: Optional[int] = None):
    """CPU length vector for FIA. Zero-length rows become 1 (zero K/V + clear out)."""
    if isinstance(kv_lens, torch.Tensor):
        values = [int(x) for x in kv_lens.reshape(-1).detach().cpu().tolist()]
    else:
        values = [int(x) for x in kv_lens]
    if capture_rows is not None:
        capture_rows = int(capture_rows)
        if capture_rows < 0:
            raise ValueError(f"capture_rows must be >= 0, got {capture_rows}")
        if len(values) < capture_rows:
            values = values + [0] * (capture_rows - len(values))
        else:
            values = values[:capture_rows]
    return [1 if v <= 0 else v for v in values]


def tree_verify_needed_kv(seq_lens: SeqLens, num_draft: int) -> int:
    """Visible prefix+draft columns for one tree-verify query."""
    seq_list = _as_seq_list(seq_lens)
    if not seq_list:
        return 0
    return max(seq_list) + int(num_draft)


def tree_draft_needed_kv(seq_lens: SeqLens, num_steps: int) -> int:
    """Conservative in-graph draft bound: max(seq) + num_steps."""
    seq_list = _as_seq_list(seq_lens)
    if not seq_list:
        return 0
    return max(seq_list) + max(int(num_steps), 0)


def tree_slot_graph_fits(needed_kv: int, slot_width: Optional[int]) -> bool:
    """True when needed KV columns fit the captured slot table."""
    if slot_width is None:
        return True
    return int(needed_kv) <= int(slot_width)


def tree_slot_graph_needed_kv(
    *,
    is_target_verify: bool,
    is_tree_draft: bool,
    spec_info,
    fallback_seq_lens: SeqLens,
    draft_token_num_fallback: int,
    draft_num_steps: int,
    seq_lens: SeqLens,
) -> Optional[int]:
    """Visible KV columns this tree batch needs, or None when not a tree batch."""
    if is_target_verify:
        seq_list, _raw_bs, _source = resolve_tree_verify_mask_seq_lens(
            spec_info, fallback_seq_lens
        )
        num_draft = int(
            getattr(spec_info, "draft_token_num", None) or draft_token_num_fallback or 1
        )
        return tree_verify_needed_kv(seq_list, num_draft)
    if is_tree_draft:
        return tree_draft_needed_kv(seq_lens, draft_num_steps)
    return None


def tree_slot_graph_can_run_batch(
    *,
    slot_width: Optional[int],
    slot_gather_enabled: bool,
    is_target_verify: bool,
    is_tree_draft: bool,
    spec_info,
    fallback_seq_lens: SeqLens,
    draft_token_num_fallback: int,
    draft_num_steps: int,
    seq_lens: SeqLens,
    buckets: Optional[Sequence[int]] = None,
) -> bool:
    """Admit tree graph replay when needed KV fits a captured slot width.

    Verify uses mask-producer seq_lens and actual ``draft_token_num``.
    Draft uses ``max(seq)+num_steps`` as a conservative bound for every
    in-graph step. Padding rows must not be in the producer seq_lens.
    When ``buckets`` is set, admit if any bucket can hold the need.
    """
    if not slot_gather_enabled:
        return True
    needed = tree_slot_graph_needed_kv(
        is_target_verify=is_target_verify,
        is_tree_draft=is_tree_draft,
        spec_info=spec_info,
        fallback_seq_lens=fallback_seq_lens,
        draft_token_num_fallback=draft_token_num_fallback,
        draft_num_steps=draft_num_steps,
        seq_lens=seq_lens,
    )
    if needed is None:
        return True
    if buckets:
        return select_tree_kv_bucket(needed, buckets) is not None
    if slot_width is None:
        return True
    return tree_slot_graph_fits(needed, slot_width)


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


def gather_kv_into(
    flat_k: torch.Tensor,
    flat_v: torch.Tensor,
    slots: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
) -> None:
    """Gather token slots into preallocated ``k_out`` / ``v_out``.

    Writes through ``index_select(..., out=)`` so a captured graph can reuse
    one scratch pair across layers. ``k_out``/``v_out`` must be contiguous and
    sized ``[R, S, H, D]`` matching ``slots`` ``[R, S]``.
    """
    slots = slots.to(dtype=torch.int64, device=flat_k.device)
    rows, s_cap = int(slots.shape[0]), int(slots.shape[1])
    n_sel = rows * s_cap
    k_view = k_out.reshape(n_sel, flat_k.shape[1], flat_k.shape[2])
    v_view = v_out.reshape(n_sel, flat_v.shape[1], flat_v.shape[2])
    if (not k_view.is_contiguous()) or (not v_view.is_contiguous()):
        raise ValueError("gather_kv_into requires contiguous K/V scratch")
    idx = slots.reshape(-1)
    torch.index_select(flat_k, 0, idx, out=k_view)
    torch.index_select(flat_v, 0, idx, out=v_view)


def zero_gathered_kv_padding(
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    kv_lens: torch.Tensor,
) -> None:
    """Zero padding columns in compact K/V, including all columns of empty rows."""
    s_cap = int(k_out.shape[1])
    lens = kv_lens.reshape(-1).to(device=k_out.device, dtype=torch.int64)
    col = torch.arange(s_cap, device=k_out.device, dtype=torch.int64)
    pad = col.view(1, s_cap) >= lens.view(-1, 1)
    empty = lens <= 0
    pad = pad | empty.view(-1, 1)
    mask = pad.view(k_out.shape[0], s_cap, 1, 1)
    k_out.masked_fill_(mask, 0)
    if v_out is not k_out:
        v_out.masked_fill_(mask, 0)


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


def build_tree_verify_kv_slots_ref(
    custom_mask: torch.Tensor,
    seq_lens: SeqLens,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    out_cache_loc: torch.Tensor,
    num_draft: int,
    max_kv: Optional[int] = None,
    rows_limit: Optional[int] = None,
):
    """Oracle loop for tests. Not used on the tree-attention hot path."""
    seq_list = _as_seq_list(seq_lens)
    bs = len(seq_list)
    num_draft = int(num_draft)
    rows = bs * num_draft
    if rows_limit is not None:
        rows_limit = int(rows_limit)
        if num_draft > 0:
            n_seq = min(bs, max(rows_limit, 0) // num_draft)
        else:
            n_seq = 0
        if n_seq < bs:
            seq_list = seq_list[:n_seq]
            bs = n_seq
            rows = bs * num_draft
    assert_full_mask_layout(
        custom_mask, seq_list, num_draft, where="build_tree_verify_kv_slots_ref"
    )
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
    need_loc = bs * num_draft
    if int(draft_locs_all.numel()) < need_loc:
        raise ValueError(
            "build_tree_verify_kv_slots_ref: out_cache_loc too short: "
            f"numel={int(draft_locs_all.numel())} need={need_loc} "
            f"bs={bs} num_draft={num_draft}"
        )
    if int(req_pool.numel()) < bs:
        raise ValueError(
            "build_tree_verify_kv_slots_ref: req_pool_indices too short: "
            f"numel={int(req_pool.numel())} need={bs}"
        )
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


def _prepare_tree_verify_slot_dims(
    seq_lens: SeqLens,
    num_draft: int,
    rows_limit: Optional[int],
    max_kv: Optional[int],
):
    seq_list = _as_seq_list(seq_lens)
    bs = len(seq_list)
    num_draft = int(num_draft)
    rows = bs * num_draft
    if rows_limit is not None:
        rows_limit = int(rows_limit)
        if num_draft > 0:
            n_seq = min(bs, max(rows_limit, 0) // num_draft)
        else:
            n_seq = 0
        if n_seq < bs:
            seq_list = seq_list[:n_seq]
            bs = n_seq
            rows = bs * num_draft
    if max_kv is None:
        max_kv = max((int(s) + num_draft for s in seq_list), default=num_draft)
    return seq_list, bs, num_draft, rows, int(max_kv)


def build_tree_verify_kv_slots(
    custom_mask: torch.Tensor,
    seq_lens: SeqLens,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    out_cache_loc: torch.Tensor,
    num_draft: int,
    max_kv: Optional[int] = None,
    rows_limit: Optional[int] = None,
):
    """Visible token slots per TARGET_VERIFY query.

    Fixed-shape fill for the replay hot path. Returns
    ``(kv_slots[T, max_kv], kv_lens[T])`` with ``T = bs * num_draft``.
    Padding columns stay 0 and are ignored via ``kv_lens``.
    """
    seq_list, bs, num_draft, rows, max_kv = _prepare_tree_verify_slot_dims(
        seq_lens, num_draft, rows_limit, max_kv
    )
    assert_full_mask_layout(
        custom_mask, seq_list, num_draft, where="build_tree_verify_kv_slots"
    )
    device = req_to_token.device
    slots_out = torch.zeros((rows, max_kv), dtype=torch.int64, device=device)
    lens_out = torch.zeros((rows,), dtype=torch.int32, device=device)
    if rows == 0 or max_kv == 0:
        return slots_out, lens_out

    req_pool = req_pool_indices.reshape(-1).to(device=device, dtype=torch.int64)
    draft_locs_all = out_cache_loc.reshape(-1).to(device=device, dtype=torch.int64)
    need_loc = bs * num_draft
    if int(draft_locs_all.numel()) < need_loc:
        raise ValueError(
            "build_tree_verify_kv_slots: out_cache_loc too short: "
            f"numel={int(draft_locs_all.numel())} need={need_loc} "
            f"bs={bs} num_draft={num_draft}"
        )
    if int(req_pool.numel()) < bs:
        raise ValueError(
            "build_tree_verify_kv_slots: req_pool_indices too short: "
            f"numel={int(req_pool.numel())} need={bs}"
        )

    seq_t = torch.as_tensor(seq_list, dtype=torch.int64, device=device)
    widths = seq_t + num_draft
    max_width = int(widths.max()) if bs else 0
    if max_width > max_kv:
        raise RuntimeError(
            f"tree verify visible slots exceed max_kv={max_kv}: "
            f"max uncompacted row width={max_width}"
        )

    block = widths * num_draft
    block_start = torch.zeros((bs,), dtype=torch.int64, device=device)
    if bs > 1:
        block_start[1:] = torch.cumsum(block[:-1], dim=0)
    t_ids = torch.arange(num_draft, device=device, dtype=torch.int64)
    row_start = (block_start.unsqueeze(1) + t_ids.unsqueeze(0) * widths.unsqueeze(1)).reshape(
        rows
    )
    col = torch.arange(max_kv, device=device, dtype=torch.int64)
    widths_row = widths.unsqueeze(1).expand(bs, num_draft).reshape(rows)
    valid_col = col.unsqueeze(0) < widths_row.unsqueeze(1)
    mask_flat = custom_mask.reshape(-1).to(device=device)
    n_mask = int(mask_flat.numel())
    flat_idx = row_start.unsqueeze(1) + col.unsqueeze(0)
    safe_idx = torch.where(valid_col, flat_idx, torch.zeros_like(flat_idx))
    if n_mask <= 0:
        attend = torch.zeros((rows, max_kv), dtype=torch.bool, device=device)
    else:
        safe_idx = safe_idx.clamp(min=0, max=n_mask - 1)
        attend = mask_flat[safe_idx] & valid_col

    ctx_len = int(req_to_token.shape[1]) if req_to_token.ndim >= 2 else 0
    prefix_table = req_to_token[req_pool[:bs]].to(device=device)
    if ctx_len <= 0:
        prefix_vals = torch.zeros((bs, max_kv), dtype=torch.int64, device=device)
    else:
        prefix_col = col.clamp(max=ctx_len - 1)
        prefix_vals = prefix_table[:, prefix_col].to(dtype=torch.int64)

    seq_col = seq_t.unsqueeze(1)
    is_prefix = col.unsqueeze(0) < seq_col
    draft_off = col.unsqueeze(0) - seq_col
    is_draft = (draft_off >= 0) & (draft_off < num_draft)
    draft_table = draft_locs_all[:need_loc].view(bs, num_draft)
    if num_draft <= 0:
        draft_vals = torch.zeros((bs, max_kv), dtype=torch.int64, device=device)
    else:
        draft_off_safe = draft_off.clamp(min=0, max=num_draft - 1)
        draft_vals = torch.gather(draft_table, 1, draft_off_safe)
    cand_bs = torch.where(
        is_prefix,
        prefix_vals,
        torch.where(is_draft, draft_vals, torch.zeros_like(prefix_vals)),
    )
    cand = (
        cand_bs.unsqueeze(1)
        .expand(bs, num_draft, max_kv)
        .reshape(rows, max_kv)
        .to(dtype=torch.int64)
    )

    lens = attend.to(dtype=torch.int32).sum(dim=-1)
    pos = attend.to(dtype=torch.int64).cumsum(dim=-1) - 1
    dummy = max_kv
    index = torch.where(
        attend & (pos >= 0) & (pos < max_kv),
        pos,
        torch.full_like(pos, dummy),
    )
    padded = slots_out.new_zeros((rows, max_kv + 1))
    padded.scatter_(1, index, cand)
    slots_out.copy_(padded[:, :max_kv])
    lens_out.copy_(lens)
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
