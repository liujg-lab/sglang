"""Target tree paged FIA metadata for STANDALONE_REMOTE NPU verify.

CPU-importable. Lengths come from the FULL_MASK producer. Page tables read
logical page starts from the request mapping already written by
``prepare_for_verify``. Do not ``.item()`` / ``.tolist()`` device tensors,
and do not use ``nonzero()`` or data-dependent shapes.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch

from sglang.srt.speculative.standalone_remote.drafter.sr_tree_paged_layout import (
    IMPL_PAGED_FIA,
    validate_tree_draft_paged_records,
)
from sglang.srt.speculative.tree_attn_mask import assert_full_mask_layout
from sglang.srt.speculative.tree_shared_prefix import cpu_prefix_lengths

SeqLens = Union[torch.Tensor, Sequence[int]]

SR_TARGET_TREE_FIA_ENV = "SGLANG_NPU_SR_TARGET_TREE_FIA"
SR_TARGET_UPDATE_OVERLAP_ENV = "SGLANG_NPU_SR_TARGET_UPDATE_OVERLAP"
IMPL_TREE_PAGED_FIA = "tree_paged_fia"
TARGET_TREE_FIA_PAGE_SIZE = 128
TARGET_TREE_FIA_HEAD_DIMS = (64, 128)
TARGET_TREE_FIA_KV_ATTR = "actual_seq_lengths_kv"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def read_sr_target_tree_fia_env(env=None) -> bool:
    """Read once during initialization; default eligible Targets to FIA."""
    environ = os.environ if env is None else env
    raw = environ.get(SR_TARGET_TREE_FIA_ENV, "1")
    return str(raw).strip().lower() in _TRUTHY


def read_sr_target_update_overlap_env(env=None) -> bool:
    """Read once during NPU graph runner init; default off.

    Requests SR Target ``tree_paged_fia`` update/replay overlap. Set 1 to
    enable after the runner's own gate. Not a runtime toggle.
    """
    environ = os.environ if env is None else env
    raw = environ.get(SR_TARGET_UPDATE_OVERLAP_ENV, "0")
    return str(raw).strip().lower() in _TRUTHY


def pages_for_s_cap(s_cap: int, page_size: int) -> int:
    page = max(int(page_size), 1)
    return max((int(s_cap) + page - 1) // page, 1)


def mask_width_from_pages(pages: int, page_size: int) -> int:
    return int(pages) * max(int(page_size), 1)


def target_tree_fia_blocked_extra_combos(args) -> Optional[str]:
    """First-cut extras that keep the existing Target path."""
    if args is None:
        return None
    if bool(getattr(args, "enable_dp_attention", False)):
        return "dp attention"
    cp = int(getattr(args, "attn_cp_size", 1) or 1)
    if (
        cp > 1
        or bool(getattr(args, "enable_prefill_context_parallel", False))
        or bool(getattr(args, "enable_nsa_prefill_context_parallel", False))
    ):
        return "context parallel"
    if int(getattr(args, "pp_size", 1) or 1) > 1:
        return "pipeline parallel"
    if bool(getattr(args, "enable_two_batch_overlap", False)):
        return "two batch overlap"
    if bool(getattr(args, "enable_pdmux", False)):
        return "pdmux"
    return None


def maybe_select_target_tree_fia(
    current_impl,
    reason,
    *,
    requested: bool,
    capable: bool,
    extra_reason: Optional[str],
    page_size: int,
    role,
    verify_topk: int,
):
    """Upgrade a capable Target to tree_paged_fia, otherwise keep the prior pick."""
    if str(role or "") != "target" or int(verify_topk) <= 1:
        return current_impl, reason
    if not requested:
        return current_impl, reason or f"{SR_TARGET_TREE_FIA_ENV}=0"
    if not capable:
        return current_impl, reason
    if extra_reason:
        return current_impl, reason or extra_reason
    if int(page_size) != TARGET_TREE_FIA_PAGE_SIZE:
        return current_impl, reason or (
            f"page_size={int(page_size)} not {TARGET_TREE_FIA_PAGE_SIZE}"
        )
    return IMPL_TREE_PAGED_FIA, None


def plan_target_tree_fia_lengths(
    prefix_lens: SeqLens,
    queries: int,
    raw_bs: int,
    capture_bs: int,
) -> Tuple[List[int], List[int], List[int]]:
    """Return ``(prefixes, kv_lens_cpu, q_lens_cpu)``.

    ``kv_len[b] = P[b] + Q`` for real rows. Padding rows use ``kv_len=1``.
    Query lengths are ``[Q] * capture_bs``, not TND cumulative lengths.
    """
    prefixes = cpu_prefix_lengths(prefix_lens, int(raw_bs))
    queries = int(queries)
    capture_bs = int(capture_bs)
    raw_bs = int(raw_bs)
    if capture_bs < raw_bs:
        raise ValueError(f"capture_bs={capture_bs} smaller than raw_bs={raw_bs}")
    if queries < 1:
        raise ValueError(f"queries must be >= 1, got {queries}")
    kv_lens = [int(p) + queries for p in prefixes] + [1] * (capture_bs - raw_bs)
    q_lens = [queries] * capture_bs
    return prefixes, kv_lens, q_lens


def validate_target_tree_fia_inputs(
    prefix_lens: SeqLens,
    custom_mask: torch.Tensor,
    queries: int,
    raw_bs: int,
    capture_bs: int,
    pages: int,
    page_size: int,
    *,
    table_width: Optional[int] = None,
    where: str = "target tree fia",
) -> List[int]:
    """Reject bad lengths/mask/capacity before any metadata write."""
    if isinstance(prefix_lens, torch.Tensor):
        if prefix_lens.device.type != "cpu":
            raise ValueError(f"{where}: prefix lengths must be a CPU tensor")
        if prefix_lens.ndim != 1:
            raise ValueError(
                f"{where}: prefix lengths ndim={prefix_lens.ndim} must be 1"
            )
    prefixes, kv_lens, _q_lens = plan_target_tree_fia_lengths(
        prefix_lens, queries, raw_bs, capture_bs
    )
    queries = int(queries)
    assert_full_mask_layout(custom_mask, prefixes, queries, where=where)
    page = max(int(page_size), 1)
    cap = mask_width_from_pages(pages, page)
    needed = max(kv_lens[: int(raw_bs)], default=queries)
    if needed > cap:
        raise ValueError(
            f"{where}: needed kv {needed} exceeds page-aligned capacity {cap} "
            f"(pages={int(pages)} page_size={page})"
        )
    if table_width is not None and needed > int(table_width):
        raise ValueError(
            f"{where}: request mapping width {int(table_width)} < needed {needed}"
        )
    return prefixes


@dataclass
class SRTargetTreeFiaMetadata:
    block_tables: torch.Tensor
    blocked_mask: torch.Tensor
    kv_lens_cpu: List[int]
    q_lens_cpu: List[int]
    active_rows: torch.Tensor
    page_size: int

    @classmethod
    def allocate(cls, bs: int, queries: int, pages: int, page_size: int, device):
        page = max(int(page_size), 1)
        width = mask_width_from_pages(pages, page)
        bs = int(bs)
        queries = int(queries)
        pages = int(pages)
        return cls(
            block_tables=torch.zeros((bs, pages), dtype=torch.int32, device=device),
            blocked_mask=torch.ones(
                (bs, 1, queries, width), dtype=torch.bool, device=device
            ),
            kv_lens_cpu=[1] * bs,
            q_lens_cpu=[queries] * bs,
            active_rows=torch.zeros((bs,), dtype=torch.bool, device=device),
            page_size=page,
        )

    def clear(self, dummy_page: int = 0) -> None:
        dummy = int(dummy_page)
        self.block_tables.fill_(dummy)
        self.blocked_mask.fill_(True)
        if self.blocked_mask.shape[-1] > 0:
            self.blocked_mask[:, :, :, 0] = False
        self.active_rows.zero_()
        bs = int(self.block_tables.shape[0])
        queries = int(self.blocked_mask.shape[2])
        self.kv_lens_cpu = [1] * bs
        self.q_lens_cpu = [queries] * bs


def prime_target_tree_fia_capture_(
    md: SRTargetTreeFiaMetadata, dummy_page: int = 0
) -> None:
    """Legal capture-time contents: dummy pages and one unmasked dummy column."""
    md.clear(dummy_page=dummy_page)


def fill_target_tree_fia_page_tables_(
    md: SRTargetTreeFiaMetadata,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    prefixes: Sequence[int],
    queries: int,
    *,
    dummy_page: int = 0,
) -> None:
    """Gather physical page ids from logical page starts only."""
    raw_bs = len(prefixes)
    capture_bs = int(md.block_tables.shape[0])
    pages = int(md.block_tables.shape[1])
    page = max(int(md.page_size), 1)
    queries = int(queries)
    dummy = int(dummy_page)
    if raw_bs > capture_bs:
        raise ValueError("target tree fia batch exceeds page-table rows")
    if req_pool_indices.numel() < raw_bs:
        raise ValueError("target tree fia request pool shorter than raw_bs")
    needed = max((int(p) + queries for p in prefixes), default=queries)
    if needed > int(req_to_token.shape[1]):
        raise ValueError(
            "target tree fia request mapping shorter than P+Q: "
            f"width={int(req_to_token.shape[1])} needed={needed}"
        )
    md.block_tables.fill_(dummy)
    if raw_bs == 0 or pages == 0:
        return
    device = md.block_tables.device
    pool = req_pool_indices[:raw_bs].to(device=device, dtype=torch.int64)
    kv_len = torch.tensor(
        [int(p) + queries for p in prefixes],
        dtype=torch.int64,
        device=device,
    )
    page_starts = torch.arange(pages, dtype=torch.int64, device=device) * page
    valid = page_starts[None, :] < kv_len[:, None]
    safe_pos = torch.where(valid, page_starts[None, :], page_starts.new_zeros(()))
    slots = req_to_token.to(device=device)[pool[:, None], safe_pos]
    page_ids = (slots.to(dtype=torch.int64) // page).to(dtype=torch.int32)
    md.block_tables[:raw_bs].copy_(
        torch.where(valid, page_ids, page_ids.new_full((), dummy))
    )


def fill_target_tree_fia_mask_(
    md: SRTargetTreeFiaMetadata,
    custom_mask: torch.Tensor,
    prefixes: Sequence[int],
    queries: int,
) -> None:
    """Convert FULL_MASK (True=attend) into ``[B,1,Q,S]`` True=masked."""
    raw_bs = len(prefixes)
    capture_bs = int(md.blocked_mask.shape[0])
    queries = int(md.blocked_mask.shape[2])
    width = int(md.blocked_mask.shape[-1])
    q = int(queries)
    assert_full_mask_layout(
        custom_mask, prefixes, q, where="target tree fia mask"
    )
    md.blocked_mask.fill_(True)
    if raw_bs == 0 or q == 0 or width == 0:
        if width > 0:
            md.blocked_mask[:, :, :, 0] = False
        return
    device = md.blocked_mask.device
    flat = custom_mask.reshape(-1).to(device=device, dtype=torch.bool)
    prefix_t = torch.tensor(
        [int(p) for p in prefixes] + [0] * (capture_bs - raw_bs),
        dtype=torch.int64,
        device=device,
    )
    row_width = prefix_t + q
    active = torch.zeros((capture_bs,), dtype=torch.bool, device=device)
    if raw_bs:
        active[:raw_bs] = True
    starts = torch.zeros((capture_bs,), dtype=torch.int64, device=device)
    if capture_bs > 1:
        starts[1:] = (q * torch.where(active, row_width, row_width.new_zeros(())))[:-1].cumsum(
            0
        )
    q_idx = torch.arange(q, dtype=torch.int64, device=device)
    s_idx = torch.arange(width, dtype=torch.int64, device=device)
    index = (
        starts[:, None, None]
        + q_idx[None, :, None] * row_width[:, None, None]
        + s_idx[None, None, :]
    )
    valid = active[:, None, None] & (s_idx[None, None, :] < row_width[:, None, None])
    n = int(flat.numel())
    safe = index.clamp(0, max(n - 1, 0)) if n else index
    attend = flat[safe] if n else torch.zeros_like(valid)
    blocked = torch.where(valid, ~attend, torch.ones_like(valid))
    md.blocked_mask.copy_(blocked.unsqueeze(1))
    if capture_bs > raw_bs:
        md.blocked_mask[raw_bs:, :, :, 0] = False


def fill_target_tree_fia_metadata_(
    md: SRTargetTreeFiaMetadata,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    custom_mask: torch.Tensor,
    prefix_lens: SeqLens,
    queries: int,
    raw_bs: int,
    *,
    dummy_page: int = 0,
    table_width: Optional[int] = None,
) -> None:
    """Validate, then refresh page tables, mask, lengths, and active rows."""
    capture_bs = int(md.block_tables.shape[0])
    pages = int(md.block_tables.shape[1])
    prefixes = validate_target_tree_fia_inputs(
        prefix_lens,
        custom_mask,
        queries,
        raw_bs,
        capture_bs,
        pages,
        md.page_size,
        table_width=table_width
        if table_width is not None
        else int(req_to_token.shape[1]),
    )
    _, kv_lens, q_lens = plan_target_tree_fia_lengths(
        prefixes, queries, raw_bs, capture_bs
    )
    fill_target_tree_fia_page_tables_(
        md,
        req_to_token,
        req_pool_indices,
        prefixes,
        queries,
        dummy_page=dummy_page,
    )
    fill_target_tree_fia_mask_(md, custom_mask, prefixes, queries)
    md.kv_lens_cpu = list(kv_lens)
    md.q_lens_cpu = list(q_lens)
    md.active_rows.zero_()
    if raw_bs:
        md.active_rows[: int(raw_bs)] = True


def prefix_columns_visible(
    blocked_mask: torch.Tensor, prefixes: Sequence[int]
) -> bool:
    """True when every prefix column of every query is unmasked."""
    for b, prefix in enumerate(prefixes):
        if int(prefix) <= 0:
            continue
        if blocked_mask[b, 0, :, : int(prefix)].any():
            return False
    return True


def validate_target_tree_fia_records(
    records,
    num_layers,
    kv_attr: str = TARGET_TREE_FIA_KV_ATTR,
):
    """Require ``num_layers`` FIA v1 records with BSND KV length attrs."""
    return validate_tree_draft_paged_records(
        records,
        1,
        num_layers,
        kv_attr,
        IMPL_PAGED_FIA,
    )


def dense_target_tree_fia_reference(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    kv_lens: Sequence[int],
    blocked_mask: torch.Tensor,
    *,
    scale: float,
    page_size: int,
    n_kv_heads: int,
) -> torch.Tensor:
    """FP32 dense softmax reference for one BSND paged tree call.

    ``query`` is ``[B,Q,Hq,D]``. Caches are ``[n_pages, page_size, Hkv*D]``.
    """
    bs, queries, n_q, dim = query.shape
    page = max(int(page_size), 1)
    group = n_q // int(n_kv_heads)
    q = query.float()
    k = k_cache.float().view(-1, page, int(n_kv_heads), dim)
    v = v_cache.float().view(-1, page, int(n_kv_heads), dim)
    blocked = blocked_mask.reshape(bs, queries, -1)
    outs = []
    for b in range(bs):
        kv_len = int(kv_lens[b])
        tokens = []
        for s in range(kv_len):
            page_id = int(block_tables[b, s // page])
            tokens.append(k[page_id, s % page])
        keys = torch.stack(tokens, dim=0) if tokens else k.new_zeros((0, int(n_kv_heads), dim))
        vals = (
            torch.stack(
                [
                    v[int(block_tables[b, s // page]), s % page]
                    for s in range(kv_len)
                ],
                dim=0,
            )
            if tokens
            else v.new_zeros((0, int(n_kv_heads), dim))
        )
        keys = keys.repeat_interleave(group, dim=1)
        vals = vals.repeat_interleave(group, dim=1)
        row_block = blocked[b, :, :kv_len]
        for r in range(queries):
            scores = torch.einsum("hd,shd->hs", q[b, r], keys) * float(scale)
            scores = scores.masked_fill(row_block[r].to(dtype=torch.bool), -math.inf)
            finite = torch.isfinite(scores)
            if not finite.any():
                outs.append(q.new_zeros((n_q, dim)))
                continue
            weights = torch.softmax(scores, dim=-1)
            outs.append(torch.einsum("hs,shd->hd", weights, vals))
    return torch.stack(outs, dim=0).reshape(bs * queries, n_q * dim)
