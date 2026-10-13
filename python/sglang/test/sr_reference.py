"""Independent STANDALONE_REMOTE oracles for tests.

These helpers are not on the production path. Slot and attention formulas stay
local so a production change cannot silently rewrite the reference.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch


def _as_int_list(values) -> list[int]:
    if values is None:
        return []
    if isinstance(values, torch.Tensor):
        return [int(x) for x in values.reshape(-1).tolist()]
    return [int(x) for x in values]


def _reference_pages_per_branch(remainder: int, steps: int, page_size: int) -> int:
    page_size = max(int(page_size), 1)
    return int(math.ceil((int(remainder) + int(steps)) / page_size))


def prefix_columns_visible(blocked_mask: torch.Tensor, prefixes: Sequence[int]) -> bool:
    """True when every prefix column of every query is unmasked."""
    for b, prefix in enumerate(prefixes):
        if int(prefix) <= 0:
            continue
        if blocked_mask[b, 0, :, : int(prefix)].any():
            return False
    return True


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


def visible_token_slots_from_pages(
    shared_pages: torch.Tensor,
    branch_pages: torch.Tensor,
    prefix_lens_cpu,
    step_id: int,
    page_size: int,
    dummy_page: int = 0,
) -> list[list[int]]:
    """Oracle visible slots for one step. Host-side, for CPU tests."""
    prefixes = _as_int_list(prefix_lens_cpu)
    page = max(int(page_size), 1)
    topk = int(branch_pages.shape[1]) if branch_pages.dim() == 3 else 1
    shared = shared_pages.detach().cpu()
    branch = branch_pages.detach().cpu()
    rows = []
    for b, prefix in enumerate(prefixes):
        ctx = prefix + int(step_id) + 1
        n_sh = max(int(prefix), 0) // page
        for k in range(topk):
            slots = []
            for t in range(ctx):
                if t < n_sh * page:
                    page_id = int(shared[b, t // page])
                else:
                    bj = (t // page) - n_sh
                    page_id = (
                        int(branch[b, k, bj])
                        if bj < int(branch.shape[2])
                        else int(dummy_page)
                    )
                slots.append(page_id * page + (t % page))
            rows.append(slots)
    return rows


def tree_raw_span_len(remainder: int, topk: int, steps: int, page_size: int) -> int:
    nnp = _reference_pages_per_branch(remainder, steps, page_size)
    return int(topk) * nnp * int(page_size) - int(remainder)


def physical_tree_slot(
    page_ids: Sequence[int],
    branch: int,
    step: int,
    remainder: int,
    nnp: int,
    page_size: int,
) -> int:
    offset = int(remainder) + int(step)
    page_index = int(branch) * int(nnp) + offset // int(page_size)
    return int(page_ids[page_index]) * int(page_size) + offset % int(page_size)


def build_tree_raw_slots(
    page_ids: Sequence[int],
    remainder: int,
    topk: int,
    steps: int,
    page_size: int,
    dtype=torch.int64,
    device=None,
) -> torch.Tensor:
    """Full logical expand span from r to the last exclusive page end."""
    nnp = _reference_pages_per_branch(remainder, steps, page_size)
    span = tree_raw_span_len(remainder, topk, steps, page_size)
    r = int(remainder)
    ps = int(page_size)
    col = torch.arange(span, dtype=torch.int64)
    abs_off = r + col
    branch_span = nnp * ps
    branch = torch.div(abs_off, branch_span, rounding_mode="floor")
    within = abs_off % branch_span
    page_index = branch * nnp + torch.div(within, ps, rounding_mode="floor")
    ids = torch.as_tensor(list(page_ids), dtype=torch.int64)
    slots = ids.index_select(0, page_index) * ps + (within % ps)
    return slots.to(dtype=dtype, device=device)


def remap_slot_node_ids(
    slot_node_ids: torch.Tensor,
    parent_rows: torch.Tensor,
    n_prev_steps: int,
    scratch: torch.Tensor,
) -> None:
    """Gather every historical identity, then scatter it back.

    ``scratch`` is ``[depth_capacity, row_capacity]``. Only ``[:depth, :rows]``
    is written. Padding columns past ``rows`` stay untouched, and there is no
    active-row mask.
    """
    from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
        _execute_identity_remap,
        _validate_identity_remap,
    )

    depth = int(n_prev_steps)
    if depth <= 0:
        return
    _validate_identity_remap(slot_node_ids, parent_rows, depth, scratch)
    _execute_identity_remap(slot_node_ids, parent_rows, depth, scratch)


def lookup_candidate_slots(
    slot_node_ids: torch.Tensor,
    physical_slots: torch.Tensor,
    top_scores_index: torch.Tensor,
    batch_size: int,
    topk: int,
) -> torch.Tensor:
    """Map reply indices to still-live physical slots; missing -> -1.

    CPU reference for tests and kernel checks. The reply path does not call
    this. Lookup is confined to each request's own ``topk`` columns and keeps
    the first match in ``step`` then ``branch`` order. A node id of ``-1``
    still compares equal; it is not a special miss.
    """
    steps = int(slot_node_ids.shape[0])
    bs = int(batch_size)
    k = int(topk)
    rows = bs * k
    cands = top_scores_index
    if cands.dim() == 1:
        cands = cands.unsqueeze(0)
    if cands.dim() != 2:
        raise RuntimeError("candidate indices must be rank 1 or 2")
    n_cand = int(cands.shape[-1])
    if bs == 0 or n_cand == 0:
        return torch.empty((bs, n_cand), dtype=torch.int64, device=cands.device)
    if int(cands.shape[0]) not in (bs, 1):
        raise RuntimeError("candidate batch does not match requests")
    ids = slot_node_ids[:, :rows].reshape(steps, bs, k).tolist()
    phys = physical_slots[:, :rows].reshape(steps, bs, k).tolist()
    cand_rows = cands.tolist()
    if len(cand_rows) == 1 and bs != 1:
        cand_rows = cand_rows * bs
    out = []
    for request in range(bs):
        row = []
        for target in cand_rows[request]:
            target = int(target)
            found = -1
            for step in range(steps):
                matched = False
                for branch in range(k):
                    if int(ids[step][request][branch]) == target:
                        found = int(phys[step][request][branch])
                        matched = True
                        break
                if matched:
                    break
            row.append(found)
        out.append(row)
    return torch.tensor(out, dtype=torch.int64, device=cands.device)


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
    from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
        _export_assembled_rows,
        plan_draft_rows,
        write_draft_regions,
    )

    dev = device if device is not None else torch.device("cpu")
    plan = plan_draft_rows(
        token_rows,
        parent_rows,
        index_rows,
        topk,
        spec_steps,
        num_draft_tokens,
    )
    draft_cpu = torch.empty((plan.bs, plan.token_width), dtype=torch.int64)
    parents_cpu = torch.empty((plan.bs, plan.parent_w), dtype=torch.int64)
    indices_cpu = torch.empty((plan.bs, plan.index_w), dtype=torch.int64)
    write_draft_regions(draft_cpu, parents_cpu, indices_cpu, plan)
    return (
        _export_assembled_rows(parents_cpu, out_parents, dev),
        _export_assembled_rows(indices_cpu, out_indices, dev),
        _export_assembled_rows(draft_cpu, out_tokens, dev),
    )
