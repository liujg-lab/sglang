# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Stateless ClearTree cascade attention.

One launch reads ``req_to_token`` directly. A query row for request ``b``
sees prefix columns ``[0, prefix_len[b])`` and tree columns
``[prefix_len[b], seq_lens[b] + width)``. ``seq_lens`` is the length before
this forward's new tokens. The tree mask has only those tree columns.

The first ``S - 1`` partials split the prefix. The last partial is the tree.
``S == 1`` walks the prefix and the tree in that one partial. ``S`` is chosen
from the batch size and stays fixed for a captured graph.
"""

from __future__ import annotations

import math

import torch

MAX_SPLITS = 32
BLOCK_N = 32
_MAX_QUERY_BLOCK = 64
_EMPTY_LSE = -1.0e20


def cdiv(value: int, block: int) -> int:
    value = int(value)
    block = int(block)
    if block <= 0:
        raise ValueError(f"block must be positive, got {block}")
    return (value + block - 1) // block


def cascade_query_block(rows: int) -> int:
    """``min(64, next_power_of_2(rows))``."""
    rows = max(int(rows), 1)
    return min(_MAX_QUERY_BLOCK, 1 << (rows - 1).bit_length())


def splits_for(bs: int, kv_heads: int, sm_count: int) -> int:
    """Partials for one forward. Larger batches use fewer splits.

    ``clamp(ceil(2 * SM / (kv_heads * bs)), 1, 32)``.
    """
    bs = max(int(bs), 1)
    kv_heads = max(int(kv_heads), 1)
    sm_count = max(int(sm_count), 1)
    raw = math.ceil((2 * sm_count) / (kv_heads * bs))
    return max(1, min(MAX_SPLITS, int(raw)))


def prefix_split_bounds(split: int, prefix_len: int, prefix_splits: int) -> tuple:
    """Half-open token range of one dense prefix partial."""
    split = int(split)
    prefix_len = int(prefix_len)
    prefix_splits = int(prefix_splits)
    if prefix_splits <= 0 or split < 0 or split >= prefix_splits or prefix_len <= 0:
        return 0, 0
    base = prefix_len // prefix_splits
    rem = prefix_len % prefix_splits
    start = split * base + min(split, rem)
    length = base + (1 if split < rem else 0)
    return start, start + length


def fill_cascade_meta(
    prefix_len_buf: torch.Tensor,
    tree_mask_buf: torch.Tensor,
    prefix_rows_buf: torch.Tensor,
    *,
    prefix_lens,
    tree_mask: torch.Tensor | None,
    prefix_rows: torch.Tensor | None,
    seq_lens: torch.Tensor,
    width: int,
    padded_bs: int,
) -> None:
    """Write one forward into static buffers, including graph-padding rows.

    Real requests take ``prefix_lens`` and ``tree_mask``. Padding requests,
    and a capture that has no rows yet, take ``prefix_len = seq_len``,
    an all-ones ``prefix_rows``, and a zero tree mask. The used rows' mask
    tail is cleared, so a shorter layer cannot see the previous layer's columns.
    """
    width = int(width)
    padded_bs = int(padded_bs)
    if width < 1 or padded_bs < 1:
        raise RuntimeError(
            f"cascade metadata needs a positive width and batch, got width={width} bs={padded_bs}"
        )
    raw_bs = 0 if prefix_lens is None else len(prefix_lens)
    if raw_bs > padded_bs:
        raise RuntimeError(
            f"cascade metadata has {raw_bs} requests and padded batch {padded_bs}"
        )
    queries = padded_bs * width
    if padded_bs > int(prefix_len_buf.shape[0]) or queries > int(tree_mask_buf.shape[0]):
        raise RuntimeError(
            f"cascade metadata {padded_bs}x{width} exceeds buffers "
            f"{int(prefix_len_buf.shape[0])}x{int(tree_mask_buf.shape[0])}"
        )
    if queries > int(prefix_rows_buf.shape[0]):
        raise RuntimeError(
            f"cascade prefix rows {queries} exceed buffer {int(prefix_rows_buf.shape[0])}"
        )
    if int(seq_lens.shape[0]) < padded_bs:
        raise RuntimeError(
            f"cascade seq_lens {int(seq_lens.shape[0])} is shorter than batch {padded_bs}"
        )
    device = prefix_len_buf.device
    tree_mask_buf[:queries].zero_()
    prefix_rows_buf[:queries].fill_(1)
    if raw_bs:
        values = torch.tensor(list(prefix_lens), dtype=prefix_len_buf.dtype, device=device)
        prefix_len_buf[:raw_bs].copy_(values)
    pad = padded_bs - raw_bs
    if pad:
        prefix_len_buf[raw_bs:padded_bs].copy_(
            seq_lens[raw_bs:padded_bs].to(dtype=prefix_len_buf.dtype)
        )
    if tree_mask is not None:
        if tree_mask.dim() != 2 or int(tree_mask.shape[0]) != raw_bs * width:
            raise RuntimeError(
                f"cascade tree mask {tuple(tree_mask.shape)} != [{raw_bs * width}, cols]"
            )
        cols = int(tree_mask.shape[1])
        if cols > int(tree_mask_buf.shape[1]):
            raise RuntimeError(
                f"cascade tree columns {cols} exceed buffer {int(tree_mask_buf.shape[1])}"
            )
        if raw_bs and cols:
            tree_mask_buf[: raw_bs * width, :cols].copy_(
                tree_mask.to(device=device, dtype=torch.uint8)
            )
    if prefix_rows is not None:
        if int(prefix_rows.reshape(-1).shape[0]) != raw_bs * width:
            raise RuntimeError(
                f"cascade prefix rows {int(prefix_rows.reshape(-1).shape[0])} != {raw_bs * width}"
            )
        if raw_bs:
            prefix_rows_buf[: raw_bs * width].copy_(
                prefix_rows.reshape(-1).to(device=device, dtype=torch.int32)
            )


def reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_len: torch.Tensor,
    tree_mask: torch.Tensor,
    prefix_rows: torch.Tensor,
    sm_scale: float,
    width: int,
) -> torch.Tensor:
    """Full-softmax attention. One request's rows share ``req_to_token`` columns."""
    width = int(width)
    queries, heads, dim = q.shape
    bs = int(req_pool_indices.shape[0])
    if queries != bs * width:
        raise ValueError(f"queries {queries} != batch {bs} * width {width}")
    kv_heads = int(k_cache.shape[1])
    if heads % kv_heads != 0:
        raise ValueError(f"heads {heads} is not divisible by kv heads {kv_heads}")
    group = heads // kv_heads
    out = torch.zeros((queries, heads, dim), dtype=torch.float32, device=q.device)
    qf = q.float()
    scale = float(sm_scale)
    for batch in range(bs):
        rows = slice(batch * width, (batch + 1) * width)
        qb = qf[rows]
        pool = int(req_pool_indices[batch])
        plen = int(prefix_len[batch])
        seq = int(seq_lens[batch])
        tree_cols = max(seq + width - plen, 0)
        if plen < 0 or tree_cols < 0 or plen + tree_cols > int(req_to_token.shape[1]):
            raise ValueError(
                f"request {batch} columns [{plen}, {plen + tree_cols}) "
                f"do not fit req_to_token width {int(req_to_token.shape[1])}"
            )
        attend = prefix_rows[rows].reshape(-1).to(dtype=torch.bool)
        prefix_k = prefix_v = None
        if plen > 0:
            slots = req_to_token[pool, :plen].to(dtype=torch.long)
            prefix_k = k_cache.index_select(0, slots).float()
            prefix_v = v_cache.index_select(0, slots).float()
        tree_k = tree_v = visible = None
        if tree_cols > 0:
            slots = req_to_token[pool, plen : plen + tree_cols].to(dtype=torch.long)
            tree_k = k_cache.index_select(0, slots).float()
            tree_v = v_cache.index_select(0, slots).float()
            visible = tree_mask[rows, :tree_cols].to(dtype=torch.bool)
        for kv_head in range(kv_heads):
            qh = qb[:, kv_head * group : (kv_head + 1) * group, :]
            scores = []
            values = []
            if prefix_k is not None:
                score = torch.einsum("wgd,pd->wgp", qh, prefix_k[:, kv_head]) * scale
                score = score.masked_fill(~attend[:, None, None], float("-inf"))
                scores.append(score)
                values.append(prefix_v[:, kv_head])
            if tree_k is not None:
                score = torch.einsum("wgd,td->wgt", qh, tree_k[:, kv_head]) * scale
                score = score.masked_fill(~visible[:, None, :], float("-inf"))
                scores.append(score)
                values.append(tree_v[:, kv_head])
            if not scores:
                continue
            cat = torch.cat(scores, dim=-1)
            empty = torch.isneginf(cat).all(dim=-1, keepdim=True)
            weight = torch.softmax(cat.masked_fill(empty, 0), dim=-1)
            weight = weight.masked_fill(empty, 0)
            mixed = torch.einsum("wgt,td->wgd", weight, torch.cat(values, dim=0))
            out[rows, kv_head * group : (kv_head + 1) * group] = mixed
    return out.to(dtype=q.dtype)


def launch(
    q: torch.Tensor,
    o: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_len: torch.Tensor,
    tree_mask: torch.Tensor,
    prefix_rows: torch.Tensor,
    logits: torch.Tensor,
    lse: torch.Tensor,
    sm_scale: float,
    kv_heads: int,
    width: int,
    splits: int,
    bs: int,
) -> None:
    """Run stage 1 and the merge. Every pointer is supplied by the caller."""
    if _stage1 is None or _merge is None:
        raise RuntimeError("triton is required for ClearTree cascade attention")
    width = int(width)
    splits = int(splits)
    bs = int(bs)
    kv_heads = int(kv_heads)
    queries, q_heads, dim = q.shape
    if splits < 1 or splits > MAX_SPLITS:
        raise RuntimeError(f"cascade splits {splits} is outside 1..{MAX_SPLITS}")
    if queries != bs * width:
        raise RuntimeError(f"cascade queries {queries} != batch {bs} * width {width}")
    if q_heads < 1 or kv_heads < 1 or q_heads % kv_heads != 0:
        raise RuntimeError(f"bad GQA shape heads={q_heads} kv_heads={kv_heads}")
    if int(o.shape[0]) != queries or int(prefix_len.shape[0]) < bs:
        raise RuntimeError("cascade output or prefix_len does not cover the batch")
    if int(seq_lens.shape[0]) < bs or int(req_pool_indices.shape[0]) < bs:
        raise RuntimeError("cascade seq_lens or req_pool_indices does not cover the batch")
    if int(tree_mask.shape[0]) < queries or int(prefix_rows.shape[0]) < queries:
        raise RuntimeError("cascade mask or prefix_rows does not cover the queries")
    if (
        int(logits.shape[0]) < queries
        or int(logits.shape[1]) < q_heads
        or int(logits.shape[2]) < splits
        or int(lse.shape[0]) < queries
        or int(lse.shape[2]) < splits
    ):
        raise RuntimeError("cascade partial buffers do not cover this launch")
    import triton

    group = q_heads // kv_heads
    rows_per_req = width * group
    block_q = cascade_query_block(rows_per_req)
    tiles = cdiv(rows_per_req, block_q)
    block_d = triton.next_power_of_2(int(dim))
    _stage1[(kv_heads, splits, bs * tiles)](
        q,
        k_buffer,
        v_buffer,
        req_to_token,
        req_pool_indices,
        seq_lens,
        prefix_len,
        tree_mask,
        prefix_rows,
        logits,
        lse,
        float(sm_scale),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        req_to_token.stride(0),
        req_to_token.stride(1),
        tree_mask.stride(0),
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        lse.stride(0),
        lse.stride(1),
        int(width),
        int(tiles),
        GROUP=group,
        BLOCK_Q=block_q,
        BLOCK_N=BLOCK_N,
        BLOCK_D=block_d,
        DIM=int(dim),
        SPLITS=splits,
        num_warps=4,
        num_stages=1,
    )
    _merge[(queries, q_heads)](
        logits,
        lse,
        o,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        lse.stride(0),
        lse.stride(1),
        o.stride(0),
        o.stride(1),
        SPLITS=splits,
        BLOCK_D=block_d,
        DIM=int(dim),
        num_warps=4,
        num_stages=1,
    )


try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None

if triton is not None:

    @triton.jit
    def _stage1(
        Q,
        K,
        V,
        ReqToToken,
        ReqPool,
        SeqLens,
        PrefixLen,
        TreeMask,
        PrefixRows,
        Out,
        Lse,
        sm_scale,
        stride_q_n,
        stride_q_h,
        stride_k_t,
        stride_k_h,
        stride_v_t,
        stride_v_h,
        stride_req_pool,
        stride_req_tok,
        stride_mask_n,
        stride_o_n,
        stride_o_h,
        stride_o_s,
        stride_lse_n,
        stride_lse_h,
        width,
        tiles_per_req,
        GROUP: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        DIM: tl.constexpr,
        SPLITS: tl.constexpr,
    ):
        kv_head = tl.program_id(0)
        split = tl.program_id(1)
        program = tl.program_id(2)
        req = program // tiles_per_req
        tile = program % tiles_per_req
        prefix_len = tl.load(PrefixLen + req).to(tl.int32)
        seq_len = tl.load(SeqLens + req).to(tl.int32)
        pool = tl.load(ReqPool + req).to(tl.int64)

        rows = tl.arange(0, BLOCK_Q)
        linear = tile * BLOCK_Q + rows
        row_ok = linear < width * GROUP
        node_in_req = tl.where(row_ok, linear // GROUP, 0)
        local_h = tl.where(row_ok, linear % GROUP, 0)
        node = req * width + node_in_req
        q_head = kv_head * GROUP + local_h
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < DIM
        q = tl.load(
            Q + node[:, None] * stride_q_n + q_head[:, None] * stride_q_h + offs_d[None, :],
            mask=row_ok[:, None] & mask_d[None, :],
            other=0.0,
        )
        attend = tl.load(PrefixRows + node, mask=row_ok, other=0)

        # SPLITS == 1 walks the prefix and the tree together. Otherwise the
        # last partial is the tree and the earlier partials split the prefix.
        if SPLITS == 1:
            prefix_start = 0
            prefix_span = prefix_len
            tree_span = tl.maximum(seq_len + width - prefix_len, 0)
        else:
            prefix_splits = SPLITS - 1
            prefix_part = split < prefix_splits
            tree_part = split == SPLITS - 1
            base = prefix_len // prefix_splits
            rem = prefix_len % prefix_splits
            length = base + tl.where(split < rem, 1, 0)
            prefix_start = split * base + tl.minimum(split, rem)
            prefix_span = tl.where(prefix_part, length, 0)
            tree_span = tl.where(tree_part, tl.maximum(seq_len + width - prefix_len, 0), 0)

        acc = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)
        e_max = tl.zeros([BLOCK_Q], dtype=tl.float32) - float("inf")
        e_sum = tl.zeros([BLOCK_Q], dtype=tl.float32)
        token_base = ReqToToken + pool * stride_req_pool

        for off in range(0, prefix_span, BLOCK_N):
            off = tl.multiple_of(off, BLOCK_N)
            offs_n = off + tl.arange(0, BLOCK_N)
            in_bounds = offs_n < prefix_span
            cols = prefix_start + offs_n
            slots = tl.load(token_base + cols * stride_req_tok, mask=in_bounds, other=0).to(
                tl.int64
            )
            k = tl.load(
                K + slots[None, :] * stride_k_t + kv_head * stride_k_h + offs_d[:, None],
                mask=in_bounds[None, :] & mask_d[:, None],
                other=0.0,
            )
            qk = tl.dot(q.to(k.dtype), k) * sm_scale
            qk = tl.where(in_bounds[None, :], qk, float("-inf"))
            qk = tl.where(attend[:, None] != 0, qk, float("-inf"))
            row_max = tl.max(qk, 1)
            row_max = tl.where(row_max == float("-inf"), -1.0e20, row_max)
            n_e_max = tl.maximum(row_max, e_max)
            rescale = tl.exp(e_max - n_e_max)
            prob = tl.exp(qk - n_e_max[:, None])
            v = tl.load(
                V + slots[:, None] * stride_v_t + kv_head * stride_v_h + offs_d[None, :],
                mask=in_bounds[:, None] & mask_d[None, :],
                other=0.0,
            )
            acc = acc * rescale[:, None] + tl.dot(prob.to(v.dtype), v)
            e_sum = e_sum * rescale + tl.sum(prob, 1)
            e_max = n_e_max

        for off in range(0, tree_span, BLOCK_N):
            off = tl.multiple_of(off, BLOCK_N)
            offs_n = off + tl.arange(0, BLOCK_N)
            in_bounds = offs_n < tree_span
            cols = prefix_len + offs_n
            slots = tl.load(token_base + cols * stride_req_tok, mask=in_bounds, other=0).to(
                tl.int64
            )
            k = tl.load(
                K + slots[None, :] * stride_k_t + kv_head * stride_k_h + offs_d[:, None],
                mask=in_bounds[None, :] & mask_d[:, None],
                other=0.0,
            )
            visible = tl.load(
                TreeMask + node[:, None] * stride_mask_n + offs_n[None, :],
                mask=row_ok[:, None] & in_bounds[None, :],
                other=0,
            )
            qk = tl.dot(q.to(k.dtype), k) * sm_scale
            qk = tl.where((visible != 0) & in_bounds[None, :], qk, float("-inf"))
            row_max = tl.max(qk, 1)
            row_max = tl.where(row_max == float("-inf"), -1.0e20, row_max)
            n_e_max = tl.maximum(row_max, e_max)
            rescale = tl.exp(e_max - n_e_max)
            prob = tl.exp(qk - n_e_max[:, None])
            v = tl.load(
                V + slots[:, None] * stride_v_t + kv_head * stride_v_h + offs_d[None, :],
                mask=in_bounds[:, None] & mask_d[None, :],
                other=0.0,
            )
            acc = acc * rescale[:, None] + tl.dot(prob.to(v.dtype), v)
            e_sum = e_sum * rescale + tl.sum(prob, 1)
            e_max = n_e_max

        safe = e_sum > 0
        denom = tl.where(safe, e_sum, 1.0)
        stored = tl.where(safe[:, None], acc / denom[:, None], 0.0)
        lse = tl.where(safe, e_max + tl.log(denom), -1.0e20)
        tl.store(
            Out
            + node[:, None] * stride_o_n
            + q_head[:, None] * stride_o_h
            + split * stride_o_s
            + offs_d[None, :],
            stored,
            mask=row_ok[:, None] & mask_d[None, :],
        )
        tl.store(
            Lse + node * stride_lse_n + q_head * stride_lse_h + split,
            lse,
            mask=row_ok,
        )

    @triton.jit
    def _merge(
        Partial,
        Lse,
        Out,
        stride_n,
        stride_h,
        stride_s,
        stride_lse_n,
        stride_lse_h,
        stride_o_n,
        stride_o_h,
        SPLITS: tl.constexpr,
        BLOCK_D: tl.constexpr,
        DIM: tl.constexpr,
    ):
        row = tl.program_id(0)
        head = tl.program_id(1)
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < DIM
        maximum = tl.full((), -1.0e20, dtype=tl.float32)
        for split in range(SPLITS):
            value = tl.load(Lse + row * stride_lse_n + head * stride_lse_h + split)
            maximum = tl.maximum(maximum, value)
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        denom = tl.zeros((), dtype=tl.float32)
        for split in range(SPLITS):
            value = tl.load(Lse + row * stride_lse_n + head * stride_lse_h + split)
            live = value > -1.0e19
            weight = tl.where(live, tl.exp(value - maximum), 0.0)
            partial = tl.load(
                Partial
                + row * stride_n
                + head * stride_h
                + split * stride_s
                + offs_d,
                mask=mask_d,
                other=0.0,
            )
            acc += weight * partial
            denom += weight
        merged = tl.where(denom > 0, acc / denom, 0.0)
        tl.store(
            Out + row * stride_o_n + head * stride_o_h + offs_d,
            merged,
            mask=mask_d,
        )

else:
    _stage1 = None
    _merge = None


try:
    launch = torch.compiler.disable(launch)
except Exception:
    pass
