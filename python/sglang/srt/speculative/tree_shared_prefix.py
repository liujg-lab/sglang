"""Shared-prefix tree attention using existing torch operators only.

No accelerator imports: metadata and the production math also run in CPU tests.
Prefix KV is gathered once per request/chunk, never once per query.
"""

from dataclasses import dataclass

import torch

from sglang.srt.speculative.tree_attn_mask import assert_full_mask_layout

PREFIX_CHUNK_SIZE = 256
SHARED_PREFIX_IMPL = "shared_prefix_torch"


@dataclass
class SharedPrefixMetadata:
    prefix_slots: torch.Tensor
    node_slots: torch.Tensor
    ancestor_indices: torch.Tensor
    prefix_lens: torch.Tensor
    path_lens: torch.Tensor

    @classmethod
    def allocate(cls, bs, queries, prefix_cap, nodes, path_cap, device):
        return cls(
            torch.zeros((bs, prefix_cap), dtype=torch.int64, device=device),
            torch.zeros((bs, nodes), dtype=torch.int64, device=device),
            torch.zeros((bs, queries, path_cap), dtype=torch.int64, device=device),
            torch.zeros(bs, dtype=torch.int32, device=device),
            torch.zeros((bs, queries), dtype=torch.int32, device=device),
        )

    def clear(self):
        for tensor in vars(self).values():
            tensor.zero_()


def cpu_prefix_lengths(lengths, raw_bs):
    if isinstance(lengths, torch.Tensor):
        if lengths.device.type != "cpu":
            raise ValueError("shared-prefix preparation requires CPU length metadata")
        lengths = lengths.tolist()
    result = [int(x) for x in lengths[:raw_bs]]
    if len(result) != raw_bs or any(x < 0 for x in result):
        raise ValueError("invalid shared-prefix lengths")
    return result


def _fill_prefix(md, table, pool, lengths):
    bs = len(lengths)
    width = max(lengths, default=0)
    if bs > md.prefix_slots.shape[0] or pool.numel() < bs:
        raise ValueError("shared-prefix batch exceeds buffer capacity")
    if width > min(md.prefix_slots.shape[1], table.shape[1]):
        raise ValueError("shared-prefix length exceeds buffer capacity")
    md.clear()
    seq = torch.tensor(lengths, dtype=torch.int64, device=table.device)
    md.prefix_lens[:bs].copy_(seq)
    if width:
        cols = torch.arange(width, device=table.device)
        values = table[pool[:bs].long(), :width]
        md.prefix_slots[:bs, :width].copy_(
            torch.where(cols[None, :] < seq[:, None], values, 0)
        )
    return seq


def fill_shared_draft_(md, table, pool, lengths, *, page_size, topk, steps, step):
    """The existing branch KV remap has already put ancestors in branch slots."""
    bs = len(lengths)
    if not 0 <= step < steps or md.path_lens.shape[1] != topk:
        raise ValueError("invalid shared-prefix draft step/query count")
    if md.node_slots.shape[1] < topk * steps or md.ancestor_indices.shape[2] < step + 1:
        raise ValueError("shared-prefix draft path exceeds capacity")
    # Check on CPU before device indexing; no device scalar extraction.
    for p in lengths:
        if page_size == 1:
            last = p + (topk - 1) * steps + step
        else:
            pages = (p % page_size + steps + page_size - 1) // page_size
            last = p + (topk - 1) * pages * page_size + step
        if last >= table.shape[1]:
            raise ValueError("shared-prefix draft position exceeds request mapping")
    seq = _fill_prefix(md, table, pool, lengths)
    if not bs:
        return
    branch = torch.arange(topk, device=table.device)
    offset = torch.arange(step + 1, device=table.device)
    if page_size == 1:
        branch_stride = torch.full_like(seq, steps)
    else:
        branch_stride = (
            (seq % page_size + steps + page_size - 1) // page_size
        ) * page_size
    positions = (
        seq[:, None, None]
        + branch_stride[:, None, None] * branch[None, :, None]
        + offset
    )
    nodes = (branch[:, None] * steps + offset).reshape(-1)
    values = table[pool[:bs].long()[:, None, None], positions]
    # ReqToTokenPool stores int32 slots; index_put_ into this int64 buffer
    # rejects mixed dtypes on both CPU and NPU. Branch slots are contiguous,
    # so write the existing buffer through a view without advanced indexing.
    dst = md.node_slots[:bs, : topk * steps].view(bs, topk, steps)
    dst[:, :, : step + 1].copy_(values.to(dtype=md.node_slots.dtype))
    md.ancestor_indices[:bs, :, : step + 1].copy_(nodes.view(topk, step + 1))
    md.path_lens[:bs].fill_(step + 1)


def fill_shared_verify_(md, table, pool, lengths, out_cache_loc, mask, queries):
    """Extract tree columns of FULL_MASK, retaining its producer's row offsets."""
    bs = len(lengths)
    assert_full_mask_layout(mask, lengths, queries, where="shared-prefix verify")
    if md.path_lens.shape[1] != queries or md.node_slots.shape[1] < queries:
        raise ValueError("shared-prefix verify query/node capacity mismatch")
    if md.ancestor_indices.shape[2] < queries or out_cache_loc.numel() < bs * queries:
        raise ValueError("shared-prefix verify path/slot capacity mismatch")
    seq = _fill_prefix(md, table, pool, lengths)
    if not bs:
        return
    widths = seq + queries
    starts = torch.cat((seq.new_zeros(1), (widths * queries).cumsum(0)[:-1]))
    q = torch.arange(queries, device=table.device)
    rows = starts[:, None] + q[None, :] * widths[:, None]
    flat = mask.reshape(-1)
    # The SR producer guarantees a complete prefix. Reject a broken contract
    # asynchronously instead of silently converting a sparse mask to dense.
    prefix_cap = max(lengths, default=0)
    if prefix_cap:
        col = torch.arange(prefix_cap, device=table.device)
        valid = col[None, None, :] < seq[:, None, None]
        idx = torch.where(valid, rows[:, :, None] + col, 0)
        torch._assert_async(
            (flat[idx] | ~valid).all(), "SR tree prefix must be fully visible"
        )
    attend = flat[rows[:, :, None] + seq[:, None, None] + q]
    count = attend.int().sum(-1)
    rank = attend.long().cumsum(-1) - 1
    # A separate dummy column absorbs masked entries without overwriting a
    # valid ancestor; indices and lengths are fixed-shape device tensors.
    scratch = torch.zeros(
        (bs, queries, queries + 1), dtype=torch.int64, device=table.device
    )
    scratch.scatter_(
        -1, torch.where(attend, rank, queries), q.expand(bs, queries, queries)
    )
    md.node_slots[:bs, :queries].copy_(
        out_cache_loc.reshape(-1)[: bs * queries].view(bs, queries)
    )
    md.ancestor_indices[:bs, :, :queries].copy_(scratch[:, :, :queries])
    md.path_lens[:bs].copy_(count)


def cache_view(cache, heads, dim):
    """Only token-major MHA layouts; view must not copy the full KV cache."""
    if cache.ndim == 4 and cache.shape[-2:] == (heads, dim):
        return cache.view(-1, heads, dim)
    if cache.ndim == 3 and (
        cache.shape[-1] == heads * dim or cache.shape[-2:] == (heads, dim)
    ):
        return cache.view(-1, heads, dim)
    raise ValueError("unsupported shared-prefix KV layout")


def shared_prefix_layer_supported(layer):
    attn_type = getattr(getattr(layer, "attn_type", None), "value", None)
    return (
        attn_type == "decoder"
        and layer.qk_head_dim in (64, 128)
        and layer.qk_head_dim == layer.v_head_dim
        and layer.tp_k_head_num == layer.tp_v_head_num
        and layer.tp_k_head_num > 0
        and layer.tp_q_head_num % layer.tp_k_head_num == 0
        and not layer.is_cross_attention
        and layer.sliding_window_size in (-1, None)
        and layer.logit_cap == 0
        and getattr(layer, "quant_method", None) is None
        and getattr(layer, "pos_encoding_mode", "NONE") == "NONE"
        and getattr(layer, "xai_temperature_len", -1) <= 0
        and not getattr(layer, "use_irope", False)
    )


def _gather(cache, slots):
    return (
        cache.index_select(0, slots.reshape(-1))
        .view(*slots.shape, *cache.shape[1:])
        .float()
    )


def _online_update(scores, values, valid, maximum, denominator, accumulator):
    scores = scores.masked_fill(~valid, -torch.inf)
    new_max = torch.maximum(maximum, scores.amax(-1))
    # Finite anchor for completely empty rows avoids inf-inf; their probabilities
    # are masked to zero and their accumulator is unchanged.
    anchor = torch.where(torch.isfinite(new_max), new_max, 0.0)
    alpha = torch.exp(maximum - anchor)
    probabilities = torch.exp(scores - anchor.unsqueeze(-1)).masked_fill(~valid, 0.0)
    return (
        new_max,
        denominator * alpha + probabilities.sum(-1),
        accumulator * alpha.unsqueeze(-1) + torch.matmul(probabilities, values),
    )


def shared_prefix_attention(query, k_cache, v_cache, md, *, scale, kv_heads):
    """Production CPU/NPU math; Q is [B,Q,Hq,D], all reductions use FP32."""
    bs, queries, heads, dim = query.shape
    if heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if md.path_lens.shape != (bs, queries):
        raise ValueError("query/shared-prefix metadata shape mismatch")
    if bs == 0 or queries == 0:
        return query.new_empty((0, heads * dim))
    group = heads // kv_heads
    keys, values = cache_view(k_cache, kv_heads, dim), cache_view(
        v_cache, kv_heads, dim
    )
    q = query.float().reshape(bs, queries, kv_heads, group, dim).permute(0, 2, 1, 3, 4)
    q = q.reshape(bs, kv_heads, queries * group, dim)
    maximum = q.new_full((bs, kv_heads, queries * group), -torch.inf)
    denominator = torch.zeros_like(maximum)
    acc = torch.zeros_like(q)
    for start in range(0, md.prefix_slots.shape[1], PREFIX_CHUNK_SIZE):
        slots = md.prefix_slots[:, start : start + PREFIX_CHUNK_SIZE]
        k = _gather(keys, slots).permute(0, 2, 3, 1)
        v = _gather(values, slots).permute(0, 2, 1, 3)
        col = torch.arange(start, start + slots.shape[1], device=query.device)
        valid = (col[None, :] < md.prefix_lens[:, None]).view(bs, 1, 1, -1)
        k = k.masked_fill(~valid, 0.0)
        v = v.masked_fill(~valid.transpose(-1, -2), 0.0)
        maximum, denominator, acc = _online_update(
            torch.matmul(q, k) * scale, v, valid, maximum, denominator, acc
        )
    # Regroup the same running state for per-query path matrices.
    q = q.view(bs, kv_heads, queries, group, dim).permute(0, 2, 1, 3, 4)
    maximum = maximum.view(bs, kv_heads, queries, group).permute(0, 2, 1, 3)
    denominator = denominator.view(bs, kv_heads, queries, group).permute(0, 2, 1, 3)
    acc = acc.view(bs, kv_heads, queries, group, dim).permute(0, 2, 1, 3, 4)
    path_cap = md.ancestor_indices.shape[-1]
    if path_cap:
        slots = torch.gather(
            md.node_slots, 1, md.ancestor_indices.reshape(bs, -1)
        ).view(bs, queries, path_cap)
        k = _gather(keys, slots).permute(0, 1, 3, 4, 2)
        v = _gather(values, slots).permute(0, 1, 3, 2, 4)
        col = torch.arange(path_cap, device=query.device)
        valid = (col < md.path_lens[:, :, None])[:, :, None, None, :]
        k = k.masked_fill(~valid, 0.0)
        v = v.masked_fill(~valid.transpose(-1, -2), 0.0)
        maximum, denominator, acc = _online_update(
            torch.matmul(q, k) * scale, v, valid, maximum, denominator, acc
        )
    out = acc / denominator.clamp_min(1e-20).unsqueeze(-1)
    return out.reshape(bs * queries, heads * dim).to(query.dtype)
