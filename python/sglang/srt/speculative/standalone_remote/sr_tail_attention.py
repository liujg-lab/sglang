"""CPU-planned paged attention metadata for a packed, causal text tail."""

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class SRTailAttentionMetadata:
    block_tables: torch.Tensor
    context_lens_cpu: torch.Tensor
    context_lens_list: List[int]


def build_tail_attention_metadata(prefix_lens, extend_lens, block_tables):
    """One query row per tail token, sharing the request's physical pages.

    All lengths come from the CPU batch plan. No device-to-host reads and no
    per-query KV gather are needed. The causal boundary includes the query's
    own KV and excludes every later tail token.
    """
    if (
        len(prefix_lens) != len(extend_lens)
        or len(prefix_lens) != block_tables.shape[0]
    ):
        raise ValueError("SR tail request/length/page-table row mismatch")
    rows, contexts = [], []
    for row, (prefix, length) in enumerate(zip(prefix_lens, extend_lens)):
        if prefix < 0 or length <= 0:
            raise ValueError("SR tail requires nonnegative prefix and nonempty tail")
        rows.extend([row] * length)
        contexts.extend(range(prefix + 1, prefix + length + 1))
    indices = torch.tensor(rows, dtype=torch.int64, device=block_tables.device)
    return SRTailAttentionMetadata(
        block_tables.index_select(0, indices).to(dtype=torch.int32).contiguous(),
        torch.tensor(contexts, dtype=torch.int32),
        contexts,
    )


def tail_graph_max_pages(max_kv, page_size) -> int:
    """Pages needed to cover ``max_kv`` tokens, at least 1."""
    page_size = max(int(page_size or 1), 1)
    max_kv = max(int(max_kv or 0), 0)
    return max((max_kv + page_size - 1) // page_size, 1)


def tail_graph_fits_pages(max_seq, captured_pages, page_size) -> bool:
    return tail_graph_max_pages(max_seq, page_size) <= int(captured_pages or 0)


def widen_tail_block_tables(tables, max_pages):
    """Right-pad page columns with zeros so capture/replay share one width."""
    max_pages = int(max_pages)
    if max_pages < 1:
        raise ValueError("SR tail graph page table width must be positive")
    if tables.shape[1] >= max_pages:
        return tables
    pad = torch.zeros(
        (tables.shape[0], max_pages - tables.shape[1]),
        dtype=tables.dtype,
        device=tables.device,
    )
    return torch.cat([tables, pad], dim=1)


def copy_tail_attention_metadata_(dst, src):
    """Copy ``src`` into captured ``dst`` tensors. Graph replay needs the same storage."""
    if src.block_tables.shape[0] != dst.block_tables.shape[0]:
        raise ValueError("SR tail graph query row mismatch")
    if src.block_tables.shape[1] > dst.block_tables.shape[1]:
        raise ValueError("SR tail graph page table is too narrow")
    if src.context_lens_cpu.numel() != dst.context_lens_cpu.numel():
        raise ValueError("SR tail graph context length row mismatch")
    if len(src.context_lens_list) != len(dst.context_lens_list):
        raise ValueError("SR tail graph context length list mismatch")
    dst.block_tables.zero_()
    n_pages = src.block_tables.shape[1]
    dst.block_tables[:, :n_pages].copy_(src.block_tables)
    dst.context_lens_cpu.copy_(
        src.context_lens_cpu.to(
            dtype=dst.context_lens_cpu.dtype, device=dst.context_lens_cpu.device
        )
    )
    dst.context_lens_list[:] = list(src.context_lens_list)


def fill_tail_attention_metadata_(dst, prefix_lens, extend_lens, req_tables, dummy_slot=0):
    """Write query rows into captured ``dst`` without allocating a padded copy."""
    if dummy_slot != 0:
        raise ValueError("SR tail graph dummy queries must use reserved slot 0")
    if (
        len(prefix_lens) != len(extend_lens)
        or len(prefix_lens) != req_tables.shape[0]
    ):
        raise ValueError("SR tail request/length/page-table row mismatch")
    token_cap = int(dst.block_tables.shape[0])
    max_pages = int(dst.block_tables.shape[1])
    n_real = 0
    for prefix, length in zip(prefix_lens, extend_lens):
        if int(prefix) < 0 or int(length) <= 0:
            raise ValueError("SR tail requires nonnegative prefix and nonempty tail")
        n_real += int(length)
    if n_real > token_cap:
        raise ValueError("SR tail exceeds graph token capacity")
    src_pages = int(req_tables.shape[1])
    if src_pages > max_pages:
        raise ValueError("SR tail graph page table is too narrow")
    if dst.context_lens_cpu.numel() != token_cap:
        raise ValueError("SR tail graph context length row mismatch")
    if len(dst.context_lens_list) != token_cap:
        raise ValueError("SR tail graph context length list mismatch")
    dst.block_tables.zero_()
    dst.context_lens_cpu.fill_(1)
    offset = 0
    n_copy = src_pages
    for row, (prefix, length) in enumerate(zip(prefix_lens, extend_lens)):
        length = int(length)
        prefix = int(prefix)
        if n_copy:
            dst.block_tables[offset : offset + length, :n_copy].copy_(
                req_tables[row, :n_copy].unsqueeze(0).expand(length, -1)
            )
        for i in range(length):
            ctx = prefix + i + 1
            dst.context_lens_cpu[offset + i] = ctx
            dst.context_lens_list[offset + i] = ctx
        offset += length
    for i in range(offset, token_cap):
        dst.context_lens_list[i] = 1
    return dst


def pad_tail_attention_metadata(metadata, token_cap, dummy_slot=0):
    """Append dummy query rows that only see ``dummy_slot``.

    Dummy rows are packed after real queries. They do not share a real
    request's prefix pages, and real queries never include them in context.
    """
    if dummy_slot != 0:
        raise ValueError("SR tail graph dummy queries must use reserved slot 0")
    n_real = len(metadata.context_lens_list)
    if n_real > token_cap:
        raise ValueError("SR tail exceeds graph token capacity")
    if n_real == token_cap:
        return metadata
    pad = token_cap - n_real
    dummy_tables = torch.zeros(
        (pad, metadata.block_tables.shape[1]),
        dtype=metadata.block_tables.dtype,
        device=metadata.block_tables.device,
    )
    dummy_ctx = torch.ones(pad, dtype=metadata.context_lens_cpu.dtype)
    return SRTailAttentionMetadata(
        torch.cat([metadata.block_tables, dummy_tables], dim=0),
        torch.cat([metadata.context_lens_cpu, dummy_ctx], dim=0),
        list(metadata.context_lens_list) + [1] * pad,
    )


def validate_tail_forward_batch(batch):
    """Check the packed contract without synchronizing device tensors."""
    prefix, lengths = batch.extend_prefix_lens_cpu, batch.extend_seq_lens_cpu
    if len(prefix) != batch.batch_size or len(lengths) != batch.batch_size:
        raise ValueError("SR tail batch/request length mismatch")
    if any(p < 0 or n <= 0 for p, n in zip(prefix, lengths)):
        raise ValueError("invalid SR tail lengths")
    count = sum(lengths)
    if count != batch.extend_num_tokens:
        raise ValueError("SR tail packed token count mismatch")
    for name in ("input_ids", "positions", "out_cache_loc"):
        value = getattr(batch, name, None)
        if value is None or value.ndim != 1 or value.numel() != count:
            raise ValueError(f"SR tail {name} must have {count} rows")
    if batch.mrope_positions is not None and batch.mrope_positions.shape != (3, count):
        raise ValueError("SR tail M-RoPE width mismatch")
    if batch.seq_lens_cpu.tolist() != [p + n for p, n in zip(prefix, lengths)]:
        raise ValueError("SR tail total lengths do not match prefix + tail")
