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
