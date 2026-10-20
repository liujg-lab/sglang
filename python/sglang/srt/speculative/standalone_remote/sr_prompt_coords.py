"""Logical (Target) vs physical (Draft KV) coordinates for prompt compaction.

``origin_input_ids`` and ``fill_ids`` stay in the Draft's physical token
space. ``sr_padded_ids`` stays the Target prompt. ``sr_prompt_delta`` is the
number of prompt tokens removed. When the delta is 0 and no local prompt is
stored, every helper is the identity used by the uncompacted path.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


class SRPromptCoordError(ValueError):
    """A logical index cannot be mapped onto the compacted prompt."""


def prompt_delta(req) -> int:
    return int(getattr(req, "sr_prompt_delta", 0) or 0)


def logical_prompt_ids(req) -> List[int]:
    padded = getattr(req, "sr_padded_ids", None)
    if padded is not None:
        return list(padded)
    return list(getattr(req, "origin_input_ids", None) or [])


def incoming_padded_ids(dreq) -> List[int]:
    """Target prompt on a snapshot. Empty when this request is a delta."""
    incoming = getattr(dreq, "padded_input_ids", None)
    if not incoming:
        return []
    return [int(x) for x in incoming]


def align_target_ids(req, dreq) -> List[int]:
    """Logical sequence Target claims: incoming prompt when present, else stored."""
    prompt = incoming_padded_ids(dreq)
    if not prompt:
        prompt = list(
            getattr(req, "sr_padded_ids", None)
            or getattr(req, "origin_input_ids", None)
            or []
        )
    committed = getattr(dreq, "committed_ids", None) or []
    return list(prompt) + [int(x) for x in committed]


def prompt_replaced(req, dreq) -> bool:
    """True when a snapshot carries a different prompt than the one Draft stored."""
    incoming = incoming_padded_ids(dreq)
    if not incoming:
        return False
    return incoming != logical_prompt_ids(req)


def logical_prompt_len(req) -> int:
    return len(logical_prompt_ids(req))


def local_prompt_len(req) -> int:
    local = getattr(req, "sr_local_prompt_ids", None)
    if local is not None:
        return len(local)
    return logical_prompt_len(req) - prompt_delta(req)


def folded_ids(req) -> List[int]:
    return list(getattr(req, "origin_input_ids", None) or []) + list(
        getattr(req, "output_ids", None) or []
    )


def physical_len(req) -> int:
    return len(folded_ids(req))


def logical_len(req) -> int:
    return physical_len(req) + prompt_delta(req)


def logical_ids(req) -> List[int]:
    """Target-coordinate ids. Equals ``origin + output`` when delta is 0."""
    folded = folded_ids(req)
    local_n = local_prompt_len(req)
    if local_n < 0 or local_n > len(folded):
        raise SRPromptCoordError(
            "physical prompt is longer than the folded sequence"
        )
    return logical_prompt_ids(req) + folded[local_n:]


def to_local(req, n: int) -> int:
    """Map a post-prompt logical index onto a physical KV index.

    Dropped vision tokens sit inside the prompt, so the shift is constant only
    at and after the logical prompt. A fork inside the prompt means the prompt
    itself changed and the caller must reprefill.
    """
    index = int(n)
    if index < logical_prompt_len(req):
        raise SRPromptCoordError("index falls inside the logical prompt")
    return index - prompt_delta(req)


def logical_prefix_floor(req, physical_prefix_len: int) -> int:
    """Radix-prefix floor expressed in the logical sequence."""
    return int(physical_prefix_len) + prompt_delta(req)


def physical_ids_for_target(req, logical_target: Sequence[int]) -> List[int]:
    """Rebuild a physical fill from a logical target that shares this prompt."""
    target = [int(x) for x in logical_target]
    if prompt_delta(req) == 0 and getattr(req, "sr_local_prompt_ids", None) is None:
        return target
    prompt = logical_prompt_ids(req)
    prompt_n = len(prompt)
    if target[:prompt_n] != prompt:
        raise SRPromptCoordError("logical target prompt mismatch")
    local = list(getattr(req, "sr_local_prompt_ids") or [])
    return local + target[prompt_n:]


def lease_base_pair(req, kv_base: int) -> Tuple[int, int]:
    """Return ``(logical_base, physical_kv)`` for a tree KV lease.

    The shift applies only once KV covers the compacted prompt. Otherwise both
    values stay on the physical length so a partial prefix is not relabeled.
    """
    physical = int(kv_base)
    if physical >= local_prompt_len(req):
        return physical + prompt_delta(req), physical
    return physical, physical


def restore_compacted_mrope(req) -> bool:
    """Gather cached full-sequence M-RoPE onto the current physical ids.

    The stock rope index reads ``grid_thw`` and would compress positions if it
    ran on the shortened prompt. Returns False when the cache is missing.
    """
    if prompt_delta(req) <= 0:
        return True
    full = getattr(req, "sr_logical_mrope", None)
    keep = getattr(req, "sr_prompt_keep_index", None)
    mm = getattr(req, "multimodal_inputs", None)
    if full is None or keep is None or mm is None:
        return False
    if not torch.is_tensor(full) or full.ndim != 2 or full.shape[0] != 3:
        return False
    folded = folded_ids(req)
    local_n = local_prompt_len(req)
    prompt_n = logical_prompt_len(req)
    if local_n > len(folded) or prompt_n > full.shape[1]:
        return False
    keep_index = [int(i) for i in keep]
    if len(keep_index) != local_n:
        return False
    if any(i < 0 or i >= prompt_n for i in keep_index):
        return False
    tail_len = len(folded) - local_n
    positions = _gather_mrope(full, keep_index, prompt_n, tail_len)
    if positions.shape[1] != len(folded):
        return False
    # Decode and tail-extend both continue from this delta. Dropped vision
    # columns can hold an axis maximum, so the full table's max must win over
    # the gathered subset. A tail synthesized past the cache can be larger.
    full_max = int(full.max().item()) if full.numel() else -1
    got_max = int(positions.max().item()) if positions.numel() else -1
    max_pos = max(full_max, got_max)
    mm.mrope_positions = positions
    mm.mrope_position_delta = torch.tensor(
        [[max_pos + 1 - positions.shape[1]]], dtype=torch.int64
    )
    mm.mrope_position_delta_repeated_cache = None
    return True


def _gather_mrope(
    full: torch.Tensor,
    keep_index: Sequence[int],
    logical_prompt_n: int,
    tail_len: int,
) -> torch.Tensor:
    keep = torch.as_tensor(list(keep_index), dtype=torch.long)
    prompt = full[:, keep]
    cached_tail = full[:, int(logical_prompt_n) :]
    if tail_len <= cached_tail.shape[1]:
        tail = cached_tail[:, :tail_len]
    else:
        if cached_tail.shape[1] > 0:
            start = int(cached_tail[:, -1].max().item()) + 1
            have = cached_tail
            need = tail_len - cached_tail.shape[1]
        else:
            start = int(full.max().item()) + 1 if full.numel() else 0
            have = cached_tail
            need = tail_len
        extra = (
            torch.arange(start, start + need, dtype=full.dtype, device=full.device)
            .view(1, -1)
            .expand(3, -1)
        )
        tail = torch.cat([have, extra], dim=1)
    if prompt.numel() == 0:
        return tail
    if tail.numel() == 0:
        return prompt
    return torch.cat([prompt, tail], dim=1)
