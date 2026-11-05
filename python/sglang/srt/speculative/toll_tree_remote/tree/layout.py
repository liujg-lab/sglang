"""Verify layout built with numpy. Padding stays out of the retrive chain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from sglang.srt.speculative.tree_attn_mask import assert_full_mask_layout


@dataclass(frozen=True)
class TollVerifyLayout:
    tree_mask: np.ndarray
    positions: np.ndarray
    retrive_index: np.ndarray
    retrive_next_token: np.ndarray
    retrive_next_sibling: np.ndarray
    num_verify_tokens: int


def _hops(row: list[int]) -> list[int]:
    hops = []
    for index, parent in enumerate(row):
        if parent != -1 and not (0 <= parent < index):
            raise ValueError(f"parents[{index}]={parent} must be -1 or earlier")
        hops.append(0 if parent < 0 else hops[parent] + 1)
    return hops


def build_verify_layout(parent_rows, seq_lens, num_verify: int) -> TollVerifyLayout:
    """FULL_MASK plus retrive links.

    ``parent_rows`` are draft parents. Slot 0 is the committed tail. Positions
    past ``len(row)`` are padding: they stay out of ``retrive_next_*``.
    """
    rows = [[int(parent) for parent in row] for row in parent_rows]
    seqs = [int(value) for value in seq_lens]
    if len(rows) != len(seqs):
        raise ValueError(f"parent rows {len(rows)} != seq_lens {len(seqs)}")
    width = int(num_verify)
    if width < 2:
        raise ValueError("num_verify must include the tail and one draft slot")
    batch = len(rows)
    mask_rows = []
    positions = []
    retrive_index = np.zeros((batch, width), dtype=np.int64)
    retrive_next_token = np.full((batch, width), -1, dtype=np.int64)
    retrive_next_sibling = np.full((batch, width), -1, dtype=np.int64)
    for batch_index, row in enumerate(rows):
        if len(row) > width - 1:
            raise ValueError(f"draft row {len(row)} does not fit in {width}")
        hops = _hops(row)
        seq_len = seqs[batch_index]
        verify_parent = [-1] * width
        children: list[list[int]] = [[] for _ in range(width)]
        for draft_index, parent in enumerate(row):
            slot = draft_index + 1
            host = 0 if parent < 0 else parent + 1
            verify_parent[slot] = host
            children[host].append(slot)
        for parent_index, kids in enumerate(children):
            if not kids:
                continue
            retrive_next_token[batch_index, parent_index] = kids[0]
            for left, right in zip(kids, kids[1:]):
                retrive_next_sibling[batch_index, left] = right
        for slot in range(width):
            retrive_index[batch_index, slot] = batch_index * width + slot
            line = np.zeros(seq_len + width, dtype=bool)
            line[:seq_len] = True
            node = slot
            while node >= 0:
                line[seq_len + node] = True
                node = verify_parent[node]
            mask_rows.append(line)
            if slot == 0:
                positions.append(seq_len)
            elif slot <= len(row):
                positions.append(seq_len + hops[slot - 1] + 1)
            else:
                positions.append(seq_len)
    tree_mask = np.concatenate(mask_rows) if mask_rows else np.zeros(0, dtype=bool)
    assert_full_mask_layout(
        torch.tensor(tree_mask), seqs, width, where="toll tree parents"
    )
    return TollVerifyLayout(
        tree_mask=tree_mask,
        positions=np.asarray(positions, dtype=np.int64),
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        num_verify_tokens=width,
    )
