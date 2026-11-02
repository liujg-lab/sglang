"""Build a verify tree from OPT parent pointers.

``parents[i] == -1`` is the committed tail, which occupies verify slot 0 and is
not part of ``draft_tokens``. Every other parent is an earlier index in the
draft list (``parents[i] < i``). The flattened mask is FULL_MASK: True means
the query may attend. Prefix columns stay visible. Draft columns follow the
ancestor chain, including self.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch

from sglang.srt.speculative.tree_attn_mask import assert_full_mask_layout


@dataclass
class OptVerifyLayout:
    tree_mask: torch.Tensor
    positions: torch.Tensor
    retrive_index: torch.Tensor
    retrive_next_token: torch.Tensor
    retrive_next_sibling: torch.Tensor
    num_verify_tokens: int


def _as_long_2d(parent_index: torch.Tensor) -> torch.Tensor:
    if parent_index.dim() == 1:
        parent_index = parent_index.unsqueeze(0)
    if parent_index.dim() != 2:
        raise ValueError(
            f"parent_index must be [num_draft] or [bs, num_draft], got {tuple(parent_index.shape)}"
        )
    return parent_index.long()


def _seq_list(seq_lens: torch.Tensor, batch: int) -> List[int]:
    if seq_lens.dim() != 1 or int(seq_lens.shape[0]) != batch:
        raise ValueError(
            f"seq_lens shape {tuple(seq_lens.shape)} does not match batch {batch}"
        )
    values = [int(x) for x in seq_lens.detach().reshape(-1).tolist()]
    if any(v < 0 for v in values):
        raise ValueError(f"seq_lens must be >= 0, got {values}")
    return values


def _validate_parents(row: Sequence[int]) -> List[int]:
    """Return hops from the root for each draft node. Root hop count is 0."""
    hops: List[int] = []
    for i, parent in enumerate(row):
        if parent != -1 and not (0 <= parent < i):
            raise ValueError(
                f"parents[{i}]={parent} must be -1 or an earlier draft index"
            )
        if parent == -1:
            hops.append(0)
            continue
        hops.append(hops[parent] + 1)
    return hops


def pack_verify_draft_token(
    roots: Sequence[int],
    token_rows: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Flat verify tokens: committed tail, then the width-N draft nodes.

    ``prepare_for_verify`` allocates ``len(input_ids)`` KV slots. A 2-D
    ``[bs, width + 1]`` tensor has ``len == bs``, so tree verify asks for
    ``bs * (width + 1)`` slots and rejects the short allocation.
    """
    if len(roots) != len(token_rows):
        raise ValueError(
            f"roots and token_rows differ: {len(roots)} vs {len(token_rows)}"
        )
    if not roots:
        return torch.empty((0,), dtype=torch.long)
    rows = [
        [int(root), *[int(token) for token in row]]
        for root, row in zip(roots, token_rows)
    ]
    width_plus = len(rows[0])
    for row in rows[1:]:
        if len(row) != width_plus:
            raise ValueError(
                f"verify rows must share one width, got {width_plus} and {len(row)}"
            )
    return torch.tensor(rows, dtype=torch.long).reshape(-1).contiguous()


def build_tree_from_parents(
    parent_index: torch.Tensor,
    seq_lens: torch.Tensor,
) -> OptVerifyLayout:
    """FULL_MASK and retrive links for root plus the OPT draft nodes.

    ``num_verify_tokens`` is ``num_draft + 1``. Slot 0 of each row is the root.
    A draft node at hop ``h`` from the root (``h == 0`` when its parent is -1)
    is placed at ``seq_len + h + 1``. The longest OPT chain therefore occupies
    ``max_depth - 1`` draft positions when the newest layer was dropped.
    """
    parents = _as_long_2d(parent_index)
    batch, num_draft = int(parents.shape[0]), int(parents.shape[1])
    if num_draft < 1:
        raise ValueError("parent_index needs at least one draft node")
    seqs = _seq_list(seq_lens, batch)
    num_verify = num_draft + 1
    device = parents.device

    hops_batch = [_validate_parents([int(x) for x in row.tolist()]) for row in parents]

    mask_rows: List[bool] = []
    positions: List[int] = []
    retrive_index = torch.empty(batch, num_verify, dtype=torch.long, device=device)
    retrive_next_token = torch.full(
        (batch, num_verify), -1, dtype=torch.long, device=device
    )
    retrive_next_sibling = torch.full(
        (batch, num_verify), -1, dtype=torch.long, device=device
    )

    for b in range(batch):
        seq_len = seqs[b]
        hops = hops_batch[b]
        row_parents = [int(x) for x in parents[b].tolist()]
        verify_parent = [-1] + [0 if p < 0 else p + 1 for p in row_parents]
        children: List[List[int]] = [[] for _ in range(num_verify)]
        for draft_i, parent in enumerate(row_parents):
            verify_i = draft_i + 1
            parent_verify = 0 if parent < 0 else parent + 1
            children[parent_verify].append(verify_i)
        for parent_i, kids in enumerate(children):
            if not kids:
                continue
            retrive_next_token[b, parent_i] = kids[0]
            for left, right in zip(kids, kids[1:]):
                retrive_next_sibling[b, left] = right

        for t in range(num_verify):
            retrive_index[b, t] = b * num_verify + t
            row = [True] * seq_len + [False] * num_verify
            node = t
            while node >= 0:
                row[seq_len + node] = True
                node = verify_parent[node]
            mask_rows.extend(row)
            if t == 0:
                positions.append(seq_len)
            else:
                positions.append(seq_len + hops[t - 1] + 1)

    tree_mask = torch.tensor(mask_rows, dtype=torch.bool, device=device)
    assert_full_mask_layout(tree_mask, seqs, num_verify, where="opt tree parents")
    return OptVerifyLayout(
        tree_mask=tree_mask,
        positions=torch.tensor(positions, dtype=torch.long, device=device),
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        num_verify_tokens=num_verify,
    )
