"""Draft-layer attention mask in physical slots."""

from __future__ import annotations

import numpy as np


def layer_mask(prefix_len: int, tree, forward, width: int, slot_of_node) -> np.ndarray:
    """Mask for one physical layer.

    Columns are the prefix, every earlier physical slot (including padding),
    and this layer's ``width`` slots. A real row sees the prefix, its ancestor
    slots, and itself. A padding row sees the prefix and itself.
    """
    kept = [int(node) for node in forward]
    slots = {int(node): int(slot) for node, slot in dict(slot_of_node).items()}
    if kept:
        block = min(slots[node] for node in kept)
    else:
        block = 0
    columns = int(prefix_len) + block + int(width)
    mask = np.zeros((int(width), columns), dtype=bool)
    if prefix_len:
        mask[:, : int(prefix_len)] = True
    local_of = {}
    for node in kept:
        local_of[slots[node] - block] = node
    for local in range(int(width)):
        mask[local, int(prefix_len) + block + local] = True
        node = local_of.get(local)
        if node is None:
            continue
        parent = int(tree.parent[node])
        while parent != -1:
            mask[local, int(prefix_len) + slots[parent]] = True
            parent = int(tree.parent[parent])
    return mask
