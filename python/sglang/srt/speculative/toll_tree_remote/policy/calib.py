"""Map draft probabilities to acceptance mass with censored labels."""

from __future__ import annotations

import numpy as np


def _accepted_indices(tree, committed_tokens) -> set[int]:
    """Walk the committed token chain. The tail (parent -1) is already accepted."""
    accepted: list[int] = []
    parent = -1
    for token in committed_tokens:
        found = None
        for index, node in enumerate(tree.nodes):
            if node.parent == parent and int(node.token) == int(token):
                found = index
                break
        if found is None:
            break
        accepted.append(found)
        parent = found
    return set(accepted)


class Calibrator:
    """Per-depth scale from labeled children, then a sibling cap at 1.

    Only children of an accepted node are labeled. The committed tail counts
    as accepted, so the first draft layer is labeled. Other nodes are censored.
    """

    def __init__(self) -> None:
        self._p_sum: dict[int, float] = {}
        self._y_sum: dict[int, float] = {}
        self.labeled = 0
        self.censored = 0

    def _scale(self, depth: int) -> float:
        mass = self._p_sum.get(int(depth), 0.0)
        if mass <= 0.0:
            return 1.0
        return float(np.clip(self._y_sum.get(int(depth), 0.0) / mass, 0.0, 1.0))

    def map(self, draft_probs, depth: int) -> np.ndarray:
        """Return ``a`` for one sibling group. The entries sum to at most 1."""
        raw = np.asarray(draft_probs, dtype=np.float64).reshape(-1)
        scaled = np.clip(raw * self._scale(int(depth)), 0.0, 1.0)
        total = float(scaled.sum())
        if total > 1.0:
            scaled = scaled / total
        return scaled

    def observe(self, tree, committed_tokens) -> None:
        accepted = _accepted_indices(tree, committed_tokens)
        for index, node in enumerate(tree.nodes):
            parent_ok = node.parent == -1 or node.parent in accepted
            if not parent_ok:
                self.censored += 1
                continue
            label = 1.0 if index in accepted else 0.0
            depth = int(node.depth)
            self._p_sum[depth] = self._p_sum.get(depth, 0.0) + float(node.draft_a)
            self._y_sum[depth] = self._y_sum.get(depth, 0.0) + label
            self.labeled += 1
