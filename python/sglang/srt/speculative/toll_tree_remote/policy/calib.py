"""Map raw draft probabilities to acceptance mass with censored labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Log buckets of the raw probability. The scale may exceed 1; ``map`` then
# clips each ``a`` into [0, 1] and caps a sibling group at 1.
_DEFAULT_EDGES = np.geomspace(1e-4, 1.0, 9)


@dataclass
class _Slot:
    ema_y: float
    ema_p: float
    count: int


def _accepted_indices(tree, committed_tokens) -> set[int]:
    """Walk the committed token chain. The tail (parent -1) is already accepted."""
    accepted: list[int] = []
    parent = -1
    for token in committed_tokens:
        found = None
        for index in range(len(tree)):
            if int(tree.parent[index]) == parent and int(tree.token[index]) == int(token):
                found = index
                break
        if found is None:
            break
        accepted.append(found)
        parent = found
    return set(accepted)


class Calibrator:
    """EMA scale keyed by (depth, log bucket of the raw probability).

    Only children of an accepted node are labeled. The committed tail counts
    as accepted, so the first draft layer is labeled. Candidates that missed
    the cut stay in the tree's side buffer and are labeled the same way.
    ``map`` always reads the raw probabilities, so calling it again does not
    multiply the scale a second time.
    """

    def __init__(self, alpha: float = 0.02, edges=None) -> None:
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        knots = _DEFAULT_EDGES if edges is None else np.asarray(edges, dtype=np.float64)
        knots = np.unique(knots.reshape(-1))
        if knots.size < 2:
            raise ValueError("calibration edges need at least 2 knots")
        self.alpha = float(alpha)
        self.edges = knots
        self._slots: dict[tuple[int, int], _Slot] = {}
        self.labeled = 0
        self.censored = 0

    def bucket(self, raw_p: float) -> int:
        clipped = float(np.clip(raw_p, float(self.edges[0]), float(self.edges[-1])))
        idx = int(np.searchsorted(self.edges, clipped, side="right") - 1)
        return int(np.clip(idx, 0, int(self.edges.size) - 2))

    def _scale(self, depth: int, raw_p: float) -> float:
        slot = self._slots.get((int(depth), self.bucket(raw_p)))
        if slot is None or slot.ema_p <= 0.0:
            return 1.0
        return float(slot.ema_y / slot.ema_p)

    def map(self, draft_probs, depth: int) -> np.ndarray:
        """Return ``a`` for one sibling group. The entries sum to at most 1."""
        raw = np.asarray(draft_probs, dtype=np.float64).reshape(-1)
        scales = np.array(
            [self._scale(int(depth), float(p)) for p in raw],
            dtype=np.float64,
        )
        scaled = np.clip(raw * scales, 0.0, 1.0)
        total = float(scaled.sum())
        if total > 1.0:
            scaled = scaled / total
        return scaled

    def _update(self, depth: int, raw_p: float, label: float) -> None:
        key = (int(depth), self.bucket(raw_p))
        slot = self._slots.get(key)
        if slot is None:
            self._slots[key] = _Slot(ema_y=float(label), ema_p=float(raw_p), count=1)
            return
        keep = 1.0 - self.alpha
        slot.ema_y = keep * slot.ema_y + self.alpha * float(label)
        slot.ema_p = keep * slot.ema_p + self.alpha * float(raw_p)
        slot.count += 1

    def observe(self, tree, committed_tokens) -> None:
        accepted = _accepted_indices(tree, committed_tokens)
        for parent, _token, depth, raw_p, index in tree.calibration_rows():
            parent_ok = parent == -1 or parent in accepted
            if not parent_ok:
                self.censored += 1
                continue
            label = 1.0 if index is not None and index in accepted else 0.0
            self._update(depth, raw_p, label)
            self.labeled += 1
