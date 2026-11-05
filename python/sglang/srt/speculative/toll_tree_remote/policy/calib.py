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


def _find_child(tree, parent: int, token: int):
    """Tree index of ``token`` under ``parent``. A side-buffer candidate is absent."""
    parents = tree.parent
    tokens = tree.token
    for index in range(len(tree)):
        if int(parents[index]) == int(parent) and int(tokens[index]) == int(token):
            return index
    return None


def _next_token_by_parent(tree, committed_tokens) -> dict[int, int]:
    """Next committed token for each parent reached by the accepted chain."""
    next_token: dict[int, int] = {}
    parent = -1
    for token in committed_tokens:
        next_token[parent] = int(token)
        found = _find_child(tree, parent, int(token))
        if found is None:
            break
        parent = found
    return next_token


def _candidate_columns(tree):
    """Tree nodes followed by below-cut candidates."""
    return (
        np.concatenate([tree.parent, tree.rejected_parent]),
        np.concatenate([tree.token, tree.rejected_token]),
        np.concatenate([tree.depth, tree.rejected_depth]),
        np.concatenate([tree.raw_p, tree.rejected_raw_p]),
    )


class Calibrator:
    """EMA scale keyed by (depth, log bucket of the raw probability).

    A candidate is labeled only when its parent has a known next token.
    Children of the last committed node are censored, and a candidate that
    missed the cut is labeled by token identity, not by whether it entered
    the tree. Fewer than ``min_samples`` updates leave the scale at 1. After
    that the scale shrinks toward 1 and is clipped. ``map`` always reads the
    raw probabilities, so calling it again does not multiply the scale.
    """

    def __init__(
        self,
        alpha: float = 0.02,
        edges=None,
        min_samples: int = 8,
        prior_count: float = 8.0,
        scale_min: float = 0.05,
        scale_max: float = 20.0,
    ) -> None:
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if int(min_samples) < 1:
            raise ValueError("min_samples must be >= 1")
        if float(prior_count) < 0.0:
            raise ValueError("prior_count must be >= 0")
        if float(scale_min) < 0.0 or float(scale_min) > float(scale_max):
            raise ValueError("scale bounds must satisfy 0 <= scale_min <= scale_max")
        knots = _DEFAULT_EDGES if edges is None else np.asarray(edges, dtype=np.float64)
        knots = np.unique(knots.reshape(-1))
        if knots.size < 2:
            raise ValueError("calibration edges need at least 2 knots")
        self.alpha = float(alpha)
        self.edges = knots
        self.min_samples = int(min_samples)
        self.prior_count = float(prior_count)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self._slots: dict[tuple[int, int], _Slot] = {}
        self.labeled = 0
        self.censored = 0

    def bucket(self, raw_p: float) -> int:
        return int(self._buckets(np.asarray([raw_p], dtype=np.float64))[0])

    def _buckets(self, raw: np.ndarray) -> np.ndarray:
        clipped = np.clip(raw, float(self.edges[0]), float(self.edges[-1]))
        idx = np.searchsorted(self.edges, clipped, side="right") - 1
        return np.clip(idx, 0, int(self.edges.size) - 2).astype(np.int64)

    def _scale_value(self, slot: _Slot | None) -> float:
        if slot is None or slot.count < self.min_samples or slot.ema_p <= 0.0:
            return 1.0
        raw_scale = slot.ema_y / slot.ema_p
        weight = float(slot.count) + self.prior_count
        shrunk = (float(slot.count) * raw_scale + self.prior_count) / weight
        return float(np.clip(shrunk, self.scale_min, self.scale_max))

    def _scales_for(self, raw: np.ndarray, depth: int) -> np.ndarray:
        buckets = self._buckets(raw)
        scales = np.ones(raw.size, dtype=np.float64)
        for bucket in np.unique(buckets):
            slot = self._slots.get((int(depth), int(bucket)))
            scales[buckets == bucket] = self._scale_value(slot)
        return scales

    def map(self, draft_probs, depth: int) -> np.ndarray:
        """Return ``a`` for one sibling group. The entries sum to at most 1."""
        return self.map_batch([draft_probs], [depth])[0]

    def map_batch(self, groups, depths) -> list[np.ndarray]:
        """Map many sibling groups. Depths may differ; each group is capped at 1."""
        pieces = [np.asarray(group, dtype=np.float64).reshape(-1) for group in groups]
        if not pieces:
            return []
        raw = np.concatenate(pieces)
        depth_arr = np.concatenate(
            [
                np.full(piece.size, int(depth), dtype=np.int64)
                for piece, depth in zip(pieces, depths)
            ]
        )
        buckets = self._buckets(raw)
        scales = np.ones(raw.size, dtype=np.float64)
        for depth in np.unique(depth_arr):
            chosen = depth_arr == depth
            for bucket in np.unique(buckets[chosen]):
                slot = self._slots.get((int(depth), int(bucket)))
                scales[chosen & (buckets == bucket)] = self._scale_value(slot)
        scaled = np.clip(raw * scales, 0.0, 1.0)
        out = []
        cursor = 0
        for piece in pieces:
            chunk = scaled[cursor : cursor + piece.size].copy()
            total = float(chunk.sum())
            if total > 1.0:
                chunk /= total
            out.append(chunk)
            cursor += piece.size
        return out

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

    def _observe_tree(self, tree, next_token: dict[int, int]) -> None:
        if len(tree) == 0 and int(tree.rejected_token.size) == 0:
            return
        parent, token, depth, raw = _candidate_columns(tree)
        known = np.zeros(parent.size, dtype=bool)
        labels = np.zeros(parent.size, dtype=np.float64)
        for host, expected in next_token.items():
            mask = parent == int(host)
            known |= mask
            labels[mask & (token == int(expected))] = 1.0
        self.censored += int(np.count_nonzero(~known))
        if not np.any(known):
            return
        self.labeled += int(np.count_nonzero(known))
        depths = depth[known]
        raws = raw[known]
        ys = labels[known]
        for index in range(int(raws.size)):
            self._update(int(depths[index]), float(raws[index]), float(ys[index]))

    def observe(self, tree, committed_tokens) -> None:
        self.observe_batch([(tree, committed_tokens)])

    def observe_batch(self, items) -> None:
        """Label many trees. The result matches calling ``observe`` on each one."""
        for tree, committed_tokens in items:
            self._observe_tree(tree, _next_token_by_parent(tree, committed_tokens))
