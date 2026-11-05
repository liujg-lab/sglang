"""Variable-width TollTree stored as numpy columns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TollNode:
    token: int
    pi: float
    parent: int
    depth: int
    raw_p: float
    a: float


@dataclass(frozen=True)
class TollVerify:
    """Ancestor-closed prefix of the path-mass ranking.

    ``parents[i] == -1`` is the committed tail. Every other parent index is
    strictly smaller than ``i``.
    """

    tokens: tuple[int, ...]
    parents: tuple[int, ...]
    pis: tuple[float, ...]
    depths: tuple[int, ...]


class TollTree:
    """One request's draft nodes.

    Columns are token, path mass, parent, depth, raw probability, and the
    calibrated conditional ``a``. Path mass uses ``a``, so a child cannot
    exceed its parent while ``a`` stays in [0, 1]. Candidates that miss the
    cut stay in the side buffer and are not ranked by ``finish``.
    """

    def __init__(self, prefix_len: int) -> None:
        if int(prefix_len) < 0:
            raise ValueError(f"prefix_len must be >= 0, got {prefix_len}")
        self.prefix_len = int(prefix_len)
        self.token = np.zeros(0, dtype=np.int64)
        self.pi = np.zeros(0, dtype=np.float64)
        self.parent = np.zeros(0, dtype=np.int64)
        self.depth = np.zeros(0, dtype=np.int64)
        self.raw_p = np.zeros(0, dtype=np.float64)
        self.a = np.zeros(0, dtype=np.float64)
        self.frontier = np.zeros(0, dtype=np.int64)
        self.rejected_token = np.zeros(0, dtype=np.int64)
        self.rejected_parent = np.zeros(0, dtype=np.int64)
        self.rejected_depth = np.zeros(0, dtype=np.int64)
        self.rejected_raw_p = np.zeros(0, dtype=np.float64)
        self.rejected_a = np.zeros(0, dtype=np.float64)
        self.exited = False

    def __len__(self) -> int:
        return int(self.token.size)

    def mark_exited(self) -> None:
        self.exited = True
        self.frontier = np.zeros(0, dtype=np.int64)

    def add_child(self, parent: int, token: int, raw_p: float, a: float) -> int:
        if self.exited:
            raise RuntimeError("exited request cannot gain nodes")
        self._check_prob(raw_p, a)
        pi, depth = self._path(int(parent), float(a))
        self._extend(
            np.array([token], dtype=np.int64),
            np.array([pi], dtype=np.float64),
            np.array([parent], dtype=np.int64),
            np.array([depth], dtype=np.int64),
            np.array([raw_p], dtype=np.float64),
            np.array([a], dtype=np.float64),
        )
        return len(self) - 1

    def reject(self, parent: int, token: int, raw_p: float, a: float) -> None:
        """Keep a candidate for calibration without putting it in the tree."""
        if self.exited:
            raise RuntimeError("exited request cannot gain nodes")
        self._check_prob(raw_p, a)
        _pi, depth = self._path(int(parent), float(a))
        self.rejected_token = np.concatenate(
            [self.rejected_token, np.array([token], dtype=np.int64)]
        )
        self.rejected_parent = np.concatenate(
            [self.rejected_parent, np.array([parent], dtype=np.int64)]
        )
        self.rejected_depth = np.concatenate(
            [self.rejected_depth, np.array([depth], dtype=np.int64)]
        )
        self.rejected_raw_p = np.concatenate(
            [self.rejected_raw_p, np.array([raw_p], dtype=np.float64)]
        )
        self.rejected_a = np.concatenate(
            [self.rejected_a, np.array([a], dtype=np.float64)]
        )

    def filter_candidates(self, parent, token, raw_p, a, cut: float, limit: int) -> np.ndarray:
        """Admit the top ``limit`` candidates with path mass above ``cut``.

        Everyone else is appended to the calibration buffer. Returns the new
        tree indices of the admitted nodes, highest path mass first.
        """
        if self.exited:
            raise RuntimeError("exited request cannot gain nodes")
        parents = np.asarray(parent, dtype=np.int64).reshape(-1)
        tokens = np.asarray(token, dtype=np.int64).reshape(-1)
        raw = np.asarray(raw_p, dtype=np.float64).reshape(-1)
        cond = np.asarray(a, dtype=np.float64).reshape(-1)
        if not (parents.size == tokens.size == raw.size == cond.size):
            raise ValueError("candidate columns must have the same length")
        if np.any(parents < -1):
            raise ValueError("candidate parent must be -1 or a tree index")
        if np.any((cond < 0.0) | (cond > 1.0)) or np.any(raw < 0.0):
            raise ValueError("conditional a must be in [0, 1] and raw_p >= 0")
        base = np.ones(cond.size, dtype=np.float64)
        linked = parents >= 0
        if np.any(linked):
            if np.any(parents[linked] >= len(self)):
                raise ValueError("candidate parent is outside the tree")
            base[linked] = self.pi[parents[linked]]
        path = base * cond
        depth = np.ones(cond.size, dtype=np.int64)
        if np.any(linked):
            depth[linked] = self.depth[parents[linked]] + 1
        keep = np.flatnonzero(path > float(cut))
        if keep.size > int(limit):
            order = np.lexsort((keep, -path[keep]))
            chosen = keep[order[: int(limit)]]
        elif keep.size:
            chosen = keep[np.argsort(-path[keep], kind="stable")]
        else:
            chosen = keep
        chosen_mask = np.zeros(cond.size, dtype=bool)
        chosen_mask[chosen] = True
        start = len(self)
        if chosen.size:
            self._extend(
                tokens[chosen],
                path[chosen],
                parents[chosen],
                depth[chosen],
                raw[chosen],
                cond[chosen],
            )
        dropped = np.flatnonzero(~chosen_mask)
        if dropped.size:
            self.rejected_token = np.concatenate([self.rejected_token, tokens[dropped]])
            self.rejected_parent = np.concatenate([self.rejected_parent, parents[dropped]])
            self.rejected_depth = np.concatenate([self.rejected_depth, depth[dropped]])
            self.rejected_raw_p = np.concatenate([self.rejected_raw_p, raw[dropped]])
            self.rejected_a = np.concatenate([self.rejected_a, cond[dropped]])
        return np.arange(start, start + int(chosen.size), dtype=np.int64)

    def set_frontier(self, indices) -> None:
        if self.exited:
            raise RuntimeError("exited request cannot gain nodes")
        self.frontier = np.asarray(list(indices), dtype=np.int64)
        for index in self.frontier.tolist():
            if int(index) < 0 or int(index) >= len(self):
                raise ValueError(f"frontier index {index} is outside the tree")

    def calibration_rows(self):
        """Tree nodes, then below-cut candidates. ``index`` is None off the tree."""
        for index in range(len(self)):
            yield (
                int(self.parent[index]),
                int(self.token[index]),
                int(self.depth[index]),
                float(self.raw_p[index]),
                index,
            )
        for index in range(int(self.rejected_token.size)):
            yield (
                int(self.rejected_parent[index]),
                int(self.rejected_token[index]),
                int(self.rejected_depth[index]),
                float(self.rejected_raw_p[index]),
                None,
            )

    def ranked_indices(self) -> np.ndarray:
        """Higher path mass first. Shallower depth breaks ties, then index."""
        n = len(self)
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        return np.lexsort((np.arange(n), self.depth, -self.pi))

    def finish(self, k: int) -> TollVerify:
        """Keep the first ``k`` ranked nodes. The prefix includes its ancestors."""
        if int(k) < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        chosen = [int(i) for i in self.ranked_indices()[: min(int(k), len(self))].tolist()]
        chosen_set = set(chosen)
        for index in list(chosen):
            parent = int(self.parent[index])
            while parent != -1:
                if parent not in chosen_set:
                    raise RuntimeError(
                        f"finish({k}) dropped ancestor {parent} of node {index}"
                    )
                parent = int(self.parent[parent])
        position = {index: i for i, index in enumerate(chosen)}
        tokens = []
        parents = []
        pis = []
        depths = []
        for index in chosen:
            tokens.append(int(self.token[index]))
            pis.append(float(self.pi[index]))
            depths.append(int(self.depth[index]))
            parent = int(self.parent[index])
            if parent == -1:
                parents.append(-1)
            else:
                parent_at = position[parent]
                if parent_at >= position[index]:
                    raise RuntimeError("parent is not before its child in finish order")
                parents.append(parent_at)
        return TollVerify(tuple(tokens), tuple(parents), tuple(pis), tuple(depths))

    def _check_prob(self, raw_p: float, a: float) -> None:
        cond = float(a)
        if cond < 0.0 or cond > 1.0:
            raise ValueError(f"conditional a must be in [0, 1], got {a}")
        if float(raw_p) < 0.0:
            raise ValueError(f"raw_p must be >= 0, got {raw_p}")

    def _path(self, parent: int, cond: float) -> tuple[float, int]:
        if parent == -1:
            return cond, 1
        if parent < 0 or parent >= len(self):
            raise ValueError(f"parent {parent} is outside the tree")
        host = float(self.pi[parent])
        pi = host * cond
        if pi > host + 1e-12:
            raise ValueError("path probability increased along an edge")
        return pi, int(self.depth[parent]) + 1

    def _extend(self, token, pi, parent, depth, raw_p, a) -> None:
        self.token = np.concatenate([self.token, np.asarray(token, dtype=np.int64)])
        self.pi = np.concatenate([self.pi, np.asarray(pi, dtype=np.float64)])
        self.parent = np.concatenate([self.parent, np.asarray(parent, dtype=np.int64)])
        self.depth = np.concatenate([self.depth, np.asarray(depth, dtype=np.int64)])
        self.raw_p = np.concatenate([self.raw_p, np.asarray(raw_p, dtype=np.float64)])
        self.a = np.concatenate([self.a, np.asarray(a, dtype=np.float64)])
