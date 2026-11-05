"""Fixed-width OPT selection in numpy. This module does not import OPT."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FixedVerify:
    tokens: tuple[int, ...]
    parents: tuple[int, ...]


def _topk_last(values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Largest ``k`` along the last axis. Ties keep the smaller index."""
    k = min(int(k), int(values.shape[-1]))
    order = np.argsort(-values, axis=-1, kind="stable")[..., :k]
    taken = np.take_along_axis(values, order, axis=-1)
    return taken.astype(np.float32, copy=False), order


class _FixedState:
    def __init__(self, width: int, max_depth: int) -> None:
        self.width = int(width)
        self.max_depth = int(max_depth)
        self.weight = np.zeros((max_depth, width), dtype=np.float32)
        self.tokens = np.zeros((max_depth, width), dtype=np.int64)
        self.parents = np.zeros((max_depth, width), dtype=np.int64)
        self.depth = 0
        self.prev_e = np.float32(0.0)

    def mass(self) -> np.float32:
        flat = self.weight[: self.depth].reshape(-1)
        scores, _ = _topk_last(flat, self.width)
        return np.float32(scores.sum())


def _seed(state: _FixedState, probs: np.ndarray) -> None:
    row = np.asarray(probs, dtype=np.float32).reshape(-1)
    scores, ids = _topk_last(row, state.width)
    _write_seed(state, scores, ids)


class FixedPolicy:
    """Same N / τ loop as OPT, one method per step."""

    def __init__(self, width: int, tau: float, max_depth: int) -> None:
        self.tau = float(tau)
        self.state = _FixedState(int(width), int(max_depth))

    def seed(self, probs) -> None:
        _seed(self.state, probs)

    def expand(self, probs) -> None:
        # Record the mass before this layer so the next should_grow sees the increment.
        self.state.prev_e = self.state.mass()
        _expand(self.state, probs)

    def seed_topk(self, scores, ids) -> None:
        """Seed from an already chosen top-N. ``scores`` and ``ids`` are ``[N]``."""
        _write_seed(self.state, scores, ids)

    def expand_topk(self, scores, ids) -> None:
        """Expand from already chosen child top-N. Both arrays are ``[N, N]``."""
        self.state.prev_e = self.state.mass()
        _write_expand(self.state, scores, ids)

    def should_grow(self) -> bool:
        return _should_grow(self.state, self.tau)

    def finish(self) -> FixedVerify:
        return _finish(self.state)


def _write_seed(state: _FixedState, scores, ids) -> None:
    taken = np.asarray(scores, dtype=np.float32).reshape(-1)
    tokens = np.asarray(ids, dtype=np.int64).reshape(-1)
    if taken.size < state.width or tokens.size < state.width:
        raise ValueError(
            f"seed top-k {int(taken.size)} is shorter than width {state.width}"
        )
    state.weight[0] = taken[: state.width]
    state.tokens[0] = tokens[: state.width]
    state.parents[0] = np.arange(state.width)
    state.depth = 1


def _write_expand(state: _FixedState, scores, ids) -> None:
    taken = np.asarray(scores, dtype=np.float32)
    tokens = np.asarray(ids, dtype=np.int64)
    if taken.shape != (state.width, state.width) or tokens.shape != taken.shape:
        raise ValueError(
            f"expand top-k shape {taken.shape} != ({state.width}, {state.width})"
        )
    path = taken * state.weight[state.depth - 1][:, None]
    flat_scores = path.reshape(-1)
    flat_ids = tokens.reshape(-1)
    top_scores, top_index = _topk_last(flat_scores, state.width)
    parent_cols = top_index // state.width
    state.parents[state.depth] = parent_cols
    state.weight[state.depth] = top_scores
    state.tokens[state.depth] = flat_ids[top_index]
    state.depth += 1


def _expand(state: _FixedState, probs: np.ndarray) -> None:
    matrix = np.asarray(probs, dtype=np.float32)
    if matrix.shape[0] != state.width:
        raise ValueError(
            f"fixed expand rows {matrix.shape[0]} != width {state.width}"
        )
    child_scores, child_ids = _topk_last(matrix, state.width)
    _write_expand(state, child_scores, child_ids)


def _should_grow(state: _FixedState, tau: float) -> bool:
    grow = (state.mass() - state.prev_e) > np.float32(tau)
    return bool(grow) and state.depth < state.max_depth - 1


def _ranked(state: _FixedState) -> np.ndarray:
    """Top-N flat indices: score desc, depth asc, column asc."""
    width = state.width
    total = state.depth * width
    scores = state.weight[: state.depth].reshape(-1)
    idx = np.arange(total)
    rows = idx // width
    cols = idx % width
    order = np.argsort(cols, kind="stable")
    order = order[np.argsort(rows[order], kind="stable")]
    order = order[np.argsort(-scores[order], kind="stable")]
    return order[:width]


def _finish(state: _FixedState) -> FixedVerify:
    width = state.width
    selected = [int(v) for v in _ranked(state).tolist()]
    nodes = [(index // width, index % width) for index in sorted(selected)]
    index_of = {node: i for i, node in enumerate(nodes)}
    tokens = []
    parents = []
    for row, col in nodes:
        tokens.append(int(state.tokens[row, col]))
        if row == 0:
            parents.append(-1)
        else:
            parent_col = int(state.parents[row, col])
            parents.append(index_of[(row - 1, parent_col)])
    return FixedVerify(tuple(tokens), tuple(parents))


def fixed_verify(root_probs, expand_probs, width: int, tau: float, max_depth: int) -> FixedVerify:
    """Run the fixed N / τ rule and return the verify tokens and parents.

    ``expand_probs`` is a list of ``[N, vocab]`` child distributions. The same
    stop as OPT applies: grow while the top-N mass increases by more than τ
    and ``depth < max_depth - 1``.
    """
    policy = FixedPolicy(int(width), float(tau), int(max_depth))
    policy.seed(root_probs)
    upcoming = list(expand_probs)
    while policy.should_grow():
        if not upcoming:
            break
        policy.expand(upcoming.pop(0))
    return policy.finish()
