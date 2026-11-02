"""CPU OPT-Tree expand / stop / ancestor mask.

The control flow matches ``opt_classic_qwen3/tree.py`` ``Tree.update``:

- each layer keeps N nodes (parent path probability times that parent's top-N
  child probabilities, then a global top-N);
- E is the sum of the top-N path probabilities on every layer except the one
  just added;
- grow while ``E - E_prev > tau`` and ``depth < max_depth``;
- on stop, drop the newest layer and verify N nodes.

Draft attention defaults to the ancestor chain. ``leak=True`` reproduces the
reference ``kv_cache_mask`` that is never cleared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.speculative.opt_tree_remote.opt_tree_config import OptTreeConfig


@dataclass
class OptVerifySet:
    """N verify nodes, newest layer already removed.

    ``parents[i] == -1`` means the committed tail (the root). Any other parent
    is an index into this list and is strictly smaller than ``i``.
    ``positions`` are 1-based depths, the same values ``Tree`` returns before
    the prefix offset is added.
    """

    tokens: torch.Tensor
    depths: torch.Tensor
    cols: torch.Tensor
    parents: torch.Tensor
    positions: torch.Tensor


@dataclass
class OptTreeStep:
    """One ``Tree.update`` result.

    ``positions`` is the 1-based layer depth used before the caller adds the
    prefix offset. A non-final step fills every entry with that layer's depth.
    A final step uses each verified node's own depth.
    """

    is_final: bool
    tokens: torch.Tensor
    positions: torch.Tensor
    leak_mask: torch.Tensor
    ancestor_mask: torch.Tensor
    verify: Optional[OptVerifySet] = None


@dataclass
class OptTreeState:
    config: OptTreeConfig
    weight: torch.Tensor
    tokens: torch.Tensor
    parents: torch.Tensor
    depth: int = 0
    prev_e: Optional[torch.Tensor] = None
    kv_mask: Optional[torch.Tensor] = None
    kv_cache_mask: Optional[torch.Tensor] = None
    leak_mask: Optional[torch.Tensor] = None
    finished: bool = False

    @classmethod
    def create(cls, config: OptTreeConfig, device=None) -> "OptTreeState":
        device = torch.device("cpu") if device is None else device
        n = config.width
        d = config.max_depth
        return cls(
            config=config,
            weight=torch.zeros(d, n, dtype=torch.float32, device=device),
            tokens=torch.zeros(d, n, dtype=torch.long, device=device),
            parents=torch.zeros(d, n, dtype=torch.long, device=device),
            prev_e=torch.zeros((), dtype=torch.float32, device=device),
            kv_mask=torch.zeros(n, 0, dtype=torch.int8, device=device),
            kv_cache_mask=torch.zeros(n, n, dtype=torch.int8, device=device),
        )


def topk_probs_from_logits(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k full-vocabulary probabilities at temperature 1, in fp32.

    ``exp(logit - logsumexp)`` is the softmax of those entries. Indices match
    softmax-then-topk. The returned probabilities are not renormalized over k.
    """
    work = logits.float()
    width = int(work.shape[-1])
    k = min(max(int(k), 1), width)
    vals, idx = torch.topk(work, k, dim=-1)
    lse = torch.logsumexp(work, dim=-1, keepdim=True)
    return torch.exp(vals - lse), idx


def _probs_matrix(probs: torch.Tensor, *, seed: bool, width: int) -> torch.Tensor:
    """Return ``[vocab]`` for seed or ``[width, vocab]`` for expand."""
    t = probs.detach()
    if t.dim() == 1:
        if not seed:
            raise ValueError(
                f"expand expects [N, vocab] probabilities, got shape {tuple(probs.shape)}"
            )
        return t.float()
    if t.dim() == 2:
        if seed:
            return t[-1].float()
        if int(t.shape[0]) != width:
            raise ValueError(
                f"expand row count {int(t.shape[0])} != width {width}"
            )
        return t.float()
    if t.dim() == 3:
        if int(t.shape[0]) != 1:
            raise ValueError(f"OPT-Tree ops are bs=1, got batch {int(t.shape[0])}")
        if seed:
            return t[0, -1].float()
        if int(t.shape[1]) != width:
            raise ValueError(
                f"expand row count {int(t.shape[1])} != width {width}"
            )
        return t[0].float()
    raise ValueError(f"probabilities must be 1D, 2D, or 3D, got {tuple(probs.shape)}")


def seed(state: OptTreeState, probs: torch.Tensor) -> None:
    """Top-N of the root distribution. Depth becomes 1. Not a stopping point."""
    if state.depth != 0 or state.finished:
        raise RuntimeError("seed is only valid on an empty tree")
    n = state.config.width
    row = _probs_matrix(probs, seed=True, width=n)
    if int(row.shape[-1]) < n:
        raise ValueError(f"vocab {int(row.shape[-1])} < width {n}")
    scores, ids = torch.topk(row, k=n, dim=-1)
    state.weight[0].copy_(scores)
    state.tokens[0].copy_(ids)
    # Reference stores arange here. Logical parent of every layer-0 node is the root.
    state.parents[0].copy_(torch.arange(n, device=scores.device))
    state.depth = 1
    state.leak_mask = torch.eye(n, dtype=torch.int8, device=scores.device)


def expand_layer(state: OptTreeState, probs: torch.Tensor) -> None:
    """Append one layer of N nodes and update the leaky draft mask."""
    if state.finished or state.depth < 1:
        raise RuntimeError("expand_layer requires a seeded, unfinished tree")
    if state.depth >= state.config.max_depth:
        raise RuntimeError(
            f"expand_layer at depth {state.depth} would pass max_depth "
            f"{state.config.max_depth}"
        )
    n = state.config.width
    matrix = _probs_matrix(probs, seed=False, width=n)
    child_scores, child_ids = torch.topk(matrix, k=n, dim=-1)
    path_scores = child_scores * state.weight[state.depth - 1].unsqueeze(1)
    flat_scores = path_scores.reshape(-1)
    flat_ids = child_ids.reshape(-1)
    top_scores, top_index = torch.topk(flat_scores, k=n, dim=-1)
    parent_cols = top_index // n
    state.parents[state.depth].copy_(parent_cols)
    state.weight[state.depth].copy_(top_scores)
    state.tokens[state.depth].copy_(flat_ids[top_index])

    rows = torch.arange(n, device=parent_cols.device)
    state.kv_cache_mask[rows, parent_cols] = 1
    state.kv_mask = torch.cat([state.kv_mask[parent_cols], state.kv_cache_mask], dim=1)
    state.kv_cache_mask[rows, parent_cols] = 1
    eye = torch.eye(n, dtype=torch.int8, device=parent_cols.device)
    state.leak_mask = torch.cat([state.kv_mask, eye], dim=1)
    state.depth += 1


def acceptance_mass(state: OptTreeState) -> torch.Tensor:
    """E on ``weight[:depth-1]``, excluding the layer just added."""
    if state.depth < 2:
        raise RuntimeError("acceptance mass needs a layer that can be dropped")
    flat = state.weight[: state.depth - 1].reshape(-1)
    return torch.topk(flat, k=state.config.width, dim=-1).values.sum()


def depth_rule(state: OptTreeState) -> bool:
    """True when the tree should keep the newest layer and expand again.

    The comparison stays in fp32. This does not write ``prev_e``.
    """
    if state.prev_e is None:
        raise RuntimeError("prev_e is missing")
    e = acceptance_mass(state)
    grow = (e - state.prev_e) > state.config.tau
    return bool(grow.item()) and state.depth < state.config.max_depth


def _layer_positions(state: OptTreeState) -> torch.Tensor:
    """1-based depth of the latest filled layer, one entry per node."""
    return torch.full(
        (state.config.width,),
        state.depth,
        dtype=torch.long,
        device=state.weight.device,
    )


def ancestor_mask(state: OptTreeState, leak: bool = False) -> torch.Tensor:
    """Draft mask for the latest filled layer.

    Default: each node sees itself and its ancestors only. ``leak=True``
    returns the reference mask whose ``kv_cache_mask`` is never cleared.
    Shape is ``[N, depth * N]``.
    """
    if state.depth < 1 or state.leak_mask is None:
        raise RuntimeError("ancestor_mask requires a seeded tree")
    if leak:
        return state.leak_mask.clone()
    n = state.config.width
    d = state.depth
    mask = torch.zeros(n, d * n, dtype=torch.int8, device=state.weight.device)
    for i in range(n):
        row = d - 1
        col = i
        while True:
            mask[i, row * n + col] = 1
            if row == 0:
                break
            col = int(state.parents[row, col].item())
            row -= 1
    return mask


def _close_under_ancestors(
    selected: torch.Tensor,
    parents: torch.Tensor,
    layer_count: int,
    width: int,
) -> torch.Tensor:
    """Fixed parent walks. ``selected`` holds flat indices into ``[:layer_count]``."""
    total = int(layer_count) * int(width)
    alive = torch.zeros(total, dtype=torch.bool, device=selected.device)
    alive[selected.long()] = True
    idx = torch.arange(total, device=selected.device)
    rows = torch.div(idx, width, rounding_mode="floor")
    cols = idx % width
    for _ in range(max(int(layer_count) - 1, 0)):
        parent_cols = parents[rows, cols]
        parent_flat = (rows - 1) * width + parent_cols
        contribute = alive & (rows > 0)
        alive[parent_flat[contribute]] = True
    return torch.nonzero(alive, as_tuple=False).flatten()


def _ranked_verify_indices(
    scores: torch.Tensor, layer_count: int, width: int
) -> torch.Tensor:
    """Top-N flat indices ordered by score desc, depth asc, column asc."""
    total = int(layer_count) * int(width)
    idx = torch.arange(total, device=scores.device)
    rows = torch.div(idx, width, rounding_mode="floor")
    cols = idx % width
    order = torch.argsort(cols, stable=True)
    order = order[torch.argsort(rows[order], stable=True)]
    order = order[torch.argsort(scores[order], descending=True, stable=True)]
    return order[:width]


def finish(state: OptTreeState) -> OptVerifySet:
    """Drop the newest layer and keep the top-N nodes under the total order.

    Shallower nodes win ties, so the selected prefix already contains every
    ancestor. Closure is checked and does not add or remove nodes.
    """
    if state.finished:
        raise RuntimeError("tree is already finished")
    if state.depth < 2:
        raise RuntimeError("finish needs an expanded layer to drop")
    n = state.config.width
    layer_count = state.depth - 1
    flat = state.weight[:layer_count].reshape(-1)
    selected = _ranked_verify_indices(flat, layer_count, n)
    closed = _close_under_ancestors(
        selected, state.parents[:layer_count], layer_count, n
    )
    selected_set = {int(x) for x in selected.tolist()}
    closed_set = {int(x) for x in closed.tolist()}
    if closed_set != selected_set:
        raise RuntimeError(
            "verify top-N is not ancestor-closed; path scores must be "
            "non-increasing along edges"
        )
    kept = sorted(selected_set)
    nodes = [(idx // n, idx % n) for idx in kept]
    index_of = {node: i for i, node in enumerate(nodes)}
    tokens = []
    depths = []
    cols = []
    parents_out = []
    for row, col in nodes:
        tokens.append(int(state.tokens[row, col].item()))
        depths.append(row)
        cols.append(col)
        if row == 0:
            parents_out.append(-1)
        else:
            parent_col = int(state.parents[row, col].item())
            parent_key = (row - 1, parent_col)
            if parent_key not in index_of:
                raise RuntimeError(
                    f"missing ancestor {parent_key} for node {(row, col)}"
                )
            parents_out.append(index_of[parent_key])
    device = state.weight.device
    parent_tensor = torch.tensor(parents_out, dtype=torch.long, device=device)
    for i, parent in enumerate(parents_out):
        if parent != -1 and parent >= i:
            raise RuntimeError(f"parent index {parent} at {i} is not before the node")
    depth_tensor = torch.tensor(depths, dtype=torch.long, device=device)
    state.finished = True
    return OptVerifySet(
        tokens=torch.tensor(tokens, dtype=torch.long, device=device),
        depths=depth_tensor,
        cols=torch.tensor(cols, dtype=torch.long, device=device),
        parents=parent_tensor,
        positions=depth_tensor + 1,
    )


def update(state: OptTreeState, probs: torch.Tensor) -> OptTreeStep:
    """One original OPT step. Seed, or expand then either grow or finish."""
    if state.finished:
        raise RuntimeError("update called after finish")
    if state.depth == 0:
        seed(state, probs)
        mask = ancestor_mask(state, leak=False)
        return OptTreeStep(
            is_final=False,
            tokens=state.tokens[0].clone(),
            positions=_layer_positions(state),
            leak_mask=mask.clone(),
            ancestor_mask=mask.clone(),
        )
    expand_layer(state, probs)
    leak = ancestor_mask(state, leak=True)
    correct = ancestor_mask(state, leak=False)
    tokens = state.tokens[state.depth - 1].clone()
    positions = _layer_positions(state)
    if depth_rule(state):
        state.prev_e = acceptance_mass(state)
        return OptTreeStep(
            is_final=False,
            tokens=tokens,
            positions=positions,
            leak_mask=leak.clone(),
            ancestor_mask=correct.clone(),
        )
    verify = finish(state)
    return OptTreeStep(
        is_final=True,
        tokens=verify.tokens,
        positions=verify.positions.clone(),
        leak_mask=leak.clone(),
        ancestor_mask=correct.clone(),
        verify=verify,
    )
