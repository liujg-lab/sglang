"""CPU OPT-Tree expand / stop / ancestor mask.

Layer selection matches ``opt_classic_qwen3/tree.py`` ``Tree.update``. The stop
check does not. The reference adds layer k + 1 before judging E on layers
1..k, then drops that extra layer. This implementation judges as soon as layer
k exists, so the draft forward that only built the dropped layer is skipped.
The verify set is the same N nodes.

- each layer keeps N nodes (parent path probability times that parent's top-N
  child probabilities, then a global top-N);
- E is the sum of the top-N path probabilities on every layer added so far;
- grow while ``E - E_prev > tau`` and ``depth < max_depth - 1``;
- on stop, verify those layers. Do not add a layer that would be discarded.

Draft attention defaults to the ancestor chain. ``leak=True`` reproduces the
reference ``kv_cache_mask`` that is never cleared.

Tree tensors stay on CPU. ``seed`` and ``expand_layer`` run top-k on the
probability device and copy that layer's scores and ids back once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from sglang.srt.speculative.clear_tree_remote.clear_tree_config import ClearTreeConfig


@dataclass
class OptVerifySet:
    """N verify nodes chosen from every layer added so far.

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
class ClearTreeStep:
    """One ``Tree.update`` result.

    ``positions`` is the 1-based layer depth used before the caller adds the
    prefix offset. A non-final step fills every entry with that layer's depth
    and sets ``ancestor_mask``. A final step uses each verified node's own
    depth and leaves both masks unset.
    """

    is_final: bool
    tokens: torch.Tensor
    positions: torch.Tensor
    leak_mask: Optional[torch.Tensor] = None
    ancestor_mask: Optional[torch.Tensor] = None
    verify: Optional[OptVerifySet] = None


@dataclass
class ClearTreeState:
    config: ClearTreeConfig
    weight: torch.Tensor
    tokens: torch.Tensor
    parents: torch.Tensor
    depth: int = 0
    prev_e: Optional[torch.Tensor] = None
    anc_mask: Optional[torch.Tensor] = None
    finished: bool = False

    @classmethod
    def create(cls, config: ClearTreeConfig) -> "ClearTreeState":
        """Allocate the tree on CPU. Top-k scores and ids are copied in later."""
        n = config.width
        d = config.max_depth
        return cls(
            config=config,
            weight=torch.zeros(d, n, dtype=torch.float32),
            tokens=torch.zeros(d, n, dtype=torch.long),
            parents=torch.zeros(d, n, dtype=torch.long),
            prev_e=torch.zeros((), dtype=torch.float32),
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


def _pack_scores_and_ids(
    scores: torch.Tensor, ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy fp32 scores and int64 ids through one uint8 buffer onto CPU.

    The id region is 8-byte aligned and is not cast through float, so token
    ids above 2^24 stay exact. ``.cpu()`` is the single transfer.
    """
    if scores.device != ids.device:
        raise RuntimeError(
            f"scores device {scores.device} != ids device {ids.device}"
        )
    scores_f = scores.detach().to(dtype=torch.float32).contiguous()
    ids_i = ids.detach().to(dtype=torch.int64).contiguous()
    score_flat = scores_f.reshape(-1)
    id_flat = ids_i.reshape(-1)
    score_bytes = int(score_flat.numel()) * 4
    pad = (8 - (score_bytes % 8)) % 8
    id_bytes = int(id_flat.numel()) * 8
    raw = torch.empty(
        score_bytes + pad + id_bytes, dtype=torch.uint8, device=scores_f.device
    )
    raw[:score_bytes].view(torch.float32).copy_(score_flat)
    raw[score_bytes + pad :].view(torch.int64).copy_(id_flat)
    host = raw.cpu()
    out_scores = host[:score_bytes].view(torch.float32).reshape(scores_f.shape).clone()
    out_ids = host[score_bytes + pad :].view(torch.int64).reshape(ids_i.shape).clone()
    return out_scores, out_ids


def _host_scores_and_ids(
    scores: torch.Tensor, ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One D2H of a top-k result. CPU inputs are returned without a copy."""
    if scores.device.type == "cpu" and ids.device.type == "cpu":
        return (
            scores.detach().float().contiguous(),
            ids.detach().to(dtype=torch.long).contiguous(),
        )
    return _pack_scores_and_ids(scores, ids)


def seed_from_topk(
    state: ClearTreeState, scores: torch.Tensor, ids: torch.Tensor
) -> None:
    """Write one host top-N root. Depth becomes 1. Not a stopping point."""
    if state.depth != 0 or state.finished:
        raise RuntimeError("seed is only valid on an empty tree")
    n = state.config.width
    state.weight[0].copy_(scores)
    state.tokens[0].copy_(ids)
    # Reference stores arange here. Logical parent of every layer-0 node is the root.
    state.parents[0].copy_(torch.arange(n))
    state.depth = 1
    state.anc_mask = torch.eye(n, dtype=torch.int8)


def seed(state: ClearTreeState, probs: torch.Tensor) -> None:
    """Top-N of the root distribution. Depth becomes 1. Not a stopping point."""
    n = state.config.width
    row = _probs_matrix(probs, seed=True, width=n)
    if int(row.shape[-1]) < n:
        raise ValueError(f"vocab {int(row.shape[-1])} < width {n}")
    scores, ids = _host_scores_and_ids(*torch.topk(row, k=n, dim=-1))
    seed_from_topk(state, scores, ids)


def expand_from_topk(
    state: ClearTreeState, scores: torch.Tensor, ids: torch.Tensor
) -> None:
    """Append one layer from host ``[N, N]`` child scores and token ids."""
    if state.finished or state.depth < 1 or state.anc_mask is None:
        raise RuntimeError("expand_layer requires a seeded, unfinished tree")
    if state.depth >= state.config.max_depth:
        raise RuntimeError(
            f"expand_layer at depth {state.depth} would pass max_depth "
            f"{state.config.max_depth}"
        )
    n = state.config.width
    path_scores = scores * state.weight[state.depth - 1].unsqueeze(1)
    flat_scores = path_scores.reshape(-1)
    flat_ids = ids.reshape(-1)
    top_scores, top_index = torch.topk(flat_scores, k=n, dim=-1)
    parent_cols = top_index // n
    state.parents[state.depth].copy_(parent_cols)
    state.weight[state.depth].copy_(top_scores)
    state.tokens[state.depth].copy_(flat_ids[top_index])
    # Row i is the parent's ancestor row, including the parent, plus this node.
    eye = torch.eye(n, dtype=torch.int8)
    state.anc_mask = torch.cat([state.anc_mask[parent_cols], eye], dim=1)
    state.depth += 1


def seed_on_device(
    scores: torch.Tensor, ids: torch.Tensor, n: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Device root tokens and an identity ancestor mask.

    ``scores`` and ``ids`` are the root top-k, at least ``n`` wide. The first
    ``n`` become the layer. No host copy and no second top-k.
    """
    n = int(n)
    flat_scores = scores.reshape(-1)
    flat_ids = ids.reshape(-1)
    if int(flat_scores.numel()) < n or int(flat_ids.numel()) < n:
        raise ValueError(
            f"seed top-k {int(flat_ids.numel())} is shorter than width {n}"
        )
    top_scores = flat_scores[:n].to(dtype=torch.float32)
    tokens = flat_ids[:n].to(dtype=torch.long)
    anc = torch.eye(n, dtype=torch.int8, device=tokens.device)
    return top_scores, tokens, anc


def expand_on_device(
    weight_prev: torch.Tensor,
    anc: torch.Tensor,
    scores: torch.Tensor,
    ids: torch.Tensor,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same selection as ``expand_from_topk``, kept on ``scores``' device.

    ``scores`` and ``ids`` are ``[N, K]`` with ``K >= n``. Only the first ``n``
    children of each parent enter the path-score top-k. The returned ancestor
    mask is the parent's row plus a new identity block.
    """
    n = int(n)
    if scores.dim() != 2 or ids.dim() != 2:
        raise ValueError(
            f"expand_on_device expects [N, K] scores and ids, "
            f"got {tuple(scores.shape)} and {tuple(ids.shape)}"
        )
    if int(scores.shape[0]) != n or int(ids.shape[0]) != n:
        raise ValueError(
            f"expand_on_device row count {int(scores.shape[0])} != width {n}"
        )
    if int(scores.shape[1]) < n or int(ids.shape[1]) < n:
        raise ValueError(
            f"expand_on_device needs at least {n} children, "
            f"got {int(scores.shape[1])}"
        )
    if int(weight_prev.reshape(-1).numel()) < n:
        raise ValueError("expand_on_device weight_prev is shorter than width")
    if anc.dim() != 2 or int(anc.shape[0]) != n:
        raise ValueError(
            f"expand_on_device ancestor mask {tuple(anc.shape)} != [{n}, cols]"
        )
    child_scores = scores[:, :n]
    child_ids = ids[:, :n]
    path_scores = child_scores * weight_prev.reshape(-1)[:n].unsqueeze(1)
    flat_scores = path_scores.reshape(-1)
    flat_ids = child_ids.reshape(-1)
    top_scores, top_index = torch.topk(flat_scores, k=n, dim=-1)
    parent_cols = torch.div(top_index, n, rounding_mode="floor")
    tokens = flat_ids[top_index].to(dtype=torch.long)
    eye = torch.eye(n, dtype=anc.dtype, device=anc.device)
    anc_new = torch.cat([anc[parent_cols.to(dtype=torch.long)], eye], dim=1)
    return top_scores.to(dtype=torch.float32), tokens, parent_cols.to(dtype=torch.long), anc_new


def mirror_layer(
    state: ClearTreeState,
    top_scores: torch.Tensor,
    tokens: torch.Tensor,
    parent_cols: Optional[torch.Tensor],
    *,
    seed: bool,
) -> None:
    """Write one host layer from values already chosen on device.

    The ancestor mask is rebuilt from ``parent_cols``. This does not run top-k.
    CUDA and CPU break score ties differently, so a second top-k would point
    KV at the wrong node.
    """
    n = state.config.width
    scores = top_scores.detach().to(dtype=torch.float32, device="cpu").reshape(-1)[:n]
    toks = tokens.detach().to(dtype=torch.long, device="cpu").reshape(-1)[:n]
    if int(scores.numel()) != n or int(toks.numel()) != n:
        raise ValueError(f"mirror layer needs {n} scores and tokens")
    if seed:
        seed_from_topk(state, scores, toks)
        return
    if state.finished or state.depth < 1 or state.anc_mask is None:
        raise RuntimeError("mirror_layer requires a seeded, unfinished tree")
    if state.depth >= state.config.max_depth:
        raise RuntimeError(
            f"mirror_layer at depth {state.depth} would pass max_depth "
            f"{state.config.max_depth}"
        )
    if parent_cols is None:
        raise ValueError("mirror_layer expand needs parent_cols")
    cols = parent_cols.detach().to(dtype=torch.long, device="cpu").reshape(-1)[:n]
    if int(cols.numel()) != n:
        raise ValueError(f"mirror_layer needs {n} parent columns")
    state.parents[state.depth].copy_(cols)
    state.weight[state.depth].copy_(scores)
    state.tokens[state.depth].copy_(toks)
    eye = torch.eye(n, dtype=torch.int8)
    state.anc_mask = torch.cat([state.anc_mask[cols], eye], dim=1)
    state.depth += 1


def expand_layer(state: ClearTreeState, probs: torch.Tensor) -> None:
    """Append one layer of N nodes and extend the ancestor mask."""
    n = state.config.width
    matrix = _probs_matrix(probs, seed=False, width=n)
    child_scores, child_ids = _host_scores_and_ids(*torch.topk(matrix, k=n, dim=-1))
    expand_from_topk(state, child_scores, child_ids)


def _query_width(row: torch.Tensor, *, seed: bool) -> int:
    """Row count ``_probs_matrix`` checks. Independent of the top-k width."""
    if seed:
        return 1
    t = row.detach()
    if t.dim() == 2:
        return int(t.shape[0])
    if t.dim() == 3:
        return int(t.shape[1])
    raise ValueError(
        f"expand expects [N, vocab] probabilities, got shape {tuple(row.shape)}"
    )


def host_topk_rows(
    rows: Sequence[torch.Tensor], k: int, *, seed: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k already-softmaxed rows on their device, then one host copy.

    ``k`` is the number of ids, not the row count. Seed rows become ``[B, k]``.
    Expand rows stay ``[N, vocab]`` and become ``[B, N, k]``. CPU inputs stay
    on the no-copy path inside ``_host_scores_and_ids``.
    """
    if not rows:
        raise ValueError("host_topk_rows needs at least one row")
    k = int(k)
    matrices = [
        _probs_matrix(row, seed=seed, width=_query_width(row, seed=seed))
        for row in rows
    ]
    vocab = int(matrices[0].shape[-1])
    if vocab < k:
        raise ValueError(f"vocab {vocab} < topk {k}")
    stacked = torch.stack(matrices, dim=0)
    scores, ids = torch.topk(stacked, k=k, dim=-1)
    return _host_scores_and_ids(scores, ids)


def acceptance_mass(state: ClearTreeState) -> torch.Tensor:
    """E on ``weight[:depth]``, every layer added so far."""
    if state.depth < 1:
        raise RuntimeError("acceptance mass needs a seeded tree")
    flat = state.weight[: state.depth].reshape(-1)
    return torch.topk(flat, k=state.config.width, dim=-1).values.sum()


def depth_rule(state: ClearTreeState) -> bool:
    """True when the latest layer should be forwarded and expanded again.

    The comparison stays in fp32. This does not write ``prev_e``.
    ``depth < max_depth - 1`` is the reference ``depth < max_depth`` check
    after the extra layer, without building that layer.
    """
    if state.prev_e is None:
        raise RuntimeError("prev_e is missing")
    e = acceptance_mass(state)
    grow = (e - state.prev_e) > state.config.tau
    return bool(grow.item()) and state.depth < state.config.max_depth - 1


def _layer_positions(state: ClearTreeState) -> torch.Tensor:
    """1-based depth of the latest filled layer, one entry per node."""
    return torch.full(
        (state.config.width,),
        state.depth,
        dtype=torch.long,
        device=state.weight.device,
    )


def _leak_mask(state: ClearTreeState) -> torch.Tensor:
    """Reference mask whose ``kv_cache_mask`` is never cleared.

    Replayed from ``parents[1:depth]`` only when requested. The second write
    of each parent column matches the reference and does not change bits.
    """
    n = state.config.width
    if state.depth == 1:
        return torch.eye(n, dtype=torch.int8)
    kv_cache_mask = torch.zeros(n, n, dtype=torch.int8)
    kv_mask = torch.zeros(n, 0, dtype=torch.int8)
    rows = torch.arange(n)
    for layer in range(1, state.depth):
        parent_cols = state.parents[layer]
        kv_cache_mask[rows, parent_cols] = 1
        kv_mask = torch.cat([kv_mask[parent_cols], kv_cache_mask], dim=1)
        kv_cache_mask[rows, parent_cols] = 1
    return torch.cat([kv_mask, torch.eye(n, dtype=torch.int8)], dim=1)


def ancestor_mask(state: ClearTreeState, leak: bool = False) -> torch.Tensor:
    """Draft mask for the latest filled layer.

    Default: the stored ancestor mask, where each node sees itself and its
    ancestors only. ``leak=True`` rebuilds the reference mask whose
    ``kv_cache_mask`` is never cleared. Shape is ``[N, depth * N]``.
    """
    if state.depth < 1 or state.anc_mask is None:
        raise RuntimeError("ancestor_mask requires a seeded tree")
    if leak:
        return _leak_mask(state)
    return state.anc_mask


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


def finish(state: ClearTreeState) -> OptVerifySet:
    """Keep the top-N nodes of every layer added so far.

    Shallower nodes win ties, so the selected prefix already contains every
    ancestor. Closure is checked and does not add or remove nodes.
    """
    if state.finished:
        raise RuntimeError("tree is already finished")
    if state.depth < 1:
        raise RuntimeError("finish needs a seeded tree")
    n = state.config.width
    layer_count = state.depth
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
    token_rows, parent_rows = torch.stack(
        [state.tokens[:layer_count], state.parents[:layer_count]], dim=0
    ).tolist()
    tokens = []
    depths = []
    cols = []
    parents_out = []
    for row, col in nodes:
        tokens.append(int(token_rows[row][col]))
        depths.append(row)
        cols.append(col)
        if row == 0:
            parents_out.append(-1)
        else:
            parent_col = int(parent_rows[row][col])
            parent_key = (row - 1, parent_col)
            if parent_key not in index_of:
                raise RuntimeError(
                    f"missing ancestor {parent_key} for node {(row, col)}"
                )
            parents_out.append(index_of[parent_key])
    parent_tensor = torch.tensor(parents_out, dtype=torch.long)
    for i, parent in enumerate(parents_out):
        if parent != -1 and parent >= i:
            raise RuntimeError(f"parent index {parent} at {i} is not before the node")
    depth_tensor = torch.tensor(depths, dtype=torch.long)
    state.finished = True
    return OptVerifySet(
        tokens=torch.tensor(tokens, dtype=torch.long),
        depths=depth_tensor,
        cols=torch.tensor(cols, dtype=torch.long),
        parents=parent_tensor,
        positions=depth_tensor + 1,
    )


def step_after_layer(state: ClearTreeState) -> ClearTreeStep:
    """Judge E after a layer has been written and return the next step.

    A non-final step asks for a draft forward of the layer just added. A final
    step verifies that layer and does not forward it, and does not build a mask.
    """
    if depth_rule(state):
        state.prev_e = acceptance_mass(state)
        return ClearTreeStep(
            is_final=False,
            tokens=state.tokens[state.depth - 1].clone(),
            positions=_layer_positions(state),
            ancestor_mask=state.anc_mask,
        )
    verify = finish(state)
    return ClearTreeStep(
        is_final=True,
        tokens=verify.tokens,
        positions=verify.positions.clone(),
        verify=verify,
    )


def update(state: ClearTreeState, probs: torch.Tensor) -> ClearTreeStep:
    """Seed or expand, then judge E on every layer present."""
    if state.finished:
        raise RuntimeError("update called after finish")
    if state.depth == 0:
        seed(state, probs)
    else:
        expand_layer(state, probs)
    return step_after_layer(state)
