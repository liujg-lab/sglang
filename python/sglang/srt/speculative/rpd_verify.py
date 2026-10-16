"""Relative Probability Drop (RPD) longest-path tree verification.

A draft-tree edge v is valid iff the parent-slot logit gap
``z_parent(c*) - z_parent(c_v) <= -ln(1-tau)`` (T=1). tau=0 uses token
equality with argmax, matching greedy. Among all root paths whose every
non-root edge is valid, the longest path is committed. Ties break by
smaller sum of gaps, then leftmost sibling order.

Torch-only so CPU unit tests can import it without the rest of sglang.srt.
Production verification on NPU, CUDA, and CPU uses the compact path: the vocab
reduction and paired edge logits stay on device, and the longest path is still
chosen on CPU. The independent CPU reference lives in the test helper.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from sglang.srt.speculative.standalone_remote.sr_transfer_staging import (
    alloc_host,
    submit_copy,
    wait_event,
)

logger = logging.getLogger(__name__)
_logged_compact_devices = set()
_compact_cross_device_bytes = 0
_compact_stat_waits = 0
_COMPACT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def rpd_gap_max(tau: float) -> float:
    """Return the logit-gap threshold ``-ln(1-tau)``. ``tau==0`` is 0.0."""
    tau_f = float(tau)
    if not (0.0 <= tau_f < 1.0):
        raise ValueError(f"RPD tau must be in [0, 1), got {tau}.")
    if tau_f == 0.0:
        return 0.0
    return -math.log(1.0 - tau_f)


def _children_and_parents(
    next_token: Sequence[int], next_sibling: Sequence[int]
) -> Tuple[List[List[int]], List[int]]:
    n = len(next_token)
    children: List[List[int]] = [[] for _ in range(n)]
    parent = [-1] * n
    for u in range(n):
        v = int(next_token[u])
        while v != -1:
            if 0 <= v < n:
                children[u].append(v)
                parent[v] = u
            nxt = int(next_sibling[v]) if 0 <= v < n else -1
            v = nxt
    return children, parent


def _longest_path(
    children: List[List[int]],
    valid: Sequence[bool],
    gaps: Sequence[float],
) -> List[int]:
    """Return slot indices of the longest valid path from root (slot 0).

    Child slots are greater than their parent, so a descending scan finishes
    each subtree before its parent. Children are stored left to right; the
    reverse scan sees the right sibling first. On an equal length, replace
    unless that sibling has a strictly smaller gap. Equality and NaN then
    keep the left sibling, matching Python ``<``.
    """
    n = len(children)
    length = [1] * n
    best_child = [-1] * n
    path_gap = [0.0] * n

    for u in range(n - 1, -1, -1):
        for v in reversed(children[u]):
            if not valid[v]:
                continue
            cand_gap = float(gaps[v]) + path_gap[v]
            b = best_child[u]
            replace = False
            if b < 0:
                replace = True
            elif length[v] > length[b]:
                replace = True
            elif length[v] == length[b]:
                cur_gap = float(gaps[b]) + path_gap[b]
                # ``v`` is the later, left sibling. Keep it unless ``b`` is
                # strictly better. ``<=`` would leave a NaN gap on the right.
                if not (cur_gap < cand_gap):
                    replace = True
            if replace:
                best_child[u] = v
                length[u] = 1 + length[v]
                path_gap[u] = cand_gap
    path = [0]
    u = 0
    while best_child[u] >= 0:
        u = best_child[u]
        path.append(u)
    return path


def _compact_host_read(tensor: torch.Tensor) -> torch.Tensor:
    """Logical D2H. A tensor that is already on CPU is not a cross-device copy."""
    if tensor.device.type == "cpu":
        return tensor
    _note_cross_device_bytes(tensor)
    return tensor.detach().to(device="cpu")


def _compact_device_write(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Logical H2D. Same-device tensors, including CPU copies, are not counted."""
    if tensor.device == device:
        return tensor
    _note_cross_device_bytes(tensor)
    return tensor.detach().to(device=device)


def _log_compact_once(device: torch.device) -> None:
    """Log the shared compact path once per ``device.type``."""
    kind = device.type
    if kind in _logged_compact_devices:
        return
    _logged_compact_devices.add(kind)
    if kind == "cpu":
        logger.info(
            "Speculative RPD verify path: cpu_compact_cpu_path device=%s",
            device,
        )
        return
    logger.info("Speculative RPD verify path: %s_compact_cpu_path", kind)


def _rpd_compact_layout(logits: torch.Tensor) -> Tuple[int, int, Optional[int]]:
    if logits.dtype not in _COMPACT_DTYPES:
        raise ValueError(
            "RPD compact logits dtype must be float16, bfloat16, or float32, "
            f"got {logits.dtype}."
        )
    if logits.dim() == 2:
        return int(logits.shape[0]), int(logits.shape[1]), None
    if logits.dim() == 3:
        d0, d1, vocab = (int(size) for size in logits.shape)
        return d0 * d1, vocab, d1
    raise ValueError(f"RPD compact logits must be 2D or 3D, got dim={logits.dim()}.")


def _rpd_require_device(tensor: torch.Tensor, device: torch.device, name: str) -> None:
    if tensor.device != device:
        raise ValueError(
            f"RPD {name} device {tensor.device} != logits device {device}."
        )


def _rpd_compact_reduce(
    logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vocab max on the original dtype. Returns flat values and the first argmax."""
    values, indices = torch.max(logits, dim=-1)
    return values.reshape(-1), indices.reshape(-1)


def _as_int64(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.int64:
        return tensor
    return tensor.to(dtype=torch.int64)


def _rpd_compact_tree(
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
) -> torch.Tensor:
    """Read the tree pointers before the vocab reduction. Shape is int64[4, B, W]."""
    packed = torch.stack(
        (
            _as_int64(candidates),
            _as_int64(retrive_index),
            _as_int64(retrive_next_token),
            _as_int64(retrive_next_sibling),
        ),
        dim=0,
    )
    return _compact_host_read(packed)


def _rpd_link_rows(
    tree_cpu: torch.Tensor, rows: int, vocab: int
) -> Tuple[
    List[Tuple[int, int, int]], Optional[torch.Tensor], List[List[List[int]]]
]:
    """Walk each row once.

    Returns sibling-ordered ``(batch, child_slot, parent_slot)`` edges, a CPU
    ``int64[2, E]`` index of ``(parent_row, child_token)``, and the children
    lists from that same walk. Positions with no parent, an out-of-range parent
    retrieve, or an out-of-range child token produce no edge. ``E == 0``
    returns ``None`` instead of an index tensor.
    """
    cand = tree_cpu[0].tolist()
    retr = tree_cpu[1].tolist()
    next_token = tree_cpu[2].tolist()
    next_sibling = tree_cpu[3].tolist()
    edges: List[Tuple[int, int, int]] = []
    parent_rows: List[int] = []
    child_tokens: List[int] = []
    children_rows: List[List[List[int]]] = []
    for b in range(len(cand)):
        children, parent = _children_and_parents(next_token[b], next_sibling[b])
        children_rows.append(children)
        for v in range(len(cand[b])):
            p = parent[v]
            if p < 0:
                continue
            parent_flat = int(retr[b][p])
            child_tok = int(cand[b][v])
            if parent_flat < 0 or parent_flat >= rows:
                continue
            if child_tok < 0 or child_tok >= vocab:
                continue
            edges.append((b, v, p))
            parent_rows.append(parent_flat)
            child_tokens.append(child_tok)
    if not edges:
        return edges, None, children_rows
    return (
        edges,
        torch.tensor([parent_rows, child_tokens], dtype=torch.int64),
        children_rows,
    )


def _rpd_compact_edges(
    tree_cpu: torch.Tensor, rows: int, vocab: int
) -> Tuple[List[Tuple[int, int, int]], Optional[torch.Tensor]]:
    """Build sibling-ordered edges on the host.

    Returns ``(batch, child_slot, parent_slot)`` and a CPU ``int64[2, E]``
    index of ``(parent_row, child_token)``. The children lists from the same
    walk are dropped; production selection uses ``_rpd_link_rows`` instead.
    """
    edges, edge_index, _children = _rpd_link_rows(tree_cpu, rows, vocab)
    return edges, edge_index


def _rpd_slot_argmax(
    argmax_flat: torch.Tensor, retrive_index: torch.Tensor, rows: int
) -> torch.Tensor:
    """Gather argmax at each retrieve. Invalid rows stay -1 and are not clamped into the result."""
    retr = _as_int64(retrive_index)
    safe = retr.clamp(min=0, max=rows - 1)
    gathered = argmax_flat[safe.reshape(-1)].reshape(retr.shape)
    gathered = _as_int64(gathered)
    valid = (retr >= 0) & (retr < rows)
    return torch.where(valid, gathered, torch.full_like(gathered, -1))


def _rpd_compact_edge_values(
    logits: torch.Tensor,
    z_star_flat: torch.Tensor,
    edge_index_cpu: torch.Tensor,
    row_stride: Optional[int],
) -> torch.Tensor:
    """Upload paired indices and gather ``z*`` and ``z(c)`` in the logits dtype."""
    edge_index = _compact_device_write(edge_index_cpu, logits.device)
    parent_flat = edge_index[0]
    child_token = edge_index[1]
    edge_max = z_star_flat[parent_flat]
    if row_stride is None:
        edge_candidate = logits[parent_flat, child_token]
    else:
        row_i = torch.div(parent_flat, row_stride, rounding_mode="floor")
        row_j = torch.remainder(parent_flat, row_stride)
        edge_candidate = logits[row_i, row_j, child_token]
    return torch.stack((edge_max, edge_candidate), dim=0)


def _note_cross_device_bytes(tensor: torch.Tensor) -> None:
    global _compact_cross_device_bytes
    _compact_cross_device_bytes += int(tensor.numel()) * int(tensor.element_size())


def _compact_await_stats(
    argmax: torch.Tensor, edge_values: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Bring argmax and optional edge scalars to the host with one wait."""
    global _compact_stat_waits
    _compact_stat_waits += 1
    pending = [argmax] if edge_values is None else [argmax, edge_values]
    if all(tensor.device.type == "cpu" for tensor in pending):
        argmax_cpu = _compact_host_read(argmax)
        edge_cpu = None if edge_values is None else _compact_host_read(edge_values)
        return argmax_cpu, edge_cpu
    hosts = []
    last_event = None
    for src in pending:
        dest, _pinned = alloc_host(tuple(src.shape), src.dtype, src.device)
        _note_cross_device_bytes(src)
        last_event = submit_copy(dest, src)
        hosts.append(dest)
    wait_event(last_event)
    if edge_values is None:
        return hosts[0], None
    return hosts[0], hosts[1]


@dataclass(frozen=True)
class _RPDHostTopology:
    candidates: list
    retrieve: list
    children: list
    edge_pos: dict
    edges: tuple


def _rpd_host_topology(tree_cpu, edges, children=None):
    cand, retr, next_token, next_sibling = tree_cpu.tolist()
    if children is None:
        children = [
            _children_and_parents(n, s)[0] for n, s in zip(next_token, next_sibling)
        ]
    return _RPDHostTopology(
        cand,
        retr,
        children,
        {(b, v): i for i, (b, v, _parent) in enumerate(edges)},
        tuple(edges),
    )


def _rpd_select_host(
    topology, star, stats, gap_max, use_equality, accept_width, rows, predict_len
):
    """Shared Python selection; no tensors or device operations are created."""
    if stats is not None and hasattr(stats, "tolist"):
        stats = stats.tolist()
    if use_equality and hasattr(star, "tolist"):
        star = star.tolist()
    cand, retr = topology.candidates, topology.retrieve
    edges, edge_pos = topology.edges, topology.edge_pos
    batch = len(cand)
    width = len(cand[0]) if batch else 0
    plans = []
    for b in range(batch):
        children = topology.children[b]
        valid = [False] * width
        gaps = [0.0] * width
        for v in range(width):
            edge_i = edge_pos.get((b, v))
            if edge_i is None:
                continue
            if stats is None:
                raise RuntimeError("RPD compact is missing edge statistics.")
            z_star = float(stats[0][edge_i])
            z_candidate = float(stats[1][edge_i])
            gap = z_star - z_candidate
            gaps[v] = gap
            if use_equality:
                parent_slot = int(edges[edge_i][2])
                valid[v] = int(cand[b][v]) == int(star[b][parent_slot])
            else:
                valid[v] = gap <= gap_max
        path = _longest_path(children, valid, gaps)
        if len(path) > accept_width:
            raise RuntimeError(
                f"RPD accept path length {len(path)} exceeds accept_index "
                f"width {accept_width} for batch {b}."
            )
        flats = []
        for slot in path:
            flat = int(retr[b][slot])
            if flat < 0 or flat >= rows or flat >= predict_len:
                raise ValueError(
                    f"RPD accepted retrieve {flat} is outside logits rows "
                    f"[0, {rows}) or predicts length {predict_len}."
                )
            flats.append(flat)
        plans.append((path, flats))

    accept = [[-1] * accept_width for _ in range(batch)]
    lengths = []
    folded = {}
    for b, (path, flats) in enumerate(plans):
        lengths.append(len(path) - 1)
        accept[b][: len(flats)] = flats
        for t in range(1, len(path)):
            folded[flats[t - 1]] = int(cand[b][path[t]])
        folded[flats[-1]] = int(star[b][path[-1]])
    return accept, lengths, folded


def _rpd_compact_select(
    tree_cpu: torch.Tensor,
    edges: Sequence[Tuple[int, int, int]],
    star_cpu: torch.Tensor,
    stats_cpu: Optional[torch.Tensor],
    gap_max: float,
    use_equality: bool,
    accept_width: int,
    rows: int,
    predict_len: int,
    children: Optional[Sequence] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose every path on the host and fold predict writes before any upload.

    A path longer than ``accept_index`` or a retrieve outside the logit rows
    and ``predicts`` raises here, before the result packet is uploaded.
    ``children`` comes from the same row walk that built ``edges``.
    """
    rows_list, lengths_list, folded = _rpd_select_host(
        _rpd_host_topology(tree_cpu, edges, children),
        star_cpu.tolist(),
        None if stats_cpu is None else stats_cpu.tolist(),
        gap_max,
        use_equality,
        accept_width,
        rows,
        predict_len,
    )
    accept = torch.tensor(rows_list, dtype=torch.int64).reshape(
        len(rows_list), accept_width
    )
    lengths = torch.tensor(lengths_list, dtype=torch.int64)
    if folded:
        positions = torch.tensor(list(folded.keys()), dtype=torch.int64)
        tokens = torch.tensor(list(folded.values()), dtype=torch.int64)
    else:
        positions = torch.empty((0,), dtype=torch.int64)
        tokens = torch.empty((0,), dtype=torch.int64)
    return accept, lengths, positions, tokens


def _rpd_compact_apply(
    payload: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    device: torch.device,
) -> None:
    """Upload one result packet and write it with ``copy_`` / ``index_copy_``."""
    accept_cpu, lengths, positions, tokens = payload
    batch, accept_width = accept_cpu.shape
    n_write = int(positions.numel())
    n_accept = batch * accept_width
    blob = torch.empty((n_accept + batch + 2 * n_write,), dtype=torch.int64)
    blob[:n_accept] = accept_cpu.reshape(-1)
    blob[n_accept : n_accept + batch] = lengths
    if n_write:
        blob[n_accept + batch : n_accept + batch + n_write] = positions
        blob[n_accept + batch + n_write :] = tokens
    dev_blob = _compact_device_write(blob, device)
    accept_index.copy_(
        dev_blob[:n_accept].reshape(batch, accept_width).to(dtype=accept_index.dtype)
    )
    accept_token_num.copy_(
        dev_blob[n_accept : n_accept + batch].to(dtype=accept_token_num.dtype)
    )
    if n_write:
        predicts.index_copy_(
            0,
            dev_blob[n_accept + batch : n_accept + batch + n_write],
            dev_blob[n_accept + batch + n_write :].to(dtype=predicts.dtype),
        )


def _verify_tree_rpd_compact(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    logits: torch.Tensor,
    gap_max: float,
    use_equality: bool,
) -> None:
    if candidates.dim() != 2:
        raise ValueError(f"RPD candidates must be rank 2, got dim={candidates.dim()}.")
    batch, width = candidates.shape
    if batch == 0:
        return
    if (
        retrive_index.shape != candidates.shape
        or retrive_next_token.shape != candidates.shape
        or retrive_next_sibling.shape != candidates.shape
    ):
        raise ValueError("RPD retrieve tensors must match candidates shape.")
    if accept_index.dim() != 2 or int(accept_index.shape[0]) != batch:
        raise ValueError("RPD accept_index must be [batch, width].")
    if tuple(accept_token_num.shape) != (batch,):
        raise ValueError("RPD accept_token_num must be [batch].")
    if predicts.dim() != 1:
        raise ValueError("RPD predicts must be rank 1.")
    rows, vocab, row_stride = _rpd_compact_layout(logits)
    accept_width = int(accept_index.shape[1])
    if width < 1 or rows < 1 or vocab < 1 or accept_width < 1:
        raise ValueError(
            "RPD compact requires a positive tree width, logit rows, vocab, "
            f"and accept width, got W={width} T={rows} V={vocab} S={accept_width}."
        )
    device = logits.device
    for name, tensor in (
        ("predicts", predicts),
        ("accept_index", accept_index),
        ("accept_token_num", accept_token_num),
        ("candidates", candidates),
        ("retrive_index", retrive_index),
        ("retrive_next_token", retrive_next_token),
        ("retrive_next_sibling", retrive_next_sibling),
    ):
        _rpd_require_device(tensor, device, name)
    tree_cpu = _rpd_compact_tree(
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
    )
    z_star, argmax = _rpd_compact_reduce(logits)
    edges, edge_index, children = _rpd_link_rows(tree_cpu, rows, vocab)
    slot_argmax = _rpd_slot_argmax(argmax, retrive_index, rows)
    edge_values = None
    if edge_index is not None:
        edge_values = _rpd_compact_edge_values(logits, z_star, edge_index, row_stride)
    star_cpu, stats_cpu = _compact_await_stats(slot_argmax, edge_values)
    payload = _rpd_compact_select(
        tree_cpu,
        edges,
        star_cpu,
        stats_cpu,
        float(gap_max),
        bool(use_equality),
        accept_width,
        rows,
        int(predicts.numel()),
        children,
    )
    _rpd_compact_apply(payload, predicts, accept_index, accept_token_num, device)


def verify_tree_rpd(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    logits: torch.Tensor,
    tau: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fill greedy-compatible verify buffers using RPD longest-path.

    Args:
        predicts: mutable ``[tot]`` (or flattened logits rows + 1 in EAGLE).
            Written at parent retrieve indices and the last-slot bonus.
        accept_index: mutable ``[bs, spec_steps+1]``, pre-filled with -1.
        accept_token_num: mutable ``[bs]``.
        candidates: ``[bs, num_draft_tokens]`` draft token ids (slot 0 = root).
        retrive_index / retrive_next_token / retrive_next_sibling: ``[bs, n]``.
        logits: ``[tot, vocab]`` target logits, rows indexed by retrive_index.
        tau: RPD threshold in ``[0, 1)``.
    """
    gap_max = rpd_gap_max(tau)
    use_equality = float(tau) == 0.0
    _verify_tree_rpd_compact(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        logits=logits,
        gap_max=gap_max,
        use_equality=use_equality,
    )
    _log_compact_once(logits.device)
    return predicts, accept_index, accept_token_num
