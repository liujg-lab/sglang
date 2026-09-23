"""Relative Probability Drop (RPD) longest-path tree verification.

A draft-tree edge v is valid iff the parent-slot logit gap
``z_parent(c*) - z_parent(c_v) <= -ln(1-tau)`` (T=1). tau=0 uses token
equality with argmax, matching greedy. Among all root paths whose every
non-root edge is valid, the longest path is committed. Ties break by
smaller sum of gaps, then leftmost sibling order.

Torch-only so CPU unit tests can import it without the rest of sglang.srt.
NPU keeps the vocab reduction and paired edge logits on device and still
chooses the longest path on CPU. CUDA uses the existing kernel. Every other
device, including CPU, uses the reference.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)
_logged_rpd_cpu_fallback = False
_logged_rpd_path = False
_logged_rpd_npu_path = False
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


def edge_valid_from_logits(z_star: float, z_c: float, tau: float) -> bool:
    """Accept the edge iff the parent logit gap is within the RPD threshold.

    tau=0 requires exact equality of the two logits only when they correspond
    to the same argmax token; callers that have token ids should use
    ``c_v == c_star`` instead. This helper is the gap form for tau>0 tests.
    """
    if float(tau) == 0.0:
        return float(z_c) >= float(z_star)
    return (float(z_star) - float(z_c)) <= rpd_gap_max(tau)


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
    """Return slot indices of the longest valid path from root (slot 0)."""
    n = len(children)
    length = [1] * n
    best_child = [-1] * n
    path_gap = [0.0] * n

    def dfs(u: int) -> None:
        for v in children[u]:
            dfs(v)
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
                if cand_gap < cur_gap:
                    replace = True
            if replace:
                best_child[u] = v
                length[u] = 1 + length[v]
                path_gap[u] = cand_gap

    dfs(0)
    path = [0]
    u = 0
    while best_child[u] >= 0:
        u = best_child[u]
        path.append(u)
    return path


def _prepare_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 3:
        return logits.reshape(-1, logits.shape[-1])
    return logits


def _verify_tree_rpd_cuda(
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
    from sgl_kernel import verify_tree_rpd as verify_tree_rpd_cuda

    logits = _prepare_logits(logits)
    if not logits.is_contiguous():
        logits = logits.contiguous()
    candidates = candidates.contiguous()
    retrive_index = retrive_index.contiguous()
    retrive_next_token = retrive_next_token.contiguous()
    retrive_next_sibling = retrive_next_sibling.contiguous()
    z_star = logits.amax(dim=-1).contiguous()
    target_predict = logits.argmax(dim=-1).to(torch.int64).contiguous()
    accept_index.fill_(-1)
    verify_tree_rpd_cuda(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        logits,
        z_star,
        target_predict,
        float(gap_max),
        bool(use_equality),
    )


def _verify_tree_rpd_cpu(
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
    bs, n = candidates.shape
    logits_f32 = _prepare_logits(logits).float()
    target_predict = torch.argmax(logits_f32, dim=-1)
    z_star = logits_f32.max(dim=-1).values

    next_token = retrive_next_token.detach().to("cpu")
    next_sibling = retrive_next_sibling.detach().to("cpu")
    cand_cpu = candidates.detach().to("cpu")
    retr_cpu = retrive_index.detach().to("cpu")
    argmax_cpu = target_predict.detach().to("cpu")
    z_star_cpu = z_star.detach().to("cpu")
    logits_cpu = logits_f32.detach().to("cpu")

    accept_index.fill_(-1)
    vocab = logits_cpu.shape[-1]

    for b in range(bs):
        nt = next_token[b].tolist()
        ns = next_sibling[b].tolist()
        children, parent = _children_and_parents(nt, ns)
        valid = [False] * n
        gaps = [0.0] * n
        for v in range(n):
            p = parent[v]
            if p < 0:
                continue
            parent_flat = int(retr_cpu[b, p].item())
            child_tok = int(cand_cpu[b, v].item())
            if parent_flat < 0 or parent_flat >= z_star_cpu.numel():
                continue
            if child_tok < 0 or child_tok >= vocab:
                continue
            star_tok = int(argmax_cpu[parent_flat].item())
            z_s = float(z_star_cpu[parent_flat].item())
            z_c = float(logits_cpu[parent_flat, child_tok].item())
            gap = z_s - z_c
            gaps[v] = gap
            if use_equality:
                valid[v] = child_tok == star_tok
            else:
                valid[v] = gap <= gap_max

        path = _longest_path(children, valid, gaps)
        k = len(path) - 1
        accept_token_num[b] = k
        for t, slot in enumerate(path):
            flat = int(retr_cpu[b, slot].item())
            accept_index[b, t] = flat
        for t in range(1, len(path)):
            parent_slot = path[t - 1]
            child_slot = path[t]
            parent_flat = int(retr_cpu[b, parent_slot].item())
            child_tok = int(cand_cpu[b, child_slot].item())
            predicts[parent_flat] = child_tok
        last_flat = int(retr_cpu[b, path[-1]].item())
        predicts[last_flat] = int(argmax_cpu[last_flat].item())


def compact_cross_device_bytes() -> int:
    """Bytes moved by the compact helpers between two different devices."""
    return _compact_cross_device_bytes


def reset_compact_cross_device_bytes() -> None:
    global _compact_cross_device_bytes
    _compact_cross_device_bytes = 0


def compact_stat_waits() -> int:
    """How many times argmax and edge scalars were waited on together."""
    return _compact_stat_waits


def reset_compact_stat_waits() -> None:
    global _compact_stat_waits
    _compact_stat_waits = 0


def _compact_host_read(tensor: torch.Tensor) -> torch.Tensor:
    """Logical D2H. A tensor that is already on CPU is not a cross-device copy."""
    global _compact_cross_device_bytes
    if tensor.device.type == "cpu":
        return tensor
    _compact_cross_device_bytes += int(tensor.numel()) * int(tensor.element_size())
    return tensor.detach().to(device="cpu")


def _compact_device_write(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Logical H2D. Same-device tensors, including CPU copies, are not counted."""
    global _compact_cross_device_bytes
    if tensor.device == device:
        return tensor
    _compact_cross_device_bytes += int(tensor.numel()) * int(tensor.element_size())
    return tensor.detach().to(device=device)


def _rpd_backend_name(device_type: str, is_cuda: bool) -> str:
    """Pick the RPD backend. NPU wins even when ``is_cuda`` has been aliased."""
    if device_type == "npu":
        return "npu"
    if bool(is_cuda) or device_type == "cuda":
        return "cuda"
    return "cpu"


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


def _rpd_compact_edges(
    tree_cpu: torch.Tensor, rows: int, vocab: int
) -> Tuple[List[Tuple[int, int, int]], Optional[torch.Tensor]]:
    """Build sibling-ordered edges on the host.

    Returns ``(batch, child_slot, parent_slot)`` and a CPU ``int64[2, E]``
    index of ``(parent_row, child_token)``. Positions with no parent, an
    out-of-range parent retrieve, or an out-of-range child token produce no
    edge. ``E == 0`` returns ``None`` instead of an index tensor.
    """
    cand = tree_cpu[0].tolist()
    retr = tree_cpu[1].tolist()
    next_token = tree_cpu[2].tolist()
    next_sibling = tree_cpu[3].tolist()
    edges: List[Tuple[int, int, int]] = []
    parent_rows: List[int] = []
    child_tokens: List[int] = []
    for b in range(len(cand)):
        _, parent = _children_and_parents(next_token[b], next_sibling[b])
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
        return edges, None
    return edges, torch.tensor([parent_rows, child_tokens], dtype=torch.int64)


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


def _try_pinned_like(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    try:
        return torch.empty(
            tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True
        )
    except (RuntimeError, NotImplementedError):
        return None


def _record_stream_event(device: torch.device):
    try:
        if device.type == "cuda":
            event = torch.cuda.Event()
        elif device.type == "npu":
            event = torch.npu.Event()
        else:
            return None
        event.record()
        return event
    except (AttributeError, RuntimeError):
        return None


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
    pinned = []
    for tensor in pending:
        dest = _try_pinned_like(tensor)
        if dest is None:
            pinned = []
            break
        pinned.append(dest)
    if not pinned:
        argmax_cpu = _compact_host_read(argmax)
        edge_cpu = None if edge_values is None else _compact_host_read(edge_values)
        return argmax_cpu, edge_cpu
    for dest, src in zip(pinned, pending):
        _note_cross_device_bytes(src)
        dest.copy_(src, non_blocking=True)
    event = _record_stream_event(pending[0].device)
    if event is None:
        stream_device = pending[0].device
        if stream_device.type == "cuda":
            torch.cuda.current_stream(stream_device).synchronize()
        elif stream_device.type == "npu":
            torch.npu.current_stream(stream_device).synchronize()
    else:
        event.synchronize()
    if edge_values is None:
        return pinned[0], None
    return pinned[0], pinned[1]


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose every path on the host and fold predict writes before any upload.

    A path longer than ``accept_index`` or a retrieve outside the logit rows
    and ``predicts`` raises here, before the result packet is uploaded.
    """
    cand = tree_cpu[0].tolist()
    retr = tree_cpu[1].tolist()
    next_token = tree_cpu[2].tolist()
    next_sibling = tree_cpu[3].tolist()
    star = star_cpu.tolist()
    batch = len(cand)
    width = len(cand[0]) if batch else 0
    edge_pos = {(b, v): i for i, (b, v, _parent) in enumerate(edges)}
    plans: List[Tuple[List[int], List[int]]] = []
    for b in range(batch):
        children, _parent = _children_and_parents(next_token[b], next_sibling[b])
        valid = [False] * width
        gaps = [0.0] * width
        for v in range(width):
            edge_i = edge_pos.get((b, v))
            if edge_i is None:
                continue
            if stats_cpu is None:
                raise RuntimeError("RPD compact is missing edge statistics.")
            z_star = float(stats_cpu[0, edge_i])
            z_candidate = float(stats_cpu[1, edge_i])
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
        flats: List[int] = []
        for slot in path:
            flat = int(retr[b][slot])
            if flat < 0 or flat >= rows or flat >= predict_len:
                raise ValueError(
                    f"RPD accepted retrieve {flat} is outside logits rows "
                    f"[0, {rows}) or predicts length {predict_len}."
                )
            bonus = int(star[b][slot])
            if bonus < 0:
                raise ValueError(
                    "RPD clamp placeholder reached an accepted retrieve."
                )
            flats.append(flat)
        plans.append((path, flats))

    accept = torch.full((batch, accept_width), -1, dtype=torch.int64)
    lengths = torch.empty((batch,), dtype=torch.int64)
    writes: List[Tuple[int, int]] = []
    for b, (path, flats) in enumerate(plans):
        lengths[b] = len(path) - 1
        for t, flat in enumerate(flats):
            accept[b, t] = flat
        for t in range(1, len(path)):
            child_slot = path[t]
            writes.append((flats[t - 1], int(cand[b][child_slot])))
        last_slot = path[-1]
        writes.append((flats[-1], int(star[b][last_slot])))

    folded: dict[int, int] = {}
    for pos, tok in writes:
        folded[pos] = int(tok)
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
        raise ValueError(
            f"RPD candidates must be rank 2, got dim={candidates.dim()}."
        )
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
    edges, edge_index = _rpd_compact_edges(tree_cpu, rows, vocab)
    slot_argmax = _rpd_slot_argmax(argmax, retrive_index, rows)
    edge_values = None
    if edge_index is not None:
        edge_values = _rpd_compact_edge_values(
            logits, z_star, edge_index, row_stride
        )
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
    )
    _rpd_compact_apply(
        payload, predicts, accept_index, accept_token_num, device
    )


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
    global _logged_rpd_path, _logged_rpd_cpu_fallback, _logged_rpd_npu_path
    gap_max = rpd_gap_max(tau)
    use_equality = float(tau) == 0.0
    kwargs = dict(
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
    backend = _rpd_backend_name(logits.device.type, bool(logits.is_cuda))
    if backend == "npu":
        _verify_tree_rpd_compact(**kwargs)
        if not _logged_rpd_npu_path:
            _logged_rpd_npu_path = True
            logger.info("Speculative RPD verify path: npu_compact_cpu_path")
        return predicts, accept_index, accept_token_num
    if backend == "cuda":
        try:
            _verify_tree_rpd_cuda(**kwargs)
            if not _logged_rpd_path:
                _logged_rpd_path = True
                logger.info("Speculative RPD verify path: cuda_kernel")
            return predicts, accept_index, accept_token_num
        except (ImportError, AttributeError) as e:
            if not _logged_rpd_cpu_fallback:
                _logged_rpd_cpu_fallback = True
                logger.warning(
                    "RPD CUDA kernel unavailable (%s); falling back to CPU. "
                    "Rebuild sgl-kernel so sgl_kernel.verify_tree_rpd is installed.",
                    e,
                )
    if not _logged_rpd_path:
        _logged_rpd_path = True
        logger.info(
            "Speculative RPD verify path: cpu_reference device=%s", logits.device
        )
    _verify_tree_rpd_cpu(**kwargs)
    return predicts, accept_index, accept_token_num
