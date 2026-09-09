"""Relative Probability Drop (RPD) longest-path tree verification.

A draft-tree edge v is valid iff the parent-slot logit gap
``z_parent(c*) - z_parent(c_v) <= -ln(1-tau)`` (T=1). tau=0 uses token
equality with argmax, matching greedy. Among all root paths whose every
non-root edge is valid, the longest path is committed. Ties break by
smaller sum of gaps, then leftmost sibling order.

Torch-only so CPU unit tests can import it without the rest of sglang.srt.
"""

from __future__ import annotations

import logging
import math
from typing import List, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)
_logged_rpd_cpu_fallback = False


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
    if logits.is_cuda:
        try:
            _verify_tree_rpd_cuda(**kwargs)
            return predicts, accept_index, accept_token_num
        except (ImportError, AttributeError) as e:
            global _logged_rpd_cpu_fallback
            if not _logged_rpd_cpu_fallback:
                _logged_rpd_cpu_fallback = True
                logger.warning(
                    "RPD CUDA kernel unavailable (%s); falling back to CPU. "
                    "Rebuild sgl-kernel so sgl_kernel.verify_tree_rpd is installed.",
                    e,
                )
    _verify_tree_rpd_cpu(**kwargs)
    return predicts, accept_index, accept_token_num
