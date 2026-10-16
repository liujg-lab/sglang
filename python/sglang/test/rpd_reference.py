"""Independent RPD reference and test-facing compact counters.

Production verification does not import this module. The CPU reference keeps
its own full-logit read so a compact-path change cannot rewrite the oracle.
"""

from __future__ import annotations

import torch

import sglang.srt.speculative.rpd_verify as rpd


def _prepare_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 3:
        return logits.reshape(-1, logits.shape[-1])
    return logits


def edge_valid_from_logits(z_star: float, z_c: float, tau: float) -> bool:
    """Gap check for tests. Production selection does not call this helper.

    tau=0 requires exact equality of the two logits only when they correspond
    to the same argmax token; callers that have token ids should use
    ``c_v == c_star`` instead. This helper is the gap form for tau>0 tests.
    """
    if float(tau) == 0.0:
        return float(z_c) >= float(z_star)
    return (float(z_star) - float(z_c)) <= rpd.rpd_gap_max(tau)


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
    """Independent reference for tests. Production CPU verification uses compact."""
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
        children, parent = rpd._children_and_parents(nt, ns)
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

        path = rpd._longest_path(children, valid, gaps)
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


class _CompactStats:
    """Test-facing view of the compact transfer counters stored on rpd_verify."""

    @property
    def cross_device_bytes(self) -> int:
        return int(rpd._compact_cross_device_bytes)

    @cross_device_bytes.setter
    def cross_device_bytes(self, value: int) -> None:
        rpd._compact_cross_device_bytes = int(value)

    @property
    def stat_waits(self) -> int:
        return int(rpd._compact_stat_waits)

    @stat_waits.setter
    def stat_waits(self, value: int) -> None:
        rpd._compact_stat_waits = int(value)


def compact_cross_device_bytes() -> int:
    """Bytes moved by the compact helpers between two different devices."""
    return int(rpd._compact_cross_device_bytes)


def reset_compact_cross_device_bytes() -> None:
    rpd._compact_cross_device_bytes = 0


def compact_stat_waits() -> int:
    """How many times argmax and edge scalars were waited on together."""
    return int(rpd._compact_stat_waits)


def reset_compact_stat_waits() -> None:
    rpd._compact_stat_waits = 0
