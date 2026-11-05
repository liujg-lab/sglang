"""Layer forward helper. Softmax is fp32 at temperature 1."""

from __future__ import annotations

import torch


def topc_from_logits(logits: torch.Tensor, child_topk: int):
    """Return probabilities and ids. ``logits`` may be any floating dtype."""
    scores = torch.softmax(logits.float(), dim=-1)
    k = min(int(child_topk), int(scores.shape[-1]))
    return torch.topk(scores, k=k, dim=-1)
