"""Grow one OPT-Tree by calling ``update`` until it finishes."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from sglang.srt.speculative.clear_tree_remote.clear_tree_config import ClearTreeConfig
from sglang.srt.speculative.clear_tree_remote.clear_tree_ops import (
    ClearTreeState,
    ClearTreeStep,
    update,
)


def grow_clear_tree(config: ClearTreeConfig, next_probs) -> Tuple[List[int], List[int], List[int]]:
    """Seed and expand with ``update``. Return tokens, parents, and columns.

    ``next_probs(step, state)`` is called with ``(None, None)`` for the root
    distribution. After each non-final step it receives that step and the
    state, and must return the probabilities ``update`` expects for the next
    layer. The returned parents use ``-1`` for the committed tail.
    """
    root = next_probs(None, None)
    if not torch.is_tensor(root):
        raise TypeError("OPT-Tree root distribution must be a tensor")
    state = ClearTreeState.create(config)
    step: Optional[ClearTreeStep] = update(state, root)
    guard = 0
    while step is not None and not step.is_final:
        guard += 1
        if guard > config.max_depth:
            raise RuntimeError(
                f"OPT-Tree did not finish within max_depth {config.max_depth}"
            )
        step = update(state, next_probs(step, state))
    if step is None or step.verify is None:
        raise RuntimeError("OPT-Tree finished without a verify set")
    verify = step.verify
    return (
        [int(x) for x in verify.tokens.tolist()],
        [int(x) for x in verify.parents.tolist()],
        [int(x) for x in verify.cols.tolist()],
    )
