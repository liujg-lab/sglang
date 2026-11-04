"""Fixed-width OPT-Tree settings.

N and tau do not change inside a round. There is no AdaCost policy switch.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class OptTreeConfig:
    """Original OPT-Tree: keep N nodes per layer, stop on a probability gap.

    ``width`` is N. ``tau`` is the threshold on the increase of E. ``max_depth``
    matches ``Tree.max_depth``, including the extra slot that implementation
    fills and then drops. This tree stops before that slot is built, so a
    chain of draft nodes is still at most ``max_depth - 1`` long.
    """

    width: int
    tau: float
    max_depth: int

    def __post_init__(self) -> None:
        if int(self.width) < 1:
            raise ValueError(f"opt-tree width must be >= 1, got {self.width}")
        if int(self.max_depth) < 2:
            raise ValueError(
                f"opt-tree max_depth must be >= 2 to match Tree.max_depth, "
                f"got {self.max_depth}"
            )
        if not math.isfinite(float(self.tau)):
            raise ValueError(f"opt-tree tau must be finite, got {self.tau}")
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "tau", float(self.tau))
        object.__setattr__(self, "max_depth", int(self.max_depth))


def longest_draft_chain(config: OptTreeConfig) -> int:
    """Most draft tokens on one root-to-leaf path. The cap is ``max_depth - 1``."""
    return int(config.max_depth) - 1


def opt_draft_lookahead_enabled() -> bool:
    """Draft layer lookahead. Default off.

    The hot path overlaps the next layer's launch with host tree updates.
    It is only taken for CUDA, page size 1, batch size 1, and Triton.
    """
    value = os.environ.get("SGLANG_OPT_DRAFT_LOOKAHEAD", "")
    return value.strip().lower() in {"1", "true", "yes", "y"}
