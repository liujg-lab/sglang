"""Fixed-width OPT-Tree settings.

N and tau do not change inside a round. There is no AdaCost policy switch.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ClearTreeConfig:
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
            raise ValueError(f"clear-tree width must be >= 1, got {self.width}")
        if int(self.max_depth) < 2:
            raise ValueError(
                f"clear-tree max_depth must be >= 2 to match Tree.max_depth, "
                f"got {self.max_depth}"
            )
        if not math.isfinite(float(self.tau)):
            raise ValueError(f"clear-tree tau must be finite, got {self.tau}")
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "tau", float(self.tau))
        object.__setattr__(self, "max_depth", int(self.max_depth))


def longest_draft_chain(config: ClearTreeConfig) -> int:
    """Most draft tokens on one root-to-leaf path. The cap is ``max_depth - 1``."""
    return int(config.max_depth) - 1


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "y"}


def opt_draft_lookahead_enabled() -> bool:
    """Draft layer lookahead. Default off.

    The hot path overlaps the next layer's launch with host tree updates.
    It is only taken for CUDA, page size 1, batch size 1, and Triton.
    """
    return _env_flag("SGLANG_CLEAR_DRAFT_LOOKAHEAD")


def opt_probe_layer_wait_enabled() -> bool:
    """Force a stream sync before each lookahead ``_run``. Default off.

    This splits the previous-layer wait into ``opt_lookahead_launch_wait_us``.
    It is a diagnostic, not the online path: the host waits for the previous
    layer to finish before preparing the next launch. Leave it unset when
    reading ``opt_layer_launch_us``.
    """
    return _env_flag("SGLANG_CLEAR_PROBE_LAYER_WAIT")
