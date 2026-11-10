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


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "y"}


def opt_draft_lookahead_enabled() -> bool:
    """Draft layer lookahead. Default off.

    The hot path overlaps the next layer's launch with host tree updates.
    It is only taken for CUDA, page size 1, batch size 1, and Triton.
    """
    return _env_flag("SGLANG_OPT_DRAFT_LOOKAHEAD")


def opt_draft_kv_window():
    """Draft tree-layer KV window. Empty disables it.

    ``full`` copies the real prefix into a shadow row. A positive integer is
    the number of recent prefix tokens kept after the sink. This is read on
    the Draft only and is not a server argument.
    """
    raw = os.environ.get("SGLANG_OPT_DRAFT_KV_WINDOW", "").strip().lower()
    if not raw:
        return None
    if raw == "full":
        return "full"
    try:
        width = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"SGLANG_OPT_DRAFT_KV_WINDOW must be empty, full, or a positive int, "
            f"got {raw!r}"
        ) from exc
    if width < 1:
        raise ValueError(
            f"SGLANG_OPT_DRAFT_KV_WINDOW must be >= 1, got {width}"
        )
    return width


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def opt_draft_kv_sink() -> int:
    """Prefix tokens kept at the start of a truncated window. Default 64."""
    return _env_int("SGLANG_OPT_DRAFT_KV_SINK", 64)


def opt_draft_kv_window_min_len() -> int:
    """Real prefix length that must be exceeded before a window is used."""
    return _env_int("SGLANG_OPT_DRAFT_KV_WINDOW_MIN_LEN", 2048)


def opt_draft_kv_window_check() -> bool:
    """Recompute a windowed layer on the real row and compare logits.

    Sampling still uses the shadow-row result. Default off.
    """
    return _env_flag("SGLANG_OPT_DRAFT_KV_WINDOW_CHECK")


def opt_probe_layer_wait_enabled() -> bool:
    """Force a stream sync before each lookahead ``_run``. Default off.

    This splits the previous-layer wait into ``opt_lookahead_launch_wait_us``.
    It is a diagnostic, not the online path: the host waits for the previous
    layer to finish before preparing the next launch. Leave it unset when
    reading ``opt_layer_launch_us``.
    """
    return _env_flag("SGLANG_OPT_PROBE_LAYER_WAIT")
