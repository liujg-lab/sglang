"""Flat layer width and tau verify count for one ClearTree round.

Batch size 1 starts at width 8 and verify count 16. Every other batch starts
at width 4 and verify count 8. A gated replay stores a width replacement
here. This module does not import the ledger.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

DEFAULT_WIDTHS = (2, 4, 8, 16, 32, 64)
DEFAULT_KS = (2, 4, 8, 16, 32, 64)

# Lists, not tuples: startup replaces their contents in place, so modules
# that already imported these names keep seeing the configured values.
LAYER_WIDTHS = [2, 4, 8, 16, 32, 64]
CANDIDATE_WIDTHS = [2, 4, 8, 16, 32, 64]
BONUS_NTPBS = [3, 5, 9, 17, 33, 65]
VERIFY_KS = [2, 4, 8, 16, 32, 64]
TARGET_CAPACITY = 65
DRAFT_CAPACITY = 64

_persistent_w: Dict[int, int] = {}


def reset_persistent_widths() -> None:
    _persistent_w.clear()


def assign_persistent_width(bucket: int, width: int) -> None:
    _persistent_w[int(bucket)] = int(width)


def flat_width(batch_size: int) -> int:
    """Layer width. One request uses 8; every larger batch uses 4."""
    return 8 if int(batch_size) == 1 else 4


def flat_k(batch_size: int) -> int:
    """Verify count on tau. One request uses 16; every larger batch uses 8."""
    return 16 if int(batch_size) == 1 else 8


def layer_width_supported(n: int) -> bool:
    """True when ``n`` is a configured draft layer graph width."""
    try:
        width = int(n)
    except (TypeError, ValueError):
        return False
    return width in LAYER_WIDTHS


def _normalize_tiers(values: Sequence[int], *, name: str) -> list:
    """Unique integers >= 2, ascending. A string is not a list of tiers."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a list of integers >= 2, got {values!r}.")
    found = []
    seen = set()
    for item in values:
        try:
            tier = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} values must be integers >= 2, got {item!r}."
            ) from exc
        if tier < 2:
            raise ValueError(f"{name} values must be integers >= 2, got {tier}.")
        if tier in seen:
            continue
        seen.add(tier)
        found.append(tier)
    if not found:
        raise ValueError(f"{name} must contain at least one integer >= 2.")
    found.sort()
    return found


def install_candidates(widths: Optional[Sequence[int]], ks: Optional[Sequence[int]]) -> tuple:
    """Install W and K candidates and the buffer capacities they require.

    ``None`` uses the default tier list. The returned lists are ascending.
    """
    global TARGET_CAPACITY, DRAFT_CAPACITY
    width_list = (
        list(DEFAULT_WIDTHS)
        if widths is None
        else _normalize_tiers(widths, name="--clear-tree-widths")
    )
    k_list = (
        list(DEFAULT_KS) if ks is None else _normalize_tiers(ks, name="--clear-tree-ks")
    )
    LAYER_WIDTHS[:] = width_list
    CANDIDATE_WIDTHS[:] = width_list
    BONUS_NTPBS[:] = [item + 1 for item in width_list]
    VERIFY_KS[:] = k_list
    DRAFT_CAPACITY = max(width_list)
    TARGET_CAPACITY = max(k_list) + 1
    return list(width_list), list(k_list)


def capture_ntpb(role: Optional[str]) -> list:
    """Graph widths for this role, widest first, including the 1-token decode graph."""
    if role == "target":
        return sorted({int(k) + 1 for k in VERIFY_KS} | {1}, reverse=True)
    if role == "draft":
        return sorted({int(width) for width in LAYER_WIDTHS} | {1}, reverse=True)
    return [1]


def bonus_ntpbs() -> tuple:
    """Draft bonus graphs, widest first. Each is one wider than a layer width."""
    return tuple(sorted((int(width) for width in BONUS_NTPBS), reverse=True))

def batch_bucket(batch_size: int) -> int:
    """Replay state is shared inside 1, 2–4, and 5 or more."""
    size = max(int(batch_size), 1)
    if size == 1:
        return 1
    if size <= 4:
        return 2
    return 5

def persistent_width(batch_size: int) -> int:
    """Width served after a gated switch, or the flat table before one."""
    bucket = batch_bucket(batch_size)
    if bucket in _persistent_w:
        return int(_persistent_w[bucket])
    return flat_width(batch_size)
