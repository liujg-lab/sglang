"""Flat layer width for one ClearTree round.

Batch size 1 starts at 8. Every other batch starts at 4. A gated replay
stores a replacement here. This module does not import the ledger.
"""

from __future__ import annotations

from typing import Dict, Optional

TARGET_CAPACITY = 33
DRAFT_CAPACITY = 8
LAYER_WIDTHS = (4, 8)
BONUS_NTPBS = (5, 9)
CANDIDATE_WIDTHS = (2, 4, 8)

_persistent_w: Dict[int, int] = {}


def reset_persistent_widths() -> None:
    _persistent_w.clear()


def assign_persistent_width(bucket: int, width: int) -> None:
    _persistent_w[int(bucket)] = int(width)


def flat_width(batch_size: int) -> int:
    """Layer width. One request uses 8; every larger batch uses 4."""
    return 8 if int(batch_size) == 1 else 4

def capture_ntpb(role: Optional[str]) -> list:
    """Graph widths for this role, widest first, including the 1-token decode graph."""
    if role == "target":
        return [TARGET_CAPACITY, 17, 9, 1]
    if role == "draft":
        return [DRAFT_CAPACITY, 4, 1]
    return [1]

def bonus_ntpbs() -> tuple:
    """Draft bonus graphs. Each is one wider than a layer width."""
    return BONUS_NTPBS

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
