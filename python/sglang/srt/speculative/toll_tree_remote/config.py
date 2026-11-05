"""TollTree settings that can be checked without starting a server."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence


def _strict_buckets(values: Sequence[int], *, kind: str) -> tuple[int, ...]:
    if values is None or len(tuple(values)) == 0:
        raise ValueError(f"{kind} buckets must not be empty")
    buckets = tuple(int(v) for v in values)
    if any(v <= 0 for v in buckets):
        raise ValueError(f"{kind} buckets must be positive, got {buckets}")
    if list(buckets) != sorted(set(buckets)):
        raise ValueError(f"{kind} buckets must be strictly increasing, got {buckets}")
    return buckets


@dataclass(frozen=True)
class TollConfig:
    """Bucket and policy checks for ``TOLL_TREE_REMOTE``.

    Width 1 is rejected because ``r1_*`` is the decode graph. Verify buckets
    must be non-empty. RPD and DP attention are refused. A draft on NPU is
    refused until that path exists. This object does not read ``ServerArgs``.
    """

    width_buckets: Sequence[int] = (2, 4, 8, 16, 32)
    verify_buckets: Sequence[int] = (8, 16, 32)
    max_depth: int = 10
    child_topk: int = 16
    policy: str = "toll"
    fixed_width: Optional[int] = None
    fixed_tau: Optional[float] = None
    price_alpha: float = 0.02
    warmup_rounds: int = 64
    probe_interval: int = 64
    no_exit: bool = False
    fixed_k: Optional[int] = None
    verify_mode: str = "greedy"
    dp_attention: bool = False
    draft_on_npu: bool = False
    speculate_delta: float = 0.0
    bytes_per_token: float = 0.0

    def __post_init__(self) -> None:
        widths = _strict_buckets(self.width_buckets, kind="width")
        if widths[0] < 2:
            raise ValueError(
                f"width buckets must be >= 2 so r1_* stays the decode graph, "
                f"got {widths}"
            )
        verifies = _strict_buckets(self.verify_buckets, kind="verify")
        if int(self.max_depth) < 1:
            raise ValueError(f"max_depth must be >= 1, got {self.max_depth}")
        if int(self.child_topk) < 1:
            raise ValueError(f"child_topk must be >= 1, got {self.child_topk}")
        policy = str(self.policy)
        if policy not in ("toll", "fixed"):
            raise ValueError(f"policy must be toll or fixed, got {policy}")
        if policy == "fixed":
            if self.fixed_width is None:
                raise ValueError("fixed policy requires fixed_width")
            if self.fixed_tau is None or not math.isfinite(float(self.fixed_tau)):
                raise ValueError("fixed policy requires a finite fixed_tau")
        if self.fixed_width is not None:
            width = int(self.fixed_width)
            if width not in widths or width not in verifies:
                raise ValueError(
                    f"fixed_width {width} must be on both width buckets {widths} "
                    f"and verify buckets {verifies}"
                )
        if not 0.0 < float(self.price_alpha) <= 1.0:
            raise ValueError(
                f"price_alpha must be in (0, 1], got {self.price_alpha}"
            )
        if int(self.warmup_rounds) < 0 or int(self.probe_interval) < 1:
            raise ValueError("warmup_rounds and probe_interval must be non-negative")
        mode = str(self.verify_mode).lower()
        if mode == "rpd":
            raise ValueError("TOLL_TREE_REMOTE rejects rpd; only greedy is supported")
        if mode not in ("greedy", "auto"):
            raise ValueError(f"verify_mode must be greedy or auto, got {mode}")
        if self.dp_attention:
            raise ValueError("TOLL_TREE_REMOTE rejects dp attention")
        if self.draft_on_npu:
            raise ValueError("TOLL_TREE_REMOTE rejects an npu draft")
        if self.fixed_k is not None and int(self.fixed_k) not in verifies:
            raise ValueError(
                f"fixed_k {self.fixed_k} is not on verify buckets {verifies}"
            )
        if float(self.speculate_delta) < 0 or float(self.bytes_per_token) < 0:
            raise ValueError("speculate_delta and bytes_per_token must be >= 0")
        object.__setattr__(self, "width_buckets", widths)
        object.__setattr__(self, "verify_buckets", verifies)
        object.__setattr__(self, "max_depth", int(self.max_depth))
        object.__setattr__(self, "child_topk", int(self.child_topk))
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "price_alpha", float(self.price_alpha))
        object.__setattr__(self, "warmup_rounds", int(self.warmup_rounds))
        object.__setattr__(self, "probe_interval", int(self.probe_interval))
        object.__setattr__(self, "verify_mode", mode)
        object.__setattr__(self, "speculate_delta", float(self.speculate_delta))
        object.__setattr__(self, "bytes_per_token", float(self.bytes_per_token))
        if self.fixed_width is not None:
            object.__setattr__(self, "fixed_width", int(self.fixed_width))
        if self.fixed_tau is not None:
            object.__setattr__(self, "fixed_tau", float(self.fixed_tau))
        if self.fixed_k is not None:
            object.__setattr__(self, "fixed_k", int(self.fixed_k))

    @classmethod
    def from_server_args(cls, args, role: str):
        """Build a config from one args object. Graphs are refused.

        This does not read process-wide state and does not start a server.
        """
        if role not in ("draft", "target"):
            raise ValueError(f"role must be draft or target, got {role}")
        if not bool(getattr(args, "disable_cuda_graph", False)):
            raise ValueError("TOLL_TREE_REMOTE refuses CUDA graph")
        if not bool(getattr(args, "disable_piecewise_cuda_graph", False)):
            raise ValueError("TOLL_TREE_REMOTE refuses piecewise CUDA graph")
        return cls()

    @property
    def w_max(self) -> int:
        return int(self.width_buckets[-1])

    @property
    def k_max(self) -> int:
        return int(self.verify_buckets[-1])
