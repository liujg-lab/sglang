"""Per-path layer cost and shared target / network cost.

Forgetting factor is 1. ``snapshot`` refits the stored points, including a
hinge at the probed knee, which is the batch form of RLS on those points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PATHS = ("sync", "lookahead_cascade", "batch", "eager")
_NO_KNEE = 10**9


@dataclass(frozen=True)
class CostPoint:
    """One path's layer fit plus the shared target and network fit.

    ``ms_per_byte`` is the fitted network slope. Decision code treats
    ``beta + ms_per_byte * bytes_per_token`` as β̃ = β1 + s1/BW.
    """

    w: float
    kappa: float
    eps: float
    eps_plus: float
    row_knee: int
    beta: float
    beta_plus: float
    node_knee: int
    rtt_ms: float
    ms_per_byte: float


def _fit_hinge(rows: list[tuple[float, float, float]]) -> tuple[float, float, float, float, int]:
    """Fit ``y = w + kappa * prefix + eps * x + eps_plus * max(x - knee, 0)``.

    The no-hinge model is the default. A knee is kept only when it splits the
    sample and lowers the squared error. That least-squares fit is RLS with
    forgetting factor 1 on the stored points.
    """
    if len(rows) < 3:
        return 0.0, 0.0, 0.0, 0.0, _NO_KNEE
    prefix = np.asarray([r[0] for r in rows], dtype=np.float64)
    level = np.asarray([r[1] for r in rows], dtype=np.float64)
    target = np.asarray([r[2] for r in rows], dtype=np.float64)
    ones = np.ones(len(rows))
    base = np.column_stack([ones, prefix, level])
    base_coef, _, _, _ = np.linalg.lstsq(base, target, rcond=None)
    best_sse = float(np.sum((base @ base_coef - target) ** 2))
    best = (float(base_coef[0]), float(base_coef[1]), float(base_coef[2]), 0.0, _NO_KNEE)
    low = int(np.floor(float(level.min())))
    high = int(np.ceil(float(level.max())))
    if high - low > 64:
        knees = np.unique(np.linspace(low, high, 64).astype(int))
    else:
        knees = range(low, high)
    for knee in knees:
        above = level > float(knee)
        if not np.any(above) or not np.any(~above):
            continue
        hinge = np.maximum(level - float(knee), 0.0)
        design = np.column_stack([base, hinge])
        coef, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        sse = float(np.sum((design @ coef - target) ** 2))
        if sse < best_sse - 1e-8:
            best_sse = sse
            best = (
                float(coef[0]),
                float(coef[1]),
                float(coef[2]),
                float(coef[3]),
                int(knee),
            )
    return best


def _fit_line(pairs: list[tuple[float, float]]) -> tuple[float, float]:
    """Fit ``y = a + b * x``."""
    if len(pairs) < 2:
        return 0.0, 0.0
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    design = np.column_stack([np.ones(len(pairs)), x])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), float(coef[1])


class TierCostModel:
    """Separate layer coefficients for each draft path.

    Target and network observations are shared. ``lookahead_cascade`` is the
    b=1 path, so its ``w`` is not continuous with ``batch`` at b=2.
    """

    def __init__(self, probe_interval: int = 64) -> None:
        if int(probe_interval) < 1:
            raise ValueError("probe_interval must be >= 1")
        self.probe_interval = int(probe_interval)
        self._layer: dict[str, list[tuple[float, float, float]]] = {p: [] for p in PATHS}
        self._target: list[tuple[float, float, float]] = []
        self._net: list[tuple[float, float]] = []

    def observe_layer(self, path: str, prefix_sum: float, rows: float, ms: float) -> None:
        if path not in self._layer:
            raise ValueError(f"unknown cost path {path}")
        self._layer[path].append((float(prefix_sum), float(rows), float(ms)))

    def observe_target(self, prefix_sum: float, rows: float, ms: float) -> None:
        self._target.append((float(prefix_sum), float(rows), float(ms)))

    def observe_net(self, num_bytes: float, ms: float) -> None:
        self._net.append((float(num_bytes), float(ms)))

    def snapshot(self, path: str) -> CostPoint:
        if path not in self._layer:
            raise ValueError(f"unknown cost path {path}")
        w, kappa, eps, eps_plus, row_knee = _fit_hinge(self._layer[path])
        # Target fit: intercept is w_t, prefix slope is κ_t, row slope is β1.
        _w_t, _kappa_t, beta, beta_plus, node_knee = _fit_hinge(self._target)
        rtt_ms, ms_per_byte = _fit_line(self._net)
        return CostPoint(
            w=w,
            kappa=kappa,
            eps=eps,
            eps_plus=eps_plus,
            row_knee=row_knee,
            beta=beta,
            beta_plus=beta_plus,
            node_knee=node_knee,
            rtt_ms=rtt_ms,
            ms_per_byte=ms_per_byte,
        )
