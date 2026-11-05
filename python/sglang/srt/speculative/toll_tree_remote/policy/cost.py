"""Per-path layer cost and shared target / network cost.

Each series keeps a fixed window. A refit runs every ``probe_interval``
observations and is cached. ``snapshot`` only reads that cache.
"""

from __future__ import annotations

from collections import deque
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


def _nonneg(value: float) -> float:
    return 0.0 if float(value) < 0.0 else float(value)


def _fit_hinge(
    rows: list[tuple[float, float, float]],
    sse_drop: float,
) -> tuple[float, float, float, float, int]:
    """Fit ``y = w + kappa * prefix + eps * x + eps_plus * max(x - knee, 0)``.

    Coefficients are clamped at 0. A knee is kept only when it splits the
    sample, both the level slope and the hinge slope are positive, and the
    squared error falls by at least ``sse_drop`` relative to the line.
    """
    if len(rows) < 3:
        return 0.0, 0.0, 0.0, 0.0, _NO_KNEE
    prefix = np.asarray([r[0] for r in rows], dtype=np.float64)
    level = np.asarray([r[1] for r in rows], dtype=np.float64)
    target = np.asarray([r[2] for r in rows], dtype=np.float64)
    ones = np.ones(len(rows))
    base = np.column_stack([ones, prefix, level])
    base_coef, _, _, _ = np.linalg.lstsq(base, target, rcond=None)
    base_sse = float(np.sum((base @ base_coef - target) ** 2))
    best = (
        _nonneg(base_coef[0]),
        _nonneg(base_coef[1]),
        _nonneg(base_coef[2]),
        0.0,
        _NO_KNEE,
    )
    if base_sse <= 1e-6:
        return best
    low = int(np.floor(float(level.min())))
    high = int(np.ceil(float(level.max())))
    if high - low > 64:
        knees = np.unique(np.linspace(low, high, 64).astype(int))
    else:
        knees = range(low, high)
    best_sse = base_sse
    for knee in knees:
        above = level > float(knee)
        if not np.any(above) or not np.any(~above):
            continue
        hinge = np.maximum(level - float(knee), 0.0)
        design = np.column_stack([base, hinge])
        coef, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        sse = float(np.sum((design @ coef - target) ** 2))
        drop = (base_sse - sse) / base_sse
        if drop < float(sse_drop) or sse >= best_sse:
            continue
        if float(coef[2]) <= 0.0 or float(coef[3]) <= 0.0:
            continue
        best_sse = sse
        best = (
            _nonneg(coef[0]),
            _nonneg(coef[1]),
            _nonneg(coef[2]),
            _nonneg(coef[3]),
            int(knee),
        )
    return best


def _fit_line(pairs: list[tuple[float, float]]) -> tuple[float, float]:
    """Fit ``y = a + b * x`` and clamp both coefficients at 0."""
    if len(pairs) < 2:
        return 0.0, 0.0
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    design = np.column_stack([np.ones(len(pairs)), x])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return _nonneg(coef[0]), _nonneg(coef[1])


class TierCostModel:
    """Separate layer coefficients for each draft path.

    Target and network observations are shared. ``lookahead_cascade`` is the
    b=1 path, so its ``w`` is not continuous with ``batch`` at b=2.
    ``ready`` is false until that path has ``warmup_rounds`` samples. Until
    then the cached point is not a decision input.
    """

    def __init__(
        self,
        probe_interval: int = 64,
        warmup_rounds: int = 64,
        window: int = 256,
        sse_drop: float = 0.05,
    ) -> None:
        if int(probe_interval) < 1:
            raise ValueError("probe_interval must be >= 1")
        if int(warmup_rounds) < 0:
            raise ValueError("warmup_rounds must be >= 0")
        if int(window) < 3:
            raise ValueError("window must be >= 3")
        if float(sse_drop) < 0.0:
            raise ValueError("sse_drop must be >= 0")
        self.probe_interval = int(probe_interval)
        self.warmup_rounds = int(warmup_rounds)
        self.window = int(window)
        self.sse_drop = float(sse_drop)
        self._layer: dict[str, deque] = {
            path: deque(maxlen=self.window) for path in PATHS
        }
        self._seen: dict[str, int] = {path: 0 for path in PATHS}
        self._layer_fit: dict[str, tuple[float, float, float, float, int]] = {
            path: (0.0, 0.0, 0.0, 0.0, _NO_KNEE) for path in PATHS
        }
        self._target: deque = deque(maxlen=self.window)
        self._target_seen = 0
        self._target_fit = (0.0, 0.0, 0.0, 0.0, _NO_KNEE)
        self._net: deque = deque(maxlen=self.window)
        self._net_seen = 0
        self._net_fit = (0.0, 0.0)

    def ready(self, path: str) -> bool:
        if path not in self._layer:
            raise ValueError(f"unknown cost path {path}")
        return self._seen[path] >= self.warmup_rounds

    def observe_layer(self, path: str, prefix_sum: float, rows: float, ms: float) -> None:
        if path not in self._layer:
            raise ValueError(f"unknown cost path {path}")
        self._layer[path].append((float(prefix_sum), float(rows), float(ms)))
        self._seen[path] += 1
        if self._seen[path] % self.probe_interval == 0:
            self._layer_fit[path] = _fit_hinge(list(self._layer[path]), self.sse_drop)

    def observe_target(self, prefix_sum: float, rows: float, ms: float) -> None:
        self._target.append((float(prefix_sum), float(rows), float(ms)))
        self._target_seen += 1
        if self._target_seen % self.probe_interval == 0:
            self._target_fit = _fit_hinge(list(self._target), self.sse_drop)

    def observe_net(self, num_bytes: float, ms: float) -> None:
        self._net.append((float(num_bytes), float(ms)))
        self._net_seen += 1
        if self._net_seen % self.probe_interval == 0:
            self._net_fit = _fit_line(list(self._net))

    def snapshot(self, path: str) -> CostPoint:
        """Return the last cached fit. This does not refit."""
        if path not in self._layer:
            raise ValueError(f"unknown cost path {path}")
        w, kappa, eps, eps_plus, row_knee = self._layer_fit[path]
        _w_t, _kappa_t, beta, beta_plus, node_knee = self._target_fit
        rtt_ms, ms_per_byte = self._net_fit
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
