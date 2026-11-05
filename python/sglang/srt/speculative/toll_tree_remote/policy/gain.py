"""φ_d(x) = EMA of Σ(a − x)⁺ on a log grid. An empty table stays at 0."""

from __future__ import annotations

import numpy as np

# Positive knots are logarithmic. 0 is the endpoint Σ(a − 0)⁺.
# 0.014 and 0.1 are distinct knots; a linear 5-bin grid would merge them.
_DEFAULT_GRID = np.unique(
    np.concatenate(
        [
            np.array([0.0, 0.014, 0.1], dtype=np.float64),
            np.geomspace(1e-3, 1.0, 12),
        ]
    )
)


class GainTable:
    """Expected extra factor φ at x = cut / π̂.

    ``observe`` takes one parent's sibling acceptance values and updates
    Σ(a − x)⁺ at every grid knot. The first sample at a depth writes the
    value; later samples are an EMA. ``estimate`` interpolates neighboring
    knots. A missing depth uses the nearest shallower curve, then the nearest
    observed depth. The table returns 0 only when it has no samples at all.
    """

    def __init__(self, alpha: float = 0.02, grid=None) -> None:
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        knots = _DEFAULT_GRID if grid is None else np.asarray(grid, dtype=np.float64)
        knots = np.unique(knots.reshape(-1))
        if knots.size < 2 or np.any(knots < 0.0) or np.any(knots > 1.0):
            raise ValueError("phi grid must lie in [0, 1] and have at least 2 knots")
        self.alpha = float(alpha)
        self.grid = knots
        self._ema: dict[int, np.ndarray] = {}

    def bin_index(self, x: float) -> int:
        """Knot index of ``x``. 0.014 and 0.1 land on different knots."""
        clipped = float(np.clip(x, 0.0, 1.0))
        idx = int(np.searchsorted(self.grid, clipped, side="left"))
        return int(np.clip(idx, 0, int(self.grid.size) - 1))

    def observe(self, siblings, depth: int) -> None:
        accepted = np.asarray(siblings, dtype=np.float64).reshape(-1)
        if accepted.size == 0:
            values = np.zeros(self.grid.size, dtype=np.float64)
        else:
            values = np.maximum(accepted[:, None] - self.grid[None, :], 0.0).sum(axis=0)
        key = int(depth)
        current = self._ema.get(key)
        if current is None:
            self._ema[key] = values
            return
        self._ema[key] = (1.0 - self.alpha) * current + self.alpha * values

    def _curve_for(self, depth: int):
        key = int(depth)
        curve = self._ema.get(key)
        if curve is not None:
            return curve
        if not self._ema:
            return None
        shallower = [item for item in self._ema if item < key]
        if shallower:
            return self._ema[max(shallower)]
        nearest = min(self._ema, key=lambda item: (abs(item - key), item))
        return self._ema[nearest]

    def _interp(self, curve: np.ndarray, x: np.ndarray) -> np.ndarray:
        grid = self.grid
        y = np.empty(x.shape, dtype=np.float64)
        left = x <= float(grid[0])
        right = x >= float(grid[-1])
        middle = ~left & ~right
        y[left] = float(curve[0])
        y[right] = float(curve[-1])
        if np.any(middle):
            xm = x[middle]
            hi = np.clip(np.searchsorted(grid, xm, side="left"), 1, int(grid.size) - 1)
            lo = hi - 1
            x0 = grid[lo]
            x1 = grid[hi]
            span = x1 - x0
            weight = np.divide(xm - x0, span, out=np.zeros_like(xm), where=span != 0.0)
            y[middle] = curve[lo] + weight * (curve[hi] - curve[lo])
        return y

    def estimate_many(self, pis, cut: float, depth: int) -> np.ndarray:
        values = np.asarray(pis, dtype=np.float64).reshape(-1)
        out = np.zeros(values.size, dtype=np.float64)
        curve = self._curve_for(depth)
        if curve is None:
            return out
        positive = values > 0.0
        if not np.any(positive):
            return out
        ratio = np.zeros(values.size, dtype=np.float64)
        ratio[positive] = np.clip(float(cut) / values[positive], 0.0, 1.0)
        out[positive] = self._interp(curve, ratio[positive])
        return out

    def estimate(self, pi: float, cut: float, depth: int) -> float:
        return float(self.estimate_many([pi], cut, depth)[0])
