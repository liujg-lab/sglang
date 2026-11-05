"""φ_d(x) = EMA of Σ(a − x)⁺ on a log grid. Empty bins stay at 0."""

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
    knots. A depth with no samples returns 0.
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
        values = np.array(
            [float(np.maximum(accepted - float(x), 0.0).sum()) for x in self.grid],
            dtype=np.float64,
        )
        key = int(depth)
        current = self._ema.get(key)
        if current is None:
            self._ema[key] = values
            return
        self._ema[key] = (1.0 - self.alpha) * current + self.alpha * values

    def estimate(self, pi: float, cut: float, depth: int) -> float:
        if float(pi) <= 0.0:
            return 0.0
        curve = self._ema.get(int(depth))
        if curve is None:
            return 0.0
        x = float(np.clip(float(cut) / float(pi), 0.0, 1.0))
        grid = self.grid
        if x <= float(grid[0]):
            return float(curve[0])
        if x >= float(grid[-1]):
            return float(curve[-1])
        hi = int(np.searchsorted(grid, x, side="left"))
        lo = hi - 1
        x0 = float(grid[lo])
        x1 = float(grid[hi])
        if x1 == x0:
            return float(curve[lo])
        weight = (x - x0) / (x1 - x0)
        return float(curve[lo] + weight * (curve[hi] - curve[lo]))
