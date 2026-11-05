"""φ_d(x) with a flat prior of 1 and linear interpolation after observations."""

from __future__ import annotations

import numpy as np


class GainTable:
    """Expected extra factor φ at x = cut / π̂.

    Before any observation, φ is 1, so a node contributes its whole path mass.
    Observations are binned by x. ``estimate`` interpolates neighboring bins.
    """

    def __init__(self, bins: int = 5) -> None:
        if int(bins) < 2:
            raise ValueError("bins must be >= 2")
        self.bins = int(bins)
        self._sum: dict[tuple[int, int], float] = {}
        self._count: dict[tuple[int, int], int] = {}

    def observe(self, child_a: float, depth: int, x: float | None = None) -> None:
        if x is None:
            key = (int(depth), -1)
        else:
            clipped = float(np.clip(x, 0.0, 1.0))
            bucket = int(round(clipped * (self.bins - 1)))
            key = (int(depth), bucket)
        self._sum[key] = self._sum.get(key, 0.0) + float(child_a)
        self._count[key] = self._count.get(key, 0) + 1

    def estimate(self, pi: float, cut: float, depth: int) -> float:
        if float(pi) <= 0.0:
            return 0.0
        x = float(np.clip(float(cut) / float(pi), 0.0, 1.0))
        points = []
        for bucket in range(self.bins):
            key = (int(depth), bucket)
            count = self._count.get(key, 0)
            if count:
                center = bucket / (self.bins - 1)
                points.append((center, self._sum[key] / count))
        if not points:
            key = (int(depth), -1)
            count = self._count.get(key, 0)
            if count:
                return self._sum[key] / count
            return 1.0
        if x <= points[0][0]:
            return points[0][1]
        if x >= points[-1][0]:
            return points[-1][1]
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            if x0 <= x <= x1:
                if x1 == x0:
                    return y0
                weight = (x - x0) / (x1 - x0)
                return y0 + weight * (y1 - y0)
        return points[-1][1]
