"""Per-path layer cost and shared target / network cost.

Each series keeps a fixed window. ``observe_*`` only appends. ``refresh``
refits when a new ``probe_interval`` block has arrived, and ``snapshot``
reads that cache.
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
    ``w_t`` and ``kappa_t`` are the target intercept and prefix slope.
    A row slope is zero when that series never varied the row count.
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
    w_t: float = 0.0
    kappa_t: float = 0.0
    layer_rows_identified: bool = True
    target_rows_identified: bool = True


def _nonneg(value: float) -> float:
    return 0.0 if float(value) < 0.0 else float(value)


def _fit_hinge(
    rows: list[tuple[float, float, float]],
    sse_drop: float,
) -> tuple[float, float, float, float, int, bool]:
    """Fit ``y = w + kappa * prefix + eps * x + eps_plus * max(x - knee, 0)``.

    A column with one distinct value is dropped, and its slope is left
    unidentified, so a constant row count cannot be split into the intercept.
    Coefficients are clamped at 0. A knee is kept only when the row count
    varies, both row slopes are positive, and the squared error falls by at
    least ``sse_drop`` relative to the line.
    """
    empty = (0.0, 0.0, 0.0, 0.0, _NO_KNEE, False)
    if not rows:
        return empty
    prefix = np.asarray([item[0] for item in rows], dtype=np.float64)
    level = np.asarray([item[1] for item in rows], dtype=np.float64)
    target = np.asarray([item[2] for item in rows], dtype=np.float64)
    n = int(target.size)
    prefix_ok = n >= 2 and int(np.unique(prefix).size) >= 2
    rows_ok = n >= 2 and int(np.unique(level).size) >= 2
    columns = [np.ones(n)]
    if prefix_ok:
        columns.append(prefix)
    if rows_ok:
        columns.append(level)
    if n < len(columns):
        return empty
    base = np.column_stack(columns)
    coef, _, _, _ = np.linalg.lstsq(base, target, rcond=None)
    cursor = 0
    w = _nonneg(coef[cursor])
    cursor += 1
    kappa = 0.0
    if prefix_ok:
        kappa = _nonneg(coef[cursor])
        cursor += 1
    eps = 0.0
    if rows_ok:
        eps = _nonneg(coef[cursor])
    best = (w, kappa, eps, 0.0, _NO_KNEE, rows_ok)
    if not rows_ok or n < 3:
        return best
    base_sse = float(np.sum((base @ coef - target) ** 2))
    if base_sse <= 1e-6:
        return best
    level_index = 1 + int(prefix_ok)
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
        hinge_coef, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        sse = float(np.sum((design @ hinge_coef - target) ** 2))
        drop = (base_sse - sse) / base_sse
        if drop < float(sse_drop) or sse >= best_sse:
            continue
        if float(hinge_coef[level_index]) <= 0.0 or float(hinge_coef[-1]) <= 0.0:
            continue
        best_sse = sse
        fitted_kappa = _nonneg(hinge_coef[1]) if prefix_ok else 0.0
        best = (
            _nonneg(hinge_coef[0]),
            fitted_kappa,
            _nonneg(hinge_coef[level_index]),
            _nonneg(hinge_coef[-1]),
            int(knee),
            True,
        )
    return best


def _fit_line(pairs: list[tuple[float, float]]) -> tuple[float, float]:
    """Fit ``y = a + b * x`` and clamp both coefficients at 0."""
    if len(pairs) < 2:
        return 0.0, 0.0
    x = np.asarray([item[0] for item in pairs], dtype=np.float64)
    y = np.asarray([item[1] for item in pairs], dtype=np.float64)
    if int(np.unique(x).size) < 2:
        return _nonneg(float(np.mean(y))), 0.0
    design = np.column_stack([np.ones(len(pairs)), x])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return _nonneg(coef[0]), _nonneg(coef[1])


def _row_values(samples) -> int:
    if not samples:
        return 0
    return int(np.unique(np.asarray([item[1] for item in samples], dtype=np.float64)).size)


class TierCostModel:
    """Separate layer coefficients for each draft path.

    Target and network observations are shared. ``lookahead_cascade`` is the
    b=1 path, so its ``w`` is not continuous with ``batch`` at b=2.
    ``ready`` stays false until both this path and the target have enough
    samples and more than one row count. ``t_ar_s`` is one autoregressive
    token, in seconds. An unidentified row slope is left out of that time.
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
        self._layer_gen: dict[str, int] = {path: 0 for path in PATHS}
        self._layer_fit: dict[str, tuple] = {
            path: (0.0, 0.0, 0.0, 0.0, _NO_KNEE, False) for path in PATHS
        }
        self._target: deque = deque(maxlen=self.window)
        self._target_seen = 0
        self._target_gen = 0
        self._target_fit = (0.0, 0.0, 0.0, 0.0, _NO_KNEE, False)
        self._net: deque = deque(maxlen=self.window)
        self._net_seen = 0
        self._net_gen = 0
        self._net_fit = (0.0, 0.0)

    def ready(self, path: str) -> bool:
        if path not in self._layer:
            raise ValueError(f"unknown cost path {path}")
        if self._seen[path] < self.warmup_rounds or self._target_seen < self.warmup_rounds:
            return False
        return _row_values(self._layer[path]) >= 2 and _row_values(self._target) >= 2

    def observe_layer(self, path: str, prefix_sum: float, rows: float, ms: float) -> None:
        if path not in self._layer:
            raise ValueError(f"unknown cost path {path}")
        self._layer[path].append((float(prefix_sum), float(rows), float(ms)))
        self._seen[path] += 1

    def observe_target(self, prefix_sum: float, rows: float, ms: float) -> None:
        self._target.append((float(prefix_sum), float(rows), float(ms)))
        self._target_seen += 1

    def observe_net(self, num_bytes: float, ms: float) -> None:
        self._net.append((float(num_bytes), float(ms)))
        self._net_seen += 1

    def refresh(self) -> None:
        """Refit series that have filled another probe interval."""
        for path in PATHS:
            generation = self._seen[path] // self.probe_interval
            if generation > self._layer_gen[path]:
                self._layer_fit[path] = _fit_hinge(list(self._layer[path]), self.sse_drop)
                self._layer_gen[path] = generation
        generation = self._target_seen // self.probe_interval
        if generation > self._target_gen:
            self._target_fit = _fit_hinge(list(self._target), self.sse_drop)
            self._target_gen = generation
        generation = self._net_seen // self.probe_interval
        if generation > self._net_gen:
            self._net_fit = _fit_line(list(self._net))
            self._net_gen = generation

    def snapshot(self, path: str) -> CostPoint:
        """Return the last cached fit. This does not refit."""
        if path not in self._layer:
            raise ValueError(f"unknown cost path {path}")
        w, kappa, eps, eps_plus, row_knee, layer_rows = self._layer_fit[path]
        w_t, kappa_t, beta, beta_plus, node_knee, target_rows = self._target_fit
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
            w_t=w_t,
            kappa_t=kappa_t,
            layer_rows_identified=layer_rows,
            target_rows_identified=target_rows,
        )

    def t_ar_s(self, bs: int) -> float:
        """Seconds for one autoregressive token at batch size ``bs``.

        The row term is ``β·b`` milliseconds. An unidentified ``β`` is omitted.
        """
        if int(bs) < 1:
            return 0.0
        w_t, _kappa_t, beta, _beta_plus, _knee, rows_ok = self._target_fit
        ms = float(w_t)
        if rows_ok:
            ms += float(beta) * float(bs)
        return ms / 1000.0

    def lambda_prior(self, bs: int) -> float:
        """Tokens per second implied by ``t_ar_s``. Empty fits return 0."""
        elapsed = self.t_ar_s(bs)
        if elapsed <= 0.0 or int(bs) < 1:
            return 0.0
        return float(bs) / elapsed
