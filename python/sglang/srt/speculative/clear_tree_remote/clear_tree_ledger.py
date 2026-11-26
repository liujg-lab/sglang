"""Shadow cost ledger for CLEAR_TREE_REMOTE.

Records the time ledger and layer forwards, and fits layer and verify cost.
Draft uses the layer fit when it stops a tree. Online K reads ``T_K`` written
at round close, not the device samples Target sends back.

``SGLANG_CLEAR_LEDGER=1`` on the Draft turns the online K and W controllers
on. The Target ledger only logs that process. Default is off. Only tp rank 0
logs and fits. With the switch off, every round stays on tau and the flat
verify count: 16 for one request, 8 otherwise.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from functools import wraps
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

FORGET = 0.999
RIDGE = 1e-6
LAYER_MIN_SAMPLES = 4
LAYER_SPAN_PAD = 128.0
# Per-depth correction of the layer regression. Depth is the ``layer`` index
# of the forward (1-based); deeper forwards share the last bucket.
DEPTH_CAP = 8
DEPTH_MIN_SAMPLES = 30
# Intercept and prefix sum. The prefix prior is one token's 2 µs scaled by
# 128, so a flat prefix cannot set the slope. The capture batch is the cell
# key, not a feature.
LAYER_RIDGE = (0.0, LAYER_SPAN_PAD**2)
_PREFIX_FEATURE = 1
# Smallest centered prefix information, relative to Σx², that counts as spread.
_SLOPE_PRIOR_REL_EPS = 1e-9
VERIFY_MIN_SAMPLES = 6
_DEFAULT_BUCKETS = (256, 512, 1024, 2048, 4096)
_GAP_ACTIONS = ("prefill", "finish", "abort")


def ledger_enabled() -> bool:
    value = os.environ.get("SGLANG_CLEAR_LEDGER", "")
    return value.strip().lower() in {"1", "true", "yes", "y"}


def target_backend() -> str:
    value = os.environ.get("SGLANG_CLEAR_TARGET_BACKEND", "cuda").strip().lower()
    return value or "cuda"


def configured_buckets() -> tuple:
    raw = os.environ.get("SGLANG_CLEAR_NPU_KV_BUCKETS", "")
    if not raw.strip():
        return _DEFAULT_BUCKETS
    buckets = []
    for piece in raw.split(","):
        text = piece.strip()
        if not text:
            continue
        buckets.append(int(text))
    return tuple(sorted(set(buckets)))


def graph_bucket(active: int) -> int:
    """Smallest power of two that can hold ``active`` requests."""
    size = max(int(active), 1)
    bucket = 1
    while bucket < size:
        bucket *= 2
    return bucket


def layer_graph_bs(scheduler, active: int) -> int:
    """Capture batch a layer replay would use, else a power of two.

    The list is cached on the scheduler after CUDA-graph capture. With
    padding, this is the smallest captured batch that holds ``active``.
    Without padding, only an exact captured batch counts. A missing list, or
    no batch that can hold ``active``, falls back to ``graph_bucket``.
    """
    size = max(int(active), 1)
    capture = (
        None
        if scheduler is None
        else getattr(scheduler, "_clear_layer_capture_bs", None)
    )
    if not capture:
        return graph_bucket(size)
    buckets = sorted(int(item) for item in capture)
    if bool(getattr(scheduler, "_clear_layer_disable_padding", False)):
        if size in buckets:
            return size
        return graph_bucket(size)
    for bucket in buckets:
        if size <= bucket:
            return bucket
    return graph_bucket(size)


def smallest_bucket(needed: int, buckets: Sequence[int]) -> Optional[int]:
    """Smallest bucket that can hold ``needed``, or None when all are too small."""
    for bucket in buckets:
        if int(needed) <= int(bucket):
            return int(bucket)
    return None


def _scan_layer_models(models, width: int, g: int, *, ready: bool):
    """Graph-path fit for this width and capture batch, else another path.

    ``ready`` keeps only a fit that can already be solved. The sample count
    used to decide a narrow probe still sees a short graph fit, and does not
    fall through to an eager cell at the same capture batch.
    """
    preferred = None
    fallback = None
    for key, model in models.items():
        if len(key) < 3:
            continue
        if int(key[0]) != int(width) or int(key[2]) != int(g):
            continue
        if ready and (model.n < LAYER_MIN_SAMPLES or model.coefficients() is None):
            continue
        if str(key[1]) == "graph":
            preferred = model
            break
        if fallback is None:
            fallback = model
    if preferred is not None:
        return preferred
    return fallback


def _layer_model(scheduler, width: int, g: int):
    """Fitted layer model for this width and capture batch.

    A graph-path fit wins over eager.
    """
    ledger = bind_ledger(scheduler)
    if ledger is None:
        return None
    return _scan_layer_models(ledger.layer_models, width, g, ready=True)


def layer_sample_count(scheduler, width: int, active: int) -> int:
    """Samples in the model ``predict_layer_ms`` will use, even below four.

    ``active`` selects the capture batch. A missing ledger counts as zero.
    This does not create one.
    """
    existing = None if scheduler is None else getattr(scheduler, "_clear_ledger", None)
    if not existing:
        return 0
    g = layer_graph_bs(scheduler, active)
    model = _scan_layer_models(existing.layer_models, width, g, ready=False)
    if model is None:
        return 0
    return int(model.n)


def _layer_features(sum_ell: int) -> List[float]:
    return [1.0, float(sum_ell)]


def layer_depth_enabled() -> bool:
    """Depth correction in layer prices. ``SGLANG_CLEAR_LAYER_DEPTH=0`` keeps learning it but not using it."""
    return os.environ.get("SGLANG_CLEAR_LAYER_DEPTH", "").strip() != "0"


def _depth_key(depth) -> int:
    return max(1, min(int(depth), DEPTH_CAP))


def layer_depth_ratio(ledger, width: int, g: int, depth) -> tuple:
    """``(ratio, source)`` for a forward at this depth.

    The ratio is forgetting-weighted measured wall over the base regression,
    taken over middle pre-reply graph layers. The cell ``(w, g, d)`` is used
    once it has ``DEPTH_MIN_SAMPLES``; else the pooled ``(g, d)`` across
    widths; else 1.
    """
    if ledger is None or depth is None:
        return 1.0, "none"
    d = _depth_key(depth)
    for table, key, source in (
        (ledger.layer_depth, (int(width), int(g), d), "cell"),
        (ledger.layer_depth_pooled, (int(g), d), "pooled"),
    ):
        row = table.get(key)
        if row is not None and row[2] >= DEPTH_MIN_SAMPLES and row[1] > 0.0:
            return float(row[0]) / float(row[1]), source
    return 1.0, "none"


def _depth_factor(scheduler, width: int, g: int, depth) -> float:
    if depth is None or not layer_depth_enabled():
        return 1.0
    ledger = bind_ledger(scheduler)
    ratio, _source = layer_depth_ratio(ledger, width, g, depth)
    return float(ratio)


def layer_depth_factor(scheduler, width: int, active: int, depth) -> float:
    """The multiplier ``predict_layer_ms`` applies at this depth, 1 when unused."""
    if depth is None:
        return 1.0
    return _depth_factor(scheduler, width, layer_graph_bs(scheduler, active), depth)


def predict_layer_ms(
    scheduler, width: int, active: int, sum_ell: int, depth=None
) -> Optional[float]:
    """Draft layer fit, or None when the prefix is outside the seen span.

    The cell is this width and the capture batch of ``active``. A missing
    cell, a prefix outside that cell, or a non-positive prediction is None.
    The served width then stays on the flat K, and a replay candidate with
    this result does not pair. ``depth`` is the forward's layer index; with
    it the base fit is scaled by ``layer_depth_ratio``.
    """
    g = layer_graph_bs(scheduler, active)
    model = _layer_model(scheduler, width, g)
    if model is None:
        return None
    features = _layer_features(sum_ell)
    if not model.span_allows(features):
        return None
    pred = model.predict(features)
    if pred is None or float(pred) <= 0.0:
        return None
    return float(pred) * _depth_factor(scheduler, width, g, depth)


def layer_sum_ell_slope(
    scheduler, width: int, active: int, depth=None
) -> Optional[float]:
    """Prefix slope for this width and the capture batch of ``active``.

    None until that cell can be solved. ``depth`` scales it like the price.
    """
    g = layer_graph_bs(scheduler, active)
    model = _layer_model(scheduler, width, g)
    if model is None:
        return None
    beta = model.coefficients()
    if beta is None or len(beta) < 2:
        return None
    return float(beta[1]) * _depth_factor(scheduler, width, g, depth)


def _action_name(action) -> str:
    value = getattr(action, "value", action)
    return str(value)


def _solve(matrix: List[List[float]], rhs: List[float]) -> Optional[List[float]]:
    """Gaussian elimination with partial pivoting. None when the system is singular."""
    size = len(rhs)
    rows = [list(matrix[i]) + [rhs[i]] for i in range(size)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(rows[row][col]))
        if abs(rows[pivot][col]) < 1e-12:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        scale = rows[col][col]
        for col_j in range(col, size + 1):
            rows[col][col_j] /= scale
        for row in range(size):
            if row == col:
                continue
            factor = rows[row][col]
            if factor == 0.0:
                continue
            for col_j in range(col, size + 1):
                rows[row][col_j] -= factor * rows[col][col_j]
    return [rows[i][size] for i in range(size)]


class ForgettingRegressor:
    """One forgetting least-squares fit.

    Without ``ridge``, slopes have a tiny zero prior and the intercept does
    not. A ridge sequence is added on those diagonals, including a zero
    intercept entry. ``span_feature`` records the min and max of that input.
    """

    def __init__(
        self,
        n_features: int,
        min_samples: int,
        ridge: Optional[Sequence[float]] = None,
        span_feature: Optional[int] = None,
    ) -> None:
        self.n_features = int(n_features)
        self.min_samples = int(min_samples)
        self.ridge = None if ridge is None else tuple(float(value) for value in ridge)
        self.span_feature = None if span_feature is None else int(span_feature)
        self.span_min: Optional[float] = None
        self.span_max: Optional[float] = None
        self.n = 0
        self.xtx = [[0.0] * self.n_features for _ in range(self.n_features)]
        self.xty = [0.0] * self.n_features
        # ``(index, target, precision)``. Never forgotten; added on top of the
        # ridge. Not part of ``to_dict``.
        self.slope_prior: Optional[tuple] = None
        self._coef: Optional[List[float]] = None
        self._coef_n = -1

    def set_slope_prior(self, index: int, target: float, precision: float) -> None:
        """Pull weight ``index`` toward ``target`` with a fixed ``precision``.

        The prior does not decay with ``FORGET`` and sits on top of the ridge.
        When ``target`` is this fit's own solved weight, the solution does not
        move at the moment the prior is set.
        """
        self.slope_prior = (int(index), float(target), float(precision))
        self._coef = None
        self._coef_n = -1

    def update(self, features: Sequence[float], target: float) -> None:
        forget = FORGET
        width = self.n_features
        for i in range(width):
            self.xty[i] *= forget
            for j in range(width):
                self.xtx[i][j] *= forget
        for i in range(width):
            self.xty[i] += float(features[i]) * float(target)
            for j in range(width):
                self.xtx[i][j] += float(features[i]) * float(features[j])
        self.n += 1
        self._coef = None
        if self.span_feature is None:
            return
        value = float(features[self.span_feature])
        if self.span_min is None or self.span_max is None:
            self.span_min = value
            self.span_max = value
            return
        self.span_min = min(self.span_min, value)
        self.span_max = max(self.span_max, value)

    def coefficients(self) -> Optional[List[float]]:
        """Solved weights. The intercept is first. None until the fit is usable."""
        if self.n < self.min_samples:
            return None
        if self._coef is not None and self._coef_n == self.n:
            return self._coef
        system = [row[:] for row in self.xtx]
        rhs = list(self.xty)
        if self.ridge is None:
            for index in range(1, self.n_features):
                system[index][index] += RIDGE
        else:
            for index, value in enumerate(self.ridge):
                if index >= self.n_features:
                    break
                system[index][index] += float(value)
        prior = self.slope_prior
        if prior is not None and 0 <= int(prior[0]) < self.n_features:
            index = int(prior[0])
            system[index][index] += float(prior[2])
            rhs[index] += float(prior[2]) * float(prior[1])
        self._coef = _solve(system, rhs)
        self._coef_n = self.n
        return self._coef

    def to_dict(self) -> dict:
        """Numeric state. The next predict solves ``_coef`` again."""
        return {
            "n": int(self.n),
            "xtx": [list(row) for row in self.xtx],
            "xty": [float(value) for value in self.xty],
            "span_min": self.span_min,
            "span_max": self.span_max,
        }

    def from_dict(self, data: dict) -> None:
        """Load a snapshot. ``_coef`` stays empty until the next solve."""
        xtx = data.get("xtx")
        xty = data.get("xty")
        if (
            not isinstance(xtx, list)
            or len(xtx) != self.n_features
            or any(
                not isinstance(row, list) or len(row) != self.n_features for row in xtx
            )
        ):
            raise ValueError("regressor xtx does not match n_features")
        if not isinstance(xty, list) or len(xty) != self.n_features:
            raise ValueError("regressor xty does not match n_features")
        self.n = int(data.get("n", 0))
        self.xtx = [[float(value) for value in row] for row in xtx]
        self.xty = [float(value) for value in xty]
        span_min = data.get("span_min")
        span_max = data.get("span_max")
        self.span_min = None if span_min is None else float(span_min)
        self.span_max = None if span_max is None else float(span_max)
        self._coef = None
        self._coef_n = -1

    def span_allows(self, features: Sequence[float]) -> bool:
        """False when ``span_feature`` is past the seen range by the margin.

        The margin is the larger of 128 and the span of samples already seen.
        A regressor that does not track a feature always allows the query.
        """
        if self.span_feature is None or self.span_min is None or self.span_max is None:
            return True
        value = float(features[self.span_feature])
        span = float(self.span_max) - float(self.span_min)
        margin = max(LAYER_SPAN_PAD, span)
        low = float(self.span_min) - margin
        high = float(self.span_max) + margin
        return low <= value <= high

    def predict(self, features: Sequence[float]) -> Optional[float]:
        beta = self.coefficients()
        if beta is None:
            return None
        return sum(weight * float(value) for weight, value in zip(beta, features))


def _new_layer_model() -> ForgettingRegressor:
    return ForgettingRegressor(
        2,
        LAYER_MIN_SAMPLES,
        ridge=LAYER_RIDGE,
        span_feature=_PREFIX_FEATURE,
    )


def layer_slope_prior_enabled() -> bool:
    """Prefit slope prior on layer fits. ``SGLANG_CLEAR_LAYER_SLOPE_PRIOR=0`` turns it off."""
    return os.environ.get("SGLANG_CLEAR_LAYER_SLOPE_PRIOR", "").strip() != "0"


def _prefit_slope_prior(model) -> Optional[tuple]:
    """``(slope, precision)`` from a fit's own matrices, or None.

    The precision is the centered prefix information
    ``xtx11 - xtx01**2 / xtx00``, so the prefit level does not enter it.
    """
    index = _PREFIX_FEATURE
    if int(getattr(model, "n_features", 0)) <= index:
        return None
    beta = model.coefficients()
    if beta is None or len(beta) <= index:
        return None
    xtx = model.xtx
    weight = float(xtx[0][0])
    if not math.isfinite(weight) or weight <= 0.0:
        return None
    second = float(xtx[index][index])
    precision = second - float(xtx[0][index]) ** 2 / weight
    slope = float(beta[index])
    if not (math.isfinite(precision) and math.isfinite(slope)):
        return None
    # One repeated prefix leaves only rounding noise here.
    if precision <= _SLOPE_PRIOR_REL_EPS * max(abs(second), 1.0):
        return None
    return slope, precision


def attach_layer_slope_priors(models) -> int:
    """Freeze each installed fit's prefix slope as a prior that is never forgotten.

    Online samples keep moving the intercept and, with enough spread, the
    slope. Returns how many cells got a prior; 0 when the switch is off.
    """
    if not layer_slope_prior_enabled():
        return 0
    attached = 0
    for model in list((models or {}).values()):
        if getattr(model, "slope_prior", None) is not None:
            # Already frozen; solving again would read the prior back.
            attached += 1
            continue
        found = _prefit_slope_prior(model)
        if found is None:
            continue
        slope, precision = found
        model.set_slope_prior(_PREFIX_FEATURE, slope, precision)
        attached += 1
    return attached


class _Round:
    __slots__ = (
        "session_id",
        "rpc_seq",
        "b",
        "k",
        "w",
        "rids",
        "prefix_lens",
        "idle_before",
        "gap_actions",
        "gap_rpcs",
        "policy_ms",
        "policy_recv_ms",
        "policy_plan_ms",
        "policy_grow_ms",
        "policy_final_k_ms",
        "post_ms",
        "t_open",
        "t_stop",
        "t_send",
        "t_post",
        "draft_prefill_us",
        "bonus_hit_rids",
        "layers",
        "fallback",
        "row_short_rids",
        "prefit",
        "reply_bytes",
        "hold_ms",
        "e_rows",
        "e_detail",
        "tree_depth",
    )

    def __init__(
        self,
        session_id: str,
        rpc_seq: int,
        rids: Sequence[str],
        prefix_lens: Sequence[int],
        width: int,
        idle_before: bool,
        opened_at: float,
    ) -> None:
        self.session_id = session_id
        self.rpc_seq = int(rpc_seq)
        self.rids = [str(rid) for rid in rids]
        self.prefix_lens = [int(length) for length in prefix_lens]
        self.b = len(self.rids)
        self.k = int(width)
        self.w = int(width)
        self.idle_before = bool(idle_before)
        self.gap_actions: List[str] = []
        self.gap_rpcs: List[int] = []
        self.policy_ms = 0.0
        self.policy_recv_ms = 0.0
        self.policy_plan_ms = 0.0
        self.policy_grow_ms = 0.0
        self.policy_final_k_ms = 0.0
        self.post_ms = 0.0
        self.t_open = float(opened_at)
        self.t_stop: Optional[float] = None
        self.t_send: Optional[float] = None
        self.t_post: Optional[float] = None
        self.draft_prefill_us = 0
        self.bonus_hit_rids: List[str] = []
        self.layers: List[dict] = []
        self.fallback = False
        # Rids past their Draft req_to_token row: in the Target batch, not drafted.
        self.row_short_rids: set = set()
        self.prefit = None
        self.reply_bytes = None
        self.hold_ms = None
        # rid -> (accept ratio, path mass) at the verified K, for diagnostics.
        self.e_rows: Dict[str, tuple] = {}
        # rid -> references from ``RoundPlan.note`` for the accepted-path
        # ranks and per-tier mass, computed at close.
        self.e_detail: Dict[str, dict] = {}
        self.tree_depth = 0


def _round_price(rnd: _Round, t_recv: float, next_rids, next_prefixes):
    """Tokens and milliseconds from open to the next STEP, minus draft prefill."""
    if rnd.t_stop is None:
        return None, None
    prefill_s = float(rnd.draft_prefill_us) / 1e6
    ms = (float(t_recv) - float(rnd.t_open) - prefill_s) * 1000.0
    if ms <= 0:
        return None, None
    new_prefix = {
        str(rid): int(prefix) for rid, prefix in zip(next_rids, next_prefixes)
    }
    tokens = 0
    for rid, old in zip(rnd.rids, rnd.prefix_lens):
        if str(rid) in new_prefix:
            tokens += max(0, new_prefix[str(rid)] - int(old))
    return tokens, ms


# A tail this long covers the deepest verify window plus the bonus token.
COMMIT_TAIL = 66

# Closed rounds kept for a verify report that arrives after their close. The
# report normally rides on the STEP after next, one close later; a few spare
# rounds cover FINISH/PREFILL gaps between them.
LATE_RPD_ROUNDS = 8


def _commit_tail(req) -> Optional[List[int]]:
    """The last committed token ids of one STEP request, or None.

    A delta ends at ``base_committed_len`` (the Target sends
    ``output_ids[acked:]``); a snapshot's ``committed_ids`` is the whole
    output, which ends there too. Only the tail is kept.
    """
    mode = getattr(req, "commit_mode", None)
    if mode == "delta":
        ids = getattr(req, "delta_ids", None)
    elif mode == "snapshot":
        ids = getattr(req, "committed_ids", None)
    else:
        return None
    if not isinstance(ids, (list, tuple)):
        return None
    try:
        return [int(token) for token in ids[-COMMIT_TAIL:]]
    except (TypeError, ValueError):
        return None


def _node_ranks(scores: Sequence[float], width: int) -> Dict[int, int]:
    """1-based rank of each flat node in the verify order.

    Same key as ``_ranked_verify_indices``: score desc, depth asc, column asc.
    """
    order = sorted(
        range(len(scores)),
        key=lambda index: (-float(scores[index]), index // width, index % width),
    )
    return {index: rank for rank, index in enumerate(order, start=1)}


def _window_of(detail: dict):
    """The verify window as host lists, padding stripped, or None."""
    from sglang.srt.speculative.clear_tree_remote.clear_tree_lease import (
        strip_verify_padding,
    )

    verify = detail.get("verify")
    if verify is None:
        return None
    window = strip_verify_padding(
        verify.tokens.tolist(), verify.parents.tolist(), verify.cols.tolist()
    )
    if window is None:
        return None
    tokens, parents, cols = window
    depths = [int(x) for x in verify.depths.tolist()[: len(tokens)]]
    return tokens, parents, depths, cols


def _accept_walk(detail: dict, path: Sequence[int], bonus: Optional[int]) -> dict:
    """Ranks of the accepted window path, then one step into the whole tree.

    ``ranks`` is None when a committed token has no matching window child.
    ``next_rank`` is the whole-tree rank of the child of the last accepted
    node that carries the bonus token (the Target's own next token), and
    ``next_rank_short`` its rank in the tree without the last layer. Both
    are None when no such node exists. The node is outside the window, or
    the Target would have accepted it.
    """
    out = {"ranks": None, "next_rank": None, "next_rank_short": None}
    window = _window_of(detail)
    width = int(detail["width"])
    if window is None or width <= 0:
        return out
    tokens, parents, depths, cols = window
    scores = detail["scores"]
    ranks_of = _node_ranks(scores, width)
    ranks: List[int] = []
    parent = -1
    last_col = None
    for token in path:
        child = None
        for index, (node_parent, node_token) in enumerate(zip(parents, tokens)):
            if int(node_parent) == parent and int(node_token) == int(token):
                child = index
                break
        if child is None or int(depths[child]) != len(ranks):
            return out
        flat = int(depths[child]) * width + int(cols[child])
        if flat not in ranks_of:
            return out
        ranks.append(int(ranks_of[flat]))
        parent = child
        last_col = int(cols[child])
    out["ranks"] = ranks
    tree = detail.get("tree")
    depth = int(detail["depth"])
    row = len(ranks)
    if bonus is None or tree is None or row >= depth:
        return out
    tree_tokens, tree_parents = tree
    if row >= len(tree_tokens):
        return out
    for col in range(width):
        if int(tree_tokens[row][col]) != int(bonus):
            continue
        if row > 0 and int(tree_parents[row][col]) != int(last_col):
            continue
        flat = row * width + col
        out["next_rank"] = int(ranks_of[flat])
        if row < depth - 1:
            short = _node_ranks(scores[: (depth - 1) * width], width)
            out["next_rank_short"] = int(short[flat])
        break
    return out


def _rpd_accepts(detail: dict, mask) -> Optional[dict]:
    """RPD accepts for every K tier from the Target's acceptable-slot mask.

    Bit ``j`` of ``mask`` is window node ``j`` (Target slot ``j + 1``). A node
    is accepted when it and every ancestor are acceptable, so the accepts of
    the top-K' window are the deepest reachable node of rank <= K', plus one.
    The window is the top ``n`` ranks and closed under ancestors, so this is
    exact for K' <= n. Past ``n`` it is exact only when the window is the
    whole tree: the window is the top ``min(K, depth * width)`` nodes, so it
    is whole when ``n`` covers every flat score. That includes a tree of
    exactly K nodes. Otherwise it is None.
    """
    from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
        VERIFY_KS,
    )

    window = _window_of(detail)
    width = int(detail["width"])
    if window is None or width <= 0:
        return None
    try:
        bits = int(str(mask), 16)
    except ValueError:
        return None
    tokens, parents, depths, cols = window
    n = len(tokens)
    ranks_of = _node_ranks(detail["scores"], width)
    reach = [False] * n
    reached = []
    valid = 0
    for j in range(n):
        ok = bool((bits >> j) & 1)
        valid += int(ok)
        parent = int(parents[j])
        if parent >= j:
            return None
        reach[j] = ok and (parent < 0 or reach[parent])
        if reach[j]:
            flat = int(depths[j]) * width + int(cols[j])
            if flat not in ranks_of:
                return None
            reached.append((int(ranks_of[flat]), int(depths[j]) + 1))

    def accepts(limit: int) -> int:
        return max((length for rank, length in reached if rank <= limit), default=0)

    whole = n >= len(detail["scores"])
    accept_k = {}
    for tier in VERIFY_KS:
        tier = int(tier)
        if tier <= n:
            accept_k[str(tier)] = accepts(tier)
        elif whole:
            accept_k[str(tier)] = accepts(n)
        else:
            accept_k[str(tier)] = None
    return {"accept_k": accept_k, "rpd_valid": valid, "rpd_full": accepts(n)}


def _rid_detail(detail: dict, accepted: int, tail, mask=None) -> dict:
    """Per-request depth, accepts, accepted-path ranks and mass per tier.

    With an RPD ``mask``, also the RPD accepts per K tier; ``rpd_ok`` says
    the whole-window value matches the realized accepts.
    """
    from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
        VERIFY_KS,
    )

    ordered = detail["ordered"]
    mass_k = {}
    running = 0.0
    prefix = []
    for value in ordered:
        running += float(value)
        prefix.append(running)
    for tier in VERIFY_KS:
        mass_k[str(int(tier))] = (
            prefix[min(int(tier), len(prefix)) - 1] if prefix else 0.0
        )
    walk = {"ranks": None, "next_rank": None, "next_rank_short": None}
    grown = int(accepted) + 1
    if tail is not None and len(tail) >= grown:
        committed = list(tail)[-grown:]
        try:
            walk = _accept_walk(detail, committed[: int(accepted)], committed[-1])
        except (KeyError, TypeError, ValueError, IndexError, AttributeError):
            pass
    if walk["ranks"] is None:
        walk["next_rank"] = None
        walk["next_rank_short"] = None
    out = {
        "depth": int(detail["depth"]),
        "accepted": int(accepted),
        "ranks": walk["ranks"],
        "path_ok": walk["ranks"] is not None,
        "next_rank": walk["next_rank"],
        "next_rank_short": walk["next_rank_short"],
        "mass_k": mass_k,
    }
    if mask is not None:
        out.update(_mask_fields(detail, accepted, mask))
    return out


def _mask_fields(detail: dict, accepted: int, mask) -> dict:
    """``accept_k``, ``rpd_valid`` and ``rpd_ok`` of one request's mask.

    A bad window gives ``accept_k=None`` and ``rpd_ok=False`` instead of
    raising, so it never drops the row's other diagnostics.
    """
    try:
        rpd = _rpd_accepts(detail, mask)
    except (KeyError, TypeError, ValueError, IndexError, AttributeError):
        rpd = None
    if rpd is None:
        return {"accept_k": None, "rpd_valid": None, "rpd_ok": False}
    return {
        "accept_k": rpd["accept_k"],
        "rpd_valid": rpd["rpd_valid"],
        "rpd_ok": int(rpd["rpd_full"]) == int(accepted),
    }


def _round_expect(
    rnd: _Round, next_rids, next_prefixes, next_tails=None, rpd=None, late=None
) -> dict:
    """Realized accepts against c_i times path mass, over requests that stayed.

    ``accepted`` drops each request's bonus token, as ``settle_verified`` does.
    Every field is None when the round noted no tree. ``e_rid``,
    ``verify_quotes`` and ``theta`` are diagnostics for judging K and the
    stop afterwards; nothing reads them back. ``rpd`` is the matched report's
    ``rid -> hex`` masks; ``rpd_masks`` counts the rids whose mask gave
    per-tier accepts. When ``late`` is a dict, it receives
    ``rid -> (detail, accepted, mass_k)`` for every row with a tree, so masks
    that arrive after this close can still be priced.
    """
    empty = {
        "accepted": None,
        "e_pred": None,
        "e_mass": None,
        "e_n": None,
        "e_rid": None,
        "k_mode": None,
        "k_cover": None,
        "verify_quotes": None,
        "theta": None,
        "rpd_masks": None,
        "tree_depth": int(getattr(rnd, "tree_depth", 0) or 0) or None,
    }
    rows = getattr(rnd, "e_rows", None) or {}
    if not rows:
        return empty
    details = getattr(rnd, "e_detail", None) or {}
    tails = next_tails or {}
    found = {str(rid): int(prefix) for rid, prefix in zip(next_rids, next_prefixes)}
    old = {str(rid): int(prefix) for rid, prefix in zip(rnd.rids, rnd.prefix_lens)}
    accepted = 0
    e_pred = 0.0
    e_mass = 0.0
    count = 0
    per_rid = {}
    round_detail = None
    masks = rpd if isinstance(rpd, dict) else {}
    used_masks = 0
    for rid, (ratio, mass) in rows.items():
        if rid not in found or rid not in old:
            continue
        taken = max(int(found[rid]) - int(old[rid]) - 1, 0)
        accepted += taken
        e_pred += float(ratio) * float(mass)
        e_mass += float(mass)
        count += 1
        detail = details.get(rid)
        if detail is not None:
            mask = masks.get(rid)
            try:
                row = _rid_detail(detail, taken, tails.get(rid), mask=mask)
                per_rid[rid] = row
                if row.get("accept_k") is not None:
                    used_masks += 1
                if late is not None:
                    late[rid] = (detail, taken, row["mass_k"])
            except (KeyError, TypeError, ValueError, IndexError, AttributeError):
                pass
            if round_detail is None:
                round_detail = detail
    if count == 0:
        return empty
    empty.update(
        accepted=accepted,
        e_pred=e_pred,
        e_mass=e_mass,
        e_n=count,
        e_rid=per_rid or None,
        rpd_masks=used_masks if masks else None,
    )
    if round_detail is not None:
        quotes = round_detail.get("quotes")
        theta_value = round_detail.get("theta")
        cover = round_detail.get("k_cover")
        empty.update(
            k_mode=round_detail.get("k_mode"),
            k_cover=None if cover is None else int(cover),
            verify_quotes=(
                None
                if quotes is None
                else {str(int(k)): float(v) for k, v in sorted(quotes.items())}
            ),
            theta=None if theta_value is None else float(theta_value),
        )
    return empty


class ClearLedger:
    """One Draft process. Rank 0 constructs this; other ranks do not."""

    def __init__(
        self,
        *,
        backend: Optional[str] = None,
        buckets: Optional[Sequence[int]] = None,
        active_rids: Optional[Sequence[str]] = None,
    ) -> None:
        self.backend = (backend or target_backend()).strip().lower() or "cuda"
        self.buckets = tuple(buckets) if buckets is not None else configured_buckets()
        self.active_rids = {str(rid) for rid in (active_rids or ())}
        self.open: Optional[_Round] = None
        self.layer_models: Dict[tuple, ForgettingRegressor] = {}
        # [num, den, n]: forgetting sums of measured wall and base prediction.
        self.layer_depth: Dict[tuple, list] = {}
        self.layer_depth_pooled: Dict[tuple, list] = {}
        self.verify_model: Optional[ForgettingRegressor] = None
        self.verify_samples = 0
        self.pending_logs: List[tuple] = []
        self.split_capable = False
        self.hold_invalid = 0
        self._pending_rounds: Dict[tuple, dict] = {}
        self._pending_reports: Dict[tuple, dict] = {}
        self._pending_order: List[tuple] = []
        # Closed rounds whose RPD masks have not arrived, oldest first.
        self._pending_rpd: Dict[tuple, dict] = {}
        # One closed round whose accounting waits until after the reply.
        self.deferred: Optional[dict] = None

    def queue_log(self, kind: str, record: dict) -> None:
        self.pending_logs.append((kind, record))

    def flush_logs(self) -> None:
        pending, self.pending_logs = self.pending_logs, []
        for kind, record in pending:
            try:
                _emit(kind, record)
            except Exception:
                # Diagnostics must not interrupt resource cleanup or model updates.
                logger.warning("[CLEAR ledger] log output failed", exc_info=True)

    def drop_fitted(self) -> None:
        """Drop the open round and both regressions. Theta lives outside."""
        self.open = None
        self.layer_models.clear()
        self.reset_layer_depth()
        self.verify_model = None
        self.verify_samples = 0
        self._clear_pending()

    def reset_layer_depth(self) -> None:
        """Drop the depth ratios. They are relative to the regressions they saw."""
        self.layer_depth.clear()
        self.layer_depth_pooled.clear()

    def drop_open(self) -> None:
        """Drop the round that has not closed. Keep both regressions."""
        self.open = None
        self.split_capable = False

    def on_packet(
        self,
        action,
        rids: Sequence[str],
        session_id: str,
        rpc_seq: int,
        prefix_lens: Sequence[int],
        width: int,
        now: Optional[float] = None,
        recv_queued: bool = False,
        hold_ns: Optional[int] = None,
        hold_rpc_seq: Optional[int] = None,
        target_queued: Optional[bool] = None,
        step_bytes: Optional[int] = None,
        reply_bytes: Optional[int] = None,
        verify=None,
        scheduler=None,
        defer: bool = False,
        next_tails: Optional[Dict[str, List[int]]] = None,
    ) -> Optional[dict]:
        """Close the previous STEP when this packet is its next STEP.

        PREFILL, FINISH, and ABORT between send and the next STEP stay on the
        open round and exclude it. They do not start a round. A residence
        value marks this session as able to split H, even when the value
        does not belong to the round being closed.

        With ``defer``, a same-session close only detaches the round, its
        accept rows, and the gain ratio, and returns None. ``run_deferred``
        does the accounting after the reply. A session change still closes
        here, so its pending reports clear after that close.
        """
        if type(hold_ns) is int:
            self.split_capable = True
        stamp = time.perf_counter() if now is None else float(now)
        name = _action_name(action)
        closed = None
        if self.open is not None and self.open.t_send is not None:
            if name == "step" and defer and str(self.open.session_id) == str(
                session_id
            ):
                self.run_deferred()
                self._drop_other_sessions(str(session_id))
                self._defer_close(
                    stamp,
                    name,
                    rids,
                    prefix_lens,
                    recv_queued,
                    hold_ns=hold_ns,
                    hold_rpc_seq=hold_rpc_seq,
                    target_queued=target_queued,
                    step_bytes=step_bytes,
                    reply_bytes=reply_bytes,
                    verify=verify,
                    scheduler=scheduler,
                    next_tails=next_tails,
                )
            elif name == "step":
                closed_session = str(self.open.session_id)
                self._drop_other_sessions(str(session_id))
                closed = self._close(
                    stamp,
                    name,
                    rids,
                    prefix_lens,
                    recv_queued,
                    hold_ns=hold_ns,
                    hold_rpc_seq=hold_rpc_seq,
                    target_queued=target_queued,
                    step_bytes=step_bytes,
                    reply_bytes=reply_bytes,
                    verify=verify,
                    scheduler=scheduler,
                    next_tails=next_tails,
                )
                if closed_session != str(session_id):
                    self._clear_pending()
            elif name in _GAP_ACTIONS:
                self.open.gap_actions.append(name)
                self.open.gap_rpcs.append(int(rpc_seq))
                self._apply_gap(name, rids)
                return None
            else:
                return None
        elif self.open is not None:
            self.open = None
        if name in _GAP_ACTIONS:
            self._apply_gap(name, rids)
            return closed
        if name == "step":
            idle_before = not self.active_rids
            self.active_rids = {str(rid) for rid in rids}
            self.open = _Round(
                str(session_id),
                int(rpc_seq),
                rids,
                prefix_lens,
                int(width),
                idle_before,
                stamp,
            )
        return closed

    def _defer_close(
        self,
        t_recv: float,
        next_action: str,
        next_rids: Sequence[str],
        next_prefixes: Sequence[int],
        recv_queued: bool,
        **kwargs,
    ) -> None:
        """Detach the closing round with everything its accounting reads later."""
        from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import (
            take_pending,
        )

        rnd = self.open
        self.open = None
        self.deferred = dict(
            kwargs,
            t_recv=float(t_recv),
            next_action=next_action,
            next_rids=list(next_rids),
            next_prefixes=[int(value) for value in next_prefixes],
            recv_queued=bool(recv_queued),
            rnd=rnd,
            pending=take_pending(),
            gain=_gain_ratio(),
        )

    def run_deferred(self) -> Optional[dict]:
        """Account the detached round, if any. Safe to call when none waits."""
        args, self.deferred = self.deferred, None
        if args is None:
            return None
        return self._close(**args)

    def _apply_gap(self, name: str, rids: Sequence[str]) -> None:
        if name == "prefill":
            self.active_rids.update(str(rid) for rid in rids)
        elif name in ("finish", "abort"):
            from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import (
                forget_request,
            )

            for rid in rids:
                self.active_rids.discard(str(rid))
                forget_request(str(rid))

    def note_idle(self) -> None:
        """Flush generated rows even when no subsequent reply is sent."""
        self.run_deferred()
        self.flush_logs()

    def mark_stop(self, now: Optional[float] = None) -> None:
        """The first stop in a round wins. Later cuts stay inside ``T_K``."""
        if self.open is None or self.open.t_stop is not None:
            return
        self.open.t_stop = time.perf_counter() if now is None else float(now)

    def mark_send(self, now: Optional[float] = None) -> None:
        if self.open is None:
            return
        self.open.t_send = time.perf_counter() if now is None else float(now)

    def mark_post(self, now: Optional[float] = None) -> None:
        if self.open is None:
            return
        self.open.t_post = time.perf_counter() if now is None else float(now)

    def add_draft_prefill(self, seconds: float) -> None:
        if self.open is None:
            return
        self.open.draft_prefill_us += int(float(seconds) * 1e6)

    def note_bonus_hit(self, rid: str) -> None:
        if self.open is None:
            return
        self.open.bonus_hit_rids.append(str(rid))

    def queue_layer(self, sample: dict) -> Optional[dict]:
        if self.open is None:
            return None
        self.open.layers.append(sample)
        return sample

    def _close(
        self,
        t_recv: float,
        next_action: str,
        next_rids: Sequence[str],
        next_prefixes: Sequence[int] = (),
        recv_queued: bool = False,
        hold_ns: Optional[int] = None,
        hold_rpc_seq: Optional[int] = None,
        target_queued: Optional[bool] = None,
        step_bytes: Optional[int] = None,
        reply_bytes: Optional[int] = None,
        verify=None,
        scheduler=None,
        rnd: Optional[_Round] = None,
        pending=None,
        gain: Optional[float] = None,
        next_tails: Optional[Dict[str, List[int]]] = None,
    ) -> dict:
        """Account one closed round.

        A deferred close passes the detached ``rnd``, the accept rows taken at
        receive (``pending``), and the gain ratio seen then (``gain``).
        """
        started = time.perf_counter()
        if rnd is None:
            rnd = self.open
            self.open = None
        assert rnd is not None
        reasons = []
        gap = set(rnd.gap_actions)
        for name in _GAP_ACTIONS:
            if name in gap:
                reasons.append(name)
        # A row-short rid left this round's rids but stays in later packets.
        known = set(rnd.rids) | set(rnd.row_short_rids)
        if any(str(rid) not in known for rid in next_rids):
            reasons.append("new_rid")
        if rnd.idle_before:
            reasons.append("idle_start")
        if _target_gap(next_rids, next_prefixes, pending):
            reasons.append("target_gap")
        if rnd.fallback:
            reasons.append("fallback")
        if rnd.row_short_rids and not rnd.rids:
            # Nothing was drafted this round.
            reasons.append("row_short")
        report = _take_verify_report(rnd, verify)
        incoming = _valid_verify_report(verify)
        if incoming is not None and report is None:
            self._park_report(str(rnd.session_id), incoming)
            self._settle_rpd(str(rnd.session_id), incoming)
        if report is not None and report["fallback"]:
            reasons.append("target_fallback")
        prefit = rnd.prefit if isinstance(rnd.prefit, dict) else None
        phase = prefit.get("phase") if prefit else None
        if phase == "warmup":
            reasons.append("warmup")
        elif phase == "mixed":
            reasons.append("prefit_mixed")
        elif phase == "stale":
            reasons.append("prefit_stale")
        elif phase == "control":
            reasons.append("prefit_control")
        elif phase == "keep" and prefit.get("group") == "v":
            planned = prefit.get("bucket")
            if report is not None and not report["fallback"]:
                actual = report["bucket"]
            else:
                grown = (max(rnd.prefix_lens) if rnd.prefix_lens else 0) + int(rnd.k) + 1
                actual = smallest_bucket(grown, runtime_kv_buckets())
            if planned is None or actual is None or int(actual) != int(planned):
                reasons.append("prefit_bucket")
        draft_floor = next_action == "step" and bool(recv_queued)
        wait_ms = (
            max(0.0, (float(t_recv) - rnd.t_post) * 1000.0)
            if rnd.t_post is not None
            else 0.0
        )
        t_k_us = None
        if rnd.t_stop is not None and rnd.t_send is not None:
            t_k_us = int(
                round(((rnd.t_send - rnd.t_stop) + (t_recv - rnd.t_send)) * 1e6)
            )
        hold_ms, q_ms = self._split_hold(rnd, t_k_us, hold_ns, hold_rpc_seq)
        rnd.hold_ms = hold_ms
        rnd.reply_bytes = reply_bytes
        excluded = bool(reasons)
        verify_pred = None
        verify_residual = None
        bucket = None
        over_bucket = False
        features = None
        if t_k_us is not None and self.backend == "npu":
            # The shadow model stays empty on NPU. K selection uses T_K below.
            needed = (max(rnd.prefix_lens) if rnd.prefix_lens else 0) + rnd.k + 1
            bucket = smallest_bucket(needed, self.buckets)
            over_bucket = bucket is None
        elif t_k_us is not None:
            features = _cuda_features(rnd.b, sum(rnd.prefix_lens), rnd.k)
        # Target verified the row-short requests too, so its batch is not
        # this round's b: keep T_K, H, Q and the Θ price out of the fits.
        target_mixed = bool(rnd.row_short_rids)
        fit_verify = (
            features is not None
            and not excluded
            and not target_mixed
            and not draft_floor
            and not over_bucket
        )
        if features is not None:
            model = self._verify()
            if len(features) != model.n_features:
                features = None
                fit_verify = False
            else:
                if model.n >= VERIFY_MIN_SAMPLES:
                    verify_pred = model.predict(features)
                    if verify_pred not in (None, 0.0):
                        measured_ms = t_k_us / 1000.0
                        verify_residual = measured_ms / verify_pred - 1.0
                if fit_verify:
                    model.update(features, t_k_us / 1000.0)
                    self.verify_samples += 1
        h_pred_ms, q_pred_ms = self._predict_split(rnd)
        self._observe_round_verify(
            rnd,
            t_k_us,
            excluded or target_mixed,
            hold_ms=hold_ms,
            q_ms=q_ms,
            verify=report,
            scheduler=scheduler,
        )
        self.merge_pending(scheduler)
        seed_ms, layer_wall_ms = self._assign_layer_walls(rnd)
        price_tokens, price_ms = _round_price(rnd, t_recv, next_rids, next_prefixes)
        late = {} if report is None else None
        expect = _round_expect(
            rnd,
            next_rids,
            next_prefixes,
            next_tails,
            rpd=None if report is None else report.get("rpd"),
            late=late,
        )
        if late:
            self._stash_rpd(rnd, late, excluded, expect)
        prefit_round = prefit is not None
        price_kept = (
            price_tokens is not None
            and price_ms is not None
            and not excluded
            # price_ms includes the Target verifying the row-short requests,
            # which b no longer counts.
            and not target_mixed
            and not over_bucket
            and not prefit_round
        )
        record = {
            "rpc_seq": rnd.rpc_seq,
            "session_id": rnd.session_id,
            "b": rnd.b,
            "k": rnd.k,
            "w": rnd.w,
            "rids": list(rnd.rids),
            "prefix_lens": list(rnd.prefix_lens),
            "sum_ell": sum(rnd.prefix_lens),
            "max_ell": max(rnd.prefix_lens) if rnd.prefix_lens else 0,
            "t_stop": rnd.t_stop,
            "t_send": rnd.t_send,
            "t_post": rnd.t_post,
            "t_recv": t_recv,
            "t_k_us": t_k_us,
            "draft_floor": draft_floor,
            "wait_ms": wait_ms,
            "policy_ms": rnd.policy_ms,
            "policy_recv_ms": rnd.policy_recv_ms,
            "policy_plan_ms": rnd.policy_plan_ms,
            "policy_grow_ms": rnd.policy_grow_ms,
            "policy_final_k_ms": rnd.policy_final_k_ms,
            "post_ms": rnd.post_ms,
            "excluded": excluded,
            "exclude_reason": reasons,
            "row_short_rids": sorted(rnd.row_short_rids),
            "gap_rpc": list(rnd.gap_rpcs),
            "price_tokens": price_tokens,
            "price_ms": price_ms,
            "price_kept": price_kept,
            "draft_prefill_us": rnd.draft_prefill_us,
            "bonus_hit_rids": list(rnd.bonus_hit_rids),
            "kv_bucket": bucket,
            "over_bucket": over_bucket,
            "verify_pred_ms": verify_pred,
            "verify_residual": verify_residual,
            "verify_fit": fit_verify,
            "seed_ms": seed_ms,
            "layer_wall_ms": layer_wall_ms,
            "gain_ratio": _gain_ratio() if gain is None else float(gain),
            "hold_ms": hold_ms,
            "q_ms": q_ms,
            "target_queued": target_queued if type(target_queued) is bool else None,
            "reply_bytes": reply_bytes,
            "step_bytes": step_bytes,
            "h_pred_ms": h_pred_ms,
            "q_pred_ms": q_pred_ms,
            "prefit": prefit,
        }
        record.update(expect)
        self.queue_log("d_round", record)
        for sample in rnd.layers:
            self._finish_layer(rnd, sample, excluded, scheduler)
        self._note_price(
            rnd,
            next_rids,
            next_prefixes,
            price_tokens,
            price_ms,
            price_kept,
            skip_accept=(
                prefit_round or "target_gap" in reasons or "fallback" in reasons
            ),
            pending=pending,
            seed_ms=seed_ms,
        )
        record["close_ms"] = max(0.0, (time.perf_counter() - started) * 1000.0)
        record["close_deferred"] = pending is not None
        return record

    def _split_hold(self, rnd: _Round, t_k_us, hold_ns, hold_rpc_seq):
        """H is the Target hold for this round's reply. Q is the rest of T_K.

        A value is H only when it names this round and fits inside T_K.
        """
        if t_k_us is None:
            return None, None
        t_k_ms = float(t_k_us) / 1000.0
        if (
            hold_ns is None
            or hold_rpc_seq is None
            or int(hold_rpc_seq) != int(rnd.rpc_seq)
        ):
            return None, None
        candidate = float(hold_ns) / 1e6
        if candidate < 0 or candidate > t_k_ms:
            self.hold_invalid += 1
            return None, None
        return candidate, t_k_ms - candidate

    @staticmethod
    def _round_ell(rnd: _Round) -> int:
        """Largest committed prefix of the round, the x of the H curve."""
        return int(max(rnd.prefix_lens)) if rnd.prefix_lens else 0

    def _predict_split(self, rnd: _Round):
        from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import (
            predict_hold_ms,
            predict_q_ms,
        )

        needed = (max(rnd.prefix_lens) if rnd.prefix_lens else 0) + int(rnd.k) + 1
        bucket = smallest_bucket(needed, runtime_kv_buckets())
        hold = predict_hold_ms(rnd.b, int(rnd.k), bucket, self._round_ell(rnd))
        if hold is None:
            return None, None
        return float(hold), float(predict_q_ms(rnd.b, int(rnd.k)))

    def _observe_round_verify(
        self,
        rnd: _Round,
        t_k_us: Optional[int],
        excluded: bool,
        hold_ms: Optional[float] = None,
        q_ms: Optional[float] = None,
        verify=None,
        scheduler=None,
    ) -> None:
        """Write H and Q together, or the whole T_K before a split exists.

        A draft-floor round is not excluded: that wait is on the critical
        path and still depends on K. Width-sweep rounds update the layer
        fit only. Prefit keep rounds write the active run. Ordinary rounds
        write the online tables. An over-bucket round, an eager report, and
        a fallback report write neither H nor Q. A missing report is stashed
        until that rpc arrives.
        """
        if excluded or t_k_us is None:
            return
        from sglang.srt.speculative.clear_tree_remote.clear_tree_fit import (
            note_online_h,
            prefit_record_h,
        )
        from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import (
            observe_verify,
        )
        from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
            VERIFY_KS,
        )

        prefit = rnd.prefit if isinstance(rnd.prefit, dict) else None
        if prefit is not None and prefit.get("group") == "w":
            return
        keep = prefit is not None and prefit.get("phase") == "keep"
        if self.split_capable:
            if hold_ms is None or q_ms is None or verify is None or verify.get("fallback"):
                if (
                    verify is None
                    and hold_ms is not None
                    and q_ms is not None
                    and not excluded
                ):
                    self._stash_unpaired(
                        rnd, float(hold_ms), float(q_ms), prefit, self._round_ell(rnd)
                    )
                return
            if verify.get("path") == "eager":
                return
            batch = int(verify["b"])
            k = int(verify["k"])
            bucket = verify.get("bucket")
            if bucket is None or k not in VERIFY_KS:
                return
            self._write_hq(
                scheduler,
                rpc=int(verify["rpc_seq"]),
                batch=batch,
                k=k,
                bucket=int(bucket),
                hold_ms=float(hold_ms),
                q_ms=float(q_ms),
                prefit=prefit,
                path=str(verify.get("path") or "graph"),
                keep=keep,
                ell=self._round_ell(rnd),
            )
            return
        if int(rnd.k) not in VERIFY_KS:
            return
        needed = (max(rnd.prefix_lens) if rnd.prefix_lens else 0) + int(rnd.k) + 1
        bucket = smallest_bucket(needed, runtime_kv_buckets())
        if bucket is None:
            return
        measured = float(t_k_us) / 1000.0
        ell = self._round_ell(rnd)
        if keep:
            if prefit.get("group") == "v":
                prefit_record_h(
                    scheduler, rnd.b, int(rnd.k), int(bucket), measured, prefit, ell=ell
                )
            return
        if prefit is not None:
            return
        observe_verify(rnd.b, int(rnd.k), int(bucket), measured, ell)
        note_online_h(scheduler, rnd.b, int(rnd.k), int(bucket), measured, ell=ell)

    def _write_hq(
        self,
        scheduler,
        *,
        rpc,
        batch,
        k,
        bucket,
        hold_ms,
        q_ms,
        prefit,
        path,
        keep,
        ell=None,
    ) -> None:
        """Write one graph sample and log the b, K, and bucket that landed.

        ``ell`` is the round's largest committed prefix, kept beside H.
        """
        from sglang.srt.speculative.clear_tree_remote.clear_tree_fit import (
            note_online_h,
            prefit_record_h,
        )
        from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import (
            observe_q,
            observe_verify,
        )

        logger.info(
            "[CLEAR] fit write rpc=%s b=%s k=%s bucket=%s path=%s",
            int(rpc),
            int(batch),
            int(k),
            int(bucket),
            path,
        )
        if keep:
            if isinstance(prefit, dict) and prefit.get("group") == "v":
                prefit_record_h(
                    scheduler,
                    int(batch),
                    int(k),
                    int(bucket),
                    float(hold_ms),
                    prefit,
                    ell=ell,
                )
            return
        if prefit is not None:
            return
        observe_verify(int(batch), int(k), int(bucket), float(hold_ms), ell)
        observe_q(int(batch), int(k), float(q_ms))
        note_online_h(
            scheduler, int(batch), int(k), int(bucket), float(hold_ms), ell=ell
        )

    def _pending_key(self, session, rpc) -> tuple:
        return (str(session), int(rpc))

    def _remember_pending(self, key) -> None:
        if key in self._pending_order:
            self._pending_order.remove(key)
        self._pending_order.append(key)
        while len(self._pending_order) > 64:
            old = self._pending_order.pop(0)
            self._pending_rounds.pop(old, None)
            self._pending_reports.pop(old, None)

    def _clear_pending(self) -> None:
        self._pending_rounds.clear()
        self._pending_reports.clear()
        self._pending_order.clear()
        self._pending_rpd.clear()

    def _drop_other_sessions(self, session: str) -> None:
        for key in list(self._pending_order):
            if key[0] != str(session):
                self._pending_order.remove(key)
                self._pending_rounds.pop(key, None)
                self._pending_reports.pop(key, None)
        for key in list(self._pending_rpd):
            if key[0] != str(session):
                self._pending_rpd.pop(key, None)

    def _stash_rpd(self, rnd: _Round, late: dict, excluded: bool, expect: dict) -> None:
        """Keep a closed round's trees until its verify report arrives.

        The Target sends the next STEP from inside verify, before it stores
        this verify's report, so the report (and its RPD masks) rides on the
        STEP after next and the round is already closed. ``_settle_rpd``
        prices the masks then. Only the newest ``LATE_RPD_ROUNDS`` wait.
        """
        key = self._pending_key(rnd.session_id, rnd.rpc_seq)
        self._pending_rpd.pop(key, None)
        self._pending_rpd[key] = {
            "b": rnd.b,
            "k": rnd.k,
            "w": rnd.w,
            "excluded": bool(excluded),
            "e_pred": expect.get("e_pred"),
            "e_mass": expect.get("e_mass"),
            "rows": late,
        }
        while len(self._pending_rpd) > LATE_RPD_ROUNDS:
            self._pending_rpd.pop(next(iter(self._pending_rpd)))

    def _settle_rpd(self, session: str, report: dict) -> None:
        """Log ``d_rpd`` for the closed round a late report names. Never raises."""
        masks = report.get("rpd")
        if not masks:
            return
        held = self._pending_rpd.pop(self._pending_key(session, report["rpc_seq"]), None)
        if held is None:
            return
        try:
            rows = {}
            used = 0
            for rid, (detail, taken, mass_k) in held["rows"].items():
                mask = masks.get(rid)
                if mask is None:
                    continue
                row = {
                    "depth": int(detail["depth"]),
                    "accepted": int(taken),
                    "mass_k": mass_k,
                }
                row.update(_mask_fields(detail, taken, mask))
                used += int(row["accept_k"] is not None)
                rows[rid] = row
            self.queue_log(
                "d_rpd",
                {
                    "session_id": str(session),
                    "rpc_seq": int(report["rpc_seq"]),
                    "b": held["b"],
                    "k": held["k"],
                    "w": held["w"],
                    "excluded": held["excluded"],
                    "e_pred": held["e_pred"],
                    "e_mass": held["e_mass"],
                    "rpd_masks": used,
                    "e_rid": rows or None,
                },
            )
        except Exception:
            logger.warning("[CLEAR ledger] late RPD masks failed", exc_info=True)

    def _stash_unpaired(
        self, rnd: _Round, hold_ms: float, q_ms: float, prefit, ell=None
    ) -> None:
        key = self._pending_key(rnd.session_id, rnd.rpc_seq)
        self._pending_rounds[key] = {
            "hold_ms": float(hold_ms),
            "q_ms": float(q_ms),
            "prefit": None if prefit is None else dict(prefit),
            "ell": None if ell is None else int(ell),
        }
        self._remember_pending(key)

    def _park_report(self, session: str, report: dict) -> None:
        key = self._pending_key(session, report["rpc_seq"])
        self._pending_reports[key] = dict(report)
        self._remember_pending(key)

    def merge_pending(self, scheduler) -> None:
        """Apply reports that now have a closed round. Unmatched rounds stay."""
        from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
            VERIFY_KS,
        )

        for key in list(self._pending_order):
            report = self._pending_reports.get(key)
            held = self._pending_rounds.get(key)
            if report is None or held is None:
                continue
            self._pending_reports.pop(key, None)
            self._pending_rounds.pop(key, None)
            if key in self._pending_order:
                self._pending_order.remove(key)
            if report.get("fallback") or report.get("path") == "eager":
                continue
            bucket = report.get("bucket")
            k = report.get("k")
            batch = report.get("b")
            if (
                type(bucket) is not int
                or type(k) is not int
                or type(batch) is not int
                or int(k) not in VERIFY_KS
            ):
                continue
            prefit = held.get("prefit")
            keep = isinstance(prefit, dict) and prefit.get("phase") == "keep"
            self._write_hq(
                scheduler,
                rpc=int(report["rpc_seq"]),
                batch=int(batch),
                k=int(k),
                bucket=int(bucket),
                hold_ms=float(held["hold_ms"]),
                q_ms=float(held["q_ms"]),
                prefit=prefit,
                path=str(report.get("path") or "graph"),
                keep=keep,
                ell=held.get("ell"),
            )

    def _assign_layer_walls(self, rnd: _Round) -> tuple:
        """Wall time between forwards. The last pre-reply sample ends at stop.

        A sample with no timestamp keeps ``wall_us`` unset and the regression
        falls back to ``host_us + gpu_us``. A missing neighbour is not skipped
        over. Post-reply samples stay out of this chain.
        """
        pre = [sample for sample in rnd.layers if not sample.get("post_reply")]
        for index, sample in enumerate(pre):
            # The last one runs to the stop and carries the round's tail work.
            sample["last"] = index + 1 == len(pre)
            t0 = sample.get("t0")
            if index + 1 < len(pre):
                boundary = pre[index + 1].get("t0")
            else:
                boundary = rnd.t_stop
            if t0 is not None and boundary is not None and float(boundary) >= float(t0):
                sample["wall_us"] = int(round((float(boundary) - float(t0)) * 1e6))
            else:
                sample["wall_us"] = None
        seed_ms = None
        if pre and pre[0].get("t0") is not None:
            seed_ms = (float(pre[0]["t0"]) - float(rnd.t_open)) * 1000.0
        walls = [
            int(sample["wall_us"])
            for sample in pre
            if sample.get("wall_us") is not None
        ]
        layer_wall_ms = (sum(walls) / 1000.0) if walls else 0.0
        return seed_ms, layer_wall_ms

    def _note_price(
        self,
        rnd: _Round,
        next_rids: Sequence[str],
        next_prefixes: Sequence[int],
        price_tokens: Optional[float],
        price_ms: Optional[float],
        price_kept: bool,
        skip_accept: bool = False,
        pending=None,
        seed_ms: Optional[float] = None,
    ) -> None:
        """Kept rounds update Θ_b. A mixed round is logged and left out.

        An empty poll after the reply is only the draft-floor flag. It still
        counts as pure decode. Eager rounds past the last KV bucket stay out.
        A target gap or a fallback round drops the pending accept count
        without writing c_i.
        """
        from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import (
            discard_pending,
            observe_price,
            observe_seed,
            settle_verified,
        )

        if skip_accept:
            discard_pending(pending)
        else:
            settle_verified(next_rids, next_prefixes, pending)
        if not price_kept or price_tokens is None or price_ms is None:
            return
        observe_price(rnd.b, price_tokens, price_ms)
        # The width pair adds this common overhead to both cycles.
        observe_seed(rnd.b, seed_ms)

    def _verify(self) -> ForgettingRegressor:
        if self.verify_model is None:
            if self.backend == "npu":
                width = _verify_feature_width(self.buckets)
            else:
                width = 4
            self.verify_model = ForgettingRegressor(width, VERIFY_MIN_SAMPLES)
        return self.verify_model

    def _finish_layer(self, rnd: _Round, sample: dict, excluded: bool, scheduler=None) -> None:
        host = sample.get("host_us")
        gpu = sample.get("gpu_us")
        wall = sample.get("wall_us")
        post_reply = bool(sample.get("post_reply"))
        pred = None
        residual = None
        fitted = False
        features = None
        measured = None
        depth_ratio = None
        depth_src = None
        if wall is not None:
            measured = int(wall) / 1000.0
        elif host is not None and gpu is not None:
            measured = (int(host) + int(gpu)) / 1000.0
        if measured is not None:
            features = [1.0, float(sample["sum_ell"])]
        # A lookahead process fits only lookahead layers: an ordinary layer
        # after a failed admission is a different cost. The last lookahead
        # layer is still queued at ``t_stop``, so its wall is not a layer.
        lookahead = bool(sample.get("lookahead"))
        lookahead_mode = bool(sample.get("lookahead_mode"))
        fit_ok = not (lookahead_mode and not lookahead) and not (
            lookahead and sample.get("last")
        )
        model = None
        if features is not None and not post_reply:
            key = (
                int(sample["w"]),
                str(sample["path"]),
                int(sample["graph_bucket"]),
            )
            model = self.layer_models.get(key)
            prefit = rnd.prefit if isinstance(rnd.prefit, dict) else None
            keep = prefit is not None and prefit.get("phase") == "keep"
            if prefit is None and model is None:
                model = _new_layer_model()
                self.layer_models[key] = model
            if model is not None and model.n >= LAYER_MIN_SAMPLES:
                pred = model.predict(features)
                if pred not in (None, 0.0) and measured is not None:
                    residual = measured / pred - 1.0
            depth_ratio, depth_src = layer_depth_ratio(
                self, sample["w"], sample["graph_bucket"], sample["layer"]
            )
            if (
                not excluded
                and fit_ok
                and prefit is None
                and not sample.get("last")
                and str(sample["path"]) == "graph"
                # Host plus GPU, used without a wall, is not the same span.
                and wall is not None
                and measured is not None
                and pred not in (None, 0.0)
            ):
                self._learn_depth(sample, float(measured), float(pred))
            if not excluded and fit_ok:
                if keep:
                    from sglang.srt.speculative.clear_tree_remote.clear_tree_fit import (
                        prefit_record_layer,
                    )

                    prefit_record_layer(
                        scheduler,
                        sample["w"],
                        sample["path"],
                        sample["graph_bucket"],
                        features,
                        measured,
                        prefit,
                    )
                    fitted = True
                elif prefit is None:
                    from sglang.srt.speculative.clear_tree_remote.clear_tree_fit import (
                        note_online_layer,
                    )

                    model.update(features, measured)
                    note_online_layer(
                        scheduler,
                        sample["w"],
                        sample["path"],
                        sample["graph_bucket"],
                        features,
                        measured,
                    )
                    fitted = True
        sample_log = {
            "rpc_seq": rnd.rpc_seq,
            "session_id": rnd.session_id,
            "layer": sample["layer"],
            "active": sample["active"],
            "graph_bucket": sample["graph_bucket"],
            "w": sample["w"],
            "sum_ell": sample["sum_ell"],
            "host_us": host,
            "gpu_us": gpu,
            "wall_us": None if wall is None else int(wall),
            "path": sample["path"],
            "post_reply": post_reply,
            "pred_ms": pred,
            "residual": residual,
            "fit": fitted,
            "last": bool(sample.get("last")),
            "depth_ratio": depth_ratio,
            "depth_src": depth_src,
            "lookahead": lookahead,
            "lookahead_mode": lookahead_mode,
        }
        self.queue_log("d_layer", sample_log)

    def _learn_depth(self, sample: dict, measured: float, pred: float) -> None:
        """One middle layer into its depth cell and the pooled depth row."""
        d = _depth_key(sample["layer"])
        g = int(sample["graph_bucket"])
        for table, key in (
            (self.layer_depth, (int(sample["w"]), g, d)),
            (self.layer_depth_pooled, (g, d)),
        ):
            row = table.setdefault(key, [0.0, 0.0, 0])
            row[0] = row[0] * FORGET + measured
            row[1] = row[1] * FORGET + pred
            row[2] += 1


def _valid_verify_report(verify) -> Optional[dict]:
    """One normalized Target shape. A bad type is not a report.

    A report that omits ``path`` is a graph report. Eager and unused-draft
    fallback stay valid even when the prefix has no KV bucket.
    """
    if not isinstance(verify, dict):
        return None
    rpc = verify.get("rpc_seq")
    fallback = verify.get("fallback")
    if type(rpc) is not int or type(fallback) is not bool:
        return None
    path = verify.get("path")
    if path not in (None, "graph", "eager", "fallback"):
        return None
    if fallback or path == "fallback":
        return {"fallback": True, "rpc_seq": int(rpc), "path": "fallback"}
    batch = verify.get("b")
    k = verify.get("k")
    bucket = verify.get("bucket")
    if type(batch) is not int or type(k) is not int or batch <= 0 or k <= 0:
        return None
    if type(bucket) is int and bucket > 0:
        bucket = int(bucket)
    elif bucket is None:
        bucket = None
    else:
        return None
    out = {
        "fallback": False,
        "path": "eager" if path == "eager" else "graph",
        "rpc_seq": int(rpc),
        "b": int(batch),
        "k": int(k),
        "bucket": bucket,
    }
    # RPD acceptable-slot masks ride along only on a verify that used drafts.
    from sglang.srt.speculative.clear_tree_remote.clear_tree_protocol import (
        parse_rpd_masks,
    )

    masks = parse_rpd_masks(verify.get("rpd"))
    if masks:
        out["rpd"] = masks
    return out


def _take_verify_report(rnd: _Round, verify) -> Optional[dict]:
    """The Target shape for this round, or None when the packet does not name it."""
    report = _valid_verify_report(verify)
    if report is None or int(report["rpc_seq"]) != int(rnd.rpc_seq):
        return None
    return report


def _batch_verify_report(batch) -> Optional[dict]:
    """Copy a verified shape off the next STEP. Bad types are not a report."""
    if batch is None:
        return None
    return _valid_verify_report(
        {
            "rpc_seq": getattr(batch, "target_verify_rpc_seq", None),
            "fallback": getattr(batch, "target_verify_fallback", None),
            "b": getattr(batch, "target_verify_b", None),
            "k": getattr(batch, "target_verify_k", None),
            "bucket": getattr(batch, "target_verify_bucket", None),
            "path": getattr(batch, "target_verify_path", None),
            "rpd": getattr(batch, "target_verify_rpd", None),
        }
    )


def merge_pending_verify(scheduler) -> None:
    """Write stashed H/Q whose reports have already arrived."""
    ledger = None if scheduler is None else getattr(scheduler, "_clear_ledger", None)
    merge = None if ledger is None else getattr(ledger, "merge_pending", None)
    if callable(merge):
        merge(scheduler)


def _target_gap(
    next_rids: Sequence[str], next_prefixes: Sequence[int], pending=None
) -> bool:
    """True when a noted request grew by more than its tree depth plus one."""
    from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import (
        target_gap_open,
    )

    return target_gap_open(next_rids, next_prefixes, pending)


def _gain_ratio() -> float:
    """Current next-layer calibration. One when the Draft has no observations."""
    from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import (
        gain_ratio,
    )

    return float(gain_ratio())


def _cuda_features(batch: int, sum_ell: int, k: int) -> List[float]:
    return [1.0, float(batch), float(sum_ell), float(k + 1) * float(sum_ell)]


def _verify_feature_width(buckets: Sequence[int]) -> int:
    """NPU verify features: intercept, one K bit, bucket bits, then two batch terms."""
    from sglang.srt.speculative.clear_tree_remote.clear_tree_width import VERIFY_KS

    return 1 + len(VERIFY_KS) + max(len(buckets) - 1, 0) + 2


def _npu_features(
    batch: int, k: int, bucket: Optional[int], buckets: Sequence[int]
) -> Optional[List[float]]:
    if bucket is None:
        return None
    from sglang.srt.speculative.clear_tree_remote.clear_tree_width import VERIFY_KS

    features = [1.0]
    for known in VERIFY_KS:
        features.append(1.0 if int(k) == int(known) else 0.0)
    rest = [int(item) for item in buckets][1:]
    for item in rest:
        features.append(1.0 if int(bucket) == item else 0.0)
    features.append(float(batch))
    features.append(float(batch) * float(bucket) / 1024.0)
    return features


def _emit(kind: str, record: dict) -> None:
    logger.info(
        "[CLEAR ledger %s] %s",
        kind,
        json.dumps(record, sort_keys=True, separators=(",", ":")),
    )


def bind_ledger(scheduler) -> Optional[ClearLedger]:
    """Rank-0 ledger, or None when the switch is off or this rank does not fit."""
    existing = getattr(scheduler, "_clear_ledger", None)
    if existing is not None:
        return existing or None
    rank = int(getattr(scheduler, "tp_rank", 0) or 0)
    if rank != 0 or not ledger_enabled():
        scheduler._clear_ledger = False
        return None
    ledger = ClearLedger()
    scheduler._clear_ledger = ledger
    return ledger


def _quiet(scheduler, fn) -> None:
    if not ledger_enabled():
        return
    try:
        ledger = bind_ledger(scheduler)
        if ledger is None:
            return
        fn(ledger)
    except Exception:
        logger.warning("[CLEAR ledger] update failed", exc_info=True)


def note_target_prefill(
    session_id, rpc_seq, rids, wall_us, tp_rank=0
) -> Optional[dict]:
    """Target PREFILL wall time. Only rank 0 logs it; the STEP message is unchanged."""
    if int(tp_rank) != 0 or not ledger_enabled():
        return None
    try:
        record = {
            "action": "prefill",
            "session_id": str(session_id),
            "rpc_seq": int(rpc_seq),
            "rids": [str(rid) for rid in list(rids or [])],
            "wall_us": int(wall_us),
        }
        _emit("t_gap", record)
        return record
    except Exception:
        logger.warning("[CLEAR ledger] target prefill log failed", exc_info=True)
        return None


_POLICY_STAGE_FIELDS = {
    "recv": "policy_recv_ms",
    "plan": "policy_plan_ms",
    "grow": "policy_grow_ms",
    "final_k": "policy_final_k_ms",
}


def policy_timed(fn=None, *, stage=None):
    """Measure controller CPU work once, including nested controller calls.

    ``stage`` names the outermost call. Nested controller work stays inside
    that sample and is not added again.
    """

    def deco(func):
        @wraps(func)
        def measured(owner, *args, **kwargs):
            scheduler = getattr(owner, "_scheduler", owner)
            if (
                scheduler is None
                or not ledger_enabled()
                or getattr(scheduler, "_clear_in_post_reply", False)
            ):
                return func(owner, *args, **kwargs)
            depth = int(getattr(scheduler, "_clear_policy_depth", 0))
            scheduler._clear_policy_depth = depth + 1
            if depth == 0:
                scheduler._clear_policy_stage = stage
                started = time.perf_counter()
            else:
                started = None
            try:
                return func(owner, *args, **kwargs)
            finally:
                scheduler._clear_policy_depth = depth
                if started is not None:
                    elapsed = max(0.0, (time.perf_counter() - started) * 1000.0)
                    ledger = getattr(scheduler, "_clear_ledger", None)
                    if (
                        ledger
                        and ledger.open is not None
                        and ledger.open.t_send is None
                    ):
                        ledger.open.policy_ms += elapsed
                        attr = _POLICY_STAGE_FIELDS.get(
                            getattr(scheduler, "_clear_policy_stage", None)
                        )
                        if attr is not None:
                            setattr(
                                ledger.open,
                                attr,
                                getattr(ledger.open, attr) + elapsed,
                            )

        return measured

    if fn is None:
        return deco
    return deco(fn)


def queue_draft_log(scheduler, kind: str, record: dict) -> None:
    _quiet(scheduler, lambda ledger: ledger.queue_log(kind, record))


def flush_draft_logs(scheduler) -> None:
    _quiet(scheduler, lambda ledger: ledger.flush_logs())


def add_post_ms(scheduler, elapsed_ms: float) -> None:
    def _add(ledger):
        if ledger.open is not None:
            ledger.open.post_ms += max(0.0, float(elapsed_ms))

    _quiet(scheduler, _add)


@policy_timed(stage="recv")
def note_packet(scheduler, batch) -> None:
    # A close left over from a packet that never reached its post step runs
    # first, so every packet sees the same state order as before.
    run_deferred_close(scheduler)
    # An interrupted reply must not freeze its abandoned trees later.
    scheduler._clear_replay_to_hold = []
    serial = int(getattr(scheduler, "_clear_build_serial", 0) or 0) + 1
    scheduler._clear_build_serial = serial
    scheduler._clear_build_token = serial
    scheduler._clear_replay_packet = (
        str(getattr(batch, "session_id", "")),
        int(getattr(batch, "rpc_seq", 0) or 0),
    )
    scheduler._clear_plan_fallback = False
    scheduler._clear_round_retried = False
    reqs = list(getattr(batch, "reqs", None) or [])
    rids = [str(getattr(req, "rid", "")) for req in reqs]
    # Snapshot packets also carry output-only ``committed_ids``. The prefix
    # used for theta and c_i is always the Target base length.
    prefix = [int(getattr(req, "base_committed_len", 0) or 0) for req in reqs]
    stored = {}
    for rid, length in zip(rids, prefix):
        if rid:
            stored[rid] = int(length)
    scheduler._clear_target_prefix = stored
    tails = {}
    if ledger_enabled():
        for rid, req in zip(rids, reqs):
            tail = _commit_tail(req)
            if rid and tail is not None:
                tails[rid] = tail
    info = getattr(batch, "target_info", None)
    if isinstance(info, dict):
        from sglang.srt.speculative.clear_tree_remote.clear_tree_fit import (
            on_target_info,
        )

        on_target_info(scheduler, info)
    server = getattr(scheduler, "sr_server", None)
    hold_ns = getattr(batch, "target_residence_ns", None)
    hold_rpc = getattr(batch, "target_residence_rpc_seq", None)

    from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import (
        defer_close_enabled,
    )

    defer = defer_close_enabled()

    def _run(ledger: ClearLedger) -> None:
        if isinstance(info, dict) or type(hold_ns) is int:
            ledger.split_capable = True
        width = int(scheduler.server_args.clear_tree_width)
        ledger.on_packet(
            batch.action,
            rids,
            str(getattr(batch, "session_id", "")),
            int(getattr(batch, "rpc_seq", 0) or 0),
            prefix,
            width,
            recv_queued=bool(getattr(scheduler, "_clear_recv_queued", False)),
            hold_ns=hold_ns if type(hold_ns) is int else None,
            hold_rpc_seq=hold_rpc if type(hold_rpc) is int else None,
            target_queued=getattr(batch, "target_queued", None),
            step_bytes=getattr(server, "last_request_bytes", None),
            reply_bytes=getattr(server, "last_reply_bytes", None),
            verify=_batch_verify_report(batch),
            scheduler=scheduler,
            defer=defer,
            next_tails=tails,
        )

    _quiet(scheduler, _run)


def note_round_expect(scheduler, rid, ratio, mass, depth, detail=None) -> None:
    """Remember one request's c_i and verified path mass on the open round.

    ``detail`` holds references for the close-time diagnostics (window,
    scores, K quotes); it is never read by a decision. A ledger that is
    already bound still records the accept row when the switch is off; the
    caller omits ``detail``. This does not call ``bind_ledger`` while the
    switch is off, because that would pin ``_clear_ledger`` to False.
    """
    if scheduler is None:
        return

    def _run(ledger: ClearLedger) -> None:
        rnd = ledger.open
        if rnd is None:
            return
        rnd.e_rows[str(rid)] = (float(ratio), float(mass))
        if detail is not None:
            rnd.e_detail[str(rid)] = detail
        else:
            rnd.e_detail.pop(str(rid), None)
        rnd.tree_depth = max(int(rnd.tree_depth or 0), int(depth or 0))

    existing = getattr(scheduler, "_clear_ledger", None)
    if isinstance(existing, ClearLedger):
        try:
            _run(existing)
        except Exception:
            logger.warning("[CLEAR ledger] update failed", exc_info=True)
        return
    if not ledger_enabled():
        return
    _quiet(scheduler, _run)


def run_deferred_close(scheduler) -> None:
    """Account the round that ``note_packet`` detached. A no-op when none waits.

    The Draft loop calls this after the reply is sent and before the post
    step, so the Target never waits for it.
    """
    _quiet(scheduler, lambda ledger: ledger.run_deferred())


def note_idle(scheduler) -> None:
    _quiet(scheduler, lambda ledger: ledger.note_idle())


def mark_stop(scheduler) -> None:
    _quiet(scheduler, lambda ledger: ledger.mark_stop())


def mark_round_fallback(scheduler) -> None:
    """The open round rebuilt one request at a time and stays out of the fits."""

    def _run(ledger: ClearLedger) -> None:
        if ledger.open is not None:
            ledger.open.fallback = True

    _quiet(scheduler, _run)


def mark_round_row_short(scheduler, rids) -> None:
    """Take row-short requests out of the open STEP round.

    Their committed length is past the Draft ``req_to_token`` row, so they
    get empty drafts and the plan did not count them. They leave the round's
    rids, prefix lengths and b, so price, accept and layer accounting match
    the trees that were grown. The Target still verified them, so the round
    writes no T_K, H or Q (``row_short_rids`` on ``d_round``). A round with
    only row-short requests is excluded as ``row_short``.
    """
    drop = {str(rid) for rid in rids}

    def _run(ledger: ClearLedger) -> None:
        rnd = ledger.open
        if rnd is None or not drop:
            return
        kept = [
            (rid, prefix)
            for rid, prefix in zip(rnd.rids, rnd.prefix_lens)
            if str(rid) not in drop
        ]
        rnd.row_short_rids |= {str(rid) for rid in rnd.rids if str(rid) in drop}
        rnd.rids = [rid for rid, _prefix in kept]
        rnd.prefix_lens = [prefix for _rid, prefix in kept]
        rnd.b = len(rnd.rids)

    _quiet(scheduler, _run)


def mark_send(scheduler) -> None:
    _quiet(scheduler, lambda ledger: ledger.mark_send())


def mark_post(scheduler) -> None:
    _quiet(scheduler, lambda ledger: ledger.mark_post())


def add_draft_prefill(scheduler, seconds: float) -> None:
    _quiet(scheduler, lambda ledger: ledger.add_draft_prefill(seconds))


def note_bonus_hit(scheduler, req) -> None:
    rid = str(getattr(req, "rid", ""))
    _quiet(scheduler, lambda ledger: ledger.note_bonus_hit(rid))


def set_post_reply(scheduler, active: bool) -> None:
    if not ledger_enabled():
        return
    scheduler._clear_layer_post_reply = bool(active)


def remember_layer_path(scheduler, result) -> None:
    if not ledger_enabled():
        return
    flag = getattr(result, "can_run_cuda_graph", None)
    if flag is None:
        flag = getattr(result, "can_run_graph", None)
    if flag is None:
        path = "unknown"
    else:
        path = "graph" if bool(flag) else "eager"
    scheduler._clear_layer_path = path


def queue_layer_sample(
    scheduler,
    *,
    layer: int,
    runners,
    host_us: Optional[int],
    width: int,
    t0: Optional[float] = None,
) -> Optional[dict]:
    """Attach one sub-batch to the open round. GPU time is filled later.

    ``t0`` is the host clock taken when this layer forward began.
    """
    if not ledger_enabled():
        return None
    try:
        ledger = bind_ledger(scheduler)
        if ledger is None:
            return None
        prefix = [int(getattr(runner, "prefix_len", 0) or 0) for runner in runners]
        path = str(getattr(scheduler, "_clear_layer_path", "unknown") or "unknown")
        scheduler._clear_layer_path = "unknown"
        try:
            from sglang.srt.speculative.clear_tree_remote.clear_tree_forward import (
                lookahead_mode,
            )

            mode = bool(lookahead_mode(scheduler))
        except Exception:
            mode = False
        sample = {
            "layer": int(layer),
            "active": len(prefix),
            "graph_bucket": layer_graph_bs(scheduler, len(prefix)),
            "w": int(width),
            "sum_ell": sum(prefix),
            "host_us": None if host_us is None else int(host_us),
            "gpu_us": None,
            "path": path,
            "post_reply": bool(getattr(scheduler, "_clear_layer_post_reply", False)),
            "lookahead": bool(getattr(scheduler, "_opt_lookahead_active", False)),
            "lookahead_mode": mode,
            "t0": None if t0 is None else float(t0),
        }
        return ledger.queue_layer(sample)
    except Exception:
        logger.warning("[CLEAR ledger] layer sample failed", exc_info=True)
        return None


def remember_layer_gpu(sample, gpu_us: Optional[int]) -> None:
    if isinstance(sample, dict):
        sample["gpu_us"] = None if gpu_us is None else int(gpu_us)


def complete_verify_report(frozen, ran_graph: bool, rpd=None) -> dict:
    """Build the verify report from the shape frozen before accept.

    The bucket is ``pre-verify length + K + 1``. Eager is not fallback.
    ``K <= 0`` means the verify did not use a draft. ``rpd`` is the
    ``rid -> hex`` acceptable-slot masks of an RPD host-plan verify.
    """
    from sglang.srt.speculative.tree_attn_fallback import (
        select_tree_kv_bucket,
    )

    seq_lens = [int(item) for item in (frozen.get("seq_lens") or [])]
    k = int(frozen.get("k") or 0)
    if k <= 0:
        path = "fallback"
        bucket = None
    else:
        path = "graph" if ran_graph else "eager"
        max_ell = max(seq_lens) if seq_lens else 0
        chosen = select_tree_kv_bucket(max_ell + k + 1, runtime_kv_buckets())
        bucket = None if chosen is None else int(chosen)
    rpc = frozen.get("rpc_seq")
    report = {
        "session": frozen.get("session"),
        "rpc_seq": rpc if type(rpc) is int else None,
        "b": int(frozen.get("b") or 0),
        "k": int(k),
        "bucket": bucket,
        "path": path,
        "fallback": path == "fallback",
    }
    if path != "fallback" and isinstance(rpd, dict) and rpd:
        report["rpd"] = dict(rpd)
    return report


def runtime_kv_buckets() -> tuple:
    """Buckets the Target graph actually uses. Draft's shadow fit has its own list."""
    raw = os.environ.get("SGLANG_NPU_TREE_GRAPH_KV_BUCKETS", "")
    if not raw.strip():
        return _DEFAULT_BUCKETS
    buckets = []
    for piece in raw.split(","):
        text = piece.strip()
        if text:
            buckets.append(int(text))
    return tuple(sorted(set(buckets)))


_target_verify_model: Optional[ForgettingRegressor] = None
_target_verify_samples = 0


def reset_target_verify_model() -> None:
    """Drop the Target verify fit. Tests use this; the server does not."""
    global _target_verify_model, _target_verify_samples
    _target_verify_model = None
    _target_verify_samples = 0


def target_verify_samples() -> int:
    return _target_verify_samples


def _target_verify_regressor() -> ForgettingRegressor:
    global _target_verify_model
    if _target_verify_model is None:
        buckets = runtime_kv_buckets()
        _target_verify_model = ForgettingRegressor(
            _verify_feature_width(buckets), VERIFY_MIN_SAMPLES
        )
    return _target_verify_model


def _fit_target_device(
    batch_size: int,
    k: int,
    kv_bucket: Optional[int],
    ran_graph: bool,
    device_us: Optional[int],
) -> tuple:
    """Fit one graph verify. Eager, over-bucket, and missing times stay out.

    The label is ``device_us`` in milliseconds. Features stay ``_npu_features``.
    Returns ``(pred_ms, residual, fitted)``. Prediction uses the model from
    before this sample is added.
    """
    if device_us is None or not ran_graph or kv_bucket is None:
        return None, None, False
    buckets = runtime_kv_buckets()
    features = _npu_features(int(batch_size), int(k), int(kv_bucket), buckets)
    if features is None:
        return None, None, False
    model = _target_verify_regressor()
    if len(features) != model.n_features:
        return None, None, False
    measured = float(device_us) / 1000.0
    from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import (
        observe_verify,
    )

    observe_verify(int(batch_size), int(k), int(kv_bucket), measured)
    pred = None
    residual = None
    if model.n >= VERIFY_MIN_SAMPLES:
        pred = model.predict(features)
        if pred not in (None, 0.0):
            residual = measured / pred - 1.0
    model.update(features, measured)
    global _target_verify_samples
    _target_verify_samples += 1
    return pred, residual, True


def target_verify_record(
    *,
    rank: int,
    batch_size: int,
    k: int,
    seq_lens: Sequence[int],
    ran_graph: bool,
    kv_bucket: Optional[int],
    verify_host_us: Optional[int] = None,
    device_us: Optional[int] = None,
    verify_id: Optional[int] = None,
    rpc_seq: Optional[int] = None,
    timing: Optional[dict] = None,
) -> Optional[dict]:
    """One Target verify line. ``kv_bucket`` comes from ``select_tree_kv_bucket``.

    A prefix that does not fit the largest bucket is logged as eager.
    ``verify_host_us`` is the host time already stored on ``verify_forward``.
    ``device_us`` is set when that forward's event has completed; otherwise
    the next ``poll`` emits ``t_verify_device`` and fits there.
    A ready graph sample updates the Target verify model. Eager and
    over-bucket rows are logged only.
    ``timing`` holds the host wall stages in ``VERIFY_TIMING_FIELDS``; each
    missing or invalid one is null. They are logged only.
    """
    if not ledger_enabled() or int(rank) != 0:
        return None
    lengths = [int(length) for length in seq_lens]
    max_ell = max(lengths) if lengths else 0
    over_bucket = kv_bucket is None
    if over_bucket or not ran_graph:
        path = "eager"
    else:
        path = "graph"
    pred = None
    residual = None
    fitted = False
    if device_us is not None:
        pred, residual, fitted = _fit_target_device(
            int(batch_size),
            int(k),
            None if kv_bucket is None else int(kv_bucket),
            bool(ran_graph),
            int(device_us),
        )
    record = {
        "b": int(batch_size),
        "k": int(k),
        "max_ell": max_ell,
        "sum_ell": sum(lengths),
        "path": path,
        "kv_bucket": None if kv_bucket is None else int(kv_bucket),
        "over_bucket": over_bucket,
        "ran_graph": bool(ran_graph),
        "verify_host_us": None if verify_host_us is None else int(verify_host_us),
        "device_us": None if device_us is None else int(device_us),
        "verify_id": None if verify_id is None else int(verify_id),
        "rpc_seq": None if rpc_seq is None else int(rpc_seq),
        "verify_pred_ms": pred,
        "verify_residual": residual,
        "verify_fit": fitted,
    }
    record.update(_verify_timing_fields(timing))
    _emit("t_verify", record)
    return record


# Host wall stages of one Target verify, in microseconds (``t_verify``).
# verify_wall_us: the whole ClearTree verify call, forward included.
# accept_ready_us: verify start to the accept-ready hook that sends the
#   next STEP, so the part of this verify inside the Draft's H.
# rpd_build_us: RPD topology build and edge upload, done while the verify
#   input was built, before verify starts.
# rpd_stats_us: RPD edge statistics, D2H copies and the wait for the forward.
# rpd_select_us: RPD host path selection after the statistics.
# rpd_mask_us: acceptable-slot masks, after the next STEP was sent.
VERIFY_TIMING_FIELDS = (
    "verify_wall_us",
    "accept_ready_us",
    "rpd_build_us",
    "rpd_stats_us",
    "rpd_select_us",
    "rpd_mask_us",
)


def _verify_timing_fields(timing) -> dict:
    out = {}
    source = timing if isinstance(timing, dict) else {}
    for name in VERIFY_TIMING_FIELDS:
        value = source.get(name)
        valid = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and value >= 0
        )
        out[name] = int(value) if valid else None
    return out


class VerifyWallTimer:
    """Wall stamps of one Target verify. Never raises.

    The batch's ``sr_accept_ready_hook`` is wrapped so the moment accept is
    ready (just before the next STEP is sent) is stamped once; ``stop``
    restores the hook the scheduler installed.
    """

    def __init__(self, batch) -> None:
        self.batch = batch
        self.start = time.perf_counter_ns()
        self.ready: Optional[int] = None
        self.end: Optional[int] = None
        self._hook = None
        self._stamped = None
        try:
            hook = getattr(batch, "sr_accept_ready_hook", None)
            if hook is not None:

                def stamped(*args, **kwargs):
                    if self.ready is None:
                        self.ready = time.perf_counter_ns()
                    return hook(*args, **kwargs)

                batch.sr_accept_ready_hook = stamped
                self._hook, self._stamped = hook, stamped
        except Exception:
            self._hook = self._stamped = None

    def stop(self) -> None:
        if self.end is None:
            self.end = time.perf_counter_ns()
        try:
            if (
                self._stamped is not None
                and getattr(self.batch, "sr_accept_ready_hook", None)
                is self._stamped
            ):
                self.batch.sr_accept_ready_hook = self._hook
        except Exception:
            pass

    def fields(self) -> dict:
        return {
            "verify_wall_us": (
                None if self.end is None else (self.end - self.start) // 1000
            ),
            "accept_ready_us": (
                None if self.ready is None else (self.ready - self.start) // 1000
            ),
        }


def read_verify_host_us(metrics) -> Optional[int]:
    """Host seconds already added by ``verify_forward``, in microseconds."""
    host = getattr(metrics, "_round_host", None)
    if host is None or "verify_forward" not in host:
        return None
    return int(round(float(host["verify_forward"]) * 1e6))


VERIFY_SAMPLE_LIMIT = 8


def push_verify_sample(metrics, batch, k, bucket, device_us, ran_graph) -> None:
    """Keep one finished graph sample for the next ClearTree request."""
    if metrics is None or int(getattr(metrics, "tp_rank", 0) or 0) != 0:
        return
    if not ran_graph or bucket is None or device_us is None:
        return
    from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import VERIFY_KS

    if int(k) not in VERIFY_KS:
        return
    queue = getattr(metrics, "_clear_verify_samples", None)
    if queue is None:
        queue = []
        metrics._clear_verify_samples = queue
    queue.append([int(batch), int(k), int(bucket), int(device_us)])
    del queue[:-VERIFY_SAMPLE_LIMIT]


def take_verify_samples(metrics) -> list:
    """Return the queued samples and leave the queue empty."""
    if metrics is None:
        return []
    queue = list(getattr(metrics, "_clear_verify_samples", None) or [])
    metrics._clear_verify_samples = []
    return queue


def restore_verify_samples(metrics, samples) -> None:
    """Put samples back after a send that did not leave this process."""
    if metrics is None or not samples:
        return
    queued = list(getattr(metrics, "_clear_verify_samples", None) or [])
    metrics._clear_verify_samples = list(samples) + queued


def install_clear_device_emit(metrics) -> None:
    if getattr(metrics, "_clear_device_emit", None) is not None:
        return

    def _emit(meta, duration_ms):
        emit_verify_device(meta, duration_ms)
        device_us = int(round(float(duration_ms) * 1000.0))
        push_verify_sample(
            metrics,
            meta.get("b") or 0,
            meta.get("k") or 0,
            meta.get("kv_bucket"),
            device_us,
            bool(meta.get("ran_graph")),
        )

    metrics._clear_device_emit = _emit


def next_verify_id(metrics) -> int:
    seq = int(getattr(metrics, "_clear_verify_seq", 0) or 0) + 1
    metrics._clear_verify_seq = seq
    return seq


def consume_verify_device(metrics, meta: dict) -> Optional[int]:
    """Read the open verify event. Queue it when the device is not done yet.

    ``query`` only. This does not synchronize.
    """
    opened = getattr(metrics, "_clear_device_open", None)
    if not opened:
        return None
    start, end = opened.pop()
    opened.clear()
    try:
        ready = end.query()
    except Exception as exc:
        metrics._timing_failure("query", exc)
        return None
    if not ready:
        pending = metrics._clear_device_pending
        if len(pending) < 16:
            pending.append((start, end, meta))
        else:
            metrics.counts["device_timing_dropped_clear_queue"] += 1
        return None
    try:
        duration_ms = start.elapsed_time(end)
    except Exception as exc:
        metrics._timing_failure("elapsed", exc)
        return None
    return int(round(float(duration_ms) * 1000.0))


def emit_verify_device(meta, duration_ms) -> None:
    device_us = int(round(float(duration_ms) * 1000.0))
    pred = None
    residual = None
    fitted = False
    if ledger_enabled():
        pred, residual, fitted = _fit_target_device(
            int(meta.get("b") or 0),
            int(meta.get("k") or 0),
            meta.get("kv_bucket"),
            bool(meta.get("ran_graph")),
            device_us,
        )
    record = {
        "verify_id": meta.get("verify_id"),
        "rpc_seq": meta.get("rpc_seq"),
        "sum_ell": meta.get("sum_ell"),
        "kv_bucket": meta.get("kv_bucket"),
        "device_us": device_us,
        "verify_pred_ms": pred,
        "verify_residual": residual,
        "verify_fit": fitted,
    }
    _emit("t_verify_device", record)
