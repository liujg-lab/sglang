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
import os
import time
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

FORGET = 0.999
RIDGE = 1e-6
LAYER_MIN_SAMPLES = 4
LAYER_SPAN_PAD = 128.0
# Intercept, graph bucket, prefix sum. The prefix prior is one token's
# 2 µs scaled by 128, so a flat prefix cannot set the slope.
LAYER_RIDGE = (0.0, RIDGE, LAYER_SPAN_PAD ** 2)
_PREFIX_FEATURE = 2
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


def smallest_bucket(needed: int, buckets: Sequence[int]) -> Optional[int]:
    """Smallest bucket that can hold ``needed``, or None when all are too small."""
    for bucket in buckets:
        if int(needed) <= int(bucket):
            return int(bucket)
    return None


def _scan_layer_models(models, width: int, *, ready: bool):
    """Graph-path fit for this width, else the first other path.

    ``ready`` keeps only a fit that can already be solved. The sample count
    used to decide a narrow probe still sees a short graph fit.
    """
    preferred = None
    fallback = None
    for key, model in models.items():
        if int(key[0]) != int(width):
            continue
        if ready and (model.n < LAYER_MIN_SAMPLES or model.coefficients() is None):
            continue
        if len(key) > 1 and str(key[1]) == "graph":
            preferred = model
            break
        if fallback is None:
            fallback = model
    if preferred is not None:
        return preferred
    return fallback


def _layer_model(scheduler, width: int):
    """Fitted layer model for this width. A graph-path fit wins over eager."""
    ledger = bind_ledger(scheduler)
    if ledger is None:
        return None
    return _scan_layer_models(ledger.layer_models, width, ready=True)


def layer_sample_count(scheduler, width: int) -> int:
    """Samples in the model ``predict_layer_ms`` will use, even below four.

    A missing ledger counts as zero. This does not create one.
    """
    existing = None if scheduler is None else getattr(scheduler, "_clear_ledger", None)
    if not existing:
        return 0
    model = _scan_layer_models(existing.layer_models, width, ready=False)
    if model is None:
        return 0
    return int(model.n)


def _layer_features(active: int, sum_ell: int) -> List[float]:
    return [1.0, float(graph_bucket(active)), float(sum_ell)]


def predict_layer_ms(scheduler, width: int, active: int, sum_ell: int) -> Optional[float]:
    """Draft layer fit, or None when the prefix is outside the seen span.

    A non-positive prediction is also None. The served width then stays on
    the flat K, and a replay candidate with this result does not pair.
    """
    model = _layer_model(scheduler, width)
    if model is None:
        return None
    features = _layer_features(active, sum_ell)
    if not model.span_allows(features):
        return None
    pred = model.predict(features)
    if pred is None or float(pred) <= 0.0:
        return None
    return float(pred)


def layer_sum_ell_slope(scheduler, width: int) -> Optional[float]:
    """Coefficient of prefix length in the layer fit. None until that fit exists."""
    model = _layer_model(scheduler, width)
    if model is None:
        return None
    beta = model.coefficients()
    if beta is None or len(beta) < 3:
        return None
    return float(beta[2])


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
        self.xtx = [
            [0.0] * self.n_features for _ in range(self.n_features)
        ]
        self.xty = [0.0] * self.n_features
        self._coef: Optional[List[float]] = None
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
        if self.ridge is None:
            for index in range(1, self.n_features):
                system[index][index] += RIDGE
        else:
            for index, value in enumerate(self.ridge):
                if index >= self.n_features:
                    break
                system[index][index] += float(value)
        self._coef = _solve(system, self.xty)
        self._coef_n = self.n
        return self._coef

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
        3,
        LAYER_MIN_SAMPLES,
        ridge=LAYER_RIDGE,
        span_feature=_PREFIX_FEATURE,
    )


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
        "idle_after_post",
        "t_open",
        "t_stop",
        "t_send",
        "t_post",
        "draft_prefill_us",
        "bonus_hit_rids",
        "layers",
        "fallback",
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
        self.idle_after_post = False
        self.t_open = float(opened_at)
        self.t_stop: Optional[float] = None
        self.t_send: Optional[float] = None
        self.t_post: Optional[float] = None
        self.draft_prefill_us = 0
        self.bonus_hit_rids: List[str] = []
        self.layers: List[dict] = []
        self.fallback = False


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
        self.verify_model: Optional[ForgettingRegressor] = None
        self.verify_samples = 0

    def drop_fitted(self) -> None:
        """Drop the open round and both regressions. Theta lives outside."""
        self.open = None
        self.layer_models.clear()
        self.verify_model = None
        self.verify_samples = 0

    def on_packet(
        self,
        action,
        rids: Sequence[str],
        session_id: str,
        rpc_seq: int,
        prefix_lens: Sequence[int],
        width: int,
        now: Optional[float] = None,
    ) -> Optional[dict]:
        """Close the previous STEP when this packet is its next STEP.

        PREFILL, FINISH, and ABORT between send and the next STEP stay on the
        open round and exclude it. They do not start a round.
        """
        stamp = time.perf_counter() if now is None else float(now)
        name = _action_name(action)
        closed = None
        if self.open is not None and self.open.t_send is not None:
            if name == "step":
                closed = self._close(stamp, name, rids, prefix_lens)
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
        """The recv loop found no packet. This is a wait, not an empty server."""
        if self.open is not None and self.open.t_post is not None:
            self.open.idle_after_post = True

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
    ) -> dict:
        rnd = self.open
        self.open = None
        assert rnd is not None
        reasons = []
        gap = set(rnd.gap_actions)
        for name in _GAP_ACTIONS:
            if name in gap:
                reasons.append(name)
        known = set(rnd.rids)
        if any(str(rid) not in known for rid in next_rids):
            reasons.append("new_rid")
        if rnd.idle_before:
            reasons.append("idle_start")
        if _target_gap(next_rids, next_prefixes):
            reasons.append("target_gap")
        if rnd.fallback:
            reasons.append("fallback")
        draft_floor = (
            next_action == "step"
            and rnd.t_post is not None
            and not rnd.idle_after_post
        )
        t_k_us = None
        if rnd.t_stop is not None and rnd.t_send is not None:
            t_k_us = int(
                round(
                    (
                        (rnd.t_send - rnd.t_stop)
                        + (t_recv - rnd.t_send)
                    )
                    * 1e6
                )
            )
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
        fit_verify = (
            features is not None
            and not excluded
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
        self._observe_round_verify(rnd, t_k_us, excluded)
        seed_ms, layer_wall_ms = self._assign_layer_walls(rnd)
        price_tokens, price_ms = _round_price(
            rnd, t_recv, next_rids, next_prefixes
        )
        price_kept = (
            price_tokens is not None
            and price_ms is not None
            and not excluded
            and not over_bucket
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
            "excluded": excluded,
            "exclude_reason": reasons,
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
            "gain_ratio": _gain_ratio(),
        }
        _emit("d_round", record)
        for sample in rnd.layers:
            self._finish_layer(rnd, sample, excluded)
        self._note_price(
            rnd,
            next_rids,
            next_prefixes,
            price_tokens,
            price_ms,
            price_kept,
            skip_accept="target_gap" in reasons or "fallback" in reasons,
        )
        return record

    def _observe_round_verify(
        self, rnd: _Round, t_k_us: Optional[int], excluded: bool
    ) -> None:
        """Write Draft ``T_K`` into the table that chooses K.

        A draft-floor round is not excluded: that wait is on the critical
        path and still depends on K. An over-bucket prefix yields no bucket
        and ``observe_verify`` drops it.
        """
        if excluded or t_k_us is None:
            return
        from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import (
            observe_verify,
        )
        from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
            VERIFY_KS,
        )

        if int(rnd.k) not in VERIFY_KS:
            return
        needed = (max(rnd.prefix_lens) if rnd.prefix_lens else 0) + int(rnd.k) + 1
        observe_verify(
            rnd.b,
            int(rnd.k),
            smallest_bucket(needed, runtime_kv_buckets()),
            float(t_k_us) / 1000.0,
        )

    def _assign_layer_walls(self, rnd: _Round) -> tuple:
        """Wall time between forwards. The last pre-reply sample ends at stop.

        A sample with no timestamp keeps ``wall_us`` unset and the regression
        falls back to ``host_us + gpu_us``. A missing neighbour is not skipped
        over. Post-reply samples stay out of this chain.
        """
        pre = [sample for sample in rnd.layers if not sample.get("post_reply")]
        for index, sample in enumerate(pre):
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
            settle_verified,
        )

        if skip_accept:
            discard_pending()
        else:
            settle_verified(next_rids, next_prefixes)
        if not price_kept or price_tokens is None or price_ms is None:
            return
        observe_price(rnd.b, price_tokens, price_ms)

    def _verify(self) -> ForgettingRegressor:
        if self.verify_model is None:
            if self.backend == "npu":
                width = _verify_feature_width(self.buckets)
            else:
                width = 4
            self.verify_model = ForgettingRegressor(width, VERIFY_MIN_SAMPLES)
        return self.verify_model

    def _finish_layer(self, rnd: _Round, sample: dict, excluded: bool) -> None:
        host = sample.get("host_us")
        gpu = sample.get("gpu_us")
        wall = sample.get("wall_us")
        post_reply = bool(sample.get("post_reply"))
        pred = None
        residual = None
        fitted = False
        features = None
        measured = None
        if wall is not None:
            measured = int(wall) / 1000.0
        elif host is not None and gpu is not None:
            measured = (int(host) + int(gpu)) / 1000.0
        if measured is not None:
            features = [
                1.0,
                float(sample["graph_bucket"]),
                float(sample["sum_ell"]),
            ]
        model = None
        if features is not None and not post_reply:
            key = (int(sample["w"]), str(sample["path"]))
            model = self.layer_models.get(key)
            if model is None:
                model = _new_layer_model()
                self.layer_models[key] = model
            if model.n >= LAYER_MIN_SAMPLES:
                pred = model.predict(features)
                if pred not in (None, 0.0) and measured is not None:
                    residual = measured / pred - 1.0
            if not excluded:
                model.update(features, measured)
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
        }
        _emit("d_layer", sample_log)


def _target_gap(next_rids: Sequence[str], next_prefixes: Sequence[int]) -> bool:
    """True when a noted request grew by more than its tree depth plus one."""
    from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import (
        target_gap_open,
    )

    return target_gap_open(next_rids, next_prefixes)


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


def note_packet(scheduler, batch) -> None:
    serial = int(getattr(scheduler, "_clear_build_serial", 0) or 0) + 1
    scheduler._clear_build_serial = serial
    scheduler._clear_build_token = serial
    scheduler._clear_plan_fallback = False
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
    closed = {}

    def _run(ledger: ClearLedger) -> None:
        width = int(scheduler.server_args.clear_tree_width)
        record = ledger.on_packet(
            batch.action,
            rids,
            str(getattr(batch, "session_id", "")),
            int(getattr(batch, "rpc_seq", 0) or 0),
            prefix,
            width,
        )
        if isinstance(record, dict):
            closed["record"] = record

    _quiet(scheduler, _run)
    if "record" in closed:
        scheduler._clear_replay_closed = closed["record"]


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
        sample = {
            "layer": int(layer),
            "active": len(prefix),
            "graph_bucket": graph_bucket(len(prefix)),
            "w": int(width),
            "sum_ell": sum(prefix),
            "host_us": None if host_us is None else int(host_us),
            "gpu_us": None,
            "path": path,
            "post_reply": bool(getattr(scheduler, "_clear_layer_post_reply", False)),
            "t0": None if t0 is None else float(t0),
        }
        return ledger.queue_layer(sample)
    except Exception:
        logger.warning("[CLEAR ledger] layer sample failed", exc_info=True)
        return None


def remember_layer_gpu(sample, gpu_us: Optional[int]) -> None:
    if isinstance(sample, dict):
        sample["gpu_us"] = None if gpu_us is None else int(gpu_us)


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
) -> Optional[dict]:
    """One Target verify line. ``kv_bucket`` comes from ``select_tree_kv_bucket``.

    A prefix that does not fit the largest bucket is logged as eager.
    ``verify_host_us`` is the host time already stored on ``verify_forward``.
    ``device_us`` is set when that forward's event has completed; otherwise
    the next ``poll`` emits ``t_verify_device`` and fits there.
    A ready graph sample updates the Target verify model. Eager and
    over-bucket rows are logged only.
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
    _emit("t_verify", record)
    return record


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
