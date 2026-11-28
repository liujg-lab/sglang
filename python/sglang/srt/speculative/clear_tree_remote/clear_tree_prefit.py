"""HTTP sweep that fills ClearTree H cells before user traffic.

The Target starts this from a daemon thread. It talks only to the local
HTTP server. Draft decides which rounds to keep and writes the fit file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

PREFIT_VERIFY_WARMUP_ROUNDS = 2
PREFIT_CELL_SAMPLES = 4
_VERIFY_TAIL_ROUNDS = 2
_VERIFY_ROUNDS = PREFIT_VERIFY_WARMUP_ROUNDS + PREFIT_CELL_SAMPLES + _VERIFY_TAIL_ROUNDS
_VERIFY_BUDGET_VERSION = 1
# Layer fits need this many samples per (W, path, graph) cell. Only forwards
# from layer index 2 on are fitted, and the last forward of a lookahead tree is
# not, so a width sweep keeps ``depth - 2`` samples per round.
PREFIT_LAYER_SAMPLES = 50
PREFIT_WIDTH_MIN_DEPTH = 4
# Each width holds this many consecutive rounds of a width group; the first is
# warmup. ``prefit_plan`` on the Draft cycles widths with the same number.
PREFIT_WIDTH_ROUNDS_PER_WIDTH = 4
# A width group plans 20% over the cell target, so a few excluded keep rounds
# do not force a refill of every width group of that batch.
PREFIT_WIDTH_SAMPLE_TARGET = PREFIT_LAYER_SAMPLES + PREFIT_LAYER_SAMPLES // 5
# Verify rounds after warmup; each keeps ``depth - 2`` layer samples.
PREFIT_VERIFY_KEEP_ROUNDS = _VERIFY_ROUNDS - PREFIT_VERIFY_WARMUP_ROUNDS
_DEFAULT_WIDTH_ROUNDS = 20
# 2: prefit layer cells drop spikes; w-group max_new follows the width rotation.
_LAYER_SAMPLE_VERSION = 2


@dataclass
class PrefitConfig:
    url: str
    buckets: Sequence[int]
    max_depth: int
    ks: Sequence[int]
    widths: Sequence[int]
    batches: Sequence[int]
    token_range: Tuple[int, int]
    retries: int = 3
    seed: int = 0
    ready_timeout_s: float = 600.0
    api_key: Optional[str] = None
    fit_timeout_s: float = 60.0


@dataclass(frozen=True)
class PlanInputs:
    batches: Tuple[int, ...]
    ks: Tuple[int, ...]
    widths: Tuple[int, ...]
    buckets: Tuple[int, ...]
    max_depth: int
    rpc_batch_limit: Optional[int] = None


@dataclass
class PrefitPlan:
    inputs: PlanInputs
    groups: list
    required_h: list
    skipped_h: list
    fingerprint: str
    required_layers: list = None
    skipped_layers: list = None
    b_to_g: dict = None

    def __post_init__(self) -> None:
        if self.required_layers is None:
            self.required_layers = []
        if self.skipped_layers is None:
            self.skipped_layers = []
        if self.b_to_g is None:
            self.b_to_g = {}


def prefixes_for(buckets: Sequence[int]) -> list:
    """First bucket uses half its width. Every later bucket uses five eighths."""
    prefixes = []
    for index, bucket in enumerate(buckets):
        fraction = 0.5 if index == 0 else 0.625
        prefixes.append(int(int(bucket) * fraction))
    return prefixes


def depth_for(k: int, width: int, max_depth: int) -> int:
    if int(width) <= 0:
        return 0
    return max(0, min(math.ceil(int(k) / float(width)), int(max_depth) - 1))


def depth_for_width(k: int, width: int, max_depth: int) -> int:
    """Width-sweep depth: at least ``PREFIT_WIDTH_MIN_DEPTH`` layers.

    Layers 2 to depth - 1 are fitted, so four layers keep two samples a round.
    """
    if int(width) <= 0:
        return 0
    return min(
        max(PREFIT_WIDTH_MIN_DEPTH, math.ceil(int(k) / float(width))),
        int(max_depth) - 1,
    )


def max_new_for_verify(bucket: int, prefix: int, k: int) -> int:
    """Tokens that still leave this verify inside the planned KV bucket."""
    return int(bucket) - int(prefix) - int(k) - 4


def width_rounds(depths: Sequence[int], n_buckets: int) -> int:
    """Rounds one width group runs so every width reaches the layer target.

    Each width keeps ``PREFIT_WIDTH_ROUNDS_PER_WIDTH - 1`` rounds a cycle and
    at least ``min(depths) - 2`` fitted layers a round. The planned total is
    ``PREFIT_WIDTH_SAMPLE_TARGET`` (the cell target plus 20%), split over the
    buckets, one width group each. Shallower sweeps keep nothing and run the
    old 20 rounds.
    """
    widths = [int(depth) for depth in depths]
    if not widths:
        return _DEFAULT_WIDTH_ROUNDS
    per_round = min(widths) - 2
    if per_round < 1:
        return _DEFAULT_WIDTH_ROUNDS
    per_bucket = math.ceil(
        PREFIT_WIDTH_SAMPLE_TARGET / float(max(int(n_buckets), 1))
    )
    keep = PREFIT_WIDTH_ROUNDS_PER_WIDTH - 1
    cycles = math.ceil(per_bucket / float(keep * per_round))
    return int(cycles * PREFIT_WIDTH_ROUNDS_PER_WIDTH * len(widths))


def width_keep_rounds(rounds: int, n_widths: int) -> int:
    """Keep rounds every width gets in a width group of ``rounds`` rounds.

    Only full cycles count, so this is a lower bound for every width.
    """
    if int(n_widths) <= 0:
        return 0
    per_width = PREFIT_WIDTH_ROUNDS_PER_WIDTH
    cycles = int(rounds) // (per_width * int(n_widths))
    return int(cycles * (per_width - 1))


def max_new_for_width(depths: Sequence[int], rounds: int = _DEFAULT_WIDTH_ROUNDS) -> int:
    """Token budget that lasts ``rounds`` rounds of the width rotation.

    Round ``r`` uses width ``(r // PREFIT_WIDTH_ROUNDS_PER_WIDTH) % len(depths)``,
    the rotation of the Draft's ``prefit_plan``, and commits at most that
    width's depth plus one. Two more tokens cover the prefill token and one
    spare. With full acceptance the request runs ``rounds`` rounds and at most
    one more (the next width's warmup round); lower acceptance only adds
    rounds.
    """
    widths = [max(int(depth), 0) for depth in depths]
    if not widths:
        return 0
    per = PREFIT_WIDTH_ROUNDS_PER_WIDTH
    total = sum(widths[(r // per) % len(widths)] + 1 for r in range(int(rounds)))
    return int(total) + 2


def _ordered(values, center: int, *, include_center: bool) -> list:
    center = int(center)
    pool = []
    for value in values:
        width = int(value)
        if width <= 0:
            continue
        if width == center and not include_center:
            continue
        pool.append(width)
    return sorted(pool, key=lambda value: (abs(math.log2(value) - math.log2(center)), value))


_PLAN_BATCHES = (1, 2, 4)


def _fingerprint(inputs: PlanInputs) -> str:
    """Hash the normalized plan. A raw batch cap is not part of the identity."""
    payload = {
        "batches": [int(item) for item in inputs.batches],
        "buckets": [int(item) for item in inputs.buckets],
        "ks": [int(item) for item in inputs.ks],
        "max_depth": int(inputs.max_depth),
        "widths": [int(item) for item in inputs.widths],
        "verify_budget_version": _VERIFY_BUDGET_VERSION,
        "layer_sample_version": _LAYER_SAMPLE_VERSION,
        "layer_samples": PREFIT_LAYER_SAMPLES,
        "width_sample_target": PREFIT_WIDTH_SAMPLE_TARGET,
        "width_min_depth": PREFIT_WIDTH_MIN_DEPTH,
        "width_rounds_per_width": PREFIT_WIDTH_ROUNDS_PER_WIDTH,
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:8]


def _previous_bucket(buckets, bucket: int):
    prev = None
    for item in buckets:
        size = int(item)
        if size < int(bucket) and (prev is None or size > prev):
            prev = size
    return prev


def recorded_span(prefix: int, k: int, max_new: int):
    """KV lengths a verify round of this group can be charged to.

    The ledger records ``prefix + K + 1``. Generation can grow the prefix by
    ``max_new`` before that addend.
    """
    lo = int(prefix) + int(k) + 1
    hi = int(prefix) + int(max_new) + int(k) + 1
    return lo, hi


def span_fits_bucket(prefix: int, k: int, max_new: int, bucket: int, previous) -> bool:
    """True when every recorded length of this group lands in ``bucket``."""
    lo, hi = recorded_span(prefix, k, max_new)
    if hi > int(bucket) or lo > int(bucket):
        return False
    if previous is not None and lo <= int(previous):
        return False
    return True


def make_run_id(session_id: str, fingerprint: str, nonce: str) -> str:
    """Session-scoped run id. It does not contain a hyphen."""
    session = "".join(ch for ch in str(session_id) if ch.isalnum() or ch == "_") or "local"
    stamp = "".join(ch for ch in str(nonce) if ch.isalnum()) or "0"
    return f"{session}_{fingerprint}_{stamp}"


def build_plan(inputs: PlanInputs) -> PrefitPlan:
    """One sweep plan. The same inputs always produce the same fingerprint."""
    from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
        flat_k,
        flat_width,
    )

    raw_batches = tuple(int(item) for item in inputs.batches)
    limit = None if inputs.rpc_batch_limit is None else int(inputs.rpc_batch_limit)
    selected = {
        int(item)
        for item in raw_batches
        if limit is None or int(item) <= int(limit)
    }
    # The fingerprint covers the batches that remain, not the raw cap.
    canonical = PlanInputs(
        batches=tuple(batch for batch in _PLAN_BATCHES if batch in selected),
        ks=tuple(int(item) for item in inputs.ks),
        widths=tuple(int(item) for item in inputs.widths),
        buckets=tuple(int(item) for item in inputs.buckets),
        max_depth=int(inputs.max_depth),
        rpc_batch_limit=None,
    )
    groups = []
    required_h = []
    skipped_h = []
    prefixes = prefixes_for(canonical.buckets)
    for batch in _PLAN_BATCHES:
        if batch not in selected:
            for bucket in canonical.buckets:
                for k in canonical.ks:
                    skipped_h.append(((batch, int(k), int(bucket)), "batch_limit"))
            continue
        flat_verify = int(flat_k(batch))
        flat_w = int(flat_width(batch))
        groups.append(
            {
                "kind": "warm",
                "b": batch,
                "max_new": 8,
                "prefix": 32,
                "produces_h": [],
                "produces_layers": [],
            }
        )
        for bucket, prefix in zip(canonical.buckets, prefixes):
            previous = _previous_bucket(canonical.buckets, int(bucket))
            for k in _ordered(canonical.ks, flat_verify, include_center=True):
                depth = depth_for(k, flat_w, canonical.max_depth)
                cell = (batch, int(k), int(bucket))
                if depth <= 0:
                    skipped_h.append((cell, "depth"))
                    continue
                tokens_per_verify = depth + 1
                # The limit must outlast the prefill token and the first seven verifies.
                min_new = 2 + (_VERIFY_ROUNDS - 1) * tokens_per_verify
                first_prefix = 1 if previous is None else max(1, previous - k)
                last_prefix = max_new_for_verify(bucket, 0, k) - min_new
                if first_prefix > last_prefix:
                    reason = "room" if last_prefix < 1 else "bucket_span"
                    skipped_h.append((cell, reason))
                    logger.info(
                        "[CLEAR] skip prefit cell b=%s bucket=%s k=%s reason=%s prefix_range=%s",
                        batch,
                        bucket,
                        k,
                        reason,
                        (first_prefix, last_prefix),
                    )
                    continue
                verify_prefix = min(max(prefix, first_prefix), last_prefix)
                room = max_new_for_verify(bucket, verify_prefix, k)
                max_new = min(room, _VERIFY_ROUNDS * tokens_per_verify)
                if not span_fits_bucket(verify_prefix, k, max_new, bucket, previous):
                    skipped_h.append((cell, "bucket_span"))
                    logger.info(
                        "[CLEAR] skip prefit cell b=%s bucket=%s k=%s span=%s",
                        batch,
                        bucket,
                        k,
                        recorded_span(verify_prefix, k, max_new),
                    )
                    continue
                required_h.append(cell)
                groups.append(
                    {
                        "kind": "v",
                        "b": batch,
                        "prefix": int(verify_prefix),
                        "k": int(k),
                        "bucket": int(bucket),
                        "max_new": max_new,
                        "depth": int(depth),
                        "produces_h": [cell],
                        "produces_layers": [],
                    }
                )
            others = _ordered(canonical.widths, flat_w, include_center=False)
            if not others:
                continue
            depths = [
                depth_for_width(flat_verify, width, canonical.max_depth)
                for width in others
            ]
            d_max = max(depths) if depths else 0
            rounds = width_rounds(depths, len(canonical.buckets))
            groups.append(
                {
                    "kind": "w",
                    "b": batch,
                    "prefix": int(prefix),
                    "bucket": int(bucket),
                    "max_new": int(max_new_for_width(depths, rounds)),
                    "depth": int(d_max),
                    "rounds": int(rounds),
                    "produces_h": [],
                    "produces_layers": [],
                }
            )
    return PrefitPlan(
        inputs=canonical,
        groups=groups,
        required_h=required_h,
        skipped_h=skipped_h,
        fingerprint=_fingerprint(canonical),
    )


def plan_from_config(cfg: PrefitConfig) -> PrefitPlan:
    return build_plan(
        PlanInputs(
            batches=tuple(int(item) for item in cfg.batches),
            ks=tuple(int(item) for item in cfg.ks),
            widths=tuple(int(item) for item in cfg.widths),
            buckets=tuple(int(item) for item in cfg.buckets),
            max_depth=int(cfg.max_depth),
        )
    )


def groups_for_missing(plan: PrefitPlan, missing_h, missing_layers, b_to_g=None) -> list:
    """Groups whose yield map lists a missing cell.

    ``b_to_g`` stays in the signature for callers that still pass the capture
    pairs. It does not select a group on its own.
    """
    del b_to_g
    wanted_h = set()
    for cell in list(missing_h or []):
        if isinstance(cell, (list, tuple)) and len(cell) == 3:
            wanted_h.add((int(cell[0]), int(cell[1]), int(cell[2])))
    wanted_layers = set()
    for cell in list(missing_layers or []):
        if isinstance(cell, (list, tuple)) and len(cell) == 3:
            wanted_layers.add((int(cell[0]), str(cell[1]), int(cell[2])))
    chosen = []
    seen = set()

    def _add(group) -> None:
        key = _group_id(group)
        if key in seen:
            return
        seen.add(key)
        chosen.append(group)

    for group in plan.groups:
        produced_h = group.get("produces_h")
        if produced_h is None and group.get("kind") == "v":
            produced_h = [(int(group["b"]), int(group["k"]), int(group["bucket"]))]
        if any(tuple(cell) in wanted_h for cell in list(produced_h or [])):
            _add(group)
            continue
        produced_layers = group.get("produces_layers")
        if produced_layers and any(
            tuple(cell) in wanted_layers for cell in produced_layers
        ):
            _add(group)
    return chosen


def groups_by_id(plan: PrefitPlan, ids) -> list:
    """Groups whose stable id is in ``ids``. Unknown ids are ignored."""
    wanted = set()
    for item in list(ids or []):
        parsed = _parse_group_id(item)
        if parsed is not None:
            wanted.add(parsed)
    chosen = []
    seen = set()
    for group in plan.groups:
        key = _group_id(group)
        if key not in wanted or key in seen:
            continue
        seen.add(key)
        chosen.append(group)
    return chosen


def groups_for_refill(plan: PrefitPlan, status) -> list:
    """Refill groups named by the Draft.

    A present id list is the only selector. An older receipt without that
    list can still refill H from ``produces_h``. It does not infer layers.
    """
    if not isinstance(status, dict):
        return []
    ids = status.get("refill_groups")
    if isinstance(ids, list):
        return groups_by_id(plan, ids)
    return groups_for_missing(plan, status.get("missing_h"), [])


def _parse_group_id(item):
    if not isinstance(item, (list, tuple)) or not item or not isinstance(item[0], str):
        return None
    kind = item[0]
    try:
        if kind == "v" and len(item) == 4:
            return ("v", int(item[1]), int(item[2]), int(item[3]))
        if kind == "w" and len(item) == 3:
            return ("w", int(item[1]), int(item[2]))
        if kind == "warm" and len(item) == 2:
            return ("warm", int(item[1]))
    except (TypeError, ValueError):
        return None
    return None


def _group_id(group: dict):
    kind = group["kind"]
    if kind == "v":
        return ("v", int(group["b"]), int(group["prefix"]), int(group["k"]))
    if kind == "w":
        return ("w", int(group["b"]), int(group["prefix"]))
    return (kind, int(group.get("b") or 0))


def _rid(group: dict, n: int, run: str) -> str:
    kind = group["kind"]
    size = int(group["b"])
    if kind == "warm":
        return f"clearfit-{run}-warm-{size}-{int(n)}"
    if kind == "v":
        return (
            f"clearfit-{run}-v-{size}-{int(group['prefix'])}-"
            f"{int(group['k'])}-{int(n)}"
        )
    if kind == "w":
        return f"clearfit-{run}-w-{size}-{int(group['prefix'])}-{int(n)}"
    return f"clearfit-{run}-{kind}-{int(n)}"


def _request(
    url: str,
    method: str,
    payload: Optional[dict],
    timeout: float,
    api_key: Optional[str] = None,
):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    if not body:
        return {}
    return json.loads(body)


def _wait_until(url: str, timeout_s: float, ready) -> None:
    deadline = time.monotonic() + float(timeout_s)
    last = None
    while time.monotonic() < deadline:
        try:
            if ready(url):
                return
        except Exception as exc:
            last = exc
        time.sleep(0.5)
    raise TimeoutError(f"prefit wait failed for {url}: {last}")


def _model_ready(url: str, api_key: Optional[str] = None) -> bool:
    _request(url + "/health", "GET", None, 5.0, api_key)
    return True


def _server_idle(url: str, api_key: Optional[str] = None) -> bool:
    body = _request(url + "/v1/loads", "GET", None, 5.0, api_key)
    loads = body.get("loads") if isinstance(body, dict) else None
    if not isinstance(loads, list) or not loads:
        return False
    keys = ("num_running_reqs", "num_waiting_reqs")
    return all(
        isinstance(item, dict)
        and all(type(item.get(key)) is int and item[key] == 0 for key in keys)
        for item in loads
    )


def _steps(payload, *, verify_only: bool = False) -> int:
    if isinstance(payload, list):
        if not payload:
            return 0
        return min(_steps(item, verify_only=verify_only) for item in payload)
    meta = payload.get("meta_info") if isinstance(payload, dict) else None
    if not isinstance(meta, dict):
        meta = {}
    count = meta.get("spec_verify_ct")
    if verify_only:
        return count if type(count) is int and count >= 0 else 0
    if count is not None:
        return int(count)
    if meta.get("completion_tokens") is not None:
        return int(meta["completion_tokens"])
    if isinstance(payload, dict) and payload.get("completion_tokens") is not None:
        return int(payload["completion_tokens"])
    return 0


def _tokens(rng: random.Random, length: int, token_range: Tuple[int, int]) -> list:
    lo, hi = int(token_range[0]), int(token_range[1])
    if hi <= lo:
        hi = lo + 1
    return [rng.randrange(lo, hi) for _ in range(max(int(length), 1))]


def _post_payload(
    url: str,
    payload: dict,
    api_key: Optional[str] = None,
    *,
    verify_only: bool = False,
) -> int:
    try:
        body = _request(url + "/generate", "POST", payload, 600.0, api_key)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        logger.warning(
            "[CLEAR] prefit request failed rid=%s", payload.get("rid"), exc_info=True
        )
        return 0
    return _steps(body, verify_only=verify_only)


def _run_group(cfg: PrefitConfig, group: dict, rng: random.Random, run: str, n0: int) -> dict:
    """One batched generate. ``n0`` is the first rid counter for this try."""
    is_verify = group["kind"] == "v"
    if is_verify:
        need = _VERIFY_ROUNDS
    elif group["kind"] == "w":
        need = int(group.get("rounds") or _DEFAULT_WIDTH_ROUNDS)
    else:
        need = 0
    size = int(group["b"])
    attempts = 0
    steps = 0
    limit = max(int(cfg.retries), 1)
    while attempts < limit:
        attempts += 1
        base = int(n0) + (attempts - 1) * size
        payload = {
            "input_ids": [
                _tokens(rng, int(group.get("prefix") or 8), cfg.token_range)
                for _ in range(size)
            ],
            "rid": [_rid(group, base + index, run) for index in range(size)],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": int(group["max_new"]),
                "ignore_eos": True,
            },
        }
        steps = _post_payload(cfg.url, payload, cfg.api_key, verify_only=is_verify)
        if need == 0 or steps >= need:
            break
    record = {
        "kind": group["kind"],
        "b": size,
        "steps": steps,
        "retries": attempts,
        "step_target_met": steps >= need,
    }
    logger.info("[CLEAR] prefit group %s", record)
    return record


def _post_control(
    cfg: PrefitConfig,
    run: str,
    kind: str,
    seq: int,
    rng: random.Random,
    fingerprint: str = "",
) -> Optional[dict]:
    """POST one control op. The response body is the fit status, when present."""
    del rng
    payload = {
        "op": kind,
        "run": run,
        "seq": int(seq),
        "fingerprint": str(fingerprint or ""),
    }
    try:
        body = _request(
            cfg.url + "/clear_tree_control",
            "POST",
            payload,
            float(cfg.fit_timeout_s),
            cfg.api_key,
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        logger.warning("[CLEAR] prefit %s request failed", kind, exc_info=True)
        return None
    if isinstance(body, dict) and (
        "reason" in body or "ready" in body or "state" in body
    ):
        return body
    return None


def _is_committed(status) -> bool:
    return isinstance(status, dict) and (
        status.get("ready") is True
        or status.get("written") is True
        or status.get("reason") == "committed"
    )


def _send_control(cfg, run, kind, seq, rng, fingerprint, submit_control):
    if submit_control is not None:
        return submit_control(kind, run, int(seq), str(fingerprint))
    return _post_control(cfg, run, kind, int(seq), rng, fingerprint=str(fingerprint))


def _recover_commit(cfg, plan, run, seq, rng, wait_status, submit_control):
    """Resend the same commit, then ask for status. A missing body is not acceptance."""
    resent = _send_control(
        cfg, run, "commit", seq, rng, plan.fingerprint, submit_control
    )
    status = resent if isinstance(resent, dict) and "reason" in resent else None
    if status is None and wait_status is not None:
        status = wait_status(run, int(seq), float(cfg.fit_timeout_s))
    if _is_committed(status):
        return status
    queried = _send_control(
        cfg, run, "status", seq, rng, plan.fingerprint, submit_control
    )
    if isinstance(queried, dict) and ("reason" in queried or "ready" in queried):
        return queried
    if wait_status is None:
        return status
    return wait_status(run, int(seq), float(cfg.fit_timeout_s))


def _exchange(cfg, plan, run, op, seq, rng, wait_status, submit_control, recover=False):
    if submit_control is not None:
        sent = submit_control(op, run, int(seq), plan.fingerprint)
        if isinstance(sent, dict) and ("reason" in sent or "ready" in sent):
            return sent
        if wait_status is None:
            if recover and op == "commit":
                return _recover_commit(
                    cfg, plan, run, seq, rng, None, submit_control
                )
            return None
        status = wait_status(run, int(seq), float(cfg.fit_timeout_s))
        if status is None and recover and op == "commit":
            return _recover_commit(
                cfg, plan, run, seq, rng, wait_status, submit_control
            )
        return status
    sent = _post_control(cfg, run, op, int(seq), rng, fingerprint=plan.fingerprint)
    if isinstance(sent, dict) and ("reason" in sent or "ready" in sent):
        return sent
    if sent is True and op == "begin":
        return {"reason": "begun", "ready": False, "run": run, "seq": int(seq)}
    if sent is False or sent is None:
        if recover and op == "commit":
            return _recover_commit(cfg, plan, run, seq, rng, wait_status, submit_control)
        return None
    if wait_status is None:
        if op == "commit":
            return _recover_commit(cfg, plan, run, seq, rng, None, submit_control)
        return None
    status = wait_status(run, int(seq), float(cfg.fit_timeout_s))
    if status is None and recover and op == "commit":
        return _recover_commit(cfg, plan, run, seq, rng, wait_status, submit_control)
    return status


def _attempt(cfg, plan, run, rng, wait_status, counters, submit_control=None) -> str:
    groups = list(plan.groups)
    refill_after_seq = None
    for _commit in range(3):
        try:
            for group in groups:
                key = _group_id(group)
                n0 = int(counters.get(key, 0))
                record = _run_group(cfg, group, rng, run, n0)
                counters[key] = n0 + int(group["b"]) * int(record["retries"])
                if refill_after_seq is not None:
                    logger.info(
                        "[CLEAR prefit_refill_group] %s",
                        json.dumps(
                            {
                                "run": run,
                                "commit_seq": refill_after_seq,
                                "group_id": key,
                                "attempts": record["retries"],
                                "steps": record["steps"],
                                "step_target_met": record["step_target_met"],
                            }
                        ),
                    )
        except Exception:
            logger.warning("[CLEAR] prefit group failed", exc_info=True)
            return "request_error"
        seq = int(counters.get(("commit",), 0)) + 1
        counters[("commit",)] = seq
        status = _exchange(
            cfg, plan, run, "commit", seq, rng, wait_status, submit_control, recover=True
        )
        committed = _is_committed(status)
        status = status if isinstance(status, dict) else {"reason": "timeout"}
        reason = str(status.get("reason") or "rejected")
        groups = groups_for_refill(plan, status) if reason == "missing" else []
        logger.info(
            "[CLEAR prefit_commit] %s",
            json.dumps(
                {
                    "run": run,
                    "seq": seq,
                    "ready": committed,
                    "reason": reason,
                    "missing_h": status.get("missing_h", []),
                    "missing_layers": status.get("missing_layers", []),
                    "refill_groups": [_group_id(group) for group in groups],
                }
            ),
        )
        if committed:
            return "ready"
        if reason != "missing" or not groups:
            return reason
        refill_after_seq = seq
    return "missing"


def run_prefit(
    cfg: PrefitConfig,
    plan: Optional[PrefitPlan] = None,
    wait_status: Optional[Callable] = None,
    submit_control: Optional[Callable] = None,
    session_id: Optional[str] = None,
) -> str:
    """Sweep, then commit. HTTP 200 is not acceptance.

    A wait that fails before the first group does not commit or abandon.
    A group error does not commit; that uncommitted run is abandoned.
    A commit that might already have landed is queried, not cancelled.
    """
    logger.info("[CLEAR] prefit waiting for %s", cfg.url)
    _wait_until(
        cfg.url, cfg.ready_timeout_s, partial(_model_ready, api_key=cfg.api_key)
    )
    _wait_until(
        cfg.url, cfg.ready_timeout_s, partial(_server_idle, api_key=cfg.api_key)
    )
    started = time.monotonic()
    rng = random.Random(int(cfg.seed))
    plan = plan if plan is not None else plan_from_config(cfg)
    counters = {}
    last = "failed"
    session = str(session_id or "local")
    for attempt in (1, 2):
        run = make_run_id(session, plan.fingerprint, f"{int(attempt)}{time.time_ns()}")
        begun = _exchange(
            cfg, plan, run, "begin", 0, rng, wait_status, submit_control, recover=False
        )
        if not (
            isinstance(begun, dict)
            and (begun.get("reason") == "begun" or begun.get("ready") is True)
        ):
            last = "begin_failed" if begun is None else str(begun.get("reason") or "begin_failed")
            if _is_committed(begun):
                return "ready"
            continue
        last = _attempt(cfg, plan, run, rng, wait_status, counters, submit_control)
        if last == "ready":
            logger.info(
                "[CLEAR] prefit driver finished state=ready elapsed_s=%.3f",
                time.monotonic() - started,
            )
            return "ready"
        if last == "timeout":
            logger.info(
                "[CLEAR] prefit driver finished state=failed reason=timeout elapsed_s=%.3f",
                time.monotonic() - started,
            )
            return "timeout"
        seq = int(counters.get(("abandon",), 0)) + 1
        counters[("abandon",)] = seq
        _exchange(
            cfg, plan, run, "abandon", seq, rng, wait_status, submit_control, recover=False
        )
    logger.info(
        "[CLEAR] prefit driver finished state=failed reason=%s elapsed_s=%.3f",
        last,
        time.monotonic() - started,
    )
    return last


def _parse_args(argv=None) -> PrefitConfig:
    parser = argparse.ArgumentParser(description="Rerun a ClearTree prefit sweep")
    parser.add_argument("--url", required=True)
    parser.add_argument("--buckets", nargs="+", type=int, required=True)
    parser.add_argument("--max-depth", type=int, required=True)
    parser.add_argument("--ks", nargs="+", type=int, required=True)
    parser.add_argument("--widths", nargs="+", type=int, required=True)
    parser.add_argument("--batches", nargs="+", type=int, required=True)
    parser.add_argument("--token-lo", type=int, required=True)
    parser.add_argument("--token-hi", type=int, required=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ready-timeout-s", type=float, default=600.0)
    parser.add_argument("--api-key", default="")
    args = parser.parse_args(argv)
    return PrefitConfig(
        url=args.url,
        buckets=tuple(args.buckets),
        max_depth=int(args.max_depth),
        ks=tuple(args.ks),
        widths=tuple(args.widths),
        batches=tuple(args.batches),
        token_range=(int(args.token_lo), int(args.token_hi)),
        retries=int(args.retries),
        seed=int(args.seed),
        ready_timeout_s=float(args.ready_timeout_s),
        api_key=args.api_key or None,
    )


def _cli_wait(cfg: PrefitConfig, fingerprint: str):
    """Status query on the same control POST the commit uses."""

    def _wait(run, seq, timeout):
        del timeout
        return _post_control(
            cfg, run, "status", int(seq), random.Random(0), fingerprint=fingerprint
        )

    return _wait


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _cfg = _parse_args()
    _plan = plan_from_config(_cfg)
    run_prefit(
        _cfg,
        plan=_plan,
        wait_status=_cli_wait(_cfg, _plan.fingerprint),
        session_id=f"cli{time.time_ns()}",
    )
