"""Saved ClearTree fit and the Draft side of a prefit sweep.

The file is schema 2. It is written only when a commit says this run's
required cells each have four samples. Online updates stay in memory and
are not exported. Commit installs the run snapshot, then replays ordinary
samples taken while that run was open. Cancel drops the run and leaves the
online fits alone. A later process reuses the file only when the header
matches and the saved plan covers the cells that process needs.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from typing import Optional

from sglang.srt.speculative.clear_tree_remote.clear_tree_prefit import (
    PREFIT_CELL_SAMPLES,
    PREFIT_VERIFY_WARMUP_ROUNDS,
)

logger = logging.getLogger(__name__)

_TARGET_KEYS = (
    "backend",
    "device",
    "tp",
    "model",
    "served_model_name",
    "dtype",
    "quantization",
    "verify_capture",
    "kv_buckets",
    "disable_cuda_graph",
    "attention_backend",
    "page_size",
    "cuda_graph_bs",
)


_FIT_SCHEMA = 2
_CELL_SAMPLES = PREFIT_CELL_SAMPLES
_QUERY_BACKOFF = (5.0, 10.0, 30.0)


class PrefitState:
    """Samples that belong to one prefit run, kept apart from the online fits."""

    def __init__(self, run: str, plan) -> None:
        self.run = str(run)
        self.plan = plan
        self.counts = {}
        self.h = {}
        self.layers = {}
        self.online_h = []
        self.online_layers = []
        self.required_h = [] if plan is None else list(plan.required_h)
        self.required_layers = []
        self.skipped = []
        self.ready = False
        self.last_status = None
        self.session_id = None
        self.fingerprint = None


def fit_path() -> str:
    override = os.environ.get("SGLANG_CLEAR_FIT_PATH", "").strip()
    if override:
        return override
    return os.path.join(os.path.dirname(__file__), "clear_tree_fit.json")


def parse_rid(rid) -> Optional[dict]:
    """Parse a prefit sampling request id. Anything else is ordinary traffic.

    ``run`` is the session-scoped id and contains no hyphen. Commit and
    abandon are control messages, not request ids.
    """
    parts = str(rid).split("-")
    if len(parts) < 4 or parts[0] != "clearfit" or not parts[1]:
        return None
    run = parts[1]
    kind = parts[2]
    try:
        if kind == "warm" and len(parts) == 5:
            return {"kind": "warm", "run": run, "b": int(parts[3]), "n": int(parts[4])}
        if kind == "v" and len(parts) == 7:
            return {
                "kind": "v",
                "run": run,
                "b": int(parts[3]),
                "prefix": int(parts[4]),
                "k": int(parts[5]),
                "n": int(parts[6]),
            }
        if kind == "w" and len(parts) == 6:
            return {
                "kind": "w",
                "run": run,
                "b": int(parts[3]),
                "prefix": int(parts[4]),
                "n": int(parts[5]),
            }
    except (TypeError, ValueError):
        return None
    return None


def _group_key(parsed: dict):
    kind = parsed["kind"]
    if kind == "warm":
        return ("warm", int(parsed["b"]))
    if kind == "v":
        return ("v", int(parsed["b"]), int(parsed["prefix"]), int(parsed["k"]))
    if kind == "w":
        return ("w", int(parsed["b"]), int(parsed["prefix"]))
    return ("done",)


def _ordered_away(values, center: int) -> list:
    center = int(center)
    pool = [int(value) for value in values if int(value) != center and int(value) > 0]
    return sorted(pool, key=lambda value: (abs(math.log2(value) - math.log2(center)), value))


def draft_header(scheduler) -> dict:
    """Identity of this Draft. A later process must match it to reuse the file."""
    from sglang.srt.speculative.clear_tree_remote.clear_tree_attention import (
        cascade_enabled,
    )
    from sglang.srt.speculative.clear_tree_remote.clear_tree_config import (
        bonus_prediction_enabled,
    )
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        runtime_kv_buckets,
    )
    from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
        CANDIDATE_WIDTHS,
        VERIFY_KS,
    )

    args = scheduler.server_args
    widths = list(getattr(args, "clear_tree_widths", None) or CANDIDATE_WIDTHS)
    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = str(torch.cuda.get_device_name())
    except Exception:
        device = "cpu"
    worker = getattr(scheduler, "model_worker", None)
    runner = getattr(worker, "model_runner", None)
    try:
        cascade = bool(cascade_enabled(runner)) if runner is not None else False
    except Exception:
        cascade = False
    capture = getattr(scheduler, "_clear_layer_capture_bs", ()) or ()
    attention = getattr(args, "attention_backend", None)
    return {
        "schema": _FIT_SCHEMA,
        "ks": [int(k) for k in VERIFY_KS],
        "widths": [int(width) for width in widths],
        "buckets": [int(bucket) for bucket in runtime_kv_buckets()],
        "max_depth": int(getattr(args, "clear_tree_max_depth", 0) or 0),
        "draft": {
            "device": device,
            "tp": int(getattr(scheduler, "tp_size", 1) or 1),
            "model": str(getattr(args, "model_path", "") or ""),
            "page_size": int(getattr(args, "page_size", 1) or 1),
            "capture_bs": [int(item) for item in capture],
            "cascade": cascade,
            "bonus": bool(bonus_prediction_enabled()),
            "attention_backend": None if attention is None else str(attention),
            "disable_cuda_graph": bool(getattr(args, "disable_cuda_graph", False)),
        },
    }


def target_info(scheduler) -> dict:
    """Header the Target sends on PREFILL. Draft compares it with the file."""
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        runtime_kv_buckets,
    )
    from sglang.srt.speculative.clear_tree_remote.clear_tree_protocol import (
        parse_target_info,
    )
    from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
        capture_ntpb,
    )

    args = scheduler.server_args
    device = getattr(args, "device", None) or getattr(scheduler, "device", None)
    attention = getattr(args, "attention_backend", None)
    graph_bs = getattr(args, "cuda_graph_bs", None)
    name = ""
    try:
        import torch

        module = torch.get_device_module(device or "cpu")
        found = module.get_device_name()
        name = "" if found is None else str(found)
    except Exception:
        name = str(device or "")
    raw = {
        "backend": str(device or ""),
        "device": name,
        "tp": int(getattr(scheduler, "tp_size", 1) or 1),
        "model": str(getattr(args, "model_path", "") or ""),
        "served_model_name": getattr(args, "served_model_name", None),
        "dtype": None
        if getattr(args, "dtype", None) is None
        else str(getattr(args, "dtype")),
        "quantization": getattr(args, "quantization", None),
        "verify_capture": [int(item) for item in capture_ntpb("target")],
        "kv_buckets": [int(item) for item in runtime_kv_buckets()],
        "disable_cuda_graph": bool(getattr(args, "disable_cuda_graph", False)),
        "attention_backend": None if attention is None else str(attention),
        "page_size": int(getattr(args, "page_size", 1) or 1),
        "cuda_graph_bs": None if graph_bs is None else [int(item) for item in graph_bs],
        "prefit_batches": prefit_batches_for(scheduler),
        "rpc_batch_limit": rpc_batch_limit_for(scheduler),
    }
    return parse_target_info(raw) or {}


def prefit_batches_for(scheduler) -> list:
    """Batches 1, 2, and 4 that this process can run together."""
    args = getattr(scheduler, "server_args", None)
    limit = getattr(scheduler, "max_running_requests", None)
    if limit is None and args is not None:
        limit = getattr(args, "max_running_requests", None)
    limit = int(limit or 1)
    return [batch for batch in (1, 2, 4) if batch <= limit] or [1]


def _norm_target(info) -> Optional[dict]:
    if not isinstance(info, dict) or not info:
        return None
    out = {}
    for key in _TARGET_KEYS:
        if key not in info:
            continue
        value = info[key]
        if isinstance(value, list):
            out[key] = [int(item) for item in value]
        elif type(value) is int:
            out[key] = int(value)
        else:
            out[key] = value
    return out or None


def target_headers_match(left, right) -> bool:
    """True when both sides name the same Target."""
    a = _norm_target(left)
    b = _norm_target(right)
    return bool(a) and a == b


def _draft_matches(saved, live) -> bool:
    if not isinstance(saved, dict) or int(saved.get("schema", 0)) != _FIT_SCHEMA:
        return False
    if int(saved.get("schema", 0)) != int(live.get("schema", _FIT_SCHEMA)):
        return False
    for key in ("ks", "widths", "buckets"):
        if [int(item) for item in (saved.get(key) or [])] != [
            int(item) for item in (live.get(key) or [])
        ]:
            return False
    if int(saved.get("max_depth", -1)) != int(live.get("max_depth", -2)):
        return False
    return saved.get("draft") == live.get("draft")


def _read_fit(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _install_layers(scheduler, rows) -> int:
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        _new_layer_model,
        bind_ledger,
    )

    ledger = bind_ledger(scheduler)
    if ledger is None:
        return 0
    installed = 0
    fresh = {}
    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        model = _new_layer_model()
        model.from_dict(row)
        key = (int(row["w"]), str(row["path"]), int(row["g"]))
        fresh[key] = model
        installed += 1
    ledger.layer_models = fresh
    return installed


def load_draft_part(scheduler) -> None:
    """Load the Draft half of a matching file. H waits for the Target header."""
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        bind_ledger,
        ledger_enabled,
    )

    scheduler._clear_fit_pending = None
    if not ledger_enabled():
        scheduler._clear_fit_ready = None
        return
    if bind_ledger(scheduler) is None:
        scheduler._clear_fit_ready = None
        return
    path = fit_path()
    if not os.path.isfile(path):
        logger.info("[CLEAR] no fit file at %s", path)
        return
    try:
        payload = _read_fit(path)
    except Exception:
        logger.warning("[CLEAR] fit file is unreadable", exc_info=True)
        return
    header = payload.get("header") if isinstance(payload, dict) else None
    live = draft_header(scheduler)
    if not _draft_matches(header, live):
        logger.info("[CLEAR] fit file does not match this Draft; prefit required")
        return
    scheduler._clear_fit_pending = {
        "payload": payload,
        "h": list(payload.get("h") or []),
        "layers": list(payload.get("layers") or []),
        "target": header.get("target"),
    }
    logger.info("[CLEAR] stashed fit file for the Target header check")


def on_target_info(scheduler, info) -> None:
    """Install a covering fit, or clear ready when the plan no longer matches.

    Hardware identity and plan coverage are separate. An already installed
    file is checked again when the session or the plan changes.
    """
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        ledger_enabled,
    )
    from sglang.srt.speculative.clear_tree_remote.clear_tree_protocol import (
        parse_target_info,
    )

    if not ledger_enabled():
        scheduler._clear_fit_ready = None
        return
    parsed = parse_target_info(info)
    scheduler._clear_target_header = parsed
    pending = getattr(scheduler, "_clear_fit_pending", None)
    rejection = None
    if pending is not None:
        plan = draft_plan(scheduler)
        fingerprint = plan.fingerprint
        prepared = _prepare_install(scheduler, pending, parsed, plan)
        installed = bool(getattr(scheduler, "_clear_fit_installed", False))
        same_plan = getattr(scheduler, "_clear_fit_plan_fp", None) == fingerprint
        if prepared is not None:
            if installed and same_plan:
                scheduler._clear_fit_ready = True
                scheduler._clear_fit_rejection = None
                return
            if _install_prepared(scheduler, prepared):
                scheduler._clear_fit_plan_fp = fingerprint
                scheduler._clear_fit_rejection = None
                logger.info(
                    "[CLEAR] installed %d hold cells from the fit file",
                    len(prepared.get("h") or []),
                )
                return
        rejection = (parsed, fingerprint)
    scheduler._clear_fit_ready = False
    scheduler._clear_fit_installed = False
    scheduler._clear_fit_plan_fp = None
    if rejection is not None and rejection != getattr(
        scheduler, "_clear_fit_rejection", None
    ):
        logger.info("[CLEAR] fit file does not cover this Target; prefit required")
    scheduler._clear_fit_rejection = rejection


def _inputs_dict(inputs) -> dict:
    return {
        "batches": [int(item) for item in inputs.batches],
        "ks": [int(item) for item in inputs.ks],
        "widths": [int(item) for item in inputs.widths],
        "buckets": [int(item) for item in inputs.buckets],
        "max_depth": int(inputs.max_depth),
    }


def _inputs_cover(saved, live) -> bool:
    if not isinstance(saved, dict):
        return False
    for key in ("ks", "widths", "buckets"):
        left = sorted(int(item) for item in (saved.get(key) or []))
        right = sorted(int(item) for item in getattr(live, key))
        if left != right:
            return False
    if int(saved.get("max_depth", -1)) != int(live.max_depth):
        return False
    saved_batches = {int(item) for item in (saved.get("batches") or [])}
    return {int(item) for item in live.batches} <= saved_batches


def _parse_h_row(row):
    if not isinstance(row, dict):
        raise ValueError("h row")
    samples = row.get("samples_ms")
    if not isinstance(samples, list) or not samples:
        raise ValueError("h samples")
    key = (int(row["b"]), int(row["k"]), int(row["bucket"]))
    values = [float(item) for item in samples]
    return key, {
        "b": key[0],
        "k": key[1],
        "bucket": key[2],
        "samples_ms": values,
    }


def _parse_layer_row(row):
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        _new_layer_model,
    )

    if not isinstance(row, dict):
        raise ValueError("layer row")
    model = _new_layer_model()
    model.from_dict(row)
    key = (int(row["w"]), str(row["path"]), int(row["g"]))
    if model.n >= _CELL_SAMPLES and model.coefficients() is None:
        raise ValueError("layer regression does not solve")
    return key, model


def _prepare_install(scheduler, pending, target, plan):
    """Parse H and layer fits before any memory is replaced.

    A readable file that fails this check is a cache miss.
    """
    payload = pending.get("payload") if isinstance(pending, dict) else None
    if not isinstance(payload, dict) or int(payload.get("schema", 0)) != _FIT_SCHEMA:
        return None
    if not target_headers_match(pending.get("target"), target):
        return None
    saved = (payload.get("plan") or {}).get("inputs") if isinstance(payload.get("plan"), dict) else None
    if not _inputs_cover(saved, plan.inputs):
        return None
    try:
        h_rows = []
        h_index = {}
        for row in list(payload.get("h") or []):
            key, parsed = _parse_h_row(row)
            h_index[key] = parsed["samples_ms"]
            h_rows.append(parsed)
        models = {}
        for row in list(payload.get("layers") or []):
            key, model = _parse_layer_row(row)
            models[key] = model
    except (KeyError, TypeError, ValueError):
        return None
    required_layers, _skipped = layer_requirements(scheduler, plan)
    for cell in plan.required_h:
        if len(h_index.get(tuple(cell), ())) < _CELL_SAMPLES:
            return None
    for cell in required_layers:
        model = models.get(tuple(cell))
        if model is None or int(model.n) < _CELL_SAMPLES or model.coefficients() is None:
            return None
    return {"h": h_rows, "layers": models}


def _install_prepared(scheduler, prepared) -> bool:
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        bind_ledger,
    )
    from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import (
        replace_hold_cells,
    )

    ledger = bind_ledger(scheduler)
    if ledger is None or not isinstance(prepared, dict):
        return False
    replace_hold_cells(prepared.get("h") or [])
    ledger.layer_models = dict(prepared.get("layers") or {})
    scheduler._clear_fit_installed = True
    scheduler._clear_fit_ready = True
    scheduler._clear_prefit = None
    return True


def _positive_cap(value):
    if type(value) is not int or value <= 0:
        return None
    return int(value)


def rpc_batch_limit_for(scheduler) -> int:
    """Remote RPC batch cap. An unset value uses the server default of 32."""
    args = getattr(scheduler, "server_args", None)
    raw = None if args is None else getattr(args, "standalone_remote_max_batch_size", None)
    if raw is None:
        raw = getattr(scheduler, "standalone_remote_max_batch_size", None)
    cap = _positive_cap(raw)
    return 32 if cap is None else cap


def joint_batch_tuple(target_batches, draft_batches, *caps) -> tuple:
    """Batches both sides listed, and that sit under every positive cap."""

    def _ints(values):
        found = set()
        for item in values or ():
            if type(item) is bool:
                continue
            try:
                found.add(int(item))
            except (TypeError, ValueError):
                continue
        return found

    target_set = _ints(target_batches)
    draft_set = _ints(draft_batches)
    kept = [batch for batch in (1, 2, 4) if batch in target_set and batch in draft_set]
    if not kept:
        pool = draft_set or target_set or {1}
        kept = [min(pool)]
    limits = [cap for cap in (_positive_cap(item) for item in caps) if cap is not None]
    if limits:
        cap = min(limits)
        capped = [batch for batch in kept if batch <= cap]
        kept = capped or ([1] if 1 <= cap else [])
    return tuple(kept) if kept else (1,)


def joint_plan(scheduler):
    """The plan both roles can run. The fingerprint is this normalized plan."""
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        runtime_kv_buckets,
    )
    from sglang.srt.speculative.clear_tree_remote.clear_tree_prefit import (
        PlanInputs,
        build_plan,
    )
    from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
        CANDIDATE_WIDTHS,
        VERIFY_KS,
    )

    args = scheduler.server_args
    header = getattr(scheduler, "_clear_target_header", None) or {}
    announced = header.get("prefit_batches")
    if not isinstance(announced, list) or not announced:
        announced = (1,)
    local = tuple(int(item) for item in prefit_batches_for(scheduler))
    draft_running = getattr(scheduler, "max_running_requests", None)
    if draft_running is None and args is not None:
        draft_running = getattr(args, "max_running_requests", None)
    rpc_caps = [rpc_batch_limit_for(scheduler)]
    if type(header.get("rpc_batch_limit")) is int:
        rpc_caps.append(header.get("rpc_batch_limit"))
    batches = joint_batch_tuple(announced, local, draft_running, *rpc_caps)
    return build_plan(
        PlanInputs(
            batches=tuple(batches),
            ks=tuple(int(item) for item in (getattr(args, "clear_tree_ks", None) or VERIFY_KS)),
            widths=tuple(
                int(item)
                for item in (getattr(args, "clear_tree_widths", None) or CANDIDATE_WIDTHS)
            ),
            buckets=tuple(int(item) for item in runtime_kv_buckets()),
            max_depth=int(getattr(args, "clear_tree_max_depth", 0) or 0),
        )
    )


def draft_plan(scheduler):
    """Draft's copy of the joint plan. Target uses the same ``build_plan``."""
    return joint_plan(scheduler)


def layer_requirements(scheduler, plan):
    """Layer cells the surviving scan groups can actually sample.

    A cell is required only when a kept ``v`` or ``w`` group produces it and
    that ``(W, g)`` was captured. Cells with no such group are skipped.
    """
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        layer_graph_bs,
    )
    from sglang.srt.speculative.clear_tree_remote.clear_tree_prefit import (
        depth_for,
        depth_for_width,
    )
    from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
        LAYER_WIDTHS,
        flat_k,
        flat_width,
    )

    pairs = set()
    for item in getattr(scheduler, "_clear_layer_capture_pairs", ()) or ():
        if isinstance(item, (list, tuple)) and len(item) == 2:
            try:
                pairs.add((int(item[0]), int(item[1])))
            except (TypeError, ValueError):
                continue
    allowed = {int(item) for item in LAYER_WIDTHS}
    required = []
    produced_keys = set()
    b_to_g = {}

    def _graph(batch: int) -> int:
        if int(batch) not in b_to_g:
            b_to_g[int(batch)] = int(layer_graph_bs(scheduler, int(batch)))
        return int(b_to_g[int(batch)])

    for group in list(getattr(plan, "groups", None) or []):
        if group.get("kind") not in ("v", "w"):
            group["produces_layers"] = []
            continue
        batch = int(group.get("b") or 0)
        graph = _graph(batch)
        flat = int(flat_width(batch))
        produced = []
        if group.get("kind") == "w":
            if int(group.get("depth") or 0) >= 2:
                for width in plan.inputs.widths:
                    width = int(width)
                    if width == flat or width not in allowed or (width, graph) not in pairs:
                        continue
                    if (
                        depth_for_width(flat_k(batch), width, plan.inputs.max_depth)
                        < 2
                    ):
                        continue
                    produced.append((width, "graph", graph))
        elif (
            flat in allowed
            and (flat, graph) in pairs
            and int(group.get("k") or 0) > flat
            and depth_for(int(group.get("k") or 0), flat, plan.inputs.max_depth) >= 2
        ):
            produced.append((flat, "graph", graph))
        group["produces_layers"] = produced
        for key in produced:
            if key not in produced_keys:
                produced_keys.add(key)
                required.append(key)
    skipped = []
    seen = set()
    for batch in plan.inputs.batches:
        graph = _graph(int(batch))
        for width in plan.inputs.widths:
            key = (int(width), "graph", graph)
            if key in produced_keys or key in seen:
                continue
            seen.add(key)
            if int(width) not in allowed or (int(width), graph) not in pairs:
                skipped.append((key, "no_graph"))
            else:
                skipped.append((key, "no_group"))
    plan.b_to_g = b_to_g
    plan.required_layers = list(required)
    plan.skipped_layers = list(skipped)
    return required, skipped


def _skip_rows(items) -> list:
    rows = []
    for cell, reason in list(items or []):
        rows.append({"cell": [item for item in cell], "reason": str(reason)})
    return rows


def _missing(state: PrefitState):
    missing_h = []
    for cell in state.required_h:
        if len(state.h.get(tuple(cell), ())) < _CELL_SAMPLES:
            missing_h.append([int(item) for item in cell])
    missing_layers = []
    for cell in state.required_layers:
        model = state.layers.get(tuple(cell))
        if model is None or int(model.n) < _CELL_SAMPLES:
            missing_layers.append([cell[0], cell[1], cell[2]])
    return missing_h, missing_layers


def _b_to_g_pairs(plan) -> list:
    mapping = getattr(plan, "b_to_g", None) or {}
    return [[int(key), int(mapping[key])] for key in sorted(mapping)]


def _refill_ids(plan, missing_h, missing_layers) -> list:
    from sglang.srt.speculative.clear_tree_remote.clear_tree_prefit import (
        _group_id,
        groups_for_missing,
    )

    return [
        [item for item in _group_id(group)]
        for group in groups_for_missing(plan, missing_h, missing_layers)
    ]


def _status(run, seq, *, ready, reason, written=False, missing_h=None, missing_layers=None, skipped=None, b_to_g=None, refill_groups=None) -> dict:
    status = {
        "run": str(run),
        "seq": int(seq),
        "ready": bool(ready),
        "reason": str(reason),
        "missing_h": list(missing_h or []),
        "missing_layers": list(missing_layers or []),
        "skipped": list(skipped or []),
        "written": bool(written),
    }
    if b_to_g:
        status["b_to_g"] = [[int(pair[0]), int(pair[1])] for pair in b_to_g]
    if refill_groups is not None:
        status["refill_groups"] = [list(item) for item in refill_groups]
    return status


def _registered_groups(plan) -> set:
    """Request groups this plan can actually scan."""
    keys = set()
    for group in list(getattr(plan, "groups", None) or []):
        kind = group.get("kind")
        if kind == "warm":
            keys.add(("warm", int(group["b"])))
        elif kind == "v":
            keys.add(("v", int(group["b"]), int(group["prefix"]), int(group["k"])))
        elif kind == "w":
            keys.add(("w", int(group["b"]), int(group["prefix"])))
    return keys


def _packet_session(scheduler):
    live = getattr(scheduler, "sr_session_id", None)
    if live not in (None, ""):
        return str(live)
    holder = getattr(scheduler, "sr_state", None)
    live = None if holder is None else getattr(holder, "session_id", None)
    if live not in (None, ""):
        return str(live)
    return None


def _open_run(scheduler, run: str):
    """The begun run, when this packet's session still owns it."""
    state = getattr(scheduler, "_clear_prefit", None)
    if state is None or getattr(state, "run", None) != run:
        return None
    bound = getattr(state, "session_id", None)
    if bound not in (None, "") and _packet_session(scheduler) != str(bound):
        return None
    return state


def _stamp_prefit(info: dict, state, key) -> dict:
    info["session"] = str(getattr(state, "session_id", "") or "")
    info["group_key"] = [item for item in key]
    return info


def shape_check_covers(action) -> bool:
    """PREFILL, STEP, and CONTROL share the spec-shape check. FIT does not."""
    from sglang.srt.speculative.clear_tree_remote.clear_tree_protocol import (
        ClearAction,
    )

    return action in (ClearAction.PREFILL, ClearAction.STEP, ClearAction.CONTROL)


def control_shape_status(batch):
    """Receipt for a CONTROL message rejected before it can change a run."""
    from sglang.srt.speculative.clear_tree_remote.clear_tree_protocol import (
        parse_control,
    )

    control = parse_control(getattr(batch, "control", None))
    if control is None:
        return None
    return _status(
        control["run"],
        int(control["seq"]),
        ready=False,
        reason="spec_shape_mismatch",
    )


def _seen_runs(scheduler) -> list:
    seen = getattr(scheduler, "_clear_prefit_seen", None)
    if seen is None:
        seen = []
        scheduler._clear_prefit_seen = seen
    return seen


def _adopt_run(scheduler, run: str):
    """Return ``(state, stale)``. The first sight of a run replaces the previous one."""
    seen = _seen_runs(scheduler)
    state = getattr(scheduler, "_clear_prefit", None)
    if state is not None and getattr(state, "run", None) == run:
        return state, False
    if run in seen:
        return state, True
    plan = draft_plan(scheduler)
    required, skipped_layers = layer_requirements(scheduler, plan)
    state = PrefitState(run, plan)
    state.required_layers = required
    state.groups = _registered_groups(plan)
    state.skipped = _skip_rows(plan.skipped_h) + _skip_rows(skipped_layers)
    seen.append(run)
    scheduler._clear_prefit = state
    return state, False


def _fingerprint_ok(scheduler, run: str) -> bool:
    header = getattr(scheduler, "_clear_target_header", None)
    if not isinstance(header, dict) or "prefit_batches" not in header:
        return True
    fingerprint = draft_plan(scheduler).fingerprint
    text = str(run)
    if text[:8] == fingerprint:
        return True
    return fingerprint in text.split("_")


def _sample_owner_ok(scheduler, owner) -> bool:
    """True when this sample still belongs to the active begun run."""
    state = getattr(scheduler, "_clear_prefit", None)
    if state is None:
        return False
    if owner is None:
        return True
    if not isinstance(owner, dict):
        return False
    if str(owner.get("run") or "") != str(getattr(state, "run", "")):
        return False
    session = owner.get("session")
    bound = getattr(state, "session_id", None)
    if session not in (None, "") and bound not in (None, str(session)):
        return False
    group = owner.get("group_key")
    registered = getattr(state, "groups", None)
    if group is not None and registered is not None and tuple(group) not in registered:
        return False
    return True


def _active_uncommitted(scheduler):
    """The open run, or None once it has been committed or cleared."""
    state = getattr(scheduler, "_clear_prefit", None)
    if state is None or getattr(state, "ready", False):
        return None
    return state


def note_online_h(scheduler, batch, k, bucket, hold_ms) -> None:
    """Remember one ordinary H sample so commit can replay it after the snapshot."""
    state = None if scheduler is None else _active_uncommitted(scheduler)
    if state is None or hold_ms is None or bucket is None:
        return
    state.online_h.append((int(batch), int(k), int(bucket), float(hold_ms)))


def note_online_layer(scheduler, width, path, graph, features, measured) -> None:
    """Remember one ordinary layer sample for the same replay."""
    state = None if scheduler is None else _active_uncommitted(scheduler)
    if state is None or measured is None or features is None:
        return
    state.online_layers.append(
        (
            int(width),
            str(path),
            int(graph),
            [float(value) for value in features],
            float(measured),
        )
    )


def replay_online_updates(scheduler) -> None:
    """Apply ordinary samples taken during the run on top of the installed snapshot."""
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        _new_layer_model,
        bind_ledger,
    )
    from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import (
        observe_verify,
    )

    state = getattr(scheduler, "_clear_prefit", None)
    if state is None:
        return
    for batch, k, bucket, hold_ms in list(getattr(state, "online_h", None) or []):
        observe_verify(batch, k, bucket, hold_ms)
    ledger = bind_ledger(scheduler)
    if ledger is not None:
        for width, path, graph, features, measured in list(
            getattr(state, "online_layers", None) or []
        ):
            key = (int(width), str(path), int(graph))
            model = ledger.layer_models.get(key)
            if model is None:
                model = _new_layer_model()
                ledger.layer_models[key] = model
            model.update(features, float(measured))
    state.online_h = []
    state.online_layers = []


def prefit_record_h(scheduler, batch, k, bucket, hold_ms, owner=None) -> None:
    """Keep one H sample from the run that opened this round."""
    state = getattr(scheduler, "_clear_prefit", None)
    if state is None or hold_ms is None or not _sample_owner_ok(scheduler, owner):
        return
    key = (int(batch), int(k), int(bucket))
    state.h.setdefault(key, []).append(float(hold_ms))


def prefit_record_layer(scheduler, width, path, graph, features, measured, owner=None) -> None:
    """Keep one pre-reply layer sample on the run that opened this round."""
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        _new_layer_model,
    )

    state = getattr(scheduler, "_clear_prefit", None)
    if (
        state is None
        or measured is None
        or features is None
        or not _sample_owner_ok(scheduler, owner)
    ):
        return
    key = (int(width), str(path), int(graph))
    model = state.layers.get(key)
    if model is None:
        model = _new_layer_model()
        state.layers[key] = model
    model.update(features, float(measured))


def _state_rows(state: PrefitState):
    rows = []
    for key in sorted(state.h):
        samples = list(state.h.get(key) or [])
        if not samples:
            continue
        batch, k, bucket = key
        rows.append(
            {
                "b": int(batch),
                "k": int(k),
                "bucket": int(bucket),
                "samples_ms": [float(value) for value in samples],
            }
        )
    layers = []
    for key in sorted(state.layers, key=lambda item: (item[0], item[1], item[2])):
        row = {"w": int(key[0]), "path": str(key[1]), "g": int(key[2])}
        row.update(state.layers[key].to_dict())
        layers.append(row)
    return rows, layers


def _store_fit(written: dict) -> None:
    payload = written["payload"]
    path = fit_path()
    tmp = path + ".tmp"
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise
    logger.info("[CLEAR] wrote fit file %s cells=%d", path, len(written["h"]))


def _materialize_state(scheduler, state: PrefitState) -> dict:
    """Build the fit payload. The caller decides whether to store it."""
    target = getattr(scheduler, "_clear_target_header", None)
    if not isinstance(target, dict) or not target:
        raise RuntimeError("target header is unknown")
    header = draft_header(scheduler)
    header["target"] = {key: target[key] for key in _TARGET_KEYS if key in target}
    rows, layers = _state_rows(state)
    payload = {
        "schema": _FIT_SCHEMA,
        "header": header,
        "plan": {
            "inputs": _inputs_dict(state.plan.inputs),
            "fingerprint": state.plan.fingerprint,
        },
        "coverage": {
            "required_h": [[int(item) for item in cell] for cell in state.required_h],
            "required_layers": [list(cell) for cell in state.required_layers],
            "skipped": list(state.skipped),
        },
        "run": state.run,
        "h": rows,
        "layers": layers,
    }
    return {"h": rows, "layers": layers, "target": header["target"], "payload": payload}


def _install_written(scheduler, written) -> None:
    from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import (
        replace_hold_cells,
        reset_policy,
    )

    replace_hold_cells(written["h"])
    _install_layers(scheduler, written["layers"])
    reset_policy(keep_fits=True)
    replay_online_updates(scheduler)
    scheduler._clear_fit_ready = True
    scheduler._clear_fit_installed = True
    scheduler._clear_fit_pending = {
        "payload": written["payload"],
        "h": list(written["h"]),
        "layers": list(written["layers"]),
        "target": written["target"],
    }


def _active_state(scheduler, run: str, session=None):
    """The begun run for this session, or None when the caller must not mutate."""
    state = getattr(scheduler, "_clear_prefit", None)
    if state is None or getattr(state, "run", None) != run:
        return None
    bound = getattr(state, "session_id", None)
    if session is not None and bound not in (None, session):
        return None
    return state


def begin_prefit(scheduler, run: str, fingerprint: str, session: str, seq: int) -> dict:
    """Register this run on the session. A fingerprint mismatch changes nothing."""
    plan = draft_plan(scheduler)
    if str(fingerprint) != plan.fingerprint:
        return _status(run, seq, ready=False, reason="plan_mismatch")
    state, stale = _adopt_run(scheduler, run)
    if stale or state is None or state.run != run:
        return _status(run, seq, ready=False, reason="stale_run")
    state.session_id = str(session)
    state.fingerprint = str(fingerprint)
    return _status(
        run,
        seq,
        ready=False,
        reason="begun",
        b_to_g=_b_to_g_pairs(state.plan),
    )


def commit_prefit(
    scheduler,
    run: str,
    seq: int,
    session=None,
    fingerprint=None,
    write_file: bool = True,
) -> dict:
    """Accept this run only when it is active and its samples fill the plan."""
    pairs = None
    if fingerprint is not None and str(fingerprint) != draft_plan(scheduler).fingerprint:
        return _status(run, seq, ready=False, reason="plan_mismatch")
    if fingerprint is None and not _fingerprint_ok(scheduler, run):
        return _status(run, seq, ready=False, reason="plan_mismatch")
    state = _active_state(scheduler, run, session)
    if state is None:
        return _status(run, seq, ready=False, reason="stale_run")
    if fingerprint is not None and getattr(state, "fingerprint", None) not in (None, str(fingerprint)):
        return _status(run, seq, ready=False, reason="plan_mismatch")
    pairs = _b_to_g_pairs(state.plan)
    if state.ready and isinstance(state.last_status, dict) and state.last_status.get("ready"):
        echoed = dict(state.last_status)
        echoed["seq"] = int(seq)
        return echoed
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        merge_pending_verify,
    )

    merge_pending_verify(scheduler)
    missing_h, missing_layers = _missing(state)
    if missing_h or missing_layers:
        status = _status(
            run,
            seq,
            ready=False,
            reason="missing",
            missing_h=missing_h,
            missing_layers=missing_layers,
            skipped=state.skipped,
            b_to_g=pairs,
            refill_groups=_refill_ids(state.plan, missing_h, missing_layers),
        )
        state.last_status = status
        return status
    try:
        written = _materialize_state(scheduler, state)
        if write_file:
            _store_fit(written)
    except Exception:
        logger.error("[CLEAR] prefit save failed", exc_info=True)
        status = _status(
            run, seq, ready=False, reason="write_failed", skipped=state.skipped, b_to_g=pairs
        )
        state.last_status = status
        scheduler._clear_fit_ready = False
        return status
    _install_written(scheduler, written)
    scheduler._clear_fit_plan_fp = state.plan.fingerprint
    state.ready = True
    status = _status(
        run,
        seq,
        ready=True,
        reason="ready",
        written=True,
        skipped=state.skipped,
        b_to_g=pairs,
    )
    state.last_status = status
    return status


def status_prefit(scheduler, run: str, seq: int, session=None) -> dict:
    """Report this run. A query for another run does not change memory."""
    state = _active_state(scheduler, run, session)
    if state is None:
        return _status(run, seq, ready=False, reason="stale_run")
    if state.ready and isinstance(state.last_status, dict) and (
        state.last_status.get("ready") or state.last_status.get("written")
    ):
        echoed = dict(state.last_status)
        echoed["seq"] = int(seq)
        return echoed
    missing_h, missing_layers = _missing(state)
    if missing_h or missing_layers:
        return _status(
            run,
            seq,
            ready=False,
            reason="missing",
            missing_h=missing_h,
            missing_layers=missing_layers,
            skipped=state.skipped,
            b_to_g=_b_to_g_pairs(state.plan),
            refill_groups=_refill_ids(state.plan, missing_h, missing_layers),
        )
    return _status(
        run,
        seq,
        ready=False,
        reason="open",
        skipped=state.skipped,
        b_to_g=_b_to_g_pairs(state.plan),
    )


def abandon_prefit(scheduler, run: str, session=None, seq: int = 0) -> dict:
    """Cancel an uncommitted active run. A written run and an unknown run stay.

    Only the run buffer is dropped. Online H, Q, layers, and the serving
    policy stay as they were, including a fit installed by an earlier commit.
    """
    seq = int(seq)
    state = getattr(scheduler, "_clear_prefit", None)
    if state is None:
        return _status(run, seq, ready=False, reason="no_run")
    if getattr(state, "run", None) != run:
        return _status(run, seq, ready=bool(getattr(scheduler, "_clear_fit_ready", False)), reason="stale_run")
    bound = getattr(state, "session_id", None)
    if session is not None and bound not in (None, session):
        return _status(run, seq, ready=bool(getattr(scheduler, "_clear_fit_ready", False)), reason="stale_run")
    if state.ready or (
        isinstance(state.last_status, dict) and state.last_status.get("written")
    ):
        return _status(run, seq, ready=True, reason="committed", written=True)
    scheduler._clear_prefit = None
    return _status(run, seq, ready=False, reason="abandon", written=False)


def execute_control(scheduler, batch) -> dict:
    """Run one control op after the caller has checked version, shape, and session."""
    from sglang.srt.speculative.clear_tree_remote.clear_tree_protocol import (
        parse_control,
    )

    control = parse_control(getattr(batch, "control", None))
    session = str(getattr(batch, "session_id", "") or "")
    if control is None or not session:
        run = "" if control is None else control["run"]
        seq = 0 if control is None else int(control["seq"])
        return _status(run, seq, ready=False, reason="bad_control")
    op = control["op"]
    run = control["run"]
    seq = int(control["seq"])
    fingerprint = control["fingerprint"]
    write_file = int(getattr(scheduler, "tp_rank", 0) or 0) == 0
    if op == "begin":
        return begin_prefit(scheduler, run, fingerprint, session, seq)
    if op == "commit":
        return commit_prefit(
            scheduler,
            run,
            seq,
            session=session,
            fingerprint=fingerprint,
            write_file=write_file,
        )
    if op == "abandon":
        return abandon_prefit(scheduler, run, session=session, seq=seq)
    return status_prefit(scheduler, run, seq, session=session)


def _stale_plan(run: str) -> dict:
    from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
        flat_k,
        flat_width,
    )

    return {
        "group": "stale",
        "phase": "stale",
        "use_tau": True,
        "width": int(flat_width(1)),
        "k": int(flat_k(1)),
        "depth": None,
        "bucket": None,
        "run": run,
    }


def prefit_plan(scheduler, reqs, batch, max_ell, buckets) -> Optional[dict]:
    """Lock one full prefit group. A mixed packet stays on the flat tau plan."""
    del batch, max_ell
    if getattr(scheduler, "_clear_fit_ready", None) is True:
        return None
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        smallest_bucket,
    )
    from sglang.srt.speculative.clear_tree_remote.clear_tree_prefit import (
        depth_for,
        depth_for_width,
    )
    from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
        CANDIDATE_WIDTHS,
        flat_k,
        flat_width,
    )

    parsed = [parse_rid(getattr(req, "rid", "")) for req in list(reqs or [])]
    if not parsed or all(item is None for item in parsed):
        return None
    mixed = {"phase": "mixed", "group": "mixed", "use_tau": True}
    if any(item is None for item in parsed):
        return mixed
    runs = {item["run"] for item in parsed}
    if len(runs) != 1:
        return mixed
    run = parsed[0]["run"]
    state = _open_run(scheduler, run)
    if state is None:
        return _stale_plan(run)
    keys = {_group_key(item) for item in parsed}
    if len(keys) != 1:
        return mixed
    info = parsed[0]
    if len(parsed) != int(info["b"]):
        return mixed
    key = _group_key(info)
    registered = getattr(state, "groups", None)
    if registered is not None and key not in registered:
        return _stale_plan(run)
    size = int(info["b"])
    attempt = 0 if size <= 0 else min(int(item["n"]) for item in parsed) // size
    count_key = key + (int(attempt),)
    state.counts[count_key] = int(state.counts.get(count_key, 0)) + 1
    index = int(state.counts[count_key]) - 1
    args = scheduler.server_args
    max_depth = int(getattr(args, "clear_tree_max_depth", 0) or 0)
    widths = list(getattr(args, "clear_tree_widths", None) or CANDIDATE_WIDTHS)
    if info["kind"] == "warm":
        return _stamp_prefit(
            {
                "group": "warm",
                "phase": "warmup",
                "use_tau": True,
                "width": int(flat_width(size)),
                "k": int(flat_k(size)),
                "depth": None,
                "bucket": None,
                "run": run,
            },
            state,
            key,
        )
    if info["kind"] == "v":
        width = int(flat_width(size))
        k = int(info["k"])
        depth = depth_for(k, width, max_depth)
        bucket = smallest_bucket(int(info["prefix"]) + k + 2, buckets)
        return _stamp_prefit(
            {
                "group": "v",
                "phase": "warmup" if index < PREFIT_VERIFY_WARMUP_ROUNDS else "keep",
                "use_tau": False,
                "width": width,
                "k": k,
                "depth": int(depth),
                "bucket": None if bucket is None else int(bucket),
                "run": run,
            },
            state,
            key,
        )
    width_order = _ordered_away(widths, flat_width(size))
    if not width_order:
        return mixed
    chosen = int(width_order[(index // 4) % len(width_order)])
    k = int(flat_k(size))
    depth = depth_for_width(k, chosen, max_depth)
    return _stamp_prefit(
        {
            "group": "w",
            "phase": "warmup" if index % 4 == 0 else "keep",
            "use_tau": False,
            "width": chosen,
            "k": k,
            "depth": int(depth),
            "bucket": None,
            "run": run,
        },
        state,
        key,
    )


def prefit_config(scheduler):
    """Build the sweep from this Target's own server args."""
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        runtime_kv_buckets,
    )
    from sglang.srt.speculative.clear_tree_remote.clear_tree_prefit import PrefitConfig
    from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
        CANDIDATE_WIDTHS,
        VERIFY_KS,
    )

    args = scheduler.server_args
    host = getattr(args, "host", "") or ""
    if host in ("", "0.0.0.0"):
        host = "127.0.0.1"
    stored = getattr(scheduler, "_clear_joint_batches", None)
    if isinstance(stored, tuple) and stored:
        batches = stored
    else:
        batches = tuple(prefit_batches_for(scheduler))
    model_config = getattr(scheduler, "model_config", None)
    vocab = int(getattr(model_config, "vocab_size", 0) or 0)
    hi = min(vocab, 30000) if vocab else 30000
    lo = 1000 if hi > 1000 else 0
    if hi <= lo:
        hi = lo + 1
    api_key = getattr(args, "api_key", None)
    api_key = "" if api_key is None else str(api_key).strip()
    return PrefitConfig(
        url=f"http://{host}:{int(args.port)}",
        buckets=tuple(int(bucket) for bucket in runtime_kv_buckets()),
        max_depth=int(getattr(args, "clear_tree_max_depth", 0) or 0),
        ks=tuple(int(k) for k in (getattr(args, "clear_tree_ks", None) or VERIFY_KS)),
        widths=tuple(
            int(width)
            for width in (getattr(args, "clear_tree_widths", None) or CANDIDATE_WIDTHS)
        ),
        batches=batches,
        token_range=(lo, hi),
        api_key=api_key or None,
    )


def ensure_fit_box(scheduler) -> dict:
    box = getattr(scheduler, "_clear_fit_box", None)
    if not isinstance(box, dict) or "cond" not in box:
        box = {"cond": threading.Condition(), "items": {}}
        scheduler._clear_fit_box = box
    return box


def publish_fit_status(scheduler, status) -> None:
    if not isinstance(status, dict):
        return
    run = status.get("run")
    seq = status.get("seq")
    if not isinstance(run, str) or type(seq) is not int:
        return
    box = ensure_fit_box(scheduler)
    with box["cond"]:
        box["items"][(run, int(seq))] = dict(status)
        box["cond"].notify_all()


def wait_fit_status(scheduler, run: str, seq: int, timeout: float):
    box = ensure_fit_box(scheduler)
    deadline = time.monotonic() + float(timeout)
    key = (str(run), int(seq))
    with box["cond"]:
        while key not in box["items"]:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            box["cond"].wait(remaining)
        return dict(box["items"][key])


def _ensure_fit_runtime(scheduler) -> None:
    if getattr(scheduler, "_clear_fit_state", None) is None:
        scheduler._clear_fit_state = "unknown"
        scheduler._clear_fit_reason = None
        scheduler._clear_fit_run = None
        scheduler._clear_fit_timeouts = 0
        scheduler._clear_fit_next_at = 0.0


def set_fit_state(scheduler, state: str, reason=None, run=None, *, session=None) -> None:
    """Log one line each time the fit handshake changes."""
    current_session = str(getattr(scheduler, "sr_session_id", "") or "")
    if session is not None and session != current_session:
        return
    _ensure_fit_runtime(scheduler)
    snapshot = (current_session, state, reason, run)
    previous = getattr(scheduler, "_clear_fit_snapshot", None)
    scheduler._clear_fit_state = state
    scheduler._clear_fit_reason = reason
    scheduler._clear_fit_run = run
    scheduler._clear_fit_snapshot = snapshot
    if snapshot != previous:
        logger.info(
            "[CLEAR fit_state] %s",
            json.dumps(
                {
                    "state": state,
                    "run": run,
                    "reason": reason,
                    "session": current_session,
                }
            ),
        )


def mirror_fit_terminal(scheduler, snapshot) -> None:
    if snapshot is None:
        return
    session, state, reason, run = snapshot
    if state in ("ready", "failed"):
        set_fit_state(scheduler, state, reason=reason, run=run, session=session)


def fit_handshake_status(scheduler) -> dict:
    """What a FIT query should answer after ``on_target_info`` has run."""
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        ledger_enabled,
    )

    if not ledger_enabled():
        return {"state": "ledger_off", "ready": False, "reason": "ledger_off"}
    header = getattr(scheduler, "_clear_target_header", None)
    if isinstance(header, dict) and header.get("disable_cuda_graph") is True:
        return {
            "state": "graph_unsupported",
            "ready": False,
            "reason": "graph_unsupported",
        }
    if getattr(scheduler, "_clear_fit_ready", None) is True:
        return {"state": "ready", "ready": True, "reason": "ready"}
    plan = draft_plan(scheduler)
    return {
        "state": "need_prefit",
        "ready": False,
        "reason": "need_prefit",
        "prefit_batches": [int(item) for item in plan.inputs.batches],
    }


def allocate_rpc_seq(scheduler) -> int:
    """Rank 0 allocates the next RPC number and every rank stores that value."""
    seq = None
    if scheduler.tp_size == 1 or scheduler.tp_rank == 0:
        scheduler.sr_rpc_seq = int(getattr(scheduler, "sr_rpc_seq", 0) or 0) + 1
        seq = int(scheduler.sr_rpc_seq)
    if scheduler.tp_size > 1:
        seq = scheduler._sr_broadcast_obj(seq)
        scheduler.sr_rpc_seq = int(seq)
    return int(scheduler.sr_rpc_seq)


def ensure_control_queue(scheduler) -> dict:
    queue = getattr(scheduler, "_clear_control_queue", None)
    if not isinstance(queue, dict) or "lock" not in queue:
        queue = {"lock": threading.Lock(), "items": []}
        scheduler._clear_control_queue = queue
    return queue


def enqueue_control(scheduler, item: dict) -> None:
    queue = ensure_control_queue(scheduler)
    with queue["lock"]:
        queue["items"].append(dict(item))


def pop_control(scheduler):
    queue = getattr(scheduler, "_clear_control_queue", None)
    if not isinstance(queue, dict):
        return None
    with queue["lock"]:
        if not queue["items"]:
            return None
        return queue["items"].pop(0)


def start_prefit_driver(scheduler) -> bool:
    """Start a rank-0 scan within the current Target session's retry budget."""
    if int(getattr(scheduler, "tp_rank", 0) or 0) != 0:
        return False
    if os.environ.get("SGLANG_CLEAR_AUTO_PREFIT", "").strip() == "0":
        set_fit_state(scheduler, "failed", reason="auto_prefit_off")
        return False
    session = str(getattr(scheduler, "sr_session_id", "") or "")
    if getattr(scheduler, "_clear_prefit_spent_session", None) == session:
        set_fit_state(scheduler, "failed", reason="session_budget")
        return False
    current = getattr(scheduler, "_clear_prefit_thread", None)
    if current is not None and current.is_alive():
        return False
    cfg = prefit_config(scheduler)
    active_run = None

    def _submit(op, run, seq, fingerprint):
        nonlocal active_run
        if op == "begin":
            active_run = run
        enqueue_control(
            scheduler,
            {
                "op": op,
                "run": run,
                "seq": int(seq),
                "fingerprint": str(fingerprint),
            },
        )
        return None

    def _run() -> None:
        result = "failed"
        try:
            from sglang.srt.speculative.clear_tree_remote.clear_tree_prefit import (
                run_prefit,
            )

            result = run_prefit(
                cfg,
                wait_status=lambda run, seq, timeout: wait_fit_status(
                    scheduler, run, seq, timeout
                ),
                submit_control=_submit,
                session_id=session or "local",
            )
        except Exception:
            logger.warning("[CLEAR] prefit failed", exc_info=True)
            result = "failed"
        if result == "ready":
            set_fit_state(scheduler, "ready", reason="prefit", run=active_run, session=session)
        else:
            set_fit_state(
                scheduler, "failed", reason=str(result), run=active_run, session=session
            )

    thread = threading.Thread(target=_run, name="clear-prefit", daemon=True)
    scheduler._clear_prefit_thread = thread
    scheduler._clear_prefit_spent_session = session
    set_fit_state(scheduler, "prefitting", reason="need_prefit")
    logger.info("[CLEAR] prefit started")
    thread.start()
    return True


def note_joint_batches(scheduler, status) -> None:
    """Remember the batches the Draft said both sides can scan."""
    if not isinstance(status, dict):
        return
    batches = status.get("prefit_batches")
    if (
        isinstance(batches, list)
        and batches
        and all(type(item) is int and item > 0 for item in batches)
    ):
        scheduler._clear_joint_batches = tuple(int(item) for item in batches)


def apply_fit_query(scheduler, reply, now: float) -> None:
    """Move the handshake after one FIT reply, or a timeout when reply is None."""
    _ensure_fit_runtime(scheduler)
    status = getattr(reply, "fit_status", None) if reply is not None else None
    if not isinstance(status, dict) or "state" not in status:
        timeouts = int(getattr(scheduler, "_clear_fit_timeouts", 0) or 0) + 1
        scheduler._clear_fit_timeouts = timeouts
        if timeouts >= 3:
            set_fit_state(scheduler, "no_reply", reason="timeout")
            scheduler._clear_fit_next_at = float(now) + 60.0
        else:
            set_fit_state(scheduler, "retry", reason="timeout")
            delay = _QUERY_BACKOFF[min(timeouts - 1, len(_QUERY_BACKOFF) - 1)]
            scheduler._clear_fit_next_at = float(now) + float(delay)
        return
    scheduler._clear_fit_timeouts = 0
    note_joint_batches(scheduler, status)
    name = status.get("state")
    if name == "ready":
        set_fit_state(scheduler, "ready", reason=status.get("reason") or "ready", run=status.get("run"))
        return
    if name == "need_prefit":
        start_prefit_driver(scheduler)
        return
    if name == "ledger_off":
        set_fit_state(scheduler, "disabled", reason="ledger_off")
        return
    if name == "graph_unsupported":
        set_fit_state(scheduler, "disabled", reason="graph_unsupported")
        return
    set_fit_state(scheduler, "retry", reason=str(name or "unknown"))
    scheduler._clear_fit_next_at = float(now) + 5.0


def note_prefill_fit(scheduler, reply) -> None:
    """Second signal. A ready Draft that loses its fit is queried again.

    A PREFILL reply with neither ``fit_ready`` nor ``fit_status`` is an old Draft.
    """
    _ensure_fit_runtime(scheduler)
    ready = getattr(reply, "fit_ready", None)
    status = getattr(reply, "fit_status", None)
    if ready is None and not isinstance(status, dict):
        if scheduler._clear_fit_state in ("disabled", "legacy"):
            return
        set_fit_state(scheduler, "legacy", reason="no_fit_ready")
        return
    if scheduler._clear_fit_state == "ready" and ready is False:
        set_fit_state(scheduler, "querying", reason="fit_ready_false")
        scheduler._clear_fit_next_at = 0.0


def fit_query_due(scheduler, now: float) -> bool:
    """Whether rank 0 should open a FIT collective. This does not send."""
    _ensure_fit_runtime(scheduler)
    if getattr(scheduler, "sr_speculation_stopped", False):
        if scheduler._clear_fit_state != "breaker":
            set_fit_state(scheduler, "breaker", reason="breaker")
        return False
    if scheduler._clear_fit_state == "breaker":
        set_fit_state(scheduler, "retry", reason="recovered")
        scheduler._clear_fit_next_at = float(now)
    state = scheduler._clear_fit_state
    if state not in ("unknown", "retry", "no_reply", "querying"):
        return False
    if float(now) < float(scheduler._clear_fit_next_at or 0.0):
        return False
    if state != "querying":
        set_fit_state(scheduler, "querying", reason="query")
    return True


def mirror_fit_reply(scheduler, reply) -> None:
    """Copy a broadcast FIT result. Non-zero ranks do not start a scan."""
    _ensure_fit_runtime(scheduler)
    status = getattr(reply, "fit_status", None) if reply is not None else None
    if not isinstance(status, dict) or "state" not in status:
        return
    name = status.get("state")
    if name == "ready":
        set_fit_state(
            scheduler, "ready", reason=status.get("reason") or "ready", run=status.get("run")
        )
        return
    if name == "need_prefit":
        set_fit_state(scheduler, "prefitting", reason="need_prefit")
        return
    if name == "ledger_off":
        set_fit_state(scheduler, "disabled", reason="ledger_off")
        return
    if name == "graph_unsupported":
        set_fit_state(scheduler, "disabled", reason="graph_unsupported")
        return
    set_fit_state(scheduler, "retry", reason=str(name or "unknown"))


def fit_idle(scheduler, now: float, query) -> None:
    """Query only when this process is due. The caller is the idle branch."""
    if not fit_query_due(scheduler, now):
        return
    try:
        reply = query()
    except Exception:
        logger.warning("[CLEAR] fit query failed", exc_info=True)
        reply = None
    apply_fit_query(scheduler, reply, now)
