"""Saved ClearTree fit and the Draft side of a prefit sweep.

The file is read while the server is up and written once, when a prefit
sweep finishes. Online updates stay in memory. A matching Target header
keeps H, Q, and the layer fit across a new session.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from typing import Optional

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


class PrefitState:
    """Counts full prefit rounds and the H cells those rounds filled."""

    def __init__(self) -> None:
        self.counts = {}
        self.kept = {}
        self.finished = False
        self.cells_announced = False


def fit_path() -> str:
    override = os.environ.get("SGLANG_CLEAR_FIT_PATH", "").strip()
    if override:
        return override
    return os.path.join(os.path.dirname(__file__), "clear_tree_fit.json")


def parse_rid(rid) -> Optional[dict]:
    """Parse a prefit request id. Anything else is ordinary traffic."""
    parts = str(rid).split("-")
    if len(parts) < 3 or parts[0] != "clearfit":
        return None
    kind = parts[1]
    try:
        if kind == "warm" and len(parts) == 4:
            return {"kind": "warm", "b": int(parts[2]), "n": int(parts[3])}
        if kind == "v" and len(parts) == 6:
            return {
                "kind": "v",
                "b": int(parts[2]),
                "prefix": int(parts[3]),
                "k": int(parts[4]),
                "n": int(parts[5]),
            }
        if kind == "w" and len(parts) == 5:
            return {
                "kind": "w",
                "b": int(parts[2]),
                "prefix": int(parts[3]),
                "n": int(parts[4]),
            }
        if kind == "done" and len(parts) == 3:
            return {"kind": "done", "n": int(parts[2])}
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
        "schema": 1,
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
    }
    return parse_target_info(raw) or {}


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
    if not isinstance(saved, dict) or int(saved.get("schema", 0)) != 1:
        return False
    if int(saved.get("schema", 0)) != int(live.get("schema", 1)):
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
    layers = payload.get("layers") or []
    _install_layers(scheduler, layers)
    scheduler._clear_fit_pending = {
        "h": list(payload.get("h") or []),
        "layers": list(layers),
        "target": header.get("target"),
    }
    logger.info("[CLEAR] stashed fit file for the Target header check")


def on_target_info(scheduler, info) -> None:
    """Compare the Target header and either install H or open a prefit sweep."""
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
    if getattr(scheduler, "_clear_fit_installed", False):
        return
    pending = getattr(scheduler, "_clear_fit_pending", None)
    if pending and target_headers_match(pending.get("target"), parsed):
        from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import (
            replace_hold_cells,
        )

        rows = list(pending.get("h") or [])
        replace_hold_cells(rows)
        _install_layers(scheduler, pending.get("layers") or [])
        scheduler._clear_fit_installed = True
        scheduler._clear_fit_ready = True
        scheduler._clear_prefit = None
        logger.info("[CLEAR] installed %d hold cells from the fit file", len(rows))
        return
    scheduler._clear_fit_ready = False
    if getattr(scheduler, "_clear_prefit", None) is None:
        scheduler._clear_prefit = PrefitState()
        logger.info("[CLEAR] fit file does not match this Target; prefit open")


def _layer_rows(scheduler) -> list:
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import bind_ledger

    ledger = bind_ledger(scheduler)
    if ledger is None:
        return []
    rows = []
    for key in sorted(ledger.layer_models, key=lambda item: (item[0], item[1], item[2])):
        model = ledger.layer_models[key]
        row = {"w": int(key[0]), "path": str(key[1]), "g": int(key[2])}
        row.update(model.to_dict())
        rows.append(row)
    return rows


def save_prefit(scheduler) -> Optional[dict]:
    """Write the fit once. Refuse when the Target is unknown or H is empty.

    A successful write returns the H rows, layer rows, and Target header that
    went into the file. Refusal returns None and leaves the file alone.
    """
    from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import (
        export_hold_cells,
    )

    target = getattr(scheduler, "_clear_target_header", None)
    if not target:
        logger.warning("[CLEAR] prefit save refused: Target header is unknown")
        return None
    rows = export_hold_cells()
    if not rows:
        logger.warning("[CLEAR] prefit save refused: no hold cells")
        return None
    header = draft_header(scheduler)
    header["target"] = target
    layers = _layer_rows(scheduler)
    payload = {
        "schema": 1,
        "header": header,
        "h": rows,
        "layers": layers,
    }
    path = fit_path()
    tmp = path + ".tmp"
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.write("\n")
    os.replace(tmp, path)
    logger.info("[CLEAR] wrote fit file %s cells=%d", path, len(rows))
    return {"h": rows, "layers": layers, "target": target}


def expected_cells() -> list:
    """H cells a full default sweep fills: batch, bucket, and K."""
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        runtime_kv_buckets,
    )
    from sglang.srt.speculative.clear_tree_remote.clear_tree_width import VERIFY_KS

    cells = []
    for batch in (1, 2, 4):
        for bucket in runtime_kv_buckets():
            for k in VERIFY_KS:
                cells.append((int(batch), int(k), int(bucket)))
    return cells


def _cells_ready(state: PrefitState) -> bool:
    return all(int(state.kept.get(cell, 0)) >= 4 for cell in expected_cells())


def note_kept(scheduler, batch, k, bucket) -> None:
    """Count one retained v-group round. A full grid does not end the sweep."""
    state = getattr(scheduler, "_clear_prefit", None)
    if state is None or state.finished:
        return
    key = (int(batch), int(k), int(bucket))
    state.kept[key] = int(state.kept.get(key, 0)) + 1
    if _cells_ready(state) and not state.cells_announced:
        state.cells_announced = True
        logger.info("[CLEAR] cells complete")


def maybe_finish(scheduler, reason: str) -> None:
    """Save, drop theta, and close the sweep when the done request arrives.

    A full grid does not finish the sweep. A save error is logged and still
    closes it, so a later session cannot open a second sweep in this process.
    """
    state = getattr(scheduler, "_clear_prefit", None)
    if state is None or state.finished:
        return
    if reason != "clearfit-done":
        return
    state.finished = True
    written = None
    try:
        written = save_prefit(scheduler)
    except Exception:
        logger.error("[CLEAR] prefit save failed", exc_info=True)
    missing = [cell for cell in expected_cells() if int(state.kept.get(cell, 0)) < 4]
    logger.info("[CLEAR] prefit finished reason=%s missing=%s", reason, missing)
    from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import reset_policy

    reset_policy(keep_fits=True)
    scheduler._clear_prefit = None
    scheduler._clear_fit_ready = True
    scheduler._clear_fit_installed = True
    if written:
        scheduler._clear_fit_pending = {
            "h": list(written["h"]),
            "layers": list(written["layers"]),
            "target": written["target"],
        }


def prefit_plan(scheduler, reqs, batch, max_ell, buckets) -> Optional[dict]:
    """Lock one full prefit group. A mixed packet stays on the flat tau plan."""
    del max_ell
    if getattr(scheduler, "_clear_fit_ready", None) is True:
        return None
    state = getattr(scheduler, "_clear_prefit", None)
    if state is None or state.finished:
        return None
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        smallest_bucket,
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
    keys = {_group_key(item) for item in parsed}
    if len(keys) != 1:
        return mixed
    info = parsed[0]
    if info["kind"] == "done" or len(parsed) != int(info["b"]):
        return mixed
    key = _group_key(info)
    size = int(info["b"])
    attempt = 0 if size <= 0 else min(int(item["n"]) for item in parsed) // size
    count_key = key + (int(attempt),)
    state.counts[count_key] = int(state.counts.get(count_key, 0)) + 1
    index = int(state.counts[count_key]) - 1
    args = scheduler.server_args
    max_depth = int(getattr(args, "clear_tree_max_depth", 0) or 0)
    widths = list(getattr(args, "clear_tree_widths", None) or CANDIDATE_WIDTHS)
    if info["kind"] == "warm":
        return {
            "group": "warm",
            "phase": "warmup",
            "use_tau": True,
            "width": int(flat_width(size)),
            "k": int(flat_k(size)),
            "depth": None,
            "bucket": None,
        }
    if info["kind"] == "v":
        width = int(flat_width(size))
        k = int(info["k"])
        depth = 0 if not width else max(0, min(math.ceil(k / float(width)), max_depth - 1))
        bucket = smallest_bucket(int(info["prefix"]) + k + 2, buckets)
        return {
            "group": "v",
            "phase": "warmup" if index < 2 else "keep",
            "use_tau": False,
            "width": width,
            "k": k,
            "depth": int(depth),
            "bucket": None if bucket is None else int(bucket),
        }
    width_order = _ordered_away(widths, flat_width(size))
    if not width_order:
        return mixed
    chosen = int(width_order[(index // 4) % len(width_order)])
    k = int(flat_k(size))
    depth = 0 if not chosen else max(0, min(math.ceil(k / float(chosen)), max_depth - 1))
    return {
        "group": "w",
        "phase": "warmup" if index % 4 == 0 else "keep",
        "use_tau": False,
        "width": chosen,
        "k": k,
        "depth": int(depth),
        "bucket": None,
    }


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
    limit = getattr(scheduler, "max_running_requests", None)
    if limit is None:
        limit = getattr(args, "max_running_requests", None)
    limit = int(limit or 1)
    batches = tuple(batch for batch in (1, 2, 4) if batch <= limit) or (1,)
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


def maybe_start_prefit(scheduler, fit_ready) -> None:
    """Start at most two sweeps per process, and never while one is running."""
    if fit_ready is True:
        if not getattr(scheduler, "_clear_fit_skip_logged", False):
            logger.info("[CLEAR] 已有拟合文件，跳过预拟合")
            scheduler._clear_fit_skip_logged = True
        return
    if fit_ready is not False:
        return
    if os.environ.get("SGLANG_CLEAR_AUTO_PREFIT", "").strip() == "0":
        return
    current = getattr(scheduler, "_clear_prefit_thread", None)
    if current is not None and current.is_alive():
        return
    runs = int(getattr(scheduler, "_clear_prefit_runs", 0) or 0)
    if runs >= 2:
        return
    cfg = prefit_config(scheduler)

    def _run() -> None:
        try:
            from sglang.srt.speculative.clear_tree_remote.clear_tree_prefit import (
                run_prefit,
            )

            run_prefit(cfg)
        except Exception:
            logger.warning("[CLEAR] prefit failed", exc_info=True)

    thread = threading.Thread(target=_run, name="clear-prefit", daemon=True)
    scheduler._clear_prefit_runs = runs + 1
    scheduler._clear_prefit_thread = thread
    thread.start()
    logger.info("[CLEAR] prefit started")
