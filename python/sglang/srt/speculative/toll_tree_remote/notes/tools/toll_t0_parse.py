"""Turn one OPT cell's Target and Draft logs into one CSV row.

The first ``[SR * round]`` window is dropped. Later windows whose measured
batch size is more than 10% away from ``--expect-b`` are dropped too. A cell
needs at least ``--min-windows`` survivors. Missing keys stay empty; they are
not filled in.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path


_ROUND = re.compile(r"\[SR (Target|Draft) round\]")
_SCALAR = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)=(-?\d+(?:\.\d+)?|None)\b")


def _brace_blob(text: str, key: str):
    marker = key + "="
    start = text.find(marker)
    if start < 0:
        return None
    brace = text.find("{", start)
    if brace < 0:
        return None
    depth = 0
    for index in range(brace, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return ast.literal_eval(text[brace : index + 1])
                except (SyntaxError, ValueError):
                    return None
    return None


def _windows(text: str, role: str) -> list[dict]:
    found = []
    for match in _ROUND.finditer(text):
        if match.group(1) != role:
            continue
        line = text[match.start() : text.find("\n", match.start())]
        scalars = {key: _number(value) for key, value in _SCALAR.findall(line)}
        found.append(
            {
                "rounds": scalars.get("rounds"),
                "window_wall_ms": scalars.get("window_wall_ms"),
                "host_mean_ms": _brace_blob(line, "host_mean_ms") or {},
                "counters": _brace_blob(line, "counters") or {},
                "accept_len_mean": scalars.get("accept_len_mean"),
            }
        )
    return found[1:]


def _number(value: str):
    if value == "None":
        return None
    if "." in value:
        return float(value)
    return int(value)


def _ratio(num, den):
    if num is None or den in (None, 0):
        return None
    return float(num) / float(den)


def _keep(window: dict, expect_b: float) -> bool:
    counters = window["counters"]
    actual = _ratio(counters.get("verify_requests"), counters.get("verify_batches"))
    window["actual_b"] = actual
    if actual is None or expect_b <= 0:
        return False
    return abs(actual - expect_b) / expect_b <= 0.10


def _weighted(windows: list[dict], pick):
    total = 0.0
    weight = 0.0
    for window in windows:
        rounds = window.get("rounds") or 0
        value = pick(window)
        if value is None or not rounds:
            continue
        total += float(value) * float(rounds)
        weight += float(rounds)
    if weight == 0:
        return None
    return total / weight


def _sum(windows: list[dict], key: str, *, zero_when_missing: bool = False):
    """Sum a window counter.

    A counter that was never incremented is omitted from the log. Callers pass
    ``zero_when_missing`` only for an additive term whose sibling was measured,
    so a real zero stays zero and a missing measurement stays missing.
    """
    if not any(key in window["counters"] for window in windows):
        return 0.0 if zero_when_missing else None
    return sum(float(window["counters"].get(key) or 0) for window in windows)


def _row(target: list[dict], draft: list[dict]) -> dict:
    rounds = sum(float(window.get("rounds") or 0) for window in target)
    wall = sum(
        float(window["window_wall_ms"])
        for window in target
        if window.get("window_wall_ms") is not None
    )
    accepted = _sum(target, "accepted_tokens_including_bonus")
    measured = accepted is not None
    fallback = _sum(target, "normal_decode_fallback_requests", zero_when_missing=measured)
    breaker = _sum(target, "skip_rpc_breaker_requests", zero_when_missing=measured)
    high_bs = _sum(target, "skip_rpc_high_bs_requests", zero_when_missing=measured)
    produced = None
    if accepted is not None and None not in (fallback, breaker, high_bs):
        produced = accepted + fallback + breaker + high_bs
    layer_us = _sum(draft, "opt_layer_gpu_us")
    forwards = _sum(draft, "opt_layer_forwards")
    deferred = _sum(
        draft, "opt_layer_forwards_deferred", zero_when_missing=forwards is not None
    )
    layer_count = None if forwards is None or deferred is None else forwards + deferred
    rtt_us = _sum(target, "opt_step_rtt_us")
    rtt_n = _sum(target, "opt_step_rtt_samples")
    pre_us = _sum(draft, "opt_pre_reply_us")
    draft_rounds = sum(float(window.get("rounds") or 0) for window in draft)
    prepare = _weighted(target, lambda window: window["host_mean_ms"].get("verify_prepare"))
    forward = _weighted(target, lambda window: window["host_mean_ms"].get("verify_forward"))
    commit = _weighted(
        target, lambda window: window["host_mean_ms"].get("accept_commit_including_wait")
    )
    parts = (prepare, forward, commit)
    return {
        "windows": len(target),
        "T_ms": _ratio(wall, rounds),
        "throughput_tok_s": None if produced is None or wall == 0 else produced / wall * 1000.0,
        "alen": _weighted(target, lambda window: window.get("accept_len_mean")),
        "C_L_ms": None if layer_us is None or not layer_count else layer_us / layer_count / 1000.0,
        "verify_prepare_ms": prepare,
        "verify_forward_ms": forward,
        "accept_commit_ms": commit,
        "T_t_ms": None if any(part is None for part in parts) else sum(parts),
        "draft_round_ms": _weighted(draft, lambda window: window["host_mean_ms"].get("total")),
        "R_ms": None
        if rtt_us is None or not rtt_n or pre_us is None or not draft_rounds
        else rtt_us / rtt_n / 1000.0 - pre_us / draft_rounds / 1000.0,
        "actual_b": _weighted(target, lambda window: window.get("actual_b")),
    }


def _draft_keep(window: dict, expect_b: float) -> bool:
    """Draft windows have no verify batch size. Keep them after the first drop."""
    if "verify_batches" not in window["counters"]:
        return True
    return _keep(window, expect_b)


def parse_cell(target_log: str, draft_log: str, expect_b: float, min_windows: int) -> dict:
    target = [
        window
        for window in _windows(Path(target_log).read_text(errors="replace"), "Target")
        if _keep(window, expect_b)
    ]
    draft = [
        window
        for window in _windows(Path(draft_log).read_text(errors="replace"), "Draft")
        if _draft_keep(window, expect_b)
    ]
    row = _row(target, draft)
    row["status"] = "ok" if len(target) >= min_windows and len(draft) >= min_windows else "insufficient"
    row["draft_windows"] = len(draft)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-log", required=True)
    parser.add_argument("--draft-log", required=True)
    parser.add_argument("--expect-b", type=float, required=True)
    parser.add_argument("--min-windows", type=int, default=8)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--ell", type=int, default=0)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--context", default="")
    args = parser.parse_args()
    measured = parse_cell(args.target_log, args.draft_log, args.expect_b, args.min_windows)
    record = {
        "id": args.id,
        "group": args.group,
        "N": args.n,
        "b": args.expect_b,
        "ell": args.ell,
        "tau": args.tau,
        "repeat": args.repeat,
        **measured,
    }
    if args.context:
        record.update(json.loads(Path(args.context).read_text(encoding="utf-8")))
    path = Path(args.csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        if new_file:
            writer.writeheader()
        writer.writerow(record)
    print(json.dumps(record))
    if record["status"] != "ok":
        raise SystemExit("insufficient windows")


if __name__ == "__main__":
    main()
