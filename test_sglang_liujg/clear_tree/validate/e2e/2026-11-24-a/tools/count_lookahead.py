#!/usr/bin/env python3
"""Count Draft lookahead evidence for one raw/ directory.

Does not judge pass or fail. Prefit rounds stay out of the formal trials.
Windows have no request id, so each formal trial uses its d_round time span.
A span with no flushed rank-0 window is missing, not zero.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

SKILL = (
    Path(__file__).resolve().parents[6]
    / ".cursor/skills/clear-tree-e2e-validation/scripts/accept_logs.py"
)
spec = importlib.util.spec_from_file_location("accept_logs", SKILL)
accept = importlib.util.module_from_spec(spec)
spec.loader.exec_module(accept)

KEYS = (
    "opt_lookahead_layers",
    "opt_lookahead_group_templates",
    "opt_lookahead_fallbacks",
    "opt_lookahead_group_template_errors",
    "opt_lookahead_group_template_unshared",
    "opt_lookahead_template_fallbacks",
    "opt_lookahead_eager_layers",
    "opt_lookahead_rollbacks",
)
ADMIT = "[OPT] group lookahead admit fell back"
FELL = "[OPT] group lookahead fell back"
CAPTURED = {1, 2, 4}


def load(path: Path):
    rounds = []
    layers = []
    windows = []
    admits = []
    fells = []
    errors = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            when, _tp = accept.stamp(line)
            if ADMIT in line and when is not None:
                admits.append(when)
            if FELL in line and ADMIT not in line and when is not None:
                fells.append(when)
            if "[CLEAR ledger d_round]" in line and when is not None:
                payload = line.split("[CLEAR ledger d_round]", 1)[1].strip()
                try:
                    rounds.append((when, json.loads(payload)))
                except json.JSONDecodeError:
                    errors += 1
                continue
            if "[CLEAR ledger d_layer]" in line and when is not None:
                payload = line.split("[CLEAR ledger d_layer]", 1)[1].strip()
                try:
                    layers.append((when, json.loads(payload)))
                except json.JSONDecodeError:
                    errors += 1
                continue
            if "[SR " in line and " round]" in line and when is not None:
                rank = accept.RANK.search(line)
                reason = accept.REASON.search(line)
                if rank is None or rank.group(1) != "0" or reason is None:
                    continue
                counters = accept.parse_counters(line)
                if counters is None:
                    errors += 1
                    continue
                windows.append((when, reason.group(1), counters))
    return rounds, layers, windows, admits, fells, errors


def layer_counts(rows: list) -> str:
    one = many = eager_captured = eager_other = template_path = 0
    for _when, layer in rows:
        if layer.get("post_reply"):
            continue
        if not layer.get("lookahead"):
            continue
        active = layer.get("active")
        if active == 1:
            one += 1
        elif isinstance(active, int) and active > 1:
            many += 1
        bucket = layer.get("graph_bucket")
        if layer.get("path") == "eager" and bucket in CAPTURED:
            eager_captured += 1
        elif layer.get("path") == "eager":
            eager_other += 1
    return (
        f"pre_reply_active1_lookahead={one} "
        f"pre_reply_active_gt1_lookahead={many} "
        f"pre_reply_eager_captured_shape={eager_captured} "
        f"pre_reply_eager_other={eager_other}"
    )


def window_counts(group: list) -> str:
    if not group:
        return "windows=missing tail=missing"
    reasons = Counter(reason for _, reason, _ in group)
    totals = {
        key: sum(int(counters.get(key, 0)) for _, _, counters in group)
        for key in KEYS
    }
    last = group[-1][1]
    tail = "flushed" if last == "idle" else "missing"
    parts = [
        f"windows={len(group)}",
        f"full={reasons.get('full', 0)}",
        f"idle={reasons.get('idle', 0)}",
        f"last={last}",
        f"tail={tail}",
    ]
    parts.extend(f"{key}={totals[key]}" for key in KEYS)
    return " ".join(parts)


def in_span(when: datetime, start: datetime, end: datetime) -> bool:
    return start <= when <= end


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: count_lookahead.py <raw>")
    raw = Path(sys.argv[1])
    log = accept.find_log(raw, "draft")
    if log is None:
        raise SystemExit("draft log missing")
    rounds, layers, windows, admits, fells, errors = load(log)
    trial_rids = {}
    for directory in accept.trial_dirs(raw):
        rids = set()
        for path in sorted(directory.glob("*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("error"):
                        continue
                    meta = record.get("response_meta_info")
                    rid = meta.get("id") if isinstance(meta, dict) else None
                    if isinstance(rid, str) and rid:
                        rids.add(rid)
        trial_rids[directory.name] = rids
    formal = [(when, record) for when, record in rounds if record.get("prefit") is None]
    prefit = [(when, record) for when, record in rounds if record.get("prefit") is not None]
    groups, counts = accept.assign_trial_rounds(formal, trial_rids)
    by_key = {}
    for when, record in rounds:
        key = (record.get("session_id"), record.get("rpc_seq"))
        by_key.setdefault(key, []).append((when, record))
    print(f"draft_log {log} parse_errors={errors}")
    print(
        "round_assignment "
        + " ".join(
            f"{key}={counts[key]}"
            for key in (
                "assigned",
                "unmatched",
                "mixed_trial",
                "ambiguous_id",
                "mixed_untracked",
            )
        )
    )
    prefit_layers = []
    for when, layer in layers:
        owners = by_key.get((layer.get("session_id"), layer.get("rpc_seq")), [])
        if any(record.get("prefit") is not None for _, record in owners):
            prefit_layers.append((when, layer))
    print(f"prefit rounds={len(prefit)} {layer_counts(prefit_layers)}")
    if prefit:
        start, end = prefit[0][0], prefit[-1][0]
        span = [row for row in windows if in_span(row[0], start, end)]
        print(f"prefit {start.isoformat()} {end.isoformat()} {window_counts(span)}")
        print(
            f"prefit admit_fell_back={sum(in_span(item, start, end) for item in admits)} "
            f"group_fell_back={sum(in_span(item, start, end) for item in fells)}"
        )
    else:
        print("prefit windows=missing tail=missing")
    for trial, group in groups.items():
        if not group:
            print(f"formal_trial {trial} rounds=unobserved windows=missing tail=missing")
            continue
        start, end = group[0][0], group[-1][0]
        keys = {
            (record.get("session_id"), record.get("rpc_seq")) for _, record in group
        }
        trial_layers = [
            (when, layer)
            for when, layer in layers
            if (layer.get("session_id"), layer.get("rpc_seq")) in keys
        ]
        span = [row for row in windows if in_span(row[0], start, end)]
        print(
            f"formal_trial {trial} {start.isoformat()} {end.isoformat()} "
            f"rounds={len(group)} {layer_counts(trial_layers)}"
        )
        print(f"formal_trial {trial} {window_counts(span)}")
        print(
            f"formal_trial {trial} admit_fell_back="
            f"{sum(in_span(item, start, end) for item in admits)} "
            f"group_fell_back={sum(in_span(item, start, end) for item in fells)}"
        )


if __name__ == "__main__":
    main()
