#!/usr/bin/env python3
"""Print ClearTree client totals and log counters for one raw/ directory.

Report prefit outcomes and observed refill groups before formal round counters.
Use Draft log order to exclude scan rounds and the prefit tag to exclude late tails.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

STAMP = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:\.\d+)?(?: TP(\d+))?\]")
WALL = re.compile(r"wall=([0-9]+(?:\.[0-9]+)?)s")
STDOUT_TOK = re.compile(r"output_tok/s=([0-9]+(?:\.[0-9]+)?)")
CAPTURE = re.compile(r"Capturing: bs=(\d+), ntpb=(\d+)")
FLUSH = re.compile(r"TP(\d+)\].*Cache flushed successfully")
HTTP_FLUSH = re.compile(r'POST /flush_cache HTTP/1\.1" (\d+)')
REASON = re.compile(r"reason=(\w+)")
RANK = re.compile(r"tp_rank=(\d+)")
BONUS_KEYS = (
    "opt_bonus_forwards",
    "opt_bonus_requests",
    "opt_bonus_graph",
    "opt_bonus_eager",
)


def stamp(line: str):
    match = STAMP.search(line)
    if match is None:
        return None, None
    when = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    return when, match.group(2)


def parse_counters(line: str):
    marker = "counters="
    start = line.find(marker)
    if start < 0 or start + len(marker) >= len(line):
        return None
    if line[start + len(marker)] != "{":
        return None
    depth = 0
    begin = start + len(marker)
    for index in range(begin, len(line)):
        char = line[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return ast.literal_eval(line[begin : index + 1])
    return None


def segments(events: list, gap_s: float) -> list:
    if not events:
        return []
    groups = [[events[0]]]
    for event in events[1:]:
        previous = groups[-1][-1][0]
        if (event[0] - previous).total_seconds() > gap_s:
            groups.append([event])
        else:
            groups[-1].append(event)
    return groups


def trial_dirs(raw: Path) -> list[Path]:
    found = []
    for child in sorted(path for path in raw.iterdir() if path.is_dir()):
        if list(child.glob("*.jsonl")):
            found.append(child)
    return found


def find_log(raw: Path, role: str) -> Path | None:
    for candidate in (raw / "sync" / f"{role}.log", raw / f"{role}.log"):
        if candidate.is_file():
            return candidate
    matches = sorted(raw.rglob(f"{role}.log"))
    return matches[0] if matches else None


def number(value):
    if value is None or value == "":
        return None
    return float(value)


def client_report(directory: Path) -> None:
    stdout = directory / "client.stdout"
    text = stdout.read_text(encoding="utf-8", errors="replace") if stdout.is_file() else ""
    walls = WALL.findall(text)
    printed = STDOUT_TOK.findall(text)
    wall_s = float(walls[-1]) if walls else None
    stdout_tok_s = printed[-1] if printed else "missing"
    for path in sorted(directory.glob("*.jsonl")):
        rows = ok = tokens = 0
        accept = draft = fallback = 0.0
        token_rows = spec_rows = fallback_rows = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows += 1
                record = json.loads(line)
                if record.get("error"):
                    continue
                ok += 1
                completion = number(record.get("completion_tokens"))
                if completion is not None:
                    tokens += int(completion)
                    token_rows += 1
                accepted = number(record.get("spec_accept_token_num"))
                drafted = number(record.get("spec_draft_token_num"))
                if accepted is not None and drafted is not None:
                    accept += accepted
                    draft += drafted
                    spec_rows += 1
                fell = number(record.get("spec_fallback_ct"))
                if fell is not None:
                    fallback += fell
                    fallback_rows += 1
        if wall_s is not None and token_rows:
            tok_s = f"{tokens / wall_s:.6f}"
            wall_text = f"{wall_s:.3f}"
        else:
            tok_s = "missing"
            wall_text = "missing" if wall_s is None else f"{wall_s:.3f}"
        accept_text = str(int(accept)) if spec_rows else "missing"
        draft_text = str(int(draft)) if spec_rows else "missing"
        fallback_text = str(int(fallback)) if fallback_rows else "missing"
        token_text = str(tokens) if token_rows else "missing"
        print(
            f"client {directory.name} file={path.name} rows={rows} ok={ok} "
            f"tokens={token_text} accept={accept_text} draft={draft_text} "
            f"fallback={fallback_text} wall_s={wall_text} tok_s={tok_s} "
            f"stdout_tok_s={stdout_tok_s}"
        )


def load_server_log(path: Path):
    tagged = []
    windows = []
    finish_at = None
    generation = 0
    captures = []
    flushes = Counter()
    http_flush = Counter()
    fit = {
        "states": {},
        "commits": [],
        "refills": [],
        "started": False,
        "written": False,
        "installed": False,
        "legacy_finished": False,
        "driver": None,
    }
    parse_errors = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            when, tp = stamp(line)
            capture = CAPTURE.search(line)
            if capture:
                captures.append((capture.group(1), capture.group(2), tp or ""))
            for match in FLUSH.finditer(line):
                flushes[match.group(1)] += 1
            for match in HTTP_FLUSH.finditer(line):
                http_flush[match.group(1)] += 1
            if "[CLEAR] wrote fit file" in line:
                fit["written"] = True
                generation += 1
                finish_at = when
            elif "[CLEAR] prefit finished" in line:
                fit["legacy_finished"] = True
                generation += 1
                finish_at = when
            elif "[CLEAR] installed " in line and "hold cells from the fit file" in line:
                fit["installed"] = True
            elif "[CLEAR] prefit started" in line:
                fit["started"] = True
            elif tp in (None, "0") and "[CLEAR] prefit driver finished state=" in line:
                tail = line.split("prefit driver finished state=", 1)[1]
                fit["driver"] = tail.strip().partition(" ")[0]
            for marker, key in (
                ("[CLEAR fit_state]", "states"),
                ("[CLEAR prefit_commit]", "commits"),
                ("[CLEAR prefit_refill_group]", "refills"),
            ):
                if marker not in line:
                    continue
                try:
                    event = json.loads(line.split(marker, 1)[1])
                    if not isinstance(event, dict):
                        raise ValueError("expected a JSON object")
                    if key == "states":
                        fit[key][tp or "0"] = event
                    elif tp in (None, "0"):
                        fit[key].append(event)
                except ValueError:
                    parse_errors += 1
                break
            if when is None:
                continue
            if "[CLEAR ledger d_round]" in line:
                payload = line.split("[CLEAR ledger d_round]", 1)[1].strip()
                try:
                    tagged.append((when, json.loads(payload), generation))
                except json.JSONDecodeError:
                    parse_errors += 1
                continue
            if "[SR " in line and " round]" in line:
                rank = RANK.search(line)
                reason = REASON.search(line)
                if rank is None or rank.group(1) != "0" or reason is None:
                    continue
                counters = parse_counters(line)
                if counters is None:
                    parse_errors += 1
                    continue
                windows.append((when, reason.group(1), counters))
    rounds = [
        (when, record)
        for when, record, seen in tagged
        if seen == generation and record.get("prefit") is None
    ]
    return rounds, windows, finish_at, captures, flushes, http_flush, fit, parse_errors


def print_captures(captures: list) -> None:
    if not captures:
        print("capture missing")
        return
    by_bs: dict[str, set[str]] = {}
    tps = set()
    for bs, ntpb, tp in captures:
        by_bs.setdefault(bs, set()).add(ntpb)
        if tp:
            tps.add(tp)
    print(f"capture_lines {len(captures)} tp={','.join(sorted(tps)) or 'missing'}")
    for bs in sorted(by_bs, key=int):
        ntpb = ",".join(sorted(by_bs[bs], key=int))
        print(f"capture bs={bs} ntpb={ntpb}")


def print_flushes(flushes: Counter, http_flush: Counter) -> None:
    if not flushes and not http_flush:
        print("flush missing")
        return
    parts = [f"tp{tp}={flushes[tp]}" for tp in sorted(flushes, key=int)]
    http_200 = http_flush.get("200", 0)
    other = sum(count for code, count in http_flush.items() if code != "200")
    print(
        "flush "
        + (" ".join(parts) if parts else "cache=missing")
        + f" http_200={http_200} http_other={other}"
    )


def round_counts(group: list) -> str:
    both = excluded = hold_null_excluded = hold_invalid = prefit_set = 0
    for _, record in group:
        hold = record.get("hold_ms")
        queued = record.get("q_ms")
        is_excluded = bool(record.get("excluded"))
        if hold is not None and queued is not None:
            both += 1
        if is_excluded:
            excluded += 1
        if hold is None and is_excluded:
            hold_null_excluded += 1
        if hold is None and not is_excluded:
            hold_invalid += 1
        if record.get("prefit") is not None:
            prefit_set += 1
    return (
        f"rounds={len(group)} both={both} excluded={excluded} "
        f"hold_null_excluded={hold_null_excluded} hold_invalid={hold_invalid} "
        f"prefit_set={prefit_set}"
    )


def window_counts(group: list) -> str:
    if not group:
        return "windows=missing tail=missing"
    reasons = Counter(reason for _, reason, _ in group)
    totals = {
        key: sum(int(counters.get(key, 0)) for _, _, counters in group)
        for key in BONUS_KEYS
    }
    last = group[-1][1]
    tail = "flushed" if last == "idle" else "missing"
    return (
        f"windows={len(group)} full={reasons.get('full', 0)} "
        f"idle={reasons.get('idle', 0)} last={last} tail={tail} "
        f"forwards={totals['opt_bonus_forwards']} "
        f"requests={totals['opt_bonus_requests']} "
        f"graph={totals['opt_bonus_graph']} eager={totals['opt_bonus_eager']}"
    )


def print_windows(label: str, group: list) -> None:
    if not group:
        print(f"{label} windows=missing tail=missing")
        return
    start = group[0][0].strftime("%Y-%m-%d %H:%M:%S")
    end = group[-1][0].strftime("%Y-%m-%d %H:%M:%S")
    print(f"{label} {start} {end} {window_counts(group)}")


def assign_windows(formal: list, windows: list) -> tuple[list, list]:
    if not formal:
        return windows, [[] for _ in formal]
    starts = [group[0][0] for group in formal]
    ends = [group[-1][0] for group in formal]
    mids = []
    for index in range(len(formal) - 1):
        gap = ends[index] + (starts[index + 1] - ends[index]) / 2
        mids.append(gap)
    prefit = []
    buckets = [[] for _ in formal]
    for window in windows:
        when = window[0]
        if when < starts[0]:
            prefit.append(window)
            continue
        placed = len(formal) - 1
        for index, mid in enumerate(mids):
            if when < mid:
                placed = index
                break
        buckets[placed].append(window)
    return prefit, buckets


def refill_group_key(event: dict, group) -> tuple | None:
    run = event.get("run")
    if (
        not isinstance(run, str)
        or not isinstance(group, list)
        or not group
        or any(type(item) not in (str, int) for item in group)
    ):
        return None
    return run, tuple(group)


def print_prefit(target: dict, draft: dict, finish_at) -> None:
    states = target.get("states", {})
    leader = states.get("0", {})
    terminal = leader.get("state")
    driver = target.get("driver")
    scan_confirmed = driver == "ready" or (
        terminal == "ready" and leader.get("reason") == "prefit"
    )
    if terminal == "failed" or (terminal != "ready" and driver == "failed"):
        outcome = "failed"
    elif draft["legacy_finished"] or (draft["written"] and scan_confirmed):
        outcome = "scanned_ready"
    elif (
        terminal == "ready"
        and draft["installed"]
        and not target.get("started")
        and not draft["written"]
    ):
        outcome = "reused"
    else:
        outcome = "unobserved"
    finish = "unobserved" if finish_at is None else finish_at.isoformat()
    print(f"prefit_status {outcome} draft_finished_at={finish}")
    if states:
        parts = [
            f"tp{rank}={value.get('state', 'unobserved')} run={value.get('run') or '-'}"
            for rank, value in sorted(states.items(), key=lambda item: int(item[0]))
        ]
        print("fit_states " + " ".join(parts))

    commits = target.get("commits", [])
    requested = set()
    for event in commits:
        group_ids = event.get("refill_groups")
        if isinstance(group_ids, list):
            for group in group_ids:
                key = refill_group_key(event, group)
                if key is not None:
                    requested.add(key)
    executions = {}
    for event in target.get("refills", []):
        key = refill_group_key(event, event.get("group_id"))
        seq = event.get("commit_seq")
        if key is not None and type(seq) is int and type(event.get("attempts")) is int:
            executions[(key, seq)] = event
    seqs = {event["seq"] for event in commits if type(event.get("seq")) is int}
    complete = (
        outcome == "scanned_ready"
        and seqs == set(range(1, len(seqs) + 1))
        and any(event.get("ready") is True for event in commits)
        and all(isinstance(event.get("refill_groups"), list) for event in commits)
    )
    requested_count = len(requested) if commits or outcome == "reused" else "unobserved"
    if executions or outcome == "reused" or (complete and not requested):
        groups = len({key for key, _ in executions})
        attempts = sum(event["attempts"] for event in executions.values())
        met = sum(event.get("step_target_met") is True for event in executions.values())
        print(
            f"prefit_refill refill_groups={groups} requested={requested_count} "
            f"executions={len(executions)} http_attempts={attempts} step_target_met={met}"
        )
    else:
        print(f"prefit_refill refill_groups=unobserved requested={requested_count}")


def server_report(path: Path | None, role: str, gap_s: float, target_fit=None):
    if path is None:
        print(f"{role}_log missing")
        return None
    (
        rounds,
        windows,
        finish_at,
        captures,
        flushes,
        http_flush,
        fit,
        errors,
    ) = load_server_log(path)
    print(f"{role}_log {path} parse_errors={errors}")
    print_captures(captures)
    if role == "target":
        print_flushes(flushes, http_flush)
        return fit
    print_prefit(target_fit or {}, fit, finish_at)
    formal = segments(rounds, gap_s)
    prefit_windows, buckets = assign_windows(formal, windows)
    print_windows("prefit_windows", prefit_windows)
    for index, group in enumerate(formal, start=1):
        start = group[0][0].strftime("%Y-%m-%d %H:%M:%S")
        end = group[-1][0].strftime("%Y-%m-%d %H:%M:%S")
        print(f"formal_segment {index} {start} {end} {round_counts(group)}")
        print_windows(f"formal_segment {index}", buckets[index - 1])
    if not formal:
        print("formal_segment missing")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="one run's raw directory")
    parser.add_argument("--gap-s", type=float, default=20.0)
    args = parser.parse_args()
    raw = args.raw
    if not raw.is_dir():
        raise SystemExit(f"not a directory: {raw}")
    for directory in trial_dirs(raw):
        client_report(directory)
    target_fit = server_report(find_log(raw, "target"), "target", args.gap_s)
    server_report(find_log(raw, "draft"), "draft", args.gap_s, target_fit)


if __name__ == "__main__":
    main()
