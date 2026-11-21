#!/usr/bin/env python3
"""Print ClearTree client totals and log counters for one raw/ directory.

Numbers only. A missing window tail stays ``tail=missing`` and is not printed
as zero. Formal rounds are the gap-separated ``d_round`` lines after the last
``prefit finished`` marker. Rank-0 bonus windows before the first formal round
stay in the prefit bucket; later windows follow the midpoint between rounds.
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
            if "prefit finished" in line:
                generation += 1
                if when is not None:
                    finish_at = when
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
    if generation == 0:
        rounds = [(when, record) for when, record, _ in tagged]
    else:
        rounds = [
            (when, record) for when, record, seen in tagged if seen == generation
        ]
    return rounds, windows, finish_at, captures, flushes, http_flush, parse_errors


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


def server_report(path: Path | None, role: str, gap_s: float) -> None:
    if path is None:
        print(f"{role}_log missing")
        return
    rounds, windows, finish_at, captures, flushes, http_flush, errors = load_server_log(
        path
    )
    print(f"{role}_log {path} parse_errors={errors}")
    print_captures(captures)
    if role == "target":
        print_flushes(flushes, http_flush)
        return
    if finish_at is None:
        print("prefit_finished missing")
    else:
        print("prefit_finished " + finish_at.strftime("%Y-%m-%d %H:%M:%S"))
    groups = segments(rounds, gap_s)
    if groups and all(record.get("prefit") is not None for _, record in groups[0]):
        tail = groups.pop(0)
        span = tail[0][0].strftime("%H:%M:%S")
        print(f"prefit_tail {span} {round_counts(tail)}")
    formal = groups
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
    server_report(find_log(raw, "target"), "target", args.gap_s)
    server_report(find_log(raw, "draft"), "draft", args.gap_s)


if __name__ == "__main__":
    main()
