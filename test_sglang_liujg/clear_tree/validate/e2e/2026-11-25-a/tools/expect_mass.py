#!/usr/bin/env python3
"""Print formal d_round accept mass for one raw block. Numbers only."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def trial_dirs(raw: Path) -> list[Path]:
    found = []
    for child in sorted(path for path in raw.iterdir() if path.is_dir()):
        if list(child.glob("*.jsonl")):
            found.append(child)
    return found


def client_rids(directory: Path) -> set[str]:
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
    return rids


def load_rounds(path: Path) -> tuple[list[dict], int]:
    tagged = []
    errors = 0
    generation = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "[CLEAR] wrote fit file" in line or "[CLEAR] prefit finished" in line:
                generation += 1
            if "[CLEAR ledger d_round]" not in line:
                continue
            payload = line.split("[CLEAR ledger d_round]", 1)[1].strip()
            try:
                record = json.loads(payload)
            except json.JSONDecodeError:
                errors += 1
                continue
            tagged.append((record, generation))
    rounds = [
        record
        for record, seen in tagged
        if seen == generation and record.get("prefit") is None
    ]
    return rounds, errors


def assign(rounds: list[dict], trial_rids: dict[str, set[str]]) -> tuple[dict, Counter]:
    owners = {}
    for trial, rids in trial_rids.items():
        for rid in rids:
            if owners.setdefault(rid, trial) != trial:
                owners[rid] = None
    groups = {trial: [] for trial in trial_rids}
    counts = Counter()
    for record in rounds:
        rids = record.get("rids")
        if not isinstance(rids, list) or not all(isinstance(rid, str) for rid in rids):
            counts["unmatched"] += 1
            continue
        matches = {owners[rid] for rid in rids if rid in owners}
        if None in matches:
            counts["ambiguous_id"] += 1
        elif len(matches) > 1:
            counts["mixed_trial"] += 1
        elif not matches:
            counts["unmatched"] += 1
        elif any(rid not in owners for rid in rids):
            counts["mixed_untracked"] += 1
        else:
            groups[next(iter(matches))].append(record)
            counts["assigned"] += 1
    return groups, counts


def number(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def scored(records: list[dict]) -> list[tuple[float, float, float]]:
    rows = []
    for record in records:
        accepted = number(record.get("accepted"))
        predicted = number(record.get("e_pred"))
        mass = number(record.get("e_mass"))
        if accepted is None or predicted is None or mass is None:
            continue
        rows.append((accepted, predicted, mass))
    return rows


def fmt(value: float) -> str:
    return f"{value:.6f}"


def ratio(num: float, den: float) -> str:
    if den == 0:
        return "missing"
    return fmt(num / den)


def emit(label: str, rows: list[tuple[float, float, float]]) -> None:
    if not rows:
        print(
            f"{label} scored=0 accepted=missing e_pred=missing e_mass=missing "
            "pred_over_mass=missing accepted_over_mass=missing"
        )
        return
    accepted = sum(row[0] for row in rows)
    predicted = sum(row[1] for row in rows)
    mass = sum(row[2] for row in rows)
    print(
        f"{label} scored={len(rows)} accepted={fmt(accepted)} "
        f"e_pred={fmt(predicted)} e_mass={fmt(mass)} "
        f"pred_over_mass={ratio(predicted, mass)} "
        f"accepted_over_mass={ratio(accepted, mass)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw
    if not raw.is_dir():
        raise SystemExit(f"not a directory: {raw}")
    draft = raw / "draft.log"
    if not draft.is_file():
        raise SystemExit(f"missing {draft}")
    trial_rids = {directory.name: client_rids(directory) for directory in trial_dirs(raw)}
    rounds, errors = load_rounds(draft)
    groups, counts = assign(rounds, trial_rids)
    print(f"parse_errors={errors} formal_rounds={len(rounds)}")
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
    for trial, records in groups.items():
        rows = scored(records)
        print(f"trial {trial} rounds={len(records)} missing_fields={len(records) - len(rows)}")
        emit(f"trial {trial}", rows)
        mid = len(rows) // 2
        emit(f"trial {trial} half=first", rows[:mid])
        emit(f"trial {trial} half=second", rows[mid:])


if __name__ == "__main__":
    main()
