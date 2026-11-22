#!/usr/bin/env python3
"""Print schema-3, deferred-close, depth-ratio, and H-curve counts.

Reads one raw/ directory. Prints numbers only. Does not decide a pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


POLICY = (
    "policy_recv_ms",
    "policy_plan_ms",
    "policy_grow_ms",
    "policy_final_k_ms",
)
DIAG = ("accepted", "e_pred", "e_mass", "e_n", "tree_depth")
DEPTH_SRC = ("cell", "pooled", "none")


def find_log(raw: Path, role: str) -> Path | None:
    for candidate in (raw / f"{role}.log", raw / "sync" / f"{role}.log"):
        if candidate.is_file():
            return candidate
    matches = sorted(raw.rglob(f"{role}.log"))
    return matches[0] if matches else None


def trial_rids(raw: Path) -> dict[str, set[str]]:
    found = {}
    for child in sorted(path for path in raw.iterdir() if path.is_dir()):
        rids = set()
        for path in sorted(child.glob("*.jsonl")):
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
        if rids:
            found[child.name] = rids
    return found


def load_json_after(line: str, marker: str):
    payload = line.split(marker, 1)[1].strip()
    return json.loads(payload)


def assign(rounds: list[dict], owners: dict[str, str]) -> dict[str, list[dict]]:
    groups = {}
    for record in rounds:
        rids = record.get("rids")
        if not isinstance(rids, list) or not all(isinstance(rid, str) for rid in rids):
            continue
        matches = {owners[rid] for rid in rids if rid in owners}
        if len(matches) != 1 or any(rid not in owners for rid in rids):
            continue
        groups.setdefault(next(iter(matches)), []).append(record)
    return groups


def print_fit(path: Path | None) -> None:
    if path is None or not path.is_file():
        print("fit missing")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("h") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        print(f"fit schema={payload.get('schema') if isinstance(payload, dict) else 'missing'} h=missing")
        return
    matched = unequal = missing_ells = 0
    for row in rows:
        if not isinstance(row, dict):
            unequal += 1
            continue
        samples = row.get("samples_ms")
        ells = row.get("ells")
        if not isinstance(ells, list):
            missing_ells += 1
        elif not isinstance(samples, list) or len(ells) != len(samples):
            unequal += 1
        else:
            matched += 1
    has_q = int("q" in payload or "Q" in payload)
    print(
        f"fit schema={payload.get('schema')} h={len(rows)} "
        f"ells_match={matched} ells_unequal={unequal} ells_missing={missing_ells} "
        f"has_q={has_q}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--fit", type=Path)
    args = parser.parse_args()
    raw = args.raw
    draft_path = find_log(raw, "draft")
    if draft_path is None:
        print("draft_log missing")
        return
    text = draft_path.read_text(encoding="utf-8", errors="replace")
    print(f"no_fit_file {text.count('[CLEAR] no fit file')}")
    print(f"stashed {text.count('[CLEAR] stashed fit file')}")
    print(f"installed_from_file {text.count('hold cells from the fit file')}")
    print(f"wrote_fit_file {text.count('[CLEAR] wrote fit file')}")
    anchors = 0
    curve_on = 0
    curve_off = 0
    rounds = []
    layers = []
    for line in text.splitlines():
        if "[CLEAR] hold anchors installed" in line:
            anchors += 1
            if "curve=on" in line:
                curve_on += 1
            elif "curve=off" in line:
                curve_off += 1
        elif "[CLEAR ledger d_round]" in line:
            try:
                rounds.append(load_json_after(line, "[CLEAR ledger d_round]"))
            except json.JSONDecodeError:
                print("d_round_parse_error 1")
        elif "[CLEAR ledger d_layer]" in line:
            try:
                layers.append(load_json_after(line, "[CLEAR ledger d_layer]"))
            except json.JSONDecodeError:
                print("d_layer_parse_error 1")
    print(f"hold_anchors {anchors} curve_on {curve_on} curve_off {curve_off}")
    print_fit(args.fit)

    owners = {}
    rids = trial_rids(raw)
    for trial, ids in rids.items():
        for rid in ids:
            if owners.setdefault(rid, trial) != trial:
                owners[rid] = ""
    owners = {rid: trial for rid, trial in owners.items() if trial}
    groups = assign(
        [row for row in rounds if row.get("prefit") is None],
        owners,
    )
    by_rpc = {}
    for trial, rows in groups.items():
        both = close_true = close_false = close_missing = 0
        policy_rows = 0
        policy_max = 0.0
        diag = Counter()
        for row in rows:
            if row.get("hold_ms") is not None and row.get("q_ms") is not None:
                both += 1
            flag = row.get("close_deferred")
            if flag is True:
                close_true += 1
            elif flag is False:
                close_false += 1
            else:
                close_missing += 1
            if row.get("close_ms") is None:
                close_missing += 0
            parts = [row.get(key) for key in POLICY]
            total = row.get("policy_ms")
            if all(isinstance(part, (int, float)) for part in parts) and isinstance(total, (int, float)):
                policy_rows += 1
                policy_max = max(policy_max, abs(sum(parts) - float(total)))
            for key in DIAG:
                if key in row:
                    diag[key] += 1
            rpc = row.get("rpc_seq")
            if isinstance(rpc, int):
                by_rpc.setdefault((trial, rpc), row)
        print(
            f"trial {trial} rounds={len(rows)} hold_q={both} "
            f"close_deferred_true={close_true} close_deferred_false={close_false} "
            f"close_deferred_missing={close_missing} "
            f"close_ms_present={sum(row.get('close_ms') is not None for row in rows)} "
            f"policy_rows={policy_rows} policy_max_abs={policy_max:.6f} "
            + " ".join(f"{key}={diag[key]}" for key in DIAG)
        )

    rpc_owner = {}
    ambiguous = set()
    for trial, rpc in by_rpc:
        if rpc in ambiguous:
            continue
        previous = rpc_owner.get(rpc)
        if previous is None:
            rpc_owner[rpc] = trial
        elif previous != trial:
            ambiguous.add(rpc)
            rpc_owner.pop(rpc, None)
    depth = Counter()
    other = Counter()
    post_reply = Counter()
    for sample in layers:
        rpc = sample.get("rpc_seq")
        if not isinstance(rpc, int) or rpc not in rpc_owner:
            continue
        trial = rpc_owner[rpc]
        if sample.get("post_reply") is True and sample.get("depth_src") is None:
            post_reply[trial] += 1
            continue
        src = sample.get("depth_src")
        if src in DEPTH_SRC:
            depth[(trial, src)] += 1
        else:
            other[trial] += 1
    for trial in sorted(groups):
        print(
            f"depth {trial} "
            + " ".join(f"{src}={depth[(trial, src)]}" for src in DEPTH_SRC)
            + f" post_reply_unset={post_reply[trial]} other={other[trial]}"
        )


if __name__ == "__main__":
    main()
