"""Compare greedy outputs and top-2 logprobs for the OPT gate.

``fork`` reports the first mismatch, whether the other token is in the
baseline top-2, and the top-1 minus top-2 gap in nats. ``h`` is the number of
forked requests divided by tokens compared through the fork (inclusive) or 256.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    rows.sort(key=lambda row: int(row.get("index") or 0))
    return rows


def output_ids(row: dict) -> list[int]:
    return [int(token) for token in (row.get("output_ids") or [])]


def top2(row: dict, index: int) -> list[tuple[float, int]]:
    tops = row.get("output_top_logprobs") or []
    if index >= len(tops) or not tops[index]:
        return []
    parsed = []
    for item in tops[index]:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            parsed.append((float(item[0]), int(item[1])))
    parsed.sort(key=lambda pair: pair[0], reverse=True)
    return parsed


def first_fork(left: list[int], right: list[int]) -> int | None:
    for index, (one, other) in enumerate(zip(left, right)):
        if one != other:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def compared_tokens(length: int, fork: int | None) -> int:
    if fork is None:
        return min(length, 256)
    return min(fork + 1, 256)


def pair_report(baseline: list[dict], other: list[dict], *, use_logprobs: bool) -> dict:
    count = min(len(baseline), len(other))
    forks = []
    identical = 0
    token_total = 0
    for index in range(count):
        base_ids = output_ids(baseline[index])
        other_ids = output_ids(other[index])
        fork = first_fork(base_ids, other_ids)
        token_total += compared_tokens(min(len(base_ids), len(other_ids)), fork)
        if fork is None:
            identical += 1
            continue
        other_token = other_ids[fork] if fork < len(other_ids) else None
        ranked = top2(baseline[index], fork) if use_logprobs else []
        gap = None
        in_top2 = None
        if len(ranked) >= 2:
            gap = ranked[0][0] - ranked[1][0]
            in_top2 = other_token in {ranked[0][1], ranked[1][1]}
        elif len(ranked) == 1:
            in_top2 = other_token == ranked[0][1]
        forks.append(
            {
                "request": index,
                "position": fork,
                "baseline_token": base_ids[fork] if fork < len(base_ids) else None,
                "other_token": other_token,
                "gap_nat": gap,
                "in_top2": in_top2,
                "top": ranked[:2],
            }
        )
    return {
        "requests": count,
        "identical": identical,
        "forks": forks,
        "h": (len(forks) / token_total) if token_total else None,
        "tokens_compared": token_total,
    }


def _merge_forks(reports: list[dict]) -> dict:
    forks = []
    h_values = []
    for report in reports:
        forks.extend(report["forks"])
        if report["h"] is not None:
            h_values.append(report["h"])
    return {
        "forks": forks,
        "h": max(h_values) if h_values else None,
        "requests": reports[0]["requests"] if reports else 0,
        "identical": min(report["identical"] for report in reports) if reports else 0,
    }


def item1(a1: str, a2: str, a3_paths: list[str], spec_paths: list[str]) -> dict:
    base = load_rows(a1)
    repeat = load_rows(a2)
    same = pair_report(base, repeat, use_logprobs=False)
    cross = _merge_forks([pair_report(base, load_rows(path), use_logprobs=True) for path in a3_paths])
    spec_pair = _merge_forks([pair_report(base, load_rows(path), use_logprobs=True) for path in spec_paths])
    cross_gaps = [item["gap_nat"] for item in cross["forks"] if item["gap_nat"] is not None]
    max_cross = max(cross_gaps) if cross_gaps else None
    gap_limit = max_cross if max_cross is not None else 0.1
    bug = []
    for fork in spec_pair["forks"]:
        if fork["in_top2"] is False:
            bug.append({"request": fork["request"], "reason": "not_top2", "token": fork["other_token"]})
        if fork["gap_nat"] is not None and fork["gap_nat"] > 0.5:
            bug.append({"request": fork["request"], "reason": "gap_gt_0.5", "gap_nat": fork["gap_nat"]})
    over_limit = [
        fork for fork in spec_pair["forks"]
        if fork["gap_nat"] is not None and fork["gap_nat"] > gap_limit + 1e-12
    ]
    h_limit = None if cross["h"] is None else 3.0 * cross["h"]
    h_ok = spec_pair["h"] is None or h_limit is None or spec_pair["h"] <= h_limit + 1e-12
    deterministic = same["identical"] == same["requests"] and same["requests"] > 0
    passed = deterministic and not bug and not over_limit and h_ok
    return {
        "pass": passed,
        "bug": bool(bug) or not deterministic,
        "stop": not passed,
        "deterministic": deterministic,
        "a1_vs_a2_identical": f"{same['identical']}/{same['requests']}",
        "a1_vs_a3_h": cross["h"],
        "a1_vs_a3_max_gap": max_cross,
        "gap_limit_nat": gap_limit,
        "spec_h": spec_pair["h"],
        "spec_h_limit": h_limit,
        "spec_forks": spec_pair["forks"],
        "a1_vs_a3_forks": cross["forks"],
        "bug_forks": bug,
        "gaps_over_limit": over_limit,
    }


def hcfg(paths: list[str], alens: list[float]) -> dict:
    rows = [load_rows(path) for path in paths]
    pairs = []
    worst = 0.0
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            report = pair_report(rows[left], rows[right], use_logprobs=False)
            accept = min(alens[left], alens[right])
            product = None if report["h"] is None else report["h"] * accept
            if product is not None:
                worst = max(worst, product)
            pairs.append(
                {
                    "left": paths[left],
                    "right": paths[right],
                    "identical": f"{report['identical']}/{report['requests']}",
                    "h_cfg": report["h"],
                    "accept_len": accept,
                    "h_times_accept": product,
                }
            )
    if worst <= 0.01:
        decision = "exact"
    elif worst <= 0.05:
        decision = "approximate"
    else:
        decision = "bandit"
    return {"pairs": pairs, "worst": worst, "decision": decision}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("item1", "hcfg"), required=True)
    parser.add_argument("--a1")
    parser.add_argument("--a2")
    parser.add_argument("--a3", nargs="+")
    parser.add_argument("--spec", nargs="+")
    parser.add_argument("--outputs", nargs="*")
    parser.add_argument("--alens", nargs="*")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if args.mode == "item1":
        result = item1(args.a1, args.a2, args.a3, args.spec)
    else:
        alens = [float(item) for item in args.alens or []]
        if len(alens) != len(args.outputs or []):
            raise SystemExit("--alens must match --outputs")
        result = hcfg(args.outputs, alens)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    if args.mode == "item1" and result["stop"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
