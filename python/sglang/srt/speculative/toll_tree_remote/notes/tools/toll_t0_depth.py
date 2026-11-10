"""Accept-depth distribution for one OPT cell.

``q_d`` is the share of verify requests whose accept depth is at least ``d``.
Censoring ``c`` is the share whose accept depth equals the built tree depth.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from toll_t0_parse import _windows, parse_cell  # noqa: E402


def _histogram(windows: list[dict], prefix: str) -> dict[int, float]:
    totals: dict[int, float] = {}
    for window in windows:
        for key, value in window["counters"].items():
            if not key.startswith(prefix):
                continue
            depth = int(key[len(prefix):])
            totals[depth] = totals.get(depth, 0.0) + float(value or 0)
    return totals


def _censor_bounds(accept: dict[int, float], tree: dict[int, float]) -> dict:
    """Bounds on P(accept depth == tree depth) given the two histograms.

    Each request increments both counters together, and an accepted path
    cannot be deeper than the tree that was built. The joint table was not
    logged, so the script returns the minimum and maximum consistent with
    those two facts.
    """
    if not accept or sum(accept.values()) <= 0:
        return {"c_min": None, "c_max": None}
    total = sum(accept.values())

    def run(minimize: bool) -> float | None:
        remaining = dict(tree)
        equal = 0.0
        for depth in sorted(accept, reverse=True):
            left = accept[depth]
            hosts = sorted((key for key in remaining if key > depth), reverse=True)
            if not minimize:
                hosts = [depth] + hosts
            else:
                hosts = hosts + [depth]
            for host in hosts:
                use = min(left, remaining.get(host, 0.0))
                if host == depth:
                    equal += use
                remaining[host] = remaining.get(host, 0.0) - use
                left -= use
            if left > 1e-6:
                return None
        return equal / total

    return {"c_min": run(True), "c_max": run(False)}


def depth_report(draft_log: str, target_log: str, expect_b: float) -> dict:
    draft = _windows(Path(draft_log).read_text(errors="replace"), "Draft")
    accept = _histogram(draft, "opt_accept_depth_")
    tree = _histogram(draft, "opt_tree_depth_")
    accept_total = sum(accept.values())
    q = {}
    if accept_total:
        for depth in range(0, max(accept) + 1):
            q[depth] = sum(value for key, value in accept.items() if key >= depth) / accept_total
    bounds = _censor_bounds(accept, tree)
    points = []
    for depth in range(1, 5):
        if q.get(depth, 0) > 0:
            points.append((depth, math.log(q[depth])))
    slope = None
    if len(points) >= 2:
        mean_x = sum(item[0] for item in points) / len(points)
        mean_y = sum(item[1] for item in points) / len(points)
        var = sum((item[0] - mean_x) ** 2 for item in points)
        if var:
            slope = sum((item[0] - mean_x) * (item[1] - mean_y) for item in points) / var
    r = None if slope is None else math.exp(slope)
    measured = parse_cell(target_log, draft_log, expect_b, 1)
    series = sum(q.get(depth, 0.0) for depth in range(0, max(q) + 1 if q else 0))
    alen = measured.get("alen")
    consistent = None
    if alen is not None and q:
        consistent = abs((1.0 + sum(q.get(depth, 0.0) for depth in range(1, max(q) + 1))) - alen) / alen <= 0.03
    return {
        "q": {str(key): value for key, value in sorted(q.items())},
        "c_min": bounds["c_min"],
        "c_max": bounds["c_max"],
        "censored": bounds["c_min"] is not None and bounds["c_min"] > 0.20,
        "r": r,
        "accept_hist": accept,
        "tree_hist": tree,
        "alen": alen,
        "q_sum_plus_one": None if not q else 1.0 + sum(q.get(depth, 0.0) for depth in range(1, max(q) + 1)),
        "consistent_3pct": consistent,
        "series_mass": series,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-log", required=True)
    parser.add_argument("--target-log", required=True)
    parser.add_argument("--expect-b", type=float, required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    result = depth_report(args.draft_log, args.target_log, args.expect_b)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
