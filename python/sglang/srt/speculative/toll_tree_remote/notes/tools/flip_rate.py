#!/usr/bin/env python3
"""Margin-binned flip rates.

teacher mode (primary): a run's own text scored by a reference distribution
(e.g. Target-only prefill scoring of that text). Every position counts; a flip
is a position whose emitted token is not the reference top-1. The margin is the
reference top-1 minus top-2 logprob at that position.

  python3 flip_rate.py teacher --run run.jsonl --scores scores.jsonl

pair mode: two free-running runs on the same prompts. Positions up to and
including the first divergence count; the margin comes from --ref's own top
logprobs (run with --top-logprobs).

  python3 flip_rate.py pair --ref b1.jsonl --cmp b4.jsonl

Inputs: run JSONL with idx, output_ids (and top for --ref); scores JSONL with
idx and top, top[i] = [[logprob, token_id], ...] for output position i.
"""
import argparse
import json
import math

BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 3.0), (3.0, float("inf"))]


def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if "error" not in r:
                    out.setdefault(int(r["idx"]), []).append(r)
    return out


def margin(top):
    if len(top) < 2:
        return float("inf")
    return float(top[0][0]) - float(top[1][0])


def bin_of(m):
    for i, (lo, hi) in enumerate(BINS):
        if lo <= m < hi:
            return i
    return len(BINS) - 1


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def report(pos, flips, events, title):
    print(f"== {title}")
    print(f"{'margin bin':>12} {'positions':>10} {'flips':>6} {'rate':>10} {'95% CI':>22}")
    for i, (lo, hi) in enumerate(BINS):
        n, k = pos[i], flips[i]
        lo_ci, hi_ci = wilson(k, n)
        rate = k / n if n else 0.0
        name = f"[{lo},{hi})" if hi != float("inf") else f">={lo}"
        print(f"{name:>12} {n:>10} {k:>6} {rate:>10.2e} [{lo_ci:.2e}, {hi_ci:.2e}]")
    big = [e for e in events if e["margin"] >= 1.0]
    print(f"flips with margin >= 1 nat: {len(big)}")
    for e in sorted(big, key=lambda e: -e["margin"])[:20]:
        print("  ", json.dumps(e))
    return {"bins": [{"lo": lo, "hi": (None if hi == float("inf") else hi),
                      "positions": pos[i], "flips": flips[i]}
                     for i, (lo, hi) in enumerate(BINS)], "events": events}


def teacher(args):
    runs, scores = load(args.run), load(args.scores)
    pos, flips, events = [0] * len(BINS), [0] * len(BINS), []
    missing = 0
    for idx, rs in runs.items():
        if idx not in scores:
            missing += 1
            continue
        for occ, (r, s) in enumerate(zip(rs, scores[idx])):
            ids, top = r["output_ids"], s["top"]
            for i in range(min(len(ids), len(top))):
                if not top[i]:
                    continue
                m = margin(top[i])
                b = bin_of(m)
                pos[b] += 1
                if int(top[i][0][1]) != int(ids[i]):
                    flips[b] += 1
                    rank = next((j for j, x in enumerate(top[i]) if int(x[1]) == int(ids[i])), None)
                    events.append({"idx": idx, "occ": occ, "t": i, "margin": round(m, 4),
                                   "ref": int(top[i][0][1]), "got": int(ids[i]), "rank": rank})
    if missing:
        print(f"warning: {missing} idx without scores")
    return report(pos, flips, events, f"teacher {args.run} vs {args.scores}")


def pair(args):
    ref, cmp_ = load(args.ref), load(args.cmp)
    pos, flips, events = [0] * len(BINS), [0] * len(BINS), []
    same = total = 0
    for idx, rs in ref.items():
        for occ, (a, b) in enumerate(zip(rs, cmp_.get(idx, []))):
            total += 1
            x, y, top = a["output_ids"], b["output_ids"], a.get("top") or []
            n = min(len(x), len(y), len(top))
            t = next((i for i in range(n) if x[i] != y[i]), None)
            last = n - 1 if t is None else t
            for i in range(last + 1):
                pos[bin_of(margin(top[i]))] += 1
            if t is None:
                same += int(len(x) == len(y))
                continue
            m = margin(top[t])
            flips[bin_of(m)] += 1
            rank = next((j for j, z in enumerate(top[t]) if int(z[1]) == int(y[t])), None)
            events.append({"idx": idx, "occ": occ, "t": t, "margin": round(m, 4),
                           "ref": int(x[t]), "got": int(y[t]), "rank": rank})
    print(f"identical full outputs: {same}/{total}")
    return report(pos, flips, events, f"pair {args.ref} vs {args.cmp}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("teacher")
    t.add_argument("--run", required=True)
    t.add_argument("--scores", required=True)
    t.add_argument("--json-out")
    q = sub.add_parser("pair")
    q.add_argument("--ref", required=True)
    q.add_argument("--cmp", required=True)
    q.add_argument("--json-out")
    args = p.parse_args()
    res = teacher(args) if args.mode == "teacher" else pair(args)
    if args.json_out:
        json.dump(res, open(args.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
