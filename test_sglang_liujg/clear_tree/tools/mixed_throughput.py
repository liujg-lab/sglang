#!/usr/bin/env python3
"""Throughput of several concurrent closed-loop pools over their common full window.

Each pool is one --stream client run started at the same time (e.g. a short
pool and a long pool). The window starts --skip-s after the latest first
request start and ends at the earliest last request start, so every pool is
full the whole time. Tokens are counted from streamed arrivals inside the window.

  python3 mixed_throughput.py --pool short=short.jsonl --pool long=long.jsonl --skip-s 60
  python3 mixed_throughput.py ... --split-at 240   # V11 load step: before/after a time offset
"""
import argparse
import json


def load(path):
    reqs = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r and r.get("arrivals"):
                reqs.append(r)
    return reqs


def tokens_in(reqs, lo, hi):
    n = 0
    for r in reqs:
        prev = 0
        for t, c in r["arrivals"]:
            if c > prev and lo <= t <= hi:
                n += c - prev
            prev = max(prev, c)
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", action="append", required=True, help="name=path")
    p.add_argument("--skip-s", type=float, default=60.0)
    p.add_argument("--split-at", type=float, default=None,
                   help="seconds after window start; report the two halves separately")
    args = p.parse_args()
    pools = {k: load(v) for k, v in (x.split("=", 1) for x in args.pool)}
    first = max(min(r["t_start"] for r in rs) for rs in pools.values())
    last = min(max(r["t_start"] for r in rs) for rs in pools.values())
    lo, hi = first + args.skip_s, last
    if hi <= lo:
        raise SystemExit("no common full window; run longer or lower --skip-s")
    spans = [("all", lo, hi)]
    if args.split_at is not None:
        cut = lo + args.split_at
        spans += [("before", lo, cut), ("after", cut, hi)]
    for name, a, b in spans:
        per = {k: tokens_in(rs, a, b) / (b - a) for k, rs in pools.items()}
        total = sum(per.values())
        parts = ", ".join(f"{k} {v:.2f}" for k, v in per.items())
        print(f"{name:>6}: {b - a:7.1f} s  total {total:.2f} tok/s  ({parts})")


if __name__ == "__main__":
    main()
