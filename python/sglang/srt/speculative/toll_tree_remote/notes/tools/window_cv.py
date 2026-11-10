#!/usr/bin/env python3
"""Throughput noise vs window length, from a --stream client run.

  python3 window_cv.py --run run.jsonl --skip-s 60 --windows 1000 4000 16000

Builds one token-arrival timeline from all requests, drops the first --skip-s
seconds and everything after the last request start (the pool is draining),
cuts it into windows of k generated tokens, and reports mean, CV, lag-1
autocorrelation, and how many windows per arm a paired comparison needs to
detect a 3% or 5% difference (alpha 0.05 two-sided, power 0.8).
"""
import argparse
import json
import math
import statistics


def timeline(path):
    events, starts = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "error" in r or not r.get("arrivals"):
                continue
            starts.append(r["t_start"])
            prev = 0
            for t, c in r["arrivals"]:
                if c > prev:
                    events.append((t, c - prev))
                    prev = c
    events.sort()
    return events, max(starts) if starts else None


def windows(events, t_lo, t_hi, k):
    out, acc, t0 = [], 0, None
    for t, d in events:
        if t < t_lo or t > t_hi:
            continue
        if t0 is None:
            t0 = t
            continue  # the first event only marks the start
        acc += d
        if acc >= k:
            out.append(acc / (t - t0))
            acc, t0 = 0, t
    return out


def lag1(xs):
    if len(xs) < 3:
        return float("nan")
    m = statistics.fmean(xs)
    num = sum((a - m) * (b - m) for a, b in zip(xs, xs[1:]))
    den = sum((a - m) ** 2 for a in xs)
    return num / den if den else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--skip-s", type=float, default=60.0)
    p.add_argument("--windows", type=int, nargs="+", default=[1000, 4000, 16000])
    args = p.parse_args()
    events, last_start = timeline(args.run)
    if not events:
        raise SystemExit("no streamed arrivals in run")
    t_lo = events[0][0] + args.skip_s
    t_hi = last_start
    total = sum(d for t, d in events if t_lo <= t <= t_hi)
    print(f"span {t_hi - t_lo:.1f} s, tokens {total}, mean {total / (t_hi - t_lo):.2f} tok/s")
    print(f"{'k tokens':>9} {'n':>4} {'mean tok/s':>11} {'CV':>7} {'lag1':>6} {'s/window':>9}"
          f" {'n/arm @5%':>10} {'n/arm @3%':>10}")
    z = 1.96 + 0.84
    for k in args.windows:
        xs = windows(events, t_lo, t_hi, k)
        if len(xs) < 3:
            print(f"{k:>9} {len(xs):>4}  too few windows")
            continue
        m = statistics.fmean(xs)
        cv = statistics.stdev(xs) / m
        rho = lag1(xs)
        infl = (1 + rho) / (1 - rho) if not math.isnan(rho) and -1 < rho < 1 else 1.0
        infl = max(infl, 1.0)
        need = lambda mde: math.ceil(2 * (z * cv / mde) ** 2 * infl)
        print(f"{k:>9} {len(xs):>4} {m:>11.2f} {cv:>7.3%} {rho:>6.2f} {k / m:>9.1f}"
              f" {need(0.05):>10} {need(0.03):>10}")


if __name__ == "__main__":
    main()
