"""Summaries for round 40. usage: python analyze40.py <prefix> (reads <prefix>.*.jsonl)"""
import glob, json, math, sys
import os
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
from collections import defaultdict

pre = sys.argv[1] if len(sys.argv) > 1 else "r40m"
rows = []
for f in sorted(glob.glob(os.path.join(RES, pre + ".*.jsonl"))):
    for line in open(f):
        rows.append(json.loads(line))
print(f"{len(rows)} jobs")
arms = list(rows[0]["res"])


def stat(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, sd / math.sqrt(n), min(xs), max(xs), sum(x < 0 for x in xs)


def d(r, a, ref):
    return 100.0 * (r["res"][a]["rate"] / r["res"][ref]["rate"] - 1.0)


def name(r):
    return f"{r['world']}|{r['kind']}|b{r['b']}"


print("\n== each arm vs code (all jobs) ==")
for a in arms:
    if a == "code":
        continue
    m, se, lo, hi, neg = stat([d(r, a, "code") for r in rows])
    print(f"{a:9s} mean {m:+6.2f}%  se {se:4.2f}  min {lo:+6.2f}  max {hi:+6.2f}  neg {neg}/{len(rows)}")

mixed = [r for r in rows if r["mix"] == "mixed"]
short = [r for r in rows if r["mix"] != "mixed"]
for lab, sub in (("long-prefix (mixed) worlds", mixed), ("short-prefix worlds", short)):
    print(f"\n== {lab}: {len(sub)} jobs, vs code ==")
    for a in arms:
        if a == "code":
            continue
        m, se, lo, hi, neg = stat([d(r, a, "code") for r in sub])
        print(f"{a:9s} mean {m:+6.2f}%  se {se:4.2f}  min {lo:+6.2f}  max {hi:+6.2f}  neg {neg}")

print("\n== step by step (each vs the previous step) ==")
steps = [("k1", "code"), ("kloop_nl", "k1"), ("kloop", "kloop_nl"), ("wloop_nl", "kloop_nl"), ("wloop", "kloop"),
         ("full", "wloop"), ("full_r", "wloop"), ("off_k1", "off_code"), ("kloop", "off_k1"), ("wloop", "off_k1"),
         ("full_r", "off_k1")]
for a, ref in steps:
    if a not in arms or ref not in arms:
        continue
    m, se, lo, hi, neg = stat([d(r, a, ref) for r in rows])
    print(f"{a:9s} vs {ref:9s} mean {m:+6.2f}%  se {se:4.2f}  min {lo:+6.2f}  max {hi:+6.2f}  neg {neg}/{len(rows)}")

print("\n== per world (mean over seeds), % vs code ==")
by = defaultdict(list)
for r in rows:
    by[name(r)].append(r)
hdr = " ".join(f"{a[:8]:>8s}" for a in arms if a != "code")
print(f"{'world':28s} {'code tok/s':>10s} {hdr}")
for k, rs in by.items():
    base = sum(r["res"]["code"]["rate"] for r in rs) / len(rs)
    vals = " ".join(f"{sum(d(r, a, 'code') for r in rs) / len(rs):+8.2f}" for a in arms if a != "code")
    print(f"{k:28s} {base:10.1f} {vals}")

print("\n== W switches / events (arm: jobs with any switch, rollbacks, rejects, guard trips, hold rounds) ==")
for a in arms:
    sw = sum(1 for r in rows if r["res"][a]["switch"])
    rb = sum(sum(1 for s in r["res"][a]["switch"] if s[0] == "rollback") for r in rows)
    rj = sum(sum(1 for e in r["res"][a]["events"] if e[0] == "reject") for r in rows)
    gt = sum(sum(1 for e in r["res"][a]["events"] if e[0] == "guard_trip") for r in rows)
    hold = sum(r["res"][a]["hold"] for r in rows)
    tau = sum(r["res"][a]["tau"] for r in rows)
    tot = sum(r["res"][a]["rounds"] for r in rows)
    print(f"{a:9s} switched_jobs {sw:3d} rollbacks {rb:3d} rejects {rj:3d} guard_trips {gt:3d} hold_rounds {hold:6d}"
          f" tau_share {100.0 * tau / max(tot, 1):5.1f}%")
