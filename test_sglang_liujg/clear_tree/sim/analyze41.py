"""Round 41 summary. usage: python analyze41.py [prefix]"""
import glob, json, math, sys
import os
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
from collections import defaultdict

pre = sys.argv[1] if len(sys.argv) > 1 else "r41m"
rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(RES, pre + ".*.jsonl"))) for l in open(f)]
print(f"{len(rows)} jobs")


def stat(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, sd / math.sqrt(n), min(xs), max(xs), sum(x < 0 for x in xs), n


def d(r, a, ref):
    return 100.0 * (r["res"][a]["rate"] / r["res"][ref]["rate"] - 1.0)


def line(lab, xs):
    m, se, lo, hi, neg, n = stat(xs)
    print(f"  {lab:28s} mean {m:+6.2f}%  se {se:4.2f}  min {lo:+6.2f}  max {hi:+6.2f}  neg {neg}/{n}")


def group(r):
    if r["world"] == "bvar":
        return "b varies (2/8/4)"
    return f"b={r['b']}"


for a, ref in (("root", "code"), ("k1_root", "k1"), ("k1", "code")):
    print(f"\n== {a} vs {ref} ==")
    line("all", [d(r, a, ref) for r in rows])
    by = defaultdict(list)
    for r in rows:
        by[group(r)].append(d(r, a, ref))
    for g in sorted(by):
        line(g, by[g])
    by = defaultdict(list)
    for r in rows:
        by["mixed prefix" if r["mix"] == "mixed" else "short prefix"].append(d(r, a, ref))
    for g in sorted(by):
        line(g, by[g])

print("\n== per world (root vs code, mean of seeds) and share of rounds the rule moved ==")
byw = defaultdict(list)
for r in rows:
    byw[(r["world"], r["kind"], r["b"])].append(r)
for key in sorted(byw, key=lambda k: (k[2], k[0])):
    rs = byw[key]
    g = sum(d(r, "root", "code") for r in rs) / len(rs)
    mv = []
    for r in rs:
        rt = r["res"]["root"]["root"]
        tot = r["res"]["root"]["rounds"]
        mv.append((rt["root_n"] / tot, rt["root_w"] / tot))
    n_ = sum(x[0] for x in mv) / len(mv)
    w_ = sum(x[1] for x in mv) / len(mv)
    beta = rs[0]["res"]["root"]["beta"]
    print(f"  {key[0]:10s} {key[1]:5s} b={key[2]}  {g:+6.2f}%   narrow {100 * n_:5.1f}%  wider {100 * w_:4.1f}%   beta {beta}")
