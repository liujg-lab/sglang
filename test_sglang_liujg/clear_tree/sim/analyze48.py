"""Round 48: bias of counterfactual W estimates vs fixed-run truth. usage: python analyze48.py [prefix]"""
import glob, json, math, sys
import os
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
from collections import defaultdict

import numpy as np

PRE = sys.argv[1] if len(sys.argv) > 1 else "r48"
jobs = [json.loads(l) for f in sorted(glob.glob(os.path.join(RES, PRE + ".*.jsonl"))) for l in open(f)]
print(f"{len(jobs)} jobs")
PAIRS = ((2, 4), (4, 8), (8, 16))
M = ("a", "b", "c")
LAB = {"a": "cut, same K", "b": "cut, own K", "c": "superset D, replay"}


def grp(r):
    if r["world"] == "bvar":
        return "b varies"
    return f"b={r['b']}"


pts = []
for r in jobs:
    for lo, hi in PAIRS:
        a = r["res"][f"tc{hi}"].get("tc") or {}
        if a.get("n", 0) < 30:
            continue
        truth = 100 * (r["res"][f"tc{lo}"]["rate"] / r["res"][f"tc{hi}"]["rate"] - 1)
        est = {}
        est["a"] = 100 * ((a["Ea"] / a["Ta"]) / (a["Eh"] / a["Th"]) - 1)
        est["b"] = 100 * ((a["Eb"] / a["Tb"]) / (a["Eh"] / a["Th"]) - 1)
        est["c"] = 100 * ((a["Ec"] / a["Tc"]) / (a["Ech"] / a["Tch"]) - 1)
        n = a["n"]
        pts.append(dict(r=r, lo=lo, hi=hi, grp=grp(r), truth=truth, est=est, n=n,
                        K=(a["Kh"] / n, a["Kb"] / n, a["Kc"] / n, a["Kch"] / n),
                        d=(a["dh"] / n, a["dc"] / n, a["dch"] / n), kbne=a["kb_ne"] / n, kcne=a["kc_ne"] / n,
                        ref_c=100 * ((a["Ech"] / a["Tch"]) / (a["Eh"] / a["Th"]) - 1)))
print(f"{len(pts)} (job, pair) points; estimate and truth are 'rate(W/2) / rate(W) - 1' in %")


def summ(sel, label):
    if not sel:
        return
    tr = np.array([p["truth"] for p in sel])
    print(f"\n== {label}: {len(sel)} points, truth mean {tr.mean():+.2f}%  sd {tr.std():.2f} ==")
    for m in M:
        es = np.array([p["est"][m] for p in sel])
        bias = es - tr
        big = np.abs(tr) > 1.0
        sign = (np.sign(es[big]) == np.sign(tr[big])).mean() if big.any() else float("nan")
        corr = np.corrcoef(es, tr)[0, 1] if len(sel) > 2 else float("nan")
        print(f"  {m} {LAB[m]:20s} bias {bias.mean():+5.2f}  |bias| {np.abs(bias).mean():4.2f}  sd {bias.std():4.2f}  "
              f"sign ok (|truth|>1%) {sign:4.2f} of {int(big.sum())}  corr {corr:+.2f}")


summ(pts, "all")
for lo, hi in PAIRS:
    summ([p for p in pts if p["hi"] == hi], f"W={hi} -> W={lo}")
by = defaultdict(list)
for p in pts:
    by[p["grp"]].append(p)
for g in sorted(by):
    summ(by[g], g)

print("\n== K and depth: real W_hi round K, own-K on the cut (b), replayed K at W_lo (c) and at W_hi (c ref); depths ==")
for lo, hi in PAIRS:
    s = [p for p in pts if p["hi"] == hi]
    if not s:
        continue
    K = np.mean([p["K"] for p in s], 0)
    d = np.mean([p["d"] for p in s], 0)
    print(f"  W={hi}->{lo}: K hi {K[0]:5.1f}  b {K[1]:5.1f}  c {K[2]:5.1f}  c-ref {K[3]:5.1f}   "
          f"own K differs (b) {100 * np.mean([p['kbne'] for p in s]):4.1f}%  (c) {100 * np.mean([p['kcne'] for p in s]):4.1f}%   "
          f"depth hi {d[0]:.2f}  c lo {d[1]:.2f}  c ref {d[2]:.2f}")
print("\n== c reference (replayed W_hi on the shadow superset) vs the real W_hi round: mean "
      f"{np.mean([p['ref_c'] for p in pts]):+.2f}%  sd {np.std([p['ref_c'] for p in pts]):.2f}")

print("\n== per point (truth, a, b, c) ==")
for p in sorted(pts, key=lambda p: (p["hi"], p["r"]["b"], p["r"]["world"], p["r"]["seed"])):
    r = p["r"]
    print(f"  {p['hi']:2d}->{p['lo']:<2d} {r['world']:9s} b={r['b']} s{r['seed']}  truth {p['truth']:+6.2f}  "
          + "  ".join(f"{m} {p['est'][m]:+6.2f}" for m in M))
