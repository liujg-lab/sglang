"""Round 42 summary. usage: python analyze42.py <prefix> [prefix...]"""
import glob, json, math, sys
import os
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
from collections import defaultdict

rows = []
for pre in sys.argv[1:] or ["r42m"]:
    rows += [json.loads(l) for f in sorted(glob.glob(os.path.join(RES, pre + ".*.jsonl"))) for l in open(f)]
print(f"{len(rows)} jobs")
FIX = ("fix2", "fix4", "fix8")


def stat(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, sd / math.sqrt(n), min(xs), max(xs), sum(x < 0 for x in xs), n


def rel(r, a, ref_rate):
    return 100.0 * (r["res"][a]["rate"] / ref_rate - 1.0)


def best_fixed(r):
    return max(r["res"][f]["rate"] for f in FIX)


def line(lab, xs):
    m, se, lo, hi, neg, n = stat(xs)
    print(f"  {lab:24s} mean {m:+6.2f}%  se {se:4.2f}  min {lo:+6.2f}  max {hi:+6.2f}  neg {neg}/{n}")


def grp(r):
    if r["part"] == "F":
        return "flat-W layer cost"
    if r["world"] == "bvar":
        return "b varies"
    return f"b={r['b']}"


print("\n== sig vs code ==")
line("all", [rel(r, "sig", r["res"]["code"]["rate"]) for r in rows])
by = defaultdict(list)
for r in rows:
    by[grp(r)].append(rel(r, "sig", r["res"]["code"]["rate"]))
for g in sorted(by):
    line(g, by[g])

for a in ("code", "sig"):
    print(f"\n== {a} vs best fixed W in hindsight (same job) ==")
    line("all", [rel(r, a, best_fixed(r)) for r in rows])
    by = defaultdict(list)
    for r in rows:
        by[grp(r)].append(rel(r, a, best_fixed(r)))
    for g in sorted(by):
        line(g, by[g])

print("\n== per job: best fixed, code, sig (tok/s); sig final W, switches; signal check pred/true tokens ==")
for r in sorted(rows, key=lambda r: (r["part"], r["b"], r["world"], r["seed"])):
    bf = max(FIX, key=lambda f: r["res"][f]["rate"])
    s = r["res"]["sig"]
    sw = [(x[1], x[3], x[4]) for x in s["switch"]]
    dg = {k: (round(v[0] / v[1], 2) if v[1] else None, v[2]) for k, v in s.get("diag", {}).items() if v[2] >= 5}
    print(f"  {r['part']} {r['world']:9s} {r['kind']:4s} b={r['b']} s{r['seed']}  best {bf} {r['res'][bf]['rate']:6.1f}"
          f"  code {r['res']['code']['rate']:6.1f}  sig {s['rate']:6.1f}  sw {sw}  pred/true {dg}  r_up {s.get('r_up')}")
