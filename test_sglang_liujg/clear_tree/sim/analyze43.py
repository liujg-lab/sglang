"""Round 43 summary. usage: python analyze43.py [prefix]"""
import glob, json, math, sys
import os
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
from collections import defaultdict

pre = sys.argv[1] if len(sys.argv) > 1 else "r43m"
rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(RES, pre + ".*.jsonl"))) for l in open(f)]
print(f"{len(rows)} jobs")
FIX = ("fix2", "fix4", "fix8", "fix16")
ARMS = ("code", "code_g", "sig_g", "sat_g", "sat10_g", "satm_g")


def stat(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, sd / math.sqrt(n), min(xs), max(xs), sum(x < -0.05 for x in xs), n


def rel(r, a, ref):
    return 100.0 * (r["res"][a]["rate"] / ref - 1.0)


def best(r):
    return max(r["res"][f]["rate"] for f in FIX)


def grp(r):
    if r["world"] == "bvar":
        return "b varies"
    return f"b={r['b']}" + (" mixed" if r["mix"] == "mixed" else "")


def line(lab, xs):
    m, se, lo, hi, neg, n = stat(xs)
    print(f"  {lab:26s} {m:+6.2f}%  se {se:4.2f}  min {lo:+6.2f}  max {hi:+6.2f}  worse {neg}/{n}")


print("\n== vs code (current mirror, old layer fit) ==")
for a in ARMS[1:] + FIX:
    line(a, [rel(r, a, r["res"]["code"]["rate"]) for r in rows])

print("\n== vs best fixed W in hindsight ==")
for a in ARMS:
    line(a, [rel(r, a, best(r)) for r in rows])

print("\n== by group, vs best fixed W ==")
by = defaultdict(list)
for r in rows:
    by[grp(r)].append(r)
for g in sorted(by):
    print(f" {g} ({len(by[g])} jobs)")
    for a in ARMS:
        line(a, [rel(r, a, best(r)) for r in by[g]])

print("\n== spread of fixed widths (best - worst among 4, 8, 16) per group ==")
for g in sorted(by):
    xs = [100.0 * (max(r["res"][f]["rate"] for f in ("fix4", "fix8", "fix16")) /
                   min(r["res"][f]["rate"] for f in ("fix4", "fix8", "fix16")) - 1.0) for r in by[g]]
    print(f"  {g:14s} mean {sum(xs) / len(xs):5.2f}%  max {max(xs):5.2f}%")

print("\n== best fixed W counts ==")
cnt = defaultdict(int)
for r in rows:
    cnt[max(FIX, key=lambda f: r["res"][f]["rate"])] += 1
print(" ", dict(cnt))

print("\n== switches per job ==")
for a in ARMS:
    sw = [len(r["res"][a]["switch"]) for r in rows]
    print(f"  {a:8s} mean {sum(sw) / len(sw):.2f}  max {max(sw)}")

print("\n== saturation rates at the served width (sat_g): rounds, bind share, half share ==")
agg = defaultdict(lambda: [0, 0.0, 0.0])
for r in rows:
    for k, v in (r["res"]["sat_g"].get("sat") or {}).items():
        bk, w = k.split(":")
        key = (grp(r), w)
        agg[key][0] += v[0]
        agg[key][1] += v[0] * v[1]
        agg[key][2] += v[0] * v[2]
for key in sorted(agg):
    n, b_, h_ = agg[key]
    if n >= 200:
        print(f"  {key[0]:14s} W={key[1]:>2s}  rounds {n:6d}  bind {b_ / n:5.3f}  half {h_ / n:5.3f}")

print("\n== per job vs best fixed ==")
for r in sorted(rows, key=lambda r: (r["b"], r["world"], r["seed"])):
    bf = max(FIX, key=lambda f: r["res"][f]["rate"])
    s = " ".join(f"{a}:{rel(r, a, best(r)):+5.1f}" for a in ARMS)
    fin = r["res"]["sat_g"]["final"]
    print(f"  {r['part']} {r['world']:9s} {r['kind']:4s} b={r['b']} s{r['seed']} best {bf:5s}  {s}  sat_final {fin}")
