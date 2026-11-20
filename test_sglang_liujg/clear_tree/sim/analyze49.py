"""Round 49 summary: merge r43m, r45a, r47, r49 by job. usage: python analyze49.py"""
import glob, json, math
import os
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
from collections import defaultdict, Counter

rows = {}
for pre in ("r43m", "r45a", "r45b", "r47", "r49"):
    for f in sorted(glob.glob(os.path.join(RES, pre + ".*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            k = (r["part"], r["world"], r["b"], r["seed"], str(r["x"]))
            if k not in rows:
                rows[k] = dict(meta=r, res={})
            rows[k]["res"].update(r["res"])
FIX = ("fix2", "fix4", "fix8", "fix16")
WANT = ("code", "sat_g", "satpv_g", "net0_g", "pair_g", "pair_s")
jobs = [v for v in rows.values() if all(a in v["res"] for a in ("code", "sat_g") + FIX)]
ARMS = [a for a in WANT if all(a in v["res"] for v in jobs)]
print(f"{len(jobs)} jobs; arms {ARMS}")


def stat(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, sd / math.sqrt(n), min(xs), n


def best(v):
    return max(v["res"][f]["rate"] for f in FIX)


def bestw(v):
    return int(max(FIX, key=lambda f: v["res"][f]["rate"])[3:])


def rel(v, a, ref):
    return 100.0 * (v["res"][a]["rate"] / ref - 1.0)


def grp(v):
    r = v["meta"]
    if r["world"] == "bvar":
        return "b varies"
    return f"b={r['b']}" + (" mixed" if r["mix"] == "mixed" else "")


def line(lab, xs):
    m, se, lo, n = stat(xs)
    print(f"  {lab:12s} {m:+6.2f}%  se {se:4.2f}  min {lo:+6.2f}  n {n}")


print("\n== vs best fixed W in hindsight ==")
for a in ARMS:
    line(a, [rel(v, a, best(v)) for v in jobs])
print("\n== paired vs sat_g ==")
for a in ARMS:
    if a != "sat_g":
        xs = [rel(v, a, v["res"]["sat_g"]["rate"]) for v in jobs]
        m, se, lo, n = stat(xs)
        print(f"  {a:12s} {m:+6.2f}%  se {se:4.2f}  better {sum(x > 0 for x in xs)}/{n}")
print("\n== by group, vs best fixed ==")
by = defaultdict(list)
for v in jobs:
    by[grp(v)].append(v)
for g in sorted(by):
    print(f" {g} ({len(by[g])})  " + "  ".join(f"{a} {stat([rel(v, a, best(v)) for v in by[g]])[0]:+.2f}" for a in ARMS))


def switches(v, a):
    out = []
    for x in v["res"][a]["switch"]:
        x = list(x)
        if x[3] is None and isinstance(x[4], int):
            x[3] = 8 if x[2] == 1 else 4
        if isinstance(x[3], int) and isinstance(x[4], int):
            out.append(x)
        elif len(x) >= 5 and isinstance(x[-2], int) and isinstance(x[-1], int):
            out.append([x[0], x[1], x[2], x[-2], x[-1]])
    return out


print("\n== switches per job, reversals, final W vs best fixed (over/under/eq), probe share ==")
for a in ARMS:
    sw = rev = pairs = 0
    fin = Counter()
    pr = 0.0
    for v in jobs:
        s = switches(v, a)
        sw += len(s)
        per = defaultdict(list)
        for x in s:
            per[x[2]].append((x[3], x[4]))
        for l in per.values():
            for p, q in zip(l, l[1:]):
                pairs += 1
                rev += p[0] == q[1] and p[1] == q[0]
        if v["meta"]["world"] != "bvar":
            f = list(v["res"][a]["final"].values())
            fw = f[0] if f else 4
            bw = bestw(v)
            fin["over" if fw > bw else ("under" if fw < bw else "eq")] += 1
        kd = v["res"][a]["kind"]
        tot = sum(kd.values())
        pr += (sum(n for k, n in kd.items() if k not in ("ordinary", "off", "hold")) / tot) if tot else 0.0
    print(f"  {a:12s} sw/job {sw / len(jobs):4.2f}  reversals {rev}/{pairs}  final {dict(fin)}  "
          f"non-served rounds {100 * pr / len(jobs):4.1f}%")

for a in ("satpv_g", "pair_g", "pair_s"):
    if a not in ARMS:
        continue
    ev = Counter()
    why = Counter()
    for v in jobs:
        ev.update(v["res"][a].get("events", {}))
        why.update(str(x[5]) for x in v["res"][a]["switch"])
    print(f"\n== {a} events {dict(ev)}  switch reasons {dict(why)}")

print("\n== b=8 final W (sat_g / satp_g / best fixed) ==")
for v in sorted(jobs, key=lambda v: (v["meta"]["b"], v["meta"]["world"], v["meta"]["seed"])):
    if v["meta"]["b"] != 8:
        continue
    print(f"  {v['meta']['world']:9s} s{v['meta']['seed']}  best {bestw(v):2d}  " +
          "  ".join(f"{a} {list(v['res'][a]['final'].values())} {rel(v, a, best(v)):+5.1f}" for a in ARMS if a in ("sat_g", "satpv_g", "pair_g", "pair_s")))
