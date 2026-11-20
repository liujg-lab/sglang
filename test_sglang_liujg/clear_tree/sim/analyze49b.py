"""Round 49 with level-fit controls: every arm on the same layer model (per-(W, g) fit x shared level).
usage: python analyze49b.py"""
import glob, json, math
import os
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
from collections import defaultdict, Counter

rows = {}
for pre in ("r43m", "r45a", "r49", "r49c", "r49d"):
    for f in sorted(glob.glob(os.path.join(RES, pre + ".*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            k = (r["part"], r["world"], r["b"], r["seed"], str(r["x"]))
            rows.setdefault(k, dict(meta=r, res={}))["res"].update(r["res"])
FIXL = ("fixL2", "fixL4", "fixL8", "fixL16")
FIX = ("fix2", "fix4", "fix8", "fix16")
ARMS = ("satL_g", "satpvL_g", "pair_g", "pair_s")
OLD = ("sat_g", "satpv_g")
jobs = [v for v in rows.values() if all(a in v["res"] for a in FIXL + ARMS + FIX + OLD)]
print(f"{len(jobs)} jobs")


def stat(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, sd / math.sqrt(n), min(xs), n


def rate(v, a):
    return v["res"][a]["rate"]


def bestL(v):
    return max(rate(v, f) for f in FIXL)


def rel(v, a, ref):
    return 100.0 * (rate(v, a) / ref - 1.0)


def kind(v):
    w = v["meta"]["world"]
    return "change" if w.startswith(("chg", "rtt", "lalt", "vchg", "k32")) else "static"


def grp(v):
    r = v["meta"]
    if r["world"] == "bvar":
        return "b varies"
    return f"b={r['b']}" + (" mixed" if r["mix"] == "mixed" else "")


def line(lab, xs):
    m, se, lo, n = stat(xs)
    print(f"  {lab:12s} {m:+6.2f}%  se {se:4.2f}  min {lo:+6.2f}  n {n}")


print("\n== effect of the shared level alone: best fixL vs best fix (old fit) ==")
line("best fixL", [100 * (bestL(v) / max(rate(v, f) for f in FIX) - 1) for v in jobs])
for w in (2, 4, 8, 16):
    line(f"fixL{w}/fix{w}", [100 * (rate(v, f"fixL{w}") / rate(v, f"fix{w}") - 1) for v in jobs])

print("\n== vs best fixed W with the same level fit (fixL) ==")
for a in ARMS:
    line(a, [rel(v, a, bestL(v)) for v in jobs])
print("\n== paired vs satpvL_g ==")
for a in ARMS:
    if a != "satpvL_g":
        xs = [rel(v, a, rate(v, "satpvL_g")) for v in jobs]
        m, se, lo, n = stat(xs)
        print(f"  {a:12s} {m:+6.2f}%  se {se:4.2f}  better {sum(x > 0 for x in xs)}/{n}")
print("\n== static vs changing worlds, vs best fixL ==")
for kd in ("static", "change"):
    s = [v for v in jobs if kind(v) == kd]
    print(f" {kd} ({len(s)})  " + "  ".join(f"{a} {stat([rel(v, a, bestL(v)) for v in s])[0]:+.2f}" for a in ARMS))
print("\n== by group, vs best fixL ==")
by = defaultdict(list)
for v in jobs:
    by[grp(v)].append(v)
for g in sorted(by):
    print(f" {g} ({len(by[g])})  " + "  ".join(f"{a} {stat([rel(v, a, bestL(v)) for v in by[g]])[0]:+.2f}" for a in ARMS))


def switches(v, a):
    out = []
    for x in v["res"][a]["switch"]:
        x = list(x)
        if x[3] is None:
            x[3] = 8 if x[2] == 1 else 4
        if isinstance(x[3], int) and isinstance(x[4], int):
            out.append(x)
    return out


print("\n== switches per job, reversals, final W vs best fixL ==")
for a in ARMS:
    sw = rev = pairs = 0
    fin = Counter()
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
            fw = f[0] if f else (8 if v["meta"]["b"] == 1 else 4)
            bw = int(max(FIXL, key=lambda f: rate(v, f))[4:])
            fin["over" if fw > bw else ("under" if fw < bw else "eq")] += 1
    print(f"  {a:12s} sw/job {sw / len(jobs):4.2f}  reversals {rev}/{pairs}  final {dict(fin)}")

print("\n== per job vs best fixL ==")
for v in sorted(jobs, key=lambda v: (v["meta"]["b"], v["meta"]["world"], v["meta"]["seed"])):
    bw = max(FIXL, key=lambda f: rate(v, f))
    print(f"  {v['meta']['world']:9s} b={v['meta']['b']} s{v['meta']['seed']} best {bw:6s} " +
          "  ".join(f"{a} {rel(v, a, bestL(v)):+5.1f}" for a in ARMS))
