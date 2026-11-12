"""Summarize round 30 (run30.py): src / recheck / tight on top of v11 + D65 + D67 ("G1"). usage: python analyze30.py

Baselines from round 29: r29 (RTT worlds: v11 novd, v11, G1, fixed W) and r29s (round-28 worlds: G1), same code path
(G1 reproduced bit for bit by ext24.BatchAccount17 with everything off).
Trimmed run lists (see run30.py): round-28 world runs a change cannot affect were not run; they are identical to G1
by construction and are counted as 0 in the "incl. identical" means (A: 45 runs, B: 45 runs; C ran only short runs)."""
import json, os
from collections import defaultdict


def load(prefix):
    out = []
    for f in (f"results/{prefix}.0.jsonl", f"results/{prefix}.1.jsonl"):
        if os.path.exists(f):
            out += [json.loads(l) for l in open(f)]
    return out


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def lab(r):
    return "long" if r["req_mult"] == 10 else f"short s{r['seed']}"


key = lambda r: (r["world"], r["kind"], r["b"], r["seed"], r["req_mult"])
R29 = {key(r): r["res"] for r in load("r29") if r["part"] == "R"}
S29 = {key(r): r["res"] for r in load("r29s")}
A, B, C = load("r30a"), load("r30b"), load("r30c")
order = {"base": 0, "het": 1, "phased": 2, "bvar": 3, "high": 4, "chg x3": 5, "chg x0.3": 6, "chg8 x0.3": 7, "chg8 x3": 8, "chg8 x10": 9}
grp = lambda r: "static" if r["cmult"] is None else ("chg8" if r["world"].startswith("chg8") else "chg (r26)")


def fixed_ref(res):
    fx = {W: res[f"fixed {W}"]["rate"] for W in (2, 4, 8)}
    return max(fx.values()), max(fx, key=fx.get), fx


def paired(rows, pol, title, n_all=None):
    g = defaultdict(list)
    print(title)
    for r in sorted(rows, key=lambda r: (order[r["world"]], r["kind"], r["b"], r["req_mult"], r["seed"])):
        base = S29[key(r)]["G1"]
        x = r["res"][pol]
        d = 100 * (x["rate"] / base["rate"] - 1)
        for k in ("all", grp(r), lab(r).split()[0], r["world"]):
            g[k].append(d)
        extra = ""
        if pol == "src":
            extra = f"ev {base['cost_events']}->{x['cost_events']} (l{x['levents']}/v{x['vevents']}) checks {x['checks']} offsets {x['offsets']} probes {base['probes']}->{x['probes']}"
        if pol == "recheck":
            extra = f"rechecks {x['rc']} back {x['back']} {x['rc_log'][:3]}"
        print(f"{lab(r):8s} {r['world']:9s} {r['kind']:4s} b={r['b']} | {pol} vs G1 {d:+6.2f}% | {extra} | sw G1 {base['switch_log'][:2]} {pol} {x['switch_log'][:3]}")
    print("  means (min, max):")
    for k in ("all", "static", "chg (r26)", "chg8", "short", "long", "het", "phased", "high", "bvar", "chg8 x3", "chg8 x10"):
        if g[k]:
            print(f"   {k:9s}: {mean(g[k]):+.2f}% ({min(g[k]):+.2f}, {max(g[k]):+.2f}) n={len(g[k])}")
    if n_all and g["all"]:
        print(f"   incl. identical runs (0): {sum(g['all']) / n_all:+.3f}% over {n_all} runs ({n_all - len(g['all'])} not run)")
    print()


# ---------------- A ----------------
print("== Round 30 A (src): verify cost per token x5 at spec round 400 (vchg). src vs G1 (paired %)")
g = defaultdict(list)
for r in sorted([x for x in A if x["part"] == "V"], key=lambda r: (r["kind"], r["req_mult"], r["seed"])):
    b0, x = r["res"]["G1"], r["res"]["src"]
    d = 100 * (x["rate"] / b0["rate"] - 1)
    g[("V", r["kind"])].append(d); g[("V", "all")].append(d)
    print(f"{lab(r):8s} vchg {r['kind']:4s} b={r['b']} | src vs G1 {d:+6.2f}% | G1 ev {b0['cost_events']} K {b0['k']} sw {b0['switch_log'][:2]}")
    print(f"     src ev {x['cost_events']} (l{x['levents']}/v{x['vevents']}) checks {x['checks']} offsets {x['offsets']} "
          f"chk {x['chk_log'][:3]} probes {b0['probes']}->{x['probes']} K {x['k']} sw {x['switch_log'][:2]}")
for kk in ("all", "mid", "high"):
    if g[("V", kk)]:
        print(f"  mean {kk}: {mean(g[('V', kk)]):+.2f}% ({min(g[('V', kk)]):+.2f}, {max(g[('V', kk)]):+.2f})")
print()
print("== Round 30 A (src): RTT worlds of round 29. % vs best fixed W; v11 novd / v11 / G1 from r29")
POL = ("v11 novd", "v11", "G1", "src")
for r in sorted([x for x in A if x["part"] == "R"], key=lambda r: (r["world"], r["kind"], r["req_mult"], r["seed"])):
    res = dict(R29[key(r)]); res.update(r["res"])
    ref, bw, fx = fixed_ref(res)
    x = res["src"]
    parts = []
    for v in POL:
        d = 100 * (res[v]["rate"] / ref - 1)
        for k in ("all", r["world"], f"{r['world']} {r['kind']}"):
            g[("R", k, v)].append(d)
        parts.append(f"{v} {d:+.2f}%")
    print(f"{lab(r):8s} {r['world']:8s} {r['kind']:4s} b={r['b']} | " + " | ".join(parts) +
          f" | src: ev {x['cost_events']} checks {x['checks']} offsets {x['offsets']} probes {x['probes']} (G1 {res['G1']['probes']})")
print("  means (min):")
for k in ("all", "rtt jump", "rtt alt", "rtt jump mid", "rtt jump high", "rtt alt mid", "rtt alt high"):
    if g[("R", k, "src")]:
        print(f"   {k:13s}: " + "; ".join(f"{v} {mean(g[('R', k, v)]):+.2f}% ({min(g[('R', k, v)]):+.1f})" for v in POL))
print()
paired([x for x in A if x["part"] == "S"], "src", "== Round 30 A (src): round-28 worlds with a drift trigger under G1, src vs G1 (paired %)", 45)

# ---------------- B ----------------
paired([x for x in B if x["part"] == "S"], "recheck", "== Round 30 B (recheck): round-28 worlds with a narrowing switch under G1, recheck vs G1 (paired %)", 45)

# ---------------- C ----------------
print("== Round 30 C (tight): verify step +12+1.5b ms for K >= 16 (step). tight vs G1 (paired %)")
for r in sorted([x for x in C if x["part"] == "P"], key=lambda r: (r["b"], r["req_mult"], r["seed"])):
    b0, x = r["res"]["G1"], r["res"]["tight"]
    d = 100 * (x["rate"] / b0["rate"] - 1)
    g[("P", r["b"])].append(d); g[("P", "all")].append(d)
    print(f"{lab(r):8s} step b={r['b']} | tight vs G1 {d:+6.2f}% | K G1 {b0['k']} tight {x['k']} | W G1 {b0['w']} tight {x['w']}")
for kk in ("all", 1, 4):
    if g[("P", kk)]:
        print(f"  mean {kk}: {mean(g[('P', kk)]):+.2f}% ({min(g[('P', kk)]):+.2f}, {max(g[('P', kk)]):+.2f})")
print()
paired([x for x in C if x["part"] == "S"], "tight", "== Round 30 C (tight): round-28 worlds, tight vs G1 (paired %)")

# ---------------- 30b: K check with 4 rounds per K (src4) ----------------
E = load("r30e")
if E:
    A_ = {key(r): r["res"] for r in A}
    print("== Round 30b (src4: 4 rounds per K, tolerance from the check's pooled sd). paired % vs src (r30a) and vs G1")
    gg = defaultdict(list)
    for r in sorted(E, key=lambda r: (r["part"], r["world"], r["kind"], r["req_mult"], r["seed"])):
        x, s_ = r["res"]["src4"], A_[key(r)]["src"]
        g1 = A_[key(r)]["G1"] if r["part"] == "V" else (R29[key(r)]["G1"] if r["part"] == "R" else S29[key(r)]["G1"])
        d1, d2 = 100 * (x["rate"] / s_["rate"] - 1), 100 * (x["rate"] / g1["rate"] - 1)
        grp_ = r["part"] if r["part"] != "R" else r["world"]
        for k in ("all", grp_):
            gg[(k, "src")].append(d1); gg[(k, "G1")].append(d2)
        nc = lambda y: (sum(1 for c in y["chk_log"] if c[4]), sum(1 for c in y["chk_log"] if not c[4]))
        print(f"{lab(r):8s} {r['part']} {r['world']:9s} {r['kind']:4s} b={r['b']} | vs src {d1:+6.2f}% vs G1 {d2:+6.2f}% | "
              f"common/not src {nc(s_)} src4 {nc(x)} | ev {s_['cost_events']}->{x['cost_events']} probes {s_['probes']}->{x['probes']}")
    print("  means (min, max):")
    for k in ("all", "rtt jump", "rtt alt", "V", "S"):
        if gg[(k, "src")]:
            print(f"   {k:9s}: vs src {mean(gg[(k, 'src')]):+.2f}% ({min(gg[(k, 'src')]):+.2f}, {max(gg[(k, 'src')]):+.2f}); "
                  f"vs G1 {mean(gg[(k, 'G1')]):+.2f}% ({min(gg[(k, 'G1')]):+.2f}, {max(gg[(k, 'G1')]):+.2f}) n={len(gg[(k, 'src')])}")
    rr = [r for r in E if r["part"] == "R"]
    if rr:
        for v in ("src", "src4"):
            ds = []
            for r in rr:
                res = dict(R29[key(r)]); res.update(A_[key(r)]); res["src4"] = r["res"]["src4"]
                ref = max(res[f"fixed {W}"]["rate"] for W in (2, 4, 8))
                ds.append(100 * (res[v]["rate"] / ref - 1))
            print(f"   RTT worlds vs best fixed W: {v} {mean(ds):+.2f}% (min {min(ds):+.2f}) n={len(ds)}")

# ---------------- 30c: tight on long runs ----------------
Fr = load("r30f")
if Fr:
    paired(Fr, "tight", "== Round 30c (tight, long runs): round-28 worlds, tight vs G1 (paired %)")
    allc = [x for x in C if x["part"] == "S"] + Fr
    paired(allc, "tight", "== Round 30 C + 30c (tight, short and long): round-28 worlds, tight vs G1 (paired %)")
