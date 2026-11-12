"""Summarize round 29 (run29.py): guardrails G1 / G3 / G4 for the v11 controller. usage: python analyze29.py

W (switch worlds): % vs the best of Target-only and fixed 2 / 4 / 8 (known costs, same D42 switch).
R (RTT worlds): % vs the best fixed W (known costs).
S (round 28 worlds): "G1" vs "v11", paired (same world, seed, run length), from results/r29s.* (run29.py <proc> S);
the two S jobs of the first launch (v11 vs all, results/r29.*) are listed separately. No fixed references for S."""
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


r29 = load("r29")
r29s = load("r29s")
g = defaultdict(list)

print("== Round 29 W: speculation switch (D42), Target-only scaled so fixed W=4 wins ~12%. % vs best of Target-only and")
print("   fixed W + switch; [spec share]; switch log (ms, decision); timing-probe rounds")
for r in sorted([x for x in r29 if x["part"] == "W"], key=lambda r: (r["b"], r["req_mult"], r["seed"])):
    res = r["res"]
    ar = res["v11"]["ar"]
    refs = {k: res[k]["rate"] for k in ("fixed 2", "fixed 4", "fixed 8")}
    ref = max(list(refs.values()) + [ar])
    print(f"{lab(r):8s} b={r['b']} | Target-only {100 * (ar / ref - 1):+.1f}% | " +
          " ".join(f"{k} {100 * (v / ref - 1):+.1f}%[{res[k]['spec_share']:.2f}]" for k, v in refs.items()))
    for v in ("v11", "G3", "G4", "all"):
        x = res[v]
        d = 100 * (x["rate"] / ref - 1)
        g[("W", "all", v)].append(d); g[("W", f"b={r['b']}", v)].append(d); g[("W", lab(r).split()[0], v)].append(d)
        print(f"     {v:4s} {d:+6.2f}% [{x['spec_share']:.2f}] probes {x['probes']} gaps {x['gaps']} sw {x['n_sw']} "
              f"{x['sw_log'][:5]} W-switch {x['switch_log'][:2]}")
print("  means (min):")
for k in ("all", "b=4", "b=8", "short", "long"):
    if g[("W", k, "v11")]:
        print(f"   {k:5s}: " + "; ".join(f"{v} {mean(g[('W', k, v)]):+.2f}% ({min(g[('W', k, v)]):+.1f})" for v in ("v11", "G3", "G4", "all")))

print()
print("== Round 29 R: RTT jump 4->24 ms at spec round 400 / RTT alternating 4<->40 ms every 100 rounds from 400.")
print("   % vs best fixed W (known costs); events / suppressed / timing-probe rounds; W switches")
POLR = ("v11 novd", "v11", "G1", "G3", "all")
for r in sorted([x for x in r29 if x["part"] == "R"], key=lambda r: (r["world"], r["kind"], r["req_mult"], r["seed"])):
    res = r["res"]
    fx = {W: res[f"fixed {W}"]["rate"] for W in (2, 4, 8)}
    ref = max(fx.values())
    print(f"{lab(r):8s} {r['world']:8s} {r['kind']:4s} b={r['b']} best {max(fx, key=fx.get)} | " +
          " ".join(f"fixed {W} {100 * (v / ref - 1):+.1f}%" for W, v in fx.items()))
    for v in POLR:
        x = res[v]
        d = 100 * (x["rate"] / ref - 1)
        for k in ("all", r["world"], f"{r['world']} {r['kind']}", lab(r).split()[0]):
            g[("R", k, v)].append(d)
        print(f"     {v:8s} {d:+6.2f}% ev {x['cost_events']} supp {x['suppressed']} probes {x['probes']} "
              f"sw {x['switches']} {x['switch_log'][:3]} final {x['final']}")
print("  means (min):")
for k in ("all", "rtt jump", "rtt alt", "rtt jump mid", "rtt jump high", "rtt alt mid", "rtt alt high", "short", "long"):
    if g[("R", k, "v11")]:
        print(f"   {k:13s}: " + "; ".join(f"{v} {mean(g[('R', k, v)]):+.2f}% ({min(g[('R', k, v)]):+.1f})" for v in POLR))

print()
for r in [x for x in r29 if x["part"] == "S"]:
    a, v = r["res"]["all"], r["res"]["v11"]
    print(f"(first launch, all vs v11) {lab(r)} {r['world']} {r['kind']} b={r['b']}: {100 * (a['rate'] / v['rate'] - 1):+.2f}%")
print("== Round 29 S: round 28 worlds, 'G1' vs 'v11' (paired %); events v11/G1; suppressed (G1); probes v11/G1; first W switch")
order = {"base": 0, "het": 1, "phased": 2, "bvar": 3, "high": 4, "chg x3": 5, "chg x0.3": 6, "chg8 x0.3": 7, "chg8 x3": 8, "chg8 x10": 9}
for r in sorted(r29s, key=lambda r: (order[r["world"]], r["kind"], r["b"], r["req_mult"], r["seed"])):
    a, v = r["res"]["G1"], r["res"]["v11"]
    d = 100 * (a["rate"] / v["rate"] - 1)
    grp = "static" if r["cmult"] is None else ("chg8" if r["world"].startswith("chg8") else "chg (r26)")
    for k in ("all", grp, lab(r).split()[0], r["world"]):
        g[("S", k)].append(d)
    print(f"{lab(r):8s} {r['world']:9s} {r['kind']:4s} b={r['b']} | G1 vs v11 {d:+6.2f}% | ev {v['cost_events']}/{a['cost_events']} supp {a['suppressed']} "
          f"probes {v['probes']}/{a['probes']} | sw v11 {v['switch_log'][:2]} G1 {a['switch_log'][:2]}")
print("  means (min, max) of G1 vs v11:")
for k in ("all", "static", "chg (r26)", "chg8", "short", "long"):
    if g[("S", k)]:
        x = g[("S", k)]
        print(f"   {k:9s}: {mean(x):+.2f}% ({min(x):+.2f}, {max(x):+.2f}) n={len(x)}")
ev = [(r["world"], r["kind"], r["b"], lab(r), r["res"]["v11"]["ev_log"]) for r in r29s if r["cmult"] is None
      and any(e[2] == ["verify"] for e in r["res"]["v11"]["ev_log"])]
print(f"  static worlds with a verify-detector event under v11: {len(ev)}", ev[:6])
