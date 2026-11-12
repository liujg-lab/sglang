"""Summarize round 27 (run27.py). usage: python analyze27.py

Part A: delayed first decision vs the v11 controller with online costs (r26 "r14 online resid"), % vs the best fixed W
(references as in analyze26: r25 for the static worlds, r26 for its change worlds).
Part B: cost changes that move the best W; % vs the best fixed W and vs the best scheduled switch at the change."""
import json, os
from collections import defaultdict

def load(prefix):
    out = []
    for f in (f"results/{prefix}.0.jsonl", f"results/{prefix}.1.jsonl"):
        if os.path.exists(f):
            out += [json.loads(l) for l in open(f)]
    return out

r27, r26, r25 = load("r27"), load("r26"), load("r25")
k = lambda r: (r["world"], r["kind"], r["b"], r["seed"], r["req_mult"])
I26 = {k(r): r for r in r26}
I25 = {k(r): r["res"] for r in r25}
DEL = ("d100 reset", "d300 reset", "d300 keep")


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def run_lab(r):
    return "long" if r["req_mult"] == 10 else f"short s{r['seed']}"


print("== Round 27 A: delaying the first decision (online costs). % vs the best fixed W; switches (first)")
g = defaultdict(list)
for r in sorted([x for x in r27 if x["part"] == "A"], key=lambda r: (r["req_mult"], r["seed"], r["world"], r["kind"], r["b"])):
    base = I26[k(r)]
    res = dict(base["res"])
    if base["cmult"] is None:
        for W in (2, 4, 8):
            res[f"fixed {W}"] = I25[k(r)][f"fixed {W}"]
    res.update(r["res"])
    fx = {W: res[f"fixed {W}"]["rate"] for W in (2, 4, 8)}
    ref = max(fx.values())
    parts = [f"{run_lab(r):8s} {r['world']:8s} {r['kind']:4s} b={r['b']} best {max(fx, key=fx.get)}"]
    for v in ("r14 known resid", "r14 online resid") + DEL:
        x = res[v]
        d = 100 * (x["rate"] / ref - 1)
        for key in ("all", run_lab(r).split()[0], "change" if r["cmult"] else "static"):
            g[(key, v)].append(d)
        sl = x.get("switch_log") or []
        parts.append(f"{v} {d:+.2f}% sw {x.get('switches', 0)}" + (f" {tuple(sl[0])}" if sl else ""))
    print(" | ".join(parts))
print("  means (min):")
for key in ("all", "static", "change", "short", "long"):
    print(f"   {key:7s}: " + "; ".join(f"{v} {mean(g[(key, v)]):+.2f}% ({min(g[(key, v)]):+.1f})"
                                    for v in ("r14 known resid", "r14 online resid") + DEL if g[(key, v)]))

print()
print("== Round 27 B: Draft layer cost x0.3 / x3 / x10 at the 400th speculative round (mid, short prefixes, b = 8)")
print("   % vs the best fixed W | % vs the best of fixed and scheduled switch; [W use %]; switch log")
POL = ("flat known", "flat online", "r14 known resid", "r14 online resid") + DEL
h = defaultdict(list)
for r in sorted([x for x in r27 if x["part"] == "B"], key=lambda r: (r["world"], r["req_mult"], r["seed"])):
    res = r["res"]
    fx = {W: res[f"fixed {W}"]["rate"] for W in (2, 4, 8)}
    sc = {s: res[s]["rate"] for s in ("sched 4->2", "sched 4->8")}
    ref = max(fx.values())
    best_all = max(list(fx.values()) + list(sc.values()))
    bl = max({**{f"fixed {W}": v for W, v in fx.items()}, **sc}.items(), key=lambda kv: kv[1])[0]
    print(f"{run_lab(r):8s} {r['world']:9s} | fixed 2:{100 * (fx[2] / ref - 1):+.1f} 4:{100 * (fx[4] / ref - 1):+.1f} "
          f"8:{100 * (fx[8] / ref - 1):+.1f} | sched 4->2 {100 * (sc['sched 4->2'] / ref - 1):+.1f} 4->8 "
          f"{100 * (sc['sched 4->8'] / ref - 1):+.1f} (best overall: {bl})")
    for v in POL:
        x = res[v]
        d, d2 = 100 * (x["rate"] / ref - 1), 100 * (x["rate"] / best_all - 1)
        h[(r["world"], v)].append(d2)
        h[("all", v)].append(d2)
        tot = sum(x["w"].values()) or 1
        use = ",".join(f"{kk}:{100 * n / tot:.0f}" for kk, n in x["w"].items())
        print(f"     {v:16s} {d:+6.2f}% | {d2:+6.2f}% [{use}] sw {x.get('switches', 0)} {x.get('switch_log', [])[:4]}"
              + (f" ev {x.get('cost_events')}" if "online" in v or v in DEL else ""))
print("  means vs the best of fixed and scheduled switch (min):")
for key in ("all", "chg8 x0.3", "chg8 x3", "chg8 x10"):
    print(f"   {key:9s}: " + "; ".join(f"{v} {mean(h[(key, v)]):+.2f}% ({min(h[(key, v)]):+.1f})" for v in POL if h[(key, v)]))
