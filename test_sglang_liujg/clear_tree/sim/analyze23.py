"""Summarize round 23 (run23.py). usage: python analyze23.py"""
import json, os
from collections import defaultdict

rows = []
for f in ("results/r23.0.jsonl", "results/r23.1.jsonl"):
    if os.path.exists(f):
        rows += [json.loads(l) for l in open(f)]
ONLINE = ("flat", "derived", "derived2", "cellcal", "replay14")


def mean(x):
    return sum(x) / len(x) if x else float("nan")


print("== Round 23: W from the cost model vs other online W methods (known prefix-aware costs, flat start, seed 8).")
print("   % vs the best fixed W of the scenario; [W use %] switches / exploration rounds")
g = defaultdict(list)
order = {"base": 0, "het": 1, "phased": 2, "bvar": 3}
for r in sorted(rows, key=lambda r: (order[r["world"]], r["b"])):
    res = r["res"]
    fx = {W: res[f"fixed {W}"]["rate"] for W in (2, 4, 8)}
    bw = max(fx, key=fx.get)
    ref = fx[bw]
    parts = [f"{r['world']:6s} b={r['b']} | best fixed W={bw} {ref:6.1f} (2:{fx[2]:.1f} 4:{fx[4]:.1f} 8:{fx[8]:.1f})"]
    for v in ONLINE:
        x = res[v]
        d = 100 * (x["rate"] / ref - 1)
        g[v].append(d)
        g[(r["world"], v)].append(d)
        tot = sum(x["w"].values()) or 1
        use = ",".join(f"{k}:{100 * n / tot:.0f}" for k, n in x["w"].items())
        parts.append(f"{v} {d:+.2f}% [{use}] {x['switches']}/{x['explore']}")
    print(" | ".join(parts))
print()
print("  means over scenarios (min) vs the best fixed W:")
for v in ONLINE:
    print(f"   {v:9s}: {mean(g[v]):+.2f}% ({min(g[v]):+.1f})" + "  |  " +
          "; ".join(f"{w} {mean(g[(w, v)]):+.2f}%" for w in ("base", "het", "phased", "bvar") if g[(w, v)]))
