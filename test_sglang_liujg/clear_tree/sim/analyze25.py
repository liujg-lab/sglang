"""Summarize round 25 (run25.py): fast-converging online W vs the replay controller. usage: python analyze25.py"""
import json, os
from collections import defaultdict

rows = []
for f in ("results/r25.0.jsonl", "results/r25.1.jsonl"):
    if os.path.exists(f):
        rows += [json.loads(l) for l in open(f)]
ONLINE = ("flat", "replay14", "fast")
order = {"base": 0, "het": 1, "phased": 2, "bvar": 3, "high": 4}


def mean(x):
    return sum(x) / len(x) if x else float("nan")


print("== Round 25: fast-converging online W vs the replay controller (flat start, known prefix-aware costs).")
print("   % vs the best fixed W of the scenario; [W use %]; switches (first: round b from->to); fast: probes, extra Draft ms")
g = defaultdict(list)
for r in sorted(rows, key=lambda r: (r["req_mult"], r["seed"], order[r["world"]], r["b"])):
    res = r["res"]
    fx = {W: res[f"fixed {W}"]["rate"] for W in (2, 4, 8)}
    bw = max(fx, key=fx.get)
    ref = fx[bw]
    run = "long" if r["req_mult"] == 10 else f"short s{r['seed']}"
    parts = [f"{run:8s} {r['world']:6s} b={r['b']} | best fixed {bw} ({' '.join(f'{W}:{100 * (fx[W] / ref - 1):+.1f}' for W in (2, 4, 8))})"]
    for v in ONLINE:
        x = res[v]
        d = 100 * (x["rate"] / ref - 1)
        g[(run.split()[0], v)].append(d)
        g[(r["world"], v)].append(d)
        g[("all", v)].append(d)
        tot = sum(x["w"].values()) or 1
        use = ",".join(f"{k}:{100 * n / tot:.0f}" for k, n in x["w"].items())
        sl = x.get("switch_log") or []
        first = f" first {tuple(sl[0])}" if sl else ""
        extra = f" probes {x['probes']} +{x['extra_ms'] / 1000:.1f}s" if v == "fast" else ""
        parts.append(f"{v} {d:+.2f}% [{use}] sw {x['switches']}{first}{extra}")
    print(" | ".join(parts))
print()
print("  means (min) vs the best fixed W:")
for key in ("all", "short", "long", "base", "het", "phased", "bvar", "high"):
    print(f"   {key:6s}: " + "; ".join(f"{v} {mean(g[(key, v)]):+.2f}% ({min(g[(key, v)]):+.1f})" for v in ONLINE if g[(key, v)]))
