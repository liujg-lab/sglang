"""Summarize round 26 (run26.py): the replay controller with the online cost model. usage: python analyze26.py

References for the nine static worlds (fixed W, flat known, replay14 = r14 known model5) come from r25 (same code
path, same seeds); the cost-change worlds run their own references."""
import json, os
from collections import defaultdict

rows, r25 = [], {}
for f in ("results/r26.0.jsonl", "results/r26.1.jsonl"):
    if os.path.exists(f):
        rows += [json.loads(l) for l in open(f)]
for f in ("results/r25.0.jsonl", "results/r25.1.jsonl"):
    if os.path.exists(f):
        for l in open(f):
            r = json.loads(l)
            r25[(r["world"], r["kind"], r["b"], r["seed"], r["req_mult"])] = r["res"]
POL = ("flat known", "flat online", "r14 known model5", "r14 known resid", "r14 online resid")
order = {"base": 0, "het": 1, "phased": 2, "bvar": 3, "high": 4, "chg x3": 5, "chg x0.3": 6}


def mean(x):
    return sum(x) / len(x) if x else float("nan")


print("== Round 26: v11 replay controller with the online cost model (flat start).")
print("   % vs the best fixed W (known costs); [W use %]; switches (first); online: probe rounds, cost events, layer-time error")
g = defaultdict(list)
for r in sorted(rows, key=lambda r: (r["req_mult"], r["seed"], order[r["world"]], r["kind"], r["b"])):
    res = dict(r["res"])
    if r["cmult"] is None:
        ref25 = r25[(r["world"], r["kind"], r["b"], r["seed"], r["req_mult"])]
        for W in (2, 4, 8):
            res[f"fixed {W}"] = ref25[f"fixed {W}"]
        res["flat known"] = ref25["flat"]
        res["r14 known model5"] = ref25["replay14"]
    fx = {W: res[f"fixed {W}"]["rate"] for W in (2, 4, 8)}
    bw = max(fx, key=fx.get)
    ref = fx[bw]
    run = "long" if r["req_mult"] == 10 else f"short s{r['seed']}"
    grp = "change" if r["cmult"] is not None else "static"
    parts = [f"{run:8s} {r['world']:8s} {r['kind']:4s} b={r['b']} | best fixed {bw}"]
    for v in POL:
        x = res[v]
        d = 100 * (x["rate"] / ref - 1)
        for key in ("all", grp, run.split()[0], r["world"]):
            g[(key, v)].append(d)
        tot = sum(x["w"].values()) or 1
        use = ",".join(f"{k}:{100 * n / tot:.0f}" for k, n in x["w"].items())
        sl = x.get("switch_log") or []
        first = f" {tuple(sl[0])}" if sl else ""
        extra = ""
        if "online" in v:
            extra = f" pr {x.get('probes', 0)} ev {x.get('cost_events', 0)} lerr {100 * x.get('lt_err', 0):.1f}%"
        parts.append(f"{v} {d:+.2f}% [{use}] sw {x.get('switches', 0)}{first}{extra}")
    print(" | ".join(parts))
print()
print("  means (min) vs the best fixed W:")
for key in ("all", "static", "change", "short", "long", "base", "het", "phased", "bvar", "high", "chg x3", "chg x0.3"):
    if not g[("all", POL[0])]:
        break
    print(f"   {key:8s}: " + "; ".join(f"{v} {mean(g[(key, v)]):+.2f}% ({min(g[(key, v)]):+.1f})" for v in POL if g[(key, v)]))
