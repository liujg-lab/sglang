"""Summarize round 28 (run28.py): reset at cost events and continuous timing probes. usage: python analyze28.py

% vs the best fixed W (static worlds; references from r25 / r26) or vs the best of fixed W and the scheduled switch at
the change (chg8 worlds of r27); the unfixed controller ("r14 online resid") and the known-cost one come from r26 / r27."""
import json, os
from collections import defaultdict


def load(prefix):
    out = []
    for f in (f"results/{prefix}.0.jsonl", f"results/{prefix}.1.jsonl"):
        if os.path.exists(f):
            out += [json.loads(l) for l in open(f)]
    return out


k = lambda r: (r["world"], r["kind"], r["b"], r["seed"], r["req_mult"])
r28 = load("r28")
I25 = {k(r): r["res"] for r in load("r25")}
I26 = {k(r): r["res"] for r in load("r26")}
I27 = {k(r): r["res"] for r in load("r27") if r["part"] == "B"}
POL = ("flat online", "r14 known resid", "r14 online resid", "reset", "cprobe", "reset+cprobe")


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def lab(r):
    return "long" if r["req_mult"] == 10 else f"short s{r['seed']}"


g = defaultdict(list)
print("== Round 28: fixes for the v11 controller with online costs. % vs the best fixed W (chg8: vs the best of fixed and")
print("   the scheduled switch at the change); switches (first two); fixes: cost events / resets / continuous probe rounds")
order = {"base": 0, "het": 1, "phased": 2, "bvar": 3, "high": 4, "chg x3": 5, "chg x0.3": 6, "chg8 x0.3": 7, "chg8 x3": 8, "chg8 x10": 9}
for r in sorted(r28, key=lambda r: (order[r["world"]], r["kind"], r["b"], r["req_mult"], r["seed"])):
    key = k(r)
    if key in I27:
        res = dict(I27[key])
        ref = max([res[f"fixed {W}"]["rate"] for W in (2, 4, 8)] + [res["sched 4->2"]["rate"], res["sched 4->8"]["rate"]])
        grp = "chg8"
    else:
        res = dict(I26[key])
        src = I25[key] if r["cmult"] is None else res
        ref = max(src[f"fixed {W}"]["rate"] for W in (2, 4, 8))
        grp = "static" if r["cmult"] is None else "chg (r26)"
    res.update(r["res"])
    parts = [f"{lab(r):8s} {r['world']:9s} {r['kind']:4s} b={r['b']}"]
    for v in POL:
        x = res[v]
        d = 100 * (x["rate"] / ref - 1)
        for kk in ("all", grp, lab(r).split()[0], r["world"]):
            g[(kk, v)].append(d)
        sl = [tuple(s) for s in (x.get("switch_log") or [])[:2]]
        extra = f" ev{x.get('cost_events', 0)}/rs{x.get('resets', 0)}/cp{x.get('cprobes', 0)}" if v in ("reset", "cprobe", "reset+cprobe") else ""
        parts.append(f"{v} {d:+.2f}% sw{x.get('switches', 0)} {sl if sl else ''}{extra}")
    print(" | ".join(parts))
print()
print("  means (min):")
for kk in ("all", "static", "chg (r26)", "chg8", "short", "long", "base", "het", "phased", "bvar", "high", "chg8 x0.3", "chg8 x3", "chg8 x10"):
    if g[(kk, POL[0])]:
        print(f"   {kk:9s}: " + "; ".join(f"{v} {mean(g[(kk, v)]):+.2f}% ({min(g[(kk, v)]):+.1f})" for v in POL))
