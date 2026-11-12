"""Summarize round 20 (run20.py): flat start, costs learned online vs known. usage: python analyze20.py"""
import json, os
from collections import defaultdict

R = "results"
rows = []
for f in ("r20.0.jsonl", "r20.1.jsonl"):
    p = os.path.join(R, f)
    if os.path.exists(p):
        rows += [json.loads(l) for l in open(p)]
V = ("flat known", "flat online", "calib known", "calib online", "best fixed")
WORLDS = ("base", "tok x0.3", "change")


def mean(x):
    return sum(x) / len(x) if x else float("nan")


print("== Round 20: flat starting table; costs known vs learned online (timing probes). Seed 8. % vs the best fixed W")
print("   (rerun here); [probe / calibration share of spec rounds; cost events; layer-cost error; verify K-diff error]")
g = defaultdict(lambda: defaultdict(list))
gap = defaultdict(lambda: defaultdict(list))
for wd in WORLDS:
    for kind in ("mid", "high"):
        for b in (1, 4, 8):
            rs = [r for r in rows if (r["world"], r["kind"], r["b"]) == (wd, kind, b)]
            if not rs:
                continue
            r = rs[0]["res"]
            ref = r["best fixed"]["rate"]
            parts = [f"{wd:8s} {kind:4s} b={b} | best fixed W={r['best fixed']['W']} {ref:6.1f}"]
            for v in V[:4]:
                d = 100 * (r[v]["rate"] / ref - 1)
                g[wd][v].append(d)
                x = r[v]
                extra = ""
                if "probe_rounds" in x:
                    sr = max(1, x["spec_rounds"])
                    le = f"{100 * x['lt_err']:.1f}%" if x.get("lt_err") is not None else "-"
                    ve = f"{100 * x['vt_err']:.0f}%" if x.get("vt_err") is not None else "-"
                    extra = (f" [probe {100 * x['probe_rounds'] / sr:.1f}%, cal {100 * x['cal_rounds'] / sr:.1f}%, ev {x['cost_events']}, "
                             f"lt {le}, vK {ve}]")
                parts.append(f"{v} {d:+.2f}%{extra}")
            gap[wd]["flat: online - known"].append(100 * (r["flat online"]["rate"] / r["flat known"]["rate"] - 1))
            gap[wd]["calib: online - known"].append(100 * (r["calib online"]["rate"] / r["calib known"]["rate"] - 1))
            gap[wd]["calib online - flat known"].append(100 * (r["calib online"]["rate"] / r["flat known"]["rate"] - 1))
            print(" | ".join(parts))
print()
print("  means (min) vs the best fixed W:")
for wd in WORLDS:
    if g[wd]:
        print(f"   {wd:8s}: " + "; ".join(f"{v} {mean(x):+.2f}% ({min(x):+.1f})" for v, x in g[wd].items()))
allv = defaultdict(list)
for wd in WORLDS:
    for v, x in g[wd].items():
        allv[v] += x
print("   all     : " + "; ".join(f"{v} {mean(x):+.2f}%" for v, x in allv.items()))
print()
print("  paired gaps (same scenario):")
for wd in WORLDS:
    if gap[wd]:
        print(f"   {wd:8s}: " + "; ".join(f"{k} {mean(x):+.2f}% (min {min(x):+.1f}, max {max(x):+.1f})" for k, x in gap[wd].items()))
