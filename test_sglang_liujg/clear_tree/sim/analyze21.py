"""Summarize round 21 (run21.py): prefix-aware cost model learned online. usage: python analyze21.py"""
import json, os
from collections import defaultdict

rows = []
for f in ("results/r21.0.jsonl", "results/r21.1.jsonl"):
    if os.path.exists(f):
        rows += [json.loads(l) for l in open(f)]
MODES = ("known", "blind", "aware", "aware_noorder")


def mean(x):
    x = [v for v in x if v is not None]
    return sum(x) / len(x) if x else float("nan")


print("== Round 21: prefix-aware costs. % vs 'known' (true prefix-aware costs). Seed 8, workload mid.")
print("   [probe rounds / cost events / layer error / verify K32-K8 difference error (late half)]")
g = defaultdict(lambda: defaultdict(list))
for (mix, st) in (("short", "strong"), ("mixed", "strong"), ("long", "strong"), ("mixed", "mid")):
    for b in (1, 4, 8):
        rs = [r for r in rows if (r["mix"], r["strength"], r["b"]) == (mix, st, b)]
        if not rs:
            continue
        r = rs[0]["res"]
        ref = r["known"]["rate"]
        parts = [f"{mix:5s} {st:6s} b={b} | known {ref:6.1f} K{r['known']['k']}"]
        for m in MODES[1:]:
            x = r[m]
            d = 100 * (x["rate"] / ref - 1)
            g[(mix, st)][m].append(d)
            g["all"][m].append(d)
            el = f"{100 * x['err_l']:.1f}%" if x.get("err_l") is not None else "-"
            ev = f"{100 * x['err_v_late']:.0f}%" if x.get("err_v_late") is not None else "-"
            parts.append(f"{m} {d:+.2f}% [{x['probe_rounds']}/{x['cost_events']}/{el}/{ev}] K{x['k']}")
        print(" | ".join(parts))
print()
print("  means (min) vs known:")
for key, d in g.items():
    print(f"   {str(key):22s}: " + "; ".join(f"{m} {mean(v):+.2f}% ({min(v):+.1f})" for m, v in d.items()))
