"""Summarize round 22 (run22.py): batch-wide verify budget. usage: python analyze22.py"""
import json, os
from collections import defaultdict

rows = []
for f in ("results/r22.0.jsonl", "results/r22.1.jsonl"):
    if os.path.exists(f):
        rows += [json.loads(l) for l in open(f)]


def mean(x):
    return sum(x) / len(x) if x else float("nan")


print("== Round 22: batch-wide verify budget (per-request K_i, ragged verify, token buckets) vs one K per batch.")
print("   % vs uniform; seed 8; known costs. K share: uniform per round, unified per request (<=8 / <=16 / <=32)")
g = defaultdict(lambda: defaultdict(list))
for mix in ("short", "mixed"):
    for vm in (1.0, 10.0):
        for kind in ("mid", "het"):
            for b in (4, 8):
                rs = [r for r in rows if (r["mix"], r["vm"], r["kind"], r["b"]) == (mix, vm, kind, b)]
                if not rs:
                    continue
                r = rs[0]["res"]
                ref = r["uniform"]["rate"]
                parts = [f"{mix:5s} verify x{vm:<4g} {kind:3s} b={b} | uniform {ref:6.1f} K{r['uniform']['k']}"]
                for a in ("unified_verify", "unified_both"):
                    d = 100 * (r[a]["rate"] / ref - 1)
                    for key in ("all", mix, f"x{vm:g}", kind, f"b={b}"):
                        g[key][a].append(d)
                    parts.append(f"{a} {d:+.2f}% K{r[a]['k']}")
                print(" | ".join(parts))
print()
print("  means (min, max) vs uniform:")
for key, d in g.items():
    print(f"   {key:7s}: " + "; ".join(f"{a} {mean(v):+.2f}% ({min(v):+.1f}, {max(v):+.1f})" for a, v in d.items()))
