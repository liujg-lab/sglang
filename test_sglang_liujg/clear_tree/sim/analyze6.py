"""Summarise round 6 (results/r6p*.jsonl, stop-boundary records results/r6E.jsonl.*). Run from this directory."""
import json, glob, statistics as st
import numpy as np

rows = []
for f in sorted(glob.glob("results/r6p*.jsonl")):
    rows += [json.loads(l) for l in open(f)]
pct = lambda a, b: 100.0 * (a / b - 1)
R = lambda v: v[0] if isinstance(v, list) else v


def part(p):
    return sorted([r for r in rows if r["part"] == p], key=lambda r: (r["scen"], r["seed"]))


print("== A: cost model under context change (b=4; tok/s, vs known costs; K_v differs from true-cost choice; loss in tokens/round)")
agg = {}
for r in part("A"):
    known = R(r["res"]["known"])
    line = [f"{r['scen']:38s} s{r['seed']} Target-only={r['res']['Target-only']:.0f}"]
    for k, v in r["res"].items():
        if k == "Target-only":
            continue
        rate, kr = v
        dis = kr["kv_dis"] / max(1, kr["kv_n"]); loss = kr["kv_loss"] / max(1, kr["kv_n"])
        agg.setdefault(k, []).append(pct(rate, known))
        line.append(f"{k}: {rate:.0f} ({pct(rate, known):+.1f}%) K≠{dis:.0%} loss={loss:.3f}")
    print(" | ".join(line))
for k, v in agg.items():
    print(f"  {k:26s} vs known costs: mean {st.mean(v):+.2f}% (min {min(v):+.2f}, max {max(v):+.2f}), n={len(v)}")

print("\n== B: K_g cache scope (mixed load, RTT 4<->20 every 30 s; tok/s, vs v1, re-selections per round, K_g differs from true-cost choice)")
aggB = {}
for r in part("B"):
    v1 = R(r["res"]["every 10 rounds (v1)"])
    line = [f"s{r['seed']} Target-only={r['res']['Target-only']:.0f}"]
    for k, v in r["res"].items():
        if k == "Target-only":
            continue
        rate, kr = v
        aggB.setdefault(k, []).append(pct(rate, v1))
        line.append(f"{k}: {rate:.1f} ({pct(rate, v1):+.1f}%) reeval={kr['reevals'] / max(1, kr['rounds']):.2f} Kg≠{kr['kg_dis'] / max(1, kr['kg_n']):.0%}")
    print(" | ".join(line))
for k, v in aggB.items():
    print(f"  {k:46s} vs v1: mean {st.mean(v):+.2f}%")

print("\n== C: delta decay per committed position (tok/s, vs 0.9)")
for r in part("C"):
    base = r["res"]["decay 0.9"]
    print(f"{r['scen']:14s} s{r['seed']} " + " | ".join(f"{k}: {v:.1f} ({pct(v, base):+.1f}%)" for k, v in r["res"].items()))

print("\n== D: switch timing basis, 2 ms Target overhead (tok/s; spec share; vs max(Target-only, CT no switch))")
for r in part("D"):
    ar, ct = r["res"]["Target-only"], r["res"]["ClearTree core (no switch)"]
    best = max(ar, ct)
    line = [f"{r['scen']:22s} s{r['seed']} Target-only={ar:.0f} CT={ct:.0f}"]
    for k, v in r["res"].items():
        if k.startswith("switch"):
            line.append(f"{k.split('from ')[1]}: {v[0]:.0f} ({pct(v[0], best):+.1f}%, spec {v[1]:.0%})")
    print(" | ".join(line))

print("\n== E: stop-boundary audit of ClearTree core (learned plug-in calibration; seeds pooled). Ratios are actual /")
print("   predicted mu-hat: dG = next layer under the fixed calibration, dA = actual acceptance gain; boundary =")
print("   |margin| <= 0.1 token; flips/regret: decisions whose sign changes when rescaled by the boundary ratio of dA")
import ext5 as V
recs = {}
for f in sorted(glob.glob("results/r6E.jsonl.*")):
    for l in open(f):
        o = json.loads(l)
        recs.setdefault(o["scen"], []).extend(o["rec"])
for scen, rec in recs.items():
    a = V.audit_summary(rec)
    print(f"{scen:16s} n={a['n']:5d} all dA/mu={a['all_dA_mu']:.3f} | stop: dG/mu={a['stop_dG_mu']:.3f} dA/dG={a['stop_dA_dG']:.3f} "
          f"dA/mu={a['stop_dA_mu']:.3f} | grow dA/mu={a['grow_dA_mu']:.3f} | boundary n={a['bnd_n']} dA/mu={a['bnd_dA_mu']:.3f} "
          f"(stop {a['bnd_stop_dA_mu']:.3f}, grow {a['bnd_grow_dA_mu']:.3f}) | flips {a['flips_bnd']:.0%} of boundary, "
          f"{a['flips_all']:.1%} of all | regret {a['regret_per_1000']:.1f} tokens / 1000 decisions")

print("\n== E2: calibration variants (results/r6b*.jsonl): tok/s and stop-decision ratios")
rb = []
for f in sorted(glob.glob("results/r6b*.jsonl")):
    rb += [json.loads(l) for l in open(f)]
for r in sorted(rb, key=lambda r: (r["scen"], r["seed"])):
    base = r["res"]["learned plug-in"]["rate"]
    print(f"{r['scen']:16s} s{r['seed']} " + " | ".join(
        f"{k}: {v['rate']:.1f} ({pct(v['rate'], base):+.1f}%) all dA/mu={v['all_dA_mu']:.2f} stop dA/mu={v['stop_dA_mu']:.2f} "
        f"stop dG/mu={v['stop_dG_mu']:.2f} regret={v['regret_per_1000']:.1f}" for k, v in r["res"].items()))
for k in ("posterior-integrated delta", "true parameters"):
    v = [pct(r["res"][k]["rate"], r["res"]["learned plug-in"]["rate"]) for r in rb]
    if v:
        print(f"  {k}: vs learned plug-in mean {st.mean(v):+.2f}% (min {min(v):+.2f}, max {max(v):+.2f})")
