"""Summarise round 12 (results/r12R.json, r12S.*.jsonl). Run from this directory."""
import glob, json, statistics as st
import numpy as np

pct = lambda a, b: 100.0 * (a / b - 1)


def lines(pattern):
    return [json.loads(l) for f in sorted(glob.glob(pattern)) for l in open(f) if l.strip()]


try:
    R = json.load(open("results/r12R.json"))
except FileNotFoundError:
    R = None
if R:
    print("== R: rho by workload over (W, K) x tau (b=4, seed 9): old estimator / opportunity-based")
    for kind in ("low", "mid", "high"):
        vals = {k.split("|")[1]: v for k, v in R["rho"].items() if k.startswith(kind + "|")}
        old = [v[0] for v in vals.values()]; opp = [v[1] for v in vals.values()]
        print(f"   {kind:4s} old {np.mean(old):.3f} (range {min(old):.3f}-{max(old):.3f}, sd {np.std(old):.3f}) | "
              f"opp {np.mean(opp):.3f} (range {min(opp):.3f}-{max(opp):.3f}, sd {np.std(opp):.3f})")
        for wk in ("2,16", "4,16", "4,32", "8,32"):
            cells = [f"tau{t}: {vals[f'{wk},{t}'][0]:.3f}/{vals[f'{wk},{t}'][1]:.3f}" for t in (0.05, 0.4, 0.8)]
            print(f"        W,K={wk:5s} " + " ".join(cells))
    print(f"   opportunity-based high threshold (midpoint of mid and high means): {R['th_opp']:.3f}")

S = lines("results/r12S.*.jsonl")
if S:
    ref = {(o["scen"], o["kind"], o["b"], o["seed"]): o["res"]["tuned static"] for o in lines("results/r11S.*.jsonl")}
    r11 = {(o["scen"], o["kind"], o["b"], o["seed"]): o["res"]["v2 simplified"][0] for o in lines("results/r11b.*.jsonl")}
    bad = [k for o in S for k in [(o["scen"], o["kind"], o["b"], o["seed"])] if abs(o["res"]["v11 repro"]["rate"] - r11[k]) > 1e-6]
    print(f"\n== S: seeds 8-10; regression v11 repro == round 11: {'OK' if not bad else 'MISMATCH ' + str(bad[:3])}")
    names = list(S[0]["res"].keys())
    groups = {}
    for o in S:
        groups.setdefault((o["scen"], o["kind"], o["b"]), []).append(o)
    agg = {}
    print("   % vs round 11's tuned static (switch off at b=1), mean over seeds")
    for key in sorted(groups, key=lambda k: (k[0], str(k[1]), k[2])):
        os_ = groups[key]
        cells = []
        for n in names:
            v = st.mean(pct(o["res"][n]["rate"], ref[(o["scen"], o["kind"], o["b"], o["seed"])]) for o in os_)
            agg.setdefault((key[0], n), []).append(v)
            cells.append(f"{n}: {v:+5.1f}%")
        print(f"{key[0]:6s} {str(key[1] or ''):4s} b={key[2]} | " + " | ".join(cells))
    print("  group means (min):")
    for scen in ("pure", "mixed", "phased"):
        print(f"   {scen}: " + "; ".join(f"{n} {st.mean(agg[(scen, n)]):+.2f}% ({min(agg[(scen, n)]):+.1f})" for n in names))
    print("  all 14 scenarios: " + "; ".join(f"{n} {st.mean([v for (s_, nn), vs in agg.items() if nn == n for v in vs]):+.2f}%" for n in names))

    print("\n== Exit audit (v12 all; 15% of layer decisions; true increments from cloned trees grown 1-2 layers)")
    print("   regret_pred: greedy choice with true next-layer increments minus the choice made (one step, tokens net of Theta*cost)")
    print("   gap_struct : two-layer optimum (any kept subset, then the best bucket) minus greedy with true increments")
    for key in sorted(groups, key=lambda k: (k[0], str(k[1]), k[2])):
        a = [o["res"]["v12 all"]["audit"] for o in groups[key]]
        n = sum(x["n"] for x in a); tok = sum(x["tokens"] for x in a)
        if not n:
            print(f"   {key[0]:6s} {str(key[1] or ''):4s} b={key[2]}: no audited decisions"); continue
        rp = sum(x["regret_pred"] for x in a); gs = sum(x["gap_struct"] for x in a)
        # audited decisions are a 15% sample: scale to all decisions, express per committed token
        print(f"   {key[0]:6s} {str(key[1] or ''):4s} b={key[2]}: audited {n:5d} | regret_pred {rp / n:.3f} tok/decision, "
              f"{sum(x['regret_pos'] for x in a) / n:.0%} > 0, ~{100 * rp / 0.15 / tok:.1f}% of tokens | gap_struct {gs / n:.4f} tok/decision, "
              f"{sum(x['gap_pos'] for x in a) / n:.1%} > 0, ~{100 * gs / 0.15 / tok:.2f}% of tokens")

B = lines("results/r12b.*.jsonl")
if B:
    ref = {(o["scen"], o["kind"], o["b"], o["seed"]): o["res"]["tuned static"] for o in lines("results/r11S.*.jsonl")}
    names = list(B[0]["res"].keys())
    groups = {}
    for o in B:
        groups.setdefault((o["scen"], o["kind"], o["b"]), []).append(o)
    agg = {}
    print("\n== 12b: % vs round 11's tuned static, mean over seeds 8-10 (all with Theta per b, simplified table, conservative b=1 switch)")
    for key in sorted(groups, key=lambda k: (k[0], str(k[1]), k[2])):
        cells = []
        for n in names:
            v = st.mean(pct(o["res"][n]["rate"], ref[(o["scen"], o["kind"], o["b"], o["seed"])]) for o in groups[key])
            agg.setdefault((key[0], n), []).append(v)
            cells.append(f"{n}: {v:+5.1f}%")
        print(f"{key[0]:6s} {str(key[1] or ''):4s} b={key[2]} | " + " | ".join(cells))
    print("  group means (min):")
    for scen in ("pure", "mixed", "phased"):
        print(f"   {scen}: " + "; ".join(f"{n} {st.mean(agg[(scen, n)]):+.2f}% ({min(agg[(scen, n)]):+.1f})" for n in names))
    print("  all 14 scenarios: " + "; ".join(f"{n} {st.mean([v for (s_, nn), vs in agg.items() if nn == n for v in vs]):+.2f}%" for n in names))
