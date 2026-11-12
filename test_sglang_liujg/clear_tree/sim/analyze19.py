"""Summarize round 19 (run19.py) against the round-18 references of the same scenarios (r18A). usage: analyze19.py"""
import json, os
from collections import defaultdict

R = "results"


def load(names):
    out = []
    for f in names:
        p = os.path.join(R, f)
        if os.path.exists(p):
            out += [json.loads(l) for l in open(p)]
    return out


a18 = {(r["world"], r["kind"], r["b"], r["seed"]): r["res"] for r in load(["r18A.0.jsonl", "r18A.1.jsonl"])}
r19 = {(r["world"], r["kind"], r["b"], r["seed"]): r["res"] for r in load(["r19.0.jsonl", "r19.1.jsonl"])}
METHODS = ("fixed30", "ipw", "paired_cs", "sh")
V2 = ("paired_cs v2 eb", "paired_cs v2 nm")
_b = load(["r19B.0.jsonl", "r19B.1.jsonl"])
for r in _b:
    k = (r["world"], r["kind"], r["b"], r["seed"])
    if k in r19:
        r19[k].update(r["res"])
if _b and all(all(v in r19[k] for v in V2) for k in r19):
    METHODS = METHODS + V2
REFS = ("table", "replay14")
WORLDS = ("base", "tok x3", "tok x0.3", "change")


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def fixed_means(world, kind, b):
    ks = [k for k in a18 if k[:3] == (world, kind, b)]
    return {W: mean([a18[k][f"fixed {W}"]["rate"] for k in ks]) for W in (2, 4, 8)}


print("== Round 19: W calibration methods, one budget (200 rounds per window). % vs the hindsight best fixed W;")
print("   cal = share of spec rounds inside windows; regret = mean loss of the chosen W in the main bucket's windows")
print("   (best fixed W rate vs the chosen W's fixed rate); outcomes: switch / keep / keep-uncertain counts")
grp = defaultdict(lambda: defaultdict(list))
reg = defaultdict(list)
calsh = defaultdict(list)
outc = defaultdict(lambda: defaultdict(int))
for wd in WORLDS:
    for kind in ("mid", "high"):
        for b in (1, 4, 8):
            keys = [k for k in r19 if k[:3] == (wd, kind, b)]
            if not keys:
                continue
            fx = fixed_means(wd, kind, b)
            ref = max(fx.values())
            bw = max(fx, key=fx.get)
            fx_after = fixed_means("tok x3", kind, b) if wd == "change" else fx
            fx_before = fixed_means("base", kind, b) if wd == "change" else fx
            parts = [f"{wd:8s} {kind:4s} b={b} | best W={bw} {ref:6.1f}"]
            for v in REFS:
                d = 100 * (mean([a18[k][v]["rate"] for k in keys if k in a18]) / ref - 1)
                grp[wd][v].append(d)
                parts.append(f"{v} {d:+.1f}%")
            for m in METHODS:
                d = 100 * (mean([r19[k][m]["rate"] for k in keys]) / ref - 1)
                grp[wd][m].append(d)
                cs = mean([r19[k][m]["cal_rounds"] / max(1, r19[k][m]["spec_rounds"]) for k in keys])
                calsh[(wd, m)].append(cs)
                rg = []
                for k in keys:
                    for (rnd, eb, cls, kd, cur0, ch, nwin, oc) in r19[k][m]["events"]:
                        if eb != b:
                            continue
                        f = fx_after if kd == "cost" else fx_before
                        rg.append(100 * (1 - f[ch] / max(f.values())))
                        outc[(wd, m)][oc.split("-")[0] if oc.startswith("keep-r") else oc] += 1
                reg[(wd, m)] += rg
                parts.append(f"{m} {d:+.1f}% (cal {100 * cs:.0f}%, regret {mean(rg):.2f})")
            print(" | ".join(parts))
print()
print("  group means (min) vs the best fixed W:")
for wd in WORLDS:
    print(f"   {wd:8s}: " + "; ".join(f"{v} {mean(x):+.2f}% ({min(x):+.1f})" for v, x in grp[wd].items()))
print()
print("  per method: calibration share of rounds, mean regret of the chosen W per window, outcomes")
for m in METHODS:
    for wd in WORLDS:
        o = outc[(wd, m)]
        print(f"   {m:9s} {wd:8s}: cal {100 * mean(calsh[(wd, m)]):.1f}% | regret {mean(reg[(wd, m)]):.2f}% over "
              f"{len(reg[(wd, m)])} windows | " + ", ".join(f"{k} {v}" for k, v in sorted(o.items())))
print()
allw = defaultdict(list)
for wd in WORLDS:
    for v, x in grp[wd].items():
        allw[v] += x
print("  all 24 scenarios: " + "; ".join(f"{v} {mean(x):+.2f}%" for v, x in allw.items()))
print("  cost where the table is right (base, cells where the table is within 1% of the best), vs table:")
for v in ("replay14",) + METHODS:
    ds = [x - t for t, x in zip(grp["base"]["table"], grp["base"][v]) if t > -1.0]
    print(f"   {v:9s}: mean {mean(ds):+.2f} pt, worst {min(ds):+.2f} pt over {len(ds)} cells")
print("  recovered share of the table's loss (cells where the table loses > 1%):")
for v in ("replay14",) + METHODS:
    rec = []
    for wd in WORLDS:
        for t, x in zip(grp[wd]["table"], grp[wd][v]):
            if t < -1.0:
                rec.append((x - t) / -t)
    print(f"   {v:9s}: {100 * mean(rec):.0f}% over {len(rec)} cells")

# ---------------- round 19b long runs (r19L) vs round 18 long runs (r18L, same scenarios, seed 8) ----------------
L19 = load(["r19L.0.jsonl", "r19L.1.jsonl"])
L18 = load(["r18L.0.jsonl", "r18L.1.jsonl"])
if L19:
    lr = defaultdict(dict)
    for r in L18 + L19:
        lr[(r["world"], r["kind"], r["b"])][r["variant"]] = r["res"]
    print()
    print("== Round 19b long runs (30x requests, seed 8): % vs the table; tok x0.3 also vs the best fixed W")
    agg = defaultdict(list)
    for key in sorted(lr, key=lambda k: (k[0] != "base", k[1], k[2])):
        L = lr[key]
        if "table" not in L:
            continue
        fx = [v for v in L if v.startswith("fixed")]
        parts = [f"{key[0]:8s} {key[1]:4s} b={key[2]}"]
        for v in ("replay14", "rescore step no-confirm keep-wide") + V2:
            if v not in L:
                continue
            d = 100 * (L[v]["rate"] / L["table"]["rate"] - 1)
            agg[(key[0], v, "table")].append(d)
            txt = f"{v} {d:+.2f}%"
            if fx:
                db = 100 * (L[v]["rate"] / L[fx[0]]["rate"] - 1)
                agg[(key[0], v, "best")].append(db)
                txt += f" (vs best {db:+.2f}%)"
            if "cal_rounds" in L[v]:
                txt += f" [cal {100 * L[v]['cal_rounds'] / max(1, L[v]['spec_rounds']):.1f}%, sw {L[v]['switches']}]"
            parts.append(txt)
        print(" | ".join(parts))
    for wd in ("base", "tok x0.3"):
        print(f"   {wd:8s} means: " + "; ".join(
            f"{v} {mean(agg[(wd, v, 'table')]):+.2f}% vs table" + (f", {mean(agg[(wd, v, 'best')]):+.2f}% vs best" if agg.get((wd, v, 'best')) else "")
            for v in ("replay14", "rescore step no-confirm keep-wide") + V2 if agg.get((wd, v, "table"))))
