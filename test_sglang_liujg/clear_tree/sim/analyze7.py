"""Summarise round 7: cost fits (results/fit7r_*.json) and evaluations (results/r7_*.jsonl). Run from this directory."""
import glob, json, statistics as st

pct = lambda a, b: 100.0 * (a / b - 1)
R = lambda v: v[0] if isinstance(v, list) else v

print("== A: cost fit to the real T0 grid (39 spec cells; mean / max |relative error|; b=4 held out)")
for f in sorted(glob.glob("results/fit7r_*.json")):
    o = json.load(open(f))
    for s, v in o["fits"].items():
        th = ", ".join(f"{n}={x:.3f}" for n, x in zip(["c0", "c_b", "v_tok", "d0", "d_req", "d_tok"], v["theta"]))
        print(f"{f.split('_')[1][:-5]:3s} a={o['a']} k={o.get('k', 6)} s={o.get('s_range')} {s:10s} "
              f"in-sample {v['err_mean']:.1%} / {v['err_max']:.1%}, held out {v['ho_mean']:.1%} / {v['ho_max']:.1%} | {th}")

for tag, pattern in (("C1 world, full cost structure", "results/r7_C1full.*.jsonl"),
                     ("C2 world, full cost structure", "results/r7_C2full.*.jsonl"),
                     ("original world (rounds 1-6 costs and acceptance)", "results/r7_orig.*.jsonl")):
    rows = []
    for f in sorted(glob.glob(pattern)):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    dec = {}
    for f in sorted(glob.glob(pattern.replace("r7_", "r7b_"))):
        for l in open(f):
            if l.strip():
                o = json.loads(l)
                dec[(o["scen"], o["seed"])] = o["res"]
    if not rows:
        continue
    rows.sort(key=lambda r: (r["scen"].split()[1], int(r["scen"].split("=")[1].split()[0]), r["seed"]))
    print(f"\n== B: {tag}; tok/s; CT = ClearTree core")
    agg = {}
    for r in rows:
        x = r["res"]
        ar, prod = x["Target-only"], x["production OPT(8,0.8)+b>=7 Target-only"]
        bs, bsk = x["best static"]; bo, bok = x["best static, old rule"]
        bss = x["best static+switch"]
        c4, c8 = x["ClearTree core W=4"], x["ClearTree core W=8"]
        cbest = max(c4, c8)
        csw, cw = x["ClearTree core+switch (best W)"]
        base = bss
        line = (f"{r['scen']:10s} s{r['seed']} Target-only {ar:6.1f} | prod {prod:6.1f} | best static {bs:6.1f} ({bsk}) "
                f"old rule {bo:6.1f} ({bok}, {pct(bo, bs):+.1f}%) | +switch {bss:6.1f} | CT W4 {c4:6.1f} W8 {c8:6.1f} "
                f"| CT+switch {csw:6.1f} (W{cw}) vs static+switch {pct(csw, base):+.1f}%, vs prod {pct(csw, prod):+.1f}%, "
                f"CT no switch vs best static {pct(cbest, bs):+.1f}%")
        d = dec.get((r["scen"], r["seed"]))
        if d:
            ds, dk = d["best decoupled static"]; dss = d["best decoupled static+switch"]
            line += (f" | decoupled static {ds:6.1f} ({dk}) +switch {dss:6.1f}: CT+switch vs it {pct(csw, dss):+.1f}%, "
                     f"decoupled vs static (both +switch) {pct(dss, bss):+.1f}%")
        print(line)
        key = "RTT 20" if "RTT=20" in r["scen"] else "RTT 4"
        a = agg.setdefault(key, dict(sw=[], prod=[], ns=[], old=[], dsw=[], dvs=[]))
        a["sw"].append(pct(csw, base)); a["prod"].append(pct(csw, prod)); a["ns"].append(pct(cbest, bs)); a["old"].append(pct(bo, bs))
        if d:
            a["dsw"].append(pct(csw, dss)); a["dvs"].append(pct(dss, bss))
    for key, a in agg.items():
        print(f"  {key}: CT+switch vs best static+switch mean {st.mean(a['sw']):+.2f}% (min {min(a['sw']):+.1f}, max {max(a['sw']):+.1f}); "
              f"vs production {st.mean(a['prod']):+.2f}%; CT vs best static (no switch) {st.mean(a['ns']):+.2f}%; "
              f"old-rule best static vs real-rule best static {st.mean(a['old']):+.2f}%  (n={len(a['sw'])})")
        if a["dsw"]:
            print(f"     decoupled static+switch vs static+switch {st.mean(a['dvs']):+.2f}%; CT+switch vs decoupled static+switch "
                  f"mean {st.mean(a['dsw']):+.2f}% (min {min(a['dsw']):+.1f}, max {max(a['dsw']):+.1f}) (n={len(a['dsw'])})")

# Part B3: ClearTree with K tied to W, against the static baselines and full ClearTree (C1 world, RTT 4, no switch)
coupled = {}
for f in sorted(glob.glob("results/r7c_C1.*.jsonl")):
    for l in open(f):
        if l.strip():
            o = json.loads(l)
            coupled[(o["scen"], o["seed"])] = o["res"]
if coupled:
    full = {}
    for f in sorted(glob.glob("results/r7_C1full.*.jsonl")):
        for l in open(f):
            if l.strip():
                o = json.loads(l); full[(o["scen"], o["seed"])] = o["res"]
    dec = {}
    for f in sorted(glob.glob("results/r7b_C1full.*.jsonl")):
        for l in open(f):
            if l.strip():
                o = json.loads(l); dec[(o["scen"], o["seed"])] = o["res"]
    print("\n== B3: where ClearTree's gain comes from (C1 world, RTT 4, no switch; % vs best coupled static OPT)")
    acc = dict(dec=[], cpl=[], ct=[])
    for key in sorted(coupled, key=lambda k: (int(k[0].split("=")[1].split()[0]), k[1])):
        x, c = full.get(key), coupled[key]
        if not x:
            continue
        bs = x["best static"][0]
        ds = dec[key]["best decoupled static"][0] if key in dec else float("nan")
        cp = max(c.values()); ct = max(x["ClearTree core W=4"], x["ClearTree core W=8"])
        print(f"{key[0]:10s} s{key[1]} best static {bs:6.1f} | decoupled static {ds:6.1f} ({pct(ds, bs):+.1f}%) | "
              f"CT with K=W {cp:6.1f} ({pct(cp, bs):+.1f}%) | CT {ct:6.1f} ({pct(ct, bs):+.1f}%)")
        if key in dec:
            acc["dec"].append(pct(ds, bs))
        acc["cpl"].append(pct(cp, bs)); acc["ct"].append(pct(ct, bs))
    print("  mean: " + ", ".join(f"{k} {st.mean(v):+.2f}%" for k, v in acc.items() if v))
