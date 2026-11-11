"""Summarise round 17 (results/r17S1.*, r17S2.*, r17S3.*) and the corrected round-16 B2 rerun (r16B2fix.*).
Run from this directory."""
import glob, json, statistics as st

pct = lambda a, b: 100.0 * (a / b - 1)


def lines(pattern):
    return [json.loads(l) for f in sorted(glob.glob(pattern)) for l in open(f) if l.strip()]


def mean(xs):
    xs = list(xs)
    return st.mean(xs) if xs else float("nan")


# ---------------- S1 ----------------
A = lines("results/r17S1.*.jsonl")
if A:
    print("== S1 cross-K stopping. % vs v10 (mean over seeds 8-10). diag (v10 runs, per rule-driven stop):")
    print("   cross(g) = a two-layer option with the g forecast is positive while no one-layer option is;")
    print("   lag1(true) = the true next layer pays; cross(true) = only two true layers pay; value = sum of those")
    print("   positive values (tokens, calibrated Draft mass) / committed tokens")
    groups = {}
    for o in A:
        groups.setdefault((o["world"], o["kind"], o["b"]), []).append(o)
    agg = {}
    for key in sorted(groups):
        os_ = groups[key]
        base = mean(o["res"]["v10"]["rate"] for o in os_)
        cells = []
        for m in ("opt2", "true1", "true2"):
            v = pct(mean(o["res"][m]["rate"] for o in os_), base)
            agg.setdefault((key[0], m), []).append(v)
            cells.append(f"{m} {v:+5.1f}%")
        dg = {k: sum(o["res"]["v10"]["dg"][k] for o in os_) for k in os_[0]["res"]["v10"]["dg"]}
        tok = sum(o["res"]["v10"]["tok"] for o in os_)
        s = max(1, dg["stops"])
        used = {m: sum(o["res"][m]["dg"]["two_only_used"] for o in os_) / max(1, sum(sum(o["res"][m]["k"].values()) for o in os_))
                for m in ("opt2", "true2")}
        for k_, v_ in (("cross_g", dg["cross_opt"] / s), ("cross_t", dg["cross_true"] / s), ("lag1_t", dg["lag1_true"] / s),
                       ("val_g", 100 * dg["cross_opt_val"] / tok), ("val_t", 100 * dg["cross_true_val"] / tok)):
            agg.setdefault((key[0], k_), []).append(v_)
        print(f"{key[0]:10s} {key[1]:4s} b={key[2]} | v10 {base:6.1f} | " + " | ".join(cells)
              + f" | stops {dg['stops']:4d}: cross(g) {dg['cross_opt'] / s:5.1%} (other K {dg['cross_opt_otherK'] / max(1, dg['cross_opt']):.0%}),"
              f" lag1(true) {dg['lag1_true'] / s:5.1%}, cross(true) {dg['cross_true'] / s:5.1%} | value g {100 * dg['cross_opt_val'] / tok:.2f}%, true {100 * dg['cross_true_val'] / tok:.2f}%"
              f" | two-layer-only decisions per round: opt2 {used['opt2']:.1%}, true2 {used['true2']:.1%}")
    print("  group means (min, max):")
    for wd in sorted({k[0] for k in groups}):
        print(f"   {wd:10s}: " + "; ".join(f"{m} {mean(agg[(wd, m)]):+.2f}% ({min(agg[(wd, m)]):+.1f}, {max(agg[(wd, m)]):+.1f})" for m in ("opt2", "true1", "true2"))
              + f"; stops: cross(g) {mean(agg[(wd, 'cross_g')]):.1%}, cross(true) {mean(agg[(wd, 'cross_t')]):.1%}, lag1(true) {mean(agg[(wd, 'lag1_t')]):.1%};"
              f" value g {mean(agg[(wd, 'val_g')]):.2f}%, true {mean(agg[(wd, 'val_t')]):.2f}% of tokens")

# ---------------- S2 ----------------
B = lines("results/r17S2.*.jsonl")
R13 = lines("results/r13.*.jsonl")
ref = {}
for o in R13:
    ref[(o["kind"], o["b"], o["seed"], o["verify_mult"])] = o["res"]
if B:
    print("\n== S2 cross-K marginal calibration. Rates: mean over seeds; vs r13 best fixed (W, K) where available")
    print("   f(band) = sum actual band acceptance / sum c_i * band mass (same-tree replay, rounds after warm-up, pooled seeds);")
    print("   replay columns on v10 runs: chosen K vs K re-chosen with hindsight f (same rounds, same trees), share of rounds changed;")
    print("   best fixed K on the same trees; per-round oracle K (knows each round's actual tokens) as an upper bound")
    groups = {}
    for o in B:
        groups.setdefault((o["kind"], o["vm"], o["b"]), []).append(o)
    pooled = {}
    aggf = []
    for key in sorted(groups, key=lambda k: (k[1], ["low", "mid", "high"].index(k[0]), k[2])):
        os_ = groups[key]
        r0 = mean(o["res"]["v10"]["rate"] for o in os_)
        r1 = mean(o["res"]["v10 + f_band"]["rate"] for o in os_)
        refs = [ref.get((o["kind"], o["b"], o["seed"], o["vm"])) for o in os_]
        line = f"{key[0]:4s} verify x{key[1]:<4g} b={key[2]} | v10 {r0:6.1f} | +f_band {pct(r1, r0):+5.1f}%"
        if all(refs):
            fx = {}
            for rr in refs:
                for n, v in rr.items():
                    if n.startswith("fixed"):
                        fx.setdefault(n, []).append(v["rate"])
            fxm = {n: mean(v) for n, v in fx.items()}
            bf = max(fxm, key=fxm.get)
            line += (f" | vs best fixed {bf[6:]}: v10 {pct(r0, fxm[bf]):+5.1f}%, +f_band {pct(r1, fxm[bf]):+5.1f}%"
                     f" (r13: table {pct(mean(rr['table']['rate'] for rr in refs), fxm[bf]):+.1f}%, online K {pct(mean(rr['online K']['rate'] for rr in refs), fxm[bf]):+.1f}%)")
            aggf.append((pct(r0, fxm[bf]), pct(r1, fxm[bf]), key))
        sums = {}
        for o in os_:
            rp = o["res"]["v10"]["replay"]
            if not rp:
                continue
            for k, v in rp["bias"].items():
                band, ph, db = k.split("|")
                for kk in (band, f"{band}|{ph}", f"{band}|{db}", f"{band}|{key[0]}", f"{band}|x{key[1]:g}", f"{band}|b{key[2]}"):
                    s = pooled.setdefault(kk, [0.0, 0.0, 0])
                    s[0] += v[0]; s[1] += v[1]; s[2] += v[2]
                s = sums.setdefault(band, [0.0, 0.0])
                s[0] += v[0]; s[1] += v[1]
        f_ = " ".join(f"f{b_}={v[1] / v[0]:.2f}" for b_, v in sorted(sums.items(), key=lambda x: int(x[0])) if v[0] > 0)
        rps = [o["res"]["v10"]["replay"] for o in os_ if o["res"]["v10"]["replay"]]
        if rps:
            ch = mean(r["rate_chosen"] for r in rps)
            co = pct(mean(r["rate_corrected"] for r in rps), ch)
            bfk = max(rps[0]["rate_fixed"], key=lambda k: mean(r["rate_fixed"][k] for r in rps))
            bfx = pct(mean(r["rate_fixed"][bfk] for r in rps), ch)
            orc = pct(mean(r["rate_oracle_round"] for r in rps), ch)
            sh = mean(r["share_changed"] for r in rps)
            line += f" | {f_} | replay: re-chosen with f {co:+.2f}% ({sh:.0%} rounds), best fixed K={bfk} {bfx:+.2f}%, per-round oracle {orc:+.2f}%"
        kl = {}
        for o in os_:
            for n in ("v10", "v10 + f_band"):
                for k, v in o["res"][n]["k"].items():
                    kl.setdefault(n, {}).setdefault(k, 0)
                    kl[n][k] += v
        line += " | K use " + "; ".join(n.replace("v10 + ", "+") + " " + ",".join(f"{k}:{v / sum(d.values()):.0%}" for k, v in sorted(d.items(), key=lambda x: int(x[0]))) for n, d in kl.items())
        print(line)
    if aggf:
        print(f"  mean vs best fixed (W, K): v10 {mean(a for a, _, _ in aggf):+.2f}%, +f_band {mean(b for _, b, _ in aggf):+.2f}%;"
              f" verify x10 only: v10 {mean(a for a, _, k in aggf if k[1] == 10):+.2f}%, +f_band {mean(b for _, b, k in aggf if k[1] == 10):+.2f}%")
    print("  pooled f(band) = actual / predicted marginal (v10 runs):")
    for band in ("8", "16", "32"):
        keys = sorted(k for k in pooled if k.split("|")[0] == band)
        print(f"   band ..{band}: " + "; ".join(f"{k.split('|', 1)[1] if '|' in k else 'all'} {pooled[k][1] / pooled[k][0]:.2f}" for k in keys if pooled[k][0] > 0))

# ---------------- S3 ----------------
C = lines("results/r17S3.*.jsonl")
if C:
    print("\n== S3 the unused K = 32 becomes cheaper (penalty 15 + 2 b ms removed from spec round 150 on). Rates mean over seeds")
    groups = {}
    for o in C:
        groups.setdefault((o["world"], o["kind"], o["b"]), []).append(o)
    names = ["oracle table", "stale table", "v2", "v2 + probe 1%", "v2 + probe 3%", "v2 + near-tie"]
    agg = {}
    for key in sorted(groups):
        os_ = groups[key]
        r = {n: mean(o["res"][n]["rate"] for o in os_) for n in names if n in os_[0]["res"]}
        refn = "oracle table" if key[0] == "cheap32" else "stale table"
        cells = []
        for n in names:
            if n not in r or n == refn:
                continue
            v = pct(r[n], r[refn])
            agg.setdefault((key[0], n), []).append(v)
            bu = sum(o["res"][n]["bursts"] for o in os_)
            pr = sum(o["res"][n]["probes"] for o in os_)
            tr = sum(o["res"][n]["probe_trig"] for o in os_)
            c32 = [o["res"][n]["corr"].get(f"{key[2]},32") for o in os_]
            c32 = [c for c in c32 if c is not None]
            cells.append(f"{n} {v:+5.2f}% [b{bu} p{pr} t{tr}" + (f" c32 {mean(c32):+.0f}" if c32 else "") + "]")
        if key[0] == "cheap32":
            k32 = {n: sum(o["res"][n]["k"].get("32", 0) for o in os_) / sum(sum(o["res"][n]["k"].values()) for o in os_) for n in r}
            extra = f" | K=32 share: oracle {k32['oracle table']:.0%}, stale {k32['stale table']:.0%}, probe3% {k32['v2 + probe 3%']:.0%}"
        else:
            extra = ""
        print(f"{key[0]:8s} {key[1]:4s} b={key[2]} | {refn} {r[refn]:6.1f} | " + " | ".join(cells) + extra)
    print("  group means (min) vs the reference (cheap32: oracle table; nochange: stale table = right table):")
    for wd in ("cheap32", "nochange"):
        print(f"   {wd:8s}: " + "; ".join(f"{n} {mean(agg[(wd, n)]):+.2f}% ({min(agg[(wd, n)]):+.2f})" for n in names if (wd, n) in agg))
    if any(k[0] == "cheap32" for k in agg):
        st_ = agg[("cheap32", "stale table")]
        for n in ("v2", "v2 + probe 1%", "v2 + probe 3%", "v2 + near-tie"):
            rec = [(x - s) / (-s) for x, s in zip(agg[("cheap32", n)], st_) if s < -0.3]
            print(f"   recovered by {n} (cells where stale loses > 0.3%): " + (f"{mean(rec):.0%} over {len(rec)} cells" if rec else "no such cells"))

# ---------------- round 16 B2 rerun ----------------
B16 = lines("results/r16B.*.jsonl")
B2f = lines("results/r16B2fix.*.jsonl")
B2o = lines("results/r16B2.*.jsonl")
if B16 and B2f:
    print("\n== Round 16 B2 rerun with the n0 fix (drift v2 no longer overwrites the Theta prior weight); % vs the oracle table")
    v2f = {(o["kind"], o["b"], o["seed"]): o["res"]["stale table + drift v2"] for o in B2f}
    v2o = {(o["kind"], o["b"], o["seed"]): o["res"]["stale table + drift v2"] for o in B2o}
    groups = {}
    for o in B16:
        groups.setdefault((o["kind"], o["b"]), []).append(o)
    rec_f, rec_o, dv_f, dv_o, sv, cost_f = [], [], [], [], [], []
    for key in sorted(groups):
        os_ = groups[key]
        o_ = mean(o["res"]["oracle table"]["rate"] for o in os_)
        s_ = mean(o["res"]["stale table"]["rate"] for o in os_)
        df = mean(v2f[(o["kind"], o["b"], o["seed"])]["rate"] for o in os_)
        do = mean(v2o[(o["kind"], o["b"], o["seed"])]["rate"] for o in os_) if v2o else float("nan")
        bu = sum(v2f[(o["kind"], o["b"], o["seed"])]["bursts"] for o in os_)
        sv.append(pct(s_, o_)); dv_f.append(pct(df, o_)); dv_o.append(pct(do, o_))
        if o_ - s_ > 0.005 * o_:
            rec_f.append((df - s_) / (o_ - s_)); rec_o.append((do - s_) / (o_ - s_))
        else:
            cost_f.append(pct(df, s_))
        print(f"   {key[0]:4s} b={key[1]} | stale {pct(s_, o_):+5.2f}% | v2 fixed {pct(df, o_):+5.2f}% (old {pct(do, o_):+5.2f}%) | bursts {bu}")
    print(f"  mean: stale {mean(sv):+.2f}%, v2 fixed {mean(dv_f):+.2f}% (old {mean(dv_o):+.2f}%); recovered (stale loses > 0.5%) fixed {mean(rec_f):.0%}"
          f" (old {mean(rec_o):.0%}); cells where stale loses <= 0.5%: v2 vs stale {', '.join(f'{c:+.2f}%' for c in cost_f)}")
