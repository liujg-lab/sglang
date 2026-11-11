"""Summarise round 16 (results/r16A.*, r16B.*). Run from this directory."""
import glob, json, statistics as st

pct = lambda a, b: 100.0 * (a / b - 1)


def lines(pattern):
    return [json.loads(l) for f in sorted(glob.glob(pattern)) for l in open(f) if l.strip()]


A = lines("results/r16A.*.jsonl")
if A:
    groups = {}
    for o in A:
        groups.setdefault((o["world"], o["kind"], o["b"]), []).append(o)
    print("== A: % vs the current text (early-stop K, growth over K_max); mean over seeds 8-10; [waste = calibrated mass grown beyond the chosen K]")
    agg = {}
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2])):
        os_ = groups[key]
        base = st.mean(o["res"]["early"]["rate"] for o in os_)
        cells = []
        for m in ("enum", "consistent"):
            v = pct(st.mean(o["res"][m]["rate"] for o in os_), base)
            agg.setdefault((key[0], m), []).append(v)
            cells.append(f"{m}: {v:+5.1f}%")
        w = {m: st.mean(o["res"][m]["waste"] for o in os_) for m in ("early", "enum", "consistent")}
        print(f"{key[0]:10s} {key[1]:4s} b={key[2]} | early {base:6.1f} | " + " | ".join(cells)
              + " | waste " + " / ".join(f"{m} {v:.1%}" for m, v in w.items()))
    print("  group means (min, max) vs early:")
    for wd in sorted({k[0] for k in groups}):
        print(f"   {wd:10s}: " + "; ".join(f"{m} {st.mean(agg[(wd, m)]):+.2f}% ({min(agg[(wd, m)]):+.1f}, {max(agg[(wd, m)]):+.1f})" for m in ("enum", "consistent")))

B = lines("results/r16B.*.jsonl")
if B:
    groups = {}
    for o in B:
        groups.setdefault((o["kind"], o["b"]), []).append(o)
    print("\n== B: verify cost x10 from spec round 150 on; % vs the oracle table; recovered = (drift - stale) / (oracle - stale)")
    rec, sv, dv = [], [], []
    for key in sorted(groups):
        os_ = groups[key]
        o_ = st.mean(o["res"]["oracle table"]["rate"] for o in os_)
        s_ = st.mean(o["res"]["stale table"]["rate"] for o in os_)
        d_ = st.mean(o["res"]["stale table + drift"]["rate"] for o in os_)
        r = (d_ - s_) / (o_ - s_) if o_ - s_ > 1e-9 else float("nan")
        sv.append(pct(s_, o_)); dv.append(pct(d_, o_))
        if o_ - s_ > 0.005 * o_:
            rec.append(r)
        pb = sum(o["res"]["stale table + drift"]["probes"] for o in os_); bu = sum(o["res"]["stale table + drift"]["bursts"] for o in os_)
        print(f"   {key[0]:4s} b={key[1]} | oracle {o_:6.1f} | stale {pct(s_, o_):+5.1f}% | stale+drift {pct(d_, o_):+5.1f}% | recovered {r:.0%} | probes {pb}, bursts {bu} | "
              f"corr {os_[0]['res']['stale table + drift']['corr']}")
    print(f"  mean: stale {st.mean(sv):+.2f}%, stale+drift {st.mean(dv):+.2f}%; recovered (cells where stale loses > 0.5%) mean {st.mean(rec):.0%}" if rec else "")

B2 = lines("results/r16B2.*.jsonl")
if B and B2:
    v2 = {(o["kind"], o["b"], o["seed"]): o["res"]["stale table + drift v2"] for o in B2}
    print("\n== B2: drift detection v2 (sustained level shift of the K in use, then a burst over all K); % vs the oracle table")
    rec, dv = [], []
    for key in sorted(groups):
        os_ = groups[key]
        o_ = st.mean(o["res"]["oracle table"]["rate"] for o in os_)
        s_ = st.mean(o["res"]["stale table"]["rate"] for o in os_)
        d2 = st.mean(v2[(o["kind"], o["b"], o["seed"])]["rate"] for o in os_)
        bu = sum(v2[(o["kind"], o["b"], o["seed"])]["bursts"] for o in os_)
        dv.append(pct(d2, o_))
        r = (d2 - s_) / (o_ - s_) if o_ - s_ > 1e-9 else float("nan")
        if o_ - s_ > 0.005 * o_:
            rec.append(r)
        print(f"   {key[0]:4s} b={key[1]} | stale {pct(s_, o_):+5.1f}% | v2 {pct(d2, o_):+5.1f}% | recovered {r:.0%} | bursts {bu} | corr {v2[(os_[0]['kind'], os_[0]['b'], os_[0]['seed'])]['corr']}")
    print(f"  mean: v2 {st.mean(dv):+.2f}%; recovered (cells where stale loses > 0.5%) mean {st.mean(rec):.0%}" if rec else "")
