"""Summarise round 10 (results/r10W.*, r10S.*). Run from this directory."""
import glob, json, statistics as st

pct = lambda a, b: 100.0 * (a / b - 1)
R = lambda v: v[0] if isinstance(v, list) else v


def lines(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        out += [json.loads(l) for l in open(f) if l.strip()]
    return out


W = lines("results/r10W.*.jsonl")
if W:
    print("== W: high workload, b=8. Overall rate | full-batch rate | full-batch tokens/request/round, ms/round, layers/round | drain time share")
    agg = {}
    for o in sorted(W, key=lambda o: o["seed"]):
        ref, reff = o["res"]["static (2,16,0.05)"]["rate"], o["res"]["static (2,16,0.05)"]["full_rate"]
        print(f"seed {o['seed']}")
        for k, v in o["res"].items():
            agg.setdefault(k, []).append((pct(v["rate"], ref), pct(v["full_rate"], reff), v["drain_time_share"]))
            print(f"   {k:40s} {v['rate']:7.1f} ({pct(v['rate'], ref):+5.1f}%) | full {v['full_rate']:7.1f} ({pct(v['full_rate'], reff):+5.1f}%) "
                  f"| {v['full_tok_req_round']:.2f} tok {v['full_ms_round']:6.1f} ms {v['full_layers_round']:.2f} layers | drain {v['drain_time_share']:.0%} "
                  f"(drain rate {v.get('drain_rate', 0):.0f})")
    print("  mean over seeds vs static (2,16,0.05): overall / full batch / drain share")
    for k, v in agg.items():
        print(f"   {k:40s} {st.mean(x[0] for x in v):+5.1f}% / {st.mean(x[1] for x in v):+5.1f}% / {st.mean(x[2] for x in v):.0%}")

S = lines("results/r10S.*.jsonl")
if S:
    print("\n== S: seeds 8-10 (out of sample). % vs tuned static + switch (pure: per-workload best on seed 7; mixed/phased: "
          "round-8 best for the mix); [spec time share]")
    groups = {}
    for o in S:
        groups.setdefault((o["scen"], o["kind"], o["b"]), []).append(o["res"])
    agg = {}
    for key in sorted(groups, key=lambda k: (k[0], str(k[1]), k[2])):
        rs = groups[key]
        names = [k for k in rs[0] if k != "tuned static+switch"]
        cells = []
        for k in names:
            vals = [pct(R(r[k]), R(r["tuned static+switch"])) for r in rs if k in r]
            share = [r[k][-1] for r in rs if isinstance(r.get(k), list)]
            cells.append(f"{k}: {st.mean(vals):+5.1f}%" + (f" [{st.mean(share):.0%}]" if share else ""))
            agg.setdefault((key[0], k), []).append(st.mean(vals))
        tuned = rs[0]["tuned static+switch"]
        print(f"{key[0]:6s} {str(key[1] or ''):4s} b={key[2]} n={len(rs)} tuned {tuned[1]} [{st.mean(r['tuned static+switch'][2] for r in rs):.0%}] | " + " | ".join(cells))
    print("  group means (mean over scenarios of the seed mean; min in brackets):")
    for scen in ("pure", "mixed", "phased"):
        ks = sorted({k for (s, k) in agg if s == scen})
        print(f"   {scen}: " + "; ".join(f"{k} {st.mean(agg[(scen, k)]):+.2f}% ({min(agg[(scen, k)]):+.1f})" for k in ks))
