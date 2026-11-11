"""Summarise round 9 (results/r9D.*, r9R.*), with round-8 references (r8G grid, r8M mixed). Run from this directory."""
import glob, json, statistics as st
import run8 as R8

pct = lambda a, b: 100.0 * (a / b - 1)
WL = ("low", "mid", "high")


def lines(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        out += [json.loads(l) for l in open(f) if l.strip()]
    return out


FIT = json.load(open("results/fit7r_C1.json"))
ar = lambda b: 1000.0 * b / (FIT["a0"] + FIT["a_b"] * b)          # Target-only tok/s (no noise)
G = R8.load_grids()
best = {k: max(g.values()) for k, g in G.items()}
D = {(o["kind"], o["b"]): o["res"] for o in lines("results/r9D.*.jsonl")}
R = lines("results/r9R.*.jsonl")
setup = next((o for o in R if o["part"] == "R-setup"), None)
RR = {(o["scen"], o["kind"], o["b"]): o["res"] for o in R if o["part"] == "R"}
M8 = {o["b"]: o["res"] for o in lines("results/r8M.*.jsonl")}

if D:
    print("== D: ClearTree core variants (W=4, seed 7). Cells: % vs the workload's best static | tokens per request per round | layer forwards per round")
    agg = {}
    for kind in WL:
        for b in R8.BS:
            r = D.get((kind, b))
            if not r:
                continue
            cells = []
            for name, v in r.items():
                d = pct(v["rate"], best[(kind, b)])
                agg.setdefault(name, []).append(d)
                cells.append(f"{name}: {d:+5.1f}% {v['tok_req_round']:.2f} {v['layers_round']:.2f}")
            print(f"{kind:5s} b={b}: " + " | ".join(cells))
    print("  mean % vs best static: " + ", ".join(f"{k} {st.mean(v):+.2f}% (n={len(v)})" for k, v in agg.items()))
    print("\n   stop-boundary audit (actual / predicted; boundary = |margin| <= 0.1 token)")
    for (kind, b), r in sorted(D.items()):
        for name, v in r.items():
            a = v.get("audit")
            if a:
                print(f"   {kind:5s} b={b} {name:10s} n={a['n']:5d} stop dG/mu {a['stop_dG_mu']:.3f} dA/mu {a['stop_dA_mu']:.3f} | grow dA/mu "
                      f"{a['grow_dA_mu']:.3f} | boundary n={a['bnd_n']} dA/mu {a['bnd_dA_mu']:.3f} flips {a['flips_bnd']:.0%} "
                      f"regret {a['regret_per_1000']:.1f}/1000")

if setup:
    print(f"\n== R setup: rule (W,K)=({setup['W']},{setup['K']}), its best-tau share of each workload's optimum {setup['frac']:.1%}; "
          f"tau table {setup['table']}; rho at tau 0.05/0.4/0.8 {setup['rho']}; thresholds {setup['thresholds']}")

if RR:
    print("\n== R pure workloads (seed 7; % vs the workload's best static on the full round-8 grid, in-sample)")
    aggp = {}
    for kind in WL:
        for b in R8.BS:
            r = RR.get(("pure", kind, b))
            if not r:
                continue
            ref = best[(kind, b)]
            row = {"global static (4,16,0.4)": G[(kind, b)]["4,16,0.4"]}
            if (kind, b) in D:
                row["ClearTree base"] = D[(kind, b)]["base"]["rate"]
                row["ClearTree K32 fd1.0"] = D[(kind, b)]["K32 fd1.0"]["rate"]
            for name, v in r.items():
                row[name] = v["rate"]
            for k, v in row.items():
                aggp.setdefault(k, []).append(pct(v, ref))
            extra = r.get(next((k for k in r if k.startswith("rho")), ""), {})
            print(f"{kind:5s} b={b} best static {ref:6.1f} | " + " | ".join(f"{k}: {pct(v, ref):+5.1f}%" for k, v in row.items())
                  + (f" | tau share {extra.get('tau_share')}" if extra else ""))
    print("  mean: " + "; ".join(f"{k} {st.mean(v):+.2f}% (min {min(v):+.1f})" for k, v in aggp.items()))

    print("\n== R mixed (interleaved, seed 8; % vs the best static for the mix from round 8)")
    for b in R8.BS:
        r = RR.get(("mixed", None, b))
        if not r or b not in M8:
            continue
        m = M8[b]
        ref = max(m["static"].values())
        row = {"ClearTree base (r8)": max(v for k, v in m.items() if k.startswith("ClearTree"))}
        row.update({k: v["rate"] for k, v in r.items()})
        print(f"b={b} best static {ref:6.1f} | " + " | ".join(f"{k}: {pct(v, ref):+5.1f}%" for k, v in row.items())
              + f" | Target-only {ar(b):.1f}")

    print("\n== R phased (low, then high, then mid requests; seed 8; % vs the best static candidate run on the same requests)")
    for b in (4, 8):
        r = RR.get(("phased", None, b))
        if not r:
            continue
        stat = {k: v["rate"] for k, v in r.items() if k.startswith("static")}
        bk = max(stat, key=stat.get)
        print(f"b={b} best static {bk} {stat[bk]:6.1f} | " + " | ".join(f"{k}: {pct(v['rate'], stat[bk]):+5.1f}%" for k, v in r.items()
                                                                       if not k.startswith("static") or k == bk)
              + " | other statics " + ", ".join(f"{k[7:]} {pct(v, stat[bk]):+.1f}%" for k, v in stat.items() if k != bk))
