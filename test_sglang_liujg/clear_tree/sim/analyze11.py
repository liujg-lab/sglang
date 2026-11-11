"""Summarise round 11 (results/r11C.*, r11W.*, r11S.*). Run from this directory."""
import glob, json, statistics as st

pct = lambda a, b: 100.0 * (a / b - 1)
R = lambda v: v[0] if isinstance(v, list) else v


def lines(pattern):
    return [json.loads(l) for f in sorted(glob.glob(pattern)) for l in open(f) if l.strip()]


C = lines("results/r11C.*.jsonl")
if C:
    print("== C: (W, K) table fitted on seed 7 (batch accounting with Theta per b, no switch); loss of the other cells")
    for o in sorted(C, key=lambda o: (o["b"], o["kind"])):
        r = o["res"]; b_ = max(r, key=r.get)
        print(f"   b={o['b']} {o['kind']:4s} best {b_:5s} {r[b_]:6.1f} | " + " ".join(f"{k}:{pct(v, r[b_]):+.1f}%" for k, v in r.items() if k != b_))

W10 = {o["seed"]: o["res"] for o in lines("results/r10W.*.jsonl")}
W11 = lines("results/r11W.*.jsonl")
if W11 and W10:
    print("\n== W: high workload, b=8, seeds 7-9; % vs static (2,16,0.05): overall / full batch | drain rate")
    agg = {}
    for o in W11:
        ref = W10[o["seed"]]["static (2,16,0.05)"]
        for k, v in o["res"].items():
            agg.setdefault(k, []).append((pct(v["rate"], ref["rate"]), pct(v["full_rate"], ref["full_rate"]), v.get("drain_rate", 0)))
        agg.setdefault("static (2,16,0.05)", []).append((0.0, 0.0, ref.get("drain_rate", 0)))
    for k, v in agg.items():
        print(f"   {k:28s} {st.mean(x[0] for x in v):+5.1f}% / {st.mean(x[1] for x in v):+5.1f}% | drain {st.mean(x[2] for x in v):5.0f} tok/s")

S = lines("results/r11S.*.jsonl")
simp = {(o["scen"], o["kind"], o["b"], o["seed"]): o["res"]["v2 simplified"] for o in lines("results/r11b.*.jsonl")}
for o in S:
    k = (o["scen"], o["kind"], o["b"], o["seed"])
    if k in simp:
        o["res"]["v2 simplified"] = simp[k]
if S:
    ct = {}
    for o in lines("results/r10S.*.jsonl"):
        ct.setdefault((o["scen"], o["kind"], o["b"]), {})[o["seed"]] = R(o["res"]["ClearTree K16+switch"])
    print("\n== S: seeds 8-10, % vs tuned static (all with the switch off at b=1); mean over seeds; [(W,K) use of v2 full]")
    groups = {}
    for o in S:
        groups.setdefault((o["scen"], o["kind"], o["b"]), []).append(o)
    names = ["v1", "v2 Theta per b", "v2 W/K by rho", "v2 full"] + (["v2 simplified"] if simp else [])
    agg = {}
    for key in sorted(groups, key=lambda k: (k[0], str(k[1]), k[2])):
        os_ = groups[key]
        cells = []
        for n in names:
            v = st.mean(pct(R(o["res"][n]), o["res"]["tuned static"]) for o in os_)
            agg.setdefault((key[0], n), []).append(v)
            cells.append(f"{n}: {v:+5.1f}%")
        if key in ct:
            v = st.mean(pct(ct[key][o["seed"]], o["res"]["tuned static"]) for o in os_ if o["seed"] in ct[key])
            agg.setdefault((key[0], "ClearTree K16 (r10, switch at all b)"), []).append(v)
            cells.append(f"ClearTree(r10): {v:+5.1f}%")
        use = {}
        for o in os_:
            for k, c in o["res"]["v2 full"][1].items():
                use[k] = use.get(k, 0) + c
        tot = sum(use.values())
        print(f"{key[0]:6s} {str(key[1] or ''):4s} b={key[2]} | " + " | ".join(cells)
              + " | [" + ", ".join(f"{k}:{v / tot:.0%}" for k, v in sorted(use.items(), key=lambda kv: -kv[1])) + "]")
    print("  group means (min):")
    for scen in ("pure", "mixed", "phased"):
        ks = [k for k in names + ["ClearTree K16 (r10, switch at all b)"] if (scen, k) in agg]
        print(f"   {scen}: " + "; ".join(f"{k} {st.mean(agg[(scen, k)]):+.2f}% ({min(agg[(scen, k)]):+.1f})" for k in ks))
