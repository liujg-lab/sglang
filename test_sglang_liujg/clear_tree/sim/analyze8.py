"""Summarise round 8 (results/r8G, r8P, r8M). Run from this directory."""
import glob, json, statistics as st

pct = lambda a, b: 100.0 * (a / b - 1)
WL = ("low", "mid", "high")
fit = json.load(open("results/fit7r_C1.json"))
ar = lambda b: 1000.0 * b / (fit["a0"] + fit["a_b"] * b)          # Target-only tok/s (no noise)


def load(part):
    out = {}
    for f in sorted(glob.glob(f"results/r8{part}.*.jsonl")):
        for l in open(f):
            if l.strip():
                o = json.loads(l)
                out[(o["kind"], o["b"])] = o["res"]
    return out


G, P, M = load("G"), load("P"), load("M")
BS = sorted({b for _, b in G})
best = {k: max(v["grid"], key=v["grid"].get) for k, v in G.items()}

print("== A: best static (W,K,tau) per workload and b (seed 7), tok/s; Target-only for reference")
for kind in WL:
    print(f"{kind:5s} " + " | ".join(f"b={b}: {best[(kind, b)]} {G[(kind, b)]['grid'][best[(kind, b)]]:.1f} "
                                     f"(Target-only {ar(b):.1f})" for b in BS if (kind, b) in G))

print("\n== B: loss from using another workload's best config (% vs the evaluated workload's own best, same b)")
for b in BS:
    print(f"b={b}: rows = config taken from, columns = evaluated on")
    for src in WL:
        if (src, b) not in G:
            continue
        c = best[(src, b)]
        cells = []
        for dst in WL:
            if (dst, b) in G:
                g = G[(dst, b)]["grid"]
                cells.append(f"{dst}: {pct(g[c], g[best[(dst, b)]]):+5.1f}%")
        print(f"   {src:5s} {c:12s} " + "  ".join(cells))

print("\n== B2: same, with an ideal switch (each side is max(Target-only, the config); % vs max(Target-only, own best))")
for b in BS:
    for src in WL:
        if (src, b) not in G:
            continue
        c = best[(src, b)]
        cells = []
        for dst in WL:
            if (dst, b) in G:
                g = G[(dst, b)]["grid"]
                cells.append(f"{dst}: {pct(max(ar(b), g[c]), max(ar(b), g[best[(dst, b)]])):+5.1f}%")
        print(f"   b={b} {src:5s} {c:12s} " + "  ".join(cells))

print("\n== C: one config for everything (all workloads and b)")
keys = [k for k in G]
cfgs = list(next(iter(G.values()))["grid"])
score = {c: [G[k]["grid"][c] / G[k]["grid"][best[k]] for k in keys] for c in cfgs}
by_min = sorted(cfgs, key=lambda c: min(score[c]), reverse=True)
by_mean = sorted(cfgs, key=lambda c: st.mean(score[c]), reverse=True)
for name, c in (("max-min", by_min[0]), ("max-mean", by_mean[0]), ("round-7 global", "4,16,0.4")):
    v = score[c]
    worst = min(range(len(keys)), key=lambda i: v[i])
    print(f"   {name:15s} {c:12s} mean {100 * (st.mean(v) - 1):+.2f}%, worst {100 * (v[worst] - 1):+.2f}% at {keys[worst]}")
for kind in WL:
    ks = [k for k in keys if k[0] == kind]
    if not ks:
        continue
    c = max(cfgs, key=lambda c: min(G[k]["grid"][c] / G[k]["grid"][best[k]] for k in ks))
    print(f"   {kind:5s} one config across b: {c} worst {min(pct(G[k]['grid'][c], G[k]['grid'][best[k]]) for k in ks):+.2f}%")

if P:
    print("\n== D: ClearTree core with no tuning, per workload (better of W=4 and W=8; % vs that workload's best static; "
          "and vs the mid workload's best static config transferred)")
    for kind in WL:
        line = []
        for b in BS:
            if (kind, b) not in P or (kind, b) not in G or ("mid", b) not in G:
                continue
            ct = max(P[(kind, b)].values())
            g = G[(kind, b)]["grid"]
            line.append(f"b={b}: {ct:.1f} ({pct(ct, g[best[(kind, b)]]):+.1f}% vs own best, "
                        f"{pct(ct, g[best[('mid', b)]]):+.1f}% vs mid's config)")
        print(f"   {kind:5s} " + " | ".join(line))

if M:
    print("\n== E: mixed workload (a third from each, interleaved; seed 8), tok/s and % vs the best static for the mix")
    for (_, b), r in sorted(M.items(), key=lambda kv: kv[0][1]):
        st_ = r["static"]
        bm = max(st_, key=st_.get)
        ty, W, K, taus = r["typed tau"]
        ct = {k: v for k, v in r.items() if k.startswith("ClearTree")}
        ctb = max(ct.values())
        trans = " ".join(f"{k}'s {c}: {pct(st_[c], st_[bm]):+.1f}%" for k, c in r["best per workload"].items())
        print(f"   b={b} {r['types']} best static for mix {bm} {st_[bm]:.1f} | {trans} | typed tau (W={W},K={K},{taus}) "
              f"{ty:.1f} ({pct(ty, st_[bm]):+.1f}%) | ClearTree " + " ".join(f"{k[-3:]} {v:.1f} ({pct(v, st_[bm]):+.1f}%)" for k, v in ct.items())
              + f" | Target-only {ar(b):.1f}")
