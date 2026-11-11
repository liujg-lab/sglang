"""Compare the exit tests of the D41 switch (P8/5%): sliding window (results/r4p*.jsonl) vs disjoint blocks
(results/r5p*.jsonl) and the two-threshold test (results/r5b*.jsonl), against round 3 (results/r3b*.jsonl). Run from this directory."""
import json, statistics as st

def load(files):
    out = {}
    for f in files:
        for l in open(f"results/{f}"):
            o = json.loads(l); out[(o["scen"], o["seed"])] = o
    return out

r3, r4, r5 = load(["r3b0.jsonl", "r3b1.jsonl"]), load(["r4p0.jsonl", "r4p1.jsonl"]), load(["r5p0.jsonl", "r5p1.jsonl"])
for k, o in load(["r5b0.jsonl", "r5b1.jsonl"]).items():   # round 5b: two-threshold exit test
    r5[k]["res"].update(o["res"])
pct = lambda a, b: 100.0 * (a / b - 1)
V = ["P8/5%", "P8/5% B20x1", "P8/5% B20x2", "P8/5% B20x2+fast15"]

def get(key, label, v):
    src = r4 if v == "P8/5%" else r5
    return src[key]["res"][f"{label}+{v}"]

rows = {"sw": [], "long": []}
for key, o in r4.items():
    part = o["part"]
    if part == "sw":
        b = r3[key]["res"]; ar, prod, ct0 = b["Target-only"], b["OPT(8,0.4)+static switch b>=7"], b["ClearTree core"]
        old = b["ClearTree core+adaptive switch"]
    else:
        b = o["res"]; ar, prod, ct0 = b["Target-only"], b["production: OPT(8,0.4)+static switch b>=3"], b["ClearTree core"]
        old = None
    rows[part].append((key, ar, prod, ct0, old, {v: (get(key, "ClearTree core", v), get(key, "best static", v)) for v in V}))

for part in ("sw", "long"):
    print(f"== part {part}: CT+switch tok/s (vs best static+same switch / vs production / vs max(Target-only, CT no switch)) spec share")
    agg = {v: dict(bs=[], prod=[], best=[]) for v in V}
    for key, ar, prod, ct0, old, d in sorted(rows[part], key=lambda r: (r[0][0], r[0][1])):
        line = [f"{key[0]:34s} s{key[1]} AR={ar:4.0f} prod={prod:4.0f} CT={ct0:4.0f}"]
        for v in V:
            (ct, sh, _), (bs, _, _) = d[v]
            agg[v]["bs"].append(pct(ct, bs)); agg[v]["prod"].append(pct(ct, prod)); agg[v]["best"].append(pct(ct, max(ar, ct0)))
            line.append(f"{v}: {ct:4.0f} ({pct(ct, bs):+5.1f}/{pct(ct, prod):+5.1f}/{pct(ct, max(ar, ct0)):+5.1f}) {sh:.0%}")
        print(" | ".join(line))
    for v in V:
        f = lambda x: f"{st.mean(x):+5.1f} ({min(x):+5.1f}..{max(x):+5.1f})"
        a = agg[v]
        print(f"  {v:12s} vs best static+same switch {f(a['bs'])}  vs production {f(a['prod'])}  vs max(Target-only, CT no switch) {f(a['best'])}")
    print()
