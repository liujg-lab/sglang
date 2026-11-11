"""Summarise round 4 (results/r4p*.jsonl) against round 3 (results/r3b*.jsonl). Run from this directory."""
import json, sys, statistics as st
D = R3 = "results"
r3 = {}
for f in ("r3b0.jsonl", "r3b1.jsonl"):
    for l in open(f"{R3}/{f}"):
        o = json.loads(l); r3[(o["scen"], o["seed"])] = o["res"]
r4, lg = {}, {}
for f in ("r4p0.jsonl", "r4p1.jsonl"):
    for l in open(f"{D}/{f}"):
        o = json.loads(l)
        (r4 if o["part"] == "sw" else lg)[(o["scen"], o["seed"])] = (o["res"], o["sec"])
V = ["P15/5%", "P8/5%", "P8/10%", "P8/5%/W40"]
order = [k for k in r3]  # file order
order = sorted(order, key=lambda k: (list(dict.fromkeys(s for s, _ in r3)).index(k[0]), k[1]))
pct = lambda a, b: 100.0 * (a / b - 1)
print("== part sw: CT core + switch; columns per variant: vs best static+same switch / vs production / vs CT+old switch / spec share")
summ = {v: dict(bs=[], prod=[], old=[], ar=[]) for v in V + ["old"]}
for key in order:
    if key not in r4:
        continue
    a, sec = r4[key]; b = r3[key]
    prod = b["OPT(8,0.4)+static switch b>=7"]; ar = b["Target-only"]
    old_ct = b["ClearTree core+adaptive switch"]; old_bs = b["best static+adaptive switch"]
    row = [f"{key[0]:34s} s{key[1]} AR={ar:5.0f} prod={prod:5.0f} CT-old={old_ct:5.0f} ({pct(old_ct, old_bs):+5.1f} vs bs-old)"]
    summ["old"]["bs"].append(pct(old_ct, old_bs)); summ["old"]["prod"].append(pct(old_ct, prod)); summ["old"]["ar"].append(pct(old_ct, max(ar, b["ClearTree core"])))
    for v in V:
        ct, sh, pr = a[f"ClearTree core+{v}"]; bs = a[f"best static+{v}"][0]
        summ[v]["bs"].append(pct(ct, bs)); summ[v]["prod"].append(pct(ct, prod)); summ[v]["old"].append(pct(ct, old_ct))
        summ[v]["ar"].append(pct(ct, max(ar, b["ClearTree core"])))
        row.append(f"{v}: {ct:5.0f} {pct(ct, bs):+5.1f}/{pct(ct, prod):+5.1f}/{pct(ct, old_ct):+5.1f} sh={sh:.0%}")
    if "ClearTree core+old switch (rerun)" in a:
        row.append(f"[rerun old={a['ClearTree core+old switch (rerun)']:.1f} vs r3b {old_ct:.1f}]")
    print(" | ".join(row))
print()
for v in V + ["old"]:
    s = summ[v]
    f = lambda x: f"{st.mean(x):+5.1f} ({min(x):+5.1f}..{max(x):+5.1f})" if x else "-"
    print(f"{v:10s} vs best static+same switch {f(s['bs'])}  vs production {f(s['prod'])}  vs CT+old switch {f(s['old'])}  vs max(AR, CT no switch) {f(s['ar'])}")
print("\n== part long (factor 3, RTT 4)")
for key in sorted(lg, key=lambda k: (k[0], k[1])):
    a, sec = lg[key]
    ar = a["Target-only"]; prod = a["production: OPT(8,0.4)+static switch b>=3"]; bsv, bk = a["best static (hindsight)"]
    row = [f"{key[0]} s{key[1]} AR={ar:.0f} OPT(8,.4)={a['OPT(8,0.4)']:.0f} prod={prod:.0f} best static={bsv:.0f} [{bk}] CT={a['ClearTree core']:.0f} ({pct(a['ClearTree core'], bsv):+.1f}% vs best static)"]
    for v in V:
        ct = a[f"ClearTree core+{v}"][0]; bs = a[f"best static+{v}"][0]
        row.append(f"{v}: CT+sw={ct:.0f} bs+sw={bs:.0f} ({pct(ct, bs):+.1f} / {pct(ct, prod):+.1f} vs prod)")
    print(" | ".join(row), f"[{sec}s]")
