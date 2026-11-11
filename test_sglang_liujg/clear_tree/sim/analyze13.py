"""Summarise round 13 (results/r13.*.jsonl). Run from this directory."""
import glob, json, statistics as st

pct = lambda a, b: 100.0 * (a / b - 1)
rows = [json.loads(l) for f in sorted(glob.glob("results/r13.*.jsonl")) for l in open(f) if l.strip()]
groups = {}
for o in rows:
    groups.setdefault((o["verify_mult"], o["kind"], o["b"]), []).append(o)
ONLINE = ["table", "online K", "online W", "online W+K"]
KNOWN, UNSEEN = ("low", "mid", "high"), ("midlow", "midhigh", "flat", "flatlow")


def mean_rate(os_, name):
    return st.mean(o["res"][name]["rate"] for o in os_)


print("== % vs the best fixed (W, K) for the scenario (hindsight over the same seeds); [best fixed]; mean over seeds 8-10")
agg = {}
for key in sorted(groups, key=lambda k: (k[0], k[1] not in KNOWN, k[1], k[2])):
    os_ = groups[key]
    fixed = {n: mean_rate(os_, n) for n in os_[0]["res"] if n.startswith("fixed")}
    bf = max(fixed, key=fixed.get)
    grp = "verify x10" if key[0] != 1.0 else ("known" if key[1] in KNOWN else "unseen")
    cells = []
    for n in ONLINE:
        v = pct(mean_rate(os_, n), fixed[bf])
        agg.setdefault((grp, n), []).append(v)
        cells.append(f"{n}: {v:+5.1f}%")
    use = {}
    for o in os_:
        for k, c in o["res"]["online W+K"]["wk"].items():
            use["W" + k.split(",")[0]] = use.get("W" + k.split(",")[0], 0) + c
    kuse = {}
    for o in os_:
        for k, c in o["res"]["online W+K"]["k"].items():
            kuse["K" + k] = kuse.get("K" + k, 0) + c
    tw, tk = sum(use.values()), sum(kuse.values())
    print(f"{grp:10s} {key[1]:7s} b={key[2]} | best fixed {bf[6:]:5s} {fixed[bf]:6.1f} | " + " | ".join(cells)
          + " | W+K uses " + ",".join(f"{k}:{v / tw:.0%}" for k, v in sorted(use.items())) + " " + ",".join(f"{k}:{v / tk:.0%}" for k, v in sorted(kuse.items(), key=lambda x: int(x[0][1:]))))
print("\n  group means (min) vs best fixed (W, K):")
for grp in ("known", "unseen", "verify x10"):
    print(f"   {grp:10s}: " + "; ".join(f"{n} {st.mean(agg[(grp, n)]):+.2f}% ({min(agg[(grp, n)]):+.1f})" for n in ONLINE if (grp, n) in agg))
print("\n  online vs table (mean of per-scenario differences, percentage points):")
for grp in ("known", "unseen", "verify x10"):
    print(f"   {grp:10s}: " + "; ".join(f"{n} {st.mean(a - b for a, b in zip(agg[(grp, n)], agg[(grp, 'table')])):+.2f}" for n in ONLINE[1:]))
