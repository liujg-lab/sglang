"""Summarise round 14 (results/r14.*.jsonl). Run from this directory."""
import glob, json, statistics as st

pct = lambda a, b: 100.0 * (a / b - 1)
rows = [json.loads(l) for f in sorted(glob.glob("results/r14.*.jsonl")) for l in open(f) if l.strip()]
groups = {}
for o in rows:
    groups.setdefault((o["world"], o["kind"], o["b"]), []).append(o)
POL = ["table", "online K", "replay"]
WORLDS = ["unseen", "verify x10", "draft x0.5", "draft x2"]
mr = lambda os_, n: st.mean(o["res"][n]["rate"] for o in os_)

print("== % vs the best fixed (W, K) for the scenario (hindsight over the same seeds); mean over seeds 8-10")
agg = {}
for key in sorted(groups, key=lambda k: (WORLDS.index(k[0]), k[1], k[2])):
    os_ = groups[key]
    fixed = {n: mr(os_, n) for n in os_[0]["res"] if n.startswith("fixed")}
    bf = max(fixed, key=fixed.get)
    cells = []
    for n in POL:
        v = pct(mr(os_, n), fixed[bf])
        agg.setdefault((key[0], n), []).append(v)
        cells.append(f"{n}: {v:+5.1f}%")
    ch = {}
    for o in os_:
        for k, c in o["res"]["replay"]["choice"].items():
            ch[k] = ch.get(k, 0) + c
    tot = sum(ch.values())
    ex = sum(o["res"]["replay"]["explore"] for o in os_) / max(1, sum(o["res"]["replay"]["rounds"] for o in os_))
    print(f"{key[0]:10s} {key[1]:7s} b={key[2]} | best fixed {bf[6:]:5s} {fixed[bf]:6.1f} | " + " | ".join(cells)
          + " | replay uses " + ",".join(f"{k}:{v / tot:.0%}" for k, v in sorted(ch.items(), key=lambda x: -x[1])[:3]) + f" | explore {ex:.1%}")
print("\n  group means (min) vs best fixed (W, K):")
for g in WORLDS:
    print(f"   {g:10s}: " + "; ".join(f"{n} {st.mean(agg[(g, n)]):+.2f}% ({min(agg[(g, n)]):+.1f})" for n in POL))
print("  all 30 scenarios: " + "; ".join(f"{n} {st.mean([v for (g, nn), vs in agg.items() if nn == n for v in vs]):+.2f}%" for n in POL))
print("  vs table (percentage points): " + "; ".join(
    f"{g}: K {st.mean(a - b for a, b in zip(agg[(g, 'online K')], agg[(g, 'table')])):+.2f}, replay {st.mean(a - b for a, b in zip(agg[(g, 'replay')], agg[(g, 'table')])):+.2f}" for g in WORLDS))

print("\n== Narrowing audit: accepted tokens of a real tree of the narrower width vs the wide tree cut down (same depth, same K)")
for g in WORLDS:
    a = [o["res"]["replay"]["audit"] for o in rows if o["world"] == g]
    n = sum(x["n"] for x in a)
    if n:
        print(f"   {g:10s}: {n} request-rounds, differ {sum(x['differ'] for x in a) / n:.1%}, mean (real - cut) {sum(x['diff'] for x in a) / n:+.3f} tokens, "
              f"mean |diff| {sum(x['absdiff'] for x in a) / n:.3f}, relative to real accepted {100 * sum(x['diff'] for x in a) / max(1e-9, sum(x['real'] for x in a)):+.2f}%")

B = [json.loads(l) for f in sorted(glob.glob("results/r14b.*.jsonl")) for l in open(f) if l.strip()]
if B:
    rb = {(o["world"], o["kind"], o["b"], o["seed"]): o["res"]["replay measured layers"]["rate"] for o in B}
    print("\n== 14b: replay with measured Draft layer times, % vs best fixed (W, K); same scenarios")
    agg2 = {}
    for key in sorted(groups, key=lambda k: (WORLDS.index(k[0]), k[1], k[2])):
        os_ = groups[key]
        if not all((o["world"], o["kind"], o["b"], o["seed"]) in rb for o in os_):
            continue
        fixed = {n: mr(os_, n) for n in os_[0]["res"] if n.startswith("fixed")}
        bf = max(fixed.values())
        v = pct(st.mean(rb[(o["world"], o["kind"], o["b"], o["seed"])] for o in os_), bf)
        agg2.setdefault(key[0], []).append(v)
        print(f"   {key[0]:10s} {key[1]:7s} b={key[2]}: replay (measured layers) {v:+5.1f}% | replay (learned) {pct(mr(os_, 'replay'), bf):+5.1f}% | online K {pct(mr(os_, 'online K'), bf):+5.1f}% | table {pct(mr(os_, 'table'), bf):+5.1f}%")
    for g in WORLDS:
        if g in agg2:
            print(f"   {g:10s}: replay (measured layers) mean {st.mean(agg2[g]):+.2f}% (min {min(agg2[g]):+.1f}); learned {st.mean(agg[(g, 'replay')]):+.2f}%; online K {st.mean(agg[(g, 'online K')]):+.2f}%; table {st.mean(agg[(g, 'table')]):+.2f}%")
    allv = [v for vs in agg2.values() for v in vs]
    print(f"   all {len(allv)} scenarios: replay (measured layers) {st.mean(allv):+.2f}%")
