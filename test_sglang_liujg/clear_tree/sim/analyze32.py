"""Summarize round 32 (run32.py). usage: python analyze32.py
Paired, in % of the rate of "cur" (the design as of D69) on the same run; cur comes from r31z (round-28 worlds),
r31a (cooldown worlds), r31c (rtt alt, balt), r31d (tail) or the same r32 row."""
import json, os, sys
from collections import defaultdict


def load(name):
    out = []
    for f in (f"results/{name}.0.jsonl", f"results/{name}.1.jsonl"):
        if os.path.exists(f):
            out += [json.loads(l) for l in open(f)]
    return out


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def lab(r):
    return f"long{r['req_mult']} s{r['seed']}" if r["req_mult"] >= 10 else f"short s{r['seed']}"


def pct(x, base):
    return 100 * (x["rate"] / base["rate"] - 1)


def evs(x, n=6):
    return [(e[1], e[2][0][0]) for e in x["ev_log"][:n]]


def sws(x, n=4):
    return [(s[0], s[2], s[3]) for s in x["switch_log"][:n]]


key = lambda r: (r["world"], r["kind"], r["b"], r["seed"], r["req_mult"])
Z = {key(r): r["res"]["cur"] for r in load("r31z")}
A31 = {key(r): r["res"] for r in load("r31a")}
C31 = {key(r): r["res"] for r in load("r31c")}
D31 = {key(r): r["res"] for r in load("r31d")}


def base_of(r):
    if "cur" in r["res"]:
        return r["res"]["cur"]
    for d in ((Z,) if r["part"] == "S" else (A31, C31, D31)):
        x = d.get(key(r))
        if x is not None:
            return x if "rate" in x else x.get("cur")
    return None


def table(rows, title, pols, groups, extra=lambda x: ""):
    if not rows:
        return
    print(title)
    g = defaultdict(list)
    for r in rows:
        c = base_of(r)
        if c is None:
            print("   (no baseline)", key(r)); continue
        w = f"{r['world']} {r['kind']} b={r['b']}"
        print(f"{w:22s} {lab(r):9s} rounds {c['spec_rounds']} | cur: events {evs(c)} suppressed {c['suppressed']} checks {c['offsets']}/{c['checks']} probes {c['probes']} switches {sws(c)}")
        for p in pols:
            x = r["res"].get(p)
            if x is None:
                continue
            for k in groups(r):
                g[(k, p)].append(pct(x, c))
            print(f"    {p:5s} {pct(x, c):+6.2f}% | events {evs(x)} suppressed {x['suppressed']} checks {x['offsets']}/{x['checks']} probes {x['probes']} "
                  f"switches {sws(x)}{extra(x)}")
    print("  means (min, max):")
    for k in sorted({k for k, _ in g}):
        parts = [f"{p} {mean(g[(k, p)]):+.2f}% ({min(g[(k, p)]):+.2f}, {max(g[(k, p)]):+.2f})" for p in pols if g[(k, p)]]
        print(f"   {k:22s} n={max(len(g[(k, p)]) for p in pols)}: " + "; ".join(parts))
    print()


srt = lambda rows: sorted(rows, key=lambda r: (r["part"], r["world"], r["kind"], r["b"], r["req_mult"], r["seed"]))
P = srt(load("r32p"))
table([r for r in P if r["part"] == "R"], "== Round 32 P: cost-event limiter, RTT worlds of round 29. % vs cur", ["nocd", "ebkt"],
      lambda r: ("all rtt", r["world"], f"{r['world']} {r['kind']}"))
table([r for r in P if r["part"] == "L"], "== Round 32 P: layer cost x3 / x1 alternating every 100 rounds. % vs cur", ["nocd", "ebkt"],
      lambda r: ("all lalt", f"lalt {r['kind']}"))
rows = [r for r in P if r["part"] == "A"]
for r in rows:
    r["res"]["nocd"] = A31[key(r)]["nocd"]
table(rows, "== Round 32 P: cooldown worlds of round 31 A (nocd from r31a). % vs cur", ["nocd", "ebkt"],
      lambda r: ("all 2chg", r["world"]))
table(srt(load("r32s")), "== Round 32 S: round-28 world runs where cur suppressed a trigger. % vs cur", ["nocd", "ebkt"],
      lambda r: ("all S",))
K = srt(load("r32k"))
table([r for r in K if r["part"] == "K"], "== Round 32 K: K = 32 gets cheaper at round 400 (unused K). cpk vs cur", ["cpk"],
      lambda r: ("all k32", f"k32 b={r['b']}"), lambda x: f" | K use {x['k']} cpk {x['cpk']} trig {x['cpk_trig']}")
table([r for r in K if r["part"] == "S"], "== Round 32 K: cost of cpk without a change. cpk vs cur (r31z)", ["cpk"],
      lambda r: ("all static",), lambda x: f" | cpk {x['cpk']} trig {x['cpk_trig']}")

# ---------------- V / V2: the combined candidates on every world ----------------
Vr, V2r = srt(load("r32v")), srt(load("r32v2"))
P32 = {key(r): r["res"] for r in P if "cur" in r["res"]}
K32 = {key(r): r["res"] for r in K if "cur" in r["res"]}
CURV = {(r["part"],) + key(r): r["res"]["cur"] for r in Vr if "cur" in r["res"]}
KNOWN = {key(r): r["res"]["known"] for r in Vr if "known" in r["res"]}
NAME = {"S": "round-28 worlds", "R": "RTT", "VC": "vchg x5", "ST": "step", "A": "changes close together",
        "L": "layer cost alternating", "K": "K=32 gets cheaper", "B": "b alternating", "D": "timing spikes"}


def cur_of(r):
    c = r["res"].get("cur") or CURV.get((r["part"],) + key(r))
    for d in (Z, P32, C31, A31, K32, D31):
        if c is None and r["part"] in ("S", "R", "A", "L", "K", "B", "D"):
            y = d.get(key(r))
            if y is not None and (r["part"] == "S") == (d is Z):
                c = y if "rate" in y else y.get("cur")
    return c


def vsec(rows, pol, title):
    if not rows:
        return
    print(title)
    g, miss = defaultdict(list), 0
    for r in rows:
        x, c = r["res"][pol], cur_of(r)
        if c is None:
            miss += 1
            continue
        d_ = pct(x, c)
        sub = r["world"] if r["part"] != "S" else ("static" if r["cmult"] is None else "layer cost change")
        for k in ("all", NAME[r["part"]], f"{NAME[r['part']]}: {sub}"):
            g[k].append(d_)
        kn = f" known {pct(KNOWN[key(r)], c):+.2f}%" if r["part"] == "K" and key(r) in KNOWN else ""
        print(f"{r['part']:2s} {r['world']:13s} {r['kind']:4s} b={r['b']} {lab(r):9s} | {pol} vs cur {d_:+6.2f}%{kn} | events {len(c['ev_log'])} {evs(c, 4)} -> {len(x['ev_log'])} {evs(x, 4)} "
              f"| checks {c['offsets']}/{c['checks']} -> {x['offsets']}/{x['checks']} (abandoned {x['chk_abort']}) | probes {c['probes']} -> {x['probes']} | skipped {x['skip']} | switches {sws(c, 3)} -> {sws(x, 3)}")
    print(f"  means (min, max), runs better / worse than cur by more than 0.3% (no baseline: {miss}):")
    for k in sorted(g):
        v = g[k]
        print(f"   {k:44s} n={len(v):3d}: {mean(v):+.2f}% ({min(v):+.2f}, {max(v):+.2f}) better {sum(x > 0.3 for x in v)} worse {sum(x < -0.3 for x in v)}")
    print()


vsec(Vr, "v12", "== Round 32 V: v12 (no cooldown, rob, per-K forgetting, K-check timeout, first probes outside the cap) vs cur")
vsec(V2r, "v12c", "== Round 32 V2: v12c (no cooldown, rob with staleness rule and hysteresis, K-check timeout, first probes outside the cap; pooled forgetting) vs cur")
