"""Summarize round 31 (run31.py). usage: python analyze31.py
All comparisons are paired, in % of the rate of "cur" (the current design, v11 + D65 + D67 + D68 + D69) on the same
run. Part B ran only where it can differ from cur; the other round-28 world runs are identical by construction and
count as 0 in the "incl. identical" mean (45 runs)."""
import json, os
from collections import defaultdict


def load(tag):
    out = []
    for f in (f"results/r31{tag}.0.jsonl", f"results/r31{tag}.1.jsonl"):
        if os.path.exists(f):
            out += [json.loads(l) for l in open(f)]
    return out


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def lab(r):
    return f"long{r['req_mult']} s{r['seed']}" if r["req_mult"] >= 10 else f"short s{r['seed']}"


def pct(x, base):
    return 100 * (x["rate"] / base["rate"] - 1)


def evs(x):
    return [(e[1], e[2][0][0]) for e in x["ev_log"]]


def sws(x, n=4):
    return [(s[0], s[2], s[3]) for s in x["switch_log"][:n]]


def summary(g, keys, pols):
    for k in keys:
        parts = [f"{p} {mean(g[(k, p)]):+.2f}% ({min(g[(k, p)]):+.2f}, {max(g[(k, p)]):+.2f})" for p in pols if g[(k, p)]]
        if parts:
            print(f"   {k:14s} n={len(g[(k, pols[0])])}: " + "; ".join(parts))
    print()


key = lambda r: (r["world"], r["kind"], r["b"], r["seed"], r["req_mult"])
Z = {key(r): r["res"]["cur"] for r in load("z")}

# ---------------- A ----------------
A = load("a")
if A:
    print("== Round 31 A: cooldown with two cost changes close together (mid, short prefixes, b = 8, short runs). % vs cur")
    g = defaultdict(list)
    for r in sorted(A, key=lambda r: (r["world"], r["seed"])):
        c = r["res"]["cur"]
        print(f"{r['world']:9s} s{r['seed']} rounds {c['spec_rounds']}")
        for p, x in r["res"].items():
            if p != "cur":
                g[(r["world"], p)].append(pct(x, c)); g[("all", p)].append(pct(x, c))
            print(f"    {p:5s} {pct(x, c):+6.2f}% | events {evs(x)} suppressed {x['suppressed']} pending fired {x['pend']} | "
                  f"switches {sws(x)} | checks {x['offsets']}/{x['checks']} probes {x['probes']}")
    print("  means (min, max):")
    summary(g, ["v+l x3", "v+l x10", "l+l x10", "3chg x10", "all"], ["pend", "psrc", "nocd"])

# ---------------- T ----------------
Tr = load("t")
if Tr:
    print("== Round 31 T: probe cap, rtt alt starting late on 20x runs. bucket vs cur (cumulative cap)")
    for r in sorted(Tr, key=lambda r: (r["kind"], r["seed"])):
        c, x = r["res"]["cur"], r["res"]["bucket"]
        print(f"{r['kind']:4s} b={r['b']} s{r['seed']} rounds {c['spec_rounds']} | bucket vs cur {pct(x, c):+6.2f}% | probes {c['probes']} -> {x['probes']} "
              f"| rounds a probe waited {c['probe_wait']} -> {x['probe_wait']} | checks (common/all) {c['offsets']}/{c['checks']} -> {x['offsets']}/{x['checks']} "
              f"| events {len(c['ev_log'])} -> {len(x['ev_log'])}")
    print()

# ---------------- Z / B ----------------
if Z:
    print("== Round 31 Z: width switches of cur on the 45 round-28 world runs: gain, SE, n, gain / SE")
    rows = []
    for k, x in sorted(Z.items()):
        for s in x["sw_se"]:
            rows.append((k, s))
    zs = [s[4] / s[5] if s[5] > 0 else float("inf") for _, s in rows]
    for (k, s), z in zip(rows, zs):
        if z <= 3:
            print(f"   {k[0]:9s} {k[1]:4s} b={k[2]} s{k[3]} x{k[4]} | round {s[0]} b {s[1]} {s[2]}->{s[3]} gain {100 * s[4]:.1f}% SE {100 * s[5]:.1f}% n {s[6]} z {z:.1f}")
    print(f"   switches logged {len(rows)}; gain/SE <= 2: {sum(z <= 2 for z in zs)}, <= 3: {sum(z <= 3 for z in zs)}; "
          f"median SE {100 * sorted(s[5] for _, s in rows)[len(rows) // 2]:.2f}%; runs with a switch {len({k for k, _ in rows})}")
    by = defaultdict(list)
    for _, s in rows:
        by["wider" if s[3] > s[2] else "narrower"].append(s)
    for d, v in by.items():
        print(f"   {d}: {len(v)} switches, median n {sorted(x[6] for x in v)[len(v) // 2]}, median SE {100 * sorted(x[5] for x in v)[len(v) // 2]:.2f}%, "
              f"gain/SE <= 2: {sum(x[4] <= 2 * x[5] for x in v)}")
    print()
B = load("b")
if B:
    print("== Round 31 B: se2 (switch needs gain > 2 SE) vs cur, round-28 world runs where cur had a switch with gain <= 2 SE")
    ds = []
    for r in sorted(B, key=lambda r: (r["world"], r["kind"], r["b"], r["req_mult"], r["seed"])):
        c, x = Z[key(r)], r["res"]["se2"]
        ds.append(pct(x, c))
        print(f"{lab(r):9s} {r['world']:9s} {r['kind']:4s} b={r['b']} | se2 vs cur {pct(x, c):+6.2f}% | blocked {x['se_block']} | "
              f"switches cur {[(s[0], s[2], s[3], round(s[4] / max(s[5], 1e-9), 1)) for s in c['sw_se'][:5]]} se2 {sws(x, 5)}")
    print(f"  mean {mean(ds):+.2f}% ({min(ds):+.2f}, {max(ds):+.2f}) n={len(ds)}; incl. identical runs (0): {sum(ds) / 45:+.3f}% over 45 runs")
    print()

# ---------------- C ----------------
C = load("c")
if C:
    print("== Round 31 C: K-check offsets with a changing b (target concurrency 8 / 4 every 15 s). % vs cur")
    g = defaultdict(list)
    for r in sorted(C, key=lambda r: (r["world"], r["kind"], r["req_mult"], r["seed"])):
        c = r["res"]["cur"]
        w = r["world"] if r["world"] != "rtt alt" else f"rtt alt {r['kind']}"
        print(f"{w:13s} {lab(r)} rounds {c['spec_rounds']}")
        for p, x in r["res"].items():
            if p != "cur":
                g[(w, p)].append(pct(x, c)); g[("all balt" if r["world"] != "rtt alt" else "all rtt alt", p)].append(pct(x, c))
            print(f"    {p:12s} {pct(x, c):+6.2f}% | checks common/done/started {x['offsets']}/{len(x['chk_log'])}/{x['checks']} abandoned {x.get('chk_abort', 0)} "
                  f"held {x['chk_supp']} | probes {x['probes']} waited {x['probe_wait']} | events {evs(x)} | switches {sws(x)}")
            if p == "cur":
                print(f"         checks {[(k[1], k[2], k[3], k[4]) for k in x['chk_log'][:8]]}")
    print("  means (min, max):")
    summary(g, ["balt cb", "balt rtt", "all balt", "rtt alt mid", "rtt alt high", "all rtt alt"], ["boff", "boff+int", "cto", "boff+int+cto"])

# ---------------- D ----------------
D = load("d")
if D:
    print("== Round 31 D: timing spikes (1% stalled rounds, 0.5% single spikes). rob vs cur")
    g = defaultdict(list)
    for r in sorted(D, key=lambda r: (not r["world"].startswith("tail"), r["world"], r["req_mult"], r["seed"])):
        x = r["res"]["rob"]
        c = r["res"].get("cur") or Z.get(key(r))
        if c is None:
            continue
        grp = "tail" if r["world"].startswith("tail") else "no spikes"
        g[(grp, "rob")].append(pct(x, c)); g[(r["world"], "rob")].append(pct(x, c))
        print(f"{r['world']:13s} {lab(r):9s} rounds {c['spec_rounds']} | rob vs cur {pct(x, c):+6.2f}% | events cur {len(c['ev_log'])} {evs(c)[:6]} rob {evs(x)[:6]} "
              f"| suppressed {c['suppressed']} -> {x['suppressed']} | checks {c['offsets']}/{c['checks']} -> {x['offsets']}/{x['checks']} | probes {c['probes']} -> {x['probes']} "
              f"| skipped {x['skip']} rescreened {x['rescreen']} | switches cur {sws(c)} rob {sws(x)}")
    print("  means (min, max):")
    summary(g, ["tail base", "tail high", "tail het", "tail chg8 x10", "tail", "base", "chg8 x10", "no spikes"], ["rob"])
