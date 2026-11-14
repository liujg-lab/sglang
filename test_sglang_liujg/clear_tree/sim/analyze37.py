"""Summarize round 37 (run37.py). usage: python analyze37.py [k|bias|x|eval|seed13|guard]
Paired percents are against scope on the same run, except x, which uses r34s.
The printed verdict applies the lines in results.md 7.40. It does not write r37_adopt.json.
"""
import json, os, sys
from collections import defaultdict

STATIC = {("base", "mid", 4), ("base", "mid", 8), ("high", "high", 4), ("high", "high", 8),
          ("het", "het", 4), ("het", "het", 8)}
BEST = {("base", "mid", 4): 4, ("base", "mid", 8): 4, ("high", "high", 4): 2, ("high", "high", 8): 2,
        ("het", "het", 4): 4, ("het", "het", 8): 2}
CHANGING = {"phased", "bvar", "chg x0.3", "chg x3", "chg8 x0.3", "chg8 x3", "chg8 x10"}
POLS = ("scope", "win200", "wide30", "both")


def load(name):
    out = []
    for f in (f"results/{name}.0.jsonl", f"results/{name}.1.jsonl"):
        if os.path.exists(f):
            out += [json.loads(l) for l in open(f) if l.strip()]
    return out


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def pct(a, b):
    return 100.0 * (a["rate"] / b["rate"] - 1.0)


def key(r):
    return (r["world"], r["kind"], r["b"], r["seed"], r["req_mult"])


def wkey(r):
    return (r["world"], r["kind"], r["b"])


def went_wide(x):
    return any(s[1] >= 2 and s[3] == 8 for s in x["switch_log"])


def first_to2(x):
    for s in x["switch_log"]:
        if s[1] >= 2 and s[3] == 2:
            return s[0]
    return None


def left_two(x):
    seen = False
    for s in x["switch_log"]:
        if s[1] < 2:
            continue
        if s[3] == 2:
            seen = True
        elif seen and s[2] == 2:
            return True
    return False


def k_report():
    rows = load("r37k")
    print(f"k rows {len(rows)} (expect 2)")
    for r in rows:
        print(f"seed {r['seed']}")
        counts = {}
        for name, x in r["res"].items():
            k32 = x["k"].get("32", 0)
            counts[name] = k32
            print(f"  {name:14} K {x['k']}  K32 {k32}  snaps {x.get('k_snap', [])[:4]} ... final rounds {x['spec_rounds']}")
        kf, kr, of, ore = (counts["known-fixed"], counts["known-replay"],
                           counts["online-fixed"], counts["online-replay"])
        learn = of >= 0.5 * ore
        width = of <= 2 * kf and ore > 2 * kf
        print(f"  startup-learning {learn}  from-W {width}  (online-fixed {of}, online-replay {ore}, known-fixed {kf})")


def _ratio(tok, mt):
    return tok / mt if mt else float("nan")


def bias_report():
    rows = load("r37b")
    print(f"bias rows {len(rows)} (expect 6)")
    act_tok = act_mt = sh_tok = sh_mt = 0.0
    gaps = defaultdict(list)
    for r in rows:
        x = r["res"]["scope"]
        print(f"{r['world']:8} {r['kind']:5} b={r['b']} rate {x['rate']:.4f}")
        for row in x.get("bias", []):
            if row["W"] == row["Wn"]:
                act_tok += row["rep_tok"]
                act_mt += row["rep_mt"]
                sh_tok += row["sh_tok"]
                sh_mt += row["sh_mt"]
            else:
                cur = None
                for other in x["bias"]:
                    if other["b"] == row["b"] and other["W"] == row["W"] and other["Wn"] == row["W"]:
                        cur = other
                if cur is None or cur["rep_mt"] <= 0 or row["rep_mt"] <= 0 or row["sh_mt"] <= 0:
                    continue
                g_rep = _ratio(row["rep_tok"], row["rep_mt"]) / _ratio(cur["rep_tok"], cur["rep_mt"]) - 1
                g_sh = _ratio(row["sh_tok"], row["sh_mt"]) / _ratio(cur["rep_tok"], cur["rep_mt"]) - 1
                gaps[(row["b"], row["W"], row["Wn"])].append((g_rep - g_sh) * 100.0)
    tok_d = sh_tok / act_tok - 1 if act_tok else float("nan")
    mt_d = sh_mt / act_mt - 1 if act_mt else float("nan")
    bad = abs(tok_d) > 0.01 or abs(mt_d) > 0.01
    print(f"same-width shadow vs served: tok {tok_d * 100:+.3f}%  mt {mt_d * 100:+.3f}%  unusable {bad}")
    if bad:
        print("bias diagnostic and the round-36 oracle -0.12% are unusable")
        return
    print("replay minus shadow throughput ratio, by bucket (percentage points):")
    for k in sorted(gaps):
        xs = gaps[k]
        print(f"  b={k[0]} W={k[1]} -> {k[2]}  {mean(xs):+.3f} pp  ({min(xs):+.3f}, {max(xs):+.3f}) n={len(xs)}"
              f"  within ±1 {abs(mean(xs)) <= 1.0}")


def x_report():
    rows = load("r37x")
    base = {}
    for r in load("r34s"):
        base[key(r)] = r["res"]["scope"]
    print(f"x rows {len(rows)} (expect 45)")
    groups = defaultdict(list)
    for r in rows:
        b = base.get(key(r))
        x = r["res"]["nox"]
        if b is None:
            print("missing", key(r))
            continue
        d = pct(x, b)
        groups["all"].append(d)
        groups["static" if wkey(r) in STATIC else "other"].append(d)
        groups["changing" if r["world"] in CHANGING else "held"].append(d)
        if x["n_explore"] or x["n_cprobe"]:
            print("nonzero probe", key(r), x["n_explore"], x["n_cprobe"])
    for name in ("all", "static", "changing", "held", "other"):
        xs = groups[name]
        if xs:
            print(f"  {name}: {mean(xs):+.3f}% ({min(xs):+.2f}, {max(xs):+.2f}) n={len(xs)}")


def _collect(rows, extra=None):
    """List of (wkey, pol, pct vs in-row scope, row, pol record)."""
    out = []
    for r in rows:
        scope = r["res"]["scope"]
        for p in POLS:
            if p not in r["res"]:
                continue
            out.append((wkey(r), p, pct(r["res"][p], scope) if p != "scope" else 0.0, r, r["res"][p], scope))
    if extra:
        out += _collect(extra)
    return out


def _world_means(items, pol):
    worlds = sorted({w for w, p, *_ in items if p == pol})
    means = {}
    for w in worlds:
        xs = [d for ww, p, d, *_ in items if ww == w and p == pol]
        means[w] = mean(xs)
    return means


def _phased(rows):
    bad = []
    detail = []
    for r in rows:
        if not (r["world"] == "phased" and r["b"] == 8):
            continue
        for p in ("win200", "both"):
            if p not in r["res"]:
                continue
            x, sc = r["res"][p], r["res"]["scope"]
            final = x["final"].get("8")
            back = left_two(x)
            at, sat = first_to2(x), first_to2(sc)
            late = sat is None or at is None or at > sat
            ok = final == 2 and not back and not late
            detail.append((key(r), p, final, at, sat, back, ok))
            if p == "win200" and not ok:
                bad.append(key(r))
    return bad, detail


def _static_mismatch(rows):
    n = {"scope": 0, "win200": 0, "wide30": 0, "both": 0}
    for r in rows:
        if wkey(r) not in STATIC:
            continue
        best = BEST[wkey(r)]
        for p in POLS:
            if p not in r["res"]:
                continue
            got = r["res"][p]["final"].get(str(r["b"]))
            if got != best:
                n[p] += 1
    return n


def _wide_runs(rows):
    n = {p: 0 for p in POLS}
    for r in rows:
        for p in POLS:
            if p in r["res"] and went_wide(r["res"][p]):
                n[p] += 1
    return n


def _judge(rows, label):
    items = _collect(rows)
    print(f"{label} rows {len(rows)}")
    for p in ("win200", "wide30", "both"):
        xs = [d for _w, pp, d, *_ in items if pp == p]
        print(f"  {p}: {mean(xs):+.4f}% ({min(xs):+.2f}, {max(xs):+.2f}) n={len(xs)}")
    print("per-world means:")
    for p in ("win200", "wide30", "both"):
        for w, m in _world_means(items, p).items():
            if m < -0.30 or p == "win200":
                print(f"  {w} {p} {m:+.3f}%")
    bad, detail = _phased(rows)
    print("phased b=8 (final, at, scope_at, left_two, ok):")
    for d in detail:
        print(" ", d)
    sm = _static_mismatch(rows)
    print("static finals differing from best fixed W", sm)
    wide = _wide_runs(rows)
    print("runs that reached W=8 at b>=2", wide)
    over = {p: sum(1 for _w, pp, d, *_ in items if pp == p and False) for p in POLS}
    worlds = {}
    for p in ("win200", "wide30", "both"):
        wm = _world_means(items, p)
        worlds[p] = [w for w, m in wm.items() if m < -0.30]
        xs = [d for _w, pp, d, *_ in items if pp == p]
        m = mean(xs)
        print(f"  {p} worlds below -0.30: {worlds[p]}")
    # mechanical lines
    def arm_mean(p):
        return mean([d for _w, pp, d, *_ in items if pp == p])

    wmean, w30, both = arm_mean("win200"), arm_mean("wide30"), arm_mean("both")
    win_side = (not bad) and sm["win200"] <= sm["scope"] and not worlds["win200"]
    if wmean <= 0.10:
        win = "reject"
    elif wmean >= 0.30 and win_side:
        win = "adopt"
    else:
        win = "inconclusive"
    if w30 < -0.05 or worlds["wide30"]:
        wide_v = "reject"
    elif wide["scope"] == 0:
        wide_v = "inconclusive"
    elif w30 >= -0.05 and not worlds["wide30"] and wide["wide30"] < wide["scope"]:
        wide_v = "adopt"
    else:
        wide_v = "reject"
    joint = "no"
    if win == "adopt" and wide_v == "adopt":
        joint = "adopt" if both >= wmean - 0.10 and not worlds["both"] else "win200-only"
    elif win == "adopt":
        joint = "win200-only"
    print(f"VERDICT {label}: win200 {win}  wide30 {wide_v}  joint {joint}")
    print("unused", over)
    return win, wide_v, joint


def eval_():
    _judge(load("r37e"), "eval45")


def seed13():
    ev, extra = load("r37e"), load("r37s")
    print(f"seed13 rows {len(extra)} (expect 15)")
    for r in extra:
        sc = r["res"]["scope"]
        bits = []
        for p in ("win200", "wide30", "both"):
            bits.append(f"{p} {pct(r['res'][p], sc):+.2f}%")
        print(f"  {r['world']:12} {r['kind']:5} b={r['b']} " + " ".join(bits))
    _judge(ev + extra, "combined60")


def guard():
    rows = load("r37g")
    print(f"guard rows {len(rows)} (expect 28)")
    pols = [p for p in POLS if p != "scope" and any(p in r["res"] for r in rows)]
    groups = defaultdict(list)
    for r in rows:
        sc = r["res"]["scope"]
        line = f"{r['world']:14} {r['kind']:5} b={r['b']} s{r['seed']} x{r['req_mult']}"
        bits = []
        for p in pols:
            d = pct(r["res"][p], sc)
            groups[("all", p)].append(d)
            groups[(r["world"], p)].append(d)
            bits.append(f"{p} {d:+.2f}%")
        print(line, "|", " ".join(bits))
    for name in ("all",) + tuple(sorted({r["world"] for r in rows})):
        parts = []
        for p in pols:
            xs = groups.get((name, p), [])
            if xs:
                parts.append(f"{p} {mean(xs):+.2f}% n={len(xs)}")
        if parts:
            print(f"  {name}: " + "; ".join(parts))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "eval"
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")
    {"k": k_report, "bias": bias_report, "x": x_report, "eval": eval_,
     "seed13": seed13, "guard": guard}[which]()
