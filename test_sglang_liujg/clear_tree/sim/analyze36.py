"""Summarize round 36 (run36.py). usage: python analyze36.py [pilot|eval|fixed|guard|seed10|all] [r36b|r36]
The tag defaults to r36b, the rerun after the cost-event fix. r36 reads the first run.
Paired percents are against scope on the same run. For eval and fixed, scope of the 45 runs is r34s.
Fixed W is always r36f. The pilot ranking prints the pre-registered choice; it does not write the param file.
"""
import json, os, sys
from collections import defaultdict

TAG = "r36b"
FILES = {
    "r36b": dict(pilot="r36bp", eval="r36be", fixed="r36f", guard="r36bg", seed10="r36bs"),
    "r36": dict(pilot="r36p", eval="r36e", fixed="r36f", guard="r36g", seed10="r36s"),
}

PARAMS = ("c015h14w8", "c015h10w10", "c015h14w6", "c020h12w6", "c015h16w10", "c020h14w8")
START = "c015h14w8"
STATIC = {("base", "mid", 4), ("base", "mid", 8), ("high", "high", 4), ("high", "high", 8),
          ("het", "het", 4), ("het", "het", 8)}


def load(name):
    out = []
    for f in (f"results/{name}.0.jsonl", f"results/{name}.1.jsonl"):
        if os.path.exists(f):
            out += [json.loads(l) for l in open(f)]
    return out


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def pct(a, b):
    return 100.0 * (a["rate"] / b["rate"] - 1.0)


def key(r):
    return (r["world"], r["kind"], r["b"], r["seed"], r["req_mult"])


def wkey(r):
    return (r["world"], r["kind"], r["b"])


def final_w(x, b):
    return x["final"].get(str(b), x["final"].get(b))


def switches_to(x, b, new):
    return [s for s in x["switch_log"] if s[1] == b and s[3] == new]


def show(rows, pols, base_of, groups):
    g = defaultdict(list)
    sw = defaultdict(int)
    for r in rows:
        base = base_of(r)
        if base is None:
            print("no baseline", key(r))
            continue
        for p in pols:
            x = r["res"].get(p)
            if x is None:
                continue
            for gk in groups(r):
                g[(gk, p)].append(pct(x, base))
            sw[p] += x["switches"]
        line = f"{r['world']:12} {r['kind']:5} b={r['b']} s{r['seed']} x{r['req_mult']}"
        bits = []
        for p in pols:
            x = r["res"].get(p)
            if x is None:
                continue
            bits.append(f"{p} {pct(x, base):+6.2f}% W {x['final']} sw {x['switch_log'][:3]}")
        print(line, "|", " || ".join(bits))
    print("means (min, max):")
    names = []
    for gk, _p in g:
        if gk not in names:
            names.append(gk)
    for gk in names:
        parts = []
        for p in pols:
            xs = g.get((gk, p), [])
            if xs:
                parts.append(f"{p} {mean(xs):+.2f}% ({min(xs):+.2f}, {max(xs):+.2f}) n={len(xs)}")
        print(f"  {gk}: " + "; ".join(parts))
    print("switches", dict(sw))
    return g


def pilot():
    rows = load(FILES[TAG]["pilot"])
    print(f"pilot rows {len(rows)}")
    if len(rows) != 15:
        print("expected 15")
    scores = {}
    for p in PARAMS:
        ds, bad = [], False
        for r in rows:
            if "scope" not in r["res"] or p not in r["res"]:
                continue
            d = r["res"][p]["rate"] / r["res"]["scope"]["rate"] - 1.0
            ds.append(d)
            if d < -0.01:
                bad = True
        scores[p] = (mean(ds) * 100.0, bad, len(ds))
    for p, (m, bad, n) in scores.items():
        print(f"  {p:12} {m:+.3f}% bad={bad} n={n}")
    ok = [p for p, (_m, bad, n) in scores.items() if n and not bad]
    pool = ok or list(scores)
    best = max(pool, key=lambda p: scores[p][0])
    choice = START if abs(scores[best][0] - scores[START][0]) <= 0.05 else best
    if not ok:
        choice = START
        print("every setting has a run below -1%; freeze the start")
    print("freeze", choice, scores[choice])
    show(rows, PARAMS, lambda r: r["res"]["scope"], lambda r: ["all", r["world"]])


def scope34():
    out = {}
    for r in load("r34s"):
        out[key(r)] = r["res"]["scope"]
    return out


def eval_():
    rows = load(FILES[TAG]["eval"])
    base = scope34()
    print(f"eval rows {len(rows)}")
    pols = []
    for r in rows:
        for p in r["res"]:
            if p not in pols:
                pols.append(p)

    def base_of(r):
        return base.get(key(r))

    g = show(rows, pols, base_of, lambda r: ["all", f"{r['world']} {r['kind']} b={r['b']}"])
    # adoption checks, printed, not decided here beyond the numbers
    print("per-world means:")
    worlds = sorted({wkey(r) for r in rows})
    for w in worlds:
        sub = [r for r in rows if wkey(r) == w]
        for p in pols:
            xs = [pct(r["res"][p], base[key(r)]) for r in sub if p in r["res"] and key(r) in base]
            if xs:
                print(f"  {w} {p} {mean(xs):+.2f}% n={len(xs)}")
    print("phased b=8 and chg x0.3 b=4 seed 8:")
    for r in rows:
        if (r["world"] == "phased" and r["b"] == 8) or (
                r["world"] == "chg x0.3" and r["kind"] == "mid" and r["b"] == 4 and r["seed"] == 8):
            bsc = base.get(key(r))
            print(" ", key(r), "scope", None if bsc is None else (bsc["final"], bsc["switch_log"][:4]))
            for p in pols:
                x = r["res"][p]
                print("   ", p, x["final"], x["switch_log"][:4])
    st = {"scope": 0}
    for r in load("r34s"):
        if (r["world"], r["kind"], r["b"]) in STATIC:
            st["scope"] += r["res"]["scope"]["switches"]
    for r in rows:
        if (r["world"], r["kind"], r["b"]) in STATIC:
            for p in pols:
                st[p] = st.get(p, 0) + r["res"][p]["switches"]
    print("static switches", st)


def fixed():
    rows = load(FILES[TAG]["fixed"])
    base = scope34()
    print(f"fixed rows {len(rows)}")
    gaps = []
    for r in rows:
        b = base.get(key(r))
        if b is None:
            continue
        best = max(r["res"], key=lambda p: r["res"][p]["rate"])
        gap = pct(b, r["res"][best])
        gaps.append(gap)
        print(f"{r['world']:12} {r['kind']:5} b={r['b']} s{r['seed']} x{r['req_mult']} "
              f"best {best} scope {gap:+.2f}% vs best")
    print(f"scope vs best fixed: mean {mean(gaps):+.2f}% min {min(gaps):+.2f} max {max(gaps):+.2f} n={len(gaps)}")


def seed10():
    """45 runs (r36e vs r34s) plus seed 10 short (r36s vs its own scope)."""
    ev, extra = load(FILES[TAG]["eval"]), load(FILES[TAG]["seed10"])
    base = scope34()
    print(f"seed10 rows {len(extra)} (expect 15); eval rows {len(ev)}")
    pol = "c015h14w8"
    ratios = []
    for r in ev:
        b = base.get(key(r))
        if b is None or pol not in r["res"]:
            print("missing", key(r))
            continue
        ratios.append((wkey(r), pct(r["res"][pol], b)))
    for r in extra:
        b = r["res"].get("scope")
        if b is None or pol not in r["res"]:
            print("missing seed10", key(r))
            continue
        ratios.append((wkey(r), pct(r["res"][pol], b)))
        print(f"  {r['world']:12} {r['kind']:5} b={r['b']} s10 "
              f"{pct(r['res'][pol], b):+.2f}% W {r['res'][pol]['final']} "
              f"sw {r['res'][pol]['switch_log'][:4]}")
    print(f"combined mean {mean([d for _w, d in ratios]):+.3f}% n={len(ratios)}")
    worlds = sorted({w for w, _d in ratios})
    print("world means:")
    for w in worlds:
        xs = [d for ww, d in ratios if ww == w]
        print(f"  {w} {mean(xs):+.2f}% n={len(xs)}")
    st = {"scope": 0, pol: 0}
    for r in load("r34s"):
        if wkey(r) in STATIC:
            st["scope"] += r["res"]["scope"]["switches"]
    for r in ev:
        if wkey(r) in STATIC and pol in r["res"]:
            st[pol] += r["res"][pol]["switches"]
    for r in extra:
        if wkey(r) in STATIC:
            st["scope"] += r["res"]["scope"]["switches"]
            st[pol] += r["res"][pol]["switches"]
    print("static switches", st)


def guard():
    rows = load(FILES[TAG]["guard"])
    print(f"guard rows {len(rows)}")
    pols = []
    for r in rows:
        for p in r["res"]:
            if p != "scope" and p not in pols:
                pols.append(p)
    show(rows, pols, lambda r: r["res"].get("scope"),
         lambda r: ["all", r["world"], f"{r['world']} {r['kind']} b={r['b']}"])


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if len(sys.argv) > 2:
        TAG = sys.argv[2]
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")
    fn = {"pilot": pilot, "eval": eval_, "fixed": fixed, "guard": guard, "seed10": seed10}
    if which == "all":
        for f in (pilot, eval_, fixed, guard):
            f()
            print()
    else:
        fn[which]()
