"""Round 39: the three tail fixes together, plus explore decay.
usage (conda sglang env): python run39.py <proc 0|1> <part>

  eval    seeds 17, 18, 19 short (3x) and seed 17 long (10x). 15 worlds, 60 runs.
          scope, b3, b4. r39e.
  seed20  the 15 worlds, seed 20, short (3x). Same three arms. r39s.
          Pre-registered only when neither arm is adopted and at least one is inconclusive.
  guard   the 28 guard worlds of round 36. Reads results/r39_adopt.json. r39g.
check() runs first. It does not write r39*.
"""
import json, os, sys, time
import numpy as np
import ext19 as Z
import ext29 as E22
import ext30 as E
import run13 as U
import run25 as R25
import run26 as R26
import run27 as R27
import run32 as R

Z.WL.setdefault("high", U.WL["high"])
Z.WL.setdefault("low", U.WL["low"])

BASE = dict(cooldown=False, rob=True, rob_scope=True)
N0 = 6.0


def arms():
    scope = ("scope", dict(BASE, klog=True))
    b3 = ("b3", dict(BASE, klog=True, win200=200, se_z=2.0, vprior=N0))
    b4 = ("b4", dict(BASE, klog=True, win200=200, se_z=2.0, vprior=N0, decay=True))
    return [scope, b3, b4]


def s_jobs(runs):
    out = [("S", (n, k, m, b, ph), s, mu, None) for (s, mu) in runs for (n, k, m, b, ph) in R25.WORLDS]
    out += [("S", (n, k, m, b, None), s, mu, c) for (s, mu) in runs for (n, k, m, b, c) in R26.CHANGE + R27.NEWW]
    return out


EVAL_RUNS = ((17, 3), (18, 3), (19, 3), (17, 10))


def _one(cls, part, name, kind, mix, b, phase, seed, mult, cm, kw):
    saved = R.P.BatchAccount19
    R.P.BatchAccount19 = cls
    try:
        out = R.one(part, name, kind, mix, b, phase, seed, mult, cm, kw)
    finally:
        R.P.BatchAccount19 = saved
    p = getattr(cls, "current", None)
    if p is not None:
        out["switch_log"] = [list(s) for s in p.switch_log]
        out["n_explore"] = int(getattr(p, "n_explore", 0))
        out["n_cprobe"] = int(getattr(p, "n_cprobe", 0))
        if p.k_snap:
            out["k_snap"] = p.k_snap
    return out


def emit(part, name, kind, mix, b, phase, seed, mult, cm, kw):
    return _one(E.BatchAccount23, part, name, kind, mix, b, phase, seed, mult, cm, kw)


def _cell(cls, kw):
    return _one(cls, "S", "bvar", "het", "mixed", 8, None, 9, 3, None, kw)


def _canon(v):
    return json.loads(json.dumps(v))


def _same_record(got, rec, label):
    for key, exp in rec.items():
        assert key in got, (label, key)
        gv, ev = _canon(got[key]), _canon(exp)
        assert gv == ev, (label, key, gv, ev)


def _pairs(n, mean_a, sd_a, time=1.0, ctok=10.0, ctm=1.0):
    """n pairs whose candidate-token sample sd is sd_a. The other three components are constant."""
    xs = np.full(n, mean_a)
    if sd_a:
        half = sd_a * np.sqrt((n - 1) / 2.0)
        xs[0] += half
        xs[1] -= half
    st = [0.0, 0.0, 0.0, 0.0, 0, 0.0, np.zeros((4, 4))]
    for x in xs:
        v = np.array([float(x), time, ctok, ctm])
        st[0] += v[0]
        st[1] += v[1]
        st[2] += v[2]
        st[3] += v[3]
        st[4] += 1
        st[6] += np.outer(v, v)
    return st


def _gain(st):
    return (st[0] / st[1]) / (st[2] / st[3]) - 1.0


def _bucket(cur, other, st):
    return dict(cur=cur, follow=False, stats={other: st}, n=st[4])


def check():
    rec = None
    for path in ("results/r34s.0.jsonl", "results/r34s.1.jsonl"):
        for line in open(path):
            r = json.loads(line)
            if (r["world"], r["kind"], r["b"], r["seed"], r["req_mult"]) == ("bvar", "het", 8, 9, 3):
                rec = r["res"]["scope"]
    assert rec is not None, "r34s cell missing"

    # A2, before the long cells: 10 pairs, gain 3.07% with SE 5% stays; gain 6% with SE 1% switches.
    gate = E.BatchAccount23(se_z=2.0, win200=200)
    held = _pairs(10, 10.307, 0.05 * np.sqrt(10) / 0.1)
    g, se = _gain(held), gate._gain_se(held, _gain(held))
    assert abs(g - 0.0307) < 1e-9 and abs(se - 0.05) < 1e-9, (g, se)
    s = _bucket(4, 8, held)
    gate.ctx14[4] = s
    gate._rule3_decide(s, 4, 4)
    assert gate.n_switch == 0 and s["cur"] == 4, (gate.n_switch, s["cur"])
    assert gate.n_se_block == 1, gate.n_se_block
    moved = _pairs(10, 10.6, 0.01 * np.sqrt(10) / 0.1)
    g, se = _gain(moved), gate._gain_se(moved, _gain(moved))
    assert abs(g - 0.06) < 1e-9 and abs(se - 0.01) < 1e-9, (g, se)
    s = _bucket(4, 8, moved)
    gate.ctx14[4] = s
    gate._rule3_decide(s, 4, 4)
    assert gate.n_switch == 1 and s["cur"] == 8, (gate.n_switch, s["cur"], gate.switch_log)

    # A4: 40 negative pairs drop the rate to 1%; a switch, then a layer event, each restore 5%.
    dec = E.BatchAccount23(decay=True, se_z=2.0, win200=200)
    neg = _pairs(40, 9.0, 0.0)
    assert _gain(neg) < 0, _gain(neg)
    s = _bucket(4, 8, neg)
    dec.ctx14[4] = s
    assert dec._explore_p(s) == 0.01, dec._explore_p(s)
    dec._switch_to(s, 4, 4, 2, -0.1, 0.0, 40)
    assert dec._explore_p(s) == 0.05, dec._explore_p(s)
    s["stats"][4] = _pairs(40, 9.0, 0.0)
    assert s["cur"] == 2 and dec._explore_p(s) == 0.01
    dec.world = type("W", (), {"n_T": 0})()
    dec.rounds = 0
    dec._layer_event()
    assert s["stats"] == {} and dec._explore_p(s) == 0.05, (s["stats"], dec._explore_p(s))
    print("check units ok", file=sys.stderr, flush=True)

    print("check cell scope", file=sys.stderr, flush=True)
    got = _cell(E.BatchAccount23, dict(BASE))
    _same_record(got, rec, "scope")

    print("check cell b3 vs ext29", file=sys.stderr, flush=True)
    kw = dict(BASE, win200=200, vprior=N0)
    a = _cell(E.BatchAccount23, kw)
    b = _cell(E22.BatchAccount22, kw)
    _same_record(a, b, "b3")
    _same_record(b, a, "ext29")


def eval_jobs():
    jobs = [(j, arms(), None) for j in s_jobs(EVAL_RUNS)]
    assert len(jobs) == 60, len(jobs)
    return jobs


def seed20_jobs():
    seen = {}
    for (_pt, key, _s, _m, cm) in R.s_jobs():
        if key not in seen:
            seen[key] = cm
    worlds = [("S", key, 20, 3, cm) for key, cm in seen.items()]
    assert len(worlds) == 15, len(worlds)
    return [(j, arms(), None) for j in worlds]


def guard_jobs():
    adopted = json.load(open("results/r39_adopt.json"))["arm"]
    table = {lab: kw for lab, kw in arms()}
    assert adopted in table and adopted != "scope", adopted
    pols = [table["scope"], table[adopted]]
    out = [(j, pols, None) for j in R.rtt_jobs(["rtt jump", "rtt alt"])]
    tw = [("tail base", "mid", "short", 4, None), ("tail high", "high", "short", 8, None),
          ("tail het", "het", "mixed", 4, None), ("tail chg8 x10", "mid", "short", 8, 10.0)]
    out += [(("D", (n, k, m, b, None), s, 3, c), pols, None) for s in (8, 9) for (n, k, m, b, c) in tw]
    out += [(("D", (n, k, m, b, None), 8, 10, c), pols, None) for (n, k, m, b, c) in tw[:2]]
    out += [(("L", ("lalt x3", kd, "short", 8, None), s, mu, None), pols, None)
            for (s, mu) in R.RUNS for kd in ("mid", "high")]
    assert len(out) == 28, len(out)
    return out


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")
    check()
    print("check ok", file=sys.stderr, flush=True)
    proc = int(sys.argv[1])
    part = sys.argv[2]
    jobs = {"eval": eval_jobs, "seed20": seed20_jobs, "guard": guard_jobs}[part]()
    tag = {"eval": "r39e", "seed20": "r39s", "guard": "r39g"}[part]
    for i, ((pt, (name, kind, mix, b, phase), seed, mult, cm), pols, _scale) in enumerate(jobs):
        if i % 2 != proc:
            continue
        print(f"{tag} start {i} {name} b={b} seed={seed} x{mult}", file=sys.stderr, flush=True)
        t0 = time.time()
        res = {}
        for lab, kw in pols:
            res[lab] = emit(pt, name, kind, mix, b, phase, seed, mult, cm, kw)
        sec = round(time.time() - t0, 1)
        print(json.dumps(dict(part=pt, run=tag, world=name, kind=kind, mix=mix, b=b, phase=phase,
                              seed=seed, req_mult=mult, cmult=cm, res=res, sec=sec,
                              python=sys.version.split()[0])), flush=True)
        print(f"{tag} done {i} {name} b={b} seed={seed} x{mult} {sec}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
