"""Round 38: verify-cost startup, plus three diagnostics.
usage (conda sglang env): python run38.py <proc 0|1> <part>

  p      seed 7 short. het b=4, base b=4, high b=8. vprior n0 in {2, 6, 20}. r38p.
  b      seed 7 short, b=1. mid, high, het. scope with biaslog, and known-cost fixed W=2/4/8. r38b.
  w      six static worlds, seed 7 short, explog and biaslog. r38w.
  g      low acceptance, b=2 and b=4, layer cost x1 / x0.3 / x0.1 from round 0,
         known-cost fixed W=2/4/8, seed 7 short. r38g.
  eval   seeds 14 and 15 short, seed 14 long. scope, vnn, vprior, both. r38e.
         vprior reads results/r38_param.json. Written before this part, then not changed.
  seed16 the 15 worlds, seed 16, short. Same four arms. r38s.
         Pre-registered only when neither candidate is adopted and at least one is inconclusive.
  guard  the 28 guard worlds. Reads results/r38_adopt.json. r38g2.
check() runs first.
"""
import json, sys, time
import numpy as np
import ext19 as Z
import ext29 as E
import run13 as U
import run25 as R25
import run26 as R26
import run27 as R27
import run32 as R

Z.WL.setdefault("high", U.WL["high"])
Z.WL.setdefault("low", U.WL["low"])

BASE = dict(cooldown=False, rob=True, rob_scope=True)


def n0_of():
    return int(json.load(open("results/r38_param.json"))["n0"])


def arms(n0):
    scope = ("scope", dict(BASE, klog=True))
    vnn = ("vnn", dict(BASE, klog=True, vnn=True))
    vprior = ("vprior", dict(BASE, klog=True, vprior=n0))
    both = ("both", dict(BASE, klog=True, vnn=True, vprior=n0))
    return [scope, vnn, vprior, both]


def s_jobs(runs):
    out = [("S", (n, k, m, b, ph), s, mu, None) for (s, mu) in runs for (n, k, m, b, ph) in R25.WORLDS]
    out += [("S", (n, k, m, b, None), s, mu, c) for (s, mu) in runs for (n, k, m, b, c) in R26.CHANGE + R27.NEWW]
    return out


EVAL_RUNS = ((14, 3), (15, 3), (14, 10))


def one(part, name, kind, mix, b, phase, seed, mult, cm, kw, scale=None):
    saved_cls = R.P.BatchAccount19
    saved_world = R.V.make_world
    R.P.BatchAccount19 = E.BatchAccount22
    if scale is not None and scale != 1.0:
        def wrapped(*a, **k):
            w = saved_world(*a, **k)
            w.d0 *= scale
            w.d_req *= scale
            w.d_tok *= scale
            return w
        R.V.make_world = wrapped
    try:
        out = R.one(part, name, kind, mix, b, phase, seed, mult, cm, kw)
    finally:
        R.P.BatchAccount19 = saved_cls
        R.V.make_world = saved_world
    p = E.BatchAccount22.current if hasattr(E.BatchAccount22, "current") else None
    return out, p


def emit(part, name, kind, mix, b, phase, seed, mult, cm, kw, scale=None):
    out, p = one(part, name, kind, mix, b, phase, seed, mult, cm, kw, scale)
    out["n_explore"] = int(getattr(p, "n_explore", 0))
    out["n_cprobe"] = int(getattr(p, "n_cprobe", 0))
    if p is not None and p.k_snap:
        out["k_snap"] = p.k_snap
    if p is not None and p.bias:
        out["bias"] = p.bias_rows()
    if p is not None and p.ex:
        out["ex"] = p.ex_rows()
    return out


def _vx(b, K, suml):
    s_ = suml / 1000.0
    return [float(K == 8), float(K == 16), float(K == 32), float(b), s_, (K + 1) * s_ / 17.0]


def _cell(kw):
    return emit("S", "bvar", "het", "mixed", 8, None, 9, 3, None, kw)


def check():
    rec = None
    for path in ("results/r34s.0.jsonl", "results/r34s.1.jsonl"):
        for line in open(path):
            r = json.loads(line)
            if (r["world"], r["kind"], r["b"], r["seed"], r["req_mult"]) == ("bvar", "het", 8, 9, 3):
                rec = r["res"]["scope"]
    assert rec is not None

    got = _cell(dict(BASE))
    assert got["switches"] == rec["switches"], (got["switches"], rec["switches"])
    assert got["final"] == rec["final"], (got["final"], rec["final"])
    assert json.loads(json.dumps(got["switch_log"])) == rec["switch_log"]
    assert got["rate"] == rec["rate"], (got["rate"], rec["rate"])

    fed = {8: 90.0, 16: 8.0, 32: 3.0}
    plain = E.H._RLS(6)
    nn = E._VerifyRLS(n0=0.0, vnn=True)
    for K, ak in fed.items():
        for _ in range(80):
            x = _vx(4.0, K, 2000.0)
            plain.add(x, ak)
            nn.add(x, ak)
    raw = plain.coef()
    assert raw[5] < -1e-8, raw
    got_w = nn.coef()
    assert got_w[4] >= -1e-8 and got_w[5] >= -1e-8, got_w
    gaps = (fed[16] - fed[8], fed[32] - fed[8], fed[32] - fed[16])
    solved = (got_w[1] - got_w[0], got_w[2] - got_w[0], got_w[2] - got_w[1])
    for a, b_ in zip(solved, gaps):
        assert abs(a - b_) / abs(b_) < 1e-4, (solved, gaps, got_w)

    pos = E.H._RLS(6)
    pos_nn = E._VerifyRLS(n0=0.0, vnn=True)
    rng = np.random.default_rng(38)
    for _ in range(400):
        K = int(rng.choice([8, 16, 32]))
        b = float(rng.choice([1.0, 4.0, 8.0]))
        suml = float(rng.uniform(500.0, 8000.0))
        s_ = suml / 1000.0
        t = (20.0 + (5.0 if K == 16 else 0.0) + (12.0 if K == 32 else 0.0)
             + 1.5 * b + 3.0 * s_ + 2.0 * (K + 1) * s_ / 17.0)
        x = _vx(b, K, suml)
        pos.add(x, t)
        pos_nn.add(x, t)
    a, c = pos.coef(), pos_nn.coef()
    for i in range(6):
        den = max(abs(a[i]), 1e-6)
        assert abs(c[i] - a[i]) / den < 1e-6, (i, a, c)

    base = E.H._RLS(6)
    prior = E._VerifyRLS(n0=6.0, vnn=False)
    rng = np.random.default_rng(3806)
    for _ in range(4000):
        K = int(rng.choice([8, 16, 32]))
        b = float(rng.choice([-4.0, 0.0, 4.0]))
        suml = float(rng.uniform(-4000.0, 4000.0))
        s_ = suml / 1000.0
        t = (28.0 + (6.0 if K == 16 else 0.0) + (14.0 if K == 32 else 0.0)
             + 0.8 * b + 1.5 * s_ + 1.1 * (K + 1) * s_ / 17.0)
        x = _vx(b, K, suml)
        base.add(x, t)
        prior.add(x, t)
    a, c = base.coef(), prior.coef()
    for i in range(6):
        if abs(a[i]) < 1e-3:
            continue
        assert abs(c[i] - a[i]) / abs(a[i]) < 0.01, (i, a, c)


def p_jobs():
    worlds = [
        ("S", ("het", "het", "mixed", 4, None), 7, 3, None),
        ("S", ("base", "mid", "short", 4, None), 7, 3, None),
        ("S", ("high", "high", "short", 8, None), 7, 3, None),
    ]
    pols = [(f"n{n0}", dict(BASE, vprior=n0)) for n0 in (2, 6, 20)]
    return [(j, pols, None) for j in worlds]


def b_jobs():
    worlds = [
        ("S", ("mid", "mid", "short", 1, None), 7, 3, None),
        ("S", ("high", "high", "short", 1, None), 7, 3, None),
        ("S", ("het", "het", "mixed", 1, None), 7, 3, None),
    ]
    pols = [("scope", dict(BASE, biaslog=True))]
    pols += [(f"f{w}", dict(BASE, cost_mode="known", wmode="fixed", fixed_W=w)) for w in (2, 4, 8)]
    return [(j, pols, None) for j in worlds]


def w_jobs():
    keys = [("base", "mid", "short", 4, None), ("base", "mid", "short", 8, None),
            ("high", "high", "short", 4, None), ("high", "high", "short", 8, None),
            ("het", "het", "mixed", 4, None), ("het", "het", "mixed", 8, None)]
    pol = [("scope", dict(BASE, explog=True, biaslog=True))]
    return [(("S", key, 7, 3, None), pol, None) for key in keys]


def g_jobs():
    out = []
    for b in (2, 4):
        for scale in (1.0, 0.3, 0.1):
            job = ("S", ("low", "low", "short", b, None), 7, 3, None)
            pols = [(f"f{w}", dict(BASE, cost_mode="known", wmode="fixed", fixed_W=w)) for w in (2, 4, 8)]
            out.append((job, pols, scale))
    assert len(out) == 6, len(out)
    return out


def eval_jobs():
    n0 = n0_of()
    jobs = [(j, arms(n0), None) for j in s_jobs(EVAL_RUNS)]
    assert len(jobs) == 45, len(jobs)
    return jobs


def seed16_jobs():
    n0 = n0_of()
    seen = {}
    for (_pt, key, _s, _m, cm) in R.s_jobs():
        if key not in seen:
            seen[key] = cm
    worlds = [("S", key, 16, 3, cm) for key, cm in seen.items()]
    assert len(worlds) == 15, len(worlds)
    return [(j, arms(n0), None) for j in worlds]


def guard_jobs():
    adopted = json.load(open("results/r38_adopt.json"))["arm"]
    n0 = n0_of()
    table = {lab: kw for lab, kw in arms(n0)}
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
    check()
    print("check ok", file=sys.stderr, flush=True)
    proc = int(sys.argv[1])
    part = sys.argv[2]
    jobs = {"p": p_jobs, "b": b_jobs, "w": w_jobs, "g": g_jobs,
            "eval": eval_jobs, "seed16": seed16_jobs, "guard": guard_jobs}[part]()
    tag = {"p": "r38p", "b": "r38b", "w": "r38w", "g": "r38g",
           "eval": "r38e", "seed16": "r38s", "guard": "r38g2"}[part]
    n0 = None
    if part in ("eval", "seed16", "guard"):
        n0 = n0_of()
    for i, ((pt, (name, kind, mix, b, phase), seed, mult, cm), pols, scale) in enumerate(jobs):
        if i % 2 != proc:
            continue
        print(f"{tag} start {name} b={b} seed={seed} x{mult}", file=sys.stderr, flush=True)
        t0 = time.time()
        res = {}
        for lab, kw in pols:
            res[lab] = emit(pt, name, kind, mix, b, phase, seed, mult, cm, kw, scale)
        sec = round(time.time() - t0, 1)
        print(json.dumps(dict(part=pt, run=tag, world=name, kind=kind, mix=mix, b=b, phase=phase,
                              seed=seed, req_mult=mult, cmult=cm, lmult=scale, n0=n0, res=res,
                              sec=sec, python=sys.version.split()[0])), flush=True)
        print(f"{tag} {name} b={b} seed={seed} x{mult} {sec}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
