"""Round 37: sliding window and the wider-switch sample size, plus three diagnostics.
usage (conda sglang env): python run37.py <proc 0|1> <part>

  k      het b=4, seeds 8 and 9, short. Four cells: known/online cost x fixed W=4 / replay. r37k.
  bias   six static worlds, seed 7, short, scope with a shadow tree beside the replay. r37b.
  x      the 45 runs of run32.s_jobs with explore_p=0 and cprobe_period=0. r37x. Compared to r34s.
  eval   seeds 11 and 12 short, seed 11 long, 45 worlds. scope, win200, wide30, both. r37e.
  seed13 the 15 worlds, seed 13, short. Same four arms. r37s.
         Pre-registered only when win200 on the 45 is between +0.10% and +0.30%, or a side condition fails.
  guard  the 28 guard worlds. Reads results/r37_adopt.json for the adopted arm. r37g.
check() runs first. A huge window with wide_need=10 must match one r34s cell, explore_p=0.05 must match
the same cell, and wide_need=30 must wait for 30 pairs and the next 20-round tick.
"""
import json, sys, time
import ext9 as Q
import ext28 as E
import run25 as R25
import run26 as R26
import run27 as R27
import run32 as R

BASE = dict(cooldown=False, rob=True, rob_scope=True)
SCOPE = ("scope", dict(BASE))
WIN = ("win200", dict(BASE, win200=200))
WIDE = ("wide30", dict(BASE, win200=10 ** 9, wide_need=30))
BOTH = ("both", dict(BASE, win200=200, wide_need=30))
ARMS = {"scope": SCOPE, "win200": WIN, "wide30": WIDE, "both": BOTH}
EVAL_RUNS = ((11, 3), (12, 3), (11, 10))
HUGE = 10 ** 9


def s_jobs(runs):
    out = [("S", (n, k, m, b, ph), s, mu, None) for (s, mu) in runs for (n, k, m, b, ph) in R25.WORLDS]
    out += [("S", (n, k, m, b, None), s, mu, c) for (s, mu) in runs for (n, k, m, b, c) in R26.CHANGE + R27.NEWW]
    return out


def one(part, name, kind, mix, b, phase, seed, mult, cm, kw):
    saved = R.P.BatchAccount19
    R.P.BatchAccount19 = E.BatchAccount21
    try:
        out = R.one(part, name, kind, mix, b, phase, seed, mult, cm, kw)
    finally:
        R.P.BatchAccount19 = saved
    p = E.BatchAccount21.current if hasattr(E.BatchAccount21, "current") else None
    return out, p


def emit(part, name, kind, mix, b, phase, seed, mult, cm, kw):
    out, p = one(part, name, kind, mix, b, phase, seed, mult, cm, kw)
    out["n_explore"] = int(getattr(p, "n_explore", 0))
    out["n_cprobe"] = int(getattr(p, "n_cprobe", 0))
    if p is not None and p.k_snap:
        out["k_snap"] = p.k_snap
    if p is not None and p.bias:
        out["bias"] = p.bias_rows()
    return out


def _more(p, n, tok, other, cur_tok=128.0):
    for _ in range(n):
        rep = {4: dict(tok=cur_tok, mt=1.0, lt=0.0), other: dict(tok=float(tok), mt=1.0, lt=0.0)}
        p._post14m(4, 4, "normal", rep, [cur_tok], cur_tok, 0.0)
        p.rounds += 1


def _cell(kw):
    return emit("S", "bvar", "het", "mixed", 8, None, 9, 3, None, kw)


def check():
    p = E.BatchAccount21(**BASE, win200=HUGE, wide_need=30)
    p.ctx14[Q.bkey(4)] = dict(cur=4, follow=False, stats={}, n=0, rc=None)
    p.n_switch, p.switch_log, p.sw_se, p.rounds = 0, [], [], 0
    _more(p, 20, 128.0 * 1.10, 8)
    assert p.n_switch == 0, p.switch_log
    _more(p, 9, 128.0 * 1.10, 8)
    assert p.n_switch == 0 and p.ctx14[Q.bkey(4)]["stats"][8][4] == 29
    _more(p, 1, 128.0 * 1.10, 8)
    assert p.n_switch == 0 and p.ctx14[Q.bkey(4)]["stats"][8][4] == 30
    _more(p, 10, 128.0 * 1.10, 8)
    assert p.n_switch == 1 and p.switch_log[-1][2:] == (4, 8), p.switch_log
    assert p.sw_se[-1][6] >= 30, p.sw_se[-1]

    rec = None
    for path in ("results/r34s.0.jsonl", "results/r34s.1.jsonl"):
        for line in open(path):
            r = json.loads(line)
            if (r["world"], r["kind"], r["b"], r["seed"], r["req_mult"]) == ("bvar", "het", 8, 9, 3):
                rec = r["res"]["scope"]
    assert rec is not None

    def same(got, label):
        assert got["switches"] == rec["switches"], (label, got["switches"], rec["switches"])
        assert got["final"] == rec["final"], (label, got["final"], rec["final"])
        assert json.loads(json.dumps(got["switch_log"])) == rec["switch_log"], label
        assert got["rate"] == rec["rate"], (label, got["rate"], rec["rate"])

    same(_cell(dict(BASE, win200=HUGE, wide_need=10)), "wide10")
    same(_cell(dict(BASE, explore_p=0.05)), "explore")


def k_jobs():
    pols = [
        ("known-fixed", dict(BASE, cost_mode="known", wmode="fixed", fixed_W=4, klog=True)),
        ("known-replay", dict(BASE, cost_mode="known", wmode="replay14", klog=True)),
        ("online-fixed", dict(BASE, wmode="fixed", fixed_W=4, klog=True)),
        ("online-replay", dict(BASE, klog=True)),
    ]
    worlds = [("S", ("het", "het", "mixed", 4, None), s, 3, None) for s in (8, 9)]
    return [(j, pols) for j in worlds]


def bias_jobs():
    keys = [("base", "mid", "short", 4, None), ("base", "mid", "short", 8, None),
            ("high", "high", "short", 4, None), ("high", "high", "short", 8, None),
            ("het", "het", "mixed", 4, None), ("het", "het", "mixed", 8, None)]
    pol = [("scope", dict(BASE, biaslog=True))]
    return [(("S", key, 7, 3, None), pol) for key in keys]


def x_jobs():
    pol = [("nox", dict(BASE, explore_p=0.0, cprobe_period=0))]
    jobs = [(j, pol) for j in R.s_jobs()]
    assert len(jobs) == 45, len(jobs)
    return jobs


def eval_jobs():
    jobs = [(j, [SCOPE, WIN, WIDE, BOTH]) for j in s_jobs(EVAL_RUNS)]
    assert len(jobs) == 45, len(jobs)
    return jobs


def seed13_jobs():
    seen = {}
    for (_pt, key, _s, _m, cm) in R.s_jobs():
        if key not in seen:
            seen[key] = cm
    worlds = [("S", key, 13, 3, cm) for key, cm in seen.items()]
    assert len(worlds) == 15, len(worlds)
    return [(j, [SCOPE, WIN, WIDE, BOTH]) for j in worlds]


def guard_jobs():
    adopted = json.load(open("results/r37_adopt.json"))["arms"]
    pols = [SCOPE] + [ARMS[n] for n in adopted]
    out = [(j, pols) for j in R.rtt_jobs(["rtt jump", "rtt alt"])]
    tw = [("tail base", "mid", "short", 4, None), ("tail high", "high", "short", 8, None),
          ("tail het", "het", "mixed", 4, None), ("tail chg8 x10", "mid", "short", 8, 10.0)]
    out += [(("D", (n, k, m, b, None), s, 3, c), pols) for s in (8, 9) for (n, k, m, b, c) in tw]
    out += [(("D", (n, k, m, b, None), 8, 10, c), pols) for (n, k, m, b, c) in tw[:2]]
    out += [(("L", ("lalt x3", kd, "short", 8, None), s, mu, None), pols)
            for (s, mu) in R.RUNS for kd in ("mid", "high")]
    assert len(out) == 28, len(out)
    return out


def main():
    check()
    print("check ok", file=sys.stderr)
    proc = int(sys.argv[1])
    part = sys.argv[2]
    jobs = {"k": k_jobs, "bias": bias_jobs, "x": x_jobs, "eval": eval_jobs,
            "seed13": seed13_jobs, "guard": guard_jobs}[part]()
    tag = {"k": "r37k", "bias": "r37b", "x": "r37x", "eval": "r37e", "seed13": "r37s", "guard": "r37g"}[part]
    for i, ((pt, (name, kind, mix, b, phase), seed, mult, cm), pols) in enumerate(jobs):
        if i % 2 != proc:
            continue
        t0 = time.time()
        res = {lab: emit(pt, name, kind, mix, b, phase, seed, mult, cm, kw) for lab, kw in pols}
        print(json.dumps(dict(part=pt, run=tag, world=name, kind=kind, mix=mix, b=b, phase=phase, seed=seed,
                              req_mult=mult, cmult=cm, res=res, sec=round(time.time() - t0, 1),
                              python=sys.version.split()[0])), flush=True)


if __name__ == "__main__":
    main()
