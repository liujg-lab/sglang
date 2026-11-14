"""Round 36: CUSUM on the paired width gain, against the settled scope policy.
usage (conda sglang env): python run36.py <proc 0|1> <part> [r36b|r36]
  The tag defaults to r36b, the rerun after the cost-event fix (D77). r36 is the first run.

  pilot  seed 7, short (3x), the 15 worlds of run32.s_jobs. scope plus the six pre-registered CUSUM settings.
         r36b writes r36bp. Parameter choice is not made here.
  eval   the 45 runs of run32.s_jobs. r36b writes r36be and runs the frozen CUSUM plus win200;
         oracle stays in the first run's r36e. r36 also runs oracle. scope for these 45 is r34s.
  fixed  the same 45 runs at known cost, fixed W = 2, 4, 8. Written to r36f either way.
  guard  RTT jump/alt (12), tail (10), lalt x3 (6): scope, frozen CUSUM, win200.
         r36b writes r36bg.
  seed10 the 15 worlds, seed 10, short (3x). r36b writes r36bs and runs scope, frozen CUSUM, win200.
         Pre-registered only when the 45-run mean is between +0.10% and +0.30%, or a side condition fails.
check() runs first. The scope replay of one r34s cell must match, the CUSUM unit checks must pass,
and a layer event and a verify event must clear the CUSUM state and the win200 window.
"""
import json, sys, time
import ext9 as Q
import ext27 as E
import run32 as R

SCOPE = ("scope", dict(cooldown=False, rob=True, rob_scope=True))
WIN = ("win200", dict(cooldown=False, rob=True, rob_scope=True, win200=200))
ORACLE = ("oracle", dict(cooldown=False, rob=True, rob_scope=True, oracle=True))
PARAMS = ((0.015, 14.0, 8.0), (0.015, 10.0, 10.0), (0.015, 14.0, 6.0),
          (0.020, 12.0, 6.0), (0.015, 16.0, 10.0), (0.020, 14.0, 8.0))
TAG = "r36b"
TAGS = {
    "r36b": dict(pilot="r36bp", eval="r36be", fixed="r36f", guard="r36bg", seed10="r36bs",
                 param="results/r36b_param.json", oracle=False),
    "r36": dict(pilot="r36p", eval="r36e", fixed="r36f", guard="r36g", seed10="r36s",
                param="results/r36_param.json", oracle=True),
}


def cname(k, hn, hw):
    return f"c{int(round(k * 1000)):03d}h{int(hn)}w{int(hw)}"


def cusum_pol(k, hn, hw):
    return (cname(k, hn, hw), dict(cooldown=False, rob=True, rob_scope=True,
                                   cusum_k=k, cusum_hn=hn, cusum_hw=hw))


def frozen():
    d = json.load(open(TAGS[TAG]["param"]))
    return cusum_pol(d["k"], d["hn"], d["hw"])


def one(part, name, kind, mix, b, phase, seed, mult, cm, kw):
    saved = R.P.BatchAccount19
    R.P.BatchAccount19 = E.BatchAccount20
    try:
        return R.one(part, name, kind, mix, b, phase, seed, mult, cm, kw)
    finally:
        R.P.BatchAccount19 = saved


def _feed(p, pairs, other, cur_tok=128.0):
    b = 4
    p.ctx14[Q.bkey(b)] = dict(cur=4, follow=False, stats={}, n=0, rc=None)
    p.n_switch, p.switch_log, p.sw_se = 0, [], []
    p.rounds = 0
    for a, w in pairs:
        rep = {4: dict(tok=cur_tok, mt=0.0, lt=0.0), w: dict(tok=float(a), mt=0.0, lt=0.0)}
        p._post14m(b, 4, "normal", rep, [cur_tok], cur_tok, 0.0)
        p.rounds += 1
    return p


def check():
    kw = dict(cooldown=False, rob=True, rob_scope=True, cusum_k=0.015, cusum_hn=14.0, cusum_hw=8.0)
    neg = _feed(E.BatchAccount20(**kw), [(64.0, 2)] * 30, 2)
    arm = neg.ctx14[Q.bkey(4)]["cz"]["arm"][2]
    assert neg.n_switch == 0 and arm["S"] == 0.0, (neg.n_switch, arm["S"])

    pos = E.BatchAccount20(**kw)
    _feed(pos, [(160.0, 2)] * 5, 2)
    assert pos.n_switch == 0
    _feed_more = pos
    _feed_more._post14m(4, 4, "normal",
                        {4: dict(tok=128.0, mt=0.0, lt=0.0), 2: dict(tok=160.0, mt=0.0, lt=0.0)},
                        [128.0], 128.0, 0.0)
    assert pos.n_switch == 1 and pos.switch_log[-1][2:] == (4, 2), pos.switch_log

    wide = E.BatchAccount20(**kw)
    _feed(wide, [(160.0, 8)] * 4, 8)
    assert wide.n_switch == 0
    wide._post14m(4, 4, "normal",
                  {4: dict(tok=128.0, mt=0.0, lt=0.0), 8: dict(tok=160.0, mt=0.0, lt=0.0)},
                  [128.0], 128.0, 0.0)
    assert wide.n_switch == 1 and wide.switch_log[-1][2:] == (4, 8), wide.switch_log

    hold = E.BatchAccount20(cooldown=False, rob=True, rob_scope=True, win200=200)
    _feed(hold, [(128.0 * 1.02, 2)] * 40, 2)
    assert hold.n_switch == 0
    go = E.BatchAccount20(cooldown=False, rob=True, rob_scope=True, win200=200)
    _feed(go, [(128.0 * 1.04, 2)] * 40, 2)
    assert go.n_switch == 1 and go.switch_log[-1][2:] == (4, 2), go.switch_log

    def _more(p, pairs, other, cur_tok=128.0):
        for a, w in pairs:
            rep = {4: dict(tok=cur_tok, mt=0.0, lt=0.0), w: dict(tok=float(a), mt=0.0, lt=0.0)}
            p._post14m(4, 4, "normal", rep, [cur_tok], cur_tok, 0.0)
            p.rounds += 1

    def _cleared(kind):
        p = E.BatchAccount20(cooldown=False, rob=True, rob_scope=True, cusum_k=0.015,
                             cusum_hn=14.0, cusum_hw=8.0, win200=200)
        p.world = type("W", (), {"n_T": 0})()
        _feed(p, [(160.0, 2)] * 4, 2)
        st = p.ctx14[Q.bkey(4)]
        assert st["cz"]["arm"][2]["n"] == 4 and st["cz"]["arm"][2]["S"] > 0
        assert len(st["hist"][2]) == 4
        if kind == "layer":
            p._layer_event()
        else:
            p._verify_event()
        st = p.ctx14[Q.bkey(4)]
        assert "cz" not in st and "hist" not in st, (kind, list(st))
        _more(p, [(160.0, 2)], 2)
        arm = p.ctx14[Q.bkey(4)]["cz"]["arm"][2]
        assert arm["n"] == 1 and arm["S"] > 0, (kind, arm)

    _cleared("layer")
    _cleared("verify")

    again = E.BatchAccount20(cooldown=False, rob=True, rob_scope=True, win200=200)
    again.world = type("W", (), {"n_T": 0})()
    _feed(again, [(128.0 * 1.02, 2)] * 200, 2)
    assert again.n_switch == 0
    assert len(again.ctx14[Q.bkey(4)]["hist"][2]) == 200
    again._layer_event()
    assert "hist" not in again.ctx14[Q.bkey(4)]
    _more(again, [(128.0 * 1.04, 2)] * 40, 2)
    assert again.n_switch == 1 and again.switch_log[-1][2:] == (4, 2), again.switch_log
    assert again.sw_se[-1][6] == 40, again.sw_se[-1]

    off = E.BatchAccount20(cooldown=False, rob=True, rob_scope=True)
    base = R.P.BatchAccount19(cooldown=False, rob=True, rob_scope=True)
    seq = [(200.0, 8)] * 10 + [(110.0, 8)] * 10
    _feed(off, seq, 8, cur_tok=100.0)
    _feed(base, seq, 8, cur_tok=100.0)
    assert off.n_switch == base.n_switch and off.switch_log == base.switch_log

    got = one("S", "bvar", "het", "mixed", 8, None, 9, 3, None, dict(SCOPE[1]))
    rec = None
    for path in ("results/r34s.0.jsonl", "results/r34s.1.jsonl"):
        for line in open(path):
            r = json.loads(line)
            if (r["world"], r["kind"], r["b"], r["seed"], r["req_mult"]) == ("bvar", "het", 8, 9, 3):
                rec = r["res"]["scope"]
    assert rec is not None
    assert got["switches"] == rec["switches"], (got["switches"], rec["switches"])
    assert got["final"] == rec["final"], (got["final"], rec["final"])
    assert json.loads(json.dumps(got["switch_log"])) == rec["switch_log"]
    assert got["rate"] == rec["rate"], (got["rate"], rec["rate"])


def pilot_jobs():
    seen = {}
    for (_pt, key, _seed, _mult, cm) in R.s_jobs():
        if key not in seen:
            seen[key] = cm
    worlds = [("S", key, 7, 3, cm) for key, cm in seen.items()]
    assert len(worlds) == 15, len(worlds)
    pols = [SCOPE] + [cusum_pol(*p) for p in PARAMS]
    assert len(pols) == 7
    return [(j, pols) for j in worlds]


def eval_jobs():
    pols = [frozen(), WIN] + ([ORACLE] if TAGS[TAG]["oracle"] else [])
    jobs = [(j, pols) for j in R.s_jobs()]
    assert len(jobs) == 45, len(jobs)
    return jobs


def fixed_jobs():
    pols = [(f"fixed {W}", dict(cost_mode="known", wmode="fixed", fixed_W=W,
                                cooldown=False, rob=True, rob_scope=True)) for W in (2, 4, 8)]
    return [(j, pols) for j in R.s_jobs()]


def seed10_jobs():
    seen = {}
    for (_pt, key, _seed, _mult, cm) in R.s_jobs():
        if key not in seen:
            seen[key] = cm
    worlds = [("S", key, 10, 3, cm) for key, cm in seen.items()]
    assert len(worlds) == 15, len(worlds)
    pols = [SCOPE, frozen(), WIN] + ([ORACLE] if TAGS[TAG]["oracle"] else [])
    return [(j, pols) for j in worlds]


def guard_jobs():
    pols = [SCOPE, frozen(), WIN]
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
    global TAG
    check()
    print("check ok", file=sys.stderr)
    proc = int(sys.argv[1])
    part = sys.argv[2]
    TAG = sys.argv[3] if len(sys.argv) > 3 else "r36b"
    jobs = {"pilot": pilot_jobs, "eval": eval_jobs, "fixed": fixed_jobs,
            "guard": guard_jobs, "seed10": seed10_jobs}[part]()
    tag = TAGS[TAG][part]
    for i, ((pt, (name, kind, mix, b, phase), seed, mult, cm), pols) in enumerate(jobs):
        if i % 2 != proc:
            continue
        t0 = time.time()
        res = {lab: one(pt, name, kind, mix, b, phase, seed, mult, cm, kw) for lab, kw in pols}
        print(json.dumps(dict(part=pt, run=tag, world=name, kind=kind, mix=mix, b=b, phase=phase, seed=seed,
                              req_mult=mult, cmult=cm, res=res, sec=round(time.time() - t0, 1),
                              python=sys.version.split()[0])), flush=True)


if __name__ == "__main__":
    main()
