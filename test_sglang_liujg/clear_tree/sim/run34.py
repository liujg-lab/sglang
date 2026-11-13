"""Round 34: two checks from round 31, on the settled scope policy (no cooldown, rob, rob_scope).
usage (conda sglang env): python run34.py <proc 0|1> [s|b]
  s  se2 (a switch also needs gain > 2 SE) vs scope on the 45 round-28 world runs. Written to r34s.
  b  boff (a common K-check shift is stored per b bucket) vs scope on balt cb and balt rtt. Written to r34b.
Does not rerun round 31. Does not add a K-check interval.
"""
import json, sys, time
import numpy as np
import ext9 as Q
import ext26 as P
import run32 as R

SCOPE = ("scope", dict(cooldown=False, rob=True, rob_scope=True))
SE2 = ("se2", dict(cooldown=False, rob=True, rob_scope=True, se_z=2.0))
BOFF = ("boff", dict(cooldown=False, rob=True, rob_scope=True, boff=True))


def _feed(p, toks):
    b = 4
    p.ctx14[Q.bkey(b)] = dict(cur=4, follow=False, stats={}, n=0, rc=None)
    p.n_switch, p.n_se_block = 0, 0
    p.switch_log, p.sw_se = [], []
    for a in toks:
        rep = {4: dict(tok=100.0, mt=0.0, lt=0.0), 8: dict(tok=float(a), mt=0.0, lt=0.0)}
        p._post14m(b, 4, "normal", rep, [100.0], 100.0, 0.0)
    return p.ctx14[Q.bkey(b)]["stats"].get(8)


def _check_ready(p, bk):
    p.world = type("W", (), {"n_T": 1})()
    p.rounds = 10
    p.rv.A[:] = np.eye(len(p.rv.y))
    p.rv.y[:] = 1.0
    p.rv._w = None
    p.check = dict(bk=bk, vp=[], at=0,
                   res={K: [10.0, 10.0, 10.0, 10.0] for K in p.k_options})
    return p.rv.y.copy()


def check():
    kw = dict(cooldown=False, rob=True, rob_scope=True, se_z=2.0)
    blocked = P.BatchAccount19(**kw)
    st = _feed(blocked, [200.0] * 10 + [10.0] * 10)
    g = (st[0] / st[1]) / (st[2] / st[3]) - 1
    se = blocked._gain_se(st, g)
    assert g > 0.03 and g <= 2.0 * se, (g, se)
    assert blocked.n_switch == 0 and blocked.n_se_block >= 1

    allowed = P.BatchAccount19(**kw)
    _feed(allowed, [105.0] * 20)
    assert allowed.n_switch == 1 and allowed.switch_log[-1][2:] == (4, 8)
    assert allowed.n_se_block == 0

    bk = 8
    kept = P.BatchAccount19(cooldown=False, rob=True, rob_scope=True, boff=True)
    y0 = _check_ready(kept, bk)
    kept._end_check()
    assert kept.voff.get(bk) == 10.0
    assert np.allclose(kept.rv.y, y0)
    assert kept.n_offsets == 1

    moved = P.BatchAccount19(cooldown=False, rob=True, rob_scope=True, boff=False)
    y1 = _check_ready(moved, bk)
    moved._end_check()
    assert not moved.voff
    assert not np.allclose(moved.rv.y, y1)


def jobs_s():
    jobs = [(j, [SCOPE, SE2]) for j in R.s_jobs()]
    assert len(jobs) == 45, len(jobs)
    return jobs


def jobs_b():
    return [(("B", (nm, "mid", "short", 8, None), s, mu, None), [SCOPE, BOFF])
            for (s, mu) in R.RUNS for nm in ("balt cb", "balt rtt")]


def main():
    check()
    print("check ok", file=sys.stderr)
    proc = int(sys.argv[1])
    part = sys.argv[2] if len(sys.argv) > 2 else "s"
    if part == "s":
        joblist, tag = jobs_s(), "r34s"
    elif part == "b":
        joblist, tag = jobs_b(), "r34b"
    else:
        raise SystemExit("part?")
    for i, ((pt, (name, kind, mix, b, phase), seed, mult, cm), pols) in enumerate(joblist):
        if i % 2 != proc:
            continue
        t0 = time.time()
        res = {lab: R.one(pt, name, kind, mix, b, phase, seed, mult, cm, kw) for lab, kw in pols}
        print(json.dumps(dict(part=pt, run=tag, world=name, kind=kind, mix=mix, b=b, phase=phase, seed=seed,
                              req_mult=mult, cmult=cm, res=res, sec=round(time.time() - t0, 1),
                              python=sys.version.split()[0])), flush=True)


if __name__ == "__main__":
    main()
