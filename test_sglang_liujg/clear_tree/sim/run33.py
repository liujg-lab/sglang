"""Round 33: the spike gate's counter covers only ordinary rounds of the width the controller is using.
usage (conda sglang env): python run33.py <proc 0|1> [a|b]
  a (default) phased b = 8 and the 10 tail worlds, written to r33a. Already run; do not repeat.
  b RTT jump / RTT alt (12 runs, scope only) and phased b = 8 seed 10 short (cur and scope), written to r33b.
"""
import json, sys, time
import ext26 as P
import run32 as R

SCOPE = ("scope", dict(cooldown=False, rob=True, rob_scope=True))


def check():
    p = P.BatchAccount19(rob=True, rob_scope=True, cooldown=False)
    p._force_w, p._force_k, p._tag = True, False, "cprobe"
    p.streak[(4, 2)] = 2
    p.cell_nl[(4, 2)] = 10
    assert p._streak((4, 2), 0.9) == 2
    assert p._skip(p.cell_nl, (4, 2), (4, 2), 0.9) is False
    assert p.streak[(4, 2)] == 2

    p._force_w, p._tag = False, "normal"
    p.cell_nl[(4, 4)] = 10
    p.streak[(4, 4)] = 0
    for i, expect in ((1, True), (2, True), (3, False)):
        p.streak[(4, 4)] = p._streak((4, 4), 0.8)
        skipped = p._skip(p.cell_nl, (4, 4), (4, 4), 0.8)
        assert skipped is expect, (i, skipped, p.streak[(4, 4)])

    p._tag = "explore"
    p.streak[(4, 8)] = 1
    assert p._streak((4, 8), 0.9) == 1

    p.streak[(1, 4)] = 2
    p.streak[(2, 2)] = 3
    p.streak[("v", 8)] = 4
    p._clear_layer_streak(4)
    p._clear_layer_streak(2)
    assert p.streak[(1, 4)] == 0 and p.streak[(2, 2)] == 0 and p.streak[("v", 8)] == 4


def jobs():
    vp = [SCOPE]
    out = [(("S", ("phased", "het", "mixed", 8, "lowhighmid"), s, mu, None), vp) for (s, mu) in R.RUNS]
    tw = [("tail base", "mid", "short", 4, None), ("tail high", "high", "short", 8, None),
          ("tail het", "het", "mixed", 4, None), ("tail chg8 x10", "mid", "short", 8, 10.0)]
    out += [(("D", (n, k, m, b, None), s, 3, c), vp) for s in (8, 9) for (n, k, m, b, c) in tw]
    out += [(("D", (n, k, m, b, None), 8, 10, c), vp) for (n, k, m, b, c) in tw[:2]]
    return out


def jobs_b():
    out = [(j, [SCOPE]) for j in R.rtt_jobs(["rtt jump", "rtt alt"])]
    out.append((("S", ("phased", "het", "mixed", 8, "lowhighmid"), 10, 3, None), [R.CUR, SCOPE]))
    return out


def one(pt, name, kind, mix, b, phase, seed, mult, cm, kw):
    made = []
    orig = P.BatchAccount19.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        made.append(self)

    P.BatchAccount19.__init__ = init
    try:
        out = R.one(pt, name, kind, mix, b, phase, seed, mult, cm, kw)
    finally:
        P.BatchAccount19.__init__ = orig
    p = made[-1]
    out["why"] = p.why
    out["admit_left"] = p.admit
    return out


def main():
    check()
    proc = int(sys.argv[1])
    part = sys.argv[2] if len(sys.argv) > 2 else "a"
    if part == "a":
        joblist, tag = jobs(), "r33a"
    elif part == "b":
        joblist, tag = jobs_b(), "r33b"
    else:
        raise SystemExit("part?")
    for i, ((pt, (name, kind, mix, b, phase), seed, mult, cm), pols) in enumerate(joblist):
        if i % 2 != proc:
            continue
        t0 = time.time()
        res = {lab: one(pt, name, kind, mix, b, phase, seed, mult, cm, kw) for lab, kw in pols}
        print(json.dumps(dict(part=pt, run=tag, world=name, kind=kind, mix=mix, b=b, phase=phase, seed=seed,
                              req_mult=mult, cmult=cm, res=res, sec=round(time.time() - t0, 1),
                              python=sys.version.split()[0])), flush=True)


if __name__ == "__main__":
    main()
