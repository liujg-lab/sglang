"""Remaining simulation tests. Part A: planner cost (measured, lagged) and time-varying delta.
Part B: speculation switch under alternating RTT, and seed robustness."""
import json, sys, time
import numpy as np
import feasibility as F
import ext as E
from calibrated import CAL


def emit(**kw):
    print(json.dumps(kw), flush=True)


def part_a():
    for b in (1, 4, 6, 8):
        w = F.World(b=b, rtt=4.0, misspec=False, seed=7, **CAL)
        W = 4
        res = {}
        for name, kw in [("fixed 0.1ms", dict(cost="fixed", cP=0.1)),
                         ("measured", dict(cost="measured")),
                         ("lag, fixed 0.1ms", dict(cost="fixed", cP=0.1, lag=True)),
                         ("lag, measured", dict(cost="measured", lag=True))]:
            pol = E.ClearTreeFast(W, "oracle", **kw)
            thr = F.simulate(w, pol)[0]
            res[name] = [thr, float(np.mean(pol.charged))]
        emit(test="planner", b=b, res=res)
    for mis, b in ((True, 1), (True, 4), (False, 1), (False, 4)):
        w = F.World(b=b, rtt=4.0, misspec=mis, seed=7, **CAL)
        W = 4 if not (mis and b == 1) else 8
        res = {}
        for name, kw in [("learned+beta", dict(learn_beta=True)), ("learned+beta+dyn", dict(learn_beta=True, dyn_delta=True))]:
            res[name] = F.simulate(w, E.ClearTreeFast(W, "learned", cP=0.1, **kw))[0]
        emit(test="dyn_delta", misspec=mis, b=b, res=res)


def part_b():
    grid = [(N, t) for N in (4, 8) for t in (0.9, 0.6, 0.4, 0.2)]
    cases = [(b, (4.0, 20.0), 3000.0) for b in (4, 6, 8)] + [(4, 20.0, None), (8, 4.0, None)]
    for b, rtt, period in cases:
        w = F.World(b=b, rtt=rtt, misspec=False, seed=7, **CAL)
        sim = lambda pol: E.simulate_t(w, pol, period_ms=period)[0]
        res = {"Target-only": sim(F.AR()), "OPT(8,0.4)": sim(F.OPT(8, 0.4)),
               "OPT(8,0.4)+static switch b>=7": sim(E.StaticSwitch(F.OPT(8, 0.4))),
               "OPT(8,0.4)+adaptive switch": sim(E.SpecSwitch(F.OPT(8, 0.4)))}
        g = {k: sim(F.OPT(*k)) for k in grid}
        bk = max(g, key=g.get)
        res[f"best static {bk}"] = g[bk]
        res[f"best static {bk}+adaptive switch"] = sim(E.SpecSwitch(F.OPT(*bk)))
        res["ClearTree learned"] = sim(E.ClearTreeFast(4, "learned", cP=0.1))
        sw = E.SpecSwitch(E.ClearTreeFast(4, "learned", cP=0.1))
        res["ClearTree learned+adaptive switch"] = sim(sw)
        res["(spec time share)"] = sw.spec_share[0] / max(1e-9, sw.spec_share[1])
        emit(test="switch", b=b, rtt=rtt, period=period, res=res)
    for seed in (8, 9):
        for b in (4, 8):
            w = F.World(b=b, rtt=4.0, misspec=False, seed=seed, **CAL)
            g = {f"{N},{t}": F.simulate(w, F.OPT(N, t))[0] for N in F.KS for t in F.TAUS}
            bk = max(g, key=g.get)
            emit(test="seed", seed=seed, b=b, best_static=[bk, g[bk]], default=g["8,0.4"],
                 ct_oracle=F.simulate(w, E.ClearTreeFast(4, "oracle", cP=0.1))[0],
                 ct_learned=F.simulate(w, E.ClearTreeFast(4, "learned", cP=0.1))[0])


if __name__ == "__main__":
    {"A": part_a, "B": part_b}[sys.argv[1]]()
