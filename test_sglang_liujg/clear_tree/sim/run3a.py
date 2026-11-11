"""Simulation round 3, phase 1: conservatism factor (seeds) and model-guided slow loop."""
import json, sys
import numpy as np
import feasibility as F
import ext2 as X
import ext3 as Y
from calibrated import CAL


def emit(**kw):
    print(json.dumps(kw), flush=True)


def core(w, W=4, fd=1.0, slow=None, mode="learned"):
    return Y.CT3(W, mode, cost=Y.CostModel3(w, "learned", p_force=0.03, fd=fd), slow=slow, n_snap=4, kreuse=10,
                 learn_beta=True, dyn_delta=True)


def part_cons():
    for b in (1, 4, 8):
        for seed in (7, 8, 9):
            w = X.World2(b=b, rtt=4.0, seed=seed, **CAL)
            res = {str(fd): X.simulate3(w, core(w, W=8 if b == 1 else 4, fd=fd)) for fd in (1.0, 1.15, 1.3, 1.5, 2.0)}
            emit(test="cons", b=b, seed=seed, res=res)


def part_slow():
    for b in (1, 4, 8):
        for seed in (7, 8):
            w = X.World2(b=b, rtt=4.0, seed=seed, **CAL)
            res = {"fixed W=4": X.simulate3(w, core(w, W=4)), "fixed W=8": X.simulate3(w, core(w, W=8))}
            res["UCB"] = X.simulate3(w, core(w, W=8, slow=X.SlowLoop()))
            g = Y.GuidedSlowLoop2()
            res["guided"] = X.simulate3(w, core(w, W=8, slow=g))
            res["guided final W"] = {k: v["cur"] for k, v in g.ctx.items()}
            res["guided switches"] = g.switches
            emit(test="slow", b=b, seed=seed, res=res)
    for seed in (7, 8):
        w = X.World2(b=4, rtt=4.0, seed=seed, noise=0.05, jitter=1.0, n_req=400, L_range=(32, 512), **CAL)
        conc = [(25000, 1), (50000, 4), (75000, 8), (100000, 4)]
        sim = lambda pol: X.simulate3(w, pol, conc=conc)
        g = Y.GuidedSlowLoop2()
        res = {"fixed W=4": sim(core(w, W=4)), "UCB": sim(core(w, W=8, slow=X.SlowLoop())), "guided": sim(core(w, W=8, slow=g)),
               "guided final W": {k: v["cur"] for k, v in g.ctx.items()}}
        emit(test="slow_mixed", seed=seed, res=res)


if __name__ == "__main__":
    {"cons": part_cons, "slow": part_slow}[sys.argv[1]]()
