"""Round 12 follow-up: a structure-free acceptance signal. Per round, the first-position top-1 hit (the Target's
token equals the Draft's top-1 at the committed position) is evaluated before any tree is built, so it should not
depend on W, K or tau. Same grid as run12 part R (b=4, seed 9). Prints mean and spread per workload."""
import json
import numpy as np
import ext7 as S
import run8 as R8
import run10 as T
import run12 as U


class Top1(S.OPTDecoupled):
    def __init__(self, W, K, tau):
        super().__init__(W, K, tau)
        self.hit = self.n = 0

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        for r, j in slots:
            self.hit += int(world.reqs[r]["H"][j] == 0); self.n += 1
        return super().round(world, slots, rng, rtt, rnd, trees_out)


out = {}
for kind in R8.WORKLOADS:
    vals = []
    for W, K in ((2, 16), (4, 16), (4, 32), (8, 32)):
        for tau in (0.05, 0.4, 0.8):
            p = Top1(W, K, tau)
            T.sim(R8.make_world(U.FIT, kind, 4, 9, n_req=12), p, W)
            vals.append(p.hit / p.n)
    out[kind] = vals
    print(f"{kind:4s} top-1 hit at the first position: mean {np.mean(vals):.3f}, range {min(vals):.3f}-{max(vals):.3f}, sd {np.std(vals):.3f}")
json.dump(out, open("results/d12_top1.json", "w"))
