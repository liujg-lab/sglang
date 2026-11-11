import sys, json, time
import numpy as np
import feasibility as F
from calibrated import CAL
for b in (1, 4):
    w = F.World(b=b, rtt=4.0, misspec=True, seed=7, **CAL)
    W = 8 if b == 1 else 4
    pol = F.ClearTree(W, "learned", cP=0.1, learn_beta=True)
    thr = F.simulate(w, pol)[0]
    print(json.dumps({"b": b, "W": W, "learned+beta": thr, "beta_hat": pol.bhat}), flush=True)
