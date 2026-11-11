import numpy as np, feasibility as F
from calibrated import CAL
for ns in (16, 4):
    F.N_SNAP = ns
    for b in (1, 4, 8):
        w = F.World(b=b, rtt=4.0, misspec=False, seed=7, n_req=b, L=128, **CAL)
        pol = F.ClearTree(4 if b > 1 else 8, "oracle", cP=0.1)
        thr = F.simulate(w, pol)[0]
        a = np.asarray(pol.plan_times) * 1000
        print(f"N_snap={ns} b={b}: plan ms/call mean {a.mean():.2f} p99 {np.percentile(a, 99):.2f}  (throughput {thr:.0f})", flush=True)
