"""Simulation round 3, phase 2: full evaluation of the adjusted v1 core against the baselines.
usage: run3b.py <part 0|1> <fd> <slow: guided|fixed>"""
import json, sys
import numpy as np
import feasibility as F
import ext as E
import ext2 as X
import ext3 as Y
from calibrated import CAL

GRID = [(N, t) for N in (4, 8, 16) for t in (0.9, 0.6, 0.4, 0.2)]


def scenarios():
    out = []
    for b in (1, 4, 6, 8):
        for rtt in (4.0, 20.0):
            out.append((f"b={b} RTT={rtt:g}", dict(b=b, rtt=rtt), {}))
    for b in (1, 4):
        out.append((f"misspec b={b} RTT=4", dict(b=b, rtt=4.0, misspec=True), {}))
    mixed = dict(b=4, rtt=4.0, noise=0.05, jitter=1.0, n_req=400, L_range=(32, 512))
    conc = [(25000, 1), (50000, 4), (75000, 8), (100000, 4)]
    out.append(("mixed load RTT=4", mixed, dict(conc=conc)))
    out.append(("mixed load RTT 4<->20 every 30s", dict(mixed, rtt=(4.0, 20.0)), dict(conc=conc, period_ms=30000.0)))
    out.append(("short outputs L=32 b=4", dict(b=4, rtt=4.0, L=32, n_req=256), {}))
    return out


def cm_(w, fd):
    cm = Y.CostModel3(w, "learned", p_force=0.03, fd=fd)
    cm.fd_b1 = 1.0
    return cm


def main():
    part, fd, slow = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    jobs = [(s, seed) for s in scenarios() for seed in (7, 8)]
    for k, ((name, wkw, skw), seed) in enumerate(jobs):
        if k % 2 != part:
            continue
        w = X.World2(seed=seed, **wkw, **CAL)
        sim = lambda pol: X.simulate3(w, pol, **skw)
        core = lambda: Y.CT3(8 if slow == "guided" else 4, "learned",
                             cost=cm_(w, fd),
                             slow=Y.GuidedSlowLoop2() if slow == "guided" else None,
                             n_snap=4, kreuse=10, learn_beta=True, dyn_delta=True)
        res = {"Target-only": sim(F.AR()),
               "OPT(8,0.4)": sim(X.OPT2(8, 0.4)),
               "OPT(8,0.4)+static switch b>=7": sim(E.StaticSwitch(X.OPT2(8, 0.4))),
               "OPT(8,0.4)+adaptive switch": sim(E.SpecSwitch(X.OPT2(8, 0.4)))}
        g = {f"{N},{t}": sim(X.OPT2(N, t)) for N, t in GRID}
        bk = max(g, key=g.get)
        N, t = bk.split(",")
        res["best static (hindsight)"] = [g[bk], bk]
        res["best static+adaptive switch"] = sim(E.SpecSwitch(X.OPT2(int(N), float(t))))
        res["ClearTree core"] = sim(core())
        res["ClearTree core+adaptive switch"] = sim(E.SpecSwitch(core()))
        print(json.dumps(dict(scen=name, seed=seed, res=res)), flush=True)


if __name__ == "__main__":
    main()
