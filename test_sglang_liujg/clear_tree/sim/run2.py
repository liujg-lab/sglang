"""Second batch of simulation tests. Part A: stat layer, K reuse, unknown r_i. Part B: cost model, slow loop and
mixed workload, switch backoff, snapshot-library selection bias. One JSON line per result row."""
import json, sys, time
import numpy as np
import feasibility as F
import ext as E
import ext2 as X
from calibrated import CAL


def emit(**kw):
    print(json.dumps(kw), flush=True)


def part_a():
    # E1 statistical decision layer
    for name, kw, mode in [("wellspec b=4", dict(b=4), "oracle"), ("wellspec b=8", dict(b=8), "oracle"),
                           ("misspec b=4", dict(b=4, misspec=True), "learned")]:
        w = X.World2(rtt=4.0, seed=7, **kw, **CAL)
        for ns in (16, 4):
            res = {}
            for var, flags in [("plain", {}), ("bias", dict(bias_corr=True)), ("cdec", dict(cdec=True)),
                               ("bias+cdec", dict(bias_corr=True, cdec=True))]:
                pol = X.CT2(4, mode, cost=X.CostModel(w), n_snap=ns, learn_beta=(mode == "learned"), **flags)
                thr = X.simulate3(w, pol)
                res[var] = [thr, pol.stats["cdec_calls"] / max(1, pol.stats["decisions"] + pol.stats["rounds"]),
                            pol.stats["plan_ms"] / max(1, pol.stats["rounds"])]
            emit(test="stat", scen=name, n_snap=ns, res=res)
    # E4 reuse K_g
    for b in (4, 8):
        w = X.World2(b=b, rtt=4.0, seed=7, **CAL)
        res = {}
        for k in (1, 5, 20):
            pol = X.CT2(4, "oracle", cost=X.CostModel(w), kreuse=k)
            res[k] = [X.simulate3(w, pol), pol.stats["plan_ms"] / max(1, pol.stats["rounds"])]
        emit(test="kreuse", b=b, res=res)
    # E6 unknown remaining budget, short outputs
    for L in (24, 48):
        for b in (1, 4):
            w = X.World2(b=b, rtt=4.0, seed=7, n_req=int(max(16, 8 * b) * 256 / L), L=L, **CAL)
            res = {"knows r_i": X.simulate3(w, X.CT2(4 if b > 1 else 8, "oracle", cost=X.CostModel(w))),
                   "ignores r_i": X.simulate3(w, X.CT2(4 if b > 1 else 8, "oracle", cost=X.CostModel(w), ignore_r=True)),
                   "OPT(8,0.4)": X.simulate3(w, X.OPT2(8, 0.4))}
            emit(test="budget", L=L, b=b, res=res)


GRID = [(N, t) for N in (4, 8, 16) for t in (0.9, 0.6, 0.4, 0.2)]


def part_b():
    # E2 cost model under noisy costs
    for b in (1, 4, 8):
        w = X.World2(b=b, rtt=4.0, seed=7, noise=0.05, jitter=1.0, **CAL)
        W = 8 if b == 1 else 4
        res = {}
        for var, cm in [("known", X.CostModel(w)), ("learned + forced K 5%", X.CostModel(w, "learned", p_force=0.05)),
                        ("learned, no forced K", X.CostModel(w, "learned")),
                        ("Target cost x0.7", X.CostModel(w, ft=0.7)), ("Target cost x1.3", X.CostModel(w, ft=1.3)),
                        ("Draft cost x0.7", X.CostModel(w, fd=0.7)), ("Draft cost x1.3", X.CostModel(w, fd=1.3))]:
            pol = X.CT2(W, "oracle", cost=cm)
            res[var] = [X.simulate3(w, pol), {str(k): v for k, v in pol.Kv_hist.items()}]
        g = {f"{N},{t}": X.simulate3(w, X.OPT2(N, t)) for N, t in GRID}
        bk = max(g, key=g.get)
        res["best static"] = [g[bk], bk]
        emit(test="cost", b=b, res=res)
    # E3 slow loop (stationary) and mixed workload
    for b in (1, 4, 8):
        w = X.World2(b=b, rtt=4.0, seed=7, **CAL)
        res = {f"W={W}": X.simulate3(w, X.CT2(W, "oracle", cost=X.CostModel(w))) for W in (4, 8, 16)}
        sl = X.SlowLoop()
        res["slow loop"] = X.simulate3(w, X.CT2(8, "oracle", cost=X.CostModel(w), slow=sl))
        res["arm counts"] = {str(k): {str(a): len(v) for a, v in s["hist"].items()} for k, s in sl.ctx.items()}
        emit(test="slowloop", b=b, res=res)
    w = X.World2(b=4, rtt=4.0, seed=7, noise=0.05, jitter=1.0, n_req=600, L_range=(32, 512), **CAL)
    conc = [(30000, 1), (60000, 4), (90000, 8), (120000, 4)]
    sim = lambda pol: X.simulate3(w, pol, conc=conc)
    res = {"Target-only": sim(F.AR()), "OPT(8,0.4)": sim(X.OPT2(8, 0.4))}
    g = {f"{N},{t}": sim(X.OPT2(N, t)) for N, t in GRID}
    bk = max(g, key=g.get)
    res["best single static"] = [g[bk], bk]
    res["CT oracle W=4"] = sim(X.CT2(4, "oracle", cost=X.CostModel(w)))
    res["CT oracle W=8"] = sim(X.CT2(8, "oracle", cost=X.CostModel(w)))
    res["CT oracle slow loop"] = sim(X.CT2(8, "oracle", cost=X.CostModel(w), slow=X.SlowLoop()))
    res["CT full stack (learned model+costs, forced K, slow loop, beta, dyn delta)"] = sim(
        X.CT2(8, "learned", cost=X.CostModel(w, "learned", p_force=0.05), slow=X.SlowLoop(), learn_beta=True, dyn_delta=True))
    emit(test="mixed", res=res)
    # E5 switch backoff
    for b, rtt, period in [(4, 20.0, None), (4, (4.0, 20.0), 3000.0), (4, (4.0, 20.0), 30000.0), (8, (4.0, 20.0), 30000.0)]:
        w = X.World2(b=b, rtt=rtt, seed=7, n_req=max(16, 8 * b) * 3, **CAL)
        sim = lambda pol: X.simulate3(w, pol, period_ms=period)
        res = {"Target-only": sim(F.AR()),
               "CT + switch": sim(E.SpecSwitch(X.CT2(4, "oracle", cost=X.CostModel(w)))),
               "CT + switch with backoff": sim(X.SpecSwitchBackoff(X.CT2(4, "oracle", cost=X.CostModel(w))))}
        emit(test="backoff", b=b, rtt=rtt, period=period, res=res)
    # E7 snapshot library selection bias (draft ease AR(1) along positions)
    for b in (4, 8):
        w = X.World2(b=b, rtt=4.0, seed=7, draft_ar=True, **CAL)
        res = {}
        for lib in ("oracle_ar", None, "knn_nat", "knn_ipw"):
            pol = X.CT2(4, "learned", cost=X.CostModel(w), library=lib)
            res[str(lib)] = [X.simulate3(w, pol), pol.stats["probes"]]
        g = {f"{N},{t}": X.simulate3(w, X.OPT2(N, t)) for N, t in GRID}
        bk = max(g, key=g.get)
        res["best static"] = [g[bk], bk]
        emit(test="library", b=b, res=res)


if __name__ == "__main__":
    {"A": part_a, "B": part_b}[sys.argv[1]]()
