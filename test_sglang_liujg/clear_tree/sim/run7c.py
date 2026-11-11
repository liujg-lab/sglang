"""Round 7, part B3: ClearTree core with the verify count tied to the layer width (K = W, as in real OPT),
to split ClearTree's gain into "decoupled K and W" and "per-round adaptivity". C1-style world, RTT 4.
Forced K is off (one K only). usage: run7c.py <proc 0|1> <fit json>"""
import json, sys, time
import ext as E
import ext2 as X
import ext3 as Y
import ext5 as V
import ext7 as S
import run7 as R7
from calibrated import CAL


def core_coupled(w, W):
    for m in (E, X, V):
        m.KS = (W,)
    cm = Y.CostModel3(w, "learned", p_force=0.0, fd=1.15)
    cm.fd_b1 = 1.0
    return V.CT5(W, "learned", cost=cm, slow=None, n_snap=4, kreuse=10, learn_beta=True, dyn_delta=True)


def main():
    proc, path = int(sys.argv[1]), sys.argv[2]
    fit = json.load(open(path))
    S.set_conc(fit.get("k", 6))
    akw = S.acc_kw(fit["a"], *fit.get("s_range", CAL["s_range"]))
    jobs = [(s, wkw, seed) for s, wkw in R7.scenarios() if "RTT=4" in s for seed in (7, 8)]
    for k, (s, wkw, seed) in enumerate(jobs):
        if k % 2 != proc:
            continue
        t0 = time.time()
        res = {}
        for W in (4, 8):
            w = R7.World7(fit["fits"]["full"]["theta"], fit["a0"], fit["a_b"], seed=seed, **wkw, **R7.NOISY, **akw)
            w.first_W = W
            res[f"ClearTree core K=W={W}"] = X.simulate3(w, core_coupled(w, W))
        print(json.dumps(dict(fit=path, scen=s, seed=seed, res=res, sec=round(time.time() - t0, 1))), flush=True)


if __name__ == "__main__":
    main()
