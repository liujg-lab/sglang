"""Round 7, part B2: static OPT with decoupled width W and verify count K (ext7.OPTDecoupled), same worlds,
scenarios and seeds as run7.py. Hindsight best over W in {2, 4, 8}, K in {8, 16}, tau grid; plus the D42 switch.
usage: run7b.py <proc 0|1> <fit json> <structure: full|original>"""
import json, sys, time
import feasibility as F
import ext2 as X
import ext4 as Z
import ext7 as S
import run7 as R7
from calibrated import CAL

GRID = [(W, K, t) for W in (2, 4, 8) for K in (8, 16) for t in (0.1, 0.2, 0.4, 0.6, 0.8)]


def job(fit, structure, wkw, seed):
    if structure == "original":
        S.set_conc(6)
        w = X.World2(seed=seed, **wkw, **R7.NOISY, **CAL)
    else:
        S.set_conc(fit.get("k", 6))
        akw = S.acc_kw(fit["a"], *fit.get("s_range", CAL["s_range"]))
        w = R7.World7(fit["fits"][structure]["theta"], fit["a0"], fit["a_b"], seed=seed, **wkw, **R7.NOISY, **akw)

    def sim(pol, width):
        w.first_W = width
        return X.simulate3(w, pol)

    g = {f"{W},{K},{t}": sim(S.OPTDecoupled(W, K, t), W) for W, K, t in GRID}
    bk = max(g, key=g.get)
    W, K, t = bk.split(",")
    return {"best decoupled static": [g[bk], bk], "grid": g,
            "best decoupled static+switch": sim(Z.SpecSwitch2(S.OPTDecoupled(int(W), int(K), float(t)), **R7.SW), int(W))}


def main():
    proc, path, structure = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    fit = json.load(open(path)) if structure != "original" else {"a": CAL["a"]}
    jobs = [(s, wkw, seed) for s, wkw in R7.scenarios() for seed in (7, 8)]
    for k, (s, wkw, seed) in enumerate(jobs):
        if k % 2 != proc:
            continue
        t0 = time.time()
        res = job(fit, structure, wkw, seed)
        print(json.dumps(dict(structure=structure, fit=path, scen=s, seed=seed, res=res, sec=round(time.time() - t0, 1))), flush=True)


if __name__ == "__main__":
    main()
