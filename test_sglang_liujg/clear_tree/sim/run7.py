"""Round 7, part B: re-evaluate ClearTree core in worlds whose costs were fitted to the real T0 grid (fit7.py).

World7 uses real milliseconds:
  Target verify + per-round host = c0 + c_b * b + v_tok * b * (K + 1)   [+ the first Draft layer, see below]
  Draft layer (n active requests, width W) = d0 + d_req * bucket(n) + d_tok * bucket(n) * W
  Target-only step = a0 + a_b * b
The fitted data had lookahead and KV reuse off, so the first Draft layer is on the critical path. It is
added to the Target wait with the policy's width (N for OPT, W for ClearTree), which keeps it visible to
ClearTree's learned wait table. Every policy pays the same costs.

OPT uses the real stop rule (ext7.OPTReal: grow while the top-N acceptance mass rose by more than tau).
Policies: Target-only; production rule (OPT(8, 0.8), Target-only at b >= 7); every static OPT(N, tau) on a
grid (hindsight best is reported) under the real rule, and the rounds 1-6 grid under the old rule (X.OPT2)
for comparison; hindsight-best static + D42 switch; ClearTree core with W = 4 and 8 (the better one stands
for the offline per-context W of D38), with and without the D42 switch.
Structure "original" reruns the rounds 1-6 world (CAL acceptance and costs, layer 1 free) with these baselines.
usage: run7.py <proc 0|1> <fit json> <structure: token-only|full|original>"""
import json, sys, time
import numpy as np
import feasibility as F
import ext as E
import ext2 as X
import ext4 as Z
import ext5 as V
import ext7 as S
import run3b as R
from calibrated import CAL

GRID = [(N, t) for N in (2, 4, 8, 16) for t in (0.1, 0.2, 0.4, 0.6, 0.8)]
GRID_OLD = [(N, t) for N in (4, 8, 16) for t in (0.9, 0.6, 0.4, 0.2)]          # rounds 1-6 grid, old rule
NOISY = dict(noise=0.05, jitter=1.0)
SW = dict(probe_rounds=8, share=0.05, exit_blocks=2, exit_fast=0.15)   # D42


class World7(X.World2):
    def __init__(self, theta, a0, a_b, first_layer=True, **kw):
        super().__init__(**kw)
        self.c0, self.c_b, self.v_tok, self.d0, self.d_req, self.d_tok = theta
        self.a0, self.a_b = a0, a_b
        self.first_layer, self.first_W = first_layer, 8

    def c_d(self, n, W):
        bn = F.bucket(n)
        return self.d0 + self.d_req * bn + self.d_tok * bn * W

    def T_t(self, b, K):
        t = self.c0 + self.c_b * b + self.v_tok * b * (K + 1)
        return t + (self.c_d(b, self.first_W) if self.first_layer else 0.0)

    def T_ar(self, b):
        return self.a0 + self.a_b * b


def core(w, W):
    return V.CT5(W, "learned", cost=R.cm_(w, 1.15), slow=None, n_snap=4, kreuse=10, learn_beta=True, dyn_delta=True)


def scenarios():
    out = [(f"b={b} RTT=4", dict(b=b, rtt=4.0)) for b in (1, 2, 4, 6, 8)]
    out += [(f"b={b} RTT=20", dict(b=b, rtt=20.0)) for b in (1, 4)]
    return out


def job(fit, structure, scen, wkw, seed):
    if structure == "original":
        S.set_conc(6)
        w = X.World2(seed=seed, **wkw, **NOISY, **CAL)
        w.first_W = 8
    else:
        S.set_conc(fit.get("k", 6))
        th = fit["fits"][structure]["theta"]
        akw = S.acc_kw(fit["a"], *fit.get("s_range", CAL["s_range"]))
        w = World7(th, fit["a0"], fit["a_b"], seed=seed, **wkw, **NOISY, **akw)

    def sim(pol, width):
        w.first_W = width
        return X.simulate3(w, pol)

    res = {"Target-only": sim(F.AR(), 8),
           "production OPT(8,0.8)+b>=7 Target-only": sim(E.StaticSwitch(S.OPTReal(8, 0.8)), 8)}
    go = {f"{N},{t}": sim(X.OPT2(N, t), N) for N, t in GRID_OLD}
    bo = max(go, key=go.get)
    res["best static, old rule"] = [go[bo], bo]
    g = {f"{N},{t}": sim(S.OPTReal(N, t), N) for N, t in GRID}
    bk = max(g, key=g.get)
    N, t = int(bk.split(",")[0]), float(bk.split(",")[1])
    res["best static"] = [g[bk], bk]
    res["grid"] = g
    res["best static+switch"] = sim(Z.SpecSwitch2(S.OPTReal(N, t), **SW), N)
    ct = {W: sim(core(w, W), W) for W in (4, 8)}
    bw = max(ct, key=ct.get)
    res["ClearTree core W=4"], res["ClearTree core W=8"] = ct[4], ct[8]
    res["ClearTree core+switch (best W)"] = [sim(Z.SpecSwitch2(core(w, bw), **SW), bw), bw]
    return res


def main():
    proc, path, structure = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    fit = json.load(open(path)) if structure != "original" else {"a": CAL["a"]}
    jobs = [(s, wkw, seed) for s, wkw in scenarios() for seed in (7, 8)]
    for k, (s, wkw, seed) in enumerate(jobs):
        if k % 2 != proc:
            continue
        t0 = time.time()
        res = job(fit, structure, s, wkw, seed)
        print(json.dumps(dict(structure=structure, fit=path, a=fit["a"], k=fit.get("k", 6), scen=s, seed=seed, res=res,
                              sec=round(time.time() - t0, 1))), flush=True)


if __name__ == "__main__":
    main()
