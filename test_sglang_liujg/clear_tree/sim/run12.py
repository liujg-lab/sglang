"""Round 12 (review 14): estimator dependence on tree shape, and the v10 fixes on seeds 8-10.

Part R (seed 9, b=4, 12 requests): static decoupled OPT for every workload x (W, K) in {(2,16),(4,16),(4,32),(8,32)}
  x tau in {0.05, 0.4, 0.8}; per run the old rho (stops inside the tree count as failures) and the opportunity-based
  rho (a stop counts only if an edge out of the last accepted node was verified). The high / other threshold of the
  new estimator is the midpoint of the mid and high workloads' means.                -> results/r12R.json
Part S (seeds 8-10, run10's 14 scenarios; reference = round 11's tuned static with the switch off at b=1):
  v11 repro           BatchAccount3 with every option off (must equal round 11's simplified version)
  +rho opp            opportunity-based rho, binary high / other
  +c decay+freeze     calibration sums decay 0.95 per round; costs, prices and c_i fixed within a round
  Theta global+reset  one global price whose EMA step resets to 1/n after b changes (instead of Theta per b)
  K<=16 table         (8,16) at b=1, (4,16) at b 2-5, (4,16) / high (2,16) at b>=6 (inside the measured range)
  switch tok b=1      conservative switch at b=1 (50-round blocks, two blocks losing > 20%) instead of none
  v12 all             +rho opp, +c decay+freeze, switch tok b=1 (K=32 table); audits 15% of layer decisions
  v12 all K<=16       the same with the K<=16 table
usage: run12.py <R|S> <proc 0|1>"""
import glob, json, sys, time
import numpy as np
import ext2 as X
import ext7 as S
import ext8 as N
import ext9 as Q
import ext10 as P
import run8 as R8
import run9 as R9
import run10 as T
import run11b as RB

FIT = R9.FIT


def wk_k16(b, cls):
    k = Q.bkey(b)
    if k == 1:
        return (8, 16)
    return (2, 16) if (k == 8 and cls == "high") else (4, 16)


class OPTRho(S.OPTDecoupled):
    def __init__(self, W, K, tau):
        super().__init__(W, K, tau)
        self.on = self.od = self.pd = 0

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        trees = [X.Tree2(world.reqs[r], j, world) for r, j in slots]
        params = (world.a, world.g, world.beta, 0.0)
        for t in trees:
            t.expand(self.W, rng, params)
        rem = [world.reqs[r]["L"] - j for r, j in slots]
        layers = S.grow_real_k(world, trees, self.W, self.K, self.tau, rem, rng)
        time_ = sum(world.real_cd(rng, n, self.W) for n in layers)
        com = []
        for i, t in enumerate(trees):
            a, tested = P.last_node(t, t.topk(self.K, False))
            self.on += a
            self.od += a + (1 if a < N._tree_depth(t) else 0)
            self.pd += a + (1 if tested else 0)
            com.append(min(a + 1, rem[i]))
        return com, time_ + world.real_rtt(rng, rtt) + world.real_T(rng, len(slots), self.K)


def part_R():
    out = {}
    for kind in R8.WORKLOADS:
        for W, K in ((2, 16), (4, 16), (4, 32), (8, 32)):
            for tau in (0.05, 0.4, 0.8):
                p = OPTRho(W, K, tau)
                T.sim(R8.make_world(FIT, kind, 4, 9, n_req=12), p, W)
                out[f"{kind}|{W},{K},{tau}"] = [p.on / p.od, p.on / p.pd]
    m = {k: [np.mean([v[j] for kk, v in out.items() if kk.startswith(k + "|")]) for j in (0, 1)] for k in R8.WORKLOADS}
    th = (m["mid"][1] + m["high"][1]) / 2
    return dict(rho=out, means=m, th_opp=th)


def part_S(job, th):
    scen, kind, b, seed = job
    mk = T.world_fn(scen, kind, b, seed)
    W0 = 8 if b == 1 else 4
    common = dict(rho_mode="opp", th_high=th, c_decay=0.95, freeze=True)
    variants = {
        "v11 repro": (RB.wk_min, {}, "b1off"),
        "+rho opp": (RB.wk_min, dict(rho_mode="opp", th_high=th), "b1off"),
        "+c decay+freeze": (RB.wk_min, dict(c_decay=0.95, freeze=True), "b1off"),
        "Theta global+reset": (RB.wk_min, dict(theta_mode="global_reset"), "b1off"),
        "K<=16 table": (wk_k16, {}, "b1off"),
        "switch tok b=1": (RB.wk_min, {}, "tok"),
        "v12 all": (RB.wk_min, dict(common, audit_p=0.15), "tok"),
        "v12 all K<=16": (wk_k16, common, "tok"),
    }
    out = {}
    for name, (wk, kw, sw) in variants.items():
        p = P.BatchAccount3(wk, **kw)
        pol = Q.SwitchB1(p, **T.R7.SW) if sw == "b1off" else P.SwitchTok(p, T.R7.SW)
        rate = T.sim(mk(), pol, W0)
        rec = {"rate": rate, "spec_share": round(pol.spec_time / max(1e-9, pol.total_time), 3),
               "wk": {f"{k[0]},{k[1]}": v for k, v in p.wk_log.items()}}
        if p.audit_p:
            rec["audit"] = p.audit
        out[name] = rec
    return out


def main():
    part, proc = sys.argv[1], int(sys.argv[2])
    if part == "R":
        if proc == 0:
            json.dump(part_R(), open("results/r12R.json", "w"), indent=1)
        return
    th = json.load(open("results/r12R.json"))["th_opp"]
    for n, job in enumerate(T.jobs_S()):
        if n % 2 != proc:
            continue
        t0 = time.time()
        res = part_S(job, th)
        print(json.dumps(dict(scen=job[0], kind=job[1], b=job[2], seed=job[3], res=res, sec=round(time.time() - t0, 1))), flush=True)


if __name__ == "__main__":
    main()
