"""Round 10: (S) batch accounting with the D42 switch, re-checked on seeds 8, 9, 10 (out of sample for every
config picked on seed 7); (W) why ClearTree with known acceptance parameters still trails at high acceptance, b=8.

Costs and workloads as round 8 (run8). Switch: ext4.SpecSwitch2 with the D42 settings (run7.SW).
Part S policies (scenarios: pure low/mid/high x b 1/4/8, mixed b 1/4/8, phased b 4/8):
  Target-only
  tuned static + switch      pure: the workload's best static on the seed-7 grid; mixed/phased: the best static for
                             the mix from round 8 (picked on seed 8)
  global static + switch     (4, 16, 0.4)
  batch accounting (+switch) W=8, K=32 at b=1; W=4 and K in {16, 32} at b>=4
  ClearTree K16 + switch     W=8 at b=1, W=4 at b>=4, cost margin 1.15 (round 9's best ClearTree in mixed loads)
Part W (high workload, b=8, seeds 7, 8, 9): static (W, K, 0.05) for W in {2, 4}, K in {16, 32}; ClearTree with
known acceptance parameters (oracle), cost margin 1.0, W in {2, 4} x K in {16, 32}, plus W=2 K=16 with known
costs; batch accounting W=2 K=16 and W=4 K=32. Each run records full-batch rounds and drain rounds separately.
usage: run10.py <S|W> <proc 0|1>"""
import json, sys, time
import numpy as np
import feasibility as F
import ext as E
import ext2 as X
import ext3 as Y
import ext4 as Z
import ext5 as V
import ext7 as S
import ext8 as N
import run7 as R7
import run8 as R8
import run9 as R9

FIT = R9.FIT
for n in (1, 4, 8):                      # planner timings measured for W = 4, 8, 16; W = 2 reuses W = 4 (upper bound)
    X.JIT.setdefault((n, 2), X.JIT[(n, 4)])


class Rec2:
    def __init__(self, inner, b):
        self.inner, self.b = inner, b
        self.full = [0, 0, 0.0, 0, 0]   # tokens, request-rounds, ms, rounds, layers
        self.drain = [0, 0, 0.0, 0, 0]

    def round(self, world, slots, rng, rtt, rnd):
        orig = world.real_cd
        cnt = [0]

        def cd(rng_, n, W):
            cnt[0] += 1
            return orig(rng_, n, W)
        world.real_cd = cd
        try:
            com, t = self.inner.round(world, slots, rng, rtt, rnd)
        finally:
            world.real_cd = orig
        acc = self.full if len(slots) == self.b else self.drain
        acc[0] += sum(com); acc[1] += len(slots); acc[2] += t; acc[3] += 1; acc[4] += cnt[0]
        return com, t

    def summary(self, rate):
        f, d = self.full, self.drain
        out = dict(rate=round(rate, 2))
        if f[3]:
            out.update(full_rate=round(1000 * f[0] / f[2], 2), full_tok_req_round=round(f[0] / f[1], 3),
                       full_ms_round=round(f[2] / f[3], 2), full_layers_round=round(f[4] / f[3], 3))
        out["drain_time_share"] = round(d[2] / (f[2] + d[2]), 3) if (f[2] + d[2]) else 0.0
        if d[3]:
            out["drain_rate"] = round(1000 * d[0] / d[2], 2)
        return out


def sim(w, pol, width):
    w.first_W = width
    return X.simulate3(w, pol)


def switch(inner):
    return Z.SpecSwitch2(inner, **R7.SW)


def ct(w, W, ks, fd, mode="learned", cost_mode="learned"):
    for m in (F, E, X, V):
        m.KS = ks
    cm = Y.CostModel3(w, cost_mode, p_force=0.03 if len(ks) > 1 and cost_mode == "learned" else 0.0, fd=fd)
    cm.fd_b1 = 1.0
    return V.CT5(W, mode, cost=cm, slow=None, n_snap=4, kreuse=10, learn_beta=True, dyn_delta=True)


def batch_wk(b):
    return [(8, 32)] if b == 1 else [(4, 16), (4, 32)]


# ---------- part S ----------
def tuned_config(grids, mixbest, scen, kind, b):
    if scen == "pure":
        g = grids[(kind, b)]
        return R8.parse(max(g, key=g.get))
    return mixbest[b]


def world_fn(scen, kind, b, seed):
    if scen == "pure":
        return lambda: R8.make_world(FIT, kind, b, seed)
    if scen == "mixed":
        return lambda: R8.mixed_world(FIT, b, seed)
    return lambda: R9.phased_world(b, seed)


def part_S(job, grids, mixbest):
    scen, kind, b, seed = job
    mk = world_fn(scen, kind, b, seed)
    out = {"Target-only": sim(mk(), F.AR(), 8)}
    c = tuned_config(grids, mixbest, scen, kind, b)
    sw = switch(S.OPTDecoupled(*c))
    out["tuned static+switch"] = [sim(mk(), sw, c[0]), list(c), round(sw.spec_time / max(1e-9, sw.total_time), 3)]
    sw = switch(S.OPTDecoupled(4, 16, 0.4))
    out["global static+switch"] = [sim(mk(), sw, 4), round(sw.spec_time / max(1e-9, sw.total_time), 3)]
    for W, K in batch_wk(b):
        out[f"batch W={W} K={K}"] = sim(mk(), N.BatchAccount(W, K), W)
        sw = switch(N.BatchAccount(W, K))
        out[f"batch W={W} K={K}+switch"] = [sim(mk(), sw, W), round(sw.spec_time / max(1e-9, sw.total_time), 3)]
    Wc = 8 if b == 1 else 4
    w = mk()
    sw = switch(ct(w, Wc, (8, 16, 32) if b == 1 else (16,), 1.15))
    out["ClearTree K16+switch"] = [sim(w, sw, Wc), round(sw.spec_time / max(1e-9, sw.total_time), 3)]
    return out


def jobs_S():
    jobs = []
    for seed in (8, 9, 10):
        jobs += [("pure", k, b, seed) for k in R8.WORKLOADS for b in R8.BS]
        jobs += [("mixed", None, b, seed) for b in R8.BS] + [("phased", None, b, seed) for b in (4, 8)]
    return sorted(jobs, key=lambda j: -j[2])


# ---------- part W ----------
def part_W(seed):
    b, out = 8, {}
    mk = lambda: R8.make_world(FIT, "high", b, seed)
    for W in (2, 4):
        for K in (16, 32):
            rec = Rec2(S.OPTDecoupled(W, K, 0.05), b)
            out[f"static ({W},{K},0.05)"] = rec.summary(sim(mk(), rec, W))
    for W in (2, 4):
        for K in (16, 32):
            w = mk()
            rec = Rec2(ct(w, W, (K,), 1.0, mode="oracle"), b)
            out[f"ClearTree oracle W={W} K={K}"] = rec.summary(sim(w, rec, W))
    w = mk()
    rec = Rec2(ct(w, 2, (16,), 1.0, mode="oracle", cost_mode="known"), b)
    out["ClearTree oracle W=2 K=16 known costs"] = rec.summary(sim(w, rec, 2))
    for W, K in ((2, 16), (4, 32)):
        rec = Rec2(N.BatchAccount(W, K), b)
        out[f"batch W={W} K={K}"] = rec.summary(sim(mk(), rec, W))
    return out


def main():
    part, proc = sys.argv[1], int(sys.argv[2])
    if part == "W":
        for n, seed in enumerate((7, 8, 9)):
            if n % 2 != proc:
                continue
            t0 = time.time()
            print(json.dumps(dict(part="W", seed=seed, res=part_W(seed), sec=round(time.time() - t0, 1))), flush=True)
        return
    grids = R8.load_grids()
    mixbest = {}
    import glob
    for f in sorted(glob.glob("results/r8M.*.jsonl")):
        for l in open(f):
            if l.strip():
                o = json.loads(l)
                st = o["res"]["static"]
                mixbest[o["b"]] = R8.parse(max(st, key=st.get))
    for n, job in enumerate(jobs_S()):
        if n % 2 != proc:
            continue
        t0 = time.time()
        res = part_S(job, grids, mixbest)
        print(json.dumps(dict(part="S", scen=job[0], kind=job[1], b=job[2], seed=job[3], res=res,
                              sec=round(time.time() - t0, 1))), flush=True)


if __name__ == "__main__":
    main()
