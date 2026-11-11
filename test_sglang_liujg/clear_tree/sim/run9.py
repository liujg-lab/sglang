"""Round 9: (D) why ClearTree core under-grows at high acceptance; (R) two simpler rules against per-workload
tuned static OPT. Costs: the C1 fit (run7.World7); workloads low / mid / high from run8.

Part D (seed 7, W=4, every workload x b in {1,4,8}): ClearTree core variants
  base       K in {8,16,32}, cost margin 1.15 (round 8 setting)
  K16 / K32  K fixed (no K choice, no forced K)
  fd1.0      K in {8,16,32}, cost margin 1.0
  K32 fd1.0  both
  oracle K32 fd1.0 (high workload only): known acceptance parameters
Each run records committed tokens per request per round, ms per round and Draft layer forwards per round. The
stop-boundary audit (ext5.audit_summary) runs for base and K32 fd1.0 in the mid and high workloads.

Part R: rules from ext8 with (W, K) picked on the round-8 grid; tau table and rho thresholds fitted offline
(rho measured on seed 9). Pure workloads on seed 7 (same requests as the round-8 grid, so per-workload best
static is in-sample, a strong reference); mixed (interleaved) on seed 8 for b in {1,4,8}; phased (a block of
low, then high, then mid requests) on seed 8 for b in {4, 8}, with the static candidates run there too.
usage: run9.py <D|R> <proc 0|1>"""
import json, sys, time
import numpy as np
import feasibility as F
import ext as E
import ext2 as X
import ext3 as Y
import ext5 as V
import ext7 as S
import ext8 as N
import run8 as R8

FIT = json.load(open("results/fit7r_C1.json"))
VARIANTS = {"base": ((8, 16, 32), 1.15, "learned"), "K16": ((16,), 1.15, "learned"), "K32": ((32,), 1.15, "learned"),
            "fd1.0": ((8, 16, 32), 1.0, "learned"), "K32 fd1.0": ((32,), 1.0, "learned")}


class Rec:
    def __init__(self, inner):
        self.inner, self.tok, self.req, self.ms, self.n, self.layers = inner, 0, 0, 0.0, 0, 0

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
        self.tok += sum(com); self.req += len(slots); self.ms += t; self.n += 1; self.layers += cnt[0]
        return com, t

    def summary(self, rate):
        return dict(rate=round(rate, 2), tok_req_round=round(self.tok / self.req, 3), ms_round=round(self.ms / self.n, 2),
                    layers_round=round(self.layers / self.n, 3))


def ct(w, ks, fd, mode, audit=False, W=4):
    for m in (F, E, X, V):
        m.KS = ks
    cm = Y.CostModel3(w, "learned", p_force=0.03 if len(ks) > 1 else 0.0, fd=fd)
    cm.fd_b1 = 1.0
    return V.CT5(W, mode, cost=cm, slow=None, n_snap=4, kreuse=10, learn_beta=True, dyn_delta=True, audit_stop=audit)


def run(w, pol, width):
    rec = Rec(pol)
    return rec.summary(R8.sim(w, rec, width))


def part_D(kind, b):
    out = {}
    variants = dict(VARIANTS)
    if kind == "high":
        variants["oracle K32 fd1.0"] = ((32,), 1.0, "oracle")
    for name, (ks, fd, mode) in variants.items():
        audit = kind in ("mid", "high") and name in ("base", "K32 fd1.0")
        w = R8.make_world(FIT, kind, b, 7)
        p = ct(w, ks, fd, mode, audit=audit)
        out[name] = run(w, p, 4)
        if audit and p.srec:
            out[name]["audit"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in V.audit_summary(p.srec).items()}
    return out


# ---------- part R ----------
def load_grid():
    return R8.load_grids()


def pick_rule(grids):
    """(W, K) whose best tau is closest to each workload's overall best on average; tau table for it."""
    pairs = sorted({R8.parse(k)[:2] for g in grids.values() for k in g})
    def frac(pair):
        out = []
        for k, g in grids.items():
            sub = [v for c, v in g.items() if R8.parse(c)[:2] == pair]
            out.append(max(sub) / max(g.values()))
        return float(np.mean(out))
    W, K = max(pairs, key=frac)
    table = {}
    for (kind, b), g in grids.items():
        sub = {c: v for c, v in g.items() if R8.parse(c)[:2] == (W, K)}
        table.setdefault(b, {})[kind] = R8.parse(max(sub, key=sub.get))[2]
    return W, K, table, frac((W, K))


def measure_rho(W, K):
    rho = {}
    for kind in R8.WORKLOADS:
        vals = []
        for tau in (0.05, 0.4, 0.8):
            w = R8.make_world(FIT, kind, 4, 9, n_req=12)
            p = N.OPTRhoTau(W, K, {}, (0, 0), fixed_tau=tau)
            R8.sim(w, p, W)
            vals.append(p.cum.rho())
        rho[kind] = vals
    m = {k: float(np.mean(v)) for k, v in rho.items()}
    th = ((m["low"] + m["mid"]) / 2, (m["mid"] + m["high"]) / 2)
    return rho, th


def phased_world(b, seed):
    w = R8.mixed_world(FIT, b, seed)
    order = {"low": 0, "high": 1, "mid": 2}
    rng = np.random.default_rng(seed + 11)
    w.reqs = sorted(w.reqs, key=lambda r: (order[r["type"]], rng.random()))
    return w


def rules(w_fn, W, K, table, th):
    out = {}
    for KK in sorted({K, 16, 32}):
        w = w_fn()
        p = N.BatchAccount(W, KK)
        out[f"batch accounting W={W} K={KK}"] = run(w, p, W)
        out[f"batch accounting W={W} K={KK}"]["stats"] = p.stats
    w = w_fn()
    p = N.OPTRhoTau(W, K, table, th)
    out[f"rho->tau W={W} K={K}"] = run(w, p, W)
    out[f"rho->tau W={W} K={K}"]["tau_share"] = {str(t): round(p.taus.count(t) / len(p.taus), 3) for t in sorted(set(p.taus))}
    out[f"rho->tau W={W} K={K}"]["rho"] = round(p.cum.rho(), 4)
    return out


def part_R_jobs():
    jobs = [("pure", k, b) for k in R8.WORKLOADS for b in R8.BS]
    jobs += [("mixed", None, b) for b in R8.BS] + [("phased", None, b) for b in (4, 8)]
    return sorted(jobs, key=lambda j: -j[2])


def part_R(job, W, K, table, th, grids):
    scen, kind, b = job
    if scen == "pure":
        return rules(lambda: R8.make_world(FIT, kind, b, 7), W, K, table, th)
    if scen == "mixed":
        out = rules(lambda: R8.mixed_world(FIT, b, 8), W, K, table, th)
        for name in ("K32 fd1.0", "K16"):
            ks, fd, mode = VARIANTS[name]
            w = R8.mixed_world(FIT, b, 8)
            out[f"ClearTree {name}"] = run(w, ct(w, ks, fd, mode), 4)
        return out
    out = rules(lambda: phased_world(b, 8), W, K, table, th)
    cands = {R8.parse(max(grids[(k, b)], key=grids[(k, b)].get)) for k in R8.WORKLOADS} | {(4, 16, 0.4), (4, 32, 0.4), (4, 16, 0.2)}
    for c in sorted(cands):
        out[f"static {c}"] = run(phased_world(b, 8), S.OPTDecoupled(*c), c[0])
    for name in ("base", "K32 fd1.0", "K16"):
        ks, fd, mode = VARIANTS[name]
        w = phased_world(b, 8)
        out[f"ClearTree {name}"] = run(w, ct(w, ks, fd, mode), 4)
    return out


def main():
    part, proc = sys.argv[1], int(sys.argv[2])
    if part == "D":
        jobs = sorted([(k, b) for k in R8.WORKLOADS for b in R8.BS], key=lambda j: (-j[1], j[0]))
        for n, (kind, b) in enumerate(jobs):
            if n % 2 != proc:
                continue
            t0 = time.time()
            res = part_D(kind, b)
            print(json.dumps(dict(part="D", kind=kind, b=b, res=res, sec=round(time.time() - t0, 1))), flush=True)
        return
    grids = load_grid()
    W, K, table, frac = pick_rule(grids)
    rho, th = measure_rho(W, K)
    if proc == 0:
        print(json.dumps(dict(part="R-setup", W=W, K=K, frac=round(frac, 4), table={str(b): t for b, t in table.items()},
                              rho={k: [round(x, 4) for x in v] for k, v in rho.items()}, thresholds=[round(x, 4) for x in th])), flush=True)
    for n, job in enumerate(part_R_jobs()):
        if n % 2 != proc:
            continue
        t0 = time.time()
        res = part_R(job, W, K, table, th, grids)
        print(json.dumps(dict(part="R", scen=job[0], kind=job[1], b=job[2], res=res, sec=round(time.time() - t0, 1))), flush=True)


if __name__ == "__main__":
    main()
