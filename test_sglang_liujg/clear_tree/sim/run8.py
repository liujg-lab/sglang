"""Round 8: does one static (W, K, tau) work across workloads, and can ClearTree recover what a mismatch loses?

Costs: the C1 fit to the real T0 grid (run7.World7, real ms, first Draft layer on the critical path).
Workloads differ only in acceptance (Draft top-1 concentration k = 1 for all):
  low  a=1.0, s in (0.30, 0.70): N=8 accept length about 2.5 (hypothetical, no real anchor)
  mid  a=3.0, s in (0.50, 0.90): about 4.0 (the setting that fits the real natural-prompt grid)
  high a=4.0, s in (0.85, 0.99): about 7.0 (real synthetic-fill prompts: 6.83)
Static policy: ext7.OPTDecoupled(W, K, tau) with the real stop rule; W x K x tau = {2,4,8} x {8,16,32} x
{0.05,0.1,0.2,0.4,0.8}. K = 32 at b = 8 (264 verify tokens) is outside the fitted range (at most 136).

Parts:
  G  full static grid per workload and b in {1, 4, 8}, seed 7                       -> results/r8G.<proc>.jsonl
  P  ClearTree core (W = 4 and 8, K in {8, 16, 32}) per workload and b, seed 7         -> results/r8P.<proc>.jsonl
  M  mixed workload (a third of the requests from each, interleaved), seed 8 (fresh requests, so the configs
     picked on seed 7 are not evaluated on their own selection sample): each workload's best static config,
     the best static config for the mix among the union of each workload's top 5, tau set per request type
     with a shared (W, K) (assumes the type is known), ClearTree core                -> results/r8M.<proc>.jsonl
usage: run8.py <G|P|M> <proc 0|1> [fit json, default results/fit7r_C1.json]"""
import glob, json, sys, time
import numpy as np
import feasibility as F
import ext as E
import ext2 as X
import ext3 as Y
import ext5 as V
import ext7 as S
import run7 as R7

WORKLOADS = {"low": (1.0, (0.30, 0.70)), "mid": (3.0, (0.50, 0.90)), "high": (4.0, (0.85, 0.99))}
GRID = [(W, K, t) for W in (2, 4, 8) for K in (8, 16, 32) for t in (0.05, 0.1, 0.2, 0.4, 0.8)]
BS = (1, 4, 8)
CT_KS = (8, 16, 32)


def make_world(fit, kind, b, seed, n_req=None):
    S.set_conc(1)
    a, s = WORKLOADS[kind]
    w = R7.World7(fit["fits"]["full"]["theta"], fit["a0"], fit["a_b"], seed=seed, b=b, rtt=4.0, n_req=n_req,
                  **R7.NOISY, **S.acc_kw(a, *s))
    for r in w.reqs:
        r["type"] = kind
    return w


def mixed_world(fit, b, seed):
    per = max(6, (8 * b + 2) // 3)
    parts = {k: make_world(fit, k, b, seed + 101 * i, n_req=per) for i, k in enumerate(WORKLOADS)}
    host = parts["mid"]
    reqs = [r for w in parts.values() for r in w.reqs]
    order = np.random.default_rng(seed + 7).permutation(len(reqs))
    host.reqs = [reqs[i] for i in order]
    return host


class OPTTyped(S.OPTDecoupled):
    """Decoupled static OPT whose tau depends on the request's workload type (shared W and K)."""

    def __init__(self, W, K, taus):
        super().__init__(W, K, 0.0)
        self.taus = taus

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        trees = [X.Tree2(world.reqs[r], j, world) for r, j in slots]
        params = (world.a, world.g, world.beta, 0.0)
        for t in trees:
            t.expand(self.W, rng, params)
        rem = [world.reqs[r]["L"] - j for r, j in slots]
        taus = [self.taus[world.reqs[r]["type"]] for r, _ in slots]
        K = self.K
        e_prev = [0.0] * len(trees)
        e_now = [float(np.sum(np.sort(t.scores)[-K:])) for t in trees]
        active, depth, layers = list(range(len(trees))), 1, []
        while active:
            active = [i for i in active
                      if e_now[i] - e_prev[i] > taus[i] and depth < world.DMAX - 1 and depth + 1 <= max(1, rem[i] - 1)]
            if not active:
                break
            for i in active:
                trees[i].expand(self.W, rng, params)
                e_prev[i] = e_now[i]
                e_now[i] = float(np.sum(np.sort(trees[i].scores)[-K:]))
            layers.append(len(active))
            depth += 1
        time_ = sum(world.real_cd(rng, n, self.W) for n in layers)
        com = [min(t.accepted(t.topk(K, False)) + 1, rem[i]) for i, t in enumerate(trees)]
        return com, time_ + world.real_rtt(rng, rtt) + world.real_T(rng, len(slots), K)


def core(w, W):
    for m in (F, E, X, V):
        m.KS = CT_KS
    return R7.core(w, W)


def sim(w, pol, width):
    w.first_W = width
    return X.simulate3(w, pol)


def key(c):
    return f"{c[0]},{c[1]},{c[2]}"


def parse(k):
    W, K, t = k.split(",")
    return int(W), int(K), float(t)


def part_G(fit, kind, b):
    w = make_world(fit, kind, b, 7)
    return {"grid": {key(c): sim(w, S.OPTDecoupled(*c), c[0]) for c in GRID}}


def part_P(fit, kind, b):
    w = make_world(fit, kind, b, 7)
    return {f"ClearTree core W={W}": sim(w, core(w, W), W) for W in (4, 8)}


def part_M(fit, b, grids):
    w = mixed_world(fit, b, 8)
    res = {"types": {k: sum(r["type"] == k for r in w.reqs) for k in WORKLOADS}}
    cand = set()
    best = {}
    for kind in WORKLOADS:
        g = grids[(kind, b)]
        top = sorted(g, key=g.get, reverse=True)
        best[kind] = top[0]
        cand.update(top[:5])
    rates = {k: sim(w, S.OPTDecoupled(*parse(k)), parse(k)[0]) for k in sorted(cand)}
    res["static"] = rates
    res["best per workload"] = best
    bm = max(rates, key=rates.get)
    W, K, _ = parse(bm)
    taus = {}
    for kind in WORKLOADS:
        g = grids[(kind, b)]
        sub = {k: v for k, v in g.items() if parse(k)[:2] == (W, K)}
        taus[kind] = parse(max(sub, key=sub.get))[2]
    res["typed tau"] = [sim(w, OPTTyped(W, K, taus), W), W, K, taus]
    for Wc in (4, 8):
        res[f"ClearTree core W={Wc}"] = sim(w, core(w, Wc), Wc)
    return res


def load_grids():
    grids = {}
    for f in sorted(glob.glob("results/r8G.*.jsonl")):
        for l in open(f):
            if l.strip():
                o = json.loads(l)
                grids[(o["kind"], o["b"])] = o["res"]["grid"]
    return grids


def main():
    part, proc = sys.argv[1], int(sys.argv[2])
    fit = json.load(open(sys.argv[3] if len(sys.argv) > 3 else "results/fit7r_C1.json"))
    if part in ("G", "P"):
        # heaviest first, alternating between the two processes
        jobs = sorted([(k, b) for k in WORKLOADS for b in BS], key=lambda j: (-j[1], j[0] != "low"))
    else:
        jobs = [(None, b) for b in sorted(BS, reverse=True)]
        grids = load_grids()
    for n, (kind, b) in enumerate(jobs):
        if n % 2 != proc:
            continue
        t0 = time.time()
        if part == "G":
            res = part_G(fit, kind, b)
        elif part == "P":
            res = part_P(fit, kind, b)
        else:
            res = part_M(fit, b, grids)
        print(json.dumps(dict(part=part, kind=kind, b=b, res=res, sec=round(time.time() - t0, 1))), flush=True)


if __name__ == "__main__":
    main()
