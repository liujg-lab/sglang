"""Round 15 probe: headroom of per-layer width schedules over one width per round, by paired replay.

Every round builds the tree with W=8 under the batch-accounting rule (K=32 for growth). Each option is a width per
layer (layer 1, 2, ...; the last entry repeats) and a verify count K. The tree is cut layer by layer to the schedule
(global top-W_d among children of the nodes kept one layer up), verified with the top-K nodes, and scored on the true
path (+1 bonus). Time = the round's actual time + the difference in Draft layer time (the world's own c_d, i.e.
measured costs; first layer included) + the difference in verify time (the world's T_t). Depth stays as built, so
options are compared on the same rounds at the same depths. Prints, per scenario, the best constant width, the best
schedule, and the gain of the latter. usage: diag15.py"""
import json, itertools
import numpy as np
import feasibility as F
import ext2 as X
import ext8 as N
import ext10 as P
import ext11 as O
import ext12 as RP
import run10 as T
import run13 as U
import run14 as V

SCHEDULES = [(2,), (4,), (8,), (8, 4), (8, 2), (4, 2), (8, 4, 2), (8, 8, 4), (8, 8, 2), (4, 4, 2), (8, 4, 4, 2)]
KS = (16, 32)


def cut(t, sched):
    par = RP.parents_of(t)
    by_d = {}
    for n in t.ids:
        by_d.setdefault(t.dep[n], []).append(n)
    keep, prev, widths = [], {0}, []
    for d in sorted(by_d):
        w = sched[min(d - 1, len(sched) - 1)]
        cand = sorted([n for n in by_d[d] if par.get(n) in prev], key=lambda n: t.score[n], reverse=True)[:w]
        if not cand:
            break
        keep.extend(cand); prev = set(cand); widths.append(w)
    return keep, widths


class Probe(O.BatchAccount4):
    def __init__(self, world, **kw):
        super().__init__(lambda b, c: (8, 32), verify_time=lambda bb, K: world.T_t(bb, K), **kw)
        self.w = world
        self.acc = {}

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        cap = []
        orig = world.real_cd

        def cd(r_, n, W):
            cap.append(n)
            return orig(r_, n, W)
        world.real_cd = cd
        try:
            trees_holder = []
            orig_tree = X.Tree2

            class TT(orig_tree):
                def __init__(s, *a, **k):
                    super().__init__(*a, **k); trees_holder.append(s)
            X.Tree2 = TT
            com, t_round = super().round(world, slots, rng, rtt, rnd)
        finally:
            world.real_cd = orig
            X.Tree2 = orig_tree
        if self.rounds <= self.warm:
            return com, t_round
        b = len(slots)
        trees = trees_holder[:b]
        rem = [world.reqs[r]["L"] - j for r, j in slots]
        base_layers = world.c_d(b, 8) + sum(world.c_d(m, 8) for m in cap)
        for sched in SCHEDULES:
            cuts = [cut(t, sched) for t in trees]
            # layer d (1-based) is forwarded for the requests still active; active counts per layer come from `cap`
            lt = world.c_d(b, sched[0])
            for d, m in enumerate(cap, start=2):
                lt += world.c_d(m, sched[min(d - 1, len(sched) - 1)])
            for K in KS:
                tok = sum(min(RP.accepted_in(t, RP.topk_of(t, cuts[i][0], K)) + 1, rem[i]) for i, t in enumerate(trees))
                tm = t_round + (lt - base_layers) + (world.T_t(b, K) - world.T_t(b, 32))
                a = self.acc.setdefault((sched, K), [0.0, 0.0])
                a[0] += tok; a[1] += tm
        return com, t_round


out = {}
for world_name, kind, b, vm, dm in [("base", "mid", 4, 1, 1), ("base", "mid", 8, 1, 1), ("base", "high", 4, 1, 1), ("base", "high", 8, 1, 1),
                                    ("base", "flat", 4, 1, 1), ("base", "flat", 8, 1, 1), ("base", "midlow", 4, 1, 1),
                                    ("base", "mid", 1, 1, 1), ("base", "flat", 1, 1, 1), ("verify x10", "mid", 4, 10, 1), ("draft x2", "mid", 4, 1, 2)]:
    rates = {}
    for seed in (8, 9):
        w = V.make_world(kind, b, seed, vm, dm)
        p = Probe(w)
        T.sim(w, P.SwitchTok(p, T.R7.SW), 8)
        for k, (tok, tm) in p.acc.items():
            r = rates.setdefault(k, [0.0, 0.0]); r[0] += tok; r[1] += tm
    rate = {k: 1000 * v[0] / v[1] for k, v in rates.items()}
    const = {k: v for k, v in rate.items() if len(k[0]) == 1}
    bc = max(const, key=const.get); bs = max(rate, key=rate.get)
    print(f"{world_name:10s} {kind:6s} b={b}: best constant W={bc[0][0]},K={bc[1]} {const[bc]:.1f} | best schedule {bs[0]},K={bs[1]} {rate[bs]:.1f} "
          f"({100 * (rate[bs] / const[bc] - 1):+.1f}%) | top3 " + ", ".join(f"{k[0]}K{k[1]}:{100 * (v / const[bc] - 1):+.1f}%" for k, v in sorted(rate.items(), key=lambda x: -x[1])[:3]), flush=True)
    out[f"{world_name}|{kind}|{b}"] = {f"{k[0]}|{k[1]}": v for k, v in rate.items()}
json.dump(out, open("results/d15_schedules.json", "w"), indent=1)
