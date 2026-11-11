"""Round 7 helpers: the real OPT stop rule, an adjustable Draft confidence generator, acceptance presets.

Real OPT_TREE_REMOTE (opt_tree_ops.py depth_rule): E is the sum of the top-N path probabilities over every
layer added so far (E_0 = 0). After layer k is added, the Draft forwards it and expands layer k+1 only while
E_k - E_{k-1} > tau and k < max_depth - 1. The verify set is the top-N nodes over all layers. Rounds 1-6 used
a different rule (grow while the best frontier path score >= tau, X.OPT2); OPTReal reproduces the real one.
"""
import numpy as np
import feasibility as F
import ext2 as X
from calibrated import CAL

_GEN0 = F.gen_dists


def set_conc(k):
    """Swap the Draft distribution generator for one with top-1 concentration k (k = 6 is the original)."""
    def gen(rng, s, n):
        q1 = rng.beta(max(1e-3, s * k), max(1e-3, (1 - s) * k), n)
        rho = rng.uniform(0.5, 0.95, n)
        g = rng.uniform(0.4, 0.8, n)
        pw = g[:, None] ** np.arange(F.C - 1)[None, :]
        rest = pw / pw.sum(1, keepdims=True) * ((1 - q1) * rho)[:, None]
        Q = np.concatenate([q1[:, None], rest], 1)
        return -np.sort(-Q, 1)
    F.gen_dists = _GEN0 if k == 6 else gen


def acc_kw(a, s_lo, s_hi):
    return dict(CAL, a=a, s_range=(s_lo, s_hi))


def grow_real(world, trees, N, tau, rem, rng):
    """Layer 1 is already expanded. Returns the active count of every later Draft layer forward."""
    params = (world.a, world.g, world.beta, 0.0)
    e_prev = [0.0] * len(trees)
    e_now = [float(np.sum(np.sort(t.scores)[-N:])) for t in trees]
    active, depth, layers = list(range(len(trees))), 1, []
    while active:
        active = [i for i in active
                  if e_now[i] - e_prev[i] > tau and depth < world.DMAX - 1 and depth + 1 <= max(1, rem[i] - 1)]
        if not active:
            break
        for i in active:
            trees[i].expand(N, rng, params)
            e_prev[i] = e_now[i]
            e_now[i] = float(np.sum(np.sort(trees[i].scores)[-N:]))
        layers.append(len(active))
        depth += 1
    return layers


class OPTReal(X.OPT2):
    """OPT(N, tau) with the real stop rule; same cost accounting as X.OPT2 (layer 1 rides on the wait)."""

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        trees = [X.Tree2(world.reqs[r], j, world) for r, j in slots]
        if trees_out is not None:
            trees_out.extend(trees)
        params = (world.a, world.g, world.beta, 0.0)
        for t in trees:
            t.expand(self.N, rng, params)
        rem = [world.reqs[r]["L"] - j for r, j in slots]
        layers = grow_real(world, trees, self.N, self.tau, rem, rng)
        time_ = sum(world.real_cd(rng, n, self.N) for n in layers)
        com = [min(t.accepted(t.topk(self.N, False)) + 1, rem[i]) for i, t in enumerate(trees)]
        return com, time_ + world.real_rtt(rng, rtt) + world.real_T(rng, len(slots), self.N)


def grow_real_k(world, trees, W, K, tau, rem, rng):
    """Like grow_real, but layers have width W and E is the top-K mass (K = verify count, may exceed W)."""
    params = (world.a, world.g, world.beta, 0.0)
    e_prev = [0.0] * len(trees)
    e_now = [float(np.sum(np.sort(t.scores)[-K:])) for t in trees]
    active, depth, layers = list(range(len(trees))), 1, []
    while active:
        active = [i for i in active
                  if e_now[i] - e_prev[i] > tau and depth < world.DMAX - 1 and depth + 1 <= max(1, rem[i] - 1)]
        if not active:
            break
        for i in active:
            trees[i].expand(W, rng, params)
            e_prev[i] = e_now[i]
            e_now[i] = float(np.sum(np.sort(trees[i].scores)[-K:]))
        layers.append(len(active))
        depth += 1
    return layers


class OPTDecoupled(X.OPT2):
    """Static OPT with layer width W and verify count K >= W (top-K by Draft path score over all layers).
    Same stop rule as the real OPT with E taken over the top-K nodes. Real OPT is the case K = W = N."""

    def __init__(self, W, K, tau):
        super().__init__(W, tau)
        self.W, self.K = W, K

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        trees = [X.Tree2(world.reqs[r], j, world) for r, j in slots]
        params = (world.a, world.g, world.beta, 0.0)
        for t in trees:
            t.expand(self.W, rng, params)
        rem = [world.reqs[r]["L"] - j for r, j in slots]
        layers = grow_real_k(world, trees, self.W, self.K, self.tau, rem, rng)
        time_ = sum(world.real_cd(rng, n, self.W) for n in layers)
        com = [min(t.accepted(t.topk(self.K, False)) + 1, rem[i]) for i, t in enumerate(trees)]
        return com, time_ + world.real_rtt(rng, rtt) + world.real_T(rng, len(slots), self.K)
