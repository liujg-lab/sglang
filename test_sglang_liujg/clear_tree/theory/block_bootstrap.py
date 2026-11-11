"""E1 (synthetic): per-node bootstrap vs whole-layer block bootstrap for mu-hat.

Generator with a layer-level latent factor z (shared by all parents of one layer):
  child mass   m_p = sigmoid(a + rho*z + e_p)
  shape        conc = exp(b + rho*z + f_p): higher -> sharper top child
Parents of the current layer have known values; their children are unknown.
Truth = E[dG] over fresh draws of (z, e, f) for the current parents (one z per draw).
Pool = historical layers from the same generator (each record keeps its layer id and parent rank).
  per-node: each current parent gets the vector of a random historical parent with the same rank, from any layer
  block:    one random historical layer; parent of rank r gets that layer's rank-r vector
Values are taken as child value = parent value * share (raw score == value here).
"""
import math, random, statistics

def sig(x): return 1 / (1 + math.exp(-x))

def child_shares(rng, z, C, rho):
    m = sig(0.0 + rho * z + rng.gauss(0, 0.5))
    conc = min(math.exp(0.3 + rho * z + rng.gauss(0, 0.3)), 20.0)
    raw = sorted((max(rng.random() ** conc, 1e-12) for _ in range(C)), reverse=True)
    s = sum(raw)
    return [m * r / s for r in raw]

def layer_draw(rng, P, C, rho):
    z = rng.gauss(0, 1)
    return [child_shares(rng, z, C, rho) for _ in range(P)]   # index = parent rank

def topk(vals, K): return sum(sorted(vals, reverse=True)[:K])

def dG(prev, parents, shares, W, K):
    kids = [p * s for p, sh in zip(parents, shares) for s in sh]
    kids = sorted(kids, reverse=True)[:W]
    return topk(prev + kids, K) - topk(prev, K)

def run(rho, trials=300, W=8, C=8, K=8, pool_layers=400, n=64, seed=0):
    rng = random.Random(seed)
    pool = [layer_draw(rng, W, C, rho) for _ in range(pool_layers)]
    eb_node, eb_block = [], []
    for _ in range(trials):
        parents = sorted((rng.uniform(0.05, 0.6) for _ in range(W)), reverse=True)
        prev = parents + [rng.uniform(0.3, 0.9) for _ in range(rng.randint(0, 6))]
        truth = statistics.fmean(dG(prev, parents, layer_draw(rng, W, C, rho), W, K) for _ in range(1500))
        node = statistics.fmean(
            dG(prev, parents, [rng.choice(pool)[r] for r in range(W)], W, K) for _ in range(n))
        block = statistics.fmean(dG(prev, parents, rng.choice(pool), W, K) for _ in range(n))
        eb_node.append((node - truth, truth)); eb_block.append((block - truth, truth))
    def rep(xs):
        mt = statistics.fmean(t for _, t in xs)
        b = statistics.fmean(e for e, _ in xs)
        return f"bias {b:+.4f} ({b / mt:+.1%})  MAE {statistics.fmean(abs(e) for e, _ in xs):.4f}"
    print(f"rho={rho}: per-node {rep(eb_node)} | block {rep(eb_block)}")

def main():
    for rho in (0.0, 0.5, 1.0, 2.0):
        run(rho)


if __name__ == "__main__":
    main()
