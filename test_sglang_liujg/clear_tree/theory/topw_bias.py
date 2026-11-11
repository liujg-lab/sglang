"""Experiment A (synthetic): does mu-hat need to simulate the global top-W truncation?

Generator: each node v proposes C children with values pi_v * a_j, a ~ Dirichlet-like mass*share
(sibling sum <= 1). The real next layer is the global top-W of all children of ALL layer-d nodes.
Ground truth E[dG] is estimated with many fresh draws from the same generator.
  old: children only for parents in the current top-K set, no global top-W truncation
  new: children for every layer-d parent, global top-W by score, then insert into top-K
"""
import random, statistics

def draw_children(rng, pv, C):
    raw = [rng.random() ** rng.choice([1, 2, 4]) for _ in range(C)]
    s = sum(raw); mass = rng.random()
    return [pv * r / s * mass for r in raw]

def topk_sum(vals, K):
    return sum(sorted(vals, reverse=True)[:K])

def delta_g(prev_vals, layer_vals, K):
    return topk_sum(prev_vals + layer_vals, K) - topk_sum(prev_vals, K)

def one_case(rng, K, W, C, depth):
    nodes, layer = [], [1.0]
    for _ in range(depth):                     # build layers 1..depth (global top-W)
        kids = [c for p in layer for c in draw_children(rng, p, C)]
        layer = sorted(kids, reverse=True)[:W]
        nodes += layer
    cut = sorted(nodes, reverse=True)[:K]
    m = cut[-1] if len(cut) == K else 0.0
    in_top = [p for p in layer if p >= m and p in cut]
    def sim(parents, truncate, n):
        out = []
        for _ in range(n):
            kids = [c for p in parents for c in draw_children(rng, p, C)]
            if truncate:
                kids = sorted(kids, reverse=True)[:W]
            out.append(delta_g(nodes, kids, K))
        return statistics.fmean(out)
    truth = sim(layer, True, 2000)
    return sim(in_top, False, 64) - truth, sim(layer, True, 64) - truth, truth

def main():
    rng = random.Random(7)
    for W in (2, 4, 8, 16):
        old, new, tr = [], [], []
        for _ in range(300):
            o, n, t = one_case(rng, rng.choice([4, 8, 16]), W, W, rng.randint(1, 4))
            old.append(o); new.append(n); tr.append(t)
        rel = lambda xs: statistics.fmean(xs) / max(statistics.fmean(tr), 1e-12)
        print(f"W={W:2d}  mean true dG={statistics.fmean(tr):.4f}  "
              f"old bias {statistics.fmean(old):+.4f} ({rel(old):+.1%})  MAE {statistics.fmean(map(abs, old)):.4f} | "
              f"new bias {statistics.fmean(new):+.4f} ({rel(new):+.1%})  MAE {statistics.fmean(map(abs, new)):.4f}")


if __name__ == "__main__":
    main()
