"""Numerical checks for ClearTree lemmas on random OPT-style trees.

Tree model: root value 1. At each kept node v, the Draft proposes C children with
values pi_v * a_u, where a_u >= 0 and sum_u a_u <= 1 (sibling constraint; optionally violated).
OPT layer construction: layer d+1 = global top-W among all children of layer d nodes.
G_K(d) = sum of top-K values among all nodes of layers 1..d (root excluded).
"""
import itertools, random

def build(rng, depth, W, C, sibling=True):
    layers = []
    prev = [(None, 1.0)]                      # (id, value) ; root
    nid = 0
    parent = {}
    val = {}
    for d in range(depth):
        cands = []
        for pid, pv in prev:
            raw = [rng.random() ** rng.choice([1, 2, 4]) for _ in range(C)]
            s = sum(raw)
            mass = rng.random() if sibling else rng.uniform(0.5, 3.0)
            for r in raw:
                a = r / s * mass
                if not sibling:
                    a = min(a, 1.0)            # still parent >= child, but sum may exceed 1
                cands.append((pid, pv * a))
        cands.sort(key=lambda t: -t[1])
        layer = []
        for pid, v in cands[:W]:
            parent[nid] = pid; val[nid] = v; layer.append((nid, v)); nid += 1
        layers.append(layer)
        prev = layer
    return layers, parent, val

def G(layers, K, d):
    vals = sorted((v for L in layers[:d] for _, v in L), reverse=True)
    return sum(vals[:K])

def check_a1(trials=20000, sibling=True, seed=0):
    rng = random.Random(seed)
    worst, bad, n = 0.0, 0, 0
    for _ in range(trials):
        depth, W, C, K = rng.randint(3, 8), rng.choice([1, 2, 4, 8]), rng.choice([2, 4, 8]), rng.choice([1, 2, 4, 8, 16])
        layers, _, _ = build(rng, depth, W, C, sibling)
        g = [G(layers, K, d) for d in range(depth + 1)]
        inc = [g[d] - g[d - 1] for d in range(1, depth + 1)]
        for a, b in zip(inc, inc[1:]):
            n += 1
            if b > a + 1e-12:
                bad += 1; worst = max(worst, b - a)
    return bad, n, worst

def best_first(parent, val, K):
    children = {}
    for u, p in parent.items():
        children.setdefault(p, []).append(u)
    frontier = list(children.get(None, []))
    chosen = []
    while frontier and len(chosen) < K:
        u = max(frontier, key=lambda x: (val[x], -x))
        frontier.remove(u); chosen.append(u); frontier += children.get(u, [])
    return sum(val[u] for u in chosen)

def brute(parent, val, K):
    nodes = list(val)
    best = 0.0
    for k in range(1, K + 1):
        for S in itertools.combinations(nodes, k):
            s = set(S)
            if all(parent[u] is None or parent[u] in s for u in S):
                best = max(best, sum(val[u] for u in S))
    return best

def check_best_first(trials=3000, seed=1):
    rng = random.Random(seed)
    bad = 0
    for _ in range(trials):
        layers, parent, val = build(rng, rng.randint(2, 4), rng.choice([1, 2, 3]), rng.choice([2, 3]), True)
        K = rng.randint(1, 6)
        if len(val) > 11:
            continue
        if abs(best_first(parent, val, K) - brute(parent, val, K)) > 1e-12:
            bad += 1
    return bad

def main():
    print("A1' with sibling constraint  (violations, pairs, worst):", check_a1(sibling=True))
    print("A1' without sibling constraint (violations, pairs, worst):", check_a1(sibling=False))
    print("best-first vs brute force mismatches:", check_best_first())


if __name__ == "__main__":
    main()
