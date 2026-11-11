import itertools, math, random
import oracles as O
from check_lemmas import build, G

def o2_magnitude(trials=1500, seed=3):
    rng = random.Random(seed)
    gaps, rel = [], []
    for _ in range(trials):
        b, D = rng.randint(2, 4), rng.randint(2, 4)
        u = [O.conc_gains(rng, D) for _ in range(b)]
        c = [rng.uniform(0, 1.0) for _ in range(b)]
        f = O.make_f(rng, "sub")
        vecs = list(itertools.product(range(D + 1), repeat=b))
        nested = max(O.value_of_depths(list(d), u, c, f) for d in vecs)
        free = max(sum(sum(u[i][:n[i]]) for i in range(b)) - sum(c[i] * n[i] for i in range(b))
                   - O.min_arrangement_cost(list(n), f, b) for n in vecs)
        gaps.append(free - nested)
        rel.append((free - nested) / free if free > 1e-9 else 0.0)
    pos = [g for g in gaps if g > 1e-9]
    relpos = sorted(r for r in rel if r > 1e-9)
    print(f"O2-sub free>nested: {len(pos)}/{trials}; relative gap among those: "
          f"median {relpos[len(relpos)//2]:.3%}, max {relpos[-1]:.3%}; mean over all trials {sum(rel)/len(rel):.4%}")

def o3_open_loop(trials=1500, seed=2, noise=0.0):
    rng = random.Random(seed)
    losses = []
    KS = O.KS
    for _ in range(trials):
        b, D = rng.randint(2, 3), rng.randint(3, 5)
        trees = [build(rng, D, rng.choice([2, 4]), rng.choice([2, 4]), True)[0] for _ in range(b)]
        Gk = {K: [[G(t, K, d) for d in range(D + 1)] for t in trees] for K in KS}
        c = [rng.uniform(0.0, 0.08) for _ in range(b)]
        a = rng.uniform(0.02, 0.15)
        f = lambda n, a=a: (a * {1: 1, 2: 1.15, 4: 1.35, 8: 1.7}[O.bucket(n)]) if n else 0.0
        VK = {K: rng.uniform(0, 0.05) * K for K in KS}
        def cost(d):
            return sum(f(sum(1 for x in d if x > m)) + sum(c[i] for i in range(b) if d[i] > m) for m in range(1, D))
        all_d = list(itertools.product(range(1, D + 1), repeat=b))
        oracle = max(sum(Gk[K][i][d[i]] for i in range(b)) - cost(d) - VK[K] for K in KS for d in all_d)
        P = {}
        for K in KS:
            P[K] = []
            for i in range(b):
                row = Gk[K][i][:]
                for d in range(2, D + 1):
                    inc = Gk[K][i][d] - Gk[K][i][d - 1]
                    row[d] = row[d - 1] + inc * (math.exp(rng.gauss(0, noise)) if noise else 1.0)
                P[K].append(row)
        _, K0, plan = max((sum(P[K][i][d[i]] for i in range(b)) - cost(d) - VK[K], K, d) for K in KS for d in all_d)
        losses.append(oracle - max(sum(Gk[K][i][plan[i]] for i in range(b)) - cost(plan) - VK[K] for K in KS))
    xs = sorted(losses)
    print(f"O3 A' (plan all K at round start, then fixed), noise={noise}: miss {100*sum(x>1e-9 for x in xs)/len(xs):.1f}%, "
          f"mean {sum(xs)/len(xs):.4f}, p95 {xs[int(.95*len(xs))]:.4f}, max {xs[-1]:.4f}")

def main():
    o2_magnitude(); o3_open_loop(noise=0.0); o3_open_loop(noise=0.3)


if __name__ == "__main__":
    main()
