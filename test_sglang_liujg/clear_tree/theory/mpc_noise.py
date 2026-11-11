"""Same policies as mpc_free_check.py, but planning uses noisy predictions of future increments
(fresh multiplicative noise each layer, as in oracles.o3). Realized value uses the truth; K* ex post
for MPC-Free, K_plan for Locked-K (locked verifies with K_plan) and also Locked-K + posthoc K*."""
import itertools, math, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpc_free_check import instance, KS

def run(rng, noise):
    b, D, Gk, c, f, VK = instance(rng)
    cost_from = lambda d, m0: sum(f(sum(1 for x in d if x > m)) + sum(c[i] for i in range(b) if d[i] > m) for m in range(m0, D))
    J = lambda K, d: sum(Gk[K][i][d[i]] for i in range(b)) - cost_from(d, 1) - VK[K]
    all_d = list(itertools.product(range(1, D + 1), repeat=b))
    joint = max(J(K, d) for K in KS for d in all_d)
    def pred(depth_now):
        P = {}
        for K in KS:
            P[K] = []
            for i in range(b):
                row = Gk[K][i][:]
                for d in range(depth_now + 1, D + 1):
                    row[d] = row[d - 1] + (Gk[K][i][d] - Gk[K][i][d - 1]) * math.exp(rng.gauss(0, noise))
                P[K].append(row)
        return P
    def best_under(P, K, depths, act, depth):
        best, arg = -math.inf, None
        for d in all_d:
            if any(d[i] != depths[i] for i in range(b) if i not in act) or any(d[i] < depth for i in act):
                continue
            v = sum(P[K][i][d[i]] for i in range(b)) - cost_from(d, depth) - VK[K]
            if v > best + 1e-12:
                best, arg = v, d
        return best, arg
    out = {}
    for name in ("locked", "free_tot"):
        depths, act, depth, Kcur, plans = [1] * b, set(range(b)), 1, None, 0
        while depth < D and act:
            P = pred(depth)
            if name == "locked" and Kcur is not None:
                K = Kcur; _, plan = best_under(P, K, depths, act, depth)
            else:
                cands = [(best_under(P, K, depths, act, depth), K) for K in KS]
                (v, plan), K = max(cands, key=lambda t: (t[0][0], t[1] == Kcur))
            plans += 1 if name == "locked" and Kcur is not None else len(KS)
            Kcur = K
            nxt = {i for i in act if plan[i] > depth}
            if not nxt:
                break
            for i in nxt:
                depths[i] += 1
            act, depth = nxt, depth + 1
        if name == "locked":
            out["locked"] = (joint - J(Kcur, depths), plans)
            out["locked+posthoc"] = (joint - max(J(K, depths) for K in KS), plans)
        else:
            out["free_tot"] = (joint - max(J(K, depths) for K in KS), plans)
    return out

def main(trials=1500, seed=7):
    for noise in (0.15, 0.3):
        rng = random.Random(seed)
        acc = {}
        for _ in range(trials):
            for k, v in run(rng, noise).items():
                acc.setdefault(k, []).append(v)
        print(f"noise={noise}")
        for k, rows in acc.items():
            xs = sorted(r[0] for r in rows)
            print(f"  {k:15s} miss {100*sum(x>1e-9 for x in xs)/len(xs):5.1f}%  mean {sum(xs)/len(xs):.4f}  "
                  f"p95 {xs[int(.95*len(xs))]:.4f}  max {xs[-1]:.4f}  K-plans solved per round {sum(r[1] for r in rows)/len(rows):.1f}")
if __name__ == "__main__":
    main()
