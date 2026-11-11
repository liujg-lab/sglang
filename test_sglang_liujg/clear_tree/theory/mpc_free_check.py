"""Proposition 1 audit (review round 10, issue 1).

Model: one K for the whole tree at verify time. Value of (K, depth vector d):
  J(K, d) = sum_i G_i^K(d_i) - sum_m cost(active at m) - VK[K]
G_i^K(d) = top-K revenue of request i's tree truncated to depth d (check_lemmas.G).

Policies under deterministic predictions (predictions == truth):
  joint     : max over (K, nested depth vectors)
  locked    : K chosen at layer 1 by full plan, then DP under that K every layer
  free_tot  : every layer, re-choose K by TOTAL value (built layers re-scored under K + best continuation)
  free_fut  : every layer, re-choose K by FUTURE-only value (ignores built layers' K-dependent revenue)
  mixed     : sum over layers of per-layer increments with a different K per layer (reviewer's objective;
              not realizable because verification uses one K) -- reported only as an upper number
"""
import itertools, math, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_lemmas import build, G

KS = [2, 4, 8]
BUCKETS = [1, 2, 4, 8]
def bucket(n): return next(b for b in BUCKETS if b >= n) if n else 0


def instance(rng):
    b, D = rng.randint(2, 3), rng.randint(3, 5)
    trees = [build(rng, D, rng.choice([2, 4]), rng.choice([2, 4]), True)[0] for _ in range(b)]
    Gk = {K: [[G(t, K, d) for d in range(D + 1)] for t in trees] for K in KS}
    c = [rng.uniform(0.0, 0.08) for _ in range(b)]
    a = rng.uniform(0.02, 0.15)
    f = lambda n: (a * {1: 1, 2: 1.15, 4: 1.35, 8: 1.7}[bucket(n)]) if n else 0.0
    VK = {K: rng.uniform(0, 0.05) * K for K in KS}
    return b, D, Gk, c, f, VK


def run(rng):
    b, D, Gk, c, f, VK = instance(rng)
    def cost_from(d, m0):
        return sum(f(sum(1 for x in d if x > m)) + sum(c[i] for i in range(b) if d[i] > m) for m in range(m0, D))
    def J(K, d):
        return sum(Gk[K][i][d[i]] for i in range(b)) - cost_from(d, 1) - VK[K]
    all_d = list(itertools.product(range(1, D + 1), repeat=b))
    joint = max(J(K, d) for K in KS for d in all_d)

    def feasible(d, depths, act, depth):
        return all(d[i] == depths[i] for i in range(b) if i not in act) and all(d[i] >= depth for i in act)

    def rollout(choose_k):
        depths, act, depth, Kcur, switches = [1] * b, set(range(b)), 1, None, 0
        while depth < D and act:
            K, plan = choose_k(depths, act, depth, Kcur)
            if Kcur is not None and K != Kcur:
                switches += 1
            Kcur = K
            nxt = {i for i in act if plan[i] > depth}
            if not nxt:
                break
            for i in nxt:
                depths[i] += 1
            act, depth = nxt, depth + 1
        realized = max(J(K, depths) for K in KS)  # K* on the final tree
        return realized, switches

    def best_plan_under(K, depths, act, depth, future_only):
        best, arg = -math.inf, None
        for d in all_d:
            if not feasible(d, depths, act, depth):
                continue
            if future_only:
                v = sum(Gk[K][i][d[i]] - Gk[K][i][depths[i]] for i in range(b)) - cost_from(d, depth) - VK[K]
            else:
                v = sum(Gk[K][i][d[i]] for i in range(b)) - cost_from(d, depth) - VK[K]
            if v > best + 1e-12:
                best, arg = v, d
        return best, arg

    def chooser(future_only, lock):
        def ch(depths, act, depth, Kcur):
            if lock and Kcur is not None:
                return Kcur, best_plan_under(Kcur, depths, act, depth, False)[1]
            cands = [(best_plan_under(K, depths, act, depth, future_only), K) for K in KS]
            top = max(v for (v, _), _ in cands)
            # tie-break: keep the current K
            for (v, p), K in cands:
                if Kcur is not None and K == Kcur and v >= top - 1e-12:
                    return K, p
            (v, p), K = max(cands, key=lambda t: t[0][0])
            return K, p
        return ch

    locked, _ = rollout(chooser(False, True))
    free_tot, sw_tot = rollout(chooser(False, False))
    free_fut, sw_fut = rollout(chooser(True, False))
    # reviewer's mixed-K objective: per-layer increments may each use a different K (not realizable)
    def mixed(d):
        tot = 0.0
        for m in range(1, D + 1):
            tot += max(sum(Gk[K][i][min(d[i], m)] - Gk[K][i][min(d[i], m - 1)] for i in range(b)) for K in KS)
        return tot - cost_from(d, 1) - min(VK.values())
    mix = max(mixed(d) for d in all_d)
    return joint, locked, free_tot, free_fut, mix, sw_tot, sw_fut


def counterexample():
    # reviewer's table: layer gains under K=5 / K=17 (one request, two layers)
    inc = {5: [10, 0], 17: [0, 10]}
    for K in (5, 17):
        print(f"  J(K={K}, two layers) = {sum(inc[K])}")
    print("  mixed per-layer K = 20 is not a value of any (K, tree): verification scores the whole tree with one K")
    print("  in this model the layer-2 increment under K is G^K(2) - G^K(1) of the SAME tree, so the table means")
    print("  G^5 = (10, 10), G^17 = (0, 10): both K give 10, the joint optimum is 10")


def main(trials=1500, seed=7):
    print("reviewer counterexample:")
    counterexample()
    rng = random.Random(seed)
    stats = {"locked": [], "free_tot": [], "free_fut": [], "mixed_gap": []}
    sw = {"free_tot": 0, "free_fut": 0}
    for _ in range(trials):
        joint, locked, ft, ff, mix, s1, s2 = run(rng)
        stats["locked"].append(joint - locked)
        stats["free_tot"].append(joint - ft)
        stats["free_fut"].append(joint - ff)
        stats["mixed_gap"].append(mix - joint)
        sw["free_tot"] += s1 > 0
        sw["free_fut"] += s2 > 0
    for k, xs in stats.items():
        bad = sum(x > 1e-9 for x in xs)
        print(f"{k:9s}: >0 in {bad}/{trials}, mean {sum(xs)/len(xs):.4f}, max {max(xs):.4f}")
    print(f"trials with a K switch: free_tot {sw['free_tot']}, free_fut {sw['free_fut']}")


if __name__ == "__main__":
    main()
