"""Three small oracles for ClearTree's batch-level decisions (Theta = 1 units).

O1 subset selection: one layer, pick the active set S maximizing sum(mu_i) - cost(S).
O2 nesting: synchronized depths, per-request concave gains, bucketed per-layer cost.
O3 K-parallel: real random trees (sibling-consistent), K buckets, joint optimum vs policies.
"""
import itertools, math, random
from check_lemmas import build, G

BUCKETS = [1, 2, 4, 8]
def bucket(n):
    return next(b for b in BUCKETS if b >= n) if n else 0

def make_f(rng, kind):
    a = rng.uniform(0.5, 2.0)
    if kind == "sub":      # batching never costs more than splitting
        steps = sorted(rng.uniform(1.0, 1.6) for _ in range(3))
        vals = {1: a, 2: a * steps[0], 4: a * steps[1], 8: a * steps[2]}
    else:                  # superadditive: a larger bucket costs more than two smaller ones
        vals = {1: a, 2: a * rng.uniform(2.2, 3.0), 4: a * rng.uniform(5, 7), 8: a * rng.uniform(11, 15)}
    return lambda n: vals[bucket(n)] if n else 0.0

# ---------------- O1 ----------------
def o1(trials=20000, seed=0):
    rng = random.Random(seed)
    stats = {"old(sort mu)": [0, 0.0], "new(sort mu-c, all n)": [0, 0.0], "enum(table)": [0, 0.0]}
    nonadd_bad = 0
    for _ in range(trials):
        b = rng.randint(2, 8)
        mu = [rng.uniform(0, 3) for _ in range(b)]
        c = [rng.uniform(0, 2.5) for _ in range(b)]
        f = make_f(rng, "sub")
        cost = lambda S: f(len(S)) + sum(c[i] for i in S)
        val = lambda S: sum(mu[i] for i in S) - cost(S)
        subsets = [S for k in range(b + 1) for S in itertools.combinations(range(b), k)]
        best = max(val(S) for S in subsets)
        order_mu = sorted(range(b), key=lambda i: -mu[i])
        old = max(val(order_mu[:n]) for n in [0] + BUCKETS + [b] if n <= b)
        order_s = sorted(range(b), key=lambda i: -(mu[i] - c[i]))
        new = max(val(order_s[:n]) for n in range(b + 1))
        for name, v in [("old(sort mu)", old), ("new(sort mu-c, all n)", new), ("enum(table)", best)]:
            if v < best - 1e-9:
                stats[name][0] += 1; stats[name][1] = max(stats[name][1], best - v)
        # non-additive cost: per-layer cost depends on total length through a convex term
        g = rng.uniform(0.0, 0.3)
        cost2 = lambda S: f(len(S)) + sum(c[i] for i in S) + g * sum(c[i] for i in S) ** 2
        val2 = lambda S: sum(mu[i] for i in S) - cost2(S)
        best2 = max(val2(S) for S in subsets)
        new2 = max(val2(order_s[:n]) for n in range(b + 1))
        nonadd_bad += new2 < best2 - 1e-9
    print("O1 subset selection, additive-in-bucket cost (misses, worst loss):", stats)
    print("O1 sorted rule under non-additive cost, misses:", nonadd_bad, "/", trials)

# ---------------- O2 ----------------
def conc_gains(rng, D):
    u, x = [], rng.uniform(0.5, 3.0)
    for _ in range(D):
        u.append(x); x *= rng.uniform(0.1, 0.95)
    return u

def value_of_depths(depths, u, c, f):
    D = max(depths) if depths else 0
    gain = sum(sum(u[i][:depths[i]]) for i in range(len(depths)))
    cost = sum(f(sum(1 for d in depths if d >= m)) + sum(c[i] for i in range(len(depths)) if depths[i] >= m)
               for m in range(1, D + 1))
    return gain - cost

def min_arrangement_cost(counts, f, b):
    """Cheapest step-size multiset for a 0/1 request x step matrix with row sums `counts` (Gale-Ryser)."""
    total = sum(counts)
    r = sorted(counts, reverse=True)
    best = math.inf
    def parts(n, maxp):
        if n == 0:
            yield []; return
        for p in range(min(n, maxp), 0, -1):
            for rest in parts(n - p, p):
                yield [p] + rest
    for s in parts(total, b):
        if all(sum(r[:k]) <= sum(min(x, k) for x in s) for k in range(1, b + 1)):
            best = min(best, sum(f(x) for x in s))
    return best

def o2(trials=3000, seed=1):
    rng = random.Random(seed)
    res = {}
    for kind in ["const", "sub", "super"]:
        greedy_bad = nest_vs_free = 0
        worst_g = 0.0
        for _ in range(trials):
            b, D = rng.randint(2, 4), rng.randint(2, 4)
            u = [conc_gains(rng, D) for _ in range(b)]
            c = [rng.uniform(0, 1.0) for _ in range(b)]
            if kind == "const":
                a = rng.uniform(0.5, 2.0); f = lambda n, a=a: a if n else 0.0
            else:
                f = make_f(rng, kind)
            vecs = list(itertools.product(range(D + 1), repeat=b))
            nested_opt = max(value_of_depths(list(d), u, c, f) for d in vecs)
            # greedy, irrevocable, with the corrected per-layer subset choice
            act, depths = list(range(b)), [0] * b
            for m in range(D):
                best_S, best_v = (), 0.0
                for k in range(1, len(act) + 1):
                    for S in itertools.combinations(act, k):
                        v = sum(u[i][m] - c[i] for i in S) - f(len(S))
                        if v > best_v + 1e-12:
                            best_S, best_v = S, v
                if not best_S:
                    break
                act = list(best_S)
                for i in act:
                    depths[i] += 1
            g = value_of_depths(depths, u, c, f)
            if g < nested_opt - 1e-9:
                greedy_bad += 1; worst_g = max(worst_g, nested_opt - g)
            # free arrangement (mixed depths / re-activation allowed): choose counts, cheapest arrangement
            free_opt = max(sum(sum(u[i][:n[i]]) for i in range(b)) - sum(c[i] * n[i] for i in range(b))
                           - min_arrangement_cost(list(n), f, b) for n in vecs)
            nest_vs_free += free_opt > nested_opt + 1e-9
        res[kind] = (greedy_bad, round(worst_g, 3), nest_vs_free, trials)
    print("O2 (greedy-irrevocable misses, worst loss, free-arrangement beats nested, trials):", res)

# ---------------- O3 ----------------
KS = [2, 4, 8]
def o3(trials=1500, seed=2, noise=0.0):
    rng = random.Random(seed)
    loss = {"old lead-K": [], "A: fixed K0 + DP": [], "MPC joint": []}
    for _ in range(trials):
        b, D = rng.randint(2, 3), rng.randint(3, 5)
        trees = [build(rng, D, rng.choice([2, 4]), rng.choice([2, 4]), True)[0] for _ in range(b)]
        Gk = {K: [[G(t, K, d) for d in range(D + 1)] for t in trees] for K in KS}
        c = [rng.uniform(0.0, 0.08) for _ in range(b)]
        a = rng.uniform(0.02, 0.15)
        f = lambda n, a=a: (a * {1: 1, 2: 1.15, 4: 1.35, 8: 1.7}[bucket(n)]) if n else 0.0
        VK = {K: rng.uniform(0, 0.05) * K for K in KS}
        def total(K, depths, Gt=Gk):
            # layer 1 comes from the previous round's logits: free. Forward m builds layer m+1.
            gain = sum(Gt[K][i][depths[i]] for i in range(b))
            cost = sum(f(sum(1 for d in depths if d > m)) + sum(c[i] for i in range(b) if depths[i] > m)
                       for m in range(1, D))
            return gain - cost - VK[K]
        all_d = list(itertools.product(range(1, D + 1), repeat=b))
        oracle = max(total(K, list(d)) for K in KS for d in all_d)
        # predictions used for planning: true values times noise on future increments
        def predicted(depth_now):
            P = {}
            for K in KS:
                P[K] = []
                for i in range(b):
                    row = Gk[K][i][:]
                    for d in range(depth_now + 1, D + 1):
                        inc = Gk[K][i][d] - Gk[K][i][d - 1]
                        row[d] = row[d - 1] + inc * math.exp(rng.gauss(0, noise)) if noise else Gk[K][i][d]
                    P[K].append(row)
            return P
        # old: per K choose n by mu-sort among active, lead K = best J + step value, irrevocable
        act, depth = list(range(b)), 1
        depths = [1] * b
        while depth < D and act:
            P = predicted(depth)
            cand = []
            for K in KS:
                mu = {i: P[K][i][depth + 1] - P[K][i][depth] for i in act}
                order = sorted(act, key=lambda i: -mu[i])
                best_n, best_v = 0, 0.0
                for n in [x for x in [1, 2, 4, 8] if x <= len(act)] + [len(act)]:
                    S = order[:n]
                    v = sum(mu[i] for i in S) - f(n) - sum(c[i] for i in S)
                    if v > best_v:
                        best_n, best_v = n, v
                J = sum(Gk[K][i][depths[i]] for i in range(b)) - VK[K]
                cand.append((J + best_v, best_n, K, order))
            score, n, K, order = max(cand)
            if n == 0:
                break
            act = order[:n]
            for i in act:
                depths[i] += 1
            depth += 1
        loss["old lead-K"].append(oracle - max(total(K, depths) for K in KS))
        # A: fixed K0 (middle bucket), DP (exhaustive here) over nested depths on predicted gains, then K* ex post
        P = predicted(1)
        K0 = 4
        plan = max(all_d, key=lambda d: sum(P[K0][i][d[i]] for i in range(b)) - sum(
            f(sum(1 for x in d if x > m)) + sum(c[i] for i in range(b) if d[i] > m) for m in range(1, D)))
        loss["A: fixed K0 + DP"].append(oracle - max(total(K, list(plan)) for K in KS))
        # MPC joint: each layer, plan over (K, nested depths) on predictions, execute first step only
        depths, act, depth = [1] * b, set(range(b)), 1
        while depth < D and act:
            P = predicted(depth)
            best, best_plan = -math.inf, None
            for K in KS:
                for d in all_d:
                    if any(d[i] != depths[i] for i in range(b) if i not in act):
                        continue
                    if any(d[i] < depth for i in act):
                        continue
                    v = sum(P[K][i][d[i]] for i in range(b)) - sum(
                        f(sum(1 for x in d if x > m)) + sum(c[i] for i in range(b) if d[i] > m)
                        for m in range(depth, D)) - VK[K]
                    if v > best:
                        best, best_plan = v, d
            nxt = {i for i in act if best_plan[i] > depth}
            if not nxt:
                break
            for i in nxt:
                depths[i] += 1
            act, depth = nxt, depth + 1
        loss["MPC joint"].append(oracle - max(total(K, depths) for K in KS))
    def summ(xs):
        xs = sorted(xs)
        return {"miss%": round(100 * sum(x > 1e-9 for x in xs) / len(xs), 1),
                "mean": round(sum(xs) / len(xs), 4), "p95": round(xs[int(0.95 * len(xs))], 4), "max": round(xs[-1], 4)}
    print(f"O3 K-parallel, prediction noise sigma={noise}:", {k: summ(v) for k, v in loss.items()})

if __name__ == "__main__":
    o1(); o2(); o3(noise=0.0); o3(noise=0.3)
