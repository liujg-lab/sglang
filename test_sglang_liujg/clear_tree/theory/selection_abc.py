"""Issue 2: does pool B still carry a winner's curse when it re-plans the continuation?

True per-request layer gains mu[i][m] (concave in m); bucketed step cost. Each pool sees
mu + N(0, se^2) independently. Plans are nested active-set sequences (synchronized depth).
  A    : DP on pool A, report V_A(plan_A)
  A/B  : candidates = k-best first steps under A; B re-plans each continuation and selects; report V_B
  A/B/C: same selection as A/B; pool C evaluates the selected fixed plan (no maximization); report V_C
Metrics: reported - true value of the chosen plan (bias), true regret, wrong continue/stop decisions.
"""
import random, statistics

BUCKETS = [1, 2, 4, 8]
def bucket(n): return next(x for x in BUCKETS if x >= n) if n else 0

def dp(mu, cost, b, D, first=None):
    """Nested-set DP. Returns (value, plan list of masks). first: force the first-step mask."""
    full = (1 << b) - 1
    V = [dict() for _ in range(D + 1)]
    choice = [dict() for _ in range(D + 1)]
    for m in range(D - 1, -1, -1):
        for S in range(full + 1):
            best, arg = 0.0, 0
            sub = S
            while True:
                if sub:
                    x = sum(mu[i][m] for i in range(b) if sub >> i & 1) - cost(sub) + (V[m + 1].get(sub, 0.0) if m + 1 < D else 0.0)
                    if x > best:
                        best, arg = x, sub
                if sub == 0:
                    break
                sub = (sub - 1) & S
            V[m][S], choice[m][S] = best, arg
    def unroll(m, S):
        plan = []
        while m < D and S:
            S = choice[m][S]
            if not S:
                break
            plan.append(S); m += 1
        return plan
    if first is None:
        return V[0][full], unroll(0, full)
    head = sum(mu[i][0] for i in range(b) if first >> i & 1) - cost(first)
    tail = V[1].get(first, 0.0) if D > 1 else 0.0
    return head + tail, [first] + unroll(1, first)

def value(plan, mu, cost, b):
    return sum(sum(mu[i][m] for i in range(b) if S >> i & 1) - cost(S) for m, S in enumerate(plan))

def trial(rng, se, kbest=8):
    b, D = rng.randint(3, 6), rng.randint(2, 4)
    mu = []
    for _ in range(b):
        x, row = rng.uniform(0.0, 1.0), []
        for _ in range(D):
            row.append(x); x *= rng.uniform(0.2, 0.9)
        mu.append(row)
    c = [rng.uniform(0, 0.25) for _ in range(b)]
    a = rng.uniform(0.1, 0.6)
    steps = sorted(rng.uniform(1.0, 1.6) for _ in range(3))
    f = {1: a, 2: a * steps[0], 4: a * steps[1], 8: a * steps[2]}
    cost = lambda S: f[bucket(bin(S).count("1"))] + sum(c[i] for i in range(b) if S >> i & 1)
    noisy = lambda: [[v + rng.gauss(0, se) for v in row] for row in mu]
    A, B, C = noisy(), noisy(), noisy()
    v_true_opt, _ = dp(mu, cost, b, D)
    vA, planA = dp(A, cost, b, D)
    full = (1 << b) - 1
    firsts = [s for s in range(1, full + 1)]
    scored = sorted(((dp(A, cost, b, D, first=s)[0], s) for s in firsts), reverse=True)[:kbest]
    best_B, plan_B = -1e9, []
    for _, s in scored:
        vb, pb = dp(B, cost, b, D, first=s)
        if vb > best_B:
            best_B, plan_B = vb, pb
    out = {}
    for name, plan, rep in [("A", planA, vA), ("A/B", plan_B, best_B), ("A/B/C", plan_B, value(plan_B, C, cost, b))]:
        vt = value(plan, mu, cost, b) if plan else 0.0
        go = rep > 0 and bool(plan)
        out[name] = (rep - vt, v_true_opt - vt, go and vt <= 0, (not go) and v_true_opt > 0)
    return out

def main():
    for se in (0.05, 0.15):
        rng = random.Random(11)
        acc = {k: [] for k in ("A", "A/B", "A/B/C")}
        for _ in range(1500):
            for k, v in trial(rng, se).items():
                acc[k].append(v)
        print(f"noise se={se}")
        for k, rows in acc.items():
            bias = statistics.fmean(r[0] for r in rows); reg = statistics.fmean(r[1] for r in rows)
            bad_go = sum(r[2] for r in rows); bad_stop = sum(r[3] for r in rows)
            print(f"  {k:6s} reported-true bias {bias:+.4f}  true regret {reg:.4f}  "
                  f"continue-but-worthless {bad_go}  stop-but-worthwhile {bad_stop}  (n={len(rows)})")


if __name__ == "__main__":
    main()
