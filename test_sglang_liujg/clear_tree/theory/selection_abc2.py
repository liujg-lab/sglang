"""Issue 2, follow-up: compare decision rules by realized true value (go ? true value : 0).

Same instance generator as selection_abc.py. Total sampling budget is held equal where noted:
  A          : DP on pool A (se), continue iff V_A > 0
  A/B        : k-best from A, B re-plans and selects, continue iff V_B > 0
  A/B/C      : as A/B, continue iff V_C(selected fixed plan) > 0
  P/C        : DP on pooled A+B (se/sqrt2), continue iff V_C(selected fixed plan) > 0
  P/C-margin : as P/C, continue iff V_C > -z*se_C   (z=1, accounts for C being an unbiased but noisy check)
  P-only     : DP on pooled A+B, continue iff V_P > 0  (biased valuation, best selection)
"""
import math, random, statistics
from selection_abc import dp, value, bucket

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
    noisy = lambda s: [[v + rng.gauss(0, s) for v in row] for row in mu]
    A, B, C = noisy(se), noisy(se), noisy(se)
    P = [[(A[i][m] + B[i][m]) / 2 for m in range(D)] for i in range(b)]
    v_opt, _ = dp(mu, cost, b, D)
    full = (1 << b) - 1
    res = {}
    vA, pA = dp(A, cost, b, D)
    res["A"] = (pA, vA > 0)
    scored = sorted(((dp(A, cost, b, D, first=s)[0], s) for s in range(1, full + 1)), reverse=True)[:kbest]
    vB, pB = max((dp(B, cost, b, D, first=s) for _, s in scored), key=lambda t: t[0])
    res["A/B"] = (pB, vB > 0)
    res["A/B/C"] = (pB, value(pB, C, cost, b) > 0)
    vP, pP = dp(P, cost, b, D)
    nodes = sum(bin(S).count("1") for S in pP) or 1
    vC = value(pP, C, cost, b)
    res["P/C"] = (pP, vC > 0)
    res["P/C-margin"] = (pP, vC > -1.0 * se * math.sqrt(nodes))
    res["P-only"] = (pP, vP > 0)
    out = {}
    for k, (plan, go) in res.items():
        vt = value(plan, mu, cost, b) if plan else 0.0
        out[k] = v_opt - (vt if go else 0.0)
    return out, v_opt

def main():
    for se in (0.05, 0.15):
        rng = random.Random(11)
        acc, opts = {}, []
        for _ in range(1500):
            o, v = trial(rng, se)
            opts.append(v)
            for k, x in o.items():
                acc.setdefault(k, []).append(x)
        print(f"noise se={se}: mean optimal value {statistics.fmean(opts):.4f}")
        for k, xs in acc.items():
            print(f"  {k:10s} mean regret {statistics.fmean(xs):.4f}   p95 {sorted(xs)[int(.95*len(xs))]:.4f}")


if __name__ == "__main__":
    main()
