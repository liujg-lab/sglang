"""Issue 1 follow-up: continue/stop rule after selecting the plan on the full pool P.

  P-only     : continue iff V_P(pi*) > 0
  P+C-always : continue iff V_C(pi*) > 0                 (C independent, same size as one P half)
  P+C-zone   : |V_P| > kappa*SE_P -> follow V_P; otherwise ask C: continue iff V_C > 0
Reports realized regret, false-continue rate (continued, true value <= 0),
false-stop rate (stopped, true optimum > 0), and how often C was consulted.
"""
import math, random, statistics
from selection_abc import dp, value, bucket

def trial(rng, se, kappa=1.0):
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
    P, C = noisy(se / math.sqrt(2)), noisy(se)
    v_opt, _ = dp(mu, cost, b, D)
    vP, plan = dp(P, cost, b, D)
    nodes = max(1, sum(bin(S).count("1") for S in plan))
    seP = se / math.sqrt(2) * math.sqrt(nodes)
    vt = value(plan, mu, cost, b) if plan else 0.0
    vC = value(plan, C, cost, b) if plan else 0.0
    rules = {
        "P-only": (vP > 0, False),
        "P+C-always": (vC > 0, True),
        "P+C-zone": ((vP > 0, False) if abs(vP) > kappa * seP else (vC > 0, True)),
    }
    out = {}
    for k, (go, used) in rules.items():
        go = go and bool(plan)
        out[k] = (v_opt - (vt if go else 0.0), go and vt <= 0, (not go) and v_opt > 0, used)
    return out

def main():
    for se in (0.05, 0.15, 0.3):
        rng = random.Random(23)
        acc = {}
        n = 800
        for _ in range(n):
            for k, v in trial(rng, se).items():
                acc.setdefault(k, []).append(v)
        print(f"se={se}")
        for k, rows in acc.items():
            print(f"  {k:11s} regret {statistics.fmean(r[0] for r in rows):.4f}  "
                  f"false-continue {sum(r[1] for r in rows)/n:.1%}  false-stop {sum(r[2] for r in rows)/n:.1%}  "
                  f"C used {sum(r[3] for r in rows)/n:.0%}")


if __name__ == "__main__":
    main()
