"""Selection optimism: subtract an estimated bias vs inflate the SE (review round 10, issue 3.3).

Reuses boundary_rule's generator: plan chosen by DP on pool P (winner's curse), C pool independent.
b_sel(bucket) is estimated on separate C_mon trials as mean(V_P - V_C), bucketed by b (number of requests).
Rules (all ask C_dec inside their zone, follow the stated estimate outside):
  zone      : current, |V_P| <= kappa*SE_P
  inflate   : |V_P| <= kappa*sqrt(SE_P^2 + b^2), outside follow sign(V_P)
  subtract  : |V_P - b| <= kappa*SE_P, outside follow sign(V_P - b)
"""
import math, random, statistics, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selection_abc import dp, value, bucket


def gen(rng, se):
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
    return b, v_opt, vP, seP, vt, vC, bool(plan)


def main(kappa=1.0):
    for se in (0.15, 0.3):
        rng = random.Random(41)
        mon, monz = {}, {}
        zb = lambda vP, seP: min(3, int(abs(vP) / seP))
        for _ in range(6000):
            b, _, vP, seP, _, vC, has = gen(rng, se)
            if has:
                mon.setdefault(b, []).append(vP - vC)
                monz.setdefault((b, zb(vP, seP)), []).append(vP - vC)
        bias = {b: statistics.fmean(v) for b, v in mon.items()}
        biasz = {k: statistics.fmean(v) for k, v in monz.items() if len(v) >= 20}
        rng = random.Random(43)
        n = 1500
        acc = {"zone": [], "inflate": [], "subtract": [], "subtract_cond": []}
        for _ in range(n):
            b, v_opt, vP, seP, vt, vC, has = gen(rng, se)
            bb = bias.get(b, 0.0)
            bz = biasz.get((b, zb(vP, seP)), bb) if has else 0.0
            rules = {
                "zone": (abs(vP) <= kappa * seP, vP > 0),
                "inflate": (abs(vP) <= kappa * math.sqrt(seP ** 2 + bb ** 2), vP > 0),
                "subtract": (abs(vP - bb) <= kappa * seP, vP - bb > 0),
                "subtract_cond": (abs(vP - bz) <= kappa * seP, vP - bz > 0),
            }
            for k, (in_zone, go_p) in rules.items():
                go = (vC > 0) if in_zone else go_p
                go = go and has
                acc[k].append((v_opt - (vt if go else 0.0), go and vt <= 0, (not go) and v_opt > 0, in_zone))
        print(f"se={se}  b_sel by b: " + ", ".join(f"{k}:{v:+.3f}" for k, v in sorted(bias.items())))
        print("   b_sel by (b, |V_P|/SE_P bin): " + ", ".join(f"{k}:{v:+.3f}" for k, v in sorted(biasz.items())))
        for k, rows in acc.items():
            print(f"  {k:9s} regret {statistics.fmean(r[0] for r in rows):.4f}  false-continue {sum(r[1] for r in rows)/n:.1%}"
                  f"  false-stop {sum(r[2] for r in rows)/n:.1%}  C used {sum(r[3] for r in rows)/n:.0%}")


if __name__ == "__main__":
    main()
