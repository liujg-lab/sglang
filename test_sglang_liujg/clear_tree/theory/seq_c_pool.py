"""Sequential C_dec pool (review round 10, issue 3.1).

Boundary-zone decision: true value V (continue minus stop). C samples y ~ N(V, s1^2).
Looks after n = 4, 8, ..., 32 samples. Rules:
  z2_est   : current rule, stop when |mean| > 2 * sd_hat/sqrt(n); at the cap act on sign(mean)
  z2_known : same, with per-sample sd from history (known s1)
  pocock   : constant boundary calibrated so P(any early stop | V=0) = 5% for the 8 looks (known s1)
  fixed32  : always 32 samples, act on sign
  bayes    : prior V ~ N(m0, s0^2) from the P pool; act on sign of posterior mean;
             stop when one-step expected regret reduction < cost of 4 samples
Metrics: early-stop error at V=0 (frequentist), wrong-sign rate for |V| >= s1/sqrt(32),
mean regret |V|*1[wrong], mean samples, regret + lam * samples.
"""
import math, random, statistics

LOOKS = list(range(4, 33, 4))
PHI = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
pdf = lambda x: math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def exp_regret(m, s):
    """E[regret of acting on sign(m)] when V ~ N(m, s^2): E[max(-sign(m) V, 0)]."""
    if s <= 0:
        return 0.0
    a = abs(m) / s
    return s * (pdf(a) - a * (1 - PHI(a)))


def calibrate_pocock(alpha=0.05, sims=200000, seed=1):
    rng = random.Random(seed)
    maxz = []
    for _ in range(sims):
        tot, n, mz = 0.0, 0, 0.0
        for look in LOOKS:
            while n < look:
                tot += rng.gauss(0, 1); n += 1
            mz = max(mz, abs(tot / n) * math.sqrt(n))
        maxz.append(mz)
    maxz.sort()
    return maxz[int((1 - alpha) * sims)]


def run_rule(rule, V, s1, rng, c_pock, m0, s0, lam):
    ys = []
    for look in LOOKS:
        while len(ys) < look:
            ys.append(rng.gauss(V, s1))
        n = len(ys)
        mean = sum(ys) / n
        if rule == "fixed32":
            continue
        if rule == "bayes":
            prec = 1 / s0 ** 2 + n / s1 ** 2
            pm, ps = (m0 / s0 ** 2 + n * mean / s1 ** 2) / prec, math.sqrt(1 / prec)
            now = exp_regret(pm, ps)
            # preposterior: after 4 more samples, posterior sd shrinks to ps2; expected regret ~ E over pm2
            prec2 = prec + 4 / s1 ** 2
            ps2 = math.sqrt(1 / prec2)
            spread = math.sqrt(max(ps ** 2 - ps2 ** 2, 0))
            later = sum(exp_regret(pm + spread * z, ps2) * w for z, w in GH) if n < 32 else now
            if now - later < lam * 4:
                return (pm > 0), n, True
            continue
        sd = statistics.stdev(ys) if rule == "z2_est" else s1
        z = abs(mean) / (sd / math.sqrt(n))
        thr = c_pock if rule == "pocock" else 2.0
        if z > thr:
            return (mean > 0), n, True
    mean = sum(ys) / len(ys)
    if rule == "bayes":
        prec = 1 / s0 ** 2 + len(ys) / s1 ** 2
        mean = (m0 / s0 ** 2 + len(ys) * mean / s1 ** 2) / prec
    return (mean > 0), len(ys), False

# Gauss-Hermite (probabilists') 5-point nodes/weights for E[f(Z)], Z ~ N(0,1)
GH = [(-2.8570, 0.01126), (-1.3556, 0.22208), (0.0, 0.53333), (1.3556, 0.22208), (2.8570, 0.01126)]


def main(seed=3, n_dec=20000):
    c_pock = calibrate_pocock()
    print(f"Pocock boundary for looks {LOOKS}: {c_pock:.3f} (alpha 5% two-sided, known sd)")
    s1 = 1.0
    rules = ["z2_est", "z2_known", "pocock", "fixed32", "bayes"]
    # frequentist error at V = 0 (early stops with either sign are 'decisions made with confidence')
    rng = random.Random(seed)
    print("early stop rate at V=0 (nominal 4.6% for one look at z=2):")
    for r in rules[:3]:
        k = sum(run_rule(r, 0.0, s1, rng, c_pock, 0, 1, 0)[2] for _ in range(20000))
        print(f"  {r:9s} {k/20000:.1%}")
    for s0_ratio in (0.15, 0.4):
        s0 = s0_ratio * s1  # spread of true V among boundary-zone decisions
        for lam in (0.0005, 0.002):
            rng = random.Random(seed)
            res = {r: [] for r in rules}
            for _ in range(n_dec):
                V = rng.gauss(0, s0)
                for r in rules:
                    go, n, _ = run_rule(r, V, s1, rng, c_pock, 0.0, s0, lam)
                    wrong = (go and V < 0) or ((not go) and V > 0)
                    res[r].append((abs(V) * wrong, n, wrong, abs(V) >= s1 / math.sqrt(32)))
            print(f"\nboundary spread s0={s0_ratio}*s1, sample cost lam={lam} per sample (regret units)")
            for r in rules:
                rows = res[r]
                reg = statistics.fmean(x[0] for x in rows)
                ns = statistics.fmean(x[1] for x in rows)
                big = [x for x in rows if x[3]]
                wr = sum(x[2] for x in big) / max(1, len(big))
                print(f"  {r:9s} regret {reg:.4f}  samples {ns:5.1f}  regret+lam*n {reg + lam*ns:.4f}  "
                      f"wrong sign when |V|>=s1/sqrt(32): {wr:.1%}")


if __name__ == "__main__":
    main()
