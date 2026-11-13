"""Round 31, D71 check: exponential forgetting in the pooled verify regression and a K that is never chosen.
usage (conda sglang env): python check31_rls.py

Verify regression of the simulator (ext18._RLS, forgetting 0.999 per sample, features [1{K=8}, 1{K=16}, 1{K=32}, b,
sum l / 1000, (K+1) sum l / 17000]). 6 probe samples per K, then only K = 32 is chosen. Compared: "pooled" (every
sample decays all statistics; the simulator through round 31) and "per K" (D71: the statistics of each K are kept
apart and decay only when that K gets a sample; solved on their sum). Printed: prediction error (ms) for the unused
K = 8 and K = 16 at a fixed point (b = 4, sum l = 1200), and the condition number of the pooled matrix."""
import numpy as np
import ext18 as H

KS = (8, 16, 32)
C0, CB, VT, KT = 42.5, 2.55, 0.0256, 0.002


def true_t(b, K, sl):
    return C0 + CB * b + VT * b * (K + 1) + KT * (0.5 + (K + 1) / 34.0) * sl


def vx(b, K, sl):
    s = sl / 1000.0
    return np.array([float(K == 8), float(K == 16), float(K == 32), b, s, (K + 1) * s / 17.0])


class PerK:
    def __init__(self, lam=0.999):
        self.r = {K: H._RLS(6, lam) for K in KS}

    def add(self, K, x, t):
        self.r[K].add(x, t)

    def coef(self):
        A = sum(r.A for r in self.r.values()); y = sum(r.y for r in self.r.values())
        scale = np.maximum(np.diag(A), 1e-9)
        return np.linalg.solve(A + 1e-6 * np.diag(scale) + 1e-12 * np.eye(6), y)


def main():
    rng = np.random.default_rng(31)
    pooled, perk = H._RLS(6), PerK()
    def add(b, K, sl):
        t = true_t(b, K, sl) * (1 + rng.normal(0, 0.05))
        x = vx(b, K, sl)
        pooled.add(x, t); perk.add(K, x, t)
    for _ in range(2):
        for K in KS:
            for b in (2, 4, 8):
                add(b, K, rng.uniform(400, 2400))
    marks = (0, 1000, 5000, 10000, 20000, 25000, 30000, 35000, 40000, 60000)
    n = 0
    print(f"{'rounds':>7s} | pooled: err K=8  err K=16   cond | per K: err K=8  err K=16")
    for m in marks:
        while n < m:
            add(int(rng.choice((2, 4, 8))), 32, rng.uniform(400, 2400)); n += 1
        cp, ck = pooled.coef(), perk.coef()
        e = lambda c, K: float(np.dot(c, vx(4, K, 1200.0))) - true_t(4, K, 1200.0)
        print(f"{m:7d} | {e(cp, 8):15.2f} {e(cp, 16):9.2f} {np.linalg.cond(pooled.A):6.0e} | {e(ck, 8):13.2f} {e(ck, 16):9.2f}")


if __name__ == "__main__":
    main()
