"""Split the per-round (Dinkelbach) gap into 'ignores state dynamics' and 'does not know the future'.

Same model as theory/cross_round.py. Policies, all evaluated exactly under the true landscape p:
  optimal  : knows the whole future landscape, exact DP (oracle)
  rollout  : knows the true p only inside the current round's window (j .. j+Dmax), assumes the stationary
             mean beyond it; picks d minimising T(j,d) + E[V_hat(j + a)], where V_hat is the exact
             time-to-go under the stationary-belief landscape (so it accounts for prefix growth and the
             end of the request, but not for future easy/hard stretches)
  myopic   : per-round argmax E[A] - Theta*T, Theta = its own throughput (current ClearTree rule)
"""
import os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cross_round import landscape, outcome_dist, evaluate, optimal, myopic


def rollout(p, N, T, Dmax, pbar):
    q = [pbar] * (N + 12)
    Vh = [0.0] * (N + 1)
    for j in range(N - 1, -1, -1):
        Vh[j] = min(T(j, d) + sum(pr * Vh[min(N, j + a)] for a, pr in outcome_dist(q, j, d)) for d in range(1, Dmax + 1))
    pol = lambda j: min(range(1, Dmax + 1), key=lambda d: T(j, d) + sum(pr * Vh[min(N, j + a)] for a, pr in outcome_dist(p, j, d)))
    return evaluate(pol, p, N, T)


def main():
    rng = random.Random(5)
    for kappa in (0.0, 0.002, 0.02):
        g_my, g_ro = [], []
        for _ in range(200):
            N, Dmax = 256, 10
            p = landscape(rng, N)
            pbar = sum(p) / len(p)
            T0, c, L0 = rng.uniform(20, 120), rng.uniform(5, 15), 4096
            T = lambda j, d, T0=T0, c=c: T0 + c * d + kappa * (L0 + j)
            opt = optimal(p, N, T, Dmax)
            g_my.append(myopic(p, N, T, Dmax) / opt - 1)
            g_ro.append(rollout(p, N, T, Dmax, pbar) / opt - 1)
        f = lambda g: f"mean {sum(g)/len(g):.3%}, p95 {sorted(g)[int(.95*len(g))]:.3%}"
        print(f"kappa={kappa}: time vs oracle - per-round rule: {f(g_my)} | rollout (no future info): {f(g_ro)}", flush=True)


if __name__ == "__main__":
    main()
