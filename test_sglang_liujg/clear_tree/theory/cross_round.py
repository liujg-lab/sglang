"""Issue 1: per-round Dinkelbach vs average-reward optimum when the action moves the state.

One request, fixed greedy output of N tokens (lossless => the token sequence is policy-independent).
State = token index j. p_j = probability the draft matches token j (easy/hard Markov landscape).
A round at j with chain depth d accepts k drafts w.p. prod(p_j..p_{j+k-1})*(1-p_{j+k}) (k<d), all d w.p. prod;
it then commits k+1 tokens (bonus). Round time T = T0 + c*d + kappa*(L0 + j)  (prefix grows with j).
  optimal : stochastic shortest path over j, minimizes expected total time to reach N (exact DP)
  myopic  : per round argmax_d E[A] - Theta*T, Theta iterated to the policy's own throughput
"""
import random

def landscape(rng, N, stay=0.9):
    p, hard, out = None, False, []
    for _ in range(N + 12):
        if rng.random() > stay:
            hard = not hard
        out.append(rng.uniform(0.15, 0.45) if hard else rng.uniform(0.8, 0.97))
    return out

def outcome_dist(p, j, d):
    probs, run = [], 1.0
    for k in range(d):
        probs.append(run * (1 - p[j + k]))
        run *= p[j + k]
    probs.append(run)                       # all d accepted
    return [(k + 1, pr) for k, pr in enumerate(probs)]   # tokens committed = k + 1

def evaluate(policy, p, N, T):
    """Expected total time and number of rounds under a policy (exact, backward)."""
    E = [0.0] * (N + 1)
    for j in range(N - 1, -1, -1):
        d = policy(j)
        E[j] = T(j, d) + sum(pr * E[min(N, j + a)] for a, pr in outcome_dist(p, j, d))
    return E[0]

def optimal(p, N, T, Dmax):
    E = [0.0] * (N + 1)
    for j in range(N - 1, -1, -1):
        E[j] = min(T(j, d) + sum(pr * E[min(N, j + a)] for a, pr in outcome_dist(p, j, d)) for d in range(1, Dmax + 1))
    return E[0]

def myopic(p, N, T, Dmax):
    theta = 0.1
    for _ in range(30):
        pol = lambda j, th=theta: max(range(1, Dmax + 1), key=lambda d: sum(a * pr for a, pr in outcome_dist(p, j, d)) - th * T(j, d))
        tot = evaluate(pol, p, N, T)
        new = N / tot
        if abs(new - theta) < 1e-9:
            break
        theta = new
    return tot

def main():
    rng = random.Random(5)
    for kappa in (0.0, 0.002, 0.02):
        gaps = []
        for _ in range(200):
            N, Dmax = 256, 10
            p = landscape(rng, N)
            T0, c, L0 = rng.uniform(20, 120), rng.uniform(5, 15), 4096
            T = lambda j, d, T0=T0, c=c: T0 + c * d + kappa * (L0 + j)
            opt, my = optimal(p, N, T, Dmax), myopic(p, N, T, Dmax)
            gaps.append(my / opt - 1)
        gaps.sort()
        print(f"kappa={kappa}: myopic time / optimal - 1: mean {sum(gaps)/len(gaps):.3%}, "
              f"p95 {gaps[int(.95*len(gaps))]:.3%}, max {gaps[-1]:.3%}")


if __name__ == "__main__":
    main()
