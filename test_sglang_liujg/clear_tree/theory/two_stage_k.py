"""Two-stage K problem with random layer outcomes (review round 11, issues 1, 12, 14).

True problem: grow the tree adaptively (layer outcomes are revealed after each forward), then pick
K_v on the realized tree:  max_policy E[ max_K (G^K(tree) - VK[K]) ] - E[cost].
Layer increments: inc_i^K(t) = base_i^K(t) * eps_{i,t}; eps in {1-a, 1+a}, shared across K.
eps_{i,t} = eps_{i,t-1} with prob rho, else a fresh uniform draw (rho=1: request-level effect).
Planners use certainty-equivalent expectations E[eps_t | history] (optionally with a biased mean).

Policies (all exact expectations, no Monte Carlo):
  OPT        exact Bellman over (layer, active set, revealed eps) with terminal max_K
  locked     K_plan chosen at layer 1, re-plan active sets under K_plan, verify with K_plan
  locked+K*  same, verify with argmax_K on the realized tree
  free+K*    every layer re-choose K by total value, verify with argmax_K
  Kg=2+K*    plan with the smallest K throughout, verify with argmax_K
  Kg=8+K*    plan with the largest K throughout, verify with argmax_K
"""
import itertools, math, random, sys, os
from functools import lru_cache
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_lemmas import build, G

KS = [2, 4, 8]
BUCKETS = [1, 2, 4, 8]
def bucket(n): return next(b for b in BUCKETS if b >= n)


def make(rng, rho, amp, mu_plan):
    b = 2 if rng.random() < 0.6 else 3
    D = 4
    trees = [build(rng, D, rng.choice([2, 4]), rng.choice([2, 4]), True)[0] for _ in range(b)]
    base = {K: [[G(t, K, d) - G(t, K, d - 1) for d in range(1, D + 1)] for t in trees] for K in KS}
    c = [rng.uniform(0.0, 0.08) for _ in range(b)]
    a = rng.uniform(0.02, 0.15)
    f = {n: a * {1: 1, 2: 1.15, 4: 1.35, 8: 1.7}[bucket(n)] for n in range(1, b + 1)}
    VK = {K: rng.uniform(0, 0.05) * K for K in KS}
    lo, hi = 1 - amp, 1 + amp
    p_same = rho + (1 - rho) / 2

    depth_vecs = list(itertools.product(range(1, D + 1), repeat=b))
    cost_tab = {}
    for d in depth_vecs:
        for m0 in range(1, D + 1):
            tot = 0.0
            for mm in range(m0, D):
                act = [i for i in range(b) if d[i] > mm]
                if act:
                    tot += f[len(act)] + sum(c[i] for i in act)
            cost_tab[d, m0] = tot

    def step_cost(A):
        return f[len(A)] + sum(c[i] for i in A)

    def realized(K, hist):
        return sum(base[K][i][t] * hist[i][t] for i in range(b) for t in range(len(hist[i]))) - VK[K]

    def exp_eps(last, k):  # planner's E[eps_{t+k} | eps_t = last]
        return mu_plan + (rho ** k) * (last - mu_plan)

    def plan(K, m, active, hist):
        """CE plan under K from layer m. Returns (value incl. past, next active set)."""
        best, arg = -math.inf, None
        for d in depth_vecs:
            ok = True
            for i in range(b):
                cur = len(hist[i])
                if i in active:
                    if d[i] < cur:
                        ok = False; break
                elif d[i] != cur:
                    ok = False; break
            if not ok:
                continue
            v = -cost_tab[d, m] - VK[K]
            for i in range(b):
                cur = len(hist[i])
                v += sum(base[K][i][t] * hist[i][t] for t in range(cur))
                last = hist[i][-1]
                for t in range(cur, d[i]):
                    v += base[K][i][t] * exp_eps(last, t - cur + 1)
            if v > best + 1e-12:
                best, arg = v, d
        nxt = frozenset(i for i in active if arg[i] > m)
        return best, nxt

    def outcomes(A, hist):
        """All (prob, new_hist) after building one more layer for requests in A."""
        res = [(1.0, list(hist))]
        for i in A:
            last = hist[i][-1]
            other = lo if last == hi else hi
            new = []
            for p, h in res:
                for val, q in ((last, p_same), (other, 1 - p_same)):
                    h2 = list(h); h2[i] = h[i] + (val,)
                    new.append((p * q, h2))
            res = new
        return [(p, tuple(h)) for p, h in res]

    @lru_cache(maxsize=None)
    def bellman(m, active, hist):
        stop = max(realized(K, hist) for K in KS)
        if m >= D or not active:
            return stop
        best = stop
        act = sorted(active)
        for r in range(1, len(act) + 1):
            for A in itertools.combinations(act, r):
                v = -step_cost(A) + sum(p * bellman(m + 1, frozenset(A), h) for p, h in outcomes(A, hist))
                best = max(best, v)
        return best

    def run_policy(kind):
        @lru_cache(maxsize=None)
        def val(m, active, hist, Kcur):
            if m >= D or not active:
                return final(hist, Kcur)
            if kind == "locked" or kind == "locked+K*":
                K = Kcur if Kcur is not None else max(KS, key=lambda k: plan(k, m, active, hist)[0])
                _, nxt = plan(K, m, active, hist)
            elif kind == "free+K*":
                scores = {k: plan(k, m, active, hist) for k in KS}
                top = max(s[0] for s in scores.values())
                K = Kcur if (Kcur is not None and scores[Kcur][0] >= top - 1e-12) else max(KS, key=lambda k: scores[k][0])
                nxt = scores[K][1]
            else:
                K = int(kind.split("=")[1].split("+")[0])
                _, nxt = plan(K, m, active, hist)
            if not nxt:
                return final(hist, K)
            return -step_cost(nxt) + sum(p * val(m + 1, nxt, h, K) for p, h in outcomes(nxt, hist))

        def final(hist, Kcur):
            if kind == "locked":
                return realized(Kcur if Kcur is not None else KS[0], hist)
            return max(realized(K, hist) for K in KS)
        return val

    starts = []
    for e in itertools.product((lo, hi), repeat=b):
        starts.append((0.5 ** b, tuple((x,) for x in e)))
    full = frozenset(range(b))
    out = {"OPT": sum(p * bellman(1, full, h) for p, h in starts)}
    for kind in ("locked", "locked+K*", "free+K*", "Kg=2+K*", "Kg=8+K*"):
        v = run_policy(kind)
        out[kind] = sum(p * v(1, full, h, None) for p, h in starts)
    return out


def main(trials=1000, seed=11):
    settings = [("iid", 0.0, 0.5, 1.0), ("AR rho=0.5", 0.5, 0.5, 1.0), ("AR rho=0.9", 0.9, 0.5, 1.0),
                ("request-level rho=1", 1.0, 0.5, 1.0), ("rho=0.9, planner mean +20%", 0.9, 0.5, 1.2)]
    which = os.environ.get("SETTING")
    for name, rho, amp, mu in settings:
        if which and which != name:
            continue
        rng = random.Random(seed)
        rel = {}
        for _ in range(trials):
            o = make(rng, rho, amp, mu)
            for k, v in o.items():
                if k == "OPT":
                    continue
                rel.setdefault(k, []).append((o["OPT"] - v) / max(abs(o["OPT"]), 1e-9))
        print(f"[{name}]  relative regret vs exact two-stage optimum (mean / share>0.1% / p95)")
        for k, xs in rel.items():
            xs.sort()
            print(f"   {k:10s} {100*sum(xs)/len(xs):6.2f}%  {100*sum(x > 1e-3 for x in xs)/len(xs):5.1f}%  {100*xs[int(.95*len(xs))]:6.2f}%")


if __name__ == "__main__":
    main()
