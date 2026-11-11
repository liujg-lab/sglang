"""Informative cluster size in M1 (review round 12, issue 5).

Request i has delta_i ~ N(0, sigma^2); every accepted parent hits with p_i = logistic(a + delta_i).
Each round walks the accepted chain: a hit continues to the next depth until the depth cap; the cap is
chosen adaptively from the request's OBSERVED hit rate so far (2 if < 0.6, else 6). Easy requests
therefore produce many more accepted-parent samples (cluster size depends on delta).
Fits by marginal likelihood (delta integrated, quadrature) on:
  all          every accepted parent's label (M1 as specified)
  drop-last    labels at the depth cap removed (missing by position: MAR given depth)
  drop-miss    labels removed when the round ends with a miss (outcome-dependent loss: MNAR)
  pooled       ordinary logistic, no random effect (logit of the pooled hit rate)
"""
import math, random, statistics

def logistic(x): return 1 / (1 + math.exp(-x))

NODES = [(-3.0 + 0.15 * k) for k in range(41)]
W = [math.exp(-0.5 * z * z) for z in NODES]; S = sum(W); W = [w / S for w in W]


def simulate(rng, a, sigma, n_req=1500, tokens=60, rounds=None):
    data = {"all": [], "drop-last": [], "drop-miss": []}
    for _ in range(n_req):
        p = logistic(a + rng.gauss(0, sigma))
        h = {k: 0 for k in data}; n = {k: 0 for k in data}
        made, rnds = 0, 0
        while (made < tokens) if rounds is None else (rnds < rounds):
            rnds += 1
            cap = 6 if (n["all"] and h["all"] / n["all"] >= 0.6) else 2
            depth, acc = 1, 0
            while True:
                hit = rng.random() < p
                for k in data:
                    keep = not (k == "drop-last" and depth == cap) and not (k == "drop-miss" and not hit)
                    if keep:
                        h[k] += hit; n[k] += 1
                if hit and depth < cap:
                    acc += 1; depth += 1
                else:
                    acc += hit
                    break
            made += acc + 1
        for k in data:
            data[k].append((h[k], n[k]))
    return data


def loglik(stats, a, sigma):
    tot = 0.0
    for (h, n), cnt in stats.items():
        s = 0.0
        for z, w in zip(NODES, W):
            p = logistic(a + sigma * z)
            s += w * p ** h * (1 - p) ** (n - h)
        tot += cnt * math.log(max(s, 1e-300))
    return tot


def fit(rows):
    stats = {}
    for r in rows:
        if r[1]:
            stats[r] = stats.get(r, 0) + 1
    best = max(((loglik(stats, a, s), a, s) for a in [x / 10 for x in range(-5, 16)]
                for s in [x / 10 for x in range(2, 15)]))
    _, a0, s0 = best
    best = max(((loglik(stats, a, s), a, s) for a in [a0 + x / 50 for x in range(-5, 6)]
                for s in [max(0.05, s0 + x / 50) for x in range(-5, 6)]))
    return best[1], best[2]


def main(reps=8, a=0.4, sigma=0.8, seed=9, rounds=None):
    rng = random.Random(seed)
    res = {"all": [], "drop-last": [], "drop-miss": [], "pooled": []}
    sizes = []
    for _ in range(reps):
        d = simulate(rng, a, sigma, rounds=rounds)
        for k in ("all", "drop-last", "drop-miss"):
            res[k].append(fit(d[k]))
        H = sum(h for h, n in d["all"]); N = sum(n for h, n in d["all"])
        res["pooled"].append((math.log(H / (N - H)), float("nan")))
        sizes.append(d["all"])
    lo = [n for rows in sizes for h, n in rows if h / n < 0.5]
    hi = [n for rows in sizes for h, n in rows if h / n >= 0.7]
    print(f"truth a={a}, sigma={sigma}; labels per request: hard requests {statistics.fmean(lo):.0f}, easy {statistics.fmean(hi):.0f}")
    for k, v in res.items():
        aa = [x[0] for x in v]; ss = [x[1] for x in v]
        sig = "" if k == "pooled" else f"  sigma {statistics.fmean(ss):.3f} +- {statistics.stdev(ss):.3f}"
        print(f"  {k:9s} a {statistics.fmean(aa):+.3f} +- {statistics.stdev(aa):.3f}{sig}")


if __name__ == "__main__":
    print("fixed output length (60 tokens per request):")
    main()
    print("fixed 20 rounds per request (cluster size grows with delta):")
    main(reps=6, rounds=20)
