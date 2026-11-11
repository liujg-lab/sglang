"""Snapshot-library estimators under selective expansion (review round 11, issue 3).

States at one layer: phi ~ U(0,1), unobserved u ~ N(0, su). Next-layer gain g = phi + u, observed y = g + noise.
Planner sees vhat = g - t + nu; natural expansion iff vhat > 0. Stopped states inside the band |vhat| <= h are
probed with p_band, stopped states outside with p_floor. Every decision (phi, category) is logged; y only for
expanded states. Estimand: m(phi) = E[g | phi] over all states reaching the layer (= phi).
Estimators at phi0 (kNN on phi, M neighbours among expanded states):
  unweighted   mean of y over the M nearest expanded snapshots
  ipw(w_max)   self-normalised 1/p weights (one-step propensity), clipped at w_max
  reg-adjust   kNN mean over natural snapshots, plus P(stop-category | phi) * pooled offset of probes
               (offset = mean over probes of y - natural kNN prediction at their phi, per category)
"""
import bisect, math, random, statistics


def simulate(rng, n=20000, su=0.3, sy=0.3, snu=0.2, t=1.0, h=0.2, p_band=0.01, p_floor=0.002):
    log, lib = [], []
    for _ in range(n):
        phi, u = rng.random(), rng.gauss(0, su)
        g = phi + u
        vhat = g - t + rng.gauss(0, snu)
        if vhat > 0:
            cat, p = "nat", 1.0
        elif vhat >= -h:
            cat, p = "band", p_band
        else:
            cat, p = "out", p_floor
        log.append((phi, cat))
        if cat == "nat" or rng.random() < p:
            lib.append((phi, g + rng.gauss(0, sy), cat, p))
    return log, lib


def knn(sorted_rows, key, phi0, M):
    i = bisect.bisect_left(key, phi0)
    lo, hi, out = i - 1, i, []
    while len(out) < M and (lo >= 0 or hi < len(sorted_rows)):
        if hi >= len(sorted_rows) or (lo >= 0 and phi0 - key[lo] <= key[hi] - phi0):
            out.append(sorted_rows[lo]); lo -= 1
        else:
            out.append(sorted_rows[hi]); hi += 1
    return out


def estimators(log, lib, phi0, M=64, wmaxes=(20, 100, math.inf)):
    rows = sorted(lib); key = [r[0] for r in rows]
    nb = knn(rows, key, phi0, M)
    res = {"unweighted": (statistics.fmean(r[1] for r in nb), M)}
    for wm in wmaxes:
        w = [min(1 / r[3], wm) for r in nb]
        est = sum(wi * r[1] for wi, r in zip(w, nb)) / sum(w)
        ess = sum(w) ** 2 / sum(wi * wi for wi in w)
        res[f"ipw w<={wm}"] = (est, ess)
    nat = [r for r in rows if r[2] == "nat"]; nkey = [r[0] for r in nat]
    def nat_pred(x):
        return statistics.fmean(r[1] for r in knn(nat, nkey, x, M))
    off = {}
    for cat in ("band", "out"):
        pr = [r for r in rows if r[2] == cat]
        off[cat] = statistics.fmean(r[1] - nat_pred(r[0]) for r in pr) if pr else 0.0
    lrows = sorted(log); lkey = [r[0] for r in lrows]
    nbl = knn(lrows, lkey, phi0, 4 * M)
    share = {c: sum(r[1] == c for r in nbl) / len(nbl) for c in ("nat", "band", "out")}
    est = nat_pred(phi0) + share["band"] * off["band"] + share["out"] * off["out"]
    n_probe = sum(r[2] != "nat" for r in rows)
    res["reg-adjust"] = (est, n_probe)
    return res


def main(reps=150, seed=5):
    rng = random.Random(seed)
    for phi0 in (0.5, 0.7, 0.9):
        acc = {}
        for _ in range(reps):
            log, lib = simulate(rng)
            for k, (est, ess) in estimators(log, lib, phi0).items():
                acc.setdefault(k, []).append((est - phi0, ess))
        print(f"phi0={phi0} (truth {phi0})")
        for k, rows in acc.items():
            bias = statistics.fmean(r[0] for r in rows)
            rmse = math.sqrt(statistics.fmean(r[0] ** 2 for r in rows))
            ess = statistics.fmean(r[1] for r in rows)
            lab = "probes used" if k == "reg-adjust" else "n_eff"
            print(f"   {k:14s} bias {bias:+.3f}  rmse {rmse:.3f}  {lab} {ess:7.1f}")


if __name__ == "__main__":
    main()
