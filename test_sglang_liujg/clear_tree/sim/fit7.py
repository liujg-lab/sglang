"""Round 7, part A: recalibrate the simulator's costs to the real T0 grid (a reduced S2).

For every real cell (b, N, tau) the synthetic acceptance world runs OPT(N, tau) and records, per round, the
committed tokens and the Draft layers (with the number of active requests in each). With lookahead and KV
reuse off the first Draft layer is on the critical path, so it is counted. The real round time of a cell is
then b * accept_len_sim / throughput_real, and a linear cost model is fitted to it by non-negative least
squares on relative error:

  round = RTT + c0 + c_b * b + v_tok * b * (N + 1)                        Target verify and per-round host
        + sum over layers [ d0 + d_req * bucket(n) + d_tok * bucket(n) * N ]   Draft

Structures compared: "token-only" (the original simulator: c_b = d_req = 0) and "full".
Target-only time is fitted separately: a0 + a_b * b.
Acceptance variants: a (hit intercept), k (concentration of the Draft's top-1 probability, Beta(k*s, k*(1-s));
the original generator uses k = 6) and the range of the per-request mean s.
OPT stop rule: the real one (ext7.grow_real); RULE = "score" restores the rounds 1-6 rule.
usage: fit7.py <a> <out.json> [k] [s_lo] [s_hi] [n_seeds]
"""
import itertools, json, sys
import numpy as np
import feasibility as F
import ext2 as X
from calibrated import CAL
from real_t0 import SPEC, TARGET_ONLY, ALEN_N8
import ext7 as S
from ext7 import set_conc, acc_kw
RULE = 'real'

RTT = 4.0
NAMES = ["c0", "c_b", "v_tok", "d0", "d_req", "d_tok"]


class OPTFeat(X.OPT2):
    """OPT round that returns per-round features instead of a time."""

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        trees = [X.Tree2(world.reqs[r], j, world) for r, j in slots]
        params = (world.a, world.g, world.beta, 0.0)
        for t in trees:
            t.expand(self.N, rng, params)
        rem = [world.reqs[r]["L"] - j for r, j in slots]
        b = len(slots)
        layers = [b]
        if RULE == "real":
            layers += S.grow_real(world, trees, self.N, self.tau, rem, rng)
            com = [min(t.accepted(t.topk(self.N, False)) + 1, rem[i]) for i, t in enumerate(trees)]
            return com, layers
        active, depth = list(range(len(trees))), 1
        while active and depth < world.DMAX:
            active = [i for i in active
                      if max(trees[i].score[v] for v in trees[i].frontier) >= self.tau and depth + 1 <= max(1, rem[i] - 1)]
            if not active:
                break
            for i in active:
                trees[i].expand(self.N, rng, params)
            layers.append(len(active))
            depth += 1
        com = [min(t.accepted(t.topk(self.N, False)) + 1, rem[i]) for i, t in enumerate(trees)]
        return com, layers


def features(b, N, tau, akw, seeds=(7, 8), n_req=None):
    """Mean per-round features and tokens; runs the same slot logic as simulate3 at fixed concurrency."""
    tok = nr = 0
    f = np.zeros(6)
    for seed in seeds:
        w = X.World2(seed=seed, b=b, rtt=RTT, n_req=n_req or max(48, 8 * b), **akw)
        rng = np.random.default_rng(seed)
        pol = OPTFeat(N, tau)
        slots, nxt = [], 0
        while True:
            while len(slots) < b and nxt < len(w.reqs):
                slots.append([nxt, 0]); nxt += 1
            if not slots:
                break
            com, layers = pol.round(w, slots, rng, RTT, nr)
            bb = len(slots)
            sb = sum(F.bucket(n) for n in layers)
            f += [1.0, bb, bb * (N + 1), len(layers), sb, sb * N]
            tok += sum(com); nr += 1
            keep = []
            for s, c in zip(slots, com):
                s[1] += c
                if s[1] < w.reqs[s[0]]["L"]:
                    keep.append(s)
            slots = keep
            # Stop before the drain: the real benchmark keeps the concurrency full.
            if nxt >= len(w.reqs) and len(slots) < b:
                break
    return f / nr, tok / nr


def nnls(A, y, allowed):
    """Exhaustive non-negative least squares over subsets of the allowed columns (6 columns at most)."""
    best = (np.inf, None)
    idx = [i for i in range(A.shape[1]) if allowed[i]]
    for k in range(1, len(idx) + 1):
        for sub in itertools.combinations(idx, k):
            x, *_ = np.linalg.lstsq(A[:, sub], y, rcond=None)
            if (x < 0).any():
                continue
            r = float(((A[:, sub] @ x - y) ** 2).sum())
            if r < best[0]:
                full = np.zeros(A.shape[1]); full[list(sub)] = x
                best = (r, full)
    return best[1]


def fit(cells, feat, structure):
    allowed = [True, structure == "full", True, True, structure == "full", True]
    rows, ys = [], []
    for c in cells:
        fv, tpr = feat[c]
        t_real = 1000.0 * tpr / SPEC[c]          # ms per round (tpr already sums over the b requests)
        w = 1.0 / t_real                          # relative error
        rows.append(fv * w); ys.append((t_real - RTT) * w)
    return nnls(np.array(rows), np.array(ys), allowed)


def predict(theta, fv, tpr):
    return 1000.0 * tpr / (RTT + float(fv @ theta))


def main():
    a = float(sys.argv[1]) if len(sys.argv) > 1 else CAL["a"]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    k = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
    s_lo = float(sys.argv[4]) if len(sys.argv) > 4 else CAL["s_range"][0]
    s_hi = float(sys.argv[5]) if len(sys.argv) > 5 else CAL["s_range"][1]
    seeds = (7, 8)[:int(sys.argv[6])] if len(sys.argv) > 6 else (7, 8)
    set_conc(k)
    akw = acc_kw(a, s_lo, s_hi)
    cells = sorted(SPEC)
    feat = {c: features(*c, akw=akw, seeds=seeds) for c in cells}
    print(f"acceptance a={a} k={k} s=({s_lo},{s_hi}) seeds={seeds}: sim accept length (tokens per request per round), real N=8 values {ALEN_N8}")
    for N in (2, 4, 8, 16):
        print(f"  N={N:2d} " + " ".join(f"tau{t}:{feat[(1, N, t)][1]:.2f}" for t in (0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
                                        if (1, N, t) in feat))
    # Target-only
    bs = np.array(sorted(TARGET_ONLY)); tar = np.array([1000.0 * b / TARGET_ONLY[b] for b in bs])
    a_b, a0 = np.polyfit(bs, tar, 1)
    print(f"Target-only step: a0={a0:.2f} ms + a_b={a_b:.3f} ms per request (real {dict(zip(bs.tolist(), tar.round(2).tolist()))})")
    res = {"a": a, "k": k, "s_range": [s_lo, s_hi], "a0": a0, "a_b": a_b, "fits": {}}
    for structure in ("token-only", "full"):
        th = fit(cells, feat, structure)
        errs = {c: predict(th, *feat[c]) / SPEC[c] - 1 for c in cells}
        e = np.array(list(errs.values()))
        # held out: fit without b=4, predict b=4
        th_ho = fit([c for c in cells if c[0] != 4], feat, structure)
        e_ho = np.array([predict(th_ho, *feat[c]) / SPEC[c] - 1 for c in cells if c[0] == 4])
        print(f"\n== {structure}: " + ", ".join(f"{n}={v:.3f}" for n, v in zip(NAMES, th)))
        print(f"   in-sample |err| mean {np.abs(e).mean():.1%}, max {np.abs(e).max():.1%}; "
              f"b=4 held out |err| mean {np.abs(e_ho).mean():.1%}, max {np.abs(e_ho).max():.1%}")
        for b in (1, 4, 8):
            line = []
            for N in (2, 4, 8, 16):
                for t in (0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0):
                    c = (b, N, t)
                    if c in errs:
                        line.append(f"N{N}t{t}:{predict(th, *feat[c]):.0f}/{SPEC[c]:.0f}")
            print(f"   b={b}: " + " ".join(line))
        # decomposition at b=1 and b=8, N=8, tau=0.4
        for b in (1, 8):
            fv, tpr = feat[(b, 8, 0.4)]
            parts = fv * th
            print(f"   b={b} N=8 tau=0.4 round {RTT + parts.sum():.1f} ms: Target/host const {parts[0]:.1f}, "
                  f"per-request {parts[1]:.1f}, verify tokens {parts[2]:.1f}, Draft layers {parts[3]:.1f} + "
                  f"{parts[4]:.1f} (per request) + {parts[5]:.1f} (tokens); layers/round {fv[3]:.2f}")
        res["fits"][structure] = dict(theta=th.tolist(), err_mean=float(np.abs(e).mean()), err_max=float(np.abs(e).max()),
                                      ho_mean=float(np.abs(e_ho).mean()), ho_max=float(np.abs(e_ho).max()))
    if out:
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
