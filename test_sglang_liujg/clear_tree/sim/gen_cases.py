"""Reference cases for the compiled-planner port: numpy results for mu-hat (given explicit draws) and the
nested subset DP. Written to cases.json; planner_bench.js recomputes them and reports the max difference."""
import json
import numpy as np
import feasibility as F
import ext as E

C, KS = F.C, F.KS
rng = np.random.default_rng(3)


def mu_ref(pool, idx, fs, fp, cur, dl, a, g, beta, W):
    D = pool[idx]                                            # (nR, N, nF, C)
    nR, N, nF, _ = D.shape
    p = F.sig(a + g * F.logit(D[..., 0]) + dl[:, None, None])
    w = D ** beta
    A = p[..., None] * w / w.sum(-1, keepdims=True)
    sc = (fs[:, None, :, None] * D).reshape(nR, N, nF * C)
    pv = (fp[:, None, :, None] * A).reshape(nR, N, nF * C)
    if nF * C > W:
        sel = np.argpartition(-sc, W - 1, axis=-1)[..., :W]
        pv = np.take_along_axis(pv, sel, -1)
    nT = cur.shape[1]
    top = -np.sort(-np.concatenate([np.broadcast_to(cur[:, None, :], (nR, N, nT)), pv], -1), -1)[..., :max(KS)]
    cum = np.cumsum(top, -1).mean(1)
    curc = np.cumsum(-np.sort(-cur, -1)[:, :max(KS)], -1)
    return [list(cum[:, min(K, cum.shape[1]) - 1] - curc[:, min(K, curc.shape[1]) - 1]) for K in KS]


cases = {"mu": [], "dp": []}
for _ in range(40):
    nR, W = int(rng.integers(1, 9)), int(rng.choice([4, 8, 16]))
    nF, nT, N, P = W, int(W * rng.integers(1, 4)), 16, 64
    pool = F.gen_dists(rng, 0.7, P)
    idx = rng.integers(0, P, (nR, N, nF))
    fs = rng.random((nR, nF)) * 0.5; fp = rng.random((nR, nF)) * 0.5
    cur = np.sort(rng.random((nR, nT)), -1)[:, ::-1] * 0.9
    dl = rng.normal(0, 0.5, nR)
    a, g, beta = 1.5, 1.0, float(rng.choice([1.0, 2.0]))
    cases["mu"].append(dict(nR=nR, W=W, nF=nF, nT=nT, N=N, P=P, pool=pool.ravel().tolist(), idx=idx.ravel().tolist(),
                            fs=fs.ravel().tolist(), fp=fp.ravel().tolist(), cur=cur.ravel().tolist(), dl=dl.tolist(),
                            a=a, g=g, beta=beta, ref=mu_ref(pool, idx, fs, fp, cur, dl, a, g, beta, W)))
for _ in range(60):
    n, H = int(rng.integers(1, 9)), int(rng.integers(1, 10))
    u = rng.random((n, H)) * rng.random(); el = rng.random((n, H)) < 0.85
    costs = [0.0] + [0.4 + 0.05 * k for k in range(1, n + 1)]
    act = int(rng.integers(1, 1 << n))
    xs = E.nested_dp_fast(u, el, act, lambda k: costs[k], 1.3)
    cases["dp"].append(dict(n=n, H=H, u=u.ravel().tolist(), el=el.ravel().astype(int).tolist(), costs=costs, theta=1.3,
                            ref=[np.where(np.isinf(x), -1e300, x).tolist() for x in xs]))
json.dump(cases, open("cases.json", "w"))
print("wrote", len(cases["mu"]), "mu cases and", len(cases["dp"]), "dp cases")
