"""Round 19 (usage: pairvar19.py <world> <kind> <b>). Per-round variance of W comparisons: paired (same round, replay of the narrower width from a tree built at W=8)
vs unpaired (different rounds). Every spec round is built at W=8 (rescore mode, p_explore=1, gate never passes)."""
import sys, json, math

import numpy as np
import run18 as R, ext10 as P, run10 as T

world, kind, b = sys.argv[1], sys.argv[2], int(sys.argv[3])
R.REQ_MULT = 3
w = R.make_world(world, kind, b, 8)
p = R.policy(w, wmode="rescore", p_explore=1.0, explore_to="max", gate=1e9, keep=10 ** 6, confirm=False)
T.sim(w, P.SwitchTok(p, T.R7.SW), 8 if b == 1 else 4)
recs = [r for c in p.cells.values() for r in c["recs"] if r["W"] == 8 and r["b"] == b]
th = None
rows = []
for r in recs:
    out = {}
    for Wx in (2, 4, 8):
        k = p._pickK(r["val"][Wx], r["theta"], r["b"])
        out[Wx] = (r["tok"][Wx][k], r["resid"] + p._model(r["b"], Wx, k, r["layer_m"]))
    rows.append(out)
tok8 = np.array([o[8][0] for o in rows]); tm8 = np.array([o[8][1] for o in rows])
theta = tok8.sum() / tm8.sum()
Y = {Wx: np.array([o[Wx][0] - theta * o[Wx][1] for o in rows]) for Wx in (2, 4, 8)}
A = tok8.mean()
res = {"world": world, "kind": kind, "b": b, "n": len(rows), "tok_per_round": round(float(A), 2)}
for a_, b_ in ((4, 2), (8, 4)):
    D = Y[a_] - Y[b_]
    gain = 100 * ((np.array([o[a_][0] for o in rows]).sum() / np.array([o[a_][1] for o in rows]).sum()) /
                  (np.array([o[b_][0] for o in rows]).sum() / np.array([o[b_][1] for o in rows]).sum()) - 1)
    s_pair = D.std() / A
    s_unp = math.sqrt(Y[a_].var() + Y[b_].var()) / A          # difference of two independent rounds
    s_ipw = math.sqrt(2 * (Y[a_] ** 2).mean() + 2 * (Y[b_] ** 2).mean() - D.mean() ** 2) / A   # IPW, p = 1/2
    ac = float(np.corrcoef(D[:-1], D[1:])[0, 1])
    res[f"{a_}v{b_}"] = dict(gain_pct=round(gain, 2), s_paired=round(s_pair, 3), s_unpaired=round(s_unp, 3),
                             s_ipw=round(s_ipw, 3), var_ratio_unp=round((s_unp / s_pair) ** 2, 1),
                             var_ratio_ipw=round((s_ipw / s_pair) ** 2, 1), lag1_corr_paired=round(ac, 3))
print(json.dumps(res), flush=True)
