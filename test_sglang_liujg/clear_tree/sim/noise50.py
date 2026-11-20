"""Paired gap W=8 tree vs the same tree cut to W=4, with the round's K for both (fixed K) or each width's own
K* = argmax_K [sum c_i topK mass - Theta_b * T_verify(K)] from the policy's online verify table. Gap in tokens
net of verify time (Theta * dT_verify), relative to mean round tokens. Synthetic world."""
import sys, math
sys.argv = ["x"]
import run46 as R
import ext31 as M31, ext34 as E34, ext2 as X, ext20 as V
from ext31 import mass, KS
from ext34 import FixG

class Rec(FixG):
    def __init__(self, **kw):
        super().__init__(**kw); self.rows = []
    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        com, t = super().round(world, slots, rng, rtt, rnd, trees_out)
        pv = self.prev
        if pv is None or pv["plan"]["use_tau"] or pv["snap"] is None:
            return com, t
        b = pv["b"]; th = self._theta(b)
        T = {k: self._time(b, pv["max_ell"], k) for k in KS}
        T = {k: v for k, v in T.items() if v is not None}
        K = pv["K"]
        if th is None or K not in T:
            return com, t
        cs = pv["snap"]["ratios"]
        full, cut = [], []
        for v in pv["views"]:
            L = len(v.layers)
            full.append(v.scores_upto(L))
            kept = M31.truncate(v, M31.parents_of(v.t), 4)
            cut.append(M31.kept_scores(v, kept, L))
        val = lambda sc, k: sum(c * mass(s, k) for c, s in zip(cs, sc)) - th * T[k]
        v8 = {k: val(full, k) for k in T}; v4 = {k: val(cut, k) for k in T}
        k8 = max(v8, key=v8.get); k4 = max(v4, key=v4.get)
        self.rows.append((v8[K] - v4[K], v8[k8] - v4[k4], K, k8, k4, float(sum(com))))
        return com, t

for kind, mix, b in (("mid", "short", 1), ("het", "mixed", 1), ("mid", "short", 4), ("het", "mixed", 4), ("mid", "short", 8), ("high", "short", 8)):
    w = V.make_world(kind, mix, b, 25, phase=None, req_mult=1); E34.measure_world(w); w.first_W = 4
    p = Rec(fixed=8); X.simulate3(w, p, conc=None)
    rs = p.rows; mt = sum(r[5] for r in rs) / len(rs)
    gf = sum(r[0] for r in rs) / len(rs) / mt; go = sum(r[1] for r in rs) / len(rs) / mt
    diff = sum(1 for r in rs if r[4] != r[2]) / len(rs)
    smaller = sum(1 for r in rs if r[4] < r[2]) / len(rs)
    print(f"{kind:4s} {mix:6s} b={b}  n {len(rs):4d}  gap(fixed K) {100*gf:5.2f}%  gap(own K*) {100*go:5.2f}%  "
          f"K*(W4 cut) != round K in {100*diff:4.1f}% (smaller {100*smaller:4.1f}%)", flush=True)
