"""Per-round throughput noise at fixed W (synthetic world). sigma_rel = sd(t_i - S m_i) / mean(t_i), S = sum t / sum m."""
import sys, math
sys.argv = ["x"]
import run46 as R
import ext34 as E34
import ext2 as X
import ext20 as V
from ext34 import FixG

class Rec(FixG):
    def __init__(self, **kw):
        super().__init__(**kw); self.rows = []
    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        com, t = super().round(world, slots, rng, rtt, rnd, trees_out)
        pv = self.prev
        if pv is not None and not pv["plan"]["use_tau"]:
            self.rows.append((float(sum(com)), float(t)))
        return com, t

def sig(rows):
    st = sum(t for t, _ in rows); sm = sum(m for _, m in rows); S = st / sm; mt = st / len(rows)
    var = sum((t - S * m) ** 2 for t, m in rows) / (len(rows) - 1)
    return math.sqrt(var) / mt

for kind, mix, b in (("mid", "short", 1), ("het", "mixed", 1), ("mid", "short", 4), ("het", "mixed", 4), ("mid", "short", 8), ("high", "short", 8)):
    for W in (4, 8):
        w = V.make_world(kind, mix, b, 25, phase=None, req_mult=1)
        E34.measure_world(w); w.first_W = 4
        p = Rec(fixed=W)
        X.simulate3(w, p, conc=None)
        rows = p.rows
        # lag-1 correlation of residuals
        st = sum(t for t, _ in rows); sm = sum(m for _, m in rows); S = st / sm
        e = [t - S * m for t, m in rows]; me = sum(e) / len(e)
        c1 = sum((e[i] - me) * (e[i + 1] - me) for i in range(len(e) - 1)) / sum((x - me) ** 2 for x in e)
        print(f"{kind:4s} {mix:6s} b={b} W={W:2d}  rounds {len(rows):5d}  sigma_rel {sig(rows):.3f}  lag1 {c1:+.2f}", flush=True)
