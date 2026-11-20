"""Per-round noise: realized tokens vs Draft-expected tokens (c_i weighted top-K mass + 1 per request), and the
time part alone. sigma_rel = sd(y_i - S m_i) / mean(y_i). Synthetic world."""
import sys, math
sys.argv = ["x"]
import run46 as R
import ext34 as E34, ext2 as X, ext20 as V
from ext34 import FixG

class Rec(FixG):
    def __init__(self, **kw):
        super().__init__(**kw); self.rows = []
    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        com, t = super().round(world, slots, rng, rtt, rnd, trees_out)
        pv = self.prev
        if pv is not None and not pv["plan"]["use_tau"] and pv["snap"] is not None:
            cs = pv["snap"]["ratios"]
            exp_tok = sum(c * e + 1.0 for c, e in zip(cs, pv["e_ver"]))
            self.rows.append((float(sum(com)), exp_tok, float(t)))
        return com, t

def sig(ys, ms):
    S = sum(ys) / sum(ms); my = sum(ys) / len(ys)
    return math.sqrt(sum((y - S * m) ** 2 for y, m in zip(ys, ms)) / (len(ys) - 1)) / my

def cv(xs):
    m = sum(xs) / len(xs); return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) / m

for kind, mix, b in (("mid", "short", 1), ("het", "mixed", 1), ("mid", "short", 4), ("het", "mixed", 4), ("mid", "short", 8), ("high", "short", 8)):
    w = V.make_world(kind, mix, b, 25, phase=None, req_mult=1); E34.measure_world(w); w.first_W = 4
    p = Rec(fixed=8); X.simulate3(w, p, conc=None)
    real = [r[0] for r in p.rows]; ex = [r[1] for r in p.rows]; ms = [r[2] for r in p.rows]
    bias = sum(real) / sum(ex) - 1
    print(f"{kind:4s} {mix:6s} b={b} W=8  n {len(ms):4d}  realized tok/ms {sig(real, ms):.3f}  expected tok/ms {sig(ex, ms):.3f}  "
          f"round-time CV {cv(ms):.3f}  expected/realized bias {100*bias:+.1f}%", flush=True)
