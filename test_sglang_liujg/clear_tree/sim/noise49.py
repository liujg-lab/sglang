"""Same-round paired difference: expected tokens of the built tree (W=8) minus the same tree cut to W=4 (descendants
removed), c_i weighted. Relative to mean round tokens: mean gap and sd per round; rounds for se = 0.5%."""
import sys, math
sys.argv = ["x"]
import run46 as R
import ext31 as M31, ext34 as E34, ext2 as X, ext20 as V
from ext31 import mass
from ext34 import FixG

class Rec(FixG):
    def __init__(self, **kw):
        super().__init__(**kw); self.d = []; self.tok = []
    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        com, t = super().round(world, slots, rng, rtt, rnd, trees_out)
        pv = self.prev
        if pv is not None and not pv["plan"]["use_tau"] and pv["snap"] is not None:
            cs = pv["snap"]["ratios"]; K = pv["K"]; s = 0.0
            for i, v in enumerate(pv["views"]):
                L = len(v.layers); base = mass(v.scores_upto(L), K)
                kept = M31.truncate(v, M31.parents_of(v.t), 4)
                s += cs[i] * (base - mass(M31.kept_scores(v, kept, L), K))
            self.d.append(s); self.tok.append(float(sum(com)))
        return com, t

for kind, mix, b in (("mid", "short", 1), ("het", "mixed", 1), ("mid", "short", 4), ("het", "mixed", 4), ("mid", "short", 8), ("high", "short", 8)):
    w = V.make_world(kind, mix, b, 25, phase=None, req_mult=1); E34.measure_world(w); w.first_W = 4
    p = Rec(fixed=8); X.simulate3(w, p, conc=None)
    mt = sum(p.tok) / len(p.tok); md = sum(p.d) / len(p.d)
    sd = math.sqrt(sum((x - md) ** 2 for x in p.d) / (len(p.d) - 1))
    print(f"{kind:4s} {mix:6s} b={b}  paired gap W8 vs W4 cut: mean {100*md/mt:5.2f}% of tokens, sd/round {100*sd/mt:5.2f}%, "
          f"rounds for se 0.5%: {math.ceil((sd/mt/0.005)**2)}", flush=True)
