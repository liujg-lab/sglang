"""Round 8 diagnostic: why ClearTree core trails the best static OPT in the high-acceptance workload.
Records tokens per request per round and ms per round. usage: diag8.py <b>"""
import json, sys, time
import ext3 as Y
import ext5 as V
import ext7 as S
import run3b as R
import run8 as R8


class Rec:
    def __init__(self, inner):
        self.inner, self.tok, self.req, self.ms, self.n = inner, 0, 0, 0.0, 0

    def round(self, world, slots, rng, rtt, rnd):
        com, t = self.inner.round(world, slots, rng, rtt, rnd)
        self.tok += sum(com); self.req += len(slots); self.ms += t; self.n += 1
        return com, t

    def summary(self, rate):
        return dict(rate=round(rate, 1), tok_per_req_round=round(self.tok / self.req, 2), ms_per_round=round(self.ms / self.n, 1))


def main():
    b = int(sys.argv[1])
    # The measured planner timings cover W = 4, 8, 16 only; W = 2 reuses the W = 4 entries (an upper bound).
    for n in (1, 4, 8):
        R8.X.JIT.setdefault((n, 2), R8.X.JIT[(n, 4)])
    fit = json.load(open("results/fit7r_C1.json"))
    out = {}
    for name, W, mode in (("CT learned W=2", 2, "learned"), ("CT learned W=4", 4, "learned"), ("CT oracle W=4", 4, "oracle")):
        w = R8.make_world(fit, "high", b, 7)
        for m in (R8.F, R8.E, R8.X, R8.V):
            m.KS = R8.CT_KS
        pol = Rec(V.CT5(W, mode, cost=R.cm_(w, 1.15), slow=None, n_snap=4, kreuse=10, learn_beta=True, dyn_delta=True))
        out[name] = pol.summary(R8.sim(w, pol, W))
    best = {4: (4, 32, 0.05), 8: (2, 16, 0.05)}.get(b, (4, 32, 0.4))
    for c in (best, (4, 32, 0.4), (4, 16, 0.4)):
        w = R8.make_world(fit, "high", b, 7)
        pol = Rec(S.OPTDecoupled(*c))
        out[f"static {c}"] = pol.summary(R8.sim(w, pol, c[0]))
    print(json.dumps(dict(b=b, res=out)), flush=True)


if __name__ == "__main__":
    main()
