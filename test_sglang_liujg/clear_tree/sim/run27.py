"""Round 27: delaying the first decision of the replay controller (online costs), and cost changes that move the
best W. usage (conda sglang env): python run27.py <proc 0|1>

Part A, the twelve worlds of round 26 (references and the other policies come from r26): three delayed variants of
the v11 controller with online costs ("r14 online resid" in r26):
  d100 reset   no decision before 100 controller rounds of the bucket; nothing accumulated before that
  d300 reset   the same with 300 rounds
  d300 keep    no decision before 300 rounds; everything since the probes accumulated
Part B, new worlds: mid acceptance, short prefixes, b = 8, every Draft layer cost term x0.3, x3 or x10 at the 400th
speculative round. Probed beforehand (seed 8, short): best W stays 4 at x0.3; becomes 2 at x3 (switching 4 -> 2 at the
change beats fixed 4 by 1.6%) and at x10 (by 3.9%). All policies: fixed 2 / 4 / 8, the scheduled switch 4 -> 2 and
4 -> 8 at the change, flat known / online, r14 known resid, r14 online resid and the three delayed variants.
Short runs (3x requests) seeds 8 and 9, long runs (10x) seed 8.
"""
import json, sys, time
import ext2 as X
import ext20 as V
import ext22 as E
import run25 as R
import run26 as Q

DELAY = [("d100 reset", dict(wmode="replay14", cost_mode="aware", timing="resid", first_delay=100, delay_reset=True)),
         ("d300 reset", dict(wmode="replay14", cost_mode="aware", timing="resid", first_delay=300, delay_reset=True)),
         ("d300 keep", dict(wmode="replay14", cost_mode="aware", timing="resid", first_delay=300, delay_reset=False))]
NEWW = [("chg8 x0.3", "mid", "short", 8, 0.3), ("chg8 x3", "mid", "short", 8, 3.0), ("chg8 x10", "mid", "short", 8, 10.0)]
FULL = Q.REF[:3] + [("sched 4->2", dict(sched=(4, 2))), ("sched 4->8", dict(sched=(4, 8)))] + \
       [Q.REF[3]] + Q.NEW + DELAY


class Sched(E.BatchAccount15):
    def __init__(self, w0, w1):
        super().__init__(wmode="fixed", fixed_W=w0, cost_mode="known")
        self.w1 = w1

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        if world.n_T >= Q.CHANGE_ROUND:
            self.fixed_W = self.w1
        return super().round(world, slots, rng, rtt, rnd, trees_out)


def run(world, kind, mix, b, phase, seed, mult, kw, cmult=None):
    if "sched" not in kw:
        return Q.run(world, kind, mix, b, phase, seed, mult, kw, cmult)
    w = V.make_world(kind, mix, b, seed, phase=phase, req_mult=mult)
    if cmult is not None:
        Q.add_change(w, cmult)
    w.first_W = 4
    p = Sched(*kw["sched"])
    rate = X.simulate3(w, p)
    return {"rate": rate, "spec_rounds": w.n_T, "w": {str(k): v for k, v in sorted(p.w_log.items())}}


def main():
    proc = int(sys.argv[1])
    runs = ((8, 3), (9, 3), (8, 10))
    jobs = [((n, k, m, b, ph), s, mu, None, "A") for (s, mu) in runs for (n, k, m, b, ph) in R.WORLDS]
    jobs += [((n, k, m, b, None), s, mu, c, "A") for (s, mu) in runs for (n, k, m, b, c) in Q.CHANGE]
    jobs += [((n, k, m, b, None), s, mu, c, "B") for (s, mu) in runs for (n, k, m, b, c) in NEWW]
    for i, ((name, kind, mix, b, phase), seed, mult, cm, part) in enumerate(jobs):
        if i % 2 != proc:
            continue
        t0 = time.time()
        pols = DELAY if part == "A" else FULL
        res = {lab: run(name, kind, mix, b, phase, seed, mult, kw, cm) for lab, kw in pols}
        print(json.dumps(dict(part=part, world=name, kind=kind, mix=mix, b=b, phase=phase, seed=seed, req_mult=mult, cmult=cm,
                              res=res, sec=round(time.time() - t0, 1), python=sys.version.split()[0])), flush=True)


if __name__ == "__main__":
    main()
