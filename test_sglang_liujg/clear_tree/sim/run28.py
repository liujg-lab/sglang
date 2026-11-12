"""Round 28: two fixes for the v11 replay controller with online costs (ext22.BatchAccount15).
usage (conda sglang env): python run28.py <proc 0|1>

  reset          clear the controller statistics of every b bucket at a cost event (drift detection)
  cprobe         continuous timing probes: a width other than the current one that has not run for 300 non-warm
                 rounds of the bucket gets 2 real rounds (feeds the cost model only)
  reset+cprobe   both
Worlds: the twelve of round 26 and the three of round 27 B (mid, short prefixes, b = 8, Draft layer cost x0.3 / x3 /
x10 at the 400th speculative round); short runs seeds 8 and 9, long runs seed 8. References and the unfixed
controller ("r14 online resid") come from r25 / r26 / r27.
"""
import json, sys, time
import run25 as R
import run26 as Q
import run27 as T

BASE = dict(wmode="replay14", cost_mode="aware", timing="resid")
FIX = [("reset", dict(BASE, reset_on_event=True)), ("cprobe", dict(BASE, cprobe_period=300)),
       ("reset+cprobe", dict(BASE, reset_on_event=True, cprobe_period=300))]


def run(*a):
    out = Q.run(*a)
    return out


def main():
    proc = int(sys.argv[1])
    runs = ((8, 3), (9, 3), (8, 10))
    jobs = [((n, k, m, b, ph), s, mu, None) for (s, mu) in runs for (n, k, m, b, ph) in R.WORLDS]
    jobs += [((n, k, m, b, None), s, mu, c) for (s, mu) in runs for (n, k, m, b, c) in Q.CHANGE + T.NEWW]
    for i, ((name, kind, mix, b, phase), seed, mult, cm) in enumerate(jobs):
        if i % 2 != proc:
            continue
        t0 = time.time()
        res = {}
        for lab, kw in FIX:
            import ext2 as X, ext20 as V, ext22 as E
            w = V.make_world(kind, mix, b, seed, phase=phase, req_mult=mult)
            if cm is not None:
                Q.add_change(w, cm)
            w.first_W = 4
            p = E.BatchAccount15(**kw)
            conc = (R.R.CONC if mult == 3 else R.CONC_LONG) if name == "bvar" else None
            rate = X.simulate3(w, p, conc=conc)
            res[lab] = {"rate": rate, "spec_rounds": w.n_T, "w": {str(k): v for k, v in sorted(p.w_log.items())},
                        "switches": p.n_switch, "switch_log": p.switch_log[:30], "probes": p.n_probe_rounds,
                        "cost_events": p.n_cost_events, "resets": p.n_resets, "cprobes": p.n_cprobe,
                        "lt_err": float(sum(p.err_l) / len(p.err_l)) if p.err_l else None}
        print(json.dumps(dict(world=name, kind=kind, mix=mix, b=b, phase=phase, seed=seed, req_mult=mult, cmult=cm, res=res,
                              sec=round(time.time() - t0, 1), python=sys.version.split()[0])), flush=True)


if __name__ == "__main__":
    main()
