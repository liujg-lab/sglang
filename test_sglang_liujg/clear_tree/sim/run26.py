"""Round 26: the v11 replay controller with the online cost model (ext22.BatchAccount15).
usage (conda sglang env): python run26.py <proc 0|1>

Worlds: the nine of round 25 (run25.WORLDS), plus three with a Draft layer cost change during the run (all layer
cost terms d0, d_req, d_tok multiplied at the 400th speculative round): mid acceptance, short prefixes, b = 4, x3 and
x0.3; high acceptance, short prefixes, b = 8, x0.3. Short runs (3x requests) seeds 8 and 9, long runs (10x) seed 8.
Policies (W start: flat table):
  flat online       no W control, D61 online costs (timing probes + residual drift detection)
  r14 known resid   replay controller, known costs, v11 timing (measured time - model of what ran + model of candidate)
  r14 online resid  replay controller, online costs, v11 timing  <- the v11 design
In the change worlds also: fixed 2 / 4 / 8, flat known and r14 known model5 (= round 25's replay14); in the other
worlds those come from r25 (same code path; r14 known model5 reproduces r25 exactly).
"""
import json, sys, time
import ext2 as X
import ext20 as V
import ext22 as E
import run25 as R

CHANGE = [("chg x3", "mid", "short", 4, 3.0), ("chg x0.3", "mid", "short", 4, 0.3), ("chg x0.3", "high", "short", 8, 0.3)]
NEW = [("flat online", dict(wmode="flat", cost_mode="aware")),
       ("r14 known resid", dict(wmode="replay14", cost_mode="known", timing="resid")),
       ("r14 online resid", dict(wmode="replay14", cost_mode="aware", timing="resid"))]
REF = [("fixed 2", dict(wmode="fixed", fixed_W=2, cost_mode="known")), ("fixed 4", dict(wmode="fixed", fixed_W=4, cost_mode="known")),
       ("fixed 8", dict(wmode="fixed", fixed_W=8, cost_mode="known")), ("flat known", dict(wmode="flat", cost_mode="known")),
       ("r14 known model5", dict(wmode="replay14", cost_mode="known", timing="model5"))]
CHANGE_ROUND = 400


def add_change(w, mult):
    orig = w.real_T_l

    def real_T_l(rng, b, K, suml):
        out = orig(rng, b, K, suml)
        if w.n_T == CHANGE_ROUND:
            w.d0, w.d_req, w.d_tok = w.d0 * mult, w.d_req * mult, w.d_tok * mult
        return out
    w.real_T_l = real_T_l


def run(world, kind, mix, b, phase, seed, mult, kw, cmult=None):
    w = V.make_world(kind, mix, b, seed, phase=phase, req_mult=mult)
    if cmult is not None:
        add_change(w, cmult)
    w.first_W = 4
    p = E.BatchAccount15(**kw)
    conc = None
    if world == "bvar":
        conc = R.R.CONC if mult == 3 else R.CONC_LONG
    rate = X.simulate3(w, p, conc=conc)
    out = {"rate": rate, "spec_rounds": w.n_T, "w": {str(k): v for k, v in sorted(p.w_log.items())},
           "switches": p.n_switch, "switch_log": p.switch_log[:30], "explore": p.n_explore,
           "probes": p.n_probe_rounds, "cost_events": p.n_cost_events}
    if p.err_l:
        out["lt_err"] = float(sum(p.err_l) / len(p.err_l))
    if p.err_v:
        out["vt_err"] = float(sorted(p.err_v)[len(p.err_v) // 2])
    if kw["wmode"] == "replay14":
        out["final"] = {str(k): v["cur"] for k, v in p.ctx14.items()}
    return out


def main():
    proc = int(sys.argv[1])
    jobs = [((n, k, m, b, ph), s, mu, None) for (s, mu) in ((8, 3), (9, 3), (8, 10)) for (n, k, m, b, ph) in R.WORLDS]
    jobs += [((n, k, m, b, None), s, mu, c) for (s, mu) in ((8, 3), (9, 3), (8, 10)) for (n, k, m, b, c) in CHANGE]
    for i, ((name, kind, mix, b, phase), seed, mult, cm) in enumerate(jobs):
        if i % 2 != proc:
            continue
        t0 = time.time()
        pols = NEW + (REF if cm is not None else [])
        res = {lab: run(name, kind, mix, b, phase, seed, mult, kw, cm) for lab, kw in pols}
        print(json.dumps(dict(world=name, kind=kind, mix=mix, b=b, phase=phase, seed=seed, req_mult=mult, cmult=cm, res=res,
                              sec=round(time.time() - t0, 1), python=sys.version.split()[0])), flush=True)


if __name__ == "__main__":
    main()
