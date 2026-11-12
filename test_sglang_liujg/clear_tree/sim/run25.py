"""Round 25: fast-converging online W (ext21, wmode fast) vs the replay controller (replay14), flat start.
usage (conda sglang env): python run25.py <proc 0|1>

Worlds: the seven of round 23 (run23.WORLDS: base mid, het, phased, bvar) plus high acceptance with short prefixes
(b = 4, 8), where the best fixed W (2) differs from the flat table (4). Known prefix-aware costs.
Runs: short (3x requests) with seeds 8 and 9, long (10x requests) with seed 8.
Policies: flat, fixed 2 / 4 / 8 (references), replay14, fast.
"""
import json, sys, time
import ext2 as X
import ext20 as V
import ext21 as F
import run23 as R

WORLDS = R.WORLDS + [("high", "high", "short", 4, None), ("high", "high", "short", 8, None)]
POL = [("flat", dict(wmode="flat")), ("fixed 2", dict(wmode="fixed", fixed_W=2)), ("fixed 4", dict(wmode="fixed", fixed_W=4)),
       ("fixed 8", dict(wmode="fixed", fixed_W=8)), ("replay14", dict(wmode="replay14")), ("fast", dict(wmode="fast"))]
JOBS = [(w, seed, 3) for seed in (8, 9) for w in WORLDS] + [(w, 8, 10) for w in WORLDS]
CONC_LONG = [(40000 * 10 / 3, 2), (80000 * 10 / 3, 8), (120000 * 10 / 3, 4)]


def run(name, kind, mix, b, phase, seed, mult, kw):
    w = V.make_world(kind, mix, b, seed, phase=phase, req_mult=mult)
    p = F.BatchAccount14(**kw)
    w.first_W = 4
    conc = None
    if name == "bvar":
        conc = R.CONC if mult == 3 else CONC_LONG
    rate = X.simulate3(w, p, conc=conc)
    out = {"rate": rate, "spec_rounds": w.n_T, "w": {str(k): v for k, v in sorted(p.w_log.items())},
           "switches": p.n_switch, "switch_log": p.switch_log[:40], "explore": p.n_explore}
    if kw["wmode"] == "fast":
        out.update(probes=p.n_probe, screens=p.n_screen, elim=p.n_elim, elim_log=p.elim_log[:20],
                   extra_ms=round(p.extra_ms, 1), final={str(k): v["cur"] for k, v in p.fs.items()})
    if kw["wmode"] == "replay14":
        out["final"] = {str(k): v["cur"] for k, v in p.ctx14.items()}
    return out


def main():
    proc = int(sys.argv[1])
    for n, ((name, kind, mix, b, phase), seed, mult) in enumerate(JOBS):
        if n % 2 != proc:
            continue
        t0 = time.time()
        res = {lab: run(name, kind, mix, b, phase, seed, mult, kw) for lab, kw in POL}
        print(json.dumps(dict(world=name, kind=kind, mix=mix, b=b, phase=phase, seed=seed, req_mult=mult, res=res,
                              sec=round(time.time() - t0, 1), python=sys.version.split()[0])), flush=True)


if __name__ == "__main__":
    main()
