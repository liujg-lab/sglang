"""Simulation round 4.
part 'sw':   the 13 scenarios x 2 seeds of run3b.py, re-run with the D41 switch (SpecSwitch2) in four
             settings; the best static (N, tau) is read from results/r3b*.jsonl (same seeds, deterministic).
part 'long': long-prefix regime (ext4.long_kw(3): breakeven near b = 3), b = 1..4 at RTT 4 ms, 2 seeds.
usage: run4.py <proc 0|1>   (the two processes split the combined job list)"""
import json, sys, time
import feasibility as F
import ext as E
import ext2 as X
import ext3 as Y
import ext4 as Z
import run3b as R
from calibrated import CAL

VARIANTS = {"P15/5%": dict(probe_rounds=15, share=0.05),
            "P8/5%": dict(probe_rounds=8, share=0.05),
            "P8/10%": dict(probe_rounds=8, share=0.10),
            "P8/5%/W40": dict(probe_rounds=8, share=0.05, win=40)}
LONG_FACTOR = 3.0


def best_static_r3b():
    out = {}
    for f in ("results/r3b0.jsonl", "results/r3b1.jsonl"):
        for line in open(f):
            o = json.loads(line)
            out[(o["scen"], o["seed"])] = o["res"]["best static (hindsight)"][1]
    return out


def core(w):
    return Y.CT3(4, "learned", cost=R.cm_(w, 1.15), slow=None, n_snap=4, kreuse=10, learn_beta=True, dyn_delta=True)


def run_sw(name, wkw, skw, seed, bk):
    w = X.World2(seed=seed, **wkw, **CAL)
    sim = lambda pol: X.simulate3(w, pol, **skw)
    N, t = bk.split(","); N, t = int(N), float(t)
    res = {}
    for vn, kv in VARIANTS.items():
        for label, mk in (("OPT(8,0.4)", lambda: X.OPT2(8, 0.4)), ("best static", lambda: X.OPT2(N, t)),
                          ("ClearTree core", lambda: core(w))):
            sw = Z.SpecSwitch2(mk(), **kv)
            r = sim(sw)
            res[f"{label}+{vn}"] = [r, sw.spec_time / max(1e-9, sw.total_time), sw.probes]
    if name == "b=8 RTT=20":   # determinism check against r3b (old switch)
        res["ClearTree core+old switch (rerun)"] = sim(E.SpecSwitch(core(w)))
    return res


def run_long(b, seed):
    w = X.World2(seed=seed, b=b, rtt=4.0, **dict(CAL, **Z.long_kw(LONG_FACTOR)))
    sim = lambda pol: X.simulate3(w, pol)
    res = {"Target-only": sim(F.AR()),
           "OPT(8,0.4)": sim(X.OPT2(8, 0.4)),
           "production: OPT(8,0.4)+static switch b>=3": sim(E.StaticSwitch(X.OPT2(8, 0.4), b_min=3))}
    g = {f"{N},{t}": sim(X.OPT2(N, t)) for N, t in R.GRID}
    bk = max(g, key=g.get)
    N, t = bk.split(","); N, t = int(N), float(t)
    res["best static (hindsight)"] = [g[bk], bk]
    res["grid"] = g
    res["ClearTree core"] = sim(core(w))
    for vn, kv in VARIANTS.items():
        for label, mk in (("best static", lambda: X.OPT2(N, t)), ("ClearTree core", lambda: core(w))):
            sw = Z.SpecSwitch2(mk(), b_static=3, **kv)
            r = sim(sw)
            res[f"{label}+{vn}"] = [r, sw.spec_time / max(1e-9, sw.total_time), sw.probes]
    return res


def main():
    proc = int(sys.argv[1])
    bks = best_static_r3b()
    jobs = [("sw", s, seed) for s in R.scenarios() for seed in (7, 8)]
    jobs += [("long", b, seed) for b in (1, 2, 3, 4) for seed in (7, 8)]
    for k, (part, s, seed) in enumerate(jobs):
        if k % 2 != proc:
            continue
        t0 = time.time()
        if part == "sw":
            name, wkw, skw = s
            res = run_sw(name, wkw, skw, seed, bks[(name, seed)])
        else:
            name = f"long prefix b={s} RTT=4"
            res = run_long(s, seed)
        print(json.dumps(dict(part=part, scen=name, seed=seed, res=res, sec=round(time.time() - t0, 1))), flush=True)


if __name__ == "__main__":
    main()
