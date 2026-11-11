"""Simulation round 5: exit test of the D41 switch (P8/5%) - sliding window (round 4) vs disjoint blocks.
Same jobs as run4.py (13 scenarios x 2 seeds + long prefix b = 1..4 x 2 seeds); the best static (N, tau) is
read from results/r3b*.jsonl and results/r4p*.jsonl.  usage: run5.py <proc 0|1> [b]   ('b' = round 5b: the two-threshold exit test only)"""
import json, sys, time
import ext2 as X
import ext4 as Z
import run3b as R
import run4 as R4
from calibrated import CAL

VARIANTS = {"P8/5% B20x1": dict(probe_rounds=8, share=0.05, exit_blocks=1),
            "P8/5% B20x2": dict(probe_rounds=8, share=0.05, exit_blocks=2)}


VARIANTS_B = {"P8/5% B20x2+fast15": dict(probe_rounds=8, share=0.05, exit_blocks=2, exit_fast=0.15)}


def best_static_long():
    out = {}
    for f in ("results/r4p0.jsonl", "results/r4p1.jsonl"):
        for line in open(f):
            o = json.loads(line)
            if o["part"] == "long":
                out[(o["scen"], o["seed"])] = o["res"]["best static (hindsight)"][1]
    return out


def run(w, skw, bk, b_static):
    sim = lambda pol: X.simulate3(w, pol, **skw)
    N, t = bk.split(","); N, t = int(N), float(t)
    res = {}
    for vn, kv in VARIANTS.items():
        for label, mk in (("best static", lambda: X.OPT2(N, t)), ("ClearTree core", lambda: R4.core(w))):
            sw = Z.SpecSwitch2(mk(), b_static=b_static, **kv)
            res[f"{label}+{vn}"] = [sim(sw), sw.spec_time / max(1e-9, sw.total_time), sw.probes]
    return res


def main():
    proc = int(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] == "b":
        VARIANTS.clear(); VARIANTS.update(VARIANTS_B)
    b3, bl = R4.best_static_r3b(), best_static_long()
    jobs = [("sw", s, seed) for s in R.scenarios() for seed in (7, 8)]
    jobs += [("long", b, seed) for b in (1, 2, 3, 4) for seed in (7, 8)]
    for k, (part, s, seed) in enumerate(jobs):
        if k % 2 != proc:
            continue
        t0 = time.time()
        if part == "sw":
            name, wkw, skw = s
            res = run(X.World2(seed=seed, **wkw, **CAL), skw, b3[(name, seed)], 7)
        else:
            name = f"long prefix b={s} RTT=4"
            w = X.World2(seed=seed, b=s, rtt=4.0, **dict(CAL, **Z.long_kw(R4.LONG_FACTOR)))
            res = run(w, {}, bl[(name, seed)], 3)
        print(json.dumps(dict(part=part, scen=name, seed=seed, res=res, sec=round(time.time() - t0, 1))), flush=True)


if __name__ == "__main__":
    main()
