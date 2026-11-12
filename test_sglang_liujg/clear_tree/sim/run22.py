"""Round 22: batch-wide verify budget vs one K per batch (ext19). usage (conda sglang env): python run22.py <proc 0|1>

Scenarios: prefix mix {short, mixed} (strong prefix terms) x verify token cost x{1, 10} x workload {mid, het} x
b in {4, 8}; seed 8; 3x the usual requests. Known costs; W from the v10 table; no W calibration; no switch.
Policies: uniform (v10 one K per batch), unified_verify, unified_both.
"""
import json, sys, time
import ext19 as Z
import run10 as T
import run13 as U

ALLOCS = ("uniform", "unified_verify", "unified_both")


def run(mix, vm, kind, b, seed, alloc):
    w = Z.make_world(mix, kind, b, seed, "strong", v_mult=vm)
    p = Z.BatchAccount12(U.wk_v10, alloc=alloc)
    rate = T.sim(w, p, 8 if b == 1 else 4)
    return {"rate": rate, "spec_rounds": w.n_T, "k": {str(k): v for k, v in sorted(p.k_log.items())},
            "tot": {str(k): v for k, v in sorted(p.ktot_log.items())}}


def main():
    proc = int(sys.argv[1])
    js = [(mix, vm, kind, b, 8) for mix in ("short", "mixed") for vm in (1.0, 10.0) for kind in ("mid", "het") for b in (8, 4)]
    for n, j in enumerate(js):
        if n % 2 != proc:
            continue
        t0 = time.time()
        res = {a: run(*j, a) for a in ALLOCS}
        print(json.dumps(dict(mix=j[0], vm=j[1], kind=j[2], b=j[3], seed=j[4], res=res, sec=round(time.time() - t0, 1),
                              python=sys.version.split()[0])), flush=True)


if __name__ == "__main__":
    main()
