"""Simulation round 6b: calibration variants under the stop-boundary audit. ClearTree core with learned plug-in
parameters (v1 simulator), with the path-wise integration over the delta posterior (design doc 3.4, N_param = 8),
and with the true parameters. usage: run6b.py <proc 0|1>"""
import json, sys, time
import ext2 as X
import ext4 as Z
import ext5 as V
import run3b as R
import run4 as R4
from calibrated import CAL


def main():
    proc = int(sys.argv[1])
    jobs = [(s, seed) for s in ("wellspec b=1", "wellspec b=4", "misspec b=1", "misspec b=4", "long prefix b=2")
            for seed in (7, 8)]
    for k, (scen, seed) in enumerate(jobs):
        if k % 2 != proc:
            continue
        t0 = time.time()
        b = int(scen.split("b=")[1])
        over = dict(CAL, **Z.long_kw(R4.LONG_FACTOR)) if "long" in scen else CAL
        w = X.World2(seed=seed, b=b, rtt=4.0, misspec="misspec" in scen, **over)
        res = {}
        for name, cls in (("learned plug-in", V.CT5), ("posterior-integrated delta", V.CT6), ("true parameters", V.TrueCal)):
            p = cls(4, "learned", cost=R.cm_(w, 1.15), slow=None, n_snap=4, kreuse=10, learn_beta=True,
                    dyn_delta=True, audit_stop=True)
            res[name] = dict(rate=X.simulate3(w, p), **V.audit_summary(p.srec))
        print(json.dumps(dict(scen=scen, seed=seed, res=res, sec=round(time.time() - t0, 1))), flush=True)


if __name__ == "__main__":
    main()
