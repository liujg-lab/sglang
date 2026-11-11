"""Round 12b: isolate the one-step lag and the calibration fixes; test the top-1 classifier. All variants use the
simplified table, Theta per b and the conservative b=1 switch (SwitchTok). Seeds 8-10, run10's 14 scenarios.
  base             = run12's "switch tok b=1" (round 11 rule + conservative b=1 switch)
  freeze only / c decay 0.95 only / c decay 0.99 only
  no-lag oracle    the next layer's own Draft-score increment (cloned tree grown one layer) instead of the last one
  top1 classifier  high / other by the first-position top-1 hit rate (threshold 0.84 = midpoint of mid and high)
usage: run12b.py <proc 0|1>"""
import json, sys, time
import ext10 as P
import run10 as T
import run11b as RB

VARIANTS = {"base": {}, "freeze only": dict(freeze=True), "c decay 0.95 only": dict(c_decay=0.95),
            "c decay 0.99 only": dict(c_decay=0.99), "no-lag oracle": dict(lookahead=True),
            "top1 classifier": dict(cls_mode="top1", th_top1=0.84)}


def main():
    proc = int(sys.argv[1])
    for n, job in enumerate(T.jobs_S()):
        if n % 2 != proc:
            continue
        scen, kind, b, seed = job
        t0 = time.time()
        out = {}
        for name, kw in VARIANTS.items():
            p = P.BatchAccount3(RB.wk_min, **kw)
            pol = P.SwitchTok(p, T.R7.SW)
            out[name] = {"rate": T.sim(T.world_fn(scen, kind, b, seed)(), pol, 8 if b == 1 else 4),
                         "wk": {f"{k[0]},{k[1]}": v for k, v in p.wk_log.items()}}
        print(json.dumps(dict(scen=scen, kind=kind, b=b, seed=seed, res=out, sec=round(time.time() - t0, 1))), flush=True)


if __name__ == "__main__":
    main()
