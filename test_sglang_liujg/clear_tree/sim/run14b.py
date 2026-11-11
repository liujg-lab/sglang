"""Round 14b: paired replay with measured Draft layer times (the world's c_d per active count and width, as a
deployment micro-benchmark would give) instead of the learned model. Same jobs as run14. usage: run14b.py <proc>"""
import json, sys, time
import ext10 as P
import ext12 as RP
import run10 as T
import run13 as U
import run14 as V


def main():
    proc = int(sys.argv[1])
    for n, (g, kind, b, seed, vm, dm) in enumerate(V.jobs()):
        if n % 2 != proc:
            continue
        t0 = time.time()
        w = V.make_world(kind, b, seed, vm, dm)
        p = RP.BatchAccount5(U.wk_v10, verify_time=lambda bb, K, w=w: w.T_t(bb, K), layer_time=lambda m, W, w=w: w.c_d(m, W))
        rate = T.sim(w, P.SwitchTok(p, T.R7.SW), 8 if b == 1 else 4)
        res = {"replay measured layers": {"rate": rate, "choice": {f"{k[0]},{k[1]}": v for k, v in p.choice_log.items()},
                                          "explore": p.explore_rounds, "rounds": p.rounds, "switches": p.switches}}
        print(json.dumps(dict(world=g, kind=kind, b=b, seed=seed, res=res, sec=round(time.time() - t0, 1))), flush=True)


if __name__ == "__main__":
    main()
