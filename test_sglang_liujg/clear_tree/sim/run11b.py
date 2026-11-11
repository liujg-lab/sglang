"""Round 11b: the simplified (W, K) table (keep only cells with a clear signal; all inside the cost-fit range):
b=1 -> (8, 32); b=4 -> (4, 32); b=8 -> (4, 16), or (2, 16) when the rho class is high. Theta per b. Switch off at b=1.
Same scenarios and seeds as run11 part S. usage: run11b.py <proc 0|1>"""
import json, sys, time
import ext9 as Q
import run10 as T


def wk_min(b, cls):
    k = Q.bkey(b)
    if k == 1:
        return (8, 32)
    if k == 4:
        return (4, 32)
    return (2, 16) if cls == "high" else (4, 16)


def main():
    proc = int(sys.argv[1])
    for n, job in enumerate(T.jobs_S()):
        if n % 2 != proc:
            continue
        scen, kind, b, seed = job
        t0 = time.time()
        p = Q.BatchAccount2(wk_min, theta_per_b=True)
        rate = T.sim(T.world_fn(scen, kind, b, seed)(), Q.SwitchB1(p, **T.R7.SW), 8 if b == 1 else 4)
        print(json.dumps(dict(scen=scen, kind=kind, b=b, seed=seed, res={"v2 simplified": [rate, {f"{k[0]},{k[1]}": v for k, v in p.wk_log.items()}]},
                              sec=round(time.time() - t0, 1))), flush=True)


if __name__ == "__main__":
    main()
