"""Round 14: table vs online K vs paired replay of (W, K) (ext12.BatchAccount5), plus a fixed (W, K) grid as the
hindsight reference. All policies use the revised v10 rule (top-1 classifier, frozen costs, Theta per b, conservative
b=1 switch).

Worlds (C1 costs unless noted):
  unseen       workloads midlow, midhigh, flat, flatlow (run13)
  verify x10   per-token verify cost x10; workloads mid, high
  draft x0.5   every Draft layer cost term x0.5 (Draft twice as fast); workloads mid, high
  draft x2     Draft twice as slow; workloads mid, high
b in {1, 4, 8}, seeds 8-10. usage: run14.py <proc 0|1>"""
import copy, json, sys, time
import ext7 as S
import ext10 as P
import ext11 as O
import ext12 as RP
import run7 as R7
import run10 as T
import run13 as U

FIT = U.FIT
WORLDS = {"unseen": ((1.0, 1.0), ("midlow", "midhigh", "flat", "flatlow")),
          "verify x10": ((10.0, 1.0), ("mid", "high")),
          "draft x0.5": ((1.0, 0.5), ("mid", "high")),
          "draft x2": ((1.0, 2.0), ("mid", "high"))}
FIXED = [(W, K) for W in (2, 4, 8) for K in (16, 32)]


def make_world(kind, b, seed, vm, dm):
    a, s, k = U.WL[kind]
    S.set_conc(k)
    fit = copy.deepcopy(FIT)
    th = fit["fits"]["full"]["theta"]
    th[2] *= vm
    th[3] *= dm; th[4] *= dm; th[5] *= dm
    return R7.World7(th, fit["a0"], fit["a_b"], seed=seed, b=b, rtt=4.0, **R7.NOISY, **S.acc_kw(a, *s))


def run(kind, b, seed, vm, dm, policy, wk=None):
    w = make_world(kind, b, seed, vm, dm)
    vt = lambda bb, K, w=w: w.T_t(bb, K)
    if policy == "table":
        p = O.BatchAccount4(U.wk_v10, verify_time=vt)
    elif policy == "online K":
        p = O.BatchAccount4(U.wk_v10, verify_time=vt, online_K=True)
    elif policy == "replay":
        p = RP.BatchAccount5(U.wk_v10, verify_time=vt)
    else:
        p = O.BatchAccount4(wk, verify_time=vt)
    rate = T.sim(w, P.SwitchTok(p, T.R7.SW), 8 if b == 1 else 4)
    out = {"rate": rate}
    if policy == "replay":
        out.update(choice={f"{k[0]},{k[1]}": v for k, v in p.choice_log.items()}, explore=p.explore_rounds,
                   rounds=p.rounds, switches=p.switches, audit=p.naudit)
    return out


def job(kind, b, seed, vm, dm):
    out = {pol: run(kind, b, seed, vm, dm, pol) for pol in ("table", "online K", "replay")}
    for W, K in FIXED:
        out[f"fixed {W},{K}"] = run(kind, b, seed, vm, dm, "fixed", wk=lambda bb, c, W=W, K=K: (W, K))
    return out


def jobs():
    js = []
    for g, ((vm, dm), kinds) in WORLDS.items():
        js += [(g, k, b, s, vm, dm) for k in kinds for b in (1, 4, 8) for s in (8, 9, 10)]
    return sorted(js, key=lambda j: -j[2])


def main():
    proc = int(sys.argv[1])
    for n, (g, kind, b, seed, vm, dm) in enumerate(jobs()):
        if n % 2 != proc:
            continue
        t0 = time.time()
        res = job(kind, b, seed, vm, dm)
        print(json.dumps(dict(world=g, kind=kind, b=b, seed=seed, res=res, sec=round(time.time() - t0, 1))), flush=True)


if __name__ == "__main__":
    main()
