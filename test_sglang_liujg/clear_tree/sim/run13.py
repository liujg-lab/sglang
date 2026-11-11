"""Round 13: table vs online W / K (ext11.BatchAccount4), on known and unseen workloads and on a costly-verify world.

Costs: the C1 fit (run7.World7); "verify x10" multiplies the per-token verify cost (v_tok) by 10.
Workloads (acceptance intercept a, per-request mean s, Draft top-1 concentration k; k = 6 spreads the Target's token
over more Draft ranks):
  known:  low (1.0, 0.3-0.7, 1), mid (3.0, 0.5-0.9, 1), high (4.0, 0.85-0.99, 1)
  unseen: midlow (2.0, 0.4-0.8, 1), midhigh (3.5, 0.7-0.95, 1), flat (3.0, 0.5-0.9, 6), flatlow (1.5, 0.5-0.9, 6)
Policies (all: revised v10 rule = top-1 classifier, frozen costs, Theta per b, conservative b=1 switch):
  table         (8,32) at b=1, (4,32) at b 2-4, (4,16) / high (2,16) at b>=5
  online K      table W; K from {8,16,32} by the marginal value of the extra verify nodes (verify time as measured)
  online W      W from {2,4,8} by the rank-coverage chain model with the K cap (ext11, kcap=True); K from the table
  online W+K    both
  fixed (W,K)   every (W, K) in {2,4,8} x {16,32} for all b; the best one per scenario is the hindsight reference
Seeds 8-10, b in {1, 4, 8}. usage: run13.py <proc 0|1>"""
import copy, json, sys, time
import ext7 as S
import ext10 as P
import ext11 as O
import run7 as R7
import run9 as R9
import run10 as T

FIT = R9.FIT
WL = {"low": (1.0, (0.3, 0.7), 1), "mid": (3.0, (0.5, 0.9), 1), "high": (4.0, (0.85, 0.99), 1),
      "midlow": (2.0, (0.4, 0.8), 1), "midhigh": (3.5, (0.7, 0.95), 1), "flat": (3.0, (0.5, 0.9), 6),
      "flatlow": (1.5, (0.5, 0.9), 6)}
FIXED = [(W, K) for W in (2, 4, 8) for K in (16, 32)]


def make_world(kind, b, seed, verify_mult=1.0):
    a, s, k = WL[kind]
    S.set_conc(k)
    fit = copy.deepcopy(FIT)
    th = fit["fits"]["full"]["theta"]
    th[2] *= verify_mult
    return R7.World7(th, fit["a0"], fit["a_b"], seed=seed, b=b, rtt=4.0, **R7.NOISY, **S.acc_kw(a, *s))


def wk_v10(b, cls):
    if b <= 1:
        return (8, 32)
    if b <= 4:
        return (4, 32)
    return (2, 16) if cls == "high" else (4, 16)


def run(kind, b, seed, vm, name, **kw):
    w = make_world(kind, b, seed, vm)
    p = O.BatchAccount4(kw.pop("wk", wk_v10), verify_time=lambda bb, K, w=w: w.T_t(bb, K), **kw)
    rate = T.sim(w, P.SwitchTok(p, T.R7.SW), 8 if b == 1 else 4)
    return {"rate": rate, "wk": {f"{k[0]},{k[1]}": v for k, v in p.wk_log.items()}, "k": p.k_log}


def job(kind, b, seed, vm):
    out = {"table": run(kind, b, seed, vm, "table"),
           "online K": run(kind, b, seed, vm, "online K", online_K=True),
           "online W": run(kind, b, seed, vm, "online W", online_W=True),
           "online W+K": run(kind, b, seed, vm, "online W+K", online_W=True, online_K=True)}
    for W, K in FIXED:
        out[f"fixed {W},{K}"] = run(kind, b, seed, vm, "fixed", wk=lambda bb, c, W=W, K=K: (W, K))
    return out


def jobs():
    js = [(k, b, s, 1.0) for k in WL for b in (1, 4, 8) for s in (8, 9, 10)]
    js += [(k, b, s, 10.0) for k in ("low", "mid", "high") for b in (1, 4, 8) for s in (8, 9, 10)]
    return sorted(js, key=lambda j: -j[1])


def main():
    proc = int(sys.argv[1])
    for n, (kind, b, seed, vm) in enumerate(jobs()):
        if n % 2 != proc:
            continue
        t0 = time.time()
        res = job(kind, b, seed, vm)
        print(json.dumps(dict(kind=kind, b=b, seed=seed, verify_mult=vm, res=res, sec=round(time.time() - t0, 1))), flush=True)


if __name__ == "__main__":
    main()
