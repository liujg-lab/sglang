"""Round 17 (review 16). usage: run17.py <S1|S2|S3> <proc 0|1> [one]

S1  cross-K stopping. Worlds of round 16 part A (verify x10; step: +4 + 0.5 b ms once K >= 16; step large:
    +12 + 1.5 b ms), workloads mid / high / flat, b in {1, 4, 8}, seeds 8-10. Policies (ext14.BatchAccount7, table =
    the world's own verify time): v10 (+ diag: cross-K stop counts), opt2 (two-layer optimistic option), true1
    (oracle next-layer increase), true2 (oracle next two layers).
S2  cross-K marginal calibration. Workloads low / mid / high x verify x1 / x10 x b in {1, 4, 8} x seeds 8-10.
    Policies: v10 (+ same-tree replay records), v10 + f_band online (+ replay records).
S3  an unused K becomes cheaper. The verify time of K = 32 carries a slow-path penalty of 15 + 2 b ms (and the
    deploy-time table was measured with it). World "cheap32": the penalty disappears from spec round 150 on;
    world "nochange": it stays. Workloads mid / high, b in {1, 4, 8}, seeds 8-10, 3x the usual number of requests.
    Variants: oracle table (cheap32 only), stale table, stale + drift v2 (interleaved burst), + random probes 1% / 3%,
    + near-tie probes (believed J within 2% of the believed tokens).
"""
import copy, json, sys, time
import ext7 as S
import ext10 as P
import ext14 as E
import run7 as R7
import run10 as T
import run13 as U
import run16 as R16

FIT = U.FIT


class World17(R16.World16):
    def __init__(self, *a, pen=(15.0, 2.0), cheap_round=None, **kw):
        super().__init__(*a, **kw)
        self.pen, self.cheap_round = pen, cheap_round

    def penalty(self, b, K):
        return (self.pen[0] + self.pen[1] * b) if K >= 32 else 0.0

    def base_T(self, b, K):
        t = R16.World16.base_T(self, b, K)
        if self.cheap_round is None or self.n_T < self.cheap_round:
            t += self.penalty(b, K)
        return t

    def stale_T(self, b, K):
        return R16.World16.base_T(self, b, K) + self.penalty(b, K)


def world17(kind, b, seed, cheap_round):
    a, s, k = U.WL[kind]
    S.set_conc(k)
    fit = copy.deepcopy(FIT)
    th = fit["fits"]["full"]["theta"]
    return World17(th, fit["a0"], fit["a_b"], cheap_round=cheap_round, seed=seed, b=b, rtt=4.0,
                   n_req=3 * max(16, 8 * b), **R7.NOISY, **S.acc_kw(a, *s))


def sim(w, p, b):
    return T.sim(w, P.SwitchTok(p, T.R7.SW), 8 if b == 1 else 4)


def job_S1(world, kind, b, seed):
    out = {}
    for look in (None, "opt2", "true1", "true2"):
        w = R16.make_world(kind, b, seed, vm=10.0 if world == "verify x10" else 1.0, step=world.startswith("step"),
                           step_size=(12.0, 1.5) if world == "step large" else (4.0, 0.5))
        p = E.BatchAccount7(U.wk_v10, lambda bb, K, w=w: w.T_t(bb, K), look=look, diag=(look is None))
        rate = sim(w, p, b)
        out[look or "v10"] = {"rate": rate, "k": {str(k): v for k, v in sorted(p.k_log.items())}, "dg": p.dg,
                              "tok": p.tok_total}
    return out


def job_S2(kind, vm, b, seed):
    out = {}
    for name, fb in (("v10", None), ("v10 + f_band", "online")):
        w = R16.make_world(kind, b, seed, vm=vm)
        p = E.BatchAccount7(U.wk_v10, lambda bb, K, w=w: w.T_t(bb, K), fband=fb, replay=True)
        rate = sim(w, p, b)
        out[name] = {"rate": rate, "k": {str(k): v for k, v in sorted(p.k_log.items())},
                     "f": {str(k): round(v, 3) for k, v in p.f.items()}, "replay": p.replay_summary()}
    return out


S3_VARIANTS = {
    "oracle table": dict(),
    "stale table": dict(),
    "v2": dict(probe_p=0.0),
    "v2 + probe 1%": dict(probe_p=0.01),
    "v2 + probe 3%": dict(probe_p=0.03),
    "v2 + near-tie": dict(neartie=0.02),
}


def job_S3(world, kind, b, seed):
    out = {}
    for name, kw in S3_VARIANTS.items():
        if world == "nochange" and name == "oracle table":
            continue
        w = world17(kind, b, seed, 150 if world == "cheap32" else None)
        if name == "oracle table":
            p = E.BatchAccount7(U.wk_v10, lambda bb, K, w=w: w.T_t(bb, K))
        elif name == "stale table":
            p = E.BatchAccount7(U.wk_v10, lambda bb, K, w=w: w.stale_T(bb, K))
        else:
            p = E.BatchAccount7L(U.wk_v10, lambda bb, K, w=w: w.stale_T(bb, K), **kw)
        rate = sim(w, p, b)
        out[name] = {"rate": rate, "k": {str(k): v for k, v in sorted(p.k_log.items())}, "bursts": p.n_burst,
                     "probes": getattr(p, "n_probe", 0), "probe_trig": getattr(p, "n_probe_trig", 0),
                     "corr": {f"{k[0]},{k[1]}": round(v, 2) for k, v in p.corr.items()}, "spec_rounds": w.n_T}
    return out


def jobs(part):
    if part == "S1":
        js = [("S1", wd, k, b, s) for wd in ("verify x10", "step", "step large") for k in ("mid", "high", "flat")
              for b in (1, 4, 8) for s in (8, 9, 10)]
    elif part == "S2":
        js = [("S2", k, vm, b, s) for k in ("low", "mid", "high") for vm in (1.0, 10.0) for b in (1, 4, 8) for s in (8, 9, 10)]
    else:
        js = [("S3", wd, k, b, s) for wd in ("cheap32", "nochange") for k in ("mid", "high") for b in (1, 4, 8) for s in (8, 9, 10)]
    return sorted(js, key=lambda j: -j[3])


def main():
    part, proc = sys.argv[1], int(sys.argv[2])
    one = len(sys.argv) > 3
    for n, j in enumerate(jobs(part)):
        if n % 2 != proc:
            continue
        t0 = time.time()
        if part == "S1":
            res = job_S1(*j[1:])
            head = dict(world=j[1], kind=j[2], b=j[3], seed=j[4])
        elif part == "S2":
            res = job_S2(*j[1:])
            head = dict(kind=j[1], vm=j[2], b=j[3], seed=j[4])
        else:
            res = job_S3(*j[1:])
            head = dict(world=j[1], kind=j[2], b=j[3], seed=j[4])
        print(json.dumps(dict(**head, res=res, sec=round(time.time() - t0, 1))), flush=True)
        if one:
            break


if __name__ == "__main__":
    main()
