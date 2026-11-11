"""Round 16 (review 15). W from the v10 table; Theta per b; top-1 classifier; frozen costs; conservative b=1 switch.

Part A  consistent growth/verify value. Worlds:
          verify x10  per-token verify cost x10 (the policy's table is the world's own, i.e. known costs)
          step        verify time jumps by 4 + 0.5 b ms once K >= 16 (a padding / graph-size step), almost flat from 16
                      to 32, so the net value over K can have two peaks (known costs)
          step large  the same with a 12 + 1.5 b ms jump
        Policies (ext13.BatchAccount6): early (current text), enum (all K compared), consistent (growth and verify on
        one value). Workloads mid, high, flat; b in {1, 4, 8}; seeds 8-10.
Part B  verify-time drift at run time: from spec round 150 on, the per-token verify cost becomes 10 x. Policies (all
        k_mode enum): oracle table (follows the change), stale table (pre-change table, no detection), stale table +
        drift detection. Workloads mid, high; b in {1, 4, 8}; seeds 8-10.
Part B2 drift detection v2 (ext13.BatchAccount6L: sustained level shift of the K in use, then a burst over all K).
usage: run16.py <A|B|B2> <proc 0|1>"""
import copy, json, sys, time
import ext7 as S
import ext10 as P
import ext13 as D
import run7 as R7
import run10 as T
import run13 as U

FIT = U.FIT


class World16(R7.World7):
    def __init__(self, theta, a0, a_b, step=False, switch_round=None, mult=10.0, step_size=(4.0, 0.5), **kw):
        super().__init__(theta, a0, a_b, **kw)
        self.step, self.switch_round, self.mult, self.n_T, self.step_size = step, switch_round, mult, 0, step_size

    def base_T(self, b, K):
        t = R7.World7.T_t(self, b, K)
        if self.step and K >= 16:
            t += self.step_size[0] + self.step_size[1] * b
        return t

    def T_t(self, b, K):
        t = self.base_T(b, K)
        if self.switch_round is not None and self.n_T >= self.switch_round:
            t += (self.mult - 1.0) * self.v_tok * b * (K + 1)
        return t

    def real_T(self, rng, b, K):
        self.n_T += 1
        return super().real_T(rng, b, K)


def make_world(kind, b, seed, vm=1.0, step=False, switch_round=None, step_size=(4.0, 0.5)):
    a, s, k = U.WL[kind]
    S.set_conc(k)
    fit = copy.deepcopy(FIT)
    th = fit["fits"]["full"]["theta"]
    th[2] *= vm
    return World16(th, fit["a0"], fit["a_b"], step=step, switch_round=switch_round, step_size=step_size, seed=seed, b=b, rtt=4.0,
                   **R7.NOISY, **S.acc_kw(a, *s))


def run(w, b, table, **kw):
    p = D.BatchAccount6(U.wk_v10, table, **kw)
    rate = T.sim(w, P.SwitchTok(p, T.R7.SW), 8 if b == 1 else 4)
    return {"rate": rate, "k": {str(k): v for k, v in sorted(p.k_log.items())},
            "waste": round(p.waste["beyond"] / max(1e-9, p.waste["total"]), 4), "probes": p.n_probe, "bursts": p.n_burst,
            "corr": {f"{k[0]},{k[1]}": round(v, 2) for k, v in p.corr.items()}}


def job_A(world, kind, b, seed):
    out = {}
    for mode in ("early", "enum", "consistent"):
        w = make_world(kind, b, seed, vm=10.0 if world == "verify x10" else 1.0, step=world.startswith("step"),
                       step_size=(12.0, 1.5) if world == "step large" else (4.0, 0.5))
        out[mode] = run(w, b, lambda bb, K, w=w: w.T_t(bb, K), k_mode=mode)
    return out


VARIANTS_B = ("oracle table", "stale table", "stale table + drift")


def job_B(kind, b, seed):
    out = {}
    for name in VARIANTS_B:
        w = make_world(kind, b, seed, switch_round=150)
        table = (lambda bb, K, w=w: w.T_t(bb, K)) if name == "oracle table" else (lambda bb, K, w=w: w.base_T(bb, K))
        if name == "stale table + drift v2":
            p = D.BatchAccount6L(U.wk_v10, table, k_mode="enum")
            rate = T.sim(w, P.SwitchTok(p, T.R7.SW), 8 if b == 1 else 4)
            out[name] = {"rate": rate, "k": {str(k): v for k, v in sorted(p.k_log.items())}, "probes": 0, "bursts": p.n_burst,
                         "corr": {f"{k[0]},{k[1]}": round(v, 2) for k, v in p.corr.items()}}
        else:
            out[name] = run(w, b, table, k_mode="enum", drift=(name.endswith("drift")))
        out[name]["switched_at"] = 150
        out[name]["spec_rounds"] = w.n_T
    return out


def main():
    part, proc = sys.argv[1], int(sys.argv[2])
    global VARIANTS_B
    if part == "B2":
        VARIANTS_B = ("stale table + drift v2",)
    if part == "A":
        jobs = [(wd, k, b, s) for wd in ("verify x10", "step", "step large") for k in ("mid", "high", "flat") for b in (1, 4, 8) for s in (8, 9, 10)]
    else:
        jobs = [("drift", k, b, s) for k in ("mid", "high") for b in (1, 4, 8) for s in (8, 9, 10)]
    jobs.sort(key=lambda j: -j[2])
    for n, (wd, kind, b, seed) in enumerate(jobs):
        if n % 2 != proc:
            continue
        t0 = time.time()
        res = job_A(wd, kind, b, seed) if part == "A" else job_B(kind, b, seed)
        print(json.dumps(dict(world=wd, kind=kind, b=b, seed=seed, res=res, sec=round(time.time() - t0, 1))), flush=True)


if __name__ == "__main__":
    main()
