"""Round 11: the adjusted algorithm (ext9.BatchAccount2) against round 10's version and tuned static configs.

Part C (seed 7, calibration): BatchAccount2 with fixed (W, K) in {2,4,8} x {16,32}, Theta per b, no switch, for every
  workload x b in {1,4,8}; the best (W, K) per (b, workload) becomes the table, with the workload standing for the
  rho class (low / mid / high).                                                       -> results/r11C.<proc>.jsonl
Part S (seeds 8-10, out of sample; scenarios of run10): all with the switch off at b = 1 (ext9.SwitchB1)
  tuned static            pure: workload's best static on the seed-7 grid; mixed/phased: round-8 best for the mix
  v1                      round 10's rule: ext8.BatchAccount, (W, K) = (8, 32) if the scenario's b is 1 else (4, 32)
  v2 Theta per b          BatchAccount2, (W, K) by the current b as v1, Theta per b with a cross-b prior
  v2 W/K by rho           BatchAccount2, table lookup by the current b and rho class, one global Theta
  v2 full                 both
Part W (high workload, b=8, seeds 7-9): drain behaviour of v1 W=2 K=16, v2 Theta per b with W=2 K=16, v2 full.
usage: run11.py <C|S|W> <proc 0|1>"""
import glob, json, sys, time
import feasibility as F
import ext7 as S
import ext8 as N
import ext9 as Q
import run8 as R8
import run9 as R9
import run10 as T

FIT = R9.FIT
WK = [(2, 16), (2, 32), (4, 16), (4, 32), (8, 16), (8, 32)]


def by_b(b, cls):
    return (8, 32) if b == 1 else (4, 32)


def load_table():
    rows = [json.loads(l) for f in sorted(glob.glob("results/r11C.*.jsonl")) for l in open(f) if l.strip()]
    table = {}
    for o in rows:
        best = max(o["res"], key=o["res"].get)
        table.setdefault(o["b"], {})[o["kind"]] = tuple(int(x) for x in best.split(","))
    return table


def table_fn(table):
    def f(b, cls):
        return table[Q.bkey(b)][cls]
    return f


def part_C(kind, b):
    out = {}
    for W, K in WK:
        w = R8.make_world(FIT, kind, b, 7)
        out[f"{W},{K}"] = T.sim(w, Q.BatchAccount2(lambda bb, c, W=W, K=K: (W, K)), W)
    return out


def part_S(job, grids, mixbest, table):
    scen, kind, b, seed = job
    mk = T.world_fn(scen, kind, b, seed)
    c = T.tuned_config(grids, mixbest, scen, kind, b)
    out = {"tuned static": T.sim(mk(), Q.SwitchB1(S.OPTDecoupled(*c), **T.R7.SW), c[0])}
    W1, K1 = by_b(b, None)
    out["v1"] = T.sim(mk(), Q.SwitchB1(N.BatchAccount(W1, K1), **T.R7.SW), W1)
    variants = {"v2 Theta per b": (by_b, True), "v2 W/K by rho": (table_fn(table), False), "v2 full": (table_fn(table), True)}
    for name, (wk, tpb) in variants.items():
        p = Q.BatchAccount2(wk, theta_per_b=tpb)
        out[name] = [T.sim(mk(), Q.SwitchB1(p, **T.R7.SW), W1), {f"{k[0]},{k[1]}": v for k, v in p.wk_log.items()}]
    return out


def part_W(seed, table):
    b, out = 8, {}
    mk = lambda: R8.make_world(FIT, "high", b, seed)
    rec = T.Rec2(N.BatchAccount(2, 16), b)
    out["v1 W=2 K=16"] = rec.summary(T.sim(mk(), rec, 2))
    rec = T.Rec2(Q.BatchAccount2(lambda bb, c: (2, 16), theta_per_b=True), b)
    out["v2 Theta per b, W=2 K=16"] = rec.summary(T.sim(mk(), rec, 2))
    rec = T.Rec2(Q.BatchAccount2(table_fn(table), theta_per_b=True), b)
    out["v2 full"] = rec.summary(T.sim(mk(), rec, 4))
    return out


def main():
    part, proc = sys.argv[1], int(sys.argv[2])
    if part == "C":
        jobs = sorted([(k, b) for k in R8.WORKLOADS for b in R8.BS], key=lambda j: -j[1])
        for n, (kind, b) in enumerate(jobs):
            if n % 2 == proc:
                t0 = time.time()
                print(json.dumps(dict(part="C", kind=kind, b=b, res=part_C(kind, b), sec=round(time.time() - t0, 1))), flush=True)
        return
    table = load_table()
    if part == "W":
        for n, seed in enumerate((7, 8, 9)):
            if n % 2 == proc:
                t0 = time.time()
                print(json.dumps(dict(part="W", seed=seed, res=part_W(seed, table), sec=round(time.time() - t0, 1))), flush=True)
        return
    grids = R8.load_grids()
    mixbest = {}
    for f in sorted(glob.glob("results/r8M.*.jsonl")):
        for l in open(f):
            if l.strip():
                o = json.loads(l)
                mixbest[o["b"]] = R8.parse(max(o["res"]["static"], key=o["res"]["static"].get))
    for n, job in enumerate(T.jobs_S()):
        if n % 2 == proc:
            t0 = time.time()
            res = part_S(job, grids, mixbest, table)
            print(json.dumps(dict(part="S", scen=job[0], kind=job[1], b=job[2], seed=job[3], res=res,
                                  sec=round(time.time() - t0, 1))), flush=True)


if __name__ == "__main__":
    main()
