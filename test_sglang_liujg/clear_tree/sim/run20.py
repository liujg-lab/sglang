"""Round 20: flat starting table; costs learned online from zero (timing probes) vs known costs (ext17.BatchAccount10).
usage (inside `conda activate sglang`): python run20.py <proc 0|1>

Worlds base, tok x0.3, change (run18); workloads mid / high; b in {1, 4, 8}; seed 8; 3x the usual requests.
Flat table: W = 8 at b = 1, 4 at b = 2-4, 4 at b >= 5, whatever the acceptance class (K from the v10 table, warm-up only).
Variants:
  flat known      the flat table, known costs, no W calibration (ext15.BatchAccount8, wmode table)
  flat online     the flat table, costs learned online with timing probes, no W calibration
  calib known     W calibration (paired replay + CS, round-19b fixes), known costs
  calib online    the same calibration, costs learned online with timing probes
  best fixed      the scenario's best fixed W in r18A (chosen there; rerun here, known costs)
"""
import json, os, sys, time
import ext10 as P
import ext15 as G
import ext17 as J
import run10 as T
import run18 as R

CAL = dict(method="paired_cs", open_delay=50, min_pairs=30, cs_type="nm")


def wk_flat(b, cls):
    if b <= 1:
        return (8, 32)
    if b <= 4:
        return (4, 32)
    return (4, 16)


def best_fixed_W():
    rows = [json.loads(l) for f in ("results/r18A.0.jsonl", "results/r18A.1.jsonl") for l in open(f)]
    out = {}
    for wd in ("base", "tok x0.3", "change"):
        for k in ("mid", "high"):
            for b in (1, 4, 8):
                rs = [r for r in rows if (r["world"], r["kind"], r["b"]) == (wd, k, b)]
                sc = {W: sum(r["res"][f"fixed {W}"]["rate"] for r in rs) for W in (2, 4, 8)}
                out[(wd, k, b)] = max(sc, key=sc.get)
    return out


def run(world, kind, b, seed, variant, bestW=None):
    w = R.make_world(world, kind, b, seed)
    lt = lambda m, W, w=w: w.c_d(m, W)
    vt = lambda bb, K, w=w: w.c0 + w.c_b * bb + w.v_tok * bb * (K + 1)
    tv = lambda bb, K, w=w: w.T_t(bb, K)
    if variant == "flat known":
        p = G.BatchAccount8(wk_flat, tv, lt, vt, wmode="table")
    elif variant == "best fixed":
        p = G.BatchAccount8(wk_flat, tv, lt, vt, wmode="fixed", fixed_W=bestW)
    elif variant == "flat online":
        p = J.BatchAccount10(wk_flat, tv, lt, vt, cost_mode="online", calibrate=False, **CAL)
    elif variant == "calib known":
        p = J.BatchAccount10(wk_flat, tv, lt, vt, cost_mode="known", calibrate=True, **CAL)
    else:
        p = J.BatchAccount10(wk_flat, tv, lt, vt, cost_mode="online", calibrate=True, **CAL)
    rate = T.sim(w, P.SwitchTok(p, T.R7.SW), 8 if b == 1 else 4)
    out = {"rate": rate, "w": {str(k): v for k, v in sorted(p.w_log.items())},
           "k": {str(k): v for k, v in sorted(p.k_log.items())}, "spec_rounds": w.n_T}
    if isinstance(p, J.BatchAccount10):
        le, ve, nl, nv = p.cost_errors(w)
        out.update(cal_rounds=p.cal_rounds, probe_rounds=p.n_probe_rounds, cost_events=p.n_cost_events,
                   events=p.events[:40], switches=p.n_switch, lt_err=le, vt_err=ve, n_lt=nl, n_vt=nv)
    return out


VARIANTS = ("flat known", "flat online", "calib known", "calib online", "best fixed")


def main():
    proc = int(sys.argv[1])
    bw = best_fixed_W()
    js = [(wd, k, b, 8) for wd in ("base", "tok x0.3", "change") for k in ("mid", "high") for b in (8, 4, 1)]
    for n, (wd, k, b, s) in enumerate(js):
        if n % 2 != proc:
            continue
        t0 = time.time()
        res = {v: run(wd, k, b, s, v, bestW=bw[(wd, k, b)]) for v in VARIANTS}
        res["best fixed"]["W"] = bw[(wd, k, b)]
        print(json.dumps(dict(world=wd, kind=k, b=b, seed=s, res=res, sec=round(time.time() - t0, 1),
                              python=sys.version.split()[0])), flush=True)


if __name__ == "__main__":
    main()
