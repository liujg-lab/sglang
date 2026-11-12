"""Round 19: event-triggered W calibration methods under one budget (ext16.BatchAccount9).
usage: run19.py <proc 0|1> [B|L]   (no part: the four methods; B: the fixed paired_cs variants on the same scenarios;
                                    L: the fixed variants on the 10x long scenarios of run18 part L)

Same worlds, workloads, b and seeds as run18 part A (base, tok x3, tok x0.3, change; mid / high; b in {1, 4, 8};
seeds 8-10; 3x the usual requests), so the table, fixed-W and replay14 rows of r18A are the references.
Methods: fixed30, ipw, paired_cs, sh (see ext16). Windows open at a cell's first appearance and on a cost change;
budget 200 rounds per window; no exploration outside windows.
"""
import json, sys, time
import ext10 as P
import ext16 as H
import run10 as T
import run13 as U
import run18 as R

METHODS = ("fixed30", "ipw", "paired_cs", "sh")
# Round 19b: paired_cs with the three fixes (open 50 rounds after the event, 30 pairs, empirical-Bernstein CS), and an
# ablation with the first two fixes only (plug-in normal mixture CS).
V2 = {"paired_cs v2 eb": dict(method="paired_cs", open_delay=50, min_pairs=30, cs_type="eb"),
      "paired_cs v2 nm": dict(method="paired_cs", open_delay=50, min_pairs=30, cs_type="nm")}


def run(world, kind, b, seed, method, **extra):
    w = R.make_world(world, kind, b, seed)
    lt = lambda m, W, w=w: w.c_d(m, W)
    vt = lambda bb, K, w=w: w.c0 + w.c_b * bb + w.v_tok * bb * (K + 1)
    kw = dict(V2[method]) if method in V2 else dict(method=method)
    kw.update(extra)
    p = H.BatchAccount9(U.wk_v10, lambda bb, K, w=w: w.T_t(bb, K), lt, vt, **kw)
    rate = T.sim(w, P.SwitchTok(p, T.R7.SW), 8 if b == 1 else 4)
    return {"rate": rate, "w": {str(k): v for k, v in sorted(p.w_log.items())}, "cal_rounds": p.cal_rounds,
            "spec_rounds": w.n_T, "events": p.events[:40], "switches": p.n_switch}


def main():
    proc = int(sys.argv[1])
    part = sys.argv[2] if len(sys.argv) > 2 else ""
    if part == "L":
        R.REQ_MULT = 30
        js = [(wd, k, b, 8, v) for wd in ("base", "tok x0.3") for k in ("mid", "high") for b in (8, 4) for v in V2]
        for n, (wd, k, b, s, v) in enumerate(js):
            if n % 2 != proc:
                continue
            t0 = time.time()
            res = run(wd, k, b, s, v)
            print(json.dumps(dict(world=wd, kind=k, b=b, seed=s, variant=v, res=res, sec=round(time.time() - t0, 1))), flush=True)
        return
    methods = tuple(V2) if part == "B" else METHODS
    js = sorted(R.jobs(), key=lambda j: -j[2])
    for n, (wd, k, b, s) in enumerate(js):
        if n % 2 != proc:
            continue
        t0 = time.time()
        res = {m: run(wd, k, b, s, m) for m in methods}
        print(json.dumps(dict(world=wd, kind=k, b=b, seed=s, res=res, sec=round(time.time() - t0, 1))), flush=True)


if __name__ == "__main__":
    main()
