"""Round 21: prefix-length-aware cost model learned online (ext18). usage (conda sglang env): python run21.py <proc 0|1>

Scenarios: prefix mix x strength in {(short, strong), (mixed, strong), (long, strong), (mixed, mid)}; workload mid;
b in {1, 4, 8}; seed 8; 3x the usual requests. W from the v10 table, K online, no W calibration, no speculation switch
(the Target-only path has no prefix term in this world, so the switch would be biased).
Policies: known (true prefix-aware costs, net-gain order), blind (r20 online EMA, no prefix), aware (online regression
with a prefix term, net-gain order), aware_noorder (same estimates, order by g).
"""
import json, sys, time
import ext18 as H
import run10 as T
import run13 as U

SCEN = [("short", "strong"), ("mixed", "strong"), ("long", "strong"), ("mixed", "mid")]
MODES = ("known", "blind", "aware", "aware_noorder")


def run(mix, strength, kind, b, seed, mode):
    w = H.make_world(mix, kind, b, seed, strength)
    p = H.BatchAccount11(U.wk_v10, cost_mode=mode)
    rate = T.sim(w, p, 8 if b == 1 else 4)
    mean = lambda x: float(sum(x) / len(x)) if x else None
    half = lambda x: x[len(x) // 2:]
    return {"rate": rate, "k": {str(k): v for k, v in sorted(p.k_log.items())}, "spec_rounds": w.n_T,
            "probe_rounds": p.n_probe_rounds, "cost_events": p.n_cost_events,
            "err_l": mean(p.err_l), "err_l_late": mean(half(p.err_l)), "err_v": mean(p.err_v),
            "err_v_late": mean(half(p.err_v))}


def main():
    proc = int(sys.argv[1])
    js = [(mix, st, "mid", b, 8) for mix, st in SCEN for b in (8, 4, 1)]
    for n, j in enumerate(js):
        if n % 2 != proc:
            continue
        t0 = time.time()
        res = {m: run(*j, m) for m in MODES}
        print(json.dumps(dict(mix=j[0], strength=j[1], kind=j[2], b=j[3], seed=j[4], res=res,
                              sec=round(time.time() - t0, 1), python=sys.version.split()[0])), flush=True)


if __name__ == "__main__":
    main()
