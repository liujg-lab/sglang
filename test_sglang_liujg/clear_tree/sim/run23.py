"""Round 23: W thresholds derived from the cost model vs other online W methods (ext20).
usage (conda sglang env): python run23.py <proc 0|1>

Worlds (known prefix-aware costs, strong prefix terms; seed 8; 3x the usual requests):
  base    workload mid, all prefixes short; b in {4, 8}
  het     workload het (per-request acceptance spread wide), mixed prefixes; b in {4, 8}
  phased  het + mixed prefixes, requests ordered low acceptance -> high -> mid; b in {4, 8}
  bvar    het + mixed prefixes, concurrency 2 (0-40 s) -> 8 (40-80 s) -> 4 (80-120 s)
Policies: flat, fixed 2 / 4 / 8 (references), derived, derived2, cellcal, replay14.
"""
import json, sys, time
import ext2 as X
import ext20 as V

POL = [("flat", dict(wmode="flat")), ("fixed 2", dict(wmode="fixed", fixed_W=2)), ("fixed 4", dict(wmode="fixed", fixed_W=4)),
       ("fixed 8", dict(wmode="fixed", fixed_W=8)), ("derived", dict(wmode="derived")), ("derived2", dict(wmode="derived2")),
       ("cellcal", dict(wmode="cellcal")), ("replay14", dict(wmode="replay14"))]
WORLDS = [("base", "mid", "short", 4, None), ("base", "mid", "short", 8, None),
          ("het", "het", "mixed", 4, None), ("het", "het", "mixed", 8, None),
          ("phased", "het", "mixed", 4, "lowhighmid"), ("phased", "het", "mixed", 8, "lowhighmid"),
          ("bvar", "het", "mixed", 8, None)]
CONC = [(40000, 2), (80000, 8), (120000, 4)]


def run(name, kind, mix, b, phase, kw):
    w = V.make_world(kind, mix, b, 8, phase=phase)
    p = V.BatchAccount13(**kw)
    w.first_W = 4
    rate = X.simulate3(w, p, conc=CONC if name == "bvar" else None)
    return {"rate": rate, "spec_rounds": w.n_T, "w": {str(k): v for k, v in sorted(p.w_log.items())},
            "switches": p.n_switch, "switch_log": p.switch_log[:12], "explore": p.n_explore, "deep": p.n_deep,
            "pruned": getattr(p, "n_pruned", 0)}


def main():
    proc = int(sys.argv[1])
    for n, (name, kind, mix, b, phase) in enumerate(WORLDS):
        if n % 2 != proc:
            continue
        t0 = time.time()
        res = {lab: run(name, kind, mix, b, phase, kw) for lab, kw in POL}
        print(json.dumps(dict(world=name, kind=kind, mix=mix, b=b, phase=phase, res=res, sec=round(time.time() - t0, 1),
                              python=sys.version.split()[0])), flush=True)


if __name__ == "__main__":
    main()
