"""Round 29: guardrails for the v11 controller with online costs (ext23.BatchAccount16; D64 candidate rule, ALL_M).
usage (conda sglang env): python run29.py <proc 0|1>          (parts W, R, S with POL_S)
                          python run29.py <proc 0|1> S        (part S only, with POL_S2: v11 and G1; results/r29s.*)

Baseline "v11" = round 28's reset+cprobe (D65) + verify residual drift detection (README 3.8; rounds <= 28 had only
the layer detector). Guardrails:
  G1  cooldown 300 rounds after a cost event, doubled for an event within 2 x cooldown (max 4800); timing-probe rounds
      capped at max(54, 5% of rounds)
  G3  timing-probe rounds do not update Theta_b
  G4  timing-probe rounds of a b bucket at most one per 4 rounds of the bucket, the bucket's controller pauses until
      its probes are done; no timing probe in the first 8 speculative rounds after a gap (Target-only rounds)
  all G1 + G3 + G4
Parts (short runs seeds 8, 9 at 3x requests; long run seed 8 at 10x):
  W  speculation switch (ext10.SwitchTok, D42 for b > 1): mid, short prefixes, b = 4 / 8, Target-only time scaled so that
     fixed W = 4 (known costs) beats Target-only by ~12% (factor 0.903 / 1.13, set on seed 7). Policies v11, G3, G4,
     all; references fixed 2 / 4 / 8 with known costs and the same switch, Target-only.
  R  RTT changes: mid short b = 4 and high short b = 8 (best fixed W = 2, flat 4); "rtt jump" 4 -> 24 ms at the 400th
     speculative round, "rtt alt" 4 <-> 40 ms every 100 speculative rounds from the 400th. Policies v11 without the
     verify detector ("v11 novd"), v11, G1, G3, all; references fixed 2 / 4 / 8 with known costs.
  S  the fifteen worlds of round 28 (static and layer cost changes): v11 and all.
"""
import json, sys, time
import ext2 as X
import ext10 as P
import ext20 as V
import ext23 as G
import run7 as R7
import run25 as R
import run26 as Q
import run27 as T

G1 = dict(cooldown=True, pcap=0.05)
G3 = dict(no_theta_probe=True)
G4 = dict(spread=4, gap_hold=8)
ALL = dict(G1, **G3, **G4)
FIXED = [(f"fixed {W}", dict(cost_mode="known", wmode="fixed", fixed_W=W)) for W in (2, 4, 8)]
POL_W = [("v11", {}), ("G3", G3), ("G4", G4), ("all", ALL)] + FIXED
POL_R = [("v11 novd", dict(vdrift=False)), ("v11", {}), ("G1", G1), ("G3", G3), ("all", ALL)] + FIXED
POL_S = [("v11", {}), ("all", ALL)]
POL_S2 = [("v11", {}), ("G1", G1)]      # part S rerun after the W / R results pointed at G1 alone
SW_FAC = {4: 0.903, 8: 1.13}
RTT = {"rtt jump": dict(high=24.0, at=400), "rtt alt": dict(high=40.0, at=400, period=100)}
RUNS = ((8, 3), (9, 3), (8, 10))


def one(part, name, kind, mix, b, phase, seed, mult, cm, kw):
    w = V.make_world(kind, mix, b, seed, phase=phase, req_mult=mult)
    if cm is not None:
        Q.add_change(w, cm)
    if part == "R":
        G.add_rtt(w, **RTT[name])
    if part == "W":
        G.add_switch_world(w, SW_FAC[b])
    w.first_W = 4
    p = G.BatchAccount16(**kw)
    pol = P.SwitchTok(p, R7.SW) if part == "W" else p
    conc = (R.R.CONC if mult == 3 else R.CONC_LONG) if name == "bvar" else None
    rate = X.simulate3(w, pol, conc=conc)
    out = {"rate": rate, "spec_rounds": w.n_T, "w": {str(k): v for k, v in sorted(p.w_log.items())},
           "switches": p.n_switch, "switch_log": p.switch_log[:12], "probes": p.n_probe_rounds,
           "cost_events": p.n_cost_events, "ev_log": p.ev_log[:20], "suppressed": p.n_suppressed,
           "probe_wait": p.n_probe_wait, "resets": p.n_resets, "cprobes": p.n_cprobe, "gaps": p.n_gaps,
           "final": {str(k): v["cur"] for k, v in p.ctx14.items()}}
    if p.err_l:
        out["lt_err"] = float(sum(p.err_l) / len(p.err_l))
    if part == "W":
        lg = pol.sw.log + pol.sw1.log
        out.update(spec_share=pol.spec_time / pol.total_time, sw_log=[(x[1], x[4]) for x in lg[:20]], n_sw=len(lg),
                   ar=1000.0 * b / w.T_ar(b))
    return out


def main():
    proc = int(sys.argv[1])
    only_s = len(sys.argv) > 2 and sys.argv[2] == "S"
    jobs = [("W", ("switch", "mid", "short", b, None), s, mu, None) for (s, mu) in RUNS for b in (4, 8)]
    jobs += [("R", (nm, k, "short", b, None), s, mu, None) for (s, mu) in RUNS for nm in RTT
             for (k, b) in (("mid", 4), ("high", 8))]
    jobs += [("S", (n, k, m, b, ph), s, mu, None) for (s, mu) in RUNS for (n, k, m, b, ph) in R.WORLDS]
    jobs += [("S", (n, k, m, b, None), s, mu, c) for (s, mu) in RUNS for (n, k, m, b, c) in Q.CHANGE + T.NEWW]
    if only_s:
        jobs = [j for j in jobs if j[0] == "S"]
    for i, (part, (name, kind, mix, b, phase), seed, mult, cm) in enumerate(jobs):
        if i % 2 != proc:
            continue
        t0 = time.time()
        pols = {"W": POL_W, "R": POL_R, "S": POL_S2 if only_s else POL_S}[part]
        res = {lab: one(part, name, kind, mix, b, phase, seed, mult, cm, kw) for lab, kw in pols}
        print(json.dumps(dict(part=part, world=name, kind=kind, mix=mix, b=b, phase=phase, seed=seed, req_mult=mult,
                              cmult=cm, res=res, sec=round(time.time() - t0, 1), python=sys.version.split()[0])),
              flush=True)


if __name__ == "__main__":
    main()
