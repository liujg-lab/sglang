"""Round 32: closing the gaps of round 31 (ext26.BatchAccount19). "cur" = the design as of D69 (as in round 31).
usage (conda sglang env): python run32.py <proc 0|1> <part P|S|K|V|X|Y|Y2|V2>

  P  cost-event limiter. Policies: cur (cooldown, D67), nocd (no limiter), ebkt (token bucket: 3 events, one back
     every 300 rounds).
       rtt jump / rtt alt of round 29 (mid b = 4, high b = 8; cur of rtt alt comes from r31c)
       lalt x3: layer cost x3 / x1 alternating every 100 speculative rounds from round 400 (mid b = 8, high b = 8)
       the cooldown worlds of round 31 A: ebkt only (cur, nocd in r31a)
  S  nocd and ebkt on the round-28 world runs where cur suppressed a trigger (the others are identical to cur).
  K  continuous timing probes for K (cpk: a K not run for 300 rounds of the b bucket is forced for 2 rounds; residuals
     of the same sign beyond the drift tolerance start a K check). k32: K = 32 pays 15 + 2 b ms extra until round 400
     (mid, short prefixes, b = 1 and 4): cur, cpk. Cost without a change: cpk on base mid b = 4, base mid b = 8 and
     high b = 8 (cur from r31z).
  V  the combined candidate "v12" (results/v12.json: no cooldown, rob, perk, K-check timeout 40, first-appearance
     probes outside the probe cap) on every world of rounds 28-32: the 45 round-28 world runs, rtt jump / alt, vchg
     (mid b = 4, high b = 8, mid b = 8; with cur), step (with cur), the cooldown worlds, lalt x3, k32 (also the
     replay controller with known costs), balt cb / rtt, the tail worlds. cur comes from r31 / r32 where it exists.
  Y  stage 1 of the revised candidate v12b (results/v12b.json: v12 without perk, with rob_cont) on phased b = 8, the
     tail worlds, rtt jump, k32 and step.
  V2 the final candidate v12c on every world of V.
  Y2 stage 1 of v12c (results/v12c.json: v12b with rob_stale = 50 in place of rob_cont, and rob_hys) on phased
     b = 8 and the tail worlds.
  X  ablation where v12 was slightly worse than cur (RTT, k32, step, vchg): v12 without perk; on rtt jump also v12
     without rob.
Short runs seeds 8, 9 (3x requests), long run seed 8 (10x), unless stated.
"""
import json, sys, time
import ext2 as X
import ext20 as V
import ext23 as G
import ext24 as H
import ext25 as N
import ext26 as P
import run25 as R
import run26 as Q
import run27 as T
import run29 as M
import run31 as R31

RUNS = ((8, 3), (9, 3), (8, 10))
CUR = ("cur", {})
NOCD = ("nocd", dict(cooldown=False))
EBKT = ("ebkt", dict(cooldown=False, ebkt=(3, 300)))
CPK = ("cpk", dict(cpk=(300, 2)))
FIXED = [(f"fixed {W}", dict(cost_mode="known", wmode="fixed", fixed_W=W)) for W in (2, 4, 8)]
V12 = ("v12", dict(cooldown=False, ebkt=(3, 300)))     # replaced once round 31 / 32 results are in (see V12_FILE)
V12_FILE = "results/v12.json"


def v12():
    try:
        return ("v12", json.load(open(V12_FILE)))
    except FileNotFoundError:
        return V12


def one(part, name, kind, mix, b, phase, seed, mult, cm, kw):
    kw = dict(kw)
    for k in ("ebkt", "cpk"):
        if kw.get(k):
            kw[k] = tuple(kw[k])
    w = V.make_world(kind, mix, b, seed, phase=phase, req_mult=mult)
    conc = None
    if part == "S":
        if cm is not None:
            Q.add_change(w, cm)
        if name == "bvar":
            conc = R.R.CONC if mult == 3 else R.CONC_LONG
    elif part == "R":
        G.add_rtt(w, **M.RTT[name])
    elif part == "L":
        P.add_lalt(w, mult=3.0)
    elif part == "A":
        if R31.AW[name].get("v"):
            H.add_vchg(w, 10.0, at=R31.AW[name]["v"])
        N.add_changes(w, R31.AW[name]["l"])
    elif part == "K":
        P.add_k32(w, 400)
    elif part == "VC":
        H.add_vchg(w, cm)
    elif part == "ST":
        H.add_step(w)
    elif part == "B":
        conc = R31.balt(mult)
        if name == "balt cb":
            N.add_bchg(w, 8.0, at=400)
        else:
            G.add_rtt(w, high=24.0, at=400)
    elif part == "D":
        if cm is not None:
            Q.add_change(w, cm)
        N.add_tail(w)
    w.first_W = 4
    p = P.BatchAccount19(**kw)
    rate = X.simulate3(w, p, conc=conc)
    out = {"rate": rate, "spec_rounds": w.n_T, "w": {str(k): v for k, v in sorted(p.w_log.items())},
           "k": {str(k): v for k, v in sorted(p.k_log.items())},
           "switches": p.n_switch, "switch_log": p.switch_log[:12], "sw_se": p.sw_se[:12], "probes": p.n_probe_rounds,
           "cost_events": p.n_cost_events, "ev_log": p.ev_log[:20], "suppressed": p.n_suppressed,
           "checks": p.n_checks, "offsets": p.n_offsets, "vevents": p.n_vevents, "levents": p.n_levents,
           "chk_log": p.chk_log[:20], "pend": p.n_pend, "chk_supp": p.n_chk_supp, "se_block": p.n_se_block,
           "skip": p.n_skip, "rescreen": p.n_rescreen, "probe_wait": p.n_probe_wait, "chk_abort": p.n_chk_abort,
           "cpk": p.n_cpk, "cpk_trig": p.n_cpk_trig,
           "phi": {str(k): round(v[0] / max(v[1], 1e-9), 3) for k, v in p.phi_st.items()},
           "final": {str(k): v["cur"] for k, v in p.ctx14.items()}}
    if p.err_l:
        out["lt_err"] = float(sum(p.err_l) / len(p.err_l))
    return out


def s_jobs():
    S = [("S", (n, k, m, b, ph), s, mu, None) for (s, mu) in RUNS for (n, k, m, b, ph) in R.WORLDS]
    return S + [("S", (n, k, m, b, None), s, mu, c) for (s, mu) in RUNS for (n, k, m, b, c) in Q.CHANGE + T.NEWW]


def rtt_jobs(names):
    return [("R", (nm, kd, "short", b, None), s, mu, None) for (s, mu) in RUNS for nm in names
            for (kd, b) in (("mid", 4), ("high", 8))]


def jobs(part):
    if part == "P":
        out = [(j, [CUR, NOCD, EBKT]) for j in rtt_jobs(["rtt jump"])]
        out += [(j, [NOCD, EBKT]) for j in rtt_jobs(["rtt alt"])]
        out += [(("L", ("lalt x3", kd, "short", 8, None), s, mu, None), [CUR, NOCD, EBKT]) for (s, mu) in RUNS
                for kd in ("mid", "high")]
        out += [(("A", (nm, "mid", "short", 8, None), s, 3, None), [EBKT]) for s in (8, 9, 10) for nm in R31.AW]
        return out
    if part == "S":
        z = R31._z()
        return [(j, [NOCD, EBKT]) for j in s_jobs() if z[(j[1][0], j[1][1], j[1][3], j[2], j[3])]["suppressed"] > 0]
    if part == "K":
        out = [(("K", ("k32", "mid", "short", b, None), s, mu, None), [CUR, CPK]) for (s, mu) in RUNS for b in (1, 4)]
        out += [(("S", (n, k, "short", b, None), s, mu, None), [CPK]) for (s, mu) in RUNS
                for (n, k, b) in (("base", "mid", 4), ("base", "mid", 8), ("high", "high", 8))]
        return out
    if part == "V":
        vp = [v12()]
        out = [(j, vp) for j in s_jobs()]
        out += [(j, vp) for j in rtt_jobs(["rtt jump", "rtt alt"])]
        out += [(("VC", ("vchg", kd, "short", b, None), s, mu, 5.0), [CUR] + vp) for (s, mu) in RUNS
                for (kd, b) in (("mid", 4), ("high", 8), ("mid", 8))]
        out += [(("ST", ("step", "mid", "short", b, None), s, mu, None), [CUR] + vp) for (s, mu) in RUNS for b in (1, 4)]
        out += [(("A", (nm, "mid", "short", 8, None), s, 3, None), vp) for s in (8, 9, 10) for nm in R31.AW]
        out += [(("L", ("lalt x3", kd, "short", 8, None), s, mu, None), vp) for (s, mu) in RUNS for kd in ("mid", "high")]
        out += [(("K", ("k32", "mid", "short", b, None), s, mu, None), vp + [("known", dict(cost_mode="known"))])
                for (s, mu) in RUNS for b in (1, 4)]
        out += [(("B", (nm, "mid", "short", 8, None), s, mu, None), vp) for (s, mu) in RUNS for nm in ("balt cb", "balt rtt")]
        tw = [("tail base", "mid", "short", 4, None), ("tail high", "high", "short", 8, None),
              ("tail het", "het", "mixed", 4, None), ("tail chg8 x10", "mid", "short", 8, 10.0)]
        out += [(("D", (n, k, m, b, None), s, 3, c), vp) for s in (8, 9) for (n, k, m, b, c) in tw]
        out += [(("D", (n, k, m, b, None), 8, 10, c), vp) for (n, k, m, b, c) in tw[:2]]
        return out
    if part == "X":     # ablation of v12 where it was slightly worse than cur
        va = ("v12-perk", dict(cooldown=False, rob=True, chk_to=40, cap_free=True))
        vb = ("v12-rob", dict(cooldown=False, perk=True, chk_to=40, cap_free=True))
        out = [(j, [va, vb] if j[1][0] == "rtt jump" else [va]) for j in rtt_jobs(["rtt jump", "rtt alt"])]
        out += [(("K", ("k32", "mid", "short", b, None), s, mu, None), [va]) for (s, mu) in RUNS for b in (1, 4)]
        out += [(("ST", ("step", "mid", "short", b, None), s, mu, None), [va]) for (s, mu) in RUNS for b in (1, 4)]
        out += [(("VC", ("vchg", kd, "short", b, None), s, mu, 5.0), [va]) for (s, mu) in RUNS
                for (kd, b) in (("mid", 4), ("high", 8), ("mid", 8))]
        return out
    if part == "V2":    # the final candidate v12c on every world of V (cur, v12 and known are in r32v)
        return [(j, [("v12c", json.load(open("results/v12c.json")))]) for j, _ in jobs("V")]
    if part == "Y2":    # stage 1 of v12c (results/v12c.json) on phased b = 8 and the tail worlds
        return [(j, [("v12c", json.load(open("results/v12c.json")))]) for j, _ in jobs("Y") if j[0] in ("S", "D")]
    if part == "Y":     # stage 1 of the revised candidate (results/v12b.json) where v12 lost or rob mattered
        vp = [("v12b", json.load(open("results/v12b.json")))]
        out = [(("S", ("phased", "het", "mixed", 8, "lowhighmid"), s, mu, None), vp) for (s, mu) in RUNS]
        tw = [("tail base", "mid", "short", 4, None), ("tail high", "high", "short", 8, None),
              ("tail het", "het", "mixed", 4, None), ("tail chg8 x10", "mid", "short", 8, 10.0)]
        out += [(("D", (n, k, m, b, None), s, 3, c), vp) for s in (8, 9) for (n, k, m, b, c) in tw]
        out += [(("D", (n, k, m, b, None), 8, 10, c), vp) for (n, k, m, b, c) in tw[:2]]
        out += [(j, vp) for j in rtt_jobs(["rtt jump"])]
        out += [(("K", ("k32", "mid", "short", b, None), s, mu, None), vp) for (s, mu) in RUNS for b in (1, 4)]
        out += [(("ST", ("step", "mid", "short", b, None), s, mu, None), vp) for (s, mu) in RUNS for b in (1, 4)]
        return out
    raise SystemExit("part?")


def main():
    proc, part = int(sys.argv[1]), sys.argv[2]
    for i, ((pt, (name, kind, mix, b, phase), seed, mult, cm), pols) in enumerate(jobs(part)):
        if i % 2 != proc:
            continue
        t0 = time.time()
        res = {lab: one(pt, name, kind, mix, b, phase, seed, mult, cm, kw) for lab, kw in pols}
        print(json.dumps(dict(part=pt, run=part, world=name, kind=kind, mix=mix, b=b, phase=phase, seed=seed,
                              req_mult=mult, cmult=cm, res=res, sec=round(time.time() - t0, 1),
                              python=sys.version.split()[0])), flush=True)


if __name__ == "__main__":
    main()
