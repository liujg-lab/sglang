"""Round 30: cost events by source (src), the narrow-switch recheck (recheck) and the tight next-layer gain (tight),
each on top of v11 + D65 + D67 (ext24.BatchAccount17; D64 candidate rule). Baselines and fixed references of the
round 29 worlds come from r29 / r29s ("G1" there = this round's baseline, reproduced bit for bit).
usage (conda sglang env): python run30.py <proc 0|1> <part A|B|C>

  A  src: the twelve RTT runs of round 29 (rtt jump / rtt alt, mid b=4 and high b=8) and the 45 round-28 world runs;
     new worlds "vchg": verify cost per token x5 at the 400th speculative round (K-dependent), mid short b=4 and high
     short b=8, with G1, src and fixed 2 / 4 / 8 (known costs).
  B  recheck on the 45 round-28 world runs.
  D  (round 30b) on the runs of A: src_pre (K-check residuals against the pre-trigger verify coefficients) on the RTT
     and vchg runs only; src_noise (K-check tolerance also 2.5 x the in-control residual sd of the bucket) on all.
     Not run in full: a smoke test showed src_pre left the misclassified case unchanged and src_noise misclassified
     the real verify change. Run instead:
  E  (round 30b) src4: K check with 4 rounds per K, tolerance from the check's own pooled sd; on the RTT and vchg runs
     and the round-28 world runs where src ran a K check.
  F  tight on the 15 long round-28 world runs (C ran only the short ones).
  C  tight on the 45 round-28 world runs; new worlds "step": verify time + 12 + 1.5 b ms for K >= 16 (round 16/17),
     mid short b = 1 and 4, with G1, tight and fixed 2 / 4 / 8 (known costs).
Short runs seeds 8, 9 (3x requests), long run seed 8 (10x).

Trimmed (2026-11-12) to save time; skipped runs are identical to G1 by construction (checked on base b=4 seed 8):
  A  src only on the round-28 world runs where G1 had a drift trigger (cost event or suppressed trigger); vchg: G1 and
     src only, no fixed references.
  B  recheck only on the round-28 world runs where G1 switched to a narrower width.
  C  tight only on the short round-28 world runs (seeds 8, 9); step: G1 and tight only, no fixed references.
"""
import json, sys, time
import ext2 as X
import ext20 as V
import ext23 as G
import ext24 as H
import run25 as R
import run26 as Q
import run27 as T
import run29 as M

RUNS = ((8, 3), (9, 3), (8, 10))
FIXED = [(f"fixed {W}", dict(cost_mode="known", wmode="fixed", fixed_W=W)) for W in (2, 4, 8)]
SRC, RC, TI = ("src", dict(src=True)), ("recheck", dict(recheck=True)), ("tight", dict(tight=True))
SRCP = ("src_pre", dict(src=True, src_pre=True))
SRCN = ("src_noise", dict(src=True, chk_z=2.5))
SRC4 = ("src4", dict(src=True, chk_n=4))
BASE = ("G1", {})


def one(part, name, kind, mix, b, phase, seed, mult, cm, kw):
    w = V.make_world(kind, mix, b, seed, phase=phase, req_mult=mult)
    if part == "S" and cm is not None:
        Q.add_change(w, cm)
    if part == "R":
        G.add_rtt(w, **M.RTT[name])
    if part == "V":
        H.add_vchg(w, cm)
    if part == "P":
        H.add_step(w)
    w.first_W = 4
    p = H.BatchAccount17(**kw)
    conc = (R.R.CONC if mult == 3 else R.CONC_LONG) if name == "bvar" else None
    rate = X.simulate3(w, p, conc=conc)
    out = {"rate": rate, "spec_rounds": w.n_T, "w": {str(k): v for k, v in sorted(p.w_log.items())},
           "k": {str(k): v for k, v in sorted(p.k_log.items())},
           "switches": p.n_switch, "switch_log": p.switch_log[:12], "probes": p.n_probe_rounds,
           "cost_events": p.n_cost_events, "ev_log": p.ev_log[:20], "suppressed": p.n_suppressed,
           "checks": p.n_checks, "offsets": p.n_offsets, "vevents": p.n_vevents, "levents": p.n_levents,
           "chk_log": p.chk_log[:20], "rc": p.n_rc, "back": p.n_back, "rc_log": p.rc_log[:12],
           "final": {str(k): v["cur"] for k, v in p.ctx14.items()}}
    if p.err_l:
        out["lt_err"] = float(sum(p.err_l) / len(p.err_l))
    return out


def _r29s():
    out = {}
    for f in ("results/r29s.0.jsonl", "results/r29s.1.jsonl"):
        for l in open(f):
            r = json.loads(l)
            out[(r["world"], r["kind"], r["b"], r["seed"], r["req_mult"])] = r["res"]["G1"]
    return out


def jobs(part):
    g1 = _r29s()
    k = lambda j: (j[1][0], j[1][1], j[1][3], j[2], j[3])
    S = [("S", (n, k, m, b, ph), s, mu, None) for (s, mu) in RUNS for (n, k, m, b, ph) in R.WORLDS]
    S += [("S", (n, k, m, b, None), s, mu, c) for (s, mu) in RUNS for (n, k, m, b, c) in Q.CHANGE + T.NEWW]
    if part == "F":
        return [(j, [TI]) for j in S if j[3] == 10]
    if part == "E":
        chk = {(r["world"], r["kind"], r["b"], r["seed"], r["req_mult"]) for f in ("results/r30a.0.jsonl", "results/r30a.1.jsonl")
               for r in map(json.loads, open(f)) if r["part"] == "S" and r["res"]["src"]["checks"] > 0}
        return [(j, [SRC4]) for j, _ in jobs("A") if j[0] in ("R", "V") or (j[1][0], j[1][1], j[1][3], j[2], j[3]) in chk]
    if part == "D":
        return [(j, [SRCP, SRCN] if j[0] in ("R", "V") else [SRCN]) for j, _ in jobs("A")]
    if part == "A":
        out = [("V", ("vchg", kd, "short", b, None), s, mu, 5.0) for (s, mu) in RUNS for (kd, b) in (("mid", 4), ("high", 8))]
        out += [("R", (nm, kd, "short", b, None), s, mu, None) for (s, mu) in RUNS for nm in M.RTT
                for (kd, b) in (("mid", 4), ("high", 8))]
        trig = [j for j in S if g1[k(j)]["cost_events"] > 0 or g1[k(j)]["suppressed"] > 0]
        return [(j, [SRC] if j[0] != "V" else [BASE, SRC]) for j in out] + [(j, [SRC]) for j in trig]
    if part == "B":
        return [(j, [RC]) for j in S if any(x[3] < x[2] for x in g1[k(j)]["switch_log"])]
    out = [("P", ("step", "mid", "short", b, None), s, mu, None) for (s, mu) in RUNS for b in (1, 4)]
    return [(j, [BASE, TI]) for j in out] + [(j, [TI]) for j in S if j[3] == 3]


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
