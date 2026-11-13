"""Round 31: checks of the v11 design after D66-D69 (ext25.BatchAccount18). "cur" = the current design
(v11 + D65 + D67 + D68 + D69 = ext24.BatchAccount17 with src, chk_n = 4, tight; reproduced bit for bit).
usage (conda sglang env): python run31.py <proc 0|1> <part Z|A|T|B|C|D>

  Z  cur on the 45 round-28 world runs (baseline of B and D; logs gain and SE of every width switch).
  A  cooldown (D67) with two cost changes close together; mid acceptance, short prefixes, b = 8, short runs seeds 8, 9, 10:
       v+l x3 / x10   verify cost per token x10 at speculative round 250, layer cost x3 / x10 at round 400
       l+l x10        layer cost x0.3 at round 250, x10 of the original at round 400
       3chg x10       layer cost x0.3 at round 250, back at round 600, x10 at round 700
     policies: cur, pend (suppressed events fire when the cooldown ends), psrc (cooldown per source), nocd (no cooldown).
  T  probe cap on a long run: "rtt alt" (4 <-> 40 ms every 100 speculative rounds) starting late, at round 6000 (mid
     b = 4, 20x requests, seeds 8, 9) / 3500 (high b = 8, 20x, seed 8). Policies: cur (cumulative cap), bucket.
  B  se2 (switch needs gain > 2 SE) on the round-28 world runs where cur made a switch with gain <= 2 SE (the other
     runs are identical to cur by construction).
  C  K-check offsets with a changing b: target concurrency alternates 8 / 4 every 15 s (mid, short prefixes):
       balt cb    verify cost per request + 8 ms at round 400 (K-independent, proportional to b)
       balt rtt   RTT 4 -> 24 ms at round 400 (K- and b-independent)
     policies cur, boff (offset per b bucket), boff+int (also a K-check interval of 100 rounds, doubling), cto (a K
     check unfinished after 40 rounds is abandoned), boff+int+cto; and on the "rtt alt" runs of round 29 (mid b = 4,
     high b = 8, fixed b): cur, boff+int.
  D  timing spikes (ext25.add_tail: 1% stalled rounds, 0.5% single spikes): tail base (mid b = 4), tail high
     (high b = 8), tail het (het b = 4), tail chg8 x10 (mid b = 8, layer cost x10 at round 400); cur and rob. Without
     spikes: rob on base mid b = 4 and chg8 x10 (cur from Z).
Short runs seeds 8, 9 (3x requests), long run seed 8 (10x), unless stated.
"""
import json, sys, time
import ext2 as X
import ext20 as V
import ext23 as G
import ext24 as H
import ext25 as N
import run25 as R
import run26 as Q
import run27 as T
import run29 as M

RUNS = ((8, 3), (9, 3), (8, 10))
CUR = ("cur", {})
POL_A = [CUR, ("pend", dict(pend=True)), ("psrc", dict(psrc=True)), ("nocd", dict(cooldown=False))]
POL_T = [CUR, ("bucket", dict(bucket=True))]
POL_C = [CUR, ("boff", dict(boff=True)), ("boff+int", dict(boff=True, chk_int=100)), ("cto", dict(chk_to=40)),
         ("boff+int+cto", dict(boff=True, chk_int=100, chk_to=40))]
POL_D = [CUR, ("rob", dict(rob=True))]
AW = {"v+l x3": dict(v=250, l=[(400, 3.0)]), "v+l x10": dict(v=250, l=[(400, 10.0)]),
      "l+l x10": dict(l=[(250, 0.3), (400, 10.0 / 0.3)]),
      "3chg x10": dict(l=[(250, 0.3), (600, 1.0 / 0.3), (700, 10.0)])}
LATE = {("mid", 4): 6000, ("high", 8): 3500}


def balt(mult):
    end = 120000 if mult == 3 else 400000
    return [(15000 * (i + 1), 8 if i % 2 == 0 else 4) for i in range(end // 15000)]


def one(part, name, kind, mix, b, phase, seed, mult, cm, kw):
    w = V.make_world(kind, mix, b, seed, phase=phase, req_mult=mult)
    conc = None
    if part in ("S", "D") and cm is not None:
        Q.add_change(w, cm)
    if part == "S" and name == "bvar":
        conc = R.R.CONC if mult == 3 else R.CONC_LONG
    if part == "A":
        if AW[name].get("v"):
            H.add_vchg(w, 10.0, at=AW[name]["v"])
        N.add_changes(w, AW[name]["l"])
    if part == "T":
        G.add_rtt(w, high=40.0, at=LATE[(kind, b)], period=100)
    if part == "C":
        if name == "rtt alt":
            G.add_rtt(w, **M.RTT[name])
        else:
            conc = balt(mult)
            if name == "balt cb":
                N.add_bchg(w, 8.0, at=400)
            else:
                G.add_rtt(w, high=24.0, at=400)
    if part == "D" and name.startswith("tail"):
        N.add_tail(w)
    w.first_W = 4
    p = N.BatchAccount18(**kw)
    rate = X.simulate3(w, p, conc=conc)
    out = {"rate": rate, "spec_rounds": w.n_T, "w": {str(k): v for k, v in sorted(p.w_log.items())},
           "k": {str(k): v for k, v in sorted(p.k_log.items())},
           "switches": p.n_switch, "switch_log": p.switch_log[:12], "sw_se": p.sw_se[:12], "probes": p.n_probe_rounds,
           "cost_events": p.n_cost_events, "ev_log": p.ev_log[:20], "suppressed": p.n_suppressed,
           "checks": p.n_checks, "offsets": p.n_offsets, "vevents": p.n_vevents, "levents": p.n_levents,
           "chk_log": p.chk_log[:20], "pend": p.n_pend, "chk_supp": p.n_chk_supp, "se_block": p.n_se_block,
           "skip": p.n_skip, "rescreen": p.n_rescreen, "probe_wait": p.n_probe_wait, "chk_abort": p.n_chk_abort,
           "final": {str(k): v["cur"] for k, v in p.ctx14.items()}}
    if p.err_l:
        out["lt_err"] = float(sum(p.err_l) / len(p.err_l))
    return out


def _z():
    out = {}
    for f in ("results/r31z.0.jsonl", "results/r31z.1.jsonl"):
        for l in open(f):
            r = json.loads(l)
            out[(r["world"], r["kind"], r["b"], r["seed"], r["req_mult"])] = r["res"]["cur"]
    return out


def jobs(part):
    S = [("S", (n, k, m, b, ph), s, mu, None) for (s, mu) in RUNS for (n, k, m, b, ph) in R.WORLDS]
    S += [("S", (n, k, m, b, None), s, mu, c) for (s, mu) in RUNS for (n, k, m, b, c) in Q.CHANGE + T.NEWW]
    if part == "Z":
        return [(j, [CUR]) for j in S]
    if part == "B":
        z = _z()
        weak = lambda j: any(x[4] <= 2.0 * x[5] for x in z[(j[1][0], j[1][1], j[1][3], j[2], j[3])]["sw_se"])
        return [(j, [("se2", dict(se_z=2.0))]) for j in S if weak(j)]
    if part == "A":
        return [(("A", (nm, "mid", "short", 8, None), s, 3, None), POL_A) for s in (8, 9, 10) for nm in AW]
    if part == "T":
        out = [("T", ("rtt alt late", "mid", "short", 4, None), s, 20, None) for s in (8, 9)]
        out += [("T", ("rtt alt late", "high", "short", 8, None), 8, 20, None)]
        return [(j, POL_T) for j in out]
    if part == "C":
        out = [(("C", (nm, "mid", "short", 8, None), s, mu, None), POL_C) for (s, mu) in RUNS for nm in ("balt cb", "balt rtt")]
        out += [(("C", ("rtt alt", kd, "short", b, None), s, mu, None), [CUR, POL_C[2]]) for (s, mu) in RUNS
                for (kd, b) in (("mid", 4), ("high", 8))]
        return out
    tw = [("tail base", "mid", "short", 4, None), ("tail high", "high", "short", 8, None),
          ("tail het", "het", "mixed", 4, None), ("tail chg8 x10", "mid", "short", 8, 10.0)]
    out = [(("D", (n, k, m, b, None), s, 3, c), POL_D) for s in (8, 9) for (n, k, m, b, c) in tw]
    out += [(("D", (n, k, m, b, None), 8, 10, c), POL_D) for (n, k, m, b, c) in tw[:2]]
    out += [(("D", (n, k, m, b, None), s, mu, c), [POL_D[1]]) for (s, mu) in RUNS
            for (n, k, m, b, c) in (("base", "mid", "short", 4, None), ("chg8 x10", "mid", "short", 8, 10.0))]
    return out


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
