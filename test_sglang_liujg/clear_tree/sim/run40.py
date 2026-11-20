"""Round 40 runner. usage: python run40.py <proc> <nproc> <part> [arms...]
Run from anywhere; imports the simulator in test_sglang_liujg/clear_tree/sim."""
import json, os, sys, time

SIM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(SIM)

import ext2 as X
import ext19 as Z
import ext20 as V
import ext23 as G
import ext24 as H
import ext26 as P
import run13 as U
import run25 as R25
import run26 as Q
import run27 as T
import run29 as M
import ext31 as E

Z.WL.setdefault("high", U.WL["high"])
Z.WL.setdefault("low", U.WL["low"])

S_WORLDS = [("S", w) for w in R25.WORLDS] + [("S", (n, k, m, b, c)) for (n, k, m, b, c) in Q.CHANGE + T.NEWW]
B1 = [("S", ("base", "mid", "short", 1, None)), ("S", ("het", "het", "mixed", 1, None))]
EXTRA = [("R", ("rtt jump", "mid", "short", 4, None)), ("R", ("rtt alt", "high", "short", 8, None)),
         ("K", ("k32", "mid", "short", 1, None)), ("K", ("k32", "mid", "short", 4, None)),
         ("VC", ("vchg x5", "mid", "short", 4, 5.0)), ("L", ("lalt x3", "mid", "short", 8, None))]


def one(part, key, seed, mult, kw):
    name, kind, mix, b, x = key
    phase = x if part == "S" and isinstance(x, str) else None
    cm = x if not isinstance(x, str) else None
    w = V.make_world(kind, mix, b, seed, phase=phase, req_mult=mult)
    conc = None
    if part == "S":
        if cm is not None:
            Q.add_change(w, cm)
        if name == "bvar":
            conc = R25.R.CONC if mult == 3 else R25.CONC_LONG
    elif part == "R":
        G.add_rtt(w, **M.RTT[name])
    elif part == "L":
        P.add_lalt(w, mult=3.0)
    elif part == "K":
        P.add_k32(w, 400)
    elif part == "VC":
        H.add_vchg(w, cm)
    w.first_W = 4
    p = E.CodeMirror(**kw)
    rate = X.simulate3(w, p, conc=conc)
    return {"rate": rate, "rounds": p.rounds, "w": {str(k): v for k, v in sorted(p.w_log.items())},
            "k": {str(k): v for k, v in sorted(p.k_log.items())}, "kind": p.kind_log, "src": p.src_log,
            "tau": p.n_tau, "value": p.n_value, "over": p.n_over, "excl": p.n_excl, "hold": p.n_hold_rounds,
            "switch": p.switch_log[:30], "events": p.events[:30], "final": {str(k): v for k, v in p.persist.items()},
            "level": {str(k): round(v, 2) for k, v in p.level.items()}, "lscale": round(p.lscale, 3)}


def jobs(part):
    if part == "main":
        ws = S_WORLDS + B1 + EXTRA
        return [(pt, key, s, 3) for s in (21, 22) for (pt, key) in ws]
    if part == "main2":
        ws = S_WORLDS + B1 + EXTRA
        return [(pt, key, s, 3) for s in (23, 24) for (pt, key) in ws]
    if part == "long":
        ws = [("S", ("het", "het", "mixed", 4, None)), ("S", ("phased", "het", "mixed", 8, "lowhighmid")),
              ("S", ("chg8 x10", "mid", "short", 8, 10.0)), ("R", ("rtt alt", "high", "short", 8, None)),
              ("S", ("base", "mid", "short", 1, None)), ("K", ("k32", "mid", "short", 4, None))]
        return [(pt, key, 21, 10) for (pt, key) in ws]
    if part == "smoke":
        return [("S", ("het", "het", "mixed", 4, None), 21, 1)]
    if part == "smoke2":
        ws = [("S", ("base", "mid", "short", 1, None)), ("S", ("chg8 x10", "mid", "short", 8, 10.0)),
              ("S", ("phased", "het", "mixed", 8, "lowhighmid")), ("R", ("rtt alt", "high", "short", 8, None))]
        return [(pt, key, 21, 2) for (pt, key) in ws]
    if part == "smoke3":
        ws = [("S", ("chg8 x10", "mid", "short", 8, 10.0)), ("L", ("lalt x3", "mid", "short", 8, None)),
              ("S", ("chg8 x3", "mid", "short", 8, 3.0)), ("S", ("chg x3", "mid", "short", 4, 3.0))]
        return [(pt, key, 21, 2) for (pt, key) in ws]
    if part == "abl":
        ws = [("S", ("phased", "het", "mixed", 4, "lowhighmid")), ("S", ("high", "high", "short", 8, None)),
              ("R", ("rtt alt", "high", "short", 8, None)), ("S", ("chg x0.3", "high", "short", 8, 0.3)),
              ("S", ("base", "mid", "short", 1, None))]
        return [(pt, key, s, 3) for s in (21, 22) for (pt, key) in ws]
    raise SystemExit(part)


def main():
    proc, nproc, part = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    arms = sys.argv[4:] or list(E.ARMS)
    for i, (pt, key, seed, mult) in enumerate(jobs(part)):
        if i % nproc != proc:
            continue
        t0 = time.time()
        res = {a: one(pt, key, seed, mult, E.ARMS[a]) for a in arms}
        print(json.dumps(dict(part=pt, world=key[0], kind=key[1], mix=key[2], b=key[3], x=key[4], seed=seed,
                              mult=mult, res=res, sec=round(time.time() - t0, 1))), flush=True)
        print(f"done {i} {key} s{seed} x{mult} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
