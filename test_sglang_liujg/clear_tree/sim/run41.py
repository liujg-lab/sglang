"""Round 41 runner. usage: python run41.py <proc> <nproc> <part> [arms...]
Same worlds as round 40 (23), fresh seeds 25 and 26, x3 requests."""
import json, os, sys, time

SIM = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM)
sys.path.insert(0, HERE)
os.chdir(SIM)

import ext2 as X
import ext20 as V
import ext23 as G
import ext24 as H
import ext26 as P
import run25 as R25
import run26 as Q
import run29 as M
import run40 as R40
import ext32 as E


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
    p = E.RootW(**kw)
    rate = X.simulate3(w, p, conc=conc)
    return {"rate": rate, "rounds": p.rounds, "w": {str(k): v for k, v in sorted(p.w_log.items())},
            "k": {str(k): v for k, v in sorted(p.k_log.items())}, "kind": p.kind_log,
            "root": dict(p.n_root), "final": {str(k): v for k, v in p.persist.items()},
            "beta": {f"{bk}:{lo}-{hi}": [round(s[0] / s[1], 3) if s[1] > 0 else None, s[2]]
                     for (bk, lo, hi), s in p.beta.items()},
            "switch": p.switch_log[:20]}


def jobs(part):
    ws = R40.S_WORLDS + R40.B1 + R40.EXTRA
    if part == "main":
        return [(pt, key, s, 3) for s in (25, 26) for (pt, key) in ws]
    if part == "smoke":
        return [("S", ("base", "mid", "short", 1, None), 25, 1), ("S", ("het", "het", "mixed", 4, None), 25, 1)]
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
