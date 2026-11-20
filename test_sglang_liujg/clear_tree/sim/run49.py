"""Round 49 runner (pure paired W arms pair_g / pair_s). main: reuse r43/r44 references. long: 6 worlds x10 with references. usage: python run43.py <proc> <nproc> <part> [arms...]
Every world uses the measured Draft layer cost (ext34.measure_world). main: round-40 worlds (23), seeds 25 and 26, x3."""
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
import ext33 as E33
import ext34 as E34
import ext35 as E35
import ext36 as E36
import ext37 as E37
import ext38 as E38
import ext39 as E39
import ext40 as E40
import ext41 as E41


class _All:
    ARMS = {**E34.ARMS, **E35.ARMS, **E36.ARMS, **E37.ARMS, **E38.ARMS, **E39.ARMS, **E40.ARMS, **E41.ARMS}
    SatDG = E35.SatDG


E = _All


def flatten_w(w):
    w.d_tok *= 0.2
    k4 = w.kd(4)
    w.kd = lambda _W, k4=k4: k4


def one(part, key, seed, mult, arm):
    name, kind, mix, b, x = key
    phase = x if part in ("S", "F") and isinstance(x, str) else None
    cm = x if not isinstance(x, str) else None
    w = V.make_world(kind, mix, b, seed, phase=phase, req_mult=mult)
    E34.measure_world(w)
    conc = None
    if part in ("S", "F"):
        if cm is not None:
            Q.add_change(w, cm)
        if name == "bvar":
            conc = R25.R.CONC if mult == 3 else R25.CONC_LONG
        if part == "F":
            flatten_w(w)
    elif part == "R":
        G.add_rtt(w, **M.RTT[name])
    elif part == "L":
        P.add_lalt(w, mult=3.0)
    elif part == "K":
        P.add_k32(w, 400)
    elif part == "VC":
        H.add_vchg(w, cm)
    w.first_W = 4
    cls, kw = E.ARMS[arm]
    p = cls(**kw)
    rate = X.simulate3(w, p, conc=conc)
    out = {"rate": rate, "rounds": p.rounds, "w": {str(k): v for k, v in sorted(p.w_log.items())},
           "k": {str(k): v for k, v in sorted(p.k_log.items())}, "kind": p.kind_log,
           "final": {str(k): v for k, v in p.persist.items()}, "switch": p.switch_log[:20]}
    if isinstance(p, E36._ProbeBase):
        out["n_probe"] = p.n_probe_rounds
    if isinstance(p, E41.PairW):
        out["n_sup"], out["n_timing"], out["events"] = p.n_sup, p.n_timing, dict(p.events)
        out["pair"] = p.diag()
        out["switch"] = p.switch_log[:40]
    if isinstance(p, E40.TCFix):
        out["tc"] = p.out()
    if isinstance(p, E38.IndFix):
        out["ind"] = p.out()
    if isinstance(p, E37.SatP):
        out["n_probe"] = p.n_probe_rounds
        out["n_verify"] = p.n_verify_rounds
        out["events"] = dict(p.p_events)
        out["n_up_timing"] = getattr(p, "n_up_timing", 0)
        out["switch"] = p.switch_log[:40]
    if isinstance(p, E.SatDG):
        out["satd"] = p._sat_out()
    if isinstance(p, E34.SatW):
        out["sat"] = {k: [v[0], round(v[1] / max(v[0], 1), 3), round(v[2] / max(v[0], 1), 3)] for k, v in p.sat_log.items()}
    if isinstance(p, E33.SigW):
        out["diag"] = {k: [round(v[0], 2), round(v[1], 2), v[2]] for k, v in p.diag.items()}
        out["r_up"] = {str(k): round(p._r_up(k), 3) for k in p.cal}
        out["ema"] = {str(k): {kk: round(vv, 4) for kk, vv in s.items()} for k, s in p.st.items()}
        out["n_cal"], out["n_timing"] = p.n_cal, p.n_timing
    return out


FLAT = [("F", ("base", "mid", "short", 1, None)), ("F", ("base", "mid", "short", 4, None)),
        ("F", ("base", "mid", "short", 8, None)), ("F", ("het", "het", "mixed", 4, None)),
        ("F", ("het", "het", "mixed", 8, None)), ("F", ("high", "high", "short", 8, None))]


def jobs(part):
    ws = R40.S_WORLDS + R40.B1 + R40.EXTRA
    if part == "main":
        return [(pt, key, s, 3) for s in (25, 26) for (pt, key) in ws]
    if part == "flat":
        return [(pt, key, s, 3) for s in (25, 26) for (pt, key) in FLAT]
    if part == "long":
        ws = [("S", ("base", "mid", "short", 1, None)), ("S", ("het", "het", "mixed", 1, None)),
              ("S", ("base", "mid", "short", 4, None)), ("S", ("het", "het", "mixed", 4, None)),
              ("S", ("base", "mid", "short", 8, None)), ("S", ("chg8 x3", "mid", "short", 8, 3.0))]
        return [(pt, key, 25, 10) for (pt, key) in ws]
    if part == "s25":
        return [(pt, key, 25, 3) for (pt, key) in ws]
    if part == "smoke":
        return [("S", ("base", "mid", "short", 1, None), 25, 1), ("S", ("base", "mid", "short", 4, None), 25, 1), ("S", ("het", "het", "mixed", 8, None), 25, 1)]
    raise SystemExit(part)


def main():
    proc, nproc, part = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    arms = sys.argv[4:] or list(E.ARMS)
    skip = {int(x) for x in os.environ.get("SKIP_IDX", "").split(",") if x}
    todo = [i for i in range(len(jobs(part))) if i not in skip]
    for i, (pt, key, seed, mult) in enumerate(jobs(part)):
        if i not in skip and todo.index(i) % nproc != proc or i in skip:
            continue
        t0 = time.time()
        res = {a: one(pt, key, seed, mult, a) for a in arms}
        print(json.dumps(dict(part=pt, world=key[0], kind=key[1], mix=key[2], b=key[3], x=key[4], seed=seed,
                              mult=mult, res=res, sec=round(time.time() - t0, 1))), flush=True)
        print(f"done {i} {key} s{seed} x{mult} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
