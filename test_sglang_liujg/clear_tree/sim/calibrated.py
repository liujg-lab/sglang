"""Calibrated run: acceptance ~3.5 tokens/round at b=1 with OPT(8); speculation ~ Target-only at b=7-8
(RTT 4 ms). Writes one JSON line per scenario with the full static grid for post-hoc analysis."""
import json, sys, time
import numpy as np
import feasibility as F

CAL = dict(a=1.5, s_range=(0.5, 0.9), t1=0.2)
CPS = (0.1, 0.3, 1.0)


def specs():
    out = []
    for b in (1, 4, 6, 8):
        for rtt in (4.0, 20.0):
            out.append((f"wellspec b={b} RTT={rtt:g}", dict(b=b, rtt=rtt, misspec=False)))
    for b in (1, 4):
        out.append((f"misspec b={b} RTT=4", dict(b=b, rtt=4.0, misspec=True)))
    out.append(("wellspec b=4 RTT 4<->20 every 150 rounds", dict(b=4, rtt=(4.0, 20.0), misspec=False, rtt_switch=150)))
    return out


def run(name, kw):
    w = F.World(seed=7, **kw, **CAL)
    o = {"name": name, "AR": F.simulate(w, F.AR())[0]}
    grid = {f"{N},{tau}": F.simulate(w, F.OPT(N, tau))[0] for N in F.KS for tau in F.TAUS}
    o["grid"] = grid
    ct, pt = {}, {}
    for W in F.KS:
        pol = F.ClearTree(W, "oracle", cP=CPS[0])
        ct[W] = F.simulate(w, pol)[0]
        pt[W] = pol.plan_times
    bw = max(ct, key=ct.get)
    o["ct_oracle"] = ct; o["bw"] = bw
    o["ct_learned"] = F.simulate(w, F.ClearTree(bw, "learned", cP=CPS[0]))[0]
    o["ct_cp"] = {cp: F.simulate(w, F.ClearTree(bw, "oracle", cP=cp))[0] for cp in CPS[1:]}
    arr = np.asarray(pt[bw]) * 1000
    o["plan_ms"] = [float(arr.mean()), float(np.percentile(arr, 99))]
    return o


def main():
    part = sys.argv[1]
    for k, (name, kw) in enumerate(specs()):
        if str(k % 2) != part:
            continue
        t0 = time.time()
        o = run(name, kw)
        o["sec"] = round(time.time() - t0)
        print(json.dumps(o), flush=True)


if __name__ == "__main__":
    main()
