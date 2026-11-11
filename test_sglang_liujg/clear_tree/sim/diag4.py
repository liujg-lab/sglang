"""Round-4 diagnostics.
A: long horizon (4x requests) for the RTT-20 scenarios where speculation loses, and b=8 RTT 4 where it wins:
   is the remaining loss of the D41 switch a start-up / drain effect of the short runs?
B: long prefix b=3, 4: does starting a context in Target-only (static rule) make the first probes run with an
   unwarmed planner and judge speculation too low? Compare start='static' with start='spec'; log every probe.
usage: diag4.py <A|B>"""
import json, sys, time
import feasibility as F
import ext as E
import ext2 as X
import ext4 as Z
import run4 as R4
from calibrated import CAL


def main():
    part = sys.argv[1]
    if part == "A":
        for b, rtt in ((4, 20.0), (6, 20.0), (8, 20.0), (8, 4.0)):
            t0 = time.time()
            w = X.World2(seed=7, b=b, rtt=rtt, n_req=4 * max(16, 8 * b), **CAL)
            sim = lambda pol: X.simulate3(w, pol)
            res = {"Target-only": sim(F.AR()), "ClearTree core+old switch": sim(E.SpecSwitch(R4.core(w)))}
            if rtt == 4.0:
                res["ClearTree core"] = sim(R4.core(w))
            sw = Z.SpecSwitch2(R4.core(w), probe_rounds=8, share=0.05)
            res["ClearTree core+P8/5%"] = [sim(sw), sw.spec_time / sw.total_time, sw.probes, sw.total_time / 1000.0]
            print(json.dumps(dict(part="A", scen=f"b={b} RTT={rtt:g} 4x requests", res=res, log=sw.log[:40],
                                  sec=round(time.time() - t0, 1))), flush=True)
    else:
        for b in (3, 4):
            for seed in (7, 8):
                t0 = time.time()
                w = X.World2(seed=seed, b=b, rtt=4.0, **dict(CAL, **Z.long_kw(R4.LONG_FACTOR)))
                sim = lambda pol: X.simulate3(w, pol)
                res, logs = {"ClearTree core": sim(R4.core(w))}, {}
                for start in ("static", "spec"):
                    sw = Z.SpecSwitch2(R4.core(w), probe_rounds=8, share=0.05, b_static=3, start=start)
                    res[f"ClearTree core+P8/5% start={start}"] = [sim(sw), sw.spec_time / sw.total_time, sw.probes]
                    logs[start] = sw.log[:30]
                print(json.dumps(dict(part="B", scen=f"long prefix b={b}", seed=seed, res=res, log=logs,
                                      sec=round(time.time() - t0, 1))), flush=True)


if __name__ == "__main__":
    main()
