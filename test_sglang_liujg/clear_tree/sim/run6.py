"""Simulation round 6 (review 13). Parts:
A  cost model under context change (b=4, 2 seeds): RTT 4<->20 every 30 s; RTT drifting 4..12 ms (60 s period);
   short/long prefix phases every 30 s. Cost models: EMA table (v1), level+effect, level+effect keyed by prefix
   (prefix phases only), anchored level+difference, known costs. Metric: throughput and the K choice against
   the true cost.
B  K_g cache scope: mixed load (concurrency 1->4->8->4) with RTT 4<->20 every 30 s. Re-select K_g every 10
   rounds only / also on a b change / also on a level jump / every round.
C  delta decay per committed position: 0.8 / 0.9 / 0.95 / static; misspecified b=1, 4 and well-specified b=4.
D  switch timing basis with a 2 ms Target per-iteration overhead: Target-only rate from the full cycle, from the
   kernel time as a prior, or from the kernel time throughout.
E  stop-boundary audit: every stop/grow decision of ClearTree core (records go to a separate file).
usage: run6.py <proc 0|1> <E-records output prefix>   (records go to <prefix>.<proc>)"""
import json, math, sys, time
import feasibility as F
import ext2 as X
import ext3 as Y
import ext4 as Z
import ext5 as V
import run3b as R
import run4 as R4
from calibrated import CAL

NOISY = dict(noise=0.05, jitter=1.0)
SW = dict(probe_rounds=8, share=0.05, exit_blocks=2, exit_fast=0.15)   # D42 switch


def cost(w, kind):
    if kind == "EMA table (v1)":
        return R.cm_(w, 1.15)
    if kind == "known":
        cm = Y.CostModel3(w, "known", p_force=0.03, fd=1.15); cm.fd_b1 = 1.0
        return cm
    keyed = kind.endswith("+prefix key")
    if kind.startswith("EMA table"):
        cm = V.CostModelEMAK(w, "learned", keyed=keyed, p_force=0.03, fd=1.15)
    elif kind.startswith("anchored"):
        cm = V.CostModelAnchor(w, "learned", keyed=keyed, p_force=0.03, fd=1.15)
    else:
        cm = V.CostModelLE(w, "learned", keyed=keyed, recenter=False, p_force=0.03, fd=1.15)
    cm.fd_b1 = 1.0
    return cm


def core(w, kind="EMA table (v1)", **kw):
    base = dict(slow=None, n_snap=4, kreuse=10, learn_beta=True, dyn_delta=True)
    base.update(kw)
    return V.CT5(4, "learned", cost=cost(w, kind), **base)


def job_A(scen, seed):
    w = X.World2(seed=seed, b=4, rtt=4.0, n_req=180, **NOISY, **CAL)
    if scen == "RTT 4<->20 every 30s":
        sk = dict(rtt_fn=lambda t: 4.0 if int(t // 30000) % 2 == 0 else 20.0)
    elif scen == "RTT drift 4..12 ms, 60 s period":
        sk = dict(rtt_fn=lambda t: 8.0 - 4.0 * math.cos(2 * math.pi * t / 60000.0))
    else:
        sk = dict(regime_fn=lambda t: int(t // 30000) % 2)
    kinds = ["EMA table (v1)", "level+effect", "anchored", "known"]
    if "prefix" in scen:
        kinds = ["EMA table (v1)", "EMA table+prefix key", "anchored", "anchored+prefix key", "known"]
    res = {"Target-only": V.simulate4(w, F.AR(), **sk)}
    for k in kinds:
        p = core(w, k, audit_k=True)
        res[k] = [V.simulate4(w, p, **sk), p.krec]
    return res


def job_B(seed):
    mixed = dict(b=4, rtt=4.0, n_req=400, L_range=(32, 512), **NOISY)
    w = X.World2(seed=seed, **mixed, **CAL)
    sk = dict(conc=[(25000, 1), (50000, 4), (75000, 8), (100000, 4)],
              rtt_fn=lambda t: 4.0 if int(t // 30000) % 2 == 0 else 20.0)
    variants = {"every 10 rounds (v1)": dict(kind="EMA table (v1)"),
                "every 10 rounds, +on b change": dict(kind="EMA table", kg_key=True),
                "every 10 rounds, +on b change +on level jump": dict(kind="EMA table", kg_key=True, kg_jump=True),
                "every round": dict(kind="EMA table (v1)", kreuse=1)}
    res = {"Target-only": V.simulate4(w, F.AR(), **sk)}
    for name, kv in variants.items():
        kind = kv.pop("kind")
        p = core(w, kind, audit_k=True, **kv)
        res[name] = [V.simulate4(w, p, **sk), p.krec]
    return res


def job_C(scen, seed):
    wkw = dict(b=1 if "b=1" in scen else 4, rtt=4.0, misspec="misspec" in scen)
    w = X.World2(seed=seed, **wkw, **CAL)
    res = {}
    for d in (0.8, 0.9, 0.95):
        res[f"decay {d}"] = X.simulate3(w, core(w, decay=d))
    res["static delta"] = X.simulate3(w, core(w, dyn_delta=False))
    return res


def job_D(scen, seed):
    long_ = "long" in scen
    b = int(scen.split("b=")[1].split()[0]); rtt = float(scen.split("RTT=")[1])
    over = dict(CAL, **Z.long_kw(R4.LONG_FACTOR)) if long_ else CAL
    w = V.WorldOH(seed=seed, b=b, rtt=rtt, oh=2.0, **over)
    sim = lambda pol: X.simulate3(w, pol)
    res = {"Target-only": sim(F.AR()), "ClearTree core (no switch)": sim(core(w))}
    for src in ("full", "kernel_prior", "kernel_always"):
        sw = V.SpecSwitch3(core(w), ar_source=src, b_static=3 if long_ else 7, **SW)
        res[f"switch, Target-only rate from {src}"] = [sim(sw), sw.spec_time / sw.total_time, sw.probes]
    return res


def job_E(scen, seed, out):
    long_ = "long" in scen
    b = int(scen.split("b=")[1].split()[0])
    over = dict(CAL, **Z.long_kw(R4.LONG_FACTOR)) if long_ else CAL
    w = X.World2(seed=seed, b=b, rtt=4.0, misspec="misspec" in scen, **over)
    p = core(w, audit_stop=True)
    rate = X.simulate3(w, p)
    with open(out, "a") as f:
        f.write(json.dumps(dict(scen=scen, seed=seed, rec=[[round(x, 4) if isinstance(x, float) else x for x in r]
                                                            for r in p.srec])) + "\n")
    return {"ClearTree core": rate, "records": len(p.srec)}


def jobs():
    out = []
    for seed in (7, 8):
        out.append(("B", seed, None))
    for s in ("RTT 4<->20 every 30s", "RTT drift 4..12 ms, 60 s period", "prefix short<->long every 30s, RTT 4"):
        for seed in (7, 8):
            out.append(("A", seed, s))
    for s in ("b=8 RTT=4", "b=6 RTT=4", "long prefix b=3 RTT=4", "b=4 RTT=20"):
        for seed in (7, 8):
            out.append(("D", seed, s))
    for s in ("misspec b=1", "misspec b=4", "wellspec b=4"):
        for seed in (7, 8):
            out.append(("C", seed, s))
    for s in ("wellspec b=1", "wellspec b=4", "misspec b=1", "misspec b=4", "long prefix b=2"):
        for seed in (7, 8):
            out.append(("E", seed, s))
    return out


def main():
    proc, eout = int(sys.argv[1]), sys.argv[2]
    for k, (part, seed, s) in enumerate(jobs()):
        if k % 2 != proc:
            continue
        t0 = time.time()
        if part == "A":
            res = job_A(s, seed)
        elif part == "B":
            s = "mixed load, RTT 4<->20 every 30s"; res = job_B(seed)
        elif part == "C":
            res = job_C(s, seed)
        elif part == "D":
            res = job_D(s, seed)
        else:
            res = job_E(s, seed, f"{eout}.{proc}")
        print(json.dumps(dict(part=part, scen=s, seed=seed, res=res, sec=round(time.time() - t0, 1))), flush=True)


if __name__ == "__main__":
    main()
