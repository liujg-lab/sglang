"""Round 18: online update of the layer width W (ext15.BatchAccount8). usage: run18.py <A|B|O|L|L2> <proc 0|1> [one]

Worlds (C1 costs; the v10 W table was built for "base"; every policy knows the true current costs):
  base     the C1 costs (the table should be about right)
  tok x3   the Draft per-token layer cost d_tok x3 from the start (rows expensive: the table is too wide)
  tok x0.3 d_tok x0.3 from the start (rows cheap: the table is too narrow at b >= 4)
  change   base until spec round 150, then d_tok x3 (a run-time change; the cost tables follow at once)
  (probes: d_tok x5 and d0 x3 were dropped; with d_tok x5 Target-only wins at b = 8, with d0 x3 the table stays right)
Workloads mid / high; b in {1, 4, 8}; seeds 8-10; 3x the usual number of requests.
Part A variants: table (v10), fixed W = 2 / 4 / 8 (K online; the best per scenario is the hindsight reference),
                 replay14, rescore (confirm), rescore (no confirm).
Part B (added after part A): rescore with the deployment start treated as an event (start_event=True), without
        the confirmation trial or with a 10 x 10-round one, exploring at W_max or one step wider; same scenarios.
Part O: "oracle switch" for the change world: best fixed W of base before round 150, of tok x3 after (per kind, b),
        read from the part A results.
"""
import copy, json, sys, time
import ext7 as S
import ext10 as P
import ext15 as G
import run7 as R7
import run10 as T
import run13 as U

FIT = U.FIT
WORLDS = {"base": ((1, 1, 1), None), "tok x3": ((1, 1, 3), None), "tok x0.3": ((1, 1, 0.3), None), "change": ((1, 1, 3), 150)}


class World18(R7.World7):
    def __init__(self, theta, a0, a_b, mult=(1, 1, 1), change_round=None, **kw):
        super().__init__(theta, a0, a_b, **kw)
        self.base_d = (self.d0, self.d_req, self.d_tok)
        self.mult, self.change_round, self.n_T = mult, change_round, 0
        if change_round is None:
            self._apply(mult)

    def _apply(self, m):
        self.d0, self.d_req, self.d_tok = (x * k for x, k in zip(self.base_d, m))

    def real_T(self, rng, b, K):
        self.n_T += 1
        out = super().real_T(rng, b, K)
        if self.change_round is not None and self.n_T == self.change_round:
            self._apply(self.mult)
        return out


REQ_MULT = 3


def make_world(world, kind, b, seed):
    mult, cr = WORLDS[world]
    a, s, k = U.WL[kind]
    S.set_conc(k)
    fit = copy.deepcopy(FIT)
    th = fit["fits"]["full"]["theta"]
    return World18(th, fit["a0"], fit["a_b"], mult=mult, change_round=cr, seed=seed, b=b, rtt=4.0,
                   n_req=REQ_MULT * max(16, 8 * b), **R7.NOISY, **S.acc_kw(a, *s))


def policy(w, **kw):
    lt = lambda m, W, w=w: w.c_d(m, W)
    vt = lambda bb, K, w=w: w.c0 + w.c_b * bb + w.v_tok * bb * (K + 1)
    return G.BatchAccount8(U.wk_v10, lambda bb, K, w=w: w.T_t(bb, K), lt, vt, **kw)


VARIANTS = {
    "table": dict(wmode="table"),
    "fixed 2": dict(wmode="fixed", fixed_W=2),
    "fixed 4": dict(wmode="fixed", fixed_W=4),
    "fixed 8": dict(wmode="fixed", fixed_W=8),
    "replay14": dict(wmode="replay14"),
    "rescore": dict(wmode="rescore", confirm=True),
    "rescore no-confirm": dict(wmode="rescore", confirm=False),
}


def run(world, kind, b, seed, **kw):
    w = make_world(world, kind, b, seed)
    p = policy(w, **kw)
    rate = T.sim(w, P.SwitchTok(p, T.R7.SW), 8 if b == 1 else 4)
    return {"rate": rate, "w": {str(k): v for k, v in sorted(p.w_log.items())},
            "k": {str(k): v for k, v in sorted(p.k_log.items())}, "switches": p.n_switch, "explore": p.n_explore,
            "trials": p.n_trial, "trial_ok": p.n_trial_ok, "events": p.n_events, "spec_rounds": w.n_T,
            "switch_log": p.switch_log[:20]}


VARIANTS_B = {
    "rescore start no-confirm": dict(wmode="rescore", confirm=False, start_event=True),
    "rescore start long-trial": dict(wmode="rescore", confirm=True, start_event=True, trial_blocks=10),
    "rescore step no-confirm": dict(wmode="rescore", confirm=False, start_event=True, explore_to="step"),
    "rescore step long-trial": dict(wmode="rescore", confirm=True, start_event=True, trial_blocks=10, explore_to="step"),
}


def job_A(world, kind, b, seed, variants=None):
    return {name: run(world, kind, b, seed, **kw) for name, kw in (variants or VARIANTS).items()}


def best_fixed(rows, world, kind, b):
    sc = {W: sum(r["res"][f"fixed {W}"]["rate"] for r in rows if (r["world"], r["kind"], r["b"]) == (world, kind, b))
          for W in (2, 4, 8)}
    return max(sc, key=sc.get)


def job_O(kind, b, seed, w_before, w_after):
    sched = lambda bb, w, wb=w_before, wa=w_after: wb if w.n_T < 150 else wa
    return {"oracle switch": run("change", kind, b, seed, wmode="oracle", sched=sched), "w_before": w_before, "w_after": w_after}


# Part L: 10x longer runs (30x the usual requests) to check that one-off (event) exploration amortizes while the
# replay controller's continuous exploration does not. Seed 8; b in {4, 8}; worlds base and tok x0.3.
L_VARIANTS = {"table": VARIANTS["table"], "replay14": VARIANTS["replay14"],
              "rescore no-confirm": VARIANTS["rescore no-confirm"],
              "rescore step no-confirm": VARIANTS_B["rescore step no-confirm"]}
L2_VARIANT = ("rescore step no-confirm keep-wide",
              dict(VARIANTS_B["rescore step no-confirm"], keep_wide=200))
L_BEST_FIXED = {("tok x0.3", "mid", 4): 8, ("tok x0.3", "mid", 8): 8, ("tok x0.3", "high", 4): 4, ("tok x0.3", "high", 8): 4}


# Part L2 (added after part L): the same long scenarios with exploration records kept in a separate buffer.
def jobs_L():
    js = []
    for wd in ("base", "tok x0.3"):
        for k in ("mid", "high"):
            for b in (8, 4):
                for v in L_VARIANTS:
                    js.append((wd, k, b, v))
                if (wd, k, b) in L_BEST_FIXED:
                    js.append((wd, k, b, f"fixed {L_BEST_FIXED[(wd, k, b)]}"))
    cost = {"replay14": 1.5, "rescore no-confirm": 1.3, "rescore step no-confirm": 1.3}
    return sorted(js, key=lambda j: -j[2] * cost.get(j[3], 1.0))


def jobs():
    js = [(wd, k, b, s) for wd in WORLDS for k in ("mid", "high") for b in (1, 4, 8) for s in (8, 9, 10)]
    return sorted(js, key=lambda j: -j[2])


def main():
    part, proc = sys.argv[1], int(sys.argv[2])
    one = len(sys.argv) > 3
    if part in ("L", "L2"):
        global REQ_MULT
        REQ_MULT = 30
        js = jobs_L() if part == "L" else [(wd, k, b, L2_VARIANT[0]) for wd in ("base", "tok x0.3") for k in ("mid", "high")
                                            for b in (8, 4)]
        for n, (wd, k, b, v) in enumerate(js):
            if n % 2 != proc:
                continue
            t0 = time.time()
            kw = (L_VARIANTS[v] if v in L_VARIANTS else L2_VARIANT[1] if v == L2_VARIANT[0]
                  else dict(wmode="fixed", fixed_W=int(v.split()[1])))
            res = run(wd, k, b, 8, **kw)
            print(json.dumps(dict(world=wd, kind=k, b=b, seed=8, variant=v, res=res, sec=round(time.time() - t0, 1))), flush=True)
            if one:
                break
        return
    if part in ("A", "B"):
        for n, j in enumerate(jobs()):
            if n % 2 != proc:
                continue
            t0 = time.time()
            res = job_A(*j, variants=VARIANTS_B if part == "B" else None)
            print(json.dumps(dict(world=j[0], kind=j[1], b=j[2], seed=j[3], res=res, sec=round(time.time() - t0, 1))), flush=True)
            if one:
                break
    else:
        rows = [json.loads(l) for f in ("results/r18A.0.jsonl", "results/r18A.1.jsonl") for l in open(f)]
        js = [(k, b, s) for k in ("mid", "high") for b in (1, 4, 8) for s in (8, 9, 10)]
        for n, (k, b, s) in enumerate(sorted(js, key=lambda j: -j[1])):
            if n % 2 != proc:
                continue
            t0 = time.time()
            res = job_O(k, b, s, best_fixed(rows, "base", k, b), best_fixed(rows, "tok x3", k, b))
            print(json.dumps(dict(world="change", kind=k, b=b, seed=s, res=res, sec=round(time.time() - t0, 1))), flush=True)


if __name__ == "__main__":
    main()
