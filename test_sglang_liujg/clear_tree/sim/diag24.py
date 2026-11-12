"""Round 24 diagnosis: why does paired replay underestimate a narrower W?
usage (conda sglang env): python diag24.py <proc 0|1>

The main policy runs one fixed built width W. On every non-warm round, before the main round, the same batch state is
also grown in the simulator on separate random streams (the main run is not affected), with the same price Theta,
the same c_i and the same growth rule. Per round Y = committed tokens - Theta * model time (what the calibrations
compare). For the built width W and each narrower Wn:

  G       W, own growth rule (what the policy builds)
  R0[Wn]  the replay of Wn from G, as the policy does it (R0[W]: the replay's own value of the built width)
  R0d     (ext20 only) the replay from G grown 2 extra W-wide layers (an ext20 "deep" round)
  R0c     (ext15 only) the ext20 replay rule (cut up to 2 layers earlier, K re-chosen) applied to the ext15 world
  R1[Wn]  a real Wn tree grown with G's per-layer request schedule (same exits, same depth), valued like the replay
                                               R1 - R0[Wn] = narrowing approximation (ext12.narrowed)
  R2[Wn]  a real Wn tree, own growth rule (own exits / stop), at most as many layers as G
                                               R2 - R1 = own exits / stopping
  R3[Wn]  a real Wn tree, own growth rule, no cap
                                               R3 - R2 = depth truncation
  replay estimate R0[Wn] - R0[W]     true difference R3[Wn] - G
  built 8: W=2 vs W=4 both replayed: R0[2] - R0[4] vs the truth R3[2] - R3[4]
Every variant draws off-path draft distributions from one random stream per (round, request), so variants of the same
request share draws while their trees grow alike.

Settings
  ext20  ext19.World22 (prefix-aware, strong prefix terms), BatchAccount13 (round 23 growth and replay: narrowed tree,
         may cut up to 2 layers earlier, K re-chosen); high / mid: short prefixes, het: mixed prefixes; b in {4, 8}
  ext15  run18 world "base" (round 20; no prefix terms), BatchAccount8 (round 18-20 replay: narrowed tree at the
         built depth and exits, K re-chosen, time = model time of the built layer counts); high / mid; b in {4, 8}
Built widths: 4 (replays 2) and 8 (replays 4 and 2). Seed 8; reference long-run rates: fixed 2 / 4 / 8.
"""
import json, sys, time
import numpy as np
import ext2 as X
import ext10 as P
import ext12 as RP
import ext15 as G15
import ext19 as Z
import ext20 as V
import run10 as T
import run13 as U
import run18 as R18
import run20 as R20

Z.WL.setdefault("high", U.WL["high"])        # run13 "high": a = 4.0, s in (0.85, 0.99)
FIELDS = ("tok", "time", "Y", "L", "K")


class _DiagMixin:
    def _diag_init(self, Wb, Wns, dseed=4242):
        self.Wb, self.Wns, self.dseed = Wb, tuple(Wns), dseed
        self.rows, self.k_mismatch = [], 0

    def _diag(self, world, slots, theta, rnd):
        b = len(slots)
        rids = [r for r, _ in slots]
        cx = dict(b=b, cs=[self._c(r) for r in rids], rem=[world.reqs[r]["L"] - j for r, j in slots],
                  ell=[world.reqs[r].get("p0", 0) + j for r, j in slots])
        cx["suml"] = float(sum(cx["ell"]))
        rs = lambda: [np.random.default_rng((self.dseed, rnd, i)) for i in range(b)]
        W, row = self.Wb, {}

        def put(key, o):
            tok = float(sum(o["tok"]))
            row[key] = (tok, float(o["time"]), tok - theta * float(o["time"]), int(o["layers"]), int(o["K"]))

        G = self._dgrow(world, slots, W, rs(), theta, cx)
        put("G", G)
        rep = self._drep(G, W, theta, cx)
        for Wn, o in rep.items():
            put(f"R0[{Wn}]", o)
        if rep[W]["K"] != G["K"]:
            self.k_mismatch += 1
        if hasattr(self, "_drep_deep"):
            for Wn, o in self._drep_deep(world, slots, W, rs(), theta, cx).items():
                put(f"R0d[{Wn}]", o)
        if hasattr(self, "_drep_cut"):
            for Wn, o in self._drep_cut(G, W, theta, cx).items():
                put(f"R0c[{Wn}]", o)
        row["act"] = float(sum(G["m"]))
        for Wn in self.Wns:
            T1 = self._dgrow(world, slots, Wn, rs(), theta, cx, sched=G["keeps"])
            put(f"R1[{Wn}]", self._dinh(T1, Wn, theta, cx))
            put(f"R2[{Wn}]", self._dgrow(world, slots, Wn, rs(), theta, cx, cap=len(G["keeps"])))
            T3 = self._dgrow(world, slots, Wn, rs(), theta, cx)
            put(f"R3[{Wn}]", T3)
            row[f"act[{Wn}]"] = float(sum(T3["m"]))
        self.rows.append(row)


class Diag22(_DiagMixin, V.BatchAccount13):
    """ext20 setting (round 23)."""

    def __init__(self, Wb, Wns, **kw):
        super().__init__(wmode="fixed", fixed_W=Wb, **kw)
        self._diag_init(Wb, Wns)

    def _dgrow(self, world, slots, W, rngs, theta, cx, cap=None, sched=None, deep=0):
        b, Ks, cs, ell, rem, suml_all = cx["b"], self.k_options, cx["cs"], cx["ell"], cx["rem"], cx["suml"]
        trees = [X.Tree2(world.reqs[r], j, world) for r, j in slots]
        params = (world.a, world.g, world.beta, 0.0)
        for i, t in enumerate(trees):
            t.expand(W, rngs[i], params)
        n = len(trees)
        e_now = [self._es(t) for t in trees]
        e_prev = [{K: 0.0 for K in Ks} for _ in range(n)]
        v_now = [self._vals(cs[i], e_now[i]) for i in range(n)]
        v_prev = [{K: 0.0 for K in Ks} for _ in range(n)]
        kd = world.kd(W)
        active, depth, layers, keeps = list(range(n)), 1, [], []

        def grow(keep):
            for i in keep:
                trees[i].expand(W, rngs[i], params)
                e_prev[i], e_now[i] = e_now[i], self._es(trees[i])
                v_prev[i], v_now[i] = v_now[i], self._vals(cs[i], e_now[i])
            layers.append((len(keep), float(sum(ell[i] for i in keep))))
            keeps.append(list(keep))

        while active:
            if cap is not None and len(layers) >= cap:
                break
            cand = [i for i in active if depth < world.DMAX - 1 and depth + 1 <= max(1, rem[i] - 1)]
            if not cand:
                break
            if sched is not None:
                if len(layers) >= len(sched):
                    break
                keep = [i for i in sched[len(layers)] if i in cand]
            else:
                Jk = {K: sum(v_now[i][K] for i in range(n)) - theta * world.T_core(b, K, suml_all) for K in Ks}
                stop = max(Jk.values())
                best, keep = 0.0, []
                for K in Ks:
                    g = {i: v_now[i][K] - v_prev[i][K] for i in cand}
                    sc = {i: g[i] - theta * kd * ell[i] for i in cand}
                    order = sorted(cand, key=lambda i: sc[i], reverse=True)
                    for m in self._mopts(len(order)):
                        sel = order[:m]
                        v = Jk[K] + sum(g[i] for i in sel) - theta * world.c_d_l(m, W, sum(ell[i] for i in sel)) - stop
                        if v > best + 1e-12:
                            best, keep = v, sel
            if not keep:
                break
            grow(keep)
            active = keep
            depth += 1
        deep_extra = 0
        if deep:
            cand = [i for i in range(n) if depth < world.DMAX - 1 and depth + 1 <= max(1, rem[i] - 1)]
            for _ in range(deep):
                cand = [i for i in cand if depth < world.DMAX - 1 and depth + 1 <= max(1, rem[i] - 1)]
                if not cand:
                    break
                grow(cand)
                depth += 1
                deep_extra += 1
        Jk = {K: sum(v_now[i][K] for i in range(n)) - theta * world.T_core(b, K, suml_all) for K in Ks}
        K = max(Jk, key=Jk.get)
        acc = [P.last_node(t, t.topk(K, False))[0] for t in trees]
        return dict(trees=trees, layers_l=layers, keeps=keeps, m=[m for m, _ in layers], K=K, layers=len(layers),
                    tok=[min(a + 1, rem[i]) for i, a in enumerate(acc)], deep_extra=deep_extra,
                    time=self._model_time(b, W, K, layers, suml_all))

    def _drep(self, G, W, theta, cx):
        return self._replay(G["trees"], cx["cs"], cx["rem"], cx["ell"], G["layers_l"], W, cx["b"], theta, 0)

    def _drep_deep(self, world, slots, W, rngs, theta, cx):
        Gd = self._dgrow(world, slots, W, rngs, theta, cx, deep=2)
        return self._replay(Gd["trees"], cx["cs"], cx["rem"], cx["ell"], Gd["layers_l"], W, cx["b"], theta,
                            Gd["deep_extra"])

    def _dinh(self, T1, Wn, theta, cx):
        """The replay's cut / K choice on all nodes of a real Wn tree (copy of BatchAccount13._replay, one width)."""
        w, b, cs, rem, suml_all = self.world, cx["b"], cx["cs"], cx["rem"], cx["suml"]
        trees, layers = T1["trees"], T1["layers_l"]
        Dbuilt = 1 + len(layers)
        best = None
        for d in range(max(1, Dbuilt - 2), Dbuilt + 1):
            sel = [[n for n in t.ids if t.dep[n] <= d] for t in trees]
            lt = sum(w.c_d_l(m, Wn, sl) for m, sl in layers[:d - 1]) + w.c_d_l(b, Wn, suml_all)
            for K in self.k_options:
                v = sum(cs[i] * sum(sorted((t.score[n] for n in sel[i]), reverse=True)[:K]) for i, t in enumerate(trees))
                J = v - theta * (lt + w.T_core(b, K, suml_all))
                if best is None or J > best[0]:
                    best = (J, d, K, sel, lt)
        _, d, K, sel, lt = best
        tok = [min(RP.accepted_in(t, RP.topk_of(t, sel[i], K)) + 1, rem[i]) for i, t in enumerate(trees)]
        return dict(tok=tok, time=lt + w.T_core(b, K, suml_all) + 5.0, K=K, layers=d - 1)

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        self.world = world
        theta = self._theta_b(len(slots))
        if not (self.rounds < self.warm or theta is None):
            self._diag(world, slots, theta, rnd)
        return super().round(world, slots, rng, rtt, rnd, trees_out)


class Diag18(_DiagMixin, G15.BatchAccount8):
    """ext15 setting (rounds 18-20; the replay the round-20 calibration used)."""

    def __init__(self, Wb, Wns, wk, tv, lt, vt, **kw):
        super().__init__(wk, tv, lt, vt, wmode="fixed", fixed_W=Wb, **kw)
        self._diag_init(Wb, Wns)

    def _dgrow(self, world, slots, W, rngs, theta, cx, cap=None, sched=None):
        b, Ks, cs, rem = cx["b"], self.k_options, cx["cs"], cx["rem"]
        trees = [X.Tree2(world.reqs[r], j, world) for r, j in slots]
        params = (world.a, world.g, world.beta, 0.0)
        for i, t in enumerate(trees):
            t.expand(W, rngs[i], params)
        n = len(trees)
        e_now = [self._es(t) for t in trees]
        e_prev = [{K: 0.0 for K in Ks} for _ in range(n)]
        v_now = [self._vals(cs[i], e_now[i]) for i in range(n)]
        v_prev = [{K: 0.0 for K in Ks} for _ in range(n)]
        active, depth, keeps = list(range(n)), 1, []
        while active:
            if cap is not None and len(keeps) >= cap:
                break
            cand = [i for i in active if depth < world.DMAX - 1 and depth + 1 <= max(1, rem[i] - 1)]
            if not cand:
                break
            if sched is not None:
                if len(keeps) >= len(sched):
                    break
                keep = [i for i in sched[len(keeps)] if i in cand]
            else:
                Jk = {K: sum(v_now[i][K] for i in range(n)) - theta * self.Tv(b, K) for K in Ks}
                stop = max(Jk.values())
                g1 = {K: {i: v_now[i][K] - v_prev[i][K] for i in cand} for K in Ks}
                _, keep, _, _ = self._best(cand, set(), g1, g1, Jk, stop, theta, W, (1,))
            if not keep:
                break
            for i in keep:
                trees[i].expand(W, rngs[i], params)
                e_prev[i], e_now[i] = e_now[i], self._es(trees[i])
                v_prev[i], v_now[i] = v_now[i], self._vals(cs[i], e_now[i])
            keeps.append(list(keep))
            active = keep
            depth += 1
        Jk = {K: sum(v_now[i][K] for i in range(n)) - theta * self.Tv(b, K) for K in Ks}
        K = max(Jk, key=Jk.get)
        acc = [P.last_node(t, t.topk(K, False))[0] for t in trees]
        m = [len(k) for k in keeps]
        return dict(trees=trees, keeps=keeps, m=m, K=K, layers=len(m),
                    tok=[min(a + 1, rem[i]) for i, a in enumerate(acc)], time=self._model(b, W, K, m))

    def _drep(self, G, W, theta, cx):
        b = cx["b"]
        val, tok = self._replay_tables(G["trees"], cx["cs"], cx["rem"], W)
        out = {}
        for Wn in val:
            k = self._pickK(val[Wn], theta, b)
            out[Wn] = dict(tok=[tok[Wn][k]], time=self._model(b, Wn, k, G["m"]), K=k, layers=len(G["m"]))
        return out

    def _drep_cut(self, G, W, theta, cx):
        """ext20 replay rule on this world: narrowed tree, depth cut up to 2 layers earlier, K re-chosen."""
        b, cs, rem, trees, m = cx["b"], cx["cs"], cx["rem"], G["trees"], G["m"]
        Dbuilt = 1 + len(m)
        out = {}
        for Wn in self.w_opts:
            if Wn > W:
                continue
            nodes = [RP.narrowed(t, Wn) if Wn < W else list(t.ids) for t in trees]
            depths = [Dbuilt] if Wn == W else list(range(max(1, Dbuilt - 2), Dbuilt + 1))
            best = None
            for d in depths:
                sel = [[n for n in nodes[i] if t.dep[n] <= d] for i, t in enumerate(trees)]
                for K in self.k_options:
                    v = sum(cs[i] * sum(sorted((t.score[n] for n in sel[i]), reverse=True)[:K]) for i, t in enumerate(trees))
                    tm = self._model(b, Wn, K, m[:d - 1])
                    J = v - theta * tm
                    if best is None or J > best[0]:
                        best = (J, d, K, sel, tm)
            _, d, K, sel, tm = best
            tok = [min(RP.accepted_in(t, RP.topk_of(t, sel[i], K)) + 1, rem[i]) for i, t in enumerate(trees)]
            out[Wn] = dict(tok=tok, time=tm, K=K, layers=d - 1)
        return out

    def _dinh(self, T1, Wn, theta, cx):
        """The ext15 replay valuation (built depth and exits, K re-chosen) on a real Wn tree."""
        b = cx["b"]
        val, tok = self._replay_tables(T1["trees"], cx["cs"], cx["rem"], Wn)
        k = self._pickK(val[Wn], theta, b)
        return dict(tok=[tok[Wn][k]], time=self._model(b, Wn, k, T1["m"]), K=k, layers=len(T1["m"]))

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        theta = self._theta_b(len(slots))
        if not (self.rounds < self.warm or theta is None):
            self._diag(world, slots, theta, rnd)
        return super().round(world, slots, rng, rtt, rnd, trees_out)


def summarize(rows, Wb, Wns):
    def ms(x):
        x = np.asarray(x, float)
        return float(x.mean()), (float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0)
    keys = [k for k in rows[0] if not k.startswith("act")]
    out = {"n": len(rows), "mean": {k: {f: ms([r[k][j] for r in rows])[0] for j, f in enumerate(FIELDS)} for k in keys}}
    ref = out["mean"]["G"]["tok"]
    Y = lambda k: (lambda r: r[k][2])
    D = lambda a, c: (lambda r: r[a][2] - r[c][2])
    diffs = {}
    for Wn in Wns:
        p = f"{Wb}->{Wn}"
        diffs[f"{p} replay est"] = D(f"R0[{Wn}]", f"R0[{Wb}]")
        if f"R0d[{Wn}]" in rows[0]:
            diffs[f"{p} deep-round est"] = D(f"R0d[{Wn}]", f"R0d[{Wb}]")
        if f"R0c[{Wn}]" in rows[0]:
            diffs[f"{p} cut-rule est"] = D(f"R0c[{Wn}]", f"R0c[{Wb}]")
        diffs[f"{p} TRUE"] = D(f"R3[{Wn}]", "G")
        diffs[f"{p}   narrowing R1-R0"] = D(f"R1[{Wn}]", f"R0[{Wn}]")
        diffs[f"{p}   exits/stop R2-R1"] = D(f"R2[{Wn}]", f"R1[{Wn}]")
        diffs[f"{p}   depth R3-R2"] = D(f"R3[{Wn}]", f"R2[{Wn}]")
    diffs[f"built side G-R0[{Wb}]"] = D("G", f"R0[{Wb}]")
    if 2 in Wns and 4 in Wns:
        diffs["2 vs 4 (both from 8) replay est"] = D("R0[2]", "R0[4]")
        if "R0d[2]" in rows[0]:
            diffs["2 vs 4 (both from 8) deep-round est"] = D("R0d[2]", "R0d[4]")
        if "R0c[2]" in rows[0]:
            diffs["2 vs 4 (both from 8) cut-rule est"] = D("R0c[2]", "R0c[4]")
        diffs["2 vs 4 TRUE (own trees)"] = D("R3[2]", "R3[4]")
    out["diff_pct_tok"] = {}
    for name, f in diffs.items():
        m, se = ms([f(r) for r in rows])
        out["diff_pct_tok"][name] = (100 * m / ref, 100 * se / ref)
    out["deeper_share"] = {Wn: float(np.mean([r[f"R3[{Wn}]"][3] > r["G"][3] for r in rows])) for Wn in Wns}
    out["act_layers"] = dict(G=float(np.mean([r["act"] for r in rows])),
                             **{str(Wn): float(np.mean([r[f"act[{Wn}]"] for r in rows])) for Wn in Wns})
    return out


BUILT = ((4, (2,)), (8, (4, 2)))
JOBS = [("ext20", k, mix, b) for k, mix in (("high", "short"), ("mid", "short"), ("het", "mixed")) for b in (8, 4)] + \
       [("ext15", k, "base", b) for k in ("high", "mid") for b in (8, 4)]


def run_ext20(kind, mix, b):
    res = {}
    for W in (2, 4, 8):
        w = V.make_world(kind, mix, b, 8)
        w.first_W = W
        res[f"fixed {W}"] = dict(rate=X.simulate3(w, V.BatchAccount13(wmode="fixed", fixed_W=W)))
    for Wb, Wns in BUILT:
        w = V.make_world(kind, mix, b, 8)
        w.first_W = Wb
        p = Diag22(Wb, Wns)
        rate = X.simulate3(w, p)
        res[f"built {Wb}"] = dict(summarize(p.rows, Wb, Wns), rate=rate, k_mismatch=p.k_mismatch)
    return res


def run_ext15(kind, b):
    def mk():
        w = R18.make_world("base", kind, b, 8)
        lt = lambda m, W, w=w: w.c_d(m, W)
        vt = lambda bb, K, w=w: w.c0 + w.c_b * bb + w.v_tok * bb * (K + 1)
        tv = lambda bb, K, w=w: w.T_t(bb, K)
        return w, (R20.wk_flat, tv, lt, vt)
    res = {}
    for W in (2, 4, 8):
        w, a = mk()
        res[f"fixed {W}"] = dict(rate=T.sim(w, P.SwitchTok(G15.BatchAccount8(*a, wmode="fixed", fixed_W=W), T.R7.SW),
                                            8 if b == 1 else 4))
    for Wb, Wns in BUILT:
        w, a = mk()
        p = Diag18(Wb, Wns, *a)
        rate = T.sim(w, P.SwitchTok(p, T.R7.SW), 8 if b == 1 else 4)
        res[f"built {Wb}"] = dict(summarize(p.rows, Wb, Wns), rate=rate, k_mismatch=p.k_mismatch)
    return res


def main():
    proc = int(sys.argv[1])
    for n, (setting, kind, mix, b) in enumerate(JOBS):
        if n % 2 != proc:
            continue
        t0 = time.time()
        res = run_ext20(kind, mix, b) if setting == "ext20" else run_ext15(kind, b)
        print(json.dumps(dict(setting=setting, kind=kind, mix=mix, b=b, res=res, sec=round(time.time() - t0, 1),
                              python=sys.version.split()[0])), flush=True)


if __name__ == "__main__":
    main()
