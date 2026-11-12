"""Round 25: fast-converging online W (BatchAccount14, wmode "fast") vs the replay controller (ext20 replay14).

Same world (ext19.World22, prefix-aware, known costs), v10 growth, flat starting table (W = 8 at b = 1, else 4) and
per-b-bucket state as replay14. Per round Y = committed tokens - Theta * model time.

fast, per b bucket (ext9.bkey):
  union probe round   the Draft grows, for every candidate width, the tree that width would grow by its own rule
                      (own exits, own depth, own K); the round pays the union (per layer: requests active in any
                      candidate, width = widest active candidate); the committed tokens are the current width's own
                      tree (verify unchanged). In the simulator the candidate trees are grown separately on shared
                      per-request random streams, so each candidate's Y is exact and the differences are paired.
  screening           50 rounds after a bucket first appears (and again every 2000 bucket rounds): 16 union probe
                      rounds over all widths; a width is eliminated if mean + 2.5 se of Y(W) - Y(best) is below
                      -3% of tokens per round (a one-off fixed-sample test; the smoke test with the anytime radius
                      at n = 16, about 10%, eliminated nothing); if the current width is eliminated, move to the best mean.
                      The screening rounds are exact paired data, so they also seed the neighbour comparisons
                      (probe pairs, and replay + bias samples for the narrower neighbour).
  narrower neighbour  every normal round, free replay from the current tree (ext20 replay: narrowed tree, may stop
                      up to 2 layers earlier, K re-chosen) -> D_rep = Yrep(Wn) - Yrep(cur); bias of that replay
                      from probe rounds: B = [Yrep(Wn) - Yrep(cur)] - [Y(Wn) - Y(cur)] (replay from the current
                      width's own tree vs exact). Estimate D = mean(D_rep) - mean(B), se from both parts.
  wider neighbour     only from probe rounds (exact paired differences).
  probes              5% of the normal rounds, over {narrower, current, wider} minus eliminated widths.
  decision            every 10 rounds: switch to a neighbour if D - r > eps, r = the round-19b normal-mixture radius
                      (rho = 50 sd^2, alpha 5% over 2 neighbours) evaluated with the estimate's se and n = number of
                      pairs (probe pairs for the wider one, replay rounds for the narrower one); eps = 0.25% of the committed tokens per round (EMA). Data are kept in sliding
                      windows (replay 600 rounds, probes 300); everything relative to the current width is reset
                      after a switch. Windows never close.
"""
import collections
import math
import numpy as np
import ext10 as P
import ext2 as X
import ext9 as Q
import ext20 as V
import diag24 as D          # noqa: F401  (registers the "high" workload in ext19.WL)

W_OPTS = V.W_OPTS


def cs_radius(se, n, a=0.025, rho_n=50):
    if n < 2:
        return math.inf
    return se * math.sqrt(((n + rho_n) / n) * (math.log((n + rho_n) / rho_n) + 2 * math.log(2 / a)))


class BatchAccount14(V.BatchAccount13):
    def __init__(self, wmode="fast", p_probe=0.05, screen_n=16, screen_delay=50, rescreen=2000, elim=0.03,
                 eps_rel=0.0025, every=10, win_rep=600, win_probe=300, min_probe=10, screen_z=2.5, **kw):
        super().__init__(wmode=wmode, **kw)
        self.p_probe, self.screen_n, self.screen_delay, self.rescreen, self.elim = p_probe, screen_n, screen_delay, rescreen, elim
        self.eps_rel, self.every, self.win_rep, self.win_probe, self.min_probe = eps_rel, every, win_rep, win_probe, min_probe
        self.screen_z = screen_z
        self.fs = {}
        self.n_probe, self.n_screen, self.n_elim = 0, 0, 0
        self.elim_log = []
        self.extra_ms = 0.0            # union probe time above the current width's own model time

    # ---------- one growth (copy of diag24.Diag22._dgrow, also returns acc and E^K) ----------
    def _grow(self, world, slots, W, rngs, theta, cx):
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
        while active:
            cand = [i for i in active if depth < world.DMAX - 1 and depth + 1 <= max(1, rem[i] - 1)]
            if not cand:
                break
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
            for i in keep:
                trees[i].expand(W, rngs[i], params)
                e_prev[i], e_now[i] = e_now[i], self._es(trees[i])
                v_prev[i], v_now[i] = v_now[i], self._vals(cs[i], e_now[i])
            layers.append((len(keep), float(sum(ell[i] for i in keep))))
            keeps.append(list(keep))
            active = keep
            depth += 1
        Jk = {K: sum(v_now[i][K] for i in range(n)) - theta * world.T_core(b, K, suml_all) for K in Ks}
        K = max(Jk, key=Jk.get)
        acc = [P.last_node(t, t.topk(K, False))[0] for t in trees]
        tok = [min(a + 1, rem[i]) for i, a in enumerate(acc)]
        mt = self._model_time(b, W, K, layers, suml_all)
        return dict(trees=trees, layers=layers, keeps=keeps, K=K, acc=acc, tok=tok, e=e_now, time=mt,
                    Y=float(sum(tok)) - theta * mt)

    # ---------- state per b bucket ----------
    def _st(self, bk, Wflat):
        s = self.fs.get(bk)
        if s is None:
            s = self.fs[bk] = dict(cur=Wflat, elim=set(), n=0, screen_left=0, next_screen=self.screen_delay,
                                   screen=[], rep=collections.deque(maxlen=self.win_rep),
                                   bias=collections.deque(maxlen=self.win_probe),
                                   pr={W: collections.deque(maxlen=self.win_probe) for W in W_OPTS})
        return s

    def _reset(self, s):
        s["rep"].clear()
        s["bias"].clear()
        for d in s["pr"].values():
            d.clear()

    def _neighbours(self, s):
        allowed = [W for W in W_OPTS if W not in s["elim"] or W == s["cur"]]
        i = allowed.index(s["cur"])
        lo = allowed[i - 1] if i > 0 else None
        hi = allowed[i + 1] if i + 1 < len(allowed) else None
        return lo, hi

    def _fchoose(self, s):
        s["n"] += 1
        if s["screen_left"] == 0 and s["n"] >= s["next_screen"]:
            s["screen_left"], s["screen"] = self.screen_n, []
            s["next_screen"] = s["n"] + self.rescreen
            s["elim"] = set()
        if s["screen_left"] > 0:
            s["screen_left"] -= 1
            self.n_screen += 1
            return list(W_OPTS), "screen"
        lo, hi = self._neighbours(s)
        if (lo or hi) and self.rng2.random() < self.p_probe:
            self.n_probe += 1
            return sorted({s["cur"]} | {w for w in (lo, hi) if w}), "probe"
        return [s["cur"]], "normal"

    def _after_screen(self, s, b):
        Ys = [x["Y"] for x in s["screen"]]  # list of {W: Y}
        n = len(Ys)
        means = {W: float(np.mean([y[W] for y in Ys])) for W in W_OPTS}
        best = max(means, key=means.get)
        tok = max(1e-9, self.tok_ema or 1.0)
        for W in W_OPTS:
            if W == best:
                continue
            d = np.array([y[W] - y[best] for y in Ys]) / tok
            mu, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(n))
            if mu + self.screen_z * se < -self.elim:
                s["elim"].add(W)
                self.n_elim += 1
        if len(self.elim_log) < 40:
            self.elim_log.append((self.rounds, b, sorted(s["elim"])))
        if s["cur"] in s["elim"]:
            new = max((W for W in W_OPTS if W not in s["elim"]), key=means.get)
            self._switch(s, b, new)
        # the screening rounds are exact paired data: reuse them for the neighbours of the (new) current width
        cur = s["cur"]
        lo, hi = self._neighbours(s)
        for x in s["screen"]:
            for W in (lo, hi):
                if W is not None:
                    s["pr"][W].append(x["Y"][W] - x["Y"][cur])
            if lo is not None and x["cur"] == cur and x["lo"] == lo and x["drep"] is not None:
                s["rep"].append(x["drep"])
                s["bias"].append(x["drep"] - (x["Y"][lo] - x["Y"][cur]))

    def _switch(self, s, b, new):
        if new == s["cur"]:
            return
        if len(self.switch_log) < 40:
            self.switch_log.append((self.rounds, b, s["cur"], new))
        s["cur"] = new
        self.n_switch += 1
        self._reset(s)

    def _decide(self, s, b):
        tok = max(1e-9, self.tok_ema or 1.0)
        eps = self.eps_rel
        lo, hi = self._neighbours(s)
        best, bm = None, None
        if hi is not None and len(s["pr"][hi]) >= self.min_probe:
            d = np.asarray(s["pr"][hi]) / tok
            n = len(d)
            mu, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(n))
            if mu - cs_radius(se, n) > eps:
                best, bm = hi, mu
        if lo is not None and len(s["bias"]) >= self.min_probe and len(s["rep"]) >= 30:
            r = np.asarray(s["rep"]) / tok
            bb = np.asarray(s["bias"]) / tok
            mu = float(r.mean() - bb.mean())
            se = math.sqrt(float(r.var(ddof=1)) / len(r) + float(bb.var(ddof=1)) / len(bb))
            if mu - cs_radius(se, len(r)) > eps and (bm is None or mu > bm):
                best, bm = lo, mu
        if best is not None:
            self._switch(s, b, best)

    # ---------- one round ----------
    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        if self.wmode != "fast":
            return super().round(world, slots, rng, rtt, rnd, trees_out)
        self.world = world
        b = len(slots)
        h1 = sum(int(world.reqs[r]["H"][j] == 0) for r, j in slots)
        if self.t1_n is None:
            self.t1_hit, self.t1_n = float(h1), float(b)
        else:
            self.t1_hit = 0.95 * self.t1_hit + 0.05 * h1
            self.t1_n = 0.95 * self.t1_n + 0.05 * b
        cls = self.cls()
        Wflat, Ktab = V.flat_wk(b, cls)
        theta = self._theta_b(b)
        warm = self.rounds < self.warm or theta is None
        rids = [r for r, _ in slots]
        cx = dict(b=b, cs=[self._c(r) for r in rids], rem=[world.reqs[r]["L"] - j for r, j in slots],
                  ell=[world.reqs[r]["p0"] + j for r, j in slots])
        cx["suml"] = suml_all = float(sum(cx["ell"]))
        if warm:
            # same as the flat policy during warm-up (ext20 round with wmode flat)
            self.wmode = "flat"
            try:
                return super().round(world, slots, rng, rtt, rnd, trees_out)
            finally:
                self.wmode = "fast"
        bk = Q.bkey(b)
        s = self._st(bk, Wflat)
        cands, tag = self._fchoose(s)
        cur = s["cur"]
        base = int(rng.integers(2 ** 62))
        rs = lambda: [np.random.default_rng((base, i)) for i in range(b)]
        G = {W: self._grow(world, slots, W, rs(), theta, cx) for W in cands}
        g = G[cur]
        # real Draft time: union of the candidates' layers (one candidate on normal rounds)
        Lmax = max(len(G[W]["layers"]) for W in cands)
        time_, union_model = 0.0, world.c_d_l(b, max(cands), suml_all)
        for L in range(Lmax):
            act = [W for W in cands if len(G[W]["keeps"]) > L]
            idx = set().union(*(G[W]["keeps"][L] for W in act))
            sl = float(sum(cx["ell"][i] for i in idx))
            time_ += world.real_cd_l(rng, len(idx), max(act), sl)
            union_model += world.c_d_l(len(idx), max(act), sl)
        world.first_W = max(cands)
        K = g["K"]
        com, acc = g["tok"], g["acc"]
        t_round = time_ + world.real_rtt(rng, rtt) + world.real_T_l(rng, b, K, suml_all)
        if len(cands) > 1:
            self.extra_ms += union_model - (sum(world.c_d_l(m, cur, sl) for m, sl in g["layers"])
                                            + world.c_d_l(b, cur, suml_all))
        self.w_log[cur] = self.w_log.get(cur, 0) + 1
        self.k_log[K] = self.k_log.get(K, 0) + 1
        # learning
        lo, hi = self._neighbours(s)
        if tag == "screen":
            lower = [W for W in W_OPTS if W < cur]
            drep = None
            if lower:
                rep = self._replay(g["trees"], cx["cs"], cx["rem"], cx["ell"], g["layers"], cur, b, theta, 0)
                drep = (float(sum(rep[lower[-1]]["tok"])) - theta * rep[lower[-1]]["time"]) - \
                       (float(sum(rep[cur]["tok"])) - theta * rep[cur]["time"])
            s["screen"].append(dict(Y={W: G[W]["Y"] for W in cands}, cur=cur, lo=lower[-1] if lower else None, drep=drep))
            if s["screen_left"] == 0:
                self._after_screen(s, b)
        else:
            if lo is not None:
                rep = self._replay(g["trees"], cx["cs"], cx["rem"], cx["ell"], g["layers"], cur, b, theta, 0)
                yr = lambda W: float(sum(rep[W]["tok"])) - theta * rep[W]["time"]
                d_rep = yr(lo) - yr(cur)
                s["rep"].append(d_rep)
                if tag == "probe" and lo in G:
                    s["bias"].append(d_rep - (G[lo]["Y"] - g["Y"]))
            if tag == "probe":
                for W in cands:
                    if W != cur:
                        s["pr"][W].append(G[W]["Y"] - g["Y"])
            if s["n"] % self.every == 0:
                self._decide(s, b)
        # bookkeeping (as ext20)
        for r, j in slots:
            hit = float(world.reqs[r]["H"][j] == 0)
            self.h_req[r] = hit if r not in self.h_req else 0.9 * self.h_req[r] + 0.1 * hit
        tok = float(sum(com))
        self.tok_ema = tok if self.tok_ema is None else 0.95 * self.tok_ema + 0.05 * tok
        self.tok_total += tok
        fx = t_round - time_
        self.fixed[b] = fx if b not in self.fixed else 0.9 * self.fixed[b] + 0.1 * fx
        a_ = self.alpha
        if self.g_ms is None:
            self.g_tok, self.g_ms = tok, t_round
        else:
            self.g_tok = (1 - a_) * self.g_tok + a_ * tok
            self.g_ms = (1 - a_) * self.g_ms + a_ * t_round
        v = self.tb.setdefault(b, [tok, t_round, 0])
        v[2] += 1
        ab_ = max(a_, 1.0 / v[2])
        v[0] = (1 - ab_) * v[0] + ab_ * tok
        v[1] = (1 - ab_) * v[1] + ab_ * t_round
        for i in range(b):
            e = g["e"][i][K]
            r = rids[i]
            self.req_a[r] = self.req_a.get(r, 0.0) + acc[i]
            self.req_e[r] = self.req_e.get(r, 0.0) + e
            self.c_a += acc[i]; self.c_e += e
        self.rounds += 1
        return com, t_round
