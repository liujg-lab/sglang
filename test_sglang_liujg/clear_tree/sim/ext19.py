"""Round 22: batch-wide verification budget (each request its own verify count K_i) vs one K per batch (BatchAccount12).

World22 = ext18.World21 (prefix terms) with
  v_mult        the per-token verify cost v_tok times this
  ragged verify c0 + c_b b + v_tok * bucket(sum (K_i + 1)) + sum_i kt_i + first layer, where
                kt_i = k_t (0.5 l_i + (K_i + 1) l_i / 34) (each verify row of request i reads its prefix), and bucket()
                rounds the total verify tokens up to a captured size (TOKEN_BUCKETS); with every K_i = K and no
                rounding this is the uniform World21 verify cost
BatchAccount12 (known costs, W from the v10 table, no W calibration, no speculation switch):
  alloc "uniform"         v10: one K per batch from {8, 16, 32} (J_K enumeration), growth on the same value
        "unified_verify"  growth as "uniform"; before verifying, every request's nodes (sorted by path score, values
                          c_i * score) are pooled; a node's net value is c_i * score - Theta * k_t l_i / 34; nodes with a
                          positive net value are taken best first until the total tokens reach a bucket; the bucket with
                          the best J = sum of values - Theta * ragged verify time is chosen; K_i = nodes taken from i
                          (prefix-closed: within a request the net value falls with rank)
        "unified_both"    also growth: the value of stopping is J of the unified allocation; request i's next-layer gain
                          is g_i at the smallest captured K >= its allocated K_i (8 if none)
"""
import numpy as np
import ext2 as X
import ext8 as N
import ext10 as P
import ext18 as H
import feasibility as F

TOKEN_BUCKETS = (8, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512)


def tbucket(x):
    for t in TOKEN_BUCKETS:
        if x <= t:
            return t
    return x


class World22(H.World21):
    def __init__(self, *a, v_mult=1.0, **kw):
        super().__init__(*a, **kw)
        self.v_tok *= v_mult

    def T_ragged_core(self, b, Ks, ells):
        tot = sum(k + 1 for k in Ks)
        pre = sum(self.k_t * (0.5 * l + (k + 1) * l / 34.0) for k, l in zip(Ks, ells))
        return self.c0 + self.c_b * b + self.v_tok * tbucket(tot) + pre

    def real_T_ragged(self, rng, b, Ks, ells):
        self.n_T += 1
        c = self.T_ragged_core(b, Ks, ells) + self.c_d_l(b, self.first_W, float(sum(ells)))
        return c * max(0.5, 1 + rng.normal(0, self.noise)) if self.noise else c


def make_world(mix, kind, b, seed, strength, v_mult=1.0, req_mult=3):
    import copy
    import ext7 as S
    import run7 as R7
    import run13 as U
    a, s, k = WL[kind]
    S.set_conc(k)
    fit = copy.deepcopy(U.FIT)
    th = fit["fits"]["full"]["theta"]
    kd, kt = {"mid": (0.00025, 0.0005), "strong": (0.001, 0.002)}[strength]
    return World22(th, fit["a0"], fit["a_b"], mix=mix, k_d=kd, k_t=kt, v_mult=v_mult, seed=seed, b=b, rtt=4.0,
                   n_req=req_mult * max(16, 8 * b), **R7.NOISY, **S.acc_kw(a, *s))


# workloads: mid as before; "het" spreads the per-request acceptance level widely
WL = {"mid": (3.0, (0.5, 0.9), 1), "het": (3.0, (0.2, 0.99), 1)}


class BatchAccount12(H.BatchAccount11):
    def __init__(self, wk, alloc="uniform", **kw):
        kw.setdefault("cost_mode", "known")
        super().__init__(wk, **kw)
        self.alloc = alloc
        self.ktot_log = {}

    def _alloc(self, trees, cs, ell, theta, b):
        """Unified allocation: returns (K_i list, J)."""
        w = self.world
        items = []
        for i, t in enumerate(trees):
            sc = sorted(t.scores, reverse=True)[:32]
            pen = theta * w.k_t * ell[i] / 34.0
            for s_ in sc:
                v = cs[i] * s_
                if v - pen > 0:
                    items.append((v - pen, v, i))
        items.sort(key=lambda x: -x[0])
        best = None
        for B in TOKEN_BUCKETS:
            if B < b:
                continue
            cap = B - b
            Ks = [0] * len(trees)
            val = 0.0
            for net, v, i in items[:cap]:
                Ks[i] += 1
                val += v
            J = val - theta * w.T_ragged_core(b, Ks, ell)
            if best is None or J > best[1]:
                best = (Ks, J)
            if cap >= len(items):
                break
        return best

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        if self.alloc == "uniform":
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
        W, Ktab = self.wk(b, cls)
        theta = self._theta_b(b)
        warm = self.rounds < self.warm or theta is None
        Ks_opt = self.k_options
        Kmax = max(Ks_opt)
        world.first_W = W
        trees = [X.Tree2(world.reqs[r], j, world) for r, j in slots]
        rids = [r for r, _ in slots]
        ell = [world.reqs[r]["p0"] + j for r, j in slots]
        suml_all = float(sum(ell))
        n = len(trees)
        params = (world.a, world.g, world.beta, 0.0)
        for t in trees:
            t.expand(W, rng, params)
        rem = [world.reqs[r]["L"] - j for r, j in slots]
        cs = [self._c(r) for r in rids]
        e_now = [self._es(t) for t in trees]
        e_prev = [{K: 0.0 for K in Ks_opt} for _ in range(n)]
        v_now = [self._vals(cs[i], e_now[i]) for i in range(n)]
        v_prev = [{K: 0.0 for K in Ks_opt} for _ in range(n)]
        kd = self._kd_hat(W)
        active, depth, time_ = list(range(n)), 1, 0.0
        while active:
            cand = [i for i in active if depth < world.DMAX - 1 and depth + 1 <= max(1, rem[i] - 1)]
            if not cand:
                break
            if warm:
                keep = [i for i in cand if e_now[i][Kmax] - e_prev[i][Kmax] > 0.4]
            elif self.alloc == "unified_verify":
                Jk = {K: sum(v_now[i][K] for i in range(n)) - theta * self._vt(b, K, suml_all) for K in Ks_opt}
                stop = max(Jk.values())
                best, keep = 0.0, []
                for K in Ks_opt:
                    g = {i: v_now[i][K] - v_prev[i][K] for i in cand}
                    sc = {i: g[i] - theta * kd * ell[i] for i in cand}
                    order = sorted(cand, key=lambda i: sc[i], reverse=True)
                    for m in self._mopts(len(order)):
                        sel = order[:m]
                        v = Jk[K] + sum(g[i] for i in sel) - theta * self._lt(m, W, sum(ell[i] for i in sel)) - stop
                        if v > best + 1e-12:
                            best, keep = v, sel
            else:  # unified_both
                Ks_cur, Ju = self._alloc(trees, cs, ell, theta, b)
                g = {}
                for i in cand:
                    Kt = next((K for K in Ks_opt if K >= max(Ks_cur[i], 1)), Kmax)
                    g[i] = v_now[i][Kt] - v_prev[i][Kt]
                sc = {i: g[i] - theta * kd * ell[i] for i in cand}
                order = sorted(cand, key=lambda i: sc[i], reverse=True)
                best, keep = 0.0, []
                for m in self._mopts(len(order)):
                    sel = order[:m]
                    v = sum(g[i] for i in sel) - theta * self._lt(m, W, sum(ell[i] for i in sel))
                    if v > best + 1e-12:
                        best, keep = v, sel
            if not keep:
                break
            for i in keep:
                trees[i].expand(W, rng, params)
                e_prev[i], e_now[i] = e_now[i], self._es(trees[i])
                v_prev[i], v_now[i] = v_now[i], self._vals(cs[i], e_now[i])
            sl = float(sum(ell[i] for i in keep))
            time_ += world.real_cd_l(rng, len(keep), W, sl)
            active = keep
            depth += 1
        if warm:
            Ks = [Ktab] * n
        else:
            Ks, _ = self._alloc(trees, cs, ell, theta, b)
        tot = sum(k + 1 for k in Ks)
        self.ktot_log[tbucket(tot)] = self.ktot_log.get(tbucket(tot), 0) + 1
        for k in Ks:
            kb = 8 if k <= 8 else (16 if k <= 16 else 32)
            self.k_log[kb] = self.k_log.get(kb, 0) + 1
        acc = [P.last_node(t, t.topk(Ks[i], False))[0] if Ks[i] > 0 else 0 for i, t in enumerate(trees)]
        com = [min(a + 1, rem[i]) for i, a in enumerate(acc)]
        wait = world.real_rtt(rng, rtt) + world.real_T_ragged(rng, b, Ks, ell)
        t_round = time_ + wait
        tok = float(sum(com))
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
        for i, t in enumerate(trees):
            e = N._mass(t, Ks[i]) if Ks[i] > 0 else 0.0
            r = rids[i]
            self.req_a[r] = self.req_a.get(r, 0.0) + acc[i]
            self.req_e[r] = self.req_e.get(r, 0.0) + e
            self.c_a += acc[i]; self.c_e += e
        self.rounds += 1
        return com, t_round
