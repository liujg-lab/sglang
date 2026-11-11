"""Round 17 (review 16): cross-K stopping, cross-K marginal calibration, drift on unused K.

BatchAccount7 = ext13.BatchAccount6 with k_mode "consistent" (the v10 rule: W from the table, Theta per b, top-1
classifier, frozen costs, growth and verify on one value, K* = argmax_K J_K), plus:
  look    None     the v10 rule: one-layer options, g_i^K = last layer's increase of request i's value at K
          "opt2"   also two-layer options: J_K + sum over the m kept of (g_i^K + g_i^K if i can grow two more layers)
                   - Theta * (C(m) + C(m2)), m2 = how many of the m can grow two more layers. When the K of an option is
                   already the argmax of J_K, "two layers pay" implies "one layer pays", so it only changes decisions
                   across K
          "true1"  (oracle) the next layer's own increase: every candidate tree is cloned and grown one layer with the
                   main RNG; a kept request adopts its clone, so the forecast of the Draft mass is exact (the
                   calibration is still c_i)
          "true2"  (oracle) true1 plus the layer after it (a clone of the clone), with one- and two-layer options
  diag    at every rule-driven stop (candidates left but no option beats stopping now): the two-layer value with the
          g forecast, and, from clones grown two layers with a separate RNG, the true one- and two-layer values.
          A "cross-K stop" is a stop where a two-layer option is positive and no one-layer option is.
  fband   None or "online": value of request i at K = c_i * sum over bands (k', k] with k <= K of f_k * (E^k - E^k');
          f_k = (sum of band acceptance + 5) / (sum of c_i * band mass + 5), from same-tree replay of every round at
          every K (so it does not only learn from the chosen K)
  replay  per round, on the same tree, for every K: committed tokens and the counterfactual time
          (round time - true Tv(b, K*) + true Tv(b, K)); band predictions c_i * dE and band acceptance da. Bias sums
          by (band, request phase, tree depth).

BatchAccount7L adds drift detection v2 (ext13.BatchAccount6L) with
  interleave   the burst runs burst_len2 blocks, each a random permutation of the K options (instead of 8 x K=8,
               then 8 x K=16, ...)
  probe_p      each round, with this probability, verify with a random other K; its residual (wait - table - learned
               offset) is accumulated per (b, K); once n_probe_min residuals are in, a mean beyond
               max(abs_tol2, rel_tol2 * table) and beyond 3 standard errors triggers a burst
  neartie      probe the closest other K when its believed J is within neartie * (believed tokens) of the best
"""
import math
import numpy as np
import ext2 as X
import ext5 as V
import ext8 as N
import ext10 as P
import ext13 as D
import feasibility as F


def _dbin(d):
    return "d<=3" if d <= 3 else ("d4-6" if d <= 6 else "d>=7")


class BatchAccount7(D.BatchAccount6):
    def __init__(self, wk, table, look=None, diag=False, fband=None, replay=False, f_prior=5.0, **kw):
        kw.setdefault("k_mode", "consistent")
        super().__init__(wk, table, **kw)
        self.look, self.diag, self.fband, self.replay, self.f_prior = look, diag, fband, replay, f_prior
        self.f = {K: 1.0 for K in self.k_options}
        self.fS = {K: [0.0, 0.0] for K in self.k_options}          # band -> [sum pred, sum act]
        self.bias = {}                                            # (band, phase, dbin) -> [pred, act, n]
        self.rec = []
        self.req_n = {}
        self.dg = dict(stops=0, cross_opt=0, cross_opt_val=0.0, cross_opt_otherK=0, lag1_true=0, lag1_val=0.0,
                       cross_true=0, cross_true_val=0.0, cross_true_otherK=0, two_only_used=0, layers=0)
        self.tok_total = 0.0
        self.drng = np.random.default_rng(4242)
        self._jk, self._vk = None, 0.0

    # value of one request at every K from its masses es = {K: E^K}
    def _vals(self, c, es):
        out, cum, prev = {}, 0.0, 0.0
        for K in self.k_options:
            cum += self.f[K] * (es[K] - prev)
            prev = es[K]
            out[K] = c * cum
        return out

    def _es(self, t):
        return {K: N._mass(t, K) for K in self.k_options}

    def _mopts(self, n):
        return sorted({n} | {x for x in self.buckets if x < n}) if n else []

    @staticmethod
    def _grow(t, W, rng, params):
        c = V.clone_tree(t)
        c.expand(W, rng, params)
        return c

    def _best(self, cand, can2, g1, g2, Jk, stop, theta, W, hs):
        """Best option value (minus stopping now) over K, h in hs, m; returns (value, keep, K, h)."""
        best, keep, bK, bh = 0.0, [], None, None
        for K in self.k_options:
            for h in hs:
                if h == 1:
                    score = g1[K]
                else:
                    score = {i: g1[K][i] + (g2[K][i] if i in can2 else 0.0) for i in cand}
                order = sorted(cand, key=lambda i: score[i], reverse=True)
                for m in self._mopts(len(order)):
                    sel = order[:m]
                    v = Jk[K] + sum(score[i] for i in sel) - theta * self._cost(m, W) - stop
                    if h == 2:
                        m2 = sum(1 for i in sel if i in can2)
                        if m2 == 0:
                            continue
                        v -= theta * self._cost(m2, W)
                    if v > best + 1e-12:
                        best, keep, bK, bh = v, sel, K, h
        return best, keep, bK, bh

    def _true_g(self, trees, cand, can2, cs, v_now, W, rng, params, two, cache=None):
        c1, c2, g1, g2 = {}, {}, {K: {} for K in self.k_options}, {K: {} for K in self.k_options}
        for i in cand:
            c1[i] = (cache.pop(i, None) if cache is not None else None) or self._grow(trees[i], W, rng, params)
            v1 = self._vals(cs[i], self._es(c1[i]))
            v2 = None
            if two and i in can2:
                c2[i] = self._grow(c1[i], W, rng, params)
                v2 = self._vals(cs[i], self._es(c2[i]))
            for K in self.k_options:
                g1[K][i] = v1[K] - v_now[i][K]
                g2[K][i] = (v2[K] - v1[K]) if v2 is not None else 0.0
        return c1, c2, g1, g2

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        b = len(slots)
        h1 = sum(int(world.reqs[r]["H"][j] == 0) for r, j in slots)
        if self.t1_n is None:
            self.t1_hit, self.t1_n = float(h1), float(b)
        else:
            self.t1_hit = 0.95 * self.t1_hit + 0.05 * h1
            self.t1_n = 0.95 * self.t1_n + 0.05 * b
        W, _ = self.wk(b, self.cls())
        Ks = self.k_options
        Kmax = max(Ks)
        world.first_W = W
        trees = [X.Tree2(world.reqs[r], j, world) for r, j in slots]
        rids = [r for r, _ in slots]
        n = len(trees)
        params = (world.a, world.g, world.beta, 0.0)
        for t in trees:
            t.expand(W, rng, params)
        rem = [world.reqs[r]["L"] - j for r, j in slots]
        theta = self._theta_b(b)
        warm = self.rounds < self.warm or theta is None
        cs = [self._c(r) for r in rids]
        e_now = [self._es(t) for t in trees]
        e_prev = [{K: 0.0 for K in Ks} for _ in range(n)]
        v_now = [self._vals(cs[i], e_now[i]) for i in range(n)]
        v_prev = [{K: 0.0 for K in Ks} for _ in range(n)]
        active, depth, time_ = list(range(n)), 1, 0.0
        pending, cache = [], {}
        while active:
            cand = [i for i in active if depth < world.DMAX - 1 and depth + 1 <= max(1, rem[i] - 1)]
            if not cand:
                break
            c1 = {}
            if warm:
                keep = [i for i in cand if e_now[i][Kmax] - e_prev[i][Kmax] > 0.4]
            else:
                Jk = {K: sum(v_now[i][K] for i in range(n)) - theta * self.Tv(b, K) for K in Ks}
                stop = max(Jk.values())
                kstar = max(Jk, key=Jk.get)
                can2 = {i for i in cand if depth + 1 < world.DMAX - 1 and depth + 2 <= max(1, rem[i] - 1)}
                if self.look in ("true1", "true2"):
                    c1, c2, g1, g2 = self._true_g(trees, cand, can2, cs, v_now, W, rng, params,
                                                  self.look == "true2", cache)
                else:
                    g1 = {K: {i: v_now[i][K] - v_prev[i][K] for i in cand} for K in Ks}
                    g2 = g1
                    c2 = {}
                hs = (1, 2) if self.look in ("opt2", "true2") else (1,)
                best, keep, bK, bh = self._best(cand, can2, g1, g2, Jk, stop, theta, W, hs)
                if bh == 2:
                    b1 = self._best(cand, can2, g1, g2, Jk, stop, theta, W, (1,))[0]
                    if b1 <= 0:
                        self.dg["two_only_used"] += 1
                if keep:
                    self.dg["layers"] += 1
                if not keep and self.diag:
                    self.dg["stops"] += 1
                    vo, _, Ko, _ = self._best(cand, can2, g1, g1, Jk, stop, theta, W, (2,))
                    if vo > 0:
                        self.dg["cross_opt"] += 1
                        self.dg["cross_opt_val"] += vo
                        self.dg["cross_opt_otherK"] += int(Ko != kstar)
                    _, _, tg1, tg2 = self._true_g(trees, cand, can2, cs, v_now, W, self.arng, params, True)
                    t1 = self._best(cand, can2, tg1, tg2, Jk, stop, theta, W, (1,))[0]
                    t2, _, Kt, _ = self._best(cand, can2, tg1, tg2, Jk, stop, theta, W, (2,))
                    if t1 > 0:
                        self.dg["lag1_true"] += 1
                        self.dg["lag1_val"] += t1
                    elif t2 > 0:
                        self.dg["cross_true"] += 1
                        self.dg["cross_true_val"] += t2
                        self.dg["cross_true_otherK"] += int(Kt != kstar)
                cache = {i: c2[i] for i in keep if i in c2}
            if not keep:
                break
            for i in keep:
                if i in c1:
                    trees[i] = c1[i]
                else:
                    trees[i].expand(W, rng, params)
                e_prev[i], e_now[i] = e_now[i], self._es(trees[i])
                v_prev[i], v_now[i] = v_now[i], self._vals(cs[i], e_now[i])
            dt = world.real_cd(rng, len(keep), W)
            pending.append(((F.bucket(len(keep)), W), dt))
            time_ += dt
            active = keep
            depth += 1
        for key, dt in pending:
            self._obs_cost(key, dt)
        # ---- verify count ----
        Jk = None
        if warm:
            K = self.wk(b, self.cls())[1]
        else:
            Jk = {K_: sum(v_now[i][K_] for i in range(n)) - theta * self.Tv(b, K_) for K_ in Ks}
            K = max(Jk, key=Jk.get)
        self._jk = Jk
        self._vk = sum(v_now[i][K] for i in range(n)) if Jk else 0.0
        K0 = K
        forced = self._forced_K(b, K)
        if forced is not None:
            K = int(forced)
        self.k_log[K] = self.k_log.get(K, 0) + 1
        if not warm:
            for i in range(n):
                self.waste["beyond"] += cs[i] * (e_now[i][Kmax] - e_now[i][K])
                self.waste["total"] += cs[i] * e_now[i][Kmax]
        need_all = (not warm) and (self.replay or self.fband)
        if need_all:
            aK = [{K_: P.last_node(t, t.topk(K_, False))[0] for K_ in Ks} for t in trees]
            acc = [aK[i][K] for i in range(n)]
        else:
            acc = [P.last_node(t, t.topk(K, False))[0] for t in trees]
        com = [min(a + 1, rem[i]) for i, a in enumerate(acc)]
        t_round = time_ + world.real_rtt(rng, rtt) + world.real_T(rng, b, K)
        if need_all:
            pb, ab = {K_: 0.0 for K_ in Ks}, {K_: 0.0 for K_ in Ks}
            for i, t in enumerate(trees):
                ph = "early" if self.req_n.get(rids[i], 0) < 10 else "late"
                db = _dbin(max(t.dep))
                lo = None
                for Kb in Ks:
                    pr = cs[i] * (e_now[i][Kb] - (e_now[i][lo] if lo else 0.0))
                    ac = aK[i][Kb] - (aK[i][lo] if lo else 0)
                    pb[Kb] += pr
                    ab[Kb] += ac
                    s = self.bias.setdefault((Kb, ph, db), [0.0, 0.0, 0])
                    s[0] += pr; s[1] += ac; s[2] += 1
                    lo = Kb
            for Kb in Ks:
                self.fS[Kb][0] += pb[Kb]
                self.fS[Kb][1] += ab[Kb]
                if self.fband == "online":
                    self.f[Kb] = (self.fS[Kb][1] + self.f_prior) / (self.fS[Kb][0] + self.f_prior)
            if self.replay:
                tokK = {K_: float(sum(min(aK[i][K_] + 1, rem[i]) for i in range(n))) for K_ in Ks}
                self.rec.append((b, theta, K, K0, t_round, {K_: world.T_t(b, K_) for K_ in Ks},
                                 {K_: self.Tv(b, K_) for K_ in Ks}, tokK, pb, ab))
        self._after_round(b, K, t_round - time_)
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
            e = e_now[i][K]
            r = rids[i]
            self.req_a[r] = self.req_a.get(r, 0.0) + acc[i]
            self.req_e[r] = self.req_e.get(r, 0.0) + e
            self.c_a += acc[i]; self.c_e += e
            self.req_n[r] = self.req_n.get(r, 0) + 1
        self.rounds += 1
        return com, t_round

    # ---- replay summary (S2) ----
    def replay_summary(self):
        if not self.rec:
            return None
        Ks = self.k_options
        fh = {K: (self.fS[K][1] / self.fS[K][0]) if self.fS[K][0] > 0 else 1.0 for K in Ks}

        def rate(pick):
            tok = tm = 0.0
            for (b, th, K, K0, tr, Tt, Tv, tokK, pb, ab) in self.rec:
                k = pick(b, th, K, Tt, Tv, tokK, pb)
                tok += tokK[k]
                tm += tr - Tt[K] + Tt[k]
            return 1000.0 * tok / tm

        def corr_pick(f):
            def pick(b, th, K, Tt, Tv, tokK, pb):
                best, bk, cum = None, None, 0.0
                for k in Ks:
                    cum += f[k] * pb[k]
                    v = cum - th * Tv[k]
                    if best is None or v > best:
                        best, bk = v, k
                return bk
            return pick

        chosen = rate(lambda b, th, K, Tt, Tv, tokK, pb: K)
        fixed = {k: rate(lambda b, th, K, Tt, Tv, tokK, pb, k=k: k) for k in Ks}
        corrected = rate(corr_pick(fh))
        oracle = rate(lambda b, th, K, Tt, Tv, tokK, pb: max(Ks, key=lambda k: tokK[k] - th * Tt[k]))
        nd = sum(1 for (b, th, K, K0, tr, Tt, Tv, tokK, pb, ab) in self.rec
                 if corr_pick(fh)(b, th, K, Tt, Tv, tokK, pb) != K)
        bias = {f"{k[0]}|{k[1]}|{k[2]}": [round(v[0], 2), round(v[1], 2), v[2]] for k, v in self.bias.items()}
        return dict(rounds=len(self.rec), f_hind={str(k): round(v, 3) for k, v in fh.items()},
                    rate_chosen=chosen, rate_fixed={str(k): v for k, v in fixed.items()}, rate_corrected=corrected,
                    rate_oracle_round=oracle, share_changed=round(nd / len(self.rec), 4), bias=bias)


class BatchAccount7L(D.BatchAccount6L, BatchAccount7):
    def __init__(self, *a, interleave=True, probe_p=0.0, neartie=None, n_probe_min=8, **kw):
        super().__init__(*a, **kw)
        self.interleave, self.probe_p, self.neartie, self.n_probe_min = interleave, probe_p, neartie, n_probe_min
        self.pacc = {}
        self._probe = None
        self.n_probe = self.n_probe_trig = 0

    def _burst_queue(self):
        if not self.interleave:
            return [k for k in self.k_options for _ in range(self.burst_len2)]
        q = []
        for _ in range(self.burst_len2):
            q += [int(k) for k in self.drng.permutation(list(self.k_options))]
        return q

    def _forced_K(self, b, K_now):
        self._probe = None
        st = self.lv.get(b)
        if st and st["queue"]:
            return st["queue"].pop(0)
        if st is None or st["burst"] is not None or st["off"] is None or st["n"] <= self.n_off or self.rounds < self.warm:
            return None
        alt = None
        others = [k for k in self.k_options if k != K_now]
        if self.probe_p and self.drng.random() < self.probe_p:
            alt = int(self.drng.choice(others))
        elif self.neartie is not None and self._jk:
            gaps = {k: self._jk[K_now] - self._jk[k] for k in others}
            k2 = min(gaps, key=gaps.get)
            if gaps[k2] < self.neartie * max(1e-9, self._vk):
                alt = int(k2)
        if alt is not None:
            self._probe = alt
            self.n_probe += 1
        return alt

    def _start_burst(self, st):
        st["queue"] = self._burst_queue()
        st["burst"] = {}
        self.n_burst += 1

    def _after_round(self, b, K, wait):
        D.BatchAccount6._after_round(self, b, K, wait)
        if self.rounds < self.warm:
            return
        st = self.lv.setdefault(b, dict(n=0, off=None, dev=0.0, over=0, queue=[], burst=None, cool=0))
        if st["burst"] is not None:
            st["burst"].setdefault(K, []).append(wait)
            if not st["queue"]:
                means = {k: float(np.mean(v)) for k, v in st["burst"].items()}
                ref = self.k_options[0]
                for k in self.k_options:
                    if k in means and ref in means:
                        self.corr[(b, k)] = (means[k] - means[ref]) - (self.table(b, k) - self.table(b, ref))
                self.corr[(b, ref)] = 0.0
                st.update(burst=None, n=0, off=None, dev=0.0, over=0, cool=self.cooldown)
                for key in [k for k in self.pacc if k[0] == b]:
                    del self.pacc[key]
            return
        resid = wait - self.Tv(b, K)
        if self._probe is not None:
            acc = self.pacc.setdefault((b, K), [])
            acc.append(resid - st["off"])
            del acc[:-3 * self.n_probe_min]
            if len(acc) >= self.n_probe_min and st["cool"] == 0:
                m = float(np.mean(acc))
                se = float(np.std(acc, ddof=1)) / math.sqrt(len(acc))
                tol = max(self.abs_tol2, self.rel_tol2 * self.Tv(b, K))
                if abs(m) > tol and abs(m) > 3 * se:
                    self.n_probe_trig += 1
                    self._start_burst(st)
            return
        st["n"] += 1
        if st["n"] <= self.n_off:
            st["off"] = resid if st["off"] is None else 0.9 * st["off"] + 0.1 * resid
            return
        if st["cool"] > 0:
            st["cool"] -= 1
        st["dev"] = 0.95 * st["dev"] + 0.05 * (resid - st["off"])
        tol = max(self.abs_tol2, self.rel_tol2 * self.Tv(b, K))
        st["over"] = st["over"] + 1 if abs(st["dev"]) > tol else 0
        if st["over"] >= self.sustain and st["cool"] == 0:
            self._start_burst(st)
