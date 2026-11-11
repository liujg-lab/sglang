"""Round 16 (review 15): consistent growth/verify value, full K enumeration, and verify-time drift detection.

BatchAccount6 (W from the table, Theta per b, top-1 classifier, frozen costs, as BatchAccount4) with
  k_mode "early"       current v10 text: grow with E over K_max; before verify, step up K while the next step pays
                       (stops at the first step that does not)
         "enum"        grow with E over K_max; K* = argmax_K J_K over all captured K
         "consistent"  J_K = sum_i c_i E_i^K - Theta_b * Tv(b, K); after each layer, for every K and every m,
                       value = J_K + sum of the m largest g_i^K - Theta_b * C(m, W), with g_i^K = c_i * (last layer's
                       increase of request i's top-K mass); continue with the best (K, m) if it beats max_K J_K
                       (stopping now); finally K* = argmax_K J_K
  verify-time source   table(b, K) (what the policy believes) + per-(b, K) corrections
  drift=True           every probe_every rounds of a context, force another K for probe_len rounds; compare the
                       observed wait difference (round time minus Draft layer time, EMA per (b, K)) with the believed
                       one; if they disagree by more than max(abs_tol, rel_tol * |believed|), run a burst that forces
                       every K for burst_len rounds, then set the corrections so believed differences match observed
Records: the calibrated Draft mass grown beyond the chosen K (c_i * (top-K_max mass - top-K mass), summed) as a share
of the top-K_max mass (growth paid for nodes that did not make the verify set), probes, bursts.
"""
import numpy as np
import feasibility as F
import ext2 as X
import ext8 as N
import ext10 as P
import ext11 as O


class BatchAccount6(O.BatchAccount4):
    def __init__(self, wk, table, k_mode="early", drift=False, probe_every=100, probe_len=2, burst_len=3,
                 abs_tol=1.0, rel_tol=0.15, **kw):
        super().__init__(wk, verify_time=table, online_K=True, **kw)
        self.table, self.k_mode, self.drift = table, k_mode, drift
        self.probe_every, self.probe_len, self.burst_len, self.abs_tol, self.rel_tol = probe_every, probe_len, burst_len, abs_tol, rel_tol
        self.corr = {}                       # (b, K) -> ms added to the table
        self.obs = {}                        # (b, K) -> [EMA wait, n, last round]
        self.dstate = {}                     # b -> dict(rounds, queue)
        self.n_probe = self.n_burst = 0
        self.waste = dict(beyond=0.0, total=0.0)

    def Tv(self, b, K):
        return self.table(b, K) + self.corr.get((b, K), 0.0)

    def _J(self, trees, rids, K, b, theta):
        return sum(self._c(rids[i]) * N._mass(t, K) for i, t in enumerate(trees)) - theta * self.Tv(b, K)

    # ----- drift: which K to force this round (None = choose normally) -----
    def _forced_K(self, b, K_now):
        if not self.drift or self.rounds < self.warm:
            return None
        st = self.dstate.setdefault(b, dict(rounds=0, queue=[]))
        st["rounds"] += 1
        if st["queue"]:
            return st["queue"].pop(0)
        if st["rounds"] % self.probe_every == 0:
            alt = [k for k in self.k_options if k != K_now]
            if alt:
                k = alt[(st["rounds"] // self.probe_every) % len(alt)]
                st["queue"] = [k] * (self.probe_len - 1)
                st["probe"] = (k, K_now)
                self.n_probe += 1
                return k
        return None

    def _after_round(self, b, K, wait):
        o = self.obs.setdefault((b, K), [wait, 0, 0])
        o[1] += 1
        a = 0.5 if o[1] <= 3 else 0.2
        o[0] = (1 - a) * o[0] + a * wait if o[1] > 1 else wait
        o[2] = self.rounds
        st = self.dstate.get(b)
        if not self.drift or st is None:
            return
        if st.get("probe") and not st["queue"]:
            k_alt, k_ref = st.pop("probe")
            oa, orf = self.obs.get((b, k_alt)), self.obs.get((b, k_ref))
            if oa and orf and self.rounds - orf[2] < 50:
                d_obs, d_bel = oa[0] - orf[0], self.Tv(b, k_alt) - self.Tv(b, k_ref)
                if abs(d_obs - d_bel) > max(self.abs_tol, self.rel_tol * abs(d_bel)):
                    st["queue"] = [k for k in self.k_options for _ in range(self.burst_len)]
                    st["burst"] = True
                    self.n_burst += 1
        elif st.get("burst") and not st["queue"]:
            st["burst"] = False
            ref = self.k_options[0]
            for k in self.k_options:
                ok, orf = self.obs.get((b, k)), self.obs.get((b, ref))
                if ok and orf:
                    self.corr[(b, k)] = (ok[0] - orf[0]) - (self.table(b, k) - self.table(b, ref))
            self.corr[(b, ref)] = 0.0

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        b = len(slots)
        h1 = sum(int(world.reqs[r]["H"][j] == 0) for r, j in slots)
        if self.t1_n is None:
            self.t1_hit, self.t1_n = float(h1), float(b)
        else:
            self.t1_hit = 0.95 * self.t1_hit + 0.05 * h1
            self.t1_n = 0.95 * self.t1_n + 0.05 * b
        W, _ = self.wk(b, self.cls())
        Kmax = max(self.k_options)
        world.first_W = W
        trees = [X.Tree2(world.reqs[r], j, world) for r, j in slots]
        rids = [r for r, _ in slots]
        params = (world.a, world.g, world.beta, 0.0)
        for t in trees:
            t.expand(W, rng, params)
        rem = [world.reqs[r]["L"] - j for r, j in slots]
        Ks = self.k_options
        e_prev = {K: [0.0] * len(trees) for K in Ks}
        e_now = {K: [N._mass(t, K) for t in trees] for K in Ks}
        active, depth, time_ = list(range(len(trees))), 1, 0.0
        theta = self._theta_b(b)
        pending, beyond = [], 0
        warm = self.rounds < self.warm or theta is None
        while active:
            cand = [i for i in active if depth < world.DMAX - 1 and depth + 1 <= max(1, rem[i] - 1)]
            if not cand:
                break
            if warm:
                keep = [i for i in cand if e_now[Kmax][i] - e_prev[Kmax][i] > 0.4]
            elif self.k_mode == "consistent":
                Jk = {K: self._J(trees, rids, K, b, theta) for K in Ks}
                stop = max(Jk.values())
                best, keep = 0.0, []
                for K in Ks:
                    g = {i: self._c(rids[i]) * (e_now[K][i] - e_prev[K][i]) for i in cand}
                    order = sorted(cand, key=lambda i: g[i], reverse=True)
                    for m in sorted({len(order)} | {x for x in self.buckets if x < len(order)}):
                        v = Jk[K] + sum(g[i] for i in order[:m]) - theta * self._cost(m, W) - stop
                        if v > best:
                            best, keep = v, order[:m]
            else:
                g = {i: self._c(rids[i]) * (e_now[Kmax][i] - e_prev[Kmax][i]) for i in cand}
                order = sorted(cand, key=lambda i: g[i], reverse=True)
                best_m, best_v = 0, 0.0
                for m in sorted({len(order)} | {x for x in self.buckets if x < len(order)}):
                    v = sum(g[i] for i in order[:m]) - theta * self._cost(m, W)
                    if v > best_v:
                        best_m, best_v = m, v
                keep = order[:best_m]
            if not keep:
                break
            for i in keep:
                trees[i].expand(W, rng, params)
                for K in Ks:
                    e_prev[K][i] = e_now[K][i]
                    e_now[K][i] = N._mass(trees[i], K)
            dt = world.real_cd(rng, len(keep), W)
            pending.append(((F.bucket(len(keep)), W), dt))
            time_ += dt
            active = keep
            depth += 1
        for key, dt in pending:
            self._obs_cost(key, dt)
        # ---- verify count ----
        if warm:
            K = self.wk(b, self.cls())[1]
        elif self.k_mode == "early":
            K = Ks[0]
            sc = [np.sort(np.asarray(t.scores))[::-1] for t in trees]
            for K2 in Ks[1:]:
                extra = sum(self._c(rids[i]) * float(s[K:K2].sum()) for i, s in enumerate(sc))
                if extra > theta * (self.Tv(b, K2) - self.Tv(b, K)):
                    K = K2
                else:
                    break
        else:
            Jk = {K_: self._J(trees, rids, K_, b, theta) for K_ in Ks}
            K = max(Jk, key=Jk.get)
        forced = self._forced_K(b, K)
        if forced is not None:
            K = forced
        self.k_log[K] = self.k_log.get(K, 0) + 1
        if not warm:
            for i, t in enumerate(trees):
                c = self._c(rids[i])
                mk, mx = N._mass(t, K), N._mass(t, Kmax)
                self.waste["beyond"] += c * (mx - mk)
                self.waste["total"] += c * mx
        acc = [P.last_node(t, t.topk(K, False))[0] for t in trees]
        com = [min(a + 1, rem[i]) for i, a in enumerate(acc)]
        t_round = time_ + world.real_rtt(rng, rtt) + world.real_T(rng, b, K)
        self._after_round(b, K, t_round - time_)
        tok = float(sum(com))
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
        ab = max(a_, 1.0 / v[2])
        v[0] = (1 - ab) * v[0] + ab * tok
        v[1] = (1 - ab) * v[1] + ab * t_round
        for i, t in enumerate(trees):
            e = N._mass(t, K)
            r = rids[i]
            self.req_a[r] = self.req_a.get(r, 0.0) + acc[i]
            self.req_e[r] = self.req_e.get(r, 0.0) + e
            self.c_a += acc[i]; self.c_e += e
        self.rounds += 1
        return com, t_round


class BatchAccount6L(BatchAccount6):
    """Drift detection v2 ("level"): no periodic probes. Per context, learn the offset between the observed wait for the
    K in use and the table (RTT, host) over the first n_off rounds; then follow a slow EMA of the remaining deviation.
    If it stays above max(abs_tol2, rel_tol2 * table) for `sustain` consecutive rounds, run a burst that forces every
    K for burst_len2 rounds, set the corrections from the burst means, relearn the offset, and cool down."""

    def __init__(self, *a, n_off=50, sustain=30, abs_tol2=2.0, rel_tol2=0.10, burst_len2=8, cooldown=300, **kw):
        kw["drift"] = False
        super().__init__(*a, **kw)
        self.n_off, self.sustain, self.abs_tol2, self.rel_tol2, self.burst_len2, self.cooldown = n_off, sustain, abs_tol2, rel_tol2, burst_len2, cooldown
        self.lv = {}

    def _forced_K(self, b, K_now):
        st = self.lv.get(b)
        if st and st["queue"]:
            return st["queue"].pop(0)
        return None

    def _after_round(self, b, K, wait):
        super()._after_round(b, K, wait)
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
            return
        resid = wait - self.Tv(b, K)
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
            st["queue"] = [k for k in self.k_options for _ in range(self.burst_len2)]
            st["burst"] = {}
            self.n_burst += 1
