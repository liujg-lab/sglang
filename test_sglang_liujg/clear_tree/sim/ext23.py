"""Round 29: guardrails for the v11 replay controller with online costs (D65 included), plus verify drift detection.

BatchAccount16 = ext22.BatchAccount15 (replay14, aware costs, resid timing) with:
  vdrift      verify residual drift detection as in README 3.8: per (b, K), fast EMA (0.2) of observed / predicted - 1
              of the verify core time (round wait minus the first-layer estimate); beyond drift_tol for drift_run
              observations in a row -> cost event (same handling as a layer event). Monitored once the b bucket's
              timing probes are done and the pooled regression has >= 6 samples. Earlier rounds (<= 28) had only the
              layer detector.
  G1 cooldown after a cost event, triggers are ignored for cd_len rounds (all rounds, warm-up included); an event within
              2 x cd_len of the previous one doubles cd_len (max cd_max), otherwise cd_len returns to cd0. Probe cap:
              a timing-probe round is allowed only while n_probe_rounds < max(pcap_base, pcap x rounds); otherwise the
              queue waits.
  G3 no_theta_probe: timing-probe rounds do not update Theta_b (per-b EMA) nor the global price EMA.
  G4 spread: a timing-probe round of a b bucket only if >= spread controller-eligible rounds of that bucket passed
              since its previous probe round; while a bucket's probe queue is not empty, the replay controller of
              that bucket does not accumulate (the replay would use half-learned costs). gap_hold = N: after a gap in
              the speculative rounds (Target-only rounds in between, i.e. the global round index jumped), no timing
              probe for the next N speculative rounds (the D42 switch probes speculation with 8 rounds).
Fallbacks while a regression has too few samples stay as in rounds 20-28 (5 ms per layer, 40 ms verify).
"""
import ext2 as X
import ext9 as Q
import ext10 as P
import ext20 as V
import ext22 as E
import feasibility as F


class BatchAccount16(E.BatchAccount15):
    def __init__(self, vdrift=True, cooldown=False, cd0=300, cd_max=4800, pcap=0.0, pcap_base=54,
                 no_theta_probe=False, spread=0, gap_hold=0, **kw):
        kw.setdefault("reset_on_event", True)
        kw.setdefault("cprobe_period", 300)
        super().__init__(**kw)
        self.vdrift, self.cooldown, self.cd0, self.cd_len, self.cd_max = vdrift, cooldown, cd0, cd0, cd_max
        self.pcap, self.pcap_base, self.no_theta_probe, self.spread = pcap, pcap_base, no_theta_probe, spread
        self.last_ev, self.n_suppressed, self.ev_log = None, 0, []
        self.last_probe, self.n_probe_wait = {}, 0
        self.gap_hold, self.hold, self.last_rnd, self.n_gaps = gap_hold, 0, None, 0
        self.theta_log = []

    # ---------------- hooks (round 30) ----------------
    def _g(self, t, c, K, vn, vp):
        """Predicted gain of the next layer for one request and K: the last increment (v11)."""
        return vn[K] - vp[K]

    def _probe_W(self, b, cls):
        s = self.ctx14.get(Q.bkey(b))
        return s["cur"] if s else V.flat_wk(b, cls)[0]

    # ---------------- drift detection and cost events ----------------
    def _drift(self, key, dev, trig, kind):
        f = self.fast.setdefault(key, [dev, 0])
        f[0] = 0.8 * f[0] + 0.2 * dev
        if abs(f[0]) > self.drift_tol:
            f[1] += 1
            if f[1] >= self.drift_run:
                trig.append(kind)
        else:
            f[1] = 0

    def _observe(self, b, W, K, layers, wait, suml_all):
        if self.cost_mode == "known":
            return
        trig = []
        for m, suml, dt in layers:
            tier = F.bucket(m)
            pred = self._lt(m, W, suml)
            self.rl[W].add([1.0, tier, suml / 1000.0], dt)
            if (tier, W) in self.ref:
                self._drift((tier, W), dt / max(pred, 1e-6) - 1, trig, "layer")
        bk = Q.bkey(b)
        vpred = self._vt(b, K, suml_all)
        core = wait - self._lt(b, W, suml_all)
        mon = self.vdrift and self.rv.n >= 6 and bk in self.probed and not self.probe_q.get(bk)
        self.rv.add(self._vx(b, K, suml_all), core)
        if mon:
            self._drift(("v", b, K), core / max(vpred, 1e-6) - 1, trig, "verify")
        if trig:
            self._event(trig)

    def _event(self, kinds):
        if self.cooldown and self.last_ev is not None and self.rounds < self.last_ev + self.cd_len:
            self.n_suppressed += 1
            for f in self.fast.values():
                f[1] = 0
            return
        if self.cooldown and self.last_ev is not None:
            self.cd_len = min(2 * self.cd_len, self.cd_max) if self.rounds < self.last_ev + 2 * self.cd_len else self.cd0
        self.last_ev = self.rounds
        self.n_cost_events += 1
        if len(self.ev_log) < 40:
            self.ev_log.append((self.rounds, self.world.n_T, sorted(set(kinds))))
        for r in list(self.rl.values()) + [self.rv]:
            r.A *= 0.0; r.y *= 0.0; r.n = 0; r._w = None
        self.ref, self.fast, self.probed, self.probe_q = {}, {}, set(), {}
        if self.reset_on_event:
            for st in self.ctx14.values():
                st["stats"] = {}
            self.n_resets += 1

    # ---------------- one round (ext22.BatchAccount15.round with the guardrails) ----------------
    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        self.world = world
        b = len(slots)
        if self.gap_hold and self.last_rnd is not None and rnd - self.last_rnd > 1:
            self.hold = self.gap_hold
            self.n_gaps += 1
        self.last_rnd = rnd
        h1 = sum(int(world.reqs[r]["H"][j] == 0) for r, j in slots)
        if self.t1_n is None:
            self.t1_hit, self.t1_n = float(h1), float(b)
        else:
            self.t1_hit = 0.95 * self.t1_hit + 0.05 * h1
            self.t1_n = 0.95 * self.t1_n + 0.05 * b
        cls = self.cls()
        _, Ktab = V.flat_wk(b, cls)
        theta = self._theta_b(b)
        warm = self.rounds < self.warm or theta is None
        rids = [r for r, _ in slots]
        ell = [world.reqs[r]["p0"] + j for r, j in slots]
        suml_all = float(sum(ell))
        kforce, probe, tag = None, False, "normal"
        self._probe_now = False
        bk = Q.bkey(b)
        pending = False
        if not warm and self.cost_mode != "known":
            if bk not in self.probed:
                q = [(w_, k_) for w_ in self.w_opts for k_ in self.k_options] * self.probe_passes
                self.prng2.shuffle(q)
                self.probe_q[bk] = [tuple(int(v) for v in x) for x in q]
                self.probed.add(bk)
            q = self.probe_q.get(bk)
            allow = bool(q)
            if allow and self.hold > 0:
                allow = False
            if allow and self.spread:
                allow = self.bk_cnt.get(bk, 0) - self.last_probe.get(bk, -10 ** 9) >= self.spread
            if allow and self.pcap:
                allow = self.n_probe_rounds < max(self.pcap_base, self.pcap * self.rounds)
            if q and not allow:
                self.n_probe_wait += 1
            if allow:
                W, kforce = q.pop(0)
                if W is None:                       # round 30: a probe that forces only K (W as the controller says)
                    W = self._probe_W(b, cls)
                probe = True
                self.n_probe_rounds += 1
                self.last_probe[bk] = self.bk_cnt.get(bk, 0)
                if not q:
                    for w_ in self.w_opts:
                        for t_ in (1, 2, 4, 8):
                            self.ref.setdefault((t_, w_), 1.0)
            pending = bool(self.probe_q.get(bk))
        self._probe_now = probe
        if not probe:
            W, tag = self._choose(b, cls, warm, slots, ell, theta)
            if not warm and self.wmode == "replay14":
                wc = self._cprobe(b)
                if wc is not None:
                    W, tag = wc, "cprobe"
                    self.n_cprobe += 1
        if not warm:
            self.bk_cnt[bk] = self.bk_cnt.get(bk, 0) + 1
            self.last_real[(bk, W)] = self.bk_cnt[bk]
        self.w_log[W] = self.w_log.get(W, 0) + 1
        Ks = self.k_options
        Kmax = max(Ks)
        world.first_W = W
        trees = [X.Tree2(world.reqs[r], j, world) for r, j in slots]
        n = len(trees)
        params = (world.a, world.g, world.beta, 0.0)
        for t in trees:
            t.expand(W, rng, params)
        rem = [world.reqs[r]["L"] - j for r, j in slots]
        cs = [self._c(r) for r in rids]
        e_now = [self._es(t) for t in trees]
        e_prev = [{K: 0.0 for K in Ks} for _ in range(n)]
        v_now = [self._vals(cs[i], e_now[i]) for i in range(n)]
        v_prev = [{K: 0.0 for K in Ks} for _ in range(n)]
        kd = self._kd_hat(W)
        active, depth, time_ = list(range(n)), 1, 0.0
        layers = []
        while active:
            cand = [i for i in active if depth < world.DMAX - 1 and depth + 1 <= max(1, rem[i] - 1)]
            if not cand:
                break
            if warm:
                keep = [i for i in cand if e_now[i][Kmax] - e_prev[i][Kmax] > 0.4]
            else:
                Jk = {K: sum(v_now[i][K] for i in range(n)) - theta * self._vt(b, K, suml_all) for K in Ks}
                stop = max(Jk.values())
                best, keep = 0.0, []
                for K in Ks:
                    g = {i: self._g(trees[i], cs[i], K, v_now[i], v_prev[i]) for i in cand}
                    sc = {i: g[i] - theta * kd * ell[i] for i in cand}
                    order = sorted(cand, key=lambda i: sc[i], reverse=True)
                    for m in self._mopts(len(order)):
                        sel = order[:m]
                        v = Jk[K] + sum(g[i] for i in sel) - theta * self._lt(m, W, sum(ell[i] for i in sel)) - stop
                        if v > best + 1e-12:
                            best, keep = v, sel
            if not keep:
                break
            for i in keep:
                trees[i].expand(W, rng, params)
                e_prev[i], e_now[i] = e_now[i], self._es(trees[i])
                v_prev[i], v_now[i] = v_now[i], self._vals(cs[i], e_now[i])
            sl = float(sum(ell[i] for i in keep))
            dt = world.real_cd_l(rng, len(keep), W, sl)
            if not warm and not probe and self.cost_mode != "known":
                self.err_l.append(abs(self._lt(len(keep), W, sl) / world.c_d_l(len(keep), W, sl) - 1))
            time_ += dt
            layers.append((len(keep), sl, dt))
            active = keep
            depth += 1
        if warm:
            K = Ktab
        else:
            Jk = {K_: sum(v_now[i][K_] for i in range(n)) - theta * self._vt(b, K_, suml_all) for K_ in Ks}
            K = max(Jk, key=Jk.get)
        if kforce is not None:
            K = kforce
        self.k_log[K] = self.k_log.get(K, 0) + 1
        acc = [P.last_node(t, t.topk(K, False))[0] for t in trees]
        com = [min(a + 1, rem[i]) for i, a in enumerate(acc)]
        wait = world.real_rtt(rng, rtt) + world.real_T_l(rng, b, K, suml_all)
        t_round = time_ + wait
        self._layers_now, self._W_now = layers, W
        feed = not warm and not probe and tag != "cprobe" and self.wmode == "replay14"
        if feed and self.spread and pending:
            feed = False
        if feed:
            lay2 = [(m, sl) for m, sl, _ in layers]
            rep = self._replay_m(trees, cs, rem, lay2, W, b, theta, suml_all)
            self._post14m(b, W, tag, rep, com, t_round, self._mt(b, W, K, lay2, suml_all))
        if not warm:
            self._observe(b, W, K, layers, wait, suml_all)
        for r, j in slots:
            hit = float(world.reqs[r]["H"][j] == 0)
            self.h_req[r] = hit if r not in self.h_req else 0.9 * self.h_req[r] + 0.1 * hit
        tok = float(sum(com))
        self.tok_ema = tok if self.tok_ema is None else 0.95 * self.tok_ema + 0.05 * tok
        self.tok_total += tok
        fx = t_round - time_
        self.fixed[b] = fx if b not in self.fixed else 0.9 * self.fixed[b] + 0.1 * fx
        if not (probe and self.no_theta_probe):
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
        if self.hold > 0:
            self.hold -= 1
        self.rounds += 1
        return com, t_round


# ---------------- worlds ----------------
def add_rtt(w, base=4.0, high=16.0, at=400, period=None):
    """RTT base -> high at the at-th speculative round (period=None), or alternating base / high every `period`
    speculative rounds starting at `at`. Jitter as before (exponential, mean 1 ms)."""
    orig = w.real_rtt

    def real_rtt(rng, _base):
        n = w.n_T
        if n < at:
            r = base
        elif period is None:
            r = high
        else:
            r = high if ((n - at) // period) % 2 == 0 else base
        return orig(rng, r)
    w.real_rtt = real_rtt


def add_switch_world(w, factor):
    """Make Target-only faster: a0 and a_b scaled by factor (< 1)."""
    w.a0 *= factor
    w.a_b *= factor
