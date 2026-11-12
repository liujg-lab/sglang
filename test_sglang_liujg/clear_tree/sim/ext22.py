"""Round 26: the v11 replay controller with the online cost model of D61 (BatchAccount15).

World: ext19.World22 (prefix-aware, strong prefix terms), as rounds 23 and 25.
BatchAccount15 = the v10 growth of ext18.BatchAccount11 (costs from cost_mode: "known" = the world's true functions,
"aware" = D61 online regressions with timing probes and residual drift detection) + the W choice of ext20 (flat /
fixed / replay14) + a replay that values every candidate with the policy's own cost model.

timing (how the replay controller times a candidate):
  model5   candidate time = model time of its schedule + 5 ms (rounds 23 and 25, ext20); the current width on normal
           rounds uses the measured round time
  resid    the v11 design (README 3.1): candidate time = measured round time - model time of what ran + model time of
           the candidate's schedule; the current width on explore rounds the same way
Timing-probe rounds (online modes) and warm-up rounds are not fed to the controller. Controller statistics are not
reset at cost events (v11 design).
With cost_mode known and timing model5, the policy reproduces ext20 replay14 (checked against r25).

Round 27: first_delay = N delays the first decision of each b bucket (before its first switch) until the controller
has seen N rounds of that bucket (timing-probe and warm-up rounds not counted); delay_reset = True also drops what was
accumulated before the gate (only rounds after it count).

Round 28: reset_on_event = True clears the controller statistics of every bucket at a cost event (drift detection);
cprobe_period = P > 0 runs continuous timing probes: a width other than the current one that has not run for P
non-warm rounds of the bucket gets cprobe_rounds real rounds (K chosen as usual); those rounds feed the cost model,
not the controller.
"""
import numpy as np
import ext2 as X
import ext9 as Q
import ext10 as P
import ext12 as RP
import ext20 as V
import diag24 as D          # noqa: F401  (registers the "high" workload)


class BatchAccount15(V.BatchAccount13):
    def __init__(self, wmode="replay14", cost_mode="aware", timing="resid", first_delay=0, delay_reset=False,
                 reset_on_event=False, cprobe_period=0, cprobe_rounds=2, **kw):
        super().__init__(wmode=wmode, cost_mode=cost_mode, **kw)
        self.timing = timing
        self.first_delay, self.delay_reset = first_delay, delay_reset
        self.reset_on_event, self.cprobe_period, self.cprobe_rounds = reset_on_event, cprobe_period, cprobe_rounds
        self.n_resets, self.n_cprobe = 0, 0
        self.bk_cnt, self.last_real, self.cprobe_q = {}, {}, {}

    def _observe(self, *a, **k):
        ev = self.n_cost_events
        super()._observe(*a, **k)
        if self.reset_on_event and self.n_cost_events > ev:
            for st in self.ctx14.values():
                st["stats"] = {}
            self.n_resets += 1

    def _cprobe(self, b):
        """Round 28: continuous low-rate timing probes. Every width other than the current one gets cprobe_rounds real
        rounds once it has not run for cprobe_period non-warm rounds of the bucket."""
        bk = Q.bkey(b)
        st = self.ctx14.get(bk)
        if not self.cprobe_period or st is None:
            return None
        n = self.bk_cnt.get(bk, 0)
        q = self.cprobe_q.setdefault(bk, [])
        if not q:
            for w_ in self.w_opts:
                if w_ == st["cur"]:
                    continue
                last = self.last_real.setdefault((bk, w_), n)
                if n - last >= self.cprobe_period:
                    q.extend([w_] * self.cprobe_rounds)
                    break
        return q.pop(0) if q else None

    # model time of a schedule under the policy's cost model (+ the 5 ms constant of ext20 for model5)
    def _mt(self, b, W, K, layers, suml_all):
        t = sum(self._lt(m, W, sl) for m, sl in layers) + self._lt(b, W, suml_all) + self._vt(b, K, suml_all)
        return t + (5.0 if self.timing == "model5" else 0.0)

    def _replay_m(self, trees, cs, rem, layers, W, b, theta, suml_all):
        """ext20._replay with the policy's cost model: every W' <= W, cut up to 2 layers earlier, K' by J."""
        Dbuilt = 1 + len(layers)
        out = {}
        for Wn in V.W_OPTS:
            if Wn > W:
                continue
            nodes = [RP.narrowed(t, Wn) if Wn < W else list(t.ids) for t in trees]
            depths = [Dbuilt] if Wn == W else list(range(max(1, Dbuilt - 2), Dbuilt + 1))
            best = None
            for d in depths:
                sel = [[n for n in nodes[i] if t.dep[n] <= d] for i, t in enumerate(trees)]
                lt = sum(self._lt(m, Wn, sl) for m, sl in layers[:d - 1]) + self._lt(b, Wn, suml_all)
                for K in self.k_options:
                    v = sum(cs[i] * sum(sorted((t.score[n] for n in sel[i]), reverse=True)[:K]) for i, t in enumerate(trees))
                    J = v - theta * (lt + self._vt(b, K, suml_all))
                    if best is None or J > best[0]:
                        best = (J, d, K, sel, lt)
            _, d, K, sel, lt = best
            tok = [min(RP.accepted_in(t, RP.topk_of(t, sel[i], K)) + 1, rem[i]) for i, t in enumerate(trees)]
            out[Wn] = dict(tok=float(sum(tok)), mt=lt + self._vt(b, K, suml_all) + (5.0 if self.timing == "model5" else 0.0))
        return out

    def _post14m(self, b, W, tag, rep, com, t_round, m_actual):
        s = self.ctx14[Q.bkey(b)]
        cur = s["cur"]
        if cur not in rep:
            return
        if self.timing == "model5":
            tm = lambda x: rep[x]["mt"]
        else:
            tm = lambda x: t_round - m_actual + rep[x]["mt"]
        # round 27: before the first switch of a bucket, no decision until first_delay controller rounds;
        # delay_reset: rounds before the gate are not accumulated either
        gate = s["follow"] and s["n"] < self.first_delay
        if gate and self.delay_reset:
            s["n"] += 1
            return
        if tag == "explore":
            ctok, ctm = rep[cur]["tok"], tm(cur)
        else:
            ctok, ctm = float(sum(com)), t_round
        for o in rep:
            if o == cur:
                continue
            st = s["stats"].setdefault(o, [0.0, 0.0, 0.0, 0.0, 0])
            st[0] += rep[o]["tok"]; st[1] += tm(o); st[2] += ctok; st[3] += ctm; st[4] += 1
        s["n"] += 1
        if s["n"] % 20 == 0 and not (s["follow"] and s["n"] < self.first_delay):
            best, bg = None, 0.03
            for o, st in s["stats"].items():
                need = 30 if o < cur else 10
                if st[4] < need:
                    continue
                g = (st[0] / st[1]) / (st[2] / st[3]) - 1
                if g > bg:
                    best, bg = o, g
            if best is not None:
                s["cur"], s["follow"], s["stats"] = best, False, {}
                self.n_switch += 1
                if len(self.switch_log) < 30:
                    self.switch_log.append((self.rounds, b, cur, best))

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        self.world = world
        b = len(slots)
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
        if not warm and self.cost_mode != "known":
            bk = Q.bkey(b)
            if bk not in self.probed:
                q = [(w_, k_) for w_ in self.w_opts for k_ in self.k_options] * self.probe_passes
                self.prng2.shuffle(q)
                self.probe_q[bk] = [tuple(int(v) for v in x) for x in q]
                self.probed.add(bk)
            q = self.probe_q.get(bk)
            if q:
                W, kforce = q.pop(0)
                probe = True
                self.n_probe_rounds += 1
                if not q:
                    for w_ in self.w_opts:
                        for t_ in (1, 2, 4, 8):
                            self.ref.setdefault((t_, w_), 1.0)
        if not probe:
            W, tag = self._choose(b, cls, warm, slots, ell, theta)
            if not warm and self.wmode == "replay14":
                wc = self._cprobe(b)
                if wc is not None:
                    W, tag = wc, "cprobe"
                    self.n_cprobe += 1
        if not warm:
            bk_ = Q.bkey(b)
            self.bk_cnt[bk_] = self.bk_cnt.get(bk_, 0) + 1
            self.last_real[(bk_, W)] = self.bk_cnt[bk_]
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
                    g = {i: v_now[i][K] - v_prev[i][K] for i in cand}
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
        # the controller: replay with the costs as they stand before this round's observation
        if not warm and not probe and tag != "cprobe" and self.wmode == "replay14":
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
        self.rounds += 1
        return com, t_round
