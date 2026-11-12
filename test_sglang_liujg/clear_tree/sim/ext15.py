"""Round 18: online update of the layer width W on top of the v10 rule (ext14.BatchAccount7, look=None).

All variants use measured costs: the Draft layer time lt(m, W) and the verify time come from the world's expected
values (a deployment micro-benchmark and V2, refreshed at once when the world changes, i.e. an ideal event-triggered
re-measure). Only the choice of W differs:

  table      W from the v10 table (built for the base costs), K online (v10)
  fixed      one W for every b (the per-scenario best one is the hindsight reference)
  oracle     a scheduled W (the best fixed W before / after a cost change)
  replay14   round-14 controller ported to v10: per b bucket, paired replay of narrower widths on every round and of one
             step wider on 5% exploration rounds; every 20 rounds switch if the paired gain > 3% (min 30 / 10 rounds);
             statistics reset after a switch; follows the table until its first switch
  rescore    keep the last `keep` rounds per cell (b bucket, acceptance class) with what is needed to re-score them:
             per replayable width W' and per K the calibrated value sum c_i * mass_K and the committed tokens of the
             narrowed tree, the active counts of the layers, the price, and the time residual (round time minus the
             model time of what ran, under the costs of that round). Score W' under the current costs: K' = argmax_K
             value - Theta * Tv(b, K), time = residual + model time of (W', K') now. Paired gain vs the current W over
             the records where both are replayable; time-block bootstrap (blocks of 10 records) lower 5% bound.
             Wider widths only on exploration records: steady p_explore (1%) at W_max, plus a burst of `burst` rounds at
             W_max in every cell right after a cost change. Decide every `every` rounds of a cell and after a burst.
             Gate: lower bound > gate, n >= min_n (min_n_wide for a wider candidate).
             confirm=True: a gated candidate first runs a real interleaved trial (blocks of 10 rounds, cand / cur,
             trial_blocks pairs) and is adopted if its rate beats the current one by > 1%; a rejected candidate waits
             `cooldown` rounds of the cell. confirm=False adopts on the gate.
             start_event=True also treats the deployment start as an event: every new cell whose table W is below
             W_max starts with a burst at W_max and a decision after it (added after part A showed that steady 1%
             exploration alone never reaches min_n_wide in a run of ~700-3000 rounds).
             explore_to="step" explores one width step above the current W instead of W_max (added after part A showed
             that exploring at W_max is costly where W_max is far off, e.g. b = 8 with expensive rows).
             keep_wide > 0 also keeps the last keep_wide records built wider than the cell's W in a separate buffer (added
             after part L: in 500-record windows the steady 1% exploration never holds 20 wide records, and the start
             burst is evicted after 500 rounds, so a too-narrow table was only caught right after the burst).
"""
import collections
import numpy as np
import ext2 as X
import ext8 as N
import ext9 as Q
import ext10 as P
import ext12 as RP
import ext14 as E
import feasibility as F


def _mass_nodes(t, nodes, K):
    s = sorted((t.score[n] for n in nodes), reverse=True)[:K]
    return float(sum(s))


class BatchAccount8(E.BatchAccount7):
    def __init__(self, wk, table, lt, vt_core, wmode="table", fixed_W=None, sched=None, w_options=(2, 4, 8),
                 p_explore14=0.05, every14=20, hyst14=0.03, min_n14=30, min_n_explore14=10,
                 keep=500, p_explore=0.01, burst=30, every=50, gate=0.01, min_n=100, min_n_wide=20, n_boot=200,
                 confirm=True, trial_blocks=3, trial_len=10, trial_margin=0.01, cooldown=300, boot_seed=99,
                 start_event=False, explore_to="max", keep_wide=0, **kw):
        super().__init__(wk, table, **kw)
        self.wk0 = wk
        self.lt, self.vt_core = lt, vt_core
        self.wmode, self.fixed_W, self.sched = wmode, fixed_W, sched
        self.w_opts = tuple(sorted(w_options))
        self.p14, self.every14, self.hyst14, self.min_n14, self.min_ne14 = p_explore14, every14, hyst14, min_n14, min_n_explore14
        self.keep, self.p_explore, self.burst, self.every, self.gate = keep, p_explore, burst, every, gate
        self.min_n, self.min_n_wide, self.n_boot = min_n, min_n_wide, n_boot
        self.confirm, self.trial_blocks, self.trial_len, self.trial_margin, self.cooldown = confirm, trial_blocks, trial_len, trial_margin, cooldown
        self.wrng = np.random.default_rng(31337)
        self.brng = np.random.default_rng(boot_seed)
        self.ctx14, self.cells = {}, {}
        self.w_log, self.n_switch, self.n_explore, self.n_trial, self.n_trial_ok = {}, 0, 0, 0, 0
        self.n_events, self.switch_log = 0, []
        self._probe = None
        self.start_event = start_event       # treat the deployment start as a cost event (a burst in every new cell)
        self.explore_to = explore_to         # "max": explore at W_max; "step": one width step above the current W
        self.keep_wide = keep_wide           # > 0: also keep the last keep_wide records built wider than the cell's W

    # measured layer time (deployment table), replaces the learned cell model
    def _cost(self, m, W):
        return self.lt(m, W)

    def _model(self, b, W, K, layer_m):
        return self.lt(b, W) + sum(self.lt(m, W) for m in layer_m) + self.vt_core(b, K)

    def _wx(self, cur):
        if self.explore_to == "step":
            return min(w for w in self.w_opts if w > cur)
        return self.w_opts[-1]

    # ---------------- W choice ----------------
    def _cell(self, b, cls, Wtab):
        key = (Q.bkey(b), cls)
        c = self.cells.get(key)
        if c is None:
            c = self.cells[key] = dict(cur=Wtab, follow=True, recs=collections.deque(maxlen=self.keep), n=0, queue=[],
                                       trial=None, cool={}, burst_left=0, decide_after_burst=False,
                                       wrecs=collections.deque(maxlen=max(1, self.keep_wide)))
            if self.start_event and Wtab < self.w_opts[-1] and self.burst > 0:
                c["queue"] = [(self._wx(Wtab), "explore")] * self.burst
                c["burst_left"], c["decide_after_burst"] = self.burst, True
        if c["follow"]:
            c["cur"] = Wtab
        return c

    def _choose(self, world, b, cls, Wtab, warm):
        if self.wmode == "table" or warm:
            return Wtab, "normal"
        if self.wmode == "fixed":
            return self.fixed_W, "normal"
        if self.wmode == "oracle":
            return self.sched(b, world), "normal"
        if self.wmode == "replay14":
            k = Q.bkey(b)
            s = self.ctx14.get(k)
            if s is None:
                s = self.ctx14[k] = dict(cur=Wtab, follow=True, stats={}, n=0)
            if s["follow"] and s["cur"] != Wtab:
                s["cur"], s["stats"] = Wtab, {}
            wider = [w for w in self.w_opts if w > s["cur"]]
            if wider and self.wrng.random() < self.p14:
                self.n_explore += 1
                return wider[0], "explore"
            return s["cur"], "normal"
        # rescore
        c = self._cell(b, cls, Wtab)
        if c["queue"]:
            W, tag = c["queue"].pop(0)
            return W, tag
        Wmax = self.w_opts[-1]
        if c["cur"] < Wmax and self.wrng.random() < self.p_explore:
            self.n_explore += 1
            return self._wx(c["cur"]), "explore"
        return c["cur"], "normal"

    # ---------------- replay of one round ----------------
    def _replay_tables(self, trees, cs, rem, W):
        """For every W' <= W: value[W'][K] (calibrated mass) and tok[W'][K] (committed tokens)."""
        val, tok = {}, {}
        for Wn in self.w_opts:
            if Wn > W:
                continue
            vK, tK = {K: 0.0 for K in self.k_options}, {K: 0.0 for K in self.k_options}
            for i, t in enumerate(trees):
                nodes = RP.narrowed(t, Wn) if Wn < W else list(t.ids)
                for K in self.k_options:
                    vK[K] += cs[i] * _mass_nodes(t, nodes, K)
                    tK[K] += min(RP.accepted_in(t, RP.topk_of(t, nodes, K)) + 1, rem[i])
            val[Wn], tok[Wn] = vK, tK
        return val, tok

    def _pickK(self, vK, theta, b):
        return max(self.k_options, key=lambda K: vK[K] - theta * self.vt_core(b, K))

    def _post14(self, b, W, tag, val, tok, layer_m, theta, K, com, t_round):
        s = self.ctx14[Q.bkey(b)]
        cur = s["cur"]
        resid = t_round - self._model(b, W, K, layer_m)
        if tag == "explore":
            if cur not in val:
                return
            kc = self._pickK(val[cur], theta, b)
            ctok, ctm = tok[cur][kc], resid + self._model(b, cur, kc, layer_m)
        else:
            ctok, ctm = float(sum(com)), t_round
        for Wn in val:
            if Wn == cur:
                continue
            kn = self._pickK(val[Wn], theta, b)
            v = s["stats"].setdefault(Wn, [0.0, 0.0, 0.0, 0.0, 0])
            v[0] += tok[Wn][kn]; v[1] += resid + self._model(b, Wn, kn, layer_m)
            v[2] += ctok; v[3] += ctm; v[4] += 1
        s["n"] += 1
        if s["n"] % self.every14 == 0:
            best, bg = None, self.hyst14
            for o, v in s["stats"].items():
                need = self.min_n14 if o <= cur else self.min_ne14
                if v[4] < need or v[1] <= 0 or v[3] <= 0:
                    continue
                g = (v[0] / v[1]) / (v[2] / v[3]) - 1
                if g > bg:
                    best, bg = o, g
            if best is not None:
                s["cur"], s["follow"], s["stats"] = best, False, {}
                self.n_switch += 1
                self.switch_log.append((self.rounds, b, cur, best))

    def _score(self, c, b, Wc, Wn):
        """Per-record (tok, time) for Wc and Wn on records where both are replayable, under the current costs."""
        rows = []
        recs = list(c["recs"])
        if self.keep_wide:
            seen = {id(r) for r in recs}
            recs = [r for r in c["wrecs"] if id(r) not in seen] + recs
        for r in recs:
            if r["b"] != b and Q.bkey(r["b"]) != Q.bkey(b):
                continue
            if max(Wc, Wn) > r["W"]:
                continue
            out = []
            for Wx in (Wc, Wn):
                k = self._pickK(r["val"][Wx], r["theta"], r["b"])
                out.append((r["tok"][Wx][k], r["resid"] + self._model(r["b"], Wx, k, r["layer_m"])))
            rows.append((out[0][0], out[0][1], out[1][0], out[1][1]))
        return np.array(rows) if rows else np.zeros((0, 4))

    def _gain_lb(self, rows):
        n = len(rows)
        g = (rows[:, 2].sum() / rows[:, 3].sum()) / (rows[:, 0].sum() / rows[:, 1].sum()) - 1
        bl = 10
        nb = max(1, n // bl)
        blocks = [rows[i * bl:(i + 1) * bl] if i < nb - 1 else rows[i * bl:] for i in range(nb)]
        sums = np.array([blk.sum(0) for blk in blocks])
        gs = []
        for _ in range(self.n_boot):
            idx = self.brng.integers(0, nb, nb)
            s = sums[idx].sum(0)
            gs.append((s[2] / s[3]) / (s[0] / s[1]) - 1)
        return g, float(np.quantile(gs, 0.05))

    def _decide(self, c, b):
        cur = c["cur"]
        best, best_lb = None, self.gate
        for Wn in self.w_opts:
            if Wn == cur or c["cool"].get(Wn, -1) > c["n"]:
                continue
            rows = self._score(c, b, cur, Wn)
            need = self.min_n if Wn < cur else self.min_n_wide
            if len(rows) < need:
                continue
            g, lb = self._gain_lb(rows)
            if lb > best_lb:
                best, best_lb = Wn, lb
        if best is None:
            return
        if self.confirm:
            q = []
            for _ in range(self.trial_blocks):
                q += [(best, "trial_cand")] * self.trial_len + [(cur, "trial_cur")] * self.trial_len
            c["queue"] = q + c["queue"]
            c["trial"] = dict(cand=best, cur=cur, s={"trial_cand": [0.0, 0.0], "trial_cur": [0.0, 0.0]}, left=len(q))
            self.n_trial += 1
        else:
            self._adopt(c, b, cur, best)

    def _adopt(self, c, b, cur, new):
        c["cur"], c["follow"] = new, False
        c["cool"][cur] = c["n"] + 100
        self.n_switch += 1
        self.switch_log.append((self.rounds, b, cur, new))

    def _post_rescore(self, b, cls, W, tag, val, tok, layer_m, theta, K, com, t_round):
        c = self.cells[(Q.bkey(b), cls)]
        rec = dict(b=b, W=W, layer_m=list(layer_m), theta=theta, val=val, tok=tok,
                   resid=t_round - self._model(b, W, K, layer_m))
        c["recs"].append(rec)
        if self.keep_wide and W > c["cur"]:
            c["wrecs"].append(rec)
        c["n"] += 1
        tr = c["trial"]
        if tr is not None and tag in ("trial_cand", "trial_cur"):
            tr["s"][tag][0] += float(sum(com)); tr["s"][tag][1] += t_round
            tr["left"] -= 1
            if tr["left"] <= 0:
                sc, su = tr["s"]["trial_cand"], tr["s"]["trial_cur"]
                ok = sc[1] > 0 and su[1] > 0 and (sc[0] / sc[1]) > (su[0] / su[1]) * (1 + self.trial_margin)
                if ok and c["cur"] == tr["cur"]:
                    self.n_trial_ok += 1
                    self._adopt(c, b, tr["cur"], tr["cand"])
                else:
                    c["cool"][tr["cand"]] = c["n"] + self.cooldown
                c["trial"] = None
            return
        if c["trial"] is not None:
            return
        if c["decide_after_burst"]:
            c["burst_left"] -= 1
            if c["burst_left"] <= 0:
                c["decide_after_burst"] = False
                self._decide(c, b)
            return
        if c["n"] % self.every == 0:
            self._decide(c, b)

    def _cost_event(self):
        """Called when the measured cost table changed by > 10% (ideal event-triggered re-measure)."""
        self.n_events += 1
        Wmax = self.w_opts[-1]
        for c in self.cells.values():
            c["cool"] = {}
            if c["trial"] is not None:
                c["queue"] = [q for q in c["queue"] if q[1] not in ("trial_cand", "trial_cur")]
                c["trial"] = None
            if c["cur"] < Wmax and self.burst > 0:
                c["queue"] = [(self._wx(c["cur"]), "explore")] * self.burst + c["queue"]
                c["burst_left"] = self.burst
                c["decide_after_burst"] = True
            else:
                c["burst_left"] = 1
                c["decide_after_burst"] = True

    # ---------------- one round (v10 path of BatchAccount7, look=None) ----------------
    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        b = len(slots)
        h1 = sum(int(world.reqs[r]["H"][j] == 0) for r, j in slots)
        if self.t1_n is None:
            self.t1_hit, self.t1_n = float(h1), float(b)
        else:
            self.t1_hit = 0.95 * self.t1_hit + 0.05 * h1
            self.t1_n = 0.95 * self.t1_n + 0.05 * b
        if self.wmode in ("rescore", "cal") and getattr(self, "lt_probe_events", True):
            pv = self.lt(4, 4)
            if self._probe is not None and abs(pv / self._probe - 1) > 0.10:
                self._cost_event()
            self._probe = pv
        cls = self.cls()
        Wtab, Ktab = self.wk0(b, cls)
        theta = self._theta_b(b)
        warm = self.rounds < self.warm or theta is None
        W, tag = self._choose(world, b, cls, Wtab, warm)
        self.w_log[W] = self.w_log.get(W, 0) + 1
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
        cs = [self._c(r) for r in rids]
        e_now = [self._es(t) for t in trees]
        e_prev = [{K: 0.0 for K in Ks} for _ in range(n)]
        v_now = [self._vals(cs[i], e_now[i]) for i in range(n)]
        v_prev = [{K: 0.0 for K in Ks} for _ in range(n)]
        active, depth, time_ = list(range(n)), 1, 0.0
        layer_m, layer_dt = [], []
        while active:
            cand = [i for i in active if depth < world.DMAX - 1 and depth + 1 <= max(1, rem[i] - 1)]
            if not cand:
                break
            if warm:
                keep = [i for i in cand if e_now[i][Kmax] - e_prev[i][Kmax] > 0.4]
            else:
                Jk = {K: sum(v_now[i][K] for i in range(n)) - theta * self.Tv(b, K) for K in Ks}
                stop = max(Jk.values())
                g1 = {K: {i: v_now[i][K] - v_prev[i][K] for i in cand} for K in Ks}
                _, keep, _, _ = self._best(cand, set(), g1, g1, Jk, stop, theta, W, (1,))
            if not keep:
                break
            for i in keep:
                trees[i].expand(W, rng, params)
                e_prev[i], e_now[i] = e_now[i], self._es(trees[i])
                v_prev[i], v_now[i] = v_now[i], self._vals(cs[i], e_now[i])
            dt = world.real_cd(rng, len(keep), W)
            time_ += dt
            layer_m.append(len(keep))
            layer_dt.append(dt)
            active = keep
            depth += 1
        if warm:
            K = Ktab
        else:
            Jk = {K_: sum(v_now[i][K_] for i in range(n)) - theta * self.Tv(b, K_) for K_ in Ks}
            K = max(Jk, key=Jk.get)
        if getattr(self, "_kforce", None) is not None:
            K = self._kforce
        self.k_log[K] = self.k_log.get(K, 0) + 1
        acc = [P.last_node(t, t.topk(K, False))[0] for t in trees]
        com = [min(a + 1, rem[i]) for i, a in enumerate(acc)]
        t_round = time_ + world.real_rtt(rng, rtt) + world.real_T(rng, b, K)
        if hasattr(self, "_obs_costs"):
            self._obs_costs(b, W, K, layer_m, layer_dt, t_round - time_)
        if not warm and self.wmode == "cal":
            self._post_cal(b, cls, W, tag, trees, cs, rem, layer_m, theta, K, com, t_round)
        if not warm and self.wmode in ("replay14", "rescore"):
            val, tok = self._replay_tables(trees, cs, rem, W)
            if self.wmode == "replay14":
                self._post14(b, W, tag, val, tok, layer_m, theta, K, com, t_round)
            else:
                self._post_rescore(b, cls, W, tag, val, tok, layer_m, theta, K, com, t_round)
        self._after_round(b, K, t_round - time_)
        tok_ = float(sum(com))
        self.tok_total += tok_
        fx = t_round - time_
        self.fixed[b] = fx if b not in self.fixed else 0.9 * self.fixed[b] + 0.1 * fx
        a_ = self.alpha
        if self.g_ms is None:
            self.g_tok, self.g_ms = tok_, t_round
        else:
            self.g_tok = (1 - a_) * self.g_tok + a_ * tok_
            self.g_ms = (1 - a_) * self.g_ms + a_ * t_round
        v = self.tb.setdefault(b, [tok_, t_round, 0])
        v[2] += 1
        ab_ = max(a_, 1.0 / v[2])
        v[0] = (1 - ab_) * v[0] + ab_ * tok_
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
