"""Round 30: three changes to the v11 controller with online costs, on top of D67 (ext23.BatchAccount16 with G1).

BatchAccount17 (all default off; the baseline is v11 + D65 + D67 cooldown):
  src      cost events by source (README 3.8 / section 7 candidate):
           - verify residual trigger -> a K check: each K forced for 2 rounds of the bucket (W as the controller says);
             residual r_K = mean(core - prediction) in ms. If max - min of r_K <= max(chk_tol, chk_rel x |mean r|), the
             shift is common to all K: add mean r to the three K intercepts of the verify regression (exactly, by
             shifting the RLS normal equations), no cost event, controller statistics kept. Otherwise a verify event:
             clear only the verify regression, probe only K (2 rounds per K, W as the controller says), clear the
             controller statistics (D65).
           - layer residual trigger -> a layer event: clear only the layer regressions, probe only W (2 rounds per W,
             K chosen as usual), clear the controller statistics.
           K checks are not gated by the cooldown (they are cheap and a common offset changes no decision); events are
           (G1). Check rounds count as timing-probe rounds (probe cap, controller excluded).
  src_pre  (with src) residuals of the K check are taken against the verify coefficients snapshotted when a verify
           drift counter first rose (before the regression absorbed part of the shift); a common shift sets the
           regression to snapshot + mean residual on the three K intercepts (round 30b).
  chk_z    (with src) > 0: the K-check tolerance also covers the residual noise: max(chk_tol, chk_rel x |mean r|,
           chk_z x sigma_b), sigma_b = sqrt of an EMA (0.02) of the squared verify residual of the b bucket over in-control
           rounds (monitored, no drift counter up, no check) (round 30b).
  chk_n    (with src) > 0: the K check runs chk_n rounds per K, and the tolerance also covers the check's own noise:
           max(chk_tol, chk_rel x |mean r|, 3.3 x s_pooled / sqrt(chk_n)), s_pooled = pooled within-K sd of the
           check residuals (3.3 ~ 95% point of the range of three normal means) (round 30b).
  recheck  after a switch to a narrower width W': keep the controller statistics of the switch and a snapshot of the
           layer regression coefficients of W'. After rc_rounds normal rounds at W', f = sum measured layer time /
           sum predicted with the snapshot; candidate time of W' corrected by (f - 1) x its accumulated model layer
           time; if the corrected gain < rc_back, switch back and restore the corrected statistics. Dropped if a cost
           event happens meanwhile.
  tight    next-layer gain ghat_i^K = c_i x sum over the newest layer of max(0, s_j - theta_K), theta_K = K-th largest
           path score in the tree (0 with fewer than K nodes), instead of the last increment c_i x dE^K.
"""
import copy
import numpy as np
import ext9 as Q
import ext10 as P  # noqa: F401
import ext12 as RP
import ext20 as V
import ext23 as G
import feasibility as F


class BatchAccount17(G.BatchAccount16):
    def __init__(self, src=False, chk_tol=4.0, chk_rel=0.3, recheck=False, rc_rounds=20, rc_back=0.01, tight=False,
                 src_pre=False, chk_z=0.0, chk_n=0, **kw):
        kw.setdefault("cooldown", True)
        kw.setdefault("pcap", 0.05)
        super().__init__(**kw)
        assert self.timing == "resid"
        self.src, self.chk_tol, self.chk_rel = src, chk_tol, chk_rel
        self.src_pre, self.vpre = src_pre, None
        self.chk_z, self.vvar = chk_z, {}
        self.chk_n = chk_n
        self.recheck, self.rc_rounds, self.rc_back, self.tight = recheck, rc_rounds, rc_back, tight
        self.check = None
        self.n_checks = self.n_offsets = self.n_vevents = self.n_levents = 0
        self.chk_log = []
        self.n_rc = self.n_back = 0
        self.rc_log = []

    # ---------------- tight next-layer gain ----------------
    def _g(self, t, c, K, vn, vp):
        if not self.tight:
            return vn[K] - vp[K]
        sc = t.scores
        th = float(np.sort(sc)[-K]) if len(sc) >= K else 0.0
        return c * sum(max(0.0, t.score[v] - th) for v in t.frontier)

    # ---------------- cost events by source ----------------
    def _gate(self):
        if self.cooldown and self.last_ev is not None and self.rounds < self.last_ev + self.cd_len:
            self.n_suppressed += 1
            for f in self.fast.values():
                f[1] = 0
            return False
        if self.cooldown and self.last_ev is not None:
            self.cd_len = min(2 * self.cd_len, self.cd_max) if self.rounds < self.last_ev + 2 * self.cd_len else self.cd0
        self.last_ev = self.rounds
        return True

    def _reset_stats(self):
        if self.reset_on_event:
            for st in self.ctx14.values():
                st["stats"] = {}
                st["rc"] = None
            self.n_resets += 1

    @staticmethod
    def _clear(r):
        r.A *= 0.0; r.y *= 0.0; r.n = 0; r._w = None

    def _seen(self):
        return [bk for bk in self.bk_cnt if bk in self.probed]

    def _observe(self, b, W, K, layers, wait, suml_all):
        if not self.src:
            return super()._observe(b, W, K, layers, wait, suml_all)
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
        chk = self.check is not None and self.check["bk"] == bk and self._probe_now
        if chk:
            pre = self.check.get("pre")
            ref = float(np.dot(pre, self._vx(b, K, suml_all))) if pre is not None else vpred
            self.check["res"][K].append(core - ref)
        mon = (self.vdrift and self.rv.n >= 6 and bk in self.probed and not self.probe_q.get(bk)
               and self.check is None)
        if self.src_pre and mon:
            self._vcoef_before = self.rv.coef().copy()
        self.rv.add(self._vx(b, K, suml_all), core)
        if mon and self.chk_z and not any(f[1] for k_, f in self.fast.items() if k_[0] == "v"):
            r_ = core - vpred
            v_ = self.vvar.get(bk)
            self.vvar[bk] = r_ * r_ if v_ is None else 0.98 * v_ + 0.02 * r_ * r_
        if mon:
            key = ("v", b, K)
            before = self.fast.get(key, [0.0, 0])[1]
            self._drift(key, core / max(vpred, 1e-6) - 1, trig, "verify")
            if self.src_pre:
                if before == 0 and self.fast[key][1] == 1 and self.vpre is None and self.rv.n >= 6:
                    # rv already holds this round's sample; the snapshot is taken from the coefficients used for vpred
                    self.vpre = self._vcoef_before
                if self.check is None and not any(f[1] for k_, f in self.fast.items() if k_[0] == "v"):
                    self.vpre = None
        if chk and not self.probe_q.get(bk):
            self._end_check()
        if "layer" in trig:
            self._layer_event()
        elif "verify" in trig:
            self._start_check(bk)

    def _start_check(self, bk):
        if self.probe_q.get(bk):
            return
        self.n_checks += 1
        self.check = dict(bk=bk, res={K: [] for K in self.k_options}, at=self.rounds,
                          pre=(self.vpre if self.src_pre else None))
        self.vpre = None
        self.probe_q[bk] = [(None, K) for K in self.k_options] * (self.chk_n or self.probe_passes)
        for key in [k for k in self.fast if k[0] == "v"]:
            del self.fast[key]

    def _end_check(self):
        c, self.check = self.check, None
        m = {K: float(np.mean(v)) for K, v in c["res"].items() if v}
        if not m:
            return
        mu = float(np.mean(list(m.values())))
        spread = max(m.values()) - min(m.values())
        tol = max(self.chk_tol, self.chk_rel * abs(mu))
        if self.chk_z and c["bk"] in self.vvar:
            tol = max(tol, self.chk_z * float(np.sqrt(self.vvar[c["bk"]])))
        if self.chk_n:
            dev = [x - float(np.mean(v)) for v in c["res"].values() if len(v) >= 2 for x in v]
            dof = sum(len(v) - 1 for v in c["res"].values() if len(v) >= 2)
            if dof > 0:
                tol = max(tol, 3.3 * float(np.sqrt(sum(d * d for d in dev) / dof)) / np.sqrt(self.chk_n))
        common = bool(len(m) == len(self.k_options) and spread <= tol)
        if len(self.chk_log) < 40:
            self.chk_log.append((self.rounds, self.world.n_T, round(mu, 1), round(spread, 1), bool(common), round(float(tol), 1)))
        if common:
            d = np.zeros(len(self.rv.y)); d[:3] = mu
            if c.get("pre") is not None:
                self.rv.y = self.rv.A @ (c["pre"] + d)      # back to the snapshot, shifted
            else:
                self.rv.y = self.rv.y + self.rv.A @ d
            self.rv._w = None
            self.n_offsets += 1
            return
        if not self._gate():
            return
        self.n_vevents += 1
        self.n_cost_events += 1
        if len(self.ev_log) < 40:
            self.ev_log.append((self.rounds, self.world.n_T, ["verify"]))
        self._clear(self.rv)
        for key in [k for k in self.fast if k[0] == "v"]:
            del self.fast[key]
        for bk in self._seen():
            if not self.probe_q.get(bk):
                self.probe_q[bk] = [(None, K) for K in self.k_options] * self.probe_passes
        self._reset_stats()

    def _layer_event(self):
        if not self._gate():
            return
        self.n_levents += 1
        self.n_cost_events += 1
        if len(self.ev_log) < 40:
            self.ev_log.append((self.rounds, self.world.n_T, ["layer"]))
        for r in self.rl.values():
            self._clear(r)
        self.ref = {}
        for key in [k for k in self.fast if k[0] != "v"]:
            del self.fast[key]
        self.check = None
        for bk in self._seen():
            self.probe_q[bk] = [(w_, None) for w_ in self.w_opts] * self.probe_passes
        self._reset_stats()

    # ---------------- replay with the candidate's model layer time ----------------
    def _replay_m(self, trees, cs, rem, layers, W, b, theta, suml_all):
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
            out[Wn] = dict(tok=float(sum(tok)), mt=lt + self._vt(b, K, suml_all), lt=lt)
        return out

    # ---------------- controller with the narrow-switch recheck ----------------
    def _post14m(self, b, W, tag, rep, com, t_round, m_actual):
        s = self.ctx14[Q.bkey(b)]
        cur = s["cur"]
        if cur not in rep:
            return
        tm = lambda x: t_round - m_actual + rep[x]["mt"]
        rc = s.get("rc")
        if rc is not None and rc["ev"] != self.n_cost_events:
            rc = s["rc"] = None
        if rc is not None and tag == "normal" and W == cur:
            for m, sl, dt in self._layers_now:
                rc["meas"] += dt
                rc["pred"] += max(0.1, float(np.dot(rc["coef"], [1.0, F.bucket(m), sl / 1000.0])))
            rc["n"] += 1
            if rc["n"] >= self.rc_rounds:
                s["rc"] = None
                if rc["pred"] > 0:
                    f = rc["meas"] / rc["pred"]
                    st = rc["st"]
                    tc = st[1] + (f - 1.0) * st[5]
                    g2 = (st[0] / tc) / (st[2] / st[3]) - 1
                    self.n_rc += 1
                    back = g2 < self.rc_back
                    if len(self.rc_log) < 30:
                        self.rc_log.append((self.rounds, b, rc["old"], cur, round(f, 3), round(rc["g"], 4), round(g2, 4), back))
                    if back:
                        old = rc["old_stats"]
                        old[cur] = [st[0], tc, st[2], st[3], st[4], st[5]]
                        s["cur"], s["stats"] = rc["old"], old
                        self.n_back += 1
                        self.n_switch += 1
                        if len(self.switch_log) < 30:
                            self.switch_log.append((self.rounds, b, cur, rc["old"]))
                        return
        if tag == "explore":
            ctok, ctm = rep[cur]["tok"], tm(cur)
        else:
            ctok, ctm = float(sum(com)), t_round
        for o in rep:
            if o == cur:
                continue
            st = s["stats"].setdefault(o, [0.0, 0.0, 0.0, 0.0, 0, 0.0])
            st[0] += rep[o]["tok"]; st[1] += tm(o); st[2] += ctok; st[3] += ctm; st[4] += 1; st[5] += rep[o]["lt"]
        s["n"] += 1
        if s["n"] % 20 == 0:
            best, bg = None, 0.03
            for o, st in s["stats"].items():
                need = 30 if o < cur else 10
                if st[4] < need:
                    continue
                g = (st[0] / st[1]) / (st[2] / st[3]) - 1
                if g > bg:
                    best, bg = o, g
            if best is not None:
                s["rc"] = None
                if (self.recheck and best < cur and self.cost_mode != "known" and self.rl[best].n >= 4):
                    s["rc"] = dict(old=cur, old_stats=copy.deepcopy(s["stats"]), st=list(s["stats"][best]),
                                   coef=self.rl[best].coef().copy(), meas=0.0, pred=0.0, n=0, g=bg,
                                   ev=self.n_cost_events)
                s["cur"], s["follow"], s["stats"] = best, False, {}
                self.n_switch += 1
                if len(self.switch_log) < 30:
                    self.switch_log.append((self.rounds, b, cur, best))


# ---------------- worlds ----------------
def add_vchg(w, mult, at=400):
    """Verify cost per token (v_tok, K-dependent) x mult at the at-th speculative round."""
    orig = w.real_T_l

    def real_T_l(rng, b, K, suml):
        if w.n_T == at:
            w.v_tok *= mult
        return orig(rng, b, K, suml)
    w.real_T_l = real_T_l


def add_step(w):
    """Verify time steps up for K >= 16 by 12 + 1.5 b ms (round 16/17 'step' worlds)."""
    orig = w.T_core
    w.T_core = lambda b, K, suml: orig(b, K, suml) + ((12.0 + 1.5 * b) if K >= 16 else 0.0)
