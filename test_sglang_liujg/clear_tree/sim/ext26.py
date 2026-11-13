"""Round 32: closing the gaps found in round 31, on top of ext25.BatchAccount18 (all default off; with everything
off BatchAccount19 is BatchAccount18, i.e. the design as of D69).

  ebkt     (cap, period): instead of the cooldown (D67), cost events (layer / verify) draw from a token bucket: one
           token per event, `cap` tokens at most (starts full), one token back every `period` rounds. Real changes close
           together are handled at once; a sustained storm is limited to one event per `period` rounds. Use with
           cooldown=False.
  perk     verify regression with forgetting per K (D71): the statistics of each K are kept apart and decay only when
           that K gets a sample; solved on their sum. A common shift (D68) shifts each K's statistics.
  cpk      (period, rounds): continuous timing probes for K. A K that has not run (chosen or forced) for `period`
           rounds of the b bucket is forced for `rounds` rounds (W as the controller says; these are timing-probe
           rounds: capped, excluded from the controller). If the residuals of those rounds (measured / predicted - 1)
           all have the same sign and all exceed the drift tolerance, a K check (D68) starts.
  phi      boundary calibration of the next-layer gain: ghat is multiplied by phi = EMA(actual) / EMA(predicted) of the
           b bucket (EMA step 0.02, used after 50 rounds, clipped to [phi_lo, phi_hi]). Per round and request with at
           least one grown layer: predicted = the tight ghat of its last grown layer, recomputed from the tree before
           that layer, at the K finally chosen; actual = accepted tokens of the tree's top K minus those of the top K
           of the tree without that layer (committed text). Normal rounds only.
  cap_free the full timing probes of a b bucket's first appearance (every (W, K), mandatory, once) do not draw from
           the probe cap and are not counted in it; the cap max(pcap_base, pcap x rounds) then only limits re-probes
           after events, K checks and continuous probes (with bucket=True cap_free is ignored).
  rob_cont (with rob) a layer sample is withheld from its regression only if its cell (tier, W) was also observed
           in the previous round. Cells observed now and then (continuous probes of an unused W, exploration rounds,
           rare tiers) are never screened: for them a large residual is as likely a stale model as a spike.
  rob_stale > 0 (with rob): a layer sample is never withheld if its cell (tier, W) had no sample in the last rob_stale
           rounds (an unused W between continuous probes): its model is stale, so a large residual is taken as a real
           change and the following rounds of the cell are accepted too. Cells seen more often (the W in use,
           exploration rounds, tiers after exits) are screened as before.
  rob_hys  (with rob) the outlier streak of a cell / bucket ends only when a residual falls below rob_clip / 2; a
           residual between rob_clip / 2 and rob_clip keeps it. Without it a shift close to rob_clip (RTT 4 -> 40 ms on
           a 65 ms wait) flaps around the threshold and samples keep being withheld.
  rob_scope (with rob) the 3-round gate applies only to rounds the controller chose for the width it is using.
           A probe that forces W (the queued timing probe, or a continuous probe) always enters the layer regression
           and does not read or write the layer streak. A probe that forces only K still gates layer samples. The 5%
           wider exploration round stays on the gate and does not change the layer streak, so one exploration round
           cannot complete the 3. Switching widths clears the layer streak of the old and the new width.
  arm      > 0: the drift detector of a cell is fed only while its regression (the layer regression of that W / the
           verify regression) holds at least `arm` samples since it was last cleared.

Worlds: add_lalt (layer cost alternating), add_k32 (K = 32 carries an extra verify cost that disappears).
"""
import numpy as np
import ext9 as Q
import ext12 as RP
import ext25 as N


class PerK:
    """Least squares with exponential forgetting per K: the first three features are the one-hot K."""

    def __init__(self, d=6, lam=0.999):
        self.d, self.lam = d, lam
        self.clear()

    def clear(self):
        self.Ak = [np.zeros((self.d, self.d)) for _ in range(3)]
        self.yk = [np.zeros(self.d) for _ in range(3)]
        self.nk, self.n, self._w = [0, 0, 0], 0, None

    @property
    def A(self):
        return self.Ak[0] + self.Ak[1] + self.Ak[2]

    @property
    def y(self):
        return self.yk[0] + self.yk[1] + self.yk[2]

    def add(self, x, t):
        x = np.asarray(x, float)
        k = int(np.argmax(x[:3]))
        self.Ak[k] = self.lam * self.Ak[k] + np.outer(x, x)
        self.yk[k] = self.lam * self.yk[k] + x * t
        self.nk[k] += 1
        self.n += 1
        self._w = None

    def coef(self):
        if self._w is None:
            A = self.A
            scale = np.maximum(np.diag(A), 1e-9)
            self._w = np.linalg.solve(A + 1e-6 * np.diag(scale) + 1e-12 * np.eye(self.d), self.y)
        return self._w

    def pred(self, x):
        return float(np.dot(self.coef(), x))

    def shift(self, d):
        for k in range(3):
            self.yk[k] = self.yk[k] + self.Ak[k] @ d
        self._w = None

    def mark(self, x):
        k = int(np.argmax(np.asarray(x, float)[:3]))
        return (k, self.nk[k])

    def remove(self, x, t, tag):
        k, n0 = tag
        w = self.lam ** (self.nk[k] - n0)
        self.Ak[k] = self.Ak[k] - w * np.outer(x, x)
        self.yk[k] = self.yk[k] - w * x * t
        self._w = None


class BatchAccount19(N.BatchAccount18):
    def __init__(self, ebkt=None, perk=False, cpk=None, phi=False, phi_lo=0.7, phi_hi=1.5, cap_free=False, arm=0, rob_cont=False, rob_hys=False, rob_stale=0, rob_scope=False, **kw):
        super().__init__(**kw)
        self.ebkt, self.perk, self.cpk = ebkt, perk, cpk
        self.phi, self.phi_lo, self.phi_hi, self.phi_st, self._cur_bk = phi, phi_lo, phi_hi, {}, None
        self.cap_free, self.n_free = cap_free, 0
        self.arm = arm
        self.rob_cont, self.seen = rob_cont, {}
        self.rob_hys, self.rob_stale = rob_hys, rob_stale
        self.rob_scope = rob_scope
        self._tag, self._force_w, self._force_k = "normal", False, False
        self.admit, self.why = [], []
        self.ev_tok, self.ev_tok_at = float(ebkt[0]) if ebkt else 0.0, 0
        if perk:
            self.rv = PerK(6)
        self.k_last, self.cpk_now, self.n_cpk, self.n_cpk_trig = {}, None, 0, 0

    # ---------------- cost events from a token bucket ----------------
    def _gate(self):
        if self._force or not self.ebkt:
            return super()._gate()
        cap, per = self.ebkt
        self.ev_tok = min(float(cap), self.ev_tok + (self.rounds - self.ev_tok_at) / float(per))
        self.ev_tok_at = self.rounds
        if self.ev_tok >= 1.0:
            self.ev_tok -= 1.0
            self.last_ev = self.rounds
            return True
        self.n_suppressed += 1
        for f in self.fast.values():
            f[1] = 0
        return False

    # ---------------- per-K verify regression ----------------
    @staticmethod
    def _clear(r):
        if isinstance(r, PerK):
            r.clear()
        else:
            r.A *= 0.0; r.y *= 0.0; r.n = 0; r._w = None
        r._ps = []

    def _radd(self, reg, x, t):
        note = (self.rob_scope and self.rob and reg is self.rl.get(getattr(self, "_W_now", None))
                and (self._force_w or self._tag == "explore"))
        before = float(np.dot(reg.coef(), np.asarray(x, float))) if note else None
        tag = reg.mark(x) if isinstance(reg, PerK) else None
        reg.add(x, t)
        if note:
            after = float(np.dot(reg.coef(), np.asarray(x, float)))
            self._note(self._W_now, x[1], t, before, after, False, None)
        if self.rob and self._probe_now:
            if not hasattr(reg, "_ps"):
                reg._ps = []
            if tag is not None:
                tag = (tag[0], tag[1] + 1)
            reg._ps.append((np.asarray(x, float), float(t), reg.n if tag is None else tag))

    def _rescreen(self):
        for reg in list(self.rl.values()) + [self.rv]:
            ps = getattr(reg, "_ps", [])
            reg._ps = []
            if reg.n < 6 or not ps:
                continue
            for _ in range(2):
                coef, keep = reg.coef().copy(), []
                for x, t, tag in ps:
                    p = float(np.dot(coef, x))
                    if p > 0 and abs(t / p - 1) > self.rob_clip:
                        if isinstance(reg, PerK):
                            reg.remove(x, t, tag)
                        else:
                            w = reg.lam ** (reg.n - tag)
                            reg.A = reg.A - w * np.outer(x, x)
                            reg.y = reg.y - w * x * t
                            reg._w = None
                        self.n_rescreen += 1
                    else:
                        keep.append((x, t, tag))
                ps = keep

    def _end_check(self):
        if not isinstance(self.rv, PerK) or self.boff:
            return super()._end_check()
        # ext25._end_check with the common shift applied per K
        c, self.check = self.check, None
        res = c["res"]
        if self.rob and c["vp"]:
            lim = self.rob_clip * float(np.mean(c["vp"]))
            res = {K: [x for x in v if abs(x - float(np.median(v))) <= lim] for K, v in res.items()}
        m = {K: float(np.mean(v)) for K, v in res.items() if v}
        if not m:
            return
        mu = float(np.mean(list(m.values())))
        spread = max(m.values()) - min(m.values())
        tol = max(self.chk_tol, self.chk_rel * abs(mu))
        if self.chk_n:
            dev = [x - float(np.mean(v)) for v in res.values() if len(v) >= 2 for x in v]
            dof = sum(len(v) - 1 for v in res.values() if len(v) >= 2)
            if dof > 0:
                tol = max(tol, 3.3 * float(np.sqrt(sum(d * d for d in dev) / dof)) / np.sqrt(self.chk_n))
        common = bool(len(m) == len(self.k_options) and spread <= tol)
        if len(self.chk_log) < 40:
            self.chk_log.append((self.rounds, self.world.n_T, round(mu, 1), round(spread, 1), bool(common),
                                 round(float(tol), 1)))
        if common:
            d = np.zeros(6); d[:3] = mu
            self.rv.shift(d)
            self.n_offsets += 1
            return
        self._kind = "verify"
        if not self._gate():
            return
        self._verify_event()

    # ---------------- rob: layer samples are screened only in a continuous stream ----------------
    def _streak(self, key, dev):
        if self.rob_scope and self.rob:
            layer = not (isinstance(key, tuple) and key and key[0] == "v")
            if (layer and (self._force_w or self._tag == "explore")) or ((not layer) and self._force_k):
                return self.streak.get(key, 0)
        return super()._streak(key, dev)

    def _clear_layer_streak(self, W):
        for key in list(self.streak):
            if isinstance(key, tuple) and len(key) == 2 and key[0] != "v" and key[1] == W:
                self.streak[key] = 0

    def _note(self, W, tier, measured, before, after, held, dev):
        self.admit.append(dict(at=self.rounds, tag=self._tag, W=int(W), tier=None if tier is None else int(tier),
                               ms=None if measured is None else round(float(measured), 3),
                               before=None if before is None else round(float(before), 3),
                               after=None if after is None else round(float(after), 3),
                               held=bool(held), dev=None if dev is None else round(float(dev), 3)))
        if len(self.admit) > 40:
            del self.admit[0]

    def _post14m(self, b, W, tag, rep, com, t_round, m_actual):
        n0 = self.n_switch
        super()._post14m(b, W, tag, rep, com, t_round, m_actual)
        if not self.rob_scope or self.n_switch == n0 or not self.switch_log:
            return
        _, _, old, new = self.switch_log[-1]
        gain = self.sw_se[-1][4] if self.sw_se else None
        if len(self.why) < 12:
            self.why.append(dict(at=self.switch_log[-1][0], b=b, old=old, new=new, gain=gain,
                                 lt={str(w): round(float(rep[w]["lt"]), 3) for w in rep if "lt" in rep[w]},
                                 admit=list(self.admit)))
        self.admit = []
        self._clear_layer_streak(old)
        self._clear_layer_streak(new)

    def _skip(self, cnt, ckey, skey, dev):
        if self.rob_scope and self.rob:
            if cnt is self.cell_nl and self._force_w:
                return False
            if cnt is self.cell_nv and self._force_k:
                return False
        if self.rob_cont and ckey[0] != "v" and not isinstance(ckey[0], str) and cnt is self.cell_nl:
            s = self.seen.get(ckey)
            if s is None or s[1] != self.rounds:
                s = self.seen[ckey] = (s[1] if s else None, self.rounds)
            if s[0] != self.rounds - 1:
                return False
        if self.rob_stale and cnt is self.cell_nl:
            s = self.seen.get(ckey)
            if s is None or s[1] != self.rounds:
                s = self.seen[ckey] = (s[1] if s else None, self.rounds)
            if s[0] is None or self.rounds - s[0] > self.rob_stale:
                if abs(dev) > self.rob_clip:
                    self.streak[skey] = max(self.streak.get(skey, 0), 3)    # stale model: take the change as real
                return False
        skip = super()._skip(cnt, ckey, skey, dev)
        if (skip and self.rob_scope and self.rob and cnt is self.cell_nl and self._tag == "explore"
                and abs(dev) > self.rob_clip):
            self._note(ckey[1], ckey[0], None, None, None, True, dev)
        return skip

    # ---------------- drift detector armed only on a regression with enough samples ----------------
    def _drift(self, key, dev, trig, kind):
        if self.arm and (self.rv.n if key[0] == "v" else self.rl[key[1]].n) < self.arm:
            return
        super()._drift(key, dev, trig, kind)

    # ---------------- probe cap: first-appearance probes exempt ----------------
    def _cap_ok(self):
        if not self.cap_free:
            return super()._cap_ok()
        q = self.probe_q.get(self._cur_bk)
        if q and q[0][0] is not None and q[0][1] is not None:
            self.n_free += 1
            return True
        return self.n_probe_rounds - self.n_free < max(self.pcap_base, self.pcap * self.rounds)

    # ---------------- boundary calibration of the next-layer gain ----------------
    def _phi(self):
        st = self.phi_st.get(self._cur_bk)
        if st is None or st[2] < 50 or st[1] <= 1e-9:
            return 1.0
        return min(self.phi_hi, max(self.phi_lo, st[0] / st[1]))

    def _g(self, t, c, K, vn, vp):
        g = super()._g(t, c, K, vn, vp)
        return g * self._phi() if self.phi else g

    def _replay_m(self, trees, cs, rem, layers, W, b, theta, suml_all):
        out = super()._replay_m(trees, cs, rem, layers, W, b, theta, suml_all)
        if self.phi:
            val = lambda K: sum(cs[i] * sum(sorted((t.score[n] for n in t.ids), reverse=True)[:K])
                                for i, t in enumerate(trees)) - theta * self._vt(b, K, suml_all)
            Ks = max(self.k_options, key=val)
            pred = act = 0.0
            for i, t in enumerate(trees):
                D = max(t.dep[n] for n in t.ids)
                if D < 2:
                    continue
                prev = [n for n in t.ids if t.dep[n] < D]
                sc = sorted((t.score[n] for n in prev), reverse=True)
                th = sc[Ks - 1] if len(sc) >= Ks else 0.0
                pred += cs[i] * sum(max(0.0, t.score[n] - th) for n in prev if t.dep[n] == D - 1)
                a1 = min(RP.accepted_in(t, RP.topk_of(t, list(t.ids), Ks)) + 1, rem[i])
                a0 = min(RP.accepted_in(t, RP.topk_of(t, prev, Ks)) + 1, rem[i])
                act += a1 - a0
            st = self.phi_st.setdefault(Q.bkey(b), [act, pred, 0])
            st[0] = 0.98 * st[0] + 0.02 * act
            st[1] = 0.98 * st[1] + 0.02 * pred
            st[2] += 1
        return out

    # ---------------- continuous timing probes for K ----------------
    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        self._cur_bk = Q.bkey(len(slots))
        if self.cpk and self.cost_mode != "known" and self.rounds >= self.warm:
            bk = Q.bkey(len(slots))
            n = self.bk_cnt.get(bk, 0)
            if bk in self.probed and not self.probe_q.get(bk) and self.check is None and self.cpk_now is None:
                for K in self.k_options:
                    last = self.k_last.setdefault((bk, K), n)
                    if n - last >= self.cpk[0]:
                        self.probe_q[bk] = [(None, K)] * self.cpk[1]
                        self.k_last[(bk, K)] = n
                        self.cpk_now = dict(bk=bk, K=K, res=[], at=self.rounds)
                        self.n_cpk += 1
                        break
        return super().round(world, slots, rng, rtt, rnd, trees_out)

    def _observe(self, b, W, K, layers, wait, suml_all):
        if self.cost_mode == "known":
            return
        bk = Q.bkey(b)
        cp = self.cpk_now
        if self.cpk:
            self.k_last[(bk, K)] = self.bk_cnt.get(bk, 0)
            if cp is not None and self._probe_now and bk == cp["bk"] and K == cp["K"]:
                core = wait - self._lt(b, W, suml_all) - (self.voff.get(bk, 0.0) if self.boff else 0.0)
                cp["res"].append(core / max(self._vt(b, K, suml_all), 1e-6) - 1)
        super()._observe(b, W, K, layers, wait, suml_all)
        if cp is not None and self.cpk_now is cp:
            done = len(cp["res"]) >= self.cpk[1]
            if done or self.rounds - cp["at"] > 40:
                self.cpk_now = None
                if not done:
                    self.probe_q[cp["bk"]] = [x for x in self.probe_q.get(cp["bk"], []) if x[0] is not None]
                r = cp["res"]
                if done and min(abs(x) for x in r) > self.drift_tol and (all(x > 0 for x in r) or all(x < 0 for x in r)):
                    if self.check is None and not self.probe_q.get(cp["bk"]):
                        self.n_cpk_trig += 1
                        self._start_check(cp["bk"])


# ---------------- worlds ----------------
def add_lalt(w, mult=3.0, at=400, period=100):
    """Layer cost terms alternate x mult / x 1 every `period` speculative rounds from round `at` (high first)."""
    orig = w.real_T_l
    st = {"hi": False}

    def real_T_l(rng, b, K, suml):
        out = orig(rng, b, K, suml)
        hi = w.n_T >= at and ((w.n_T - at) // period) % 2 == 0
        if hi != st["hi"]:
            m = mult if hi else 1.0 / mult
            w.d0, w.d_req, w.d_tok = w.d0 * m, w.d_req * m, w.d_tok * m
            st["hi"] = hi
        return out
    w.real_T_l = real_T_l


def add_k32(w, at=400):
    """K = 32 pays an extra 15 + 2 b ms of verify time until speculative round `at` (round 17: an unused K gets
    cheaper)."""
    orig = w.T_core
    w.T_core = lambda b, K, suml: orig(b, K, suml) + ((15.0 + 2.0 * b) if K >= 32 and w.n_T < at else 0.0)
