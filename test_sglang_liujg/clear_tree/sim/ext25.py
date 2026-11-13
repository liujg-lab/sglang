"""Round 31: checks of the v11 design after D66-D69 (README review of 2026-11-12), on top of the current design
(ext24.BatchAccount17 with src, chk_n = 4 and tight: v11 + D65 + D67 + D68 + D69).

BatchAccount18 (all default off; with everything off it reproduces BatchAccount17(src=True, chk_n=4, tight=True)
bit for bit):
  pend     a layer / verify event suppressed by the cooldown is not dropped: it is kept as pending and fired once
           when the cooldown ends (events of both kinds pending at that moment fire together).
  psrc     the cooldown (length, doubling) is kept per source: a verify event does not block a layer event and
           vice versa.
  bucket   timing-probe cap as a token bucket: pcap tokens per round, capacity pcap_base (starts full), one token
           per probe round; instead of the cumulative cap  probe rounds < max(pcap_base, pcap x rounds).
  se_z     > 0: a width switch also needs gain > se_z x SE(gain). SE by the delta method from the per-round paired
           (tokens, time) of candidate and current: psi_i = a_i/mean(a) - t_i/mean(t) - c_i/mean(c) + u_i/mean(u),
           SE = (1 + gain) x sd(psi) / sqrt(n). The gain and SE of every switch are logged in any case (sw_se).
  boff     a K check that finds a common shift adds it to an offset of the b bucket (voff[bucket]); the verify
           regression is fitted on wait - offset and its intercepts are not shifted. The offset enters only the
           drift detector and the K-check residuals (a K-independent term changes no decision).
  chk_int  > 0: after a K check of a b bucket, no new K check of that bucket for chk_int rounds of the bucket; a
           check within 2 x the interval of the previous one doubles the interval (max chk_int_max), otherwise it
           returns to chk_int. A trigger inside the interval is kept and the check runs when the interval ends.
  chk_to   > 0: a K check not finished chk_to rounds after it started is abandoned (b left the bucket, or the probe
           cap held its rounds back): its remaining forced rounds are dropped and no verdict is made. A check pending
           in another b bucket no longer switches off the verify drift detector; a trigger there abandons it.
  rob      robustness to timing spikes:
           - layer drift detector fed once per round and cell with the median residual of the cell's forwards,
             verify detector as before; both inputs clipped to +- rob_clip; drift_run counts rounds;
           - a sample with |measured / predicted - 1| > rob_clip does not enter its regression while its cell
             (tier, W) / (b bucket, K) has >= 4 samples and the outlier streak of the cell (layer) / bucket (verify),
             in rounds, is < 3;
           - samples added in timing-probe rounds are screened once when the probe queue of the bucket empties:
             those with |measured / fitted - 1| > rob_clip are removed from the regression (two passes);
           - K check: a residual further than rob_clip x mean prediction from the median of its K is dropped.

Worlds: add_changes (layer cost terms multiplied at several rounds), add_bchg (verify cost per request c_b + d at a
round: a K-independent shift proportional to b), add_tail (stalled rounds and single-forward spikes).
"""
import numpy as np
import ext9 as Q
import ext24 as H
import feasibility as F


class BatchAccount18(H.BatchAccount17):
    def __init__(self, pend=False, psrc=False, bucket=False, se_z=0.0, boff=False, chk_int=0, chk_int_max=1600,
                 rob=False, rob_clip=0.5, chk_to=0, **kw):
        kw.setdefault("src", True)
        kw.setdefault("chk_n", 4)
        kw.setdefault("tight", True)
        super().__init__(**kw)
        assert self.src and not self.src_pre and not self.chk_z and not self.recheck
        self.pend, self.psrc, self.bucket, self.se_z = pend, psrc, bucket, se_z
        self.boff, self.chk_int, self.chk_int_max, self.rob, self.rob_clip = boff, chk_int, chk_int_max, rob, rob_clip
        self._kind, self._force = "layer", False
        self.ev_at, self.cd, self.pending, self.n_pend = {}, {}, {}, 0
        self.tok_b, self.tok_at = float(self.pcap_base), 0
        self.sw_se, self.n_se_block = [], 0
        self.voff, self.chk_at, self.chk_iv, self.chk_pending, self.n_chk_supp = {}, {}, {}, set(), 0
        self.cell_nl, self.cell_nv, self.streak = {}, {}, {}
        self.n_skip = self.n_rescreen = 0
        self.chk_to, self.n_chk_abort = chk_to, 0

    # ---------------- cooldown: pending events, per source ----------------
    def _gate(self):
        if self._force:
            return True
        if not (self.pend or self.psrc):
            return super()._gate()
        k = self._kind if self.psrc else "all"
        last, cd = self.ev_at.get(k), self.cd.get(k, self.cd0)
        if self.cooldown and last is not None and self.rounds < last + cd:
            self.n_suppressed += 1
            for f in self.fast.values():
                f[1] = 0
            if self.pend:
                self.pending[self._kind] = True
            return False
        if self.cooldown and last is not None:
            self.cd[k] = min(2 * cd, self.cd_max) if self.rounds < last + 2 * cd else self.cd0
        self.ev_at[k] = self.rounds
        self.last_ev = self.rounds
        self.pending[self._kind] = False
        return True

    def _fire_pending(self):
        due = []
        for kind in ("layer", "verify"):
            if self.pending.get(kind):
                k = kind if self.psrc else "all"
                if self.rounds >= self.ev_at[k] + self.cd.get(k, self.cd0):
                    due.append(kind)
        for i, kind in enumerate(due):
            self.pending[kind] = False
            self.n_pend += 1
            self._force = i > 0 and not self.psrc      # both kinds due under one cooldown: fire together
            if kind == "layer":
                self._layer_event()
            else:
                self._kind = "verify"
                if self._gate():
                    self._verify_event()
            self._force = False

    def _layer_event(self):
        self._kind = "layer"
        n = self.n_levents
        super()._layer_event()
        if self.n_levents > n:
            self.cell_nl, self.streak = {}, {k: v for k, v in self.streak.items() if k[0] == "v"}

    def _verify_event(self):
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
        self.voff, self.cell_nv = {}, {}
        self.streak = {k: v for k, v in self.streak.items() if k[0] != "v"}

    @staticmethod
    def _clear(r):
        r.A *= 0.0; r.y *= 0.0; r.n = 0; r._w = None
        r._ps = []

    # ---------------- probe cap as a token bucket ----------------
    def _cap_ok(self):
        if not self.bucket:
            return super()._cap_ok()
        self.tok_b = min(float(self.pcap_base), self.tok_b + self.pcap * (self.rounds - self.tok_at))
        self.tok_at = self.rounds
        if self.tok_b >= 1.0:
            self.tok_b -= 1.0
            return True
        return False

    # ---------------- observation: regressions, drift detector, K check ----------------
    def _radd(self, reg, x, t):
        reg.add(x, t)
        if self.rob and self._probe_now:
            if not hasattr(reg, "_ps"):
                reg._ps = []
            reg._ps.append((np.asarray(x, float), float(t), reg.n))

    def _streak(self, key, dev):
        """rob: rounds in a row with a residual beyond rob_clip. With rob_hys (round 32) a residual between
        rob_clip / 2 and rob_clip keeps a running streak instead of ending it."""
        n = self.streak.get(key, 0)
        if abs(dev) > self.rob_clip:
            return n + 1
        return n if getattr(self, "rob_hys", False) and n > 0 and abs(dev) > self.rob_clip / 2 else 0

    def _skip(self, cnt, ckey, skey, dev):
        """rob: True if the sample is withheld from its regression (isolated outlier of an established cell)."""
        return cnt.get(ckey, 0) >= 4 and abs(dev) > self.rob_clip and self.streak.get(skey, 0) < 3

    def _rescreen(self):
        for reg in list(self.rl.values()) + [self.rv]:
            ps = getattr(reg, "_ps", [])
            reg._ps = []
            if reg.n < 6 or not ps:
                continue
            for _ in range(2):
                coef, keep = reg.coef().copy(), []
                for x, t, n0 in ps:
                    p = float(np.dot(coef, x))
                    if p > 0 and abs(t / p - 1) > self.rob_clip:
                        w = reg.lam ** (reg.n - n0)
                        reg.A = reg.A - w * np.outer(x, x)
                        reg.y = reg.y - w * x * t
                        reg._w = None
                        self.n_rescreen += 1
                    else:
                        keep.append((x, t, n0))
                ps = keep

    def _observe(self, b, W, K, layers, wait, suml_all):
        if self.cost_mode == "known":
            return
        trig = []
        c = self.rob_clip
        if not self.rob:
            for m, suml, dt in layers:
                tier = F.bucket(m)
                pred = self._lt(m, W, suml)
                self.rl[W].add([1.0, tier, suml / 1000.0], dt)
                if (tier, W) in self.ref:
                    self._drift((tier, W), dt / max(pred, 1e-6) - 1, trig, "layer")
        else:
            cells = {}
            for m, suml, dt in layers:
                tier = F.bucket(m)
                dev = dt / max(self._lt(m, W, suml), 1e-6) - 1
                cells.setdefault((tier, W), []).append((dev, [1.0, tier, suml / 1000.0], dt))
            for key, xs in cells.items():
                med = float(np.median([d for d, _, _ in xs]))
                self.streak[key] = self._streak(key, med)
                for d, x, dt in xs:
                    if self._skip(self.cell_nl, key, key, d):
                        self.n_skip += 1
                        continue
                    self._radd(self.rl[W], x, dt)
                    self.cell_nl[key] = self.cell_nl.get(key, 0) + 1
                if key in self.ref:
                    self._drift(key, float(np.clip(med, -c, c)), trig, "layer")
        bk = Q.bkey(b)
        vpred = self._vt(b, K, suml_all)
        core = wait - self._lt(b, W, suml_all)
        if self.boff:
            core = core - self.voff.get(bk, 0.0)
        if self.chk_to and self.check is not None and self.rounds - self.check["at"] > self.chk_to:
            self._abort_check()
        chk = self.check is not None and self.check["bk"] == bk and self._probe_now
        if chk:
            self.check["res"][K].append(core - vpred)
            self.check["vp"].append(vpred)
        mon = (self.vdrift and self.rv.n >= 6 and bk in self.probed and not self.probe_q.get(bk)
               and (self.check is None or (self.chk_to and self.check["bk"] != bk)))
        dev = core / max(vpred, 1e-6) - 1
        if not self.rob:
            self.rv.add(self._vx(b, K, suml_all), core)
        else:
            sk = ("v", bk)
            self.streak[sk] = self._streak(sk, dev)
            if self.rv.n >= 6 and self._skip(self.cell_nv, (bk, K), sk, dev):
                self.n_skip += 1
            else:
                self._radd(self.rv, self._vx(b, K, suml_all), core)
                self.cell_nv[(bk, K)] = self.cell_nv.get((bk, K), 0) + 1
            dev = float(np.clip(dev, -c, c))
        if mon:
            self._drift(("v", b, K), dev, trig, "verify")
        if chk and not self.probe_q.get(bk):
            self._end_check()
        if self.rob and self._probe_now and not self.probe_q.get(bk):
            self._rescreen()
        if "layer" in trig:
            self._layer_event()
        elif "verify" in trig:
            self._start_check(bk)
        if self.chk_int and bk in self.chk_pending and self.check is None and not self.probe_q.get(bk):
            if self.bk_cnt.get(bk, 0) >= self.chk_at[bk] + self.chk_iv.get(bk, self.chk_int):
                self._start_check(bk)
        if self.pend:
            self._fire_pending()

    def _abort_check(self):
        cb = self.check["bk"]
        self.probe_q[cb] = [x for x in self.probe_q.get(cb, []) if x[0] is not None]
        self.check = None
        self.n_chk_abort += 1

    def _start_check(self, bk):
        if self.probe_q.get(bk):
            return
        if self.chk_to and self.check is not None:
            self._abort_check()
        if self.chk_int:
            n = self.bk_cnt.get(bk, 0)
            last, iv = self.chk_at.get(bk), self.chk_iv.get(bk, self.chk_int)
            if last is not None and n < last + iv:
                self.n_chk_supp += 1
                self.chk_pending.add(bk)
                for key, f in self.fast.items():
                    if key[0] == "v":
                        f[1] = 0
                return
            if last is not None:
                self.chk_iv[bk] = min(2 * iv, self.chk_int_max) if n < last + 2 * iv else self.chk_int
            self.chk_at[bk] = n
            self.chk_pending.discard(bk)
        super()._start_check(bk)
        self.check["vp"] = []

    def _end_check(self):
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
            if self.boff:
                self.voff[c["bk"]] = self.voff.get(c["bk"], 0.0) + mu
            else:
                d = np.zeros(len(self.rv.y)); d[:3] = mu
                self.rv.y = self.rv.y + self.rv.A @ d
                self.rv._w = None
            self.n_offsets += 1
            return
        self._kind = "verify"
        if not self._gate():
            return
        self._verify_event()

    # ---------------- controller: gain with its standard error ----------------
    @staticmethod
    def _gain_se(st, g):
        n = st[4]
        if n < 3:
            return float("inf")
        m = np.array(st[:4]) / n
        cov = st[6] / n - np.outer(m, m)
        w = np.array([1.0 / m[0], -1.0 / m[1], -1.0 / m[2], 1.0 / m[3]])
        var = float(w @ cov @ w) * n / (n - 1)
        return (1.0 + g) * float(np.sqrt(max(var, 0.0) / n))

    def _post14m(self, b, W, tag, rep, com, t_round, m_actual):
        s = self.ctx14[Q.bkey(b)]
        cur = s["cur"]
        if cur not in rep:
            return
        tm = lambda x: t_round - m_actual + rep[x]["mt"]
        if tag == "explore":
            ctok, ctm = rep[cur]["tok"], tm(cur)
        else:
            ctok, ctm = float(sum(com)), t_round
        for o in rep:
            if o == cur:
                continue
            st = s["stats"].setdefault(o, [0.0, 0.0, 0.0, 0.0, 0, 0.0, np.zeros((4, 4))])
            a, t_ = rep[o]["tok"], tm(o)
            st[0] += a; st[1] += t_; st[2] += ctok; st[3] += ctm; st[4] += 1; st[5] += rep[o]["lt"]
            v = np.array([a, t_, ctok, ctm])
            st[6] += np.outer(v, v)
        s["n"] += 1
        if s["n"] % 20 == 0:
            best, bg, bse = None, 0.03, None
            for o, st in s["stats"].items():
                need = 30 if o < cur else 10
                if st[4] < need:
                    continue
                g = (st[0] / st[1]) / (st[2] / st[3]) - 1
                se = self._gain_se(st, g)
                if self.se_z and g <= self.se_z * se:
                    if g > 0.03:
                        self.n_se_block += 1
                    continue
                if g > bg:
                    best, bg, bse = o, g, se
            if best is not None:
                s["rc"] = None
                n_ = s["stats"][best][4]
                s["cur"], s["follow"], s["stats"] = best, False, {}
                self.n_switch += 1
                if len(self.switch_log) < 30:
                    self.switch_log.append((self.rounds, b, cur, best))
                if len(self.sw_se) < 30:
                    self.sw_se.append((self.rounds, b, cur, best, round(float(bg), 4), round(float(bse), 4), n_))


# ---------------- worlds ----------------
def add_changes(w, changes):
    """Layer cost terms (d0, d_req, d_tok) x mult at the given speculative rounds: changes = [(round, mult), ...]."""
    orig = w.real_T_l
    todo = dict(changes)

    def real_T_l(rng, b, K, suml):
        out = orig(rng, b, K, suml)
        m = todo.pop(w.n_T, None)
        if m is not None:
            w.d0, w.d_req, w.d_tok = w.d0 * m, w.d_req * m, w.d_tok * m
        return out
    w.real_T_l = real_T_l


def add_bchg(w, d, at=400):
    """Verify cost per request c_b + d ms at the at-th speculative round: K-independent, proportional to b."""
    orig = w.real_T_l
    st = {"done": False}

    def real_T_l(rng, b, K, suml):
        if not st["done"] and w.n_T >= at:
            w.c_b += d
            st["done"] = True
        return orig(rng, b, K, suml)
    w.real_T_l = real_T_l


def add_tail(w, p_round=0.01, p_fwd=0.005, seed=5):
    """Timing spikes. With probability p_round a round stalls: every layer forward x U(3, 8) and the wait + U(50, 300)
    ms. Otherwise each layer forward is x U(2, 6) and the wait + U(20, 100) ms with probability p_fwd. Drawn from a
    generator keyed by the speculative round, apart from the simulation's generator. Apply last (outermost)."""
    st = {"n": -1}

    def cur():
        if st["n"] != w.n_T:
            g = np.random.default_rng([seed, w.n_T])
            st.update(n=w.n_T, rng=g, stall=bool(g.random() < p_round), lm=float(g.uniform(3.0, 8.0)),
                      wx=float(g.uniform(50.0, 300.0)))
        return st
    o_cd, o_T = w.real_cd_l, w.real_T_l

    def real_cd_l(rng, n, W, suml):
        s = cur()
        c = o_cd(rng, n, W, suml)
        if s["stall"]:
            return c * s["lm"]
        return c * float(s["rng"].uniform(2.0, 6.0)) if s["rng"].random() < p_fwd else c

    def real_T_l(rng, b, K, suml):
        s = cur()
        stall, wx, g = s["stall"], s["wx"], s["rng"]
        c = o_T(rng, b, K, suml)
        if stall:
            return c + wx
        return c + float(g.uniform(20.0, 100.0)) if g.random() < p_fwd else c
    w.real_cd_l, w.real_T_l = real_cd_l, real_T_l
