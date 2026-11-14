"""Round 36: replace the replay controller's "every 20 rounds, gain > 3%" rule.

BatchAccount20 (all three switches off: _post14m is BatchAccount19's, bit for bit):
  cusum_k   > 0: a paired CUSUM on the linearized gain. For each candidate width o against the current width c,
            z = (tok_o - tok_c) / A_c - ((1 + k) * time_o - time_c) / T_c,
            S = max(0, S + z / sigma). Switch to o when S > H (Hn if o is narrower, Hw if wider) and at least
            5 pairs have been seen. sigma is the running sd of z with a prior (n0 = 10, sd0 = 0.10). A_c and T_c
            are the running means of the current width's tokens and time. S, the means and the z moments clear
            on a switch and on a cost event. The decision does not use Theta_b, and it is not gated on a 20-round
            tick or the 3% rule.
  win200    > 0: the 3% rule, but each candidate keeps only its most recent `win200` pairs.
  oracle    narrower candidates are scored from a tree grown at that width by its own rule, on an independent
            random stream. The served round's wall clock is unchanged. A wider width is still observed only on
            the 5% exploration rounds. The decision stays the 3% rule. An upper bound on removing replay bias,
            not an implementable policy.
"""
import math
import numpy as np
import ext2 as X
import ext9 as Q
import ext10 as P
import ext26 as N


class BatchAccount20(N.BatchAccount19):
    def __init__(self, cusum_k=0.0, cusum_hn=14.0, cusum_hw=8.0, win200=0, oracle=False, **kw):
        super().__init__(**kw)
        self.cusum_k = float(cusum_k)
        self.cusum_hn = float(cusum_hn)
        self.cusum_hw = float(cusum_hw)
        self.cusum_n0, self.cusum_sd0, self.cusum_min_n = 10.0, 0.10, 5
        self.win200 = int(win200)
        self.oracle = bool(oracle)
        self._oracle_slots = None

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        if not self.oracle:
            return super().round(world, slots, rng, rtt, rnd, trees_out)
        self._oracle_slots = slots
        try:
            return super().round(world, slots, rng, rtt, rnd, trees_out)
        finally:
            self._oracle_slots = None

    def _event(self, kinds):
        super()._event(kinds)
        self._drop_width_state()

    def _reset_stats(self):
        super()._reset_stats()
        if self.reset_on_event:
            self._drop_width_state()

    def _drop_width_state(self):
        for st in self.ctx14.values():
            st.pop("cz", None)
            st.pop("hist", None)

    def _replay_m(self, trees, cs, rem, layers, W, b, theta, suml_all):
        out = super()._replay_m(trees, cs, rem, layers, W, b, theta, suml_all)
        if self.oracle and self._oracle_slots is not None:
            self._oracle_fill(out, cs, rem, W, b, theta, suml_all)
        return out

    def _oracle_fill(self, out, cs, rem, W, b, theta, suml_all):
        world, slots = self.world, self._oracle_slots
        ell = [world.reqs[r]["p0"] + j for r, j in slots]
        for Wn in list(out):
            if Wn >= W:
                continue
            tok, mt = self._shadow(world, slots, Wn, cs, rem, ell, b, theta, suml_all)
            out[Wn]["tok"] = tok
            out[Wn]["mt"] = mt

    def _shadow(self, world, slots, W, cs, rem, ell, b, theta, suml_all):
        """One tree at width W, own exits and depth, not charged to the served round."""
        rng = np.random.default_rng((self.rounds, int(W), int(b), 36))
        trees = [X.Tree2(world.reqs[r], j, world) for r, j in slots]
        n = len(trees)
        params = (world.a, world.g, world.beta, 0.0)
        for t in trees:
            t.expand(W, rng, params)
        Ks = self.k_options
        e_now = [self._es(t) for t in trees]
        e_prev = [{K: 0.0 for K in Ks} for _ in range(n)]
        v_now = [self._vals(cs[i], e_now[i]) for i in range(n)]
        v_prev = [{K: 0.0 for K in Ks} for _ in range(n)]
        kd = self._kd_hat(W)
        active, depth, layers = list(range(n)), 1, []
        while active:
            cand = [i for i in active if depth < world.DMAX - 1 and depth + 1 <= max(1, rem[i] - 1)]
            if not cand:
                break
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
            layers.append((len(keep), float(sum(ell[i] for i in keep))))
            active = keep
            depth += 1
        Jk = {K_: sum(v_now[i][K_] for i in range(n)) - theta * self._vt(b, K_, suml_all) for K_ in Ks}
        K = max(Jk, key=Jk.get)
        acc = [P.last_node(t, t.topk(K, False))[0] for t in trees]
        tok = float(sum(min(a + 1, rem[i]) for i, a in enumerate(acc)))
        return tok, self._mt(b, W, K, layers, suml_all)

    def _post14m(self, b, W, tag, rep, com, t_round, m_actual):
        if not self.cusum_k and not self.win200:
            return super()._post14m(b, W, tag, rep, com, t_round, m_actual)
        n0 = self.n_switch
        s = self.ctx14[Q.bkey(b)]
        cur = s["cur"]
        if cur not in rep:
            return
        tm = lambda x: t_round - m_actual + rep[x]["mt"]
        if tag == "explore":
            ctok, ctm = rep[cur]["tok"], tm(cur)
        else:
            ctok, ctm = float(sum(com)), t_round
        if ctok <= 0 or ctm <= 0:
            return
        if self.cusum_k:
            cz = s.setdefault("cz", {"Ac": 0.0, "Tc": 0.0, "nc": 0, "arm": {}})
            cz["Ac"] += ctok
            cz["Tc"] += ctm
            cz["nc"] += 1
        for o in rep:
            if o == cur:
                continue
            a, t_ = rep[o]["tok"], tm(o)
            if t_ <= 0:
                continue
            self._acc(s, o, a, t_, ctok, ctm, rep[o].get("lt", 0.0))
            if self.cusum_k:
                self._cusum_update(s, o, a, t_, ctok, ctm)
        s["n"] += 1
        if self.cusum_k:
            self._cusum_decide(s, b, cur)
        elif s["n"] % 20 == 0:
            self._rule3_decide(s, b, cur)
        self._streaks(n0, b, rep)

    def _acc(self, s, o, a, t_, ctok, ctm, lt):
        st = s["stats"].setdefault(o, [0.0, 0.0, 0.0, 0.0, 0, 0.0, np.zeros((4, 4))])
        v = np.array([float(a), float(t_), float(ctok), float(ctm)])
        if self.win200:
            h = s.setdefault("hist", {}).setdefault(o, [])
            if len(h) >= self.win200:
                old = h.pop(0)
                st[0] -= old[0]
                st[1] -= old[1]
                st[2] -= old[2]
                st[3] -= old[3]
                st[6] -= np.outer(old, old)
                st[4] -= 1
            h.append(v.copy())
        st[0] += v[0]
        st[1] += v[1]
        st[2] += v[2]
        st[3] += v[3]
        st[4] += 1
        st[5] += lt
        st[6] += np.outer(v, v)

    def _cusum_update(self, s, o, a, t_, ctok, ctm):
        cz = s["cz"]
        Ac, Tc = cz["Ac"] / cz["nc"], cz["Tc"] / cz["nc"]
        z = (a - ctok) / Ac - ((1.0 + self.cusum_k) * t_ - ctm) / Tc
        arm = cz["arm"].setdefault(o, {"S": 0.0, "n": 0, "sum": 0.0, "sumsq": 0.0})
        arm["n"] += 1
        arm["sum"] += z
        arm["sumsq"] += z * z
        n = arm["n"]
        mean = arm["sum"] / n
        ss = arm["sumsq"] - n * mean * mean
        sigma = math.sqrt((self.cusum_n0 * self.cusum_sd0 ** 2 + max(ss, 0.0)) / (self.cusum_n0 + n - 1))
        arm["S"] = max(0.0, arm["S"] + z / sigma)

    def _cusum_decide(self, s, b, cur):
        cz = s.get("cz")
        if not cz:
            return
        best, bS, bg, bn = None, None, None, None
        for o in sorted(cz["arm"]):
            arm = cz["arm"][o]
            if arm["n"] < self.cusum_min_n:
                continue
            H = self.cusum_hn if o < cur else self.cusum_hw
            if arm["S"] > H and (bS is None or arm["S"] > bS):
                st = s["stats"].get(o)
                g = (st[0] / st[1]) / (st[2] / st[3]) - 1 if st and st[1] > 0 and st[3] > 0 else 0.0
                best, bS, bg, bn = o, arm["S"], g, arm["n"]
        if best is not None:
            self._switch_to(s, b, cur, best, bg, bS, bn)

    def _rule3_decide(self, s, b, cur):
        best, bg, bse, bn = None, 0.03, None, None
        for o, st in s["stats"].items():
            need = 30 if o < cur else 10
            if st[4] < need or st[1] <= 0 or st[3] <= 0:
                continue
            g = (st[0] / st[1]) / (st[2] / st[3]) - 1
            se = self._gain_se(st, g)
            if g > bg:
                best, bg, bse, bn = o, g, se, st[4]
        if best is not None:
            self._switch_to(s, b, cur, best, bg, bse, bn)

    def _switch_to(self, s, b, cur, best, gain, extra, n_):
        s["rc"] = None
        s["cur"], s["follow"], s["stats"] = best, False, {}
        s.pop("hist", None)
        s.pop("cz", None)
        self.n_switch += 1
        if len(self.switch_log) < 30:
            self.switch_log.append((self.rounds, b, cur, best))
        if len(self.sw_se) < 30:
            self.sw_se.append((self.rounds, b, cur, best, round(float(gain), 4),
                               None if extra is None else round(float(extra), 4), int(n_)))

    def _streaks(self, n0, b, rep):
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
