"""Round 37: two checks on the settled 3% width rule, plus three diagnostics.

BatchAccount21 (defaults: the base class, bit for bit):
  wide_need   minimum pairs before a switch to a wider width. The default 10 is the current rule;
              30 is the pre-registered alternative. The decision stays "every 20 rounds, gain > 3%",
              and a narrower width still needs 30 pairs. Used from the win200 path. A window of 10**9
              never fills, so the sums are the full history.
  explore_p   probability of building one step wider. The default 0.05 leaves _choose on the base class.
              0 turns that branch off and does not draw the random number. Diagnostic only.
  klog        every 250 rounds, snapshot the cumulative K counts. Diagnostic only.
  biaslog     on controller rounds, score each narrower width with a shadow tree as well as the replay,
              without replacing the replay values the decision uses. Also grow one shadow tree at the
              width that actually served the round, and accumulate its tokens and model time against
              that tree. Diagnostic only; the shadow stream is not the world's rng.
"""
import ext9 as Q
import ext20 as V
import ext27 as E


class BatchAccount21(E.BatchAccount20):
    def __init__(self, wide_need=10, explore_p=0.05, klog=False, biaslog=False, **kw):
        super().__init__(**kw)
        BatchAccount21.current = self
        self.wide_need = int(wide_need)
        self.explore_p = float(explore_p)
        self.klog = bool(klog)
        self.biaslog = bool(biaslog)
        self.k_snap = []
        self.bias = {}
        self._bias_slots = None

    def _rule3_decide(self, s, b, cur):
        best, bg, bse, bn = None, 0.03, None, None
        for o, st in s["stats"].items():
            need = 30 if o < cur else self.wide_need
            if st[4] < need or st[1] <= 0 or st[3] <= 0:
                continue
            g = (st[0] / st[1]) / (st[2] / st[3]) - 1
            se = self._gain_se(st, g)
            if g > bg:
                best, bg, bse, bn = o, g, se, st[4]
        if best is not None:
            self._switch_to(s, b, cur, best, bg, bse, bn)

    def _choose(self, b, cls, warm, slots, ell, theta):
        if self.explore_p == 0.05 or self.wmode != "replay14":
            return super()._choose(b, cls, warm, slots, ell, theta)
        Wflat = V.flat_wk(b, cls)[0]
        if warm or self.wmode == "flat":
            return Wflat, "normal"
        if self.wmode == "fixed":
            return self.fixed_W, "normal"
        bk = Q.bkey(b)
        s = self.ctx14.setdefault(bk, dict(cur=Wflat, follow=True, stats={}, n=0))
        if s["follow"]:
            s["cur"] = Wflat
        wider = [x for x in V.W_OPTS if x > s["cur"]]
        if wider and self.explore_p > 0 and self.rng2.random() < self.explore_p:
            self.n_explore += 1
            return wider[0], "explore"
        return s["cur"], "normal"

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        self._bias_slots = slots if self.biaslog else None
        try:
            out = super().round(world, slots, rng, rtt, rnd, trees_out)
        finally:
            self._bias_slots = None
        if self.klog and self.rounds % 250 == 0:
            self.k_snap.append([self.rounds, {str(k): int(v) for k, v in sorted(self.k_log.items())}])
        return out

    def _replay_m(self, trees, cs, rem, layers, W, b, theta, suml_all):
        out = super()._replay_m(trees, cs, rem, layers, W, b, theta, suml_all)
        if self.biaslog and self._bias_slots is not None and W in out:
            ell = [self.world.reqs[r]["p0"] + j for r, j in self._bias_slots]
            bk = Q.bkey(b)
            for Wn in list(out):
                if Wn > W:
                    continue
                tok, mt = self._shadow(self.world, self._bias_slots, Wn, cs, rem, ell, b, theta, suml_all)
                key = (bk, int(W), int(Wn))
                st = self.bias.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0])
                st[0] += out[Wn]["tok"]
                st[1] += out[Wn]["mt"]
                st[2] += tok
                st[3] += mt
                st[4] += 1
        return out

    def bias_rows(self):
        rows = []
        for (bk, W, Wn), (rt, rm, st, sm, n) in sorted(self.bias.items()):
            rows.append(dict(b=int(bk), W=int(W), Wn=int(Wn), n=int(n),
                             rep_tok=rt, rep_mt=rm, sh_tok=st, sh_mt=sm))
        return rows
