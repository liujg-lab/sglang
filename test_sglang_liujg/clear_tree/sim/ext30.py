"""Round 39: three tail fixes together, plus evidence-based explore decay.

BatchAccount23 (defaults: the base class, bit for bit):
  se_z    inherited. On the 3% rule, a switch also needs gain > se_z x SE(gain). se_z = 0 leaves
          _rule3_decide on the base class. The windowed rule (win200) did not apply this gate.
  decay   when True, a b bucket whose one-step-wider candidate has at least 40 pairs in the
          window and gain + 2 SE < 3% explores at 1% instead of 5%. A switch clears that
          bucket's statistics, and a cost event clears every bucket, so the rate returns to 5%.
          Narrower widths are still replayed every round. decay = False leaves _choose on the
          base class.

win200 and vprior are the base class. With both off, and with se_z = 0 and decay = False,
this class is BatchAccount22.
"""
import ext9 as Q
import ext20 as V
import ext29 as E


class BatchAccount23(E.BatchAccount22):
    def __init__(self, decay=False, **kw):
        super().__init__(**kw)
        BatchAccount23.current = self
        self.decay = bool(decay)

    def _rule3_decide(self, s, b, cur):
        if not self.se_z:
            return super()._rule3_decide(s, b, cur)
        best, bg, bse, bn = None, 0.03, None, None
        for o, st in s["stats"].items():
            need = 30 if o < cur else self.wide_need
            if st[4] < need or st[1] <= 0 or st[3] <= 0:
                continue
            g = (st[0] / st[1]) / (st[2] / st[3]) - 1
            se = self._gain_se(st, g)
            if g <= self.se_z * se:
                if g > 0.03:
                    self.n_se_block += 1
                continue
            if g > bg:
                best, bg, bse, bn = o, g, se, st[4]
        if best is not None:
            self._switch_to(s, b, cur, best, bg, bse, bn)

    def _explore_p(self, s):
        """1% once the wider neighbour is inside the window and its gain's upper bound is below 3%."""
        if not self.decay:
            return 0.05
        wider = [x for x in V.W_OPTS if x > s["cur"]]
        if not wider:
            return 0.05
        st = s["stats"].get(wider[0])
        if st is None or st[4] < 40 or st[1] <= 0 or st[3] <= 0:
            return 0.05
        g = (st[0] / st[1]) / (st[2] / st[3]) - 1.0
        se = self._gain_se(st, g)
        if g + 2.0 * se < 0.03:
            return 0.01
        return 0.05

    def _choose(self, b, cls, warm, slots, ell, theta):
        if not self.decay:
            return super()._choose(b, cls, warm, slots, ell, theta)
        Wflat = V.flat_wk(b, cls)[0]
        if warm or self.wmode == "flat":
            return Wflat, "normal"
        if self.wmode == "fixed":
            return self.fixed_W, "normal"
        if self.wmode != "replay14":
            return super()._choose(b, cls, warm, slots, ell, theta)
        bk = Q.bkey(b)
        s = self.ctx14.setdefault(bk, dict(cur=Wflat, follow=True, stats={}, n=0))
        if s["follow"]:
            s["cur"] = Wflat
        wider = [x for x in V.W_OPTS if x > s["cur"]]
        if wider and self.rng2.random() < self._explore_p(s):
            self.n_explore += 1
            return wider[0], "explore"
        return s["cur"], "normal"
