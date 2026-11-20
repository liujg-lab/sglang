"""Round 49 (exploratory, not pre-registered): pure paired W controller. Synthetic world; not a measurement.

pair_g, per b bucket:
  - superset rounds: 5 per 100 eligible rounds (Bresenham) build 2W (W <= 8) with the normal growth / stop rules.
    A due superset round waits for a round where 2W would run on the value rule, unless 2W has no layer fit yet.
  - narrow timing rounds at W/2 as before (fill to 4 layer samples, then 2 every 300 eligible rounds); not compared.
  - value of width w on a built tree: cut to w (descendants removed, same depth), K* = the policy's own _select_k
    for w, V(w) = sum c_i topK* mass + n  -  Theta_b * (layer_fit(w) for the first and every grown layer + T(K*)).
    ordinary round at W: down sample V(W/2) - V(W); superset round: up sample V(2W) - V(W) and down sample
    V(W/2) - V(W). Samples are relative to the mean round tokens of the same window (paired, ratio of means).
  - layer cost: per-(W, g) forgetting fit on samples divided by a shared level L_g; prediction = fit * L_g.
    L_g follows every layer sample of any width with a ready fit (multiplicative EMA, step 0.02 per sample).
  - every 20 eligible base rounds after a 200-round dwell: widen when >= 30 up samples and mean - se > UP_BAND;
    narrow when >= 100 down samples and mean - se > DOWN_BAND; both -> larger lower bound. Windows reset on a switch
    (up: last 150 samples, down: last 1000).
  No saturation signal, no post-switch real verification.
pair_s: DOWN_BAND = UP_BAND = 0.5 %.
"""
import math
from collections import deque

from ext31 import CodeMirror, bbucket, narrower, wider, gbucket, mass
from ext33 import NARROW_IDLE, DWELL, JUDGE_EVERY
from ext34 import GFit, Reg2
from ext40 import CutView

W_MAX = 16
LEVEL_STEP = 0.02


class LevelFit(GFit):
    """Per-(W, g) layer fit on level-normalized samples; one shared multiplicative level per g."""

    def _lvl(self, g):
        if not hasattr(self, "lvl"):
            self.lvl = {}
        return self.lvl.get(g, 1.0)

    def _layer_add(self, w, m, sl, dt):
        g = gbucket(m)
        L = self._lvl(g)
        r = self.reg.get((w, g))
        if r is not None and r.n >= 4:
            p = r.pred(sl)
            if p is not None and p > 0:
                ratio = dt / (p * L)
                ratio = min(max(ratio, 0.5), 2.0)
                L *= 1.0 + LEVEL_STEP * (ratio - 1.0)
                self.lvl[g] = L
        self.reg.setdefault((w, g), Reg2()).add(sl, dt / L)

    def _pred_layer(self, w, m, sl):
        g = gbucket(m)
        r = self.reg.get((w, g))
        if r is None:
            return None
        p = r.pred(sl)
        return None if p is None else p * self._lvl(g)

    def _slope(self, w):
        def kappa(m, w=w):
            g = gbucket(m)
            r = self.reg.get((w, g))
            if r is None:
                return None
            s = r.slope()
            return None if s is None else s * self._lvl(g)
        return kappa


def _stats(rows):
    """rows: (d, tok). Ratio of means and its se (paired; delta method)."""
    n = len(rows)
    if n < 2:
        return None, None
    sd_ = sum(d for d, _ in rows)
    st = sum(t for _, t in rows)
    if st <= 0:
        return None, None
    g = sd_ / st
    mt = st / n
    var = sum((d - g * t) ** 2 for d, t in rows) / (n - 1)
    return g, math.sqrt(var / n) / mt


class PairW(LevelFit, CodeMirror):
    def __init__(self, sup_per_100=5.0, up_band=0.005, down_band=0.01, min_up=30, min_down=100,
                 up_win=150, down_win=1000, **kw):
        super().__init__(**kw)
        self.sup_rate = float(sup_per_100) / 100.0
        self.up_band, self.down_band = float(up_band), float(down_band)
        self.min_up, self.min_down = int(min_up), int(min_down)
        self.up_win, self.down_win = int(up_win), int(down_win)
        self.ps = {}
        self.n_sup = 0
        self.n_timing = 0
        self.events = {}
        self.dec_log = []

    def _ev(self, k):
        self.events[k] = self.events.get(k, 0) + 1

    def _st(self, bk):
        s = self.ps.get(bk)
        if s is None:
            s = self.ps[bk] = dict(up=deque(maxlen=self.up_win), down=deque(maxlen=self.down_win), acc=0.0, since=0)
        return s

    def _replay(self, pv):
        return

    # ------------------------------------------------------------ width per round
    def _plan(self, b, max_ell, suml):
        self._p_max_ell = max_ell
        self._p_suml = float(suml)
        return super()._plan(b, max_ell, suml)

    def _ready(self, b, w, suml):
        if self._pred_layer(w, b, suml) is None:
            return True
        if self._theta(b) is None:
            return False
        max_ell = self._p_max_ell
        fitting = self._fitting(max_ell)
        ftier = self._covered_flat(b, max_ell, self._cover(w, fitting), fitting)
        return self._time(b, max_ell, ftier) is not None

    def _choose_w(self, b, eligible, suml):
        base = self._persist(b)
        if not eligible:
            return base, "ordinary"
        bk = bbucket(b)
        self.elig_n[bk] = self.elig_n.get(bk, 0) + 1
        nar = narrower(base)
        if nar is not None:
            key = (bk, nar)
            idle = self.narrow_idle.get(key, NARROW_IDLE)
            left = self.pw_left.get(key, 0)
            if self._layer_n(nar) < 4 and left <= 0:
                left = 4
            elif idle >= NARROW_IDLE and left <= 0:
                left = 2
            if left > 0:
                self.pw_left[key] = left - 1
                self.narrow_idle[key] = 0
                self.n_timing += 1
                return nar, "probe_w"
            self.narrow_idle[key] = idle + 1
        wid = wider(base)
        if wid is not None and wid <= W_MAX:
            s = self._st(bk)
            s["acc"] = min(s["acc"] + self.sup_rate, 2.0)
            if s["acc"] >= 1.0 and self._ready(b, wid, suml):
                s["acc"] -= 1.0
                self.n_sup += 1
                return wid, "sup"
        return base, "ordinary"

    # ------------------------------------------------------------ paired values
    def _plan_for(self, plan, w, b, max_ell, suml):
        fitting = self._fitting(max_ell)
        pl = dict(plan)
        pl["W"] = w
        pl["cover"] = self._cover(w, fitting)
        pl["layer_ms"] = self._pred_layer(w, b, suml)
        pl["locked"] = None
        return pl

    def _value(self, pv, w, theta, suml):
        plan = pv["plan"]
        b = pv["b"]
        max_ell = pv["max_ell"]
        cs = pv["snap"]["ratios"]
        cvs = []
        for v in pv["views"]:
            c = CutView(v, w)
            while c.can_reveal():
                c.reveal()
            cvs.append(c)
        dep = [len(c.layers) for c in cvs]
        K = self._select_k(cvs, dep, self._plan_for(plan, w, b, max_ell, suml), cs, b, max_ell)
        tv = self._time(b, max_ell, K)
        if tv is None:
            return None
        t = 0.0
        for m, sl in [(b, suml)] + list(pv["rows"]):
            p = self._pred_layer(w, m, sl)
            if p is None:
                return None
            t += p
        E = sum(cs[i] * mass(cvs[i].scores_upto(dep[i]), K) for i in range(len(cvs))) + len(cvs)
        return E - theta * (t + tv)

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        out = super().round(world, slots, rng, rtt, rnd, trees_out)
        pv = self.prev
        if pv is not None and self.ledger:
            self._observe(pv)
        return out

    def _observe(self, pv):
        plan = pv["plan"]
        b = pv["b"]
        bk = bbucket(b)
        served = self._persist(b)
        if plan.get("served") != served or plan["use_tau"] or pv["snap"] is None:
            return
        theta = self._theta(b)
        if theta is None:
            return
        W = plan["W"]
        kind = plan["kind"]
        suml = self._p_suml
        tok = float(sum(pv["com"]))
        s = self._st(bk)
        if kind == "ordinary" and W == served:
            nar = narrower(W)
            if nar is not None:
                vw, vn = self._value(pv, W, theta, suml), self._value(pv, nar, theta, suml)
                if vw is not None and vn is not None:
                    s["down"].append((vn - vw, tok))
            s["since"] += 1
            if s["since"] % JUDGE_EVERY == 0:
                self._judge(bk, s, served)
            return
        if kind == "sup" and W == wider(served):
            v2, vw = self._value(pv, W, theta, suml), self._value(pv, served, theta, suml)
            if v2 is not None and vw is not None:
                s["up"].append((v2 - vw, tok))
            nar = narrower(served)
            if nar is not None and vw is not None:
                vn = self._value(pv, nar, theta, suml)
                if vn is not None:
                    s["down"].append((vn - vw, tok))

    def _judge(self, bk, s, W):
        if s["since"] < DWELL:
            return
        cands = []
        wid = wider(W)
        if wid is not None and wid <= W_MAX and len(s["up"]) >= self.min_up:
            g, se = _stats(list(s["up"]))
            if g is not None and g - se > self.up_band:
                cands.append((g - se, wid, "up", g, se))
        nar = narrower(W)
        if nar is not None and len(s["down"]) >= self.min_down:
            g, se = _stats(list(s["down"]))
            if g is not None and g - se > self.down_band:
                cands.append((g - se, nar, "down", g, se))
        if not cands:
            return
        lb, tgt, why, g, se = max(cands)
        self._ev(why)
        self.persist[bk] = tgt
        self.switch_log.append(("pair", self.rounds, bk, W, tgt, why, round(g, 4), round(se, 4)))
        self.ps.pop(bk)

    def diag(self):
        out = {}
        for bk, s in self.ps.items():
            gu, su = _stats(list(s["up"]))
            gd, sd = _stats(list(s["down"]))
            out[str(bk)] = dict(nu=len(s["up"]), up=None if gu is None else round(gu, 4),
                                up_se=None if su is None else round(su, 4), nd=len(s["down"]),
                                down=None if gd is None else round(gd, 4), down_se=None if sd is None else round(sd, 4))
        out["level"] = {str(k): round(v, 3) for k, v in getattr(self, "lvl", {}).items()}
        return out


ARMS = {
    "pair_g": (PairW, dict()),
    "pair_s": (PairW, dict(down_band=0.005)),
}


# ------------------------------------------------------------ controls with the same shared layer-cost level
from ext33 import FixW
from ext34 import SatW
from ext37 import SatP


class FixL(LevelFit, FixW):
    pass


class SatL(LevelFit, SatW):
    pass


class SatPL(LevelFit, SatP):
    pass


ARMS.update({
    "fixL2": (FixL, dict(fixed=2)), "fixL4": (FixL, dict(fixed=4)),
    "fixL8": (FixL, dict(fixed=8)), "fixL16": (FixL, dict(fixed=16)),
    "satL_g": (SatL, dict()),
    "satpvL_g": (SatPL, dict(up_max=0.0, down_max=0.0)),
})
