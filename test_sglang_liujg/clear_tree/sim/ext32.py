"""Round 41 (exploratory, not pre-registered): per-round width from the root layer, on top of the round-40
mirror of the current code (ext31.CodeMirror). Synthetic world; not a measurement.

Rule (rootw=True), only on an eligible "ordinary" round (not K-probe, probe_w, explore, guard, hold):
  X(lo, hi) = sum_i c_i * sum of request i's root Draft probabilities at ranks lo+1..hi
  dA(lo, hi) ~ beta[bk, lo, hi] * X(lo, hi)      beta = sum dA / sum X, forgetting 0.999, >= 30 pairs
  dC(lo, hi) = Theta_b * sum over (first layer + last round's layer rows of this bucket) of
               [layer_fit(hi) - layer_fit(lo)]
  narrow to lo = narrower(served) when beta * X(lo, served) < dC(lo, served)
  widen to hi = wider(served) when beta * X(served, hi) > dC(served, hi)
dA comes from self-replay of the same round: tokens of the width-hi cut minus tokens of the width-lo cut, both
under the replay value rule, for every adjacent pair at or below the built width. Rounds built by this rule are
not fed to the persistent W arms (no selection bias there); their K/layer/price samples are kept.
The world has C = 8 Draft candidates, so the root band (8, 16] is empty and the rule never widens 8 -> 16.
"""
from ext31 import CodeMirror, WS, bbucket, narrower, wider, parents_of, FORGET

BETA_MIN = 30


class RootW(CodeMirror):
    def __init__(self, rootw=True, **kw):
        super().__init__(**kw)
        self.rootw = rootw
        self.beta = {}
        self._roots = None
        self._cs_now = None
        self._last_rows = {}
        self.n_root = {"root_n": 0, "root_w": 0, "kept": 0, "no_beta": 0, "no_cost": 0}

    # ---------------------------------------------------------------- feature and pieces
    def _x(self, roots, cs, lo, hi):
        return float(sum(c * float(sum(q[lo:hi])) for q, c in zip(roots, cs)))

    def _beta(self, bk, lo, hi):
        st = self.beta.get((bk, lo, hi))
        if st is None or st[2] < BETA_MIN or st[1] <= 0:
            return None
        return st[0] / st[1]

    def _dc(self, b, bk, suml, lo, hi):
        theta = self._theta(b)
        if theta is None:
            return None
        terms = [(b, suml)] + list(self._last_rows.get(bk) or [])
        s = 0.0
        for m, sl in terms:
            ph, pl = self._pred_layer(hi, m, sl), self._pred_layer(lo, m, sl)
            if ph is None or pl is None:
                return None
            s += ph - pl
        return theta * s

    # ---------------------------------------------------------------- per-round width
    def _choose_w(self, b, eligible, suml):
        w, kind = super()._choose_w(b, eligible, suml)
        if not (self.rootw and eligible and kind == "ordinary" and self._roots is not None):
            return w, kind
        bk = bbucket(b)
        nar, wid = narrower(w), wider(w)
        if nar is not None:
            beta = self._beta(bk, nar, w)
            dc = self._dc(b, bk, suml, nar, w)
            if beta is None:
                self.n_root["no_beta"] += 1
            elif dc is None:
                self.n_root["no_cost"] += 1
            elif beta * self._x(self._roots, self._cs_now, nar, w) < dc:
                self.n_root["root_n"] += 1
                return nar, "root_n"
        if wid is not None:
            beta = self._beta(bk, w, wid)
            dc = self._dc(b, bk, suml, w, wid)
            if beta is not None and dc is not None and beta * self._x(self._roots, self._cs_now, w, wid) > dc:
                self.n_root["root_w"] += 1
                return wid, "root_w"
        self.n_root["kept"] += 1
        return w, kind

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        self._roots = [world.reqs[r]["Q"][j] for r, j in slots]
        self._cs_now = [self._c(r) for r, _ in slots]
        roots, cs = self._roots, self._cs_now
        out = super().round(world, slots, rng, rtt, rnd, trees_out)
        if self.prev is not None:
            self.prev["roots"], self.prev["root_cs"] = roots, cs
            self._last_rows[bbucket(len(slots))] = list(self.prev["rows"])
        return out

    # ---------------------------------------------------------------- learning beta, keeping the arms clean
    def _replay_width(self, pv, w, pars, value=True):
        cache = pv.setdefault("_rw", {})
        key = (w, value, pv.get("guard_k") if not value else None)
        if key not in cache:
            cache[key] = super()._replay_width(pv, w, pars, value)
        return cache[key]

    def _replay(self, pv):
        plan = pv["plan"]
        if self.rootw and not plan["use_tau"] and plan["kind"] not in ("guard", "hold") and "roots" in pv:
            built = plan["W"]
            pars = [parents_of(v.t) for v in pv["views"]]
            ws = [w for w in WS if w <= built]
            toks = {w: self._replay_width(pv, w, pars, value=True)[0] for w in ws}
            bk = bbucket(pv["b"])
            for hi in ws:
                lo = narrower(hi)
                if lo is None or lo not in toks:
                    continue
                x = self._x(pv["roots"], pv["root_cs"], lo, hi)
                st = self.beta.setdefault((bk, lo, hi), [0.0, 0.0, 0])
                st[0] = st[0] * FORGET + (toks[hi] - toks[lo])
                st[1] = st[1] * FORGET + x
                st[2] += 1
        if plan["kind"] in ("root_n", "root_w"):
            return
        super()._replay(pv)


ARMS = {
    "code": dict(rootw=False),
    "root": dict(rootw=True),
    "k1": dict(rootw=False, k1=True),
    "k1_root": dict(rootw=True, k1=True),
}
