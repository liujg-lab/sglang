"""Round 45 (exploratory, not pre-registered): W chosen only from real probe rounds, both directions. Measured-cost
world and the per-(W, g) layer fit of round 43. Synthetic world; not a measurement.

No replay and no truncation estimate. Probe rounds really build the other width; the comparison is the real round
throughput (sum tokens / sum round ms) of the probe rounds against the served-width rounds of the same bucket.
Rounds on tau (K probe, missing layer fit) are not compared.

ProbeZ (both sides every period): in each period of 100 eligible rounds, Bresenham-spaced, 5 rounds at 2W and 5 at
W/2 (alternating). Windows: last 1500 base rounds, last 150 rounds per probe side. Every 20 eligible rounds, after a
200-round dwell and at least MIN_N rounds per side: gain = S_side / S_base - 1, se from the ratio-estimator delta
method; move to the side with the largest gain - Z * se when it exceeds DELTA. Statistics reset on a switch.

ProbeAC (AdaCost OnlineSpeedupOptimizerInterleaved, discrete tiers): period 100, 10 perturb rounds at the
neighbouring tier in the current probe direction. At the end of a period: perturb throughput above base
(rel gain > min_delta, default 0) -> move one tier that way and keep the direction; else flip the direction.
"""
import math
from collections import deque

from ext31 import CodeMirror, bbucket, narrower, wider
from ext34 import GFit

PERIOD = 100


def _slot(i, n_p, period=PERIOD):
    return ((i + 1) * n_p) // period > (i * n_p) // period


def _ratio_stats(rows):
    n = len(rows)
    if n < 2:
        return None, None
    st = sum(t for t, _m in rows)
    sm = sum(m for _t, m in rows)
    if sm <= 0:
        return None, None
    s = st / sm
    mm = sm / n
    var = sum((t - s * m) ** 2 for t, m in rows) / (n - 1)
    return s, math.sqrt(var / n) / mm


class _ProbeBase(CodeMirror):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.pst = {}
        self.n_probe_rounds = 0

    def _replay(self, pv):
        return

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        out = super().round(world, slots, rng, rtt, rnd, trees_out)
        pv = self.prev
        if pv is not None and self.ledger:
            plan = pv["plan"]
            if not plan["use_tau"] and plan["kind"] in ("ordinary", "probe_up", "probe_down"):
                self._record(bbucket(pv["b"]), plan["kind"], plan["W"], float(sum(pv["com"])), pv["t_round"])
        return out


class ProbeZ(_ProbeBase):
    def __init__(self, n_up=5, n_down=5, z=1.0, delta=0.005, min_n=30, base_win=1500, probe_win=150, **kw):
        super().__init__(**kw)
        self.n_up, self.n_down, self.z, self.delta, self.min_n = n_up, n_down, z, delta, min_n
        self.base_win, self.probe_win = base_win, probe_win

    def _st(self, bk):
        s = self.pst.get(bk)
        if s is None:
            s = self.pst[bk] = dict(i=0, k=0, since=0, base=deque(maxlen=self.base_win),
                                    up=deque(maxlen=self.probe_win), down=deque(maxlen=self.probe_win))
        return s

    def _choose_w(self, b, eligible, suml):
        base = self._persist(b)
        if not eligible:
            return base, "ordinary"
        bk = bbucket(b)
        s = self._st(bk)
        i = s["i"] % PERIOD
        s["i"] += 1
        n_p = self.n_up + self.n_down
        if _slot(i, n_p):
            s["k"] += 1
            want_up = s["k"] % 2 == 1
            up, down = wider(base), narrower(base)
            if want_up and up is not None:
                self.n_probe_rounds += 1
                return up, "probe_up"
            if down is not None:
                self.n_probe_rounds += 1
                return down, "probe_down"
            if up is not None:
                self.n_probe_rounds += 1
                return up, "probe_up"
        return base, "ordinary"

    def _record(self, bk, kind, W, tok, ms):
        s = self._st(bk)
        served = self.persist.get(bk)
        key = {"ordinary": "base", "probe_up": "up", "probe_down": "down"}[kind]
        s[key].append((tok, ms))
        if key != "base":
            return
        s["since"] += 1
        if s["since"] % 20 or s["since"] < 200:
            return
        sb, eb = _ratio_stats(list(s["base"]))
        if sb is None or sb <= 0:
            return
        best = None
        for side in ("up", "down"):
            rows = list(s[side])
            if len(rows) < self.min_n:
                continue
            sp, ep = _ratio_stats(rows)
            if sp is None:
                continue
            g = sp / sb - 1.0
            se = math.sqrt((ep / sb) ** 2 + (sp * eb / sb ** 2) ** 2)
            lb = g - self.z * se
            if best is None or lb > best[0]:
                best = (lb, side, g, se)
        if best is None or best[0] <= self.delta:
            return
        cur = self.persist.get(bk, 8 if bk == 1 else 4)
        tgt = wider(cur) if best[1] == "up" else narrower(cur)
        if tgt is None:
            return
        self.persist[bk] = tgt
        self.switch_log.append(("probe", self.rounds, bk, cur, tgt, round(best[2], 4), round(best[3], 4)))
        self.pst[bk] = None
        self.pst.pop(bk)


class ProbeAC(_ProbeBase):
    def __init__(self, n_p=10, min_delta=0.0, **kw):
        super().__init__(**kw)
        self.n_p, self.min_delta = n_p, min_delta

    def _st(self, bk):
        s = self.pst.get(bk)
        if s is None:
            s = self.pst[bk] = dict(i=0, dir=1, bt=0.0, bm=0.0, pt=0.0, pm=0.0, nb=0, np=0)
        return s

    def _choose_w(self, b, eligible, suml):
        base = self._persist(b)
        if not eligible:
            return base, "ordinary"
        bk = bbucket(b)
        s = self._st(bk)
        i = s["i"]
        s["i"] += 1
        # probes rejected as tau rounds are made up after the period ends (round-45 smoke stall fix)
        if (i < PERIOD and _slot(i, self.n_p)) or (i >= PERIOD and s["np"] < self.n_p and i % 5 == 0):
            tgt = wider(base) if s["dir"] > 0 else narrower(base)
            if tgt is None:
                s["dir"] = -s["dir"]
                tgt = wider(base) if s["dir"] > 0 else narrower(base)
            if tgt is not None:
                self.n_probe_rounds += 1
                return tgt, ("probe_up" if tgt > base else "probe_down")
        return base, "ordinary"

    def _record(self, bk, kind, W, tok, ms):
        s = self._st(bk)
        if kind == "ordinary":
            s["bt"] += tok; s["bm"] += ms; s["nb"] += 1
        else:
            want = "probe_up" if s["dir"] > 0 else "probe_down"
            if kind != want:
                return
            s["pt"] += tok; s["pm"] += ms; s["np"] += 1
        if s["nb"] >= PERIOD - self.n_p and s["np"] >= self.n_p:
            sb = s["bt"] / s["bm"] if s["bm"] > 0 else 0.0
            sp = s["pt"] / s["pm"] if s["pm"] > 0 else 0.0
            cur = self.persist.get(bk, 8 if bk == 1 else 4)
            d = s["dir"]
            if sb > 0 and sp / sb - 1.0 > self.min_delta:
                tgt = wider(cur) if d > 0 else narrower(cur)
                if tgt is not None:
                    self.persist[bk] = tgt
                    self.switch_log.append(("ac", self.rounds, bk, cur, tgt, round(sp / sb - 1.0, 4)))
            else:
                d = -d
            self.pst[bk] = dict(i=0, dir=d, bt=0.0, bm=0.0, pt=0.0, pm=0.0, nb=0, np=0)


class ProbeZG(GFit, ProbeZ):
    pass


class ProbeACG(GFit, ProbeAC):
    pass


ARMS = {
    "probe_z": (ProbeZG, dict()),
    "probe_ac": (ProbeACG, dict()),
}
