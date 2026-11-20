"""Round 50 (exploratory, not pre-registered): pair_g with block-wise adaptive superset rate (pairB).
Synthetic world; not a measurement.

Same paired values, own-K cut, layer fit x shared level as ext41.PairW. Per b bucket, in blocks:
  superset rate p (first block 20 %). Up samples come from superset rounds (2W), down samples from every ordinary
  round (W/2 cut) and every superset round.
  at 15 up samples: g_up - 2 se > 0.5 %          -> block ends (judge)
                    g_up + 2 se < -0.5 %         -> block ends (judge)
                    otherwise                    -> continue
  at 30 up samples: block ends (judge)
  judge (once per block): widen when g_up - 2 se > 0.5 %; narrow when >= 15 down samples and g_down - 2 se > 0.5 %;
                    both -> larger lower bound; otherwise stay.
  next block: switched -> p = 20 %; else p = max(p / 2, 5 %). Samples are cleared at every block end.
  se = max(ratio-estimator se, 1 % / sqrt(n)) (floor on the per-round sd: 1 % of round tokens).
  At the widest width (no superset rounds) the block counts would-be superset rounds (p per eligible round).
No saturation signal, no real-throughput verification, no dwell beyond the block itself.
"""
import math

from ext31 import bbucket, narrower, wider
from ext33 import NARROW_IDLE
from ext41 import PairW, _stats, W_MAX


class PairB(PairW):
    def __init__(self, p0=0.20, p_min=0.05, n1=15, n2=30, z=2.0, band=0.005, sig_floor=0.01, min_down=15, **kw):
        super().__init__(up_band=band, down_band=band, **kw)
        self.p0, self.p_min, self.n1, self.n2 = float(p0), float(p_min), int(n1), int(n2)
        self.z, self.band, self.sig_floor, self.min_dn = float(z), float(band), float(sig_floor), int(min_down)
        self.blocks = 0
        self.block_log = []

    def _st(self, bk):
        s = self.ps.get(bk)
        if s is None:
            s = self.ps[bk] = dict(up=[], down=[], acc=0.0, virt=0.0, p=self.p0)
        return s

    def _gs(self, rows):
        g, se = _stats(rows)
        if g is None:
            return None, None
        return g, max(se, self.sig_floor / math.sqrt(len(rows)))

    # ------------------------------------------------------------ width per round
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
        s = self._st(bk)
        wid = wider(base)
        if wid is None or wid > W_MAX:
            s["virt"] += s["p"]
            return base, "ordinary"
        s["acc"] = min(s["acc"] + s["p"], 2.0)
        if s["acc"] >= 1.0 and self._ready(b, wid, suml):
            s["acc"] -= 1.0
            self.n_sup += 1
            return wid, "sup"
        return base, "ordinary"

    # ------------------------------------------------------------ samples and blocks
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
        nar = narrower(served)
        if kind == "ordinary" and W == served:
            if nar is not None:
                vw, vn = self._value(pv, W, theta, suml), self._value(pv, nar, theta, suml)
                if vw is not None and vn is not None:
                    s["down"].append((vn - vw, tok))
        elif kind == "sup" and W == wider(served):
            v2, vw = self._value(pv, W, theta, suml), self._value(pv, served, theta, suml)
            if v2 is not None and vw is not None:
                s["up"].append((v2 - vw, tok))
            if nar is not None and vw is not None:
                vn = self._value(pv, nar, theta, suml)
                if vn is not None:
                    s["down"].append((vn - vw, tok))
        else:
            return
        self._block_check(bk, s, served)

    def _block_check(self, bk, s, W):
        wid = wider(W)
        has_up = wid is not None and wid <= W_MAX
        n = len(s["up"]) if has_up else int(s["virt"])
        if n < self.n1:
            return
        end = n >= self.n2
        if not end and has_up:
            g, se = self._gs(s["up"])
            if g is not None and (g - self.z * se > self.band or g + self.z * se < -self.band):
                end = True
        if not end:
            return
        self.blocks += 1
        cands = []
        gu = su = gd = sd = None
        if has_up and len(s["up"]) >= 2:
            gu, su = self._gs(s["up"])
            if gu is not None and gu - self.z * su > self.band:
                cands.append((gu - self.z * su, wid, "up", gu, su))
        nar = narrower(W)
        if nar is not None and len(s["down"]) >= self.min_dn:
            gd, sd = self._gs(s["down"])
            if gd is not None and gd - self.z * sd > self.band:
                cands.append((gd - self.z * sd, nar, "down", gd, sd))
        p = s["p"]
        if len(self.block_log) < 200:
            self.block_log.append((self.rounds, bk, W, round(p, 3), len(s["up"]), len(s["down"]),
                                   None if gu is None else round(gu, 4), None if su is None else round(su, 4),
                                   None if gd is None else round(gd, 4), None if sd is None else round(sd, 4)))
        if cands:
            lb, tgt, why, g, se = max(cands)
            self._ev(why)
            self.persist[bk] = tgt
            self.switch_log.append(("pairB", self.rounds, bk, W, tgt, why, round(g, 4), round(se, 4)))
            self.ps[bk] = dict(up=[], down=[], acc=0.0, virt=0.0, p=self.p0)
        else:
            self._ev("stay")
            s.update(up=[], down=[], virt=0.0, p=max(p / 2.0, self.p_min))

    def _judge(self, bk, s, W):
        return

    def diag(self):
        out = {str(bk): dict(p=s["p"], nu=len(s["up"]), nd=len(s["down"])) for bk, s in self.ps.items()}
        out["level"] = {str(k): round(v, 3) for k, v in getattr(self, "lvl", {}).items()}
        out["blocks"] = self.blocks
        return out


ARMS = {"pairB_g": (PairB, dict())}
