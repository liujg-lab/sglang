"""Round 45 (exploratory, not pre-registered): saturation rate proposes, real probe rounds confirm or veto.
Measured-cost world and the per-(W, g) layer fit of round 43. Synthetic world; not a measurement.

Per b bucket, on top of the sat_g signals (EMA of bind share and of the all-half indicator, ext34.SatW):
  steady probes (per 100 eligible rounds, Bresenham accumulators). A due probe waits for a round on which it
  would run on the value rule (layer fit, price and verify time ready); if the probe width has no layer fit yet it
  runs anyway (tau, not compared) so the fit can start. Tau rows are never compared.
      at 2W:  UP_MAX   * clip((B - 0.05) / 0.45, 0, 1)
      at W/2: DOWN_MAX * clip((H - 0.30) / 0.60, 0, 1)
  the SigW narrow timing rounds (probe_w) also count as W/2 rows.
  real rows: (round tokens, round ms) of non-tau rounds; base = served-width ordinary rounds (last 600),
  up / down = probe rows (last 150). g = S_side / S_base - 1, se by the ratio delta method.
Every 20 base rounds, after a 200-round dwell:
  verify mode (right after a switch; probes the old width at VERIFY_RATE per 100):
      once >= MIN_N old-width rows and >= 100 base rows: roll back when g_old - se > 0, else confirm.
      A rolled-back widening blocks saturation widening to that width for BLOCK0 eligible rounds (doubling).
      Verify ends unconfirmed (= confirm) after VERIFY_CAP base rounds.
  steady mode:
      1. widen when EMA(bind) > BIND_UP (>= 100 samples) and the target width is not blocked
      2. widen when real g_up - se > DELTA (>= MIN_N rows)
      3. narrow when real g_down - se > DELTA (>= MIN_N rows); blocks saturation widening back for BLOCK0
  W <= 16. No half fast path: narrowing needs real evidence.
Variant satpv_g: no steady probes (only verify after a saturation widening, and probe_w rows for narrowing).
"""
import math
from collections import deque

from ext31 import bbucket, narrower, wider
from ext33 import NARROW_IDLE, EMA_N, DWELL, JUDGE_EVERY
from ext34 import GFit, SatW
from ext36 import _ratio_stats

W_MAX = 16


def _clip(x):
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def _gain(base, side):
    sb, eb = _ratio_stats(base)
    sp, ep = _ratio_stats(side)
    if sb is None or sp is None or sb <= 0:
        return None
    g = sp / sb - 1.0
    se = math.sqrt((ep / sb) ** 2 + (sp * eb / sb ** 2) ** 2)
    return g, se


class SatP(GFit, SatW):
    def __init__(self, bind_up=0.25, up_max=5.0, down_max=5.0, verify_rate=25.0, min_n=30, delta=0.005,
                 rb_delta=0.0, block0=3000, base_win=600, probe_win=150, verify_cap=800, **kw):
        super().__init__(bind_up=9.0, half_down=9.0, **kw)
        self.p_bind = float(bind_up)
        self.up_max, self.down_max, self.verify_rate = float(up_max), float(down_max), float(verify_rate)
        self.min_n, self.delta, self.rb_delta = int(min_n), float(delta), float(rb_delta)
        self.block0, self.base_win, self.probe_win, self.verify_cap = int(block0), int(base_win), int(probe_win), int(verify_cap)
        self.ps = {}
        self.n_probe_rounds = 0
        self.n_verify_rounds = 0
        self.p_events = {}

    def _ps(self, bk):
        s = self.ps.get(bk)
        if s is None:
            s = self.ps[bk] = dict(base=deque(maxlen=self.base_win), up=deque(maxlen=self.probe_win),
                                   down=deque(maxlen=self.probe_win), acc_up=0.0, acc_down=0.0, acc_old=0.0,
                                   since=0, mode="steady", old=None, block={}, blen={})
        return s

    def _ev(self, k):
        self.p_events[k] = self.p_events.get(k, 0) + 1

    # ------------------------------------------------------------ width per round
    def _plan(self, b, max_ell, suml):
        self._p_max_ell = max_ell
        return super()._plan(b, max_ell, suml)

    def _probe_ok(self, b, w, suml):
        """True when a round at width w would be compared, or when w still needs layer samples."""
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
        n = self.elig_n.get(bk, 0) + 1
        self.elig_n[bk] = n
        nar = narrower(base)
        wid = wider(base)
        if wid is not None and wid > W_MAX:
            wid = None
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
        s = self._ps(bk)
        if s["mode"] == "verify":
            s["acc_old"] = min(s["acc_old"] + self.verify_rate / 100.0, 2.0)
            if s["acc_old"] >= 1.0 and self._probe_ok(b, s["old"], suml):
                s["acc_old"] -= 1.0
                self.n_verify_rounds += 1
                old = s["old"]
                return old, ("probe_up" if old > base else "probe_down")
            return base, "ordinary"
        sat = self.sat_st.get(bk)
        B = sat["bind"] if sat and sat["n"] >= 20 else 0.0
        H = sat["half"] if sat and sat["n"] >= 20 else 0.0
        if wid is not None:
            s["acc_up"] = min(s["acc_up"] + self.up_max * _clip((B - 0.05) / 0.45) / 100.0, 2.0)
        if nar is not None:
            s["acc_down"] = min(s["acc_down"] + self.down_max * _clip((H - 0.30) / 0.60) / 100.0, 2.0)
        if wid is not None and s["acc_up"] >= 1.0 and self._probe_ok(b, wid, suml):
            s["acc_up"] -= 1.0
            self.n_probe_rounds += 1
            return wid, "probe_up"
        if nar is not None and s["acc_down"] >= 1.0 and self._probe_ok(b, nar, suml):
            s["acc_down"] -= 1.0
            self.n_probe_rounds += 1
            return nar, "probe_down"
        return base, "ordinary"

    # ------------------------------------------------------------ records and decisions
    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        out = super().round(world, slots, rng, rtt, rnd, trees_out)
        pv = self.prev
        if pv is not None and self.ledger:
            self._precord(pv)
        return out

    def _precord(self, pv):
        plan = pv["plan"]
        kind = plan["kind"]
        b = pv["b"]
        bk = bbucket(b)
        served = self._persist(b)
        if plan.get("served") != served:
            return
        W = plan["W"]
        s = self._ps(bk)
        row = (float(sum(pv["com"])), float(pv["t_round"]))
        if kind == "ordinary" and W == served:
            if plan["use_tau"]:
                return
            s["base"].append(row)
            s["since"] += 1
            if s["since"] % JUDGE_EVERY == 0:
                self._judge(bk, s, served)
            return
        if kind not in ("probe_up", "probe_down", "probe_w") or W == served:
            return
        side = "up" if W > served else "down"
        if plan["use_tau"]:
            self._ev("probe_tau")
            return
        if side == "up" and W != wider(served):
            return
        if side == "down" and W != narrower(served):
            return
        s[side].append(row)

    def _judge(self, bk, s, W):
        now = self.elig_n.get(bk, 0)
        if s["mode"] == "verify":
            old = s["old"]
            side = "down" if old < W else "up"
            if len(s[side]) >= self.min_n and len(s["base"]) >= 100:
                r = _gain(list(s["base"]), list(s[side]))
                if r is not None:
                    g, se = r
                    if g - se > self.rb_delta:
                        self._ev("rollback_" + ("widen" if old < W else "narrow"))
                        if old < W:
                            ln = s["blen"].get(W, self.block0)
                            s["block"][W] = now + ln
                            s["blen"][W] = 2 * ln
                        self._switch(bk, W, old, "rollback", g, se)
                        return
                    self._ev("confirm")
                    s["mode"] = "steady"
                    return
            if s["since"] >= self.verify_cap:
                self._ev("verify_timeout")
                s["mode"] = "steady"
            return
        if s["since"] < DWELL:
            return
        sat = self.sat_st.get(bk) or {}
        wid = wider(W)
        if wid is not None and wid > W_MAX:
            wid = None
        nar = narrower(W)
        if wid is not None and s["block"].get(wid, -1) <= now and sat.get("n", 0) >= EMA_N // 2 \
                and sat.get("bind", 0.0) > self.p_bind:
            self._ev("sat_widen")
            self._switch(bk, W, wid, "sat", sat["bind"], 0.0)
            return
        if wid is not None and len(s["up"]) >= self.min_n:
            r = _gain(list(s["base"]), list(s["up"]))
            if r is not None and r[0] - r[1] > self.delta:
                self._ev("real_widen")
                self._switch(bk, W, wid, "real_up", r[0], r[1])
                return
        if nar is not None and len(s["down"]) >= self.min_n:
            r = _gain(list(s["base"]), list(s["down"]))
            if r is not None and r[0] - r[1] > self.delta:
                self._ev("real_narrow")
                self._switch(bk, W, nar, "real_down", r[0], r[1])
                return

    def _switch(self, bk, cur, tgt, why, a, b_):
        now = self.elig_n.get(bk, 0)
        self.persist[bk] = tgt
        self.switch_log.append(("satp", self.rounds, bk, cur, tgt, why, round(float(a), 4), round(float(b_), 4)))
        old = self.ps.pop(bk)
        s = self._ps(bk)
        s["block"], s["blen"] = old["block"], old["blen"]
        if why == "real_down":
            s["block"][cur] = now + self.block0
        if why != "rollback":
            s["mode"] = "verify"
            s["old"] = cur
        sat = self.sat_st.get(bk)
        if sat is not None:
            self.sat_st[bk] = dict(n=0, bind=0.0, half=0.0, tok=sat["tok"], up=0.0, down=0.0, nu=0, nd=0, since=0)


ARMS = {
    "satp_g": (SatP, dict()),
    "satpv_g": (SatP, dict(up_max=0.0, down_max=0.0)),
}
