"""Simulation round 4: the adaptive switch as adjusted by D41, and a long-prefix cost regime.

SpecSwitch2 (D36 + D41): one state per context (the batch size b).
- Start from the static rule: speculate if b < b_static, else Target-only.
- Speculating: if the measured rate over the last `win` speculative rounds of this context is below
  (1 - margin) x the Target-only rate, switch to Target-only. Round 4 re-tests a sliding window every round;
  round 5 (exit_blocks=k) tests disjoint blocks of `win` rounds and exits after k consecutive losing blocks,
  because the sliding test exits falsely when speculation is only 5-13% better (multiple testing).
- Target-only: start a probe of `probe_rounds` speculative rounds whenever the probe time spent in this
  context is at most `share` x (Target-only time + probe time) of this context. After the probe, speculate
  if the probe rate > (1 + margin) x the Target-only rate. No backoff.
The Target-only rate is measured on the last `win` Target-only rounds of the context (prior: b / T_ar(b),
i.e. the Target decode step time, which the Target can measure without speculating).

LONG: the long-prefix regime. Verify and Draft per-token costs grow with the prefix; the factor is chosen so
that OPT(8, 0.4) and Target-only break even near b = 3 at RTT 4 ms (measured: 4096-token prefix, b >= 3
Target-only is faster). This is an assumption, not a measurement of the per-token cost.
"""
import ext as E


class SpecSwitch2:
    def __init__(self, inner, margin=0.03, win=20, probe_rounds=15, share=0.05, b_static=7, start="static",
                 exit_blocks=0, exit_fast=None):
        self.inner, self.m, self.win, self.pr, self.share, self.bs = inner, margin, win, probe_rounds, share, b_static
        self.start = start   # "static": static rule; "spec": every context starts speculating (diagnostic)
        # exit test: 0 = sliding window re-tested every round (round 4); k > 0 = disjoint blocks of `win`
        # speculative rounds, exit after k consecutive losing blocks (round 5)
        self.kb = exit_blocks
        # two-threshold variant (round 5b): a block losing more than exit_fast exits at once
        self.fast = exit_fast
        self.ctx = {}
        self.log = []        # (b, clock_ms, probe rate, Target-only rate, decision)
        self.spec_time = self.total_time = 0.0
        self.probes = 0

    def _st(self, b):
        if b not in self.ctx:
            self.ctx[b] = dict(mode="spec" if (self.start == "spec" or b < self.bs) else "ar", sw=[], aw=[], probe_left=0,
                               probe_t=0.0, ar_t=0.0, blk=0, bad=0)
        return self.ctx[b]

    rate = staticmethod(E.SpecSwitch.rate)

    def _ar_rate(self, world, s, n):
        """Target-only rate: measured on the last `win` Target-only rounds of this context; prior b / T_ar(b)."""
        return self.rate(s["aw"][-self.win:]) if len(s["aw"]) >= 5 else n / world.T_ar(n)

    def round(self, world, slots, rng, rtt, rnd):
        n = len(slots)
        s = self._st(n)
        ar_rate = self._ar_rate(world, s, n)
        if s["mode"] == "ar" and s["probe_left"] == 0 and s["probe_t"] <= self.share * (s["ar_t"] + s["probe_t"]):
            s["probe_left"], s["sw"] = self.pr, []
            self.probes += 1
        probing = s["probe_left"] > 0
        spec = s["mode"] == "spec" or probing
        if spec:
            com, t = self.inner.round(world, slots, rng, rtt, rnd)
            s["sw"].append((sum(com), t)); s["sw"] = s["sw"][-self.win:]
            if probing:
                s["probe_t"] += t
        else:
            com, t = [1] * n, world.T_ar(n)
            s["aw"].append((n, t)); s["aw"] = s["aw"][-self.win:]
            s["ar_t"] += t
        self.total_time += t
        self.spec_time += t if spec else 0.0
        if s["mode"] == "spec" and not probing:
            if self.kb == 0:
                lose = len(s["sw"]) >= self.win and self.rate(s["sw"]) < (1 - self.m) * ar_rate
            else:
                s["blk"] += 1
                lose = False
                if s["blk"] >= self.win:
                    s["blk"] = 0
                    r = self.rate(s["sw"])
                    s["bad"] = s["bad"] + 1 if r < (1 - self.m) * ar_rate else 0
                    lose = s["bad"] >= self.kb or (self.fast is not None and r < (1 - self.fast) * ar_rate)
            if lose:
                s["mode"], s["blk"], s["bad"] = "ar", 0, 0
                self.log.append((n, round(self.total_time), round(1000 * self.rate(s["sw"]), 1), round(1000 * ar_rate, 1), "exit"))
        elif probing:
            s["probe_left"] -= 1
            if s["probe_left"] == 0:
                s["mode"] = "spec" if self.rate(s["sw"]) > (1 + self.m) * ar_rate else "ar"
                s["blk"] = s["bad"] = 0
                self.log.append((n, round(self.total_time), round(1000 * self.rate(s["sw"]), 1), round(1000 * ar_rate, 1), s["mode"]))
        return com, t


def long_kw(factor):
    """World overrides for the long-prefix regime: per-token verify cost t1 and Draft per-row cost cD1 both
    scaled by `factor` relative to the calibrated short-prefix world (t1 = 0.2, cD1 = 0.02)."""
    return dict(t1=0.2 * factor, cD1=0.02 * factor)
