"""Round 20: W calibration with costs learned online from zero (BatchAccount10), vs the same calibration with known costs.

Built on ext16.BatchAccount9 (event-triggered paired replay + confidence sequence, used here with the round-19b fixes:
open 50 rounds after the event, 30 pairs per arm, plug-in normal-mixture CS).

cost_mode "known"   layer time lt(m, W), verify time and the replay's model time come from the world's expected costs
                    (what rounds 14-19 assumed); cost events come from the measured table changing (ext15 probe)
          "online"  nothing is known at the start:
                    layer time   EMA per (graph tier, W) of the observed Draft layer times (step 1/n for the first 5
                                 samples, then 0.1); a cell with < 2 samples falls back to a least-squares fit
                                 d0 + d1 tier + d2 tier W over the cells that have data, else to the nearest cell scaled
                                 by tier x W, else 5 ms
                    verify time  EMA per (b, K) of the round wait minus the estimated first layer (lt(b, W)); fallback
                                 fit a + c b + v b (K + 1) over the cells with data, else the closest K at the same b, else
                                 the mean of all cells, else 40 ms
                    timing probe when a b bucket is first seen after warm-up (and after a cost event), every (W, K) in
                                 {2, 4, 8} x {8, 16, 32} runs probe_passes times in random order (W and K forced); probe
                                 rounds are not calibration rounds
                    cost event   after a probe, the probed layer cells are the reference; a fast EMA (0.2) per cell more
                                 than drift_tol away from its reference for drift_run observations in a row triggers a
                                 cost event: every cell's EMA restarts from its next sample, every b bucket is probed
                                 again, every cell's calibration window is reopened (after open_delay)
calibrate False      no W calibration (the flat table stays); probes and cost learning still run
"""
import numpy as np
import ext16 as H
import ext9 as Q
import feasibility as F


class BatchAccount10(H.BatchAccount9):
    def __init__(self, wk, table, lt, vt_core, cost_mode="online", calibrate=True, probe_passes=2, drift_tol=0.15,
                 drift_run=10, probe_seed=2718, **kw):
        super().__init__(wk, table, lt, vt_core, **kw)
        self.cost_mode, self.calibrate, self.probe_passes = cost_mode, calibrate, probe_passes
        self.drift_tol, self.drift_run = drift_tol, drift_run
        self.prng2 = np.random.default_rng(probe_seed)
        self.lc, self.vc, self.fast, self.ref = {}, {}, {}, {}
        self.probed, self.probe_q = set(), {}           # probe queues per b bucket (not per cell: the class can flip)
        self.n_probe_rounds, self.n_cost_events = 0, 0
        self._kforce = None
        if cost_mode == "online":
            self.lt, self.vt_core, self.table = self._lt_est, self._vt_est, self._vt_est
            self.lt_probe_events = False

    # ---------------- online cost estimates ----------------
    def _lt_est(self, m, W):
        bm = F.bucket(m)
        c = self.lc.get((bm, W))
        if c and c[1] >= 2:
            return c[0]
        pts = [(k[0], k[1], v[0]) for k, v in self.lc.items() if v[1] >= 2]
        if len({(p[0], p[1]) for p in pts}) >= 3:
            A = np.array([[1.0, p[0], p[0] * p[1]] for p in pts]); y = np.array([p[2] for p in pts])
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            est = float(coef[0] + coef[1] * bm + coef[2] * bm * W)
            if est > 0:
                return est if not c else 0.5 * (est + c[0])
        if c:
            return c[0]
        if pts:
            b0, w0, v0 = min(pts, key=lambda p: abs(p[0] * p[1] - bm * W))
            return v0 * (0.5 + 0.5 * (bm * W) / (b0 * w0))
        return 5.0

    def _vt_est(self, b, K):
        c = self.vc.get((b, K))
        if c and c[1] >= 2:
            return c[0]
        pts = [(k[0], k[1], v[0]) for k, v in self.vc.items() if v[1] >= 2]
        if len({(p[0], p[1]) for p in pts}) >= 3 and len({p[0] for p in pts}) >= 2 and len({p[1] for p in pts}) >= 2:
            A = np.array([[1.0, p[0], p[0] * (p[1] + 1)] for p in pts]); y = np.array([p[2] for p in pts])
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            est = float(coef[0] + coef[1] * b + coef[2] * b * (K + 1))
            if est > 0:
                return est
        same = [p for p in pts if p[0] == b]
        if same:
            return min(same, key=lambda p: abs(p[1] - K))[2]
        if c:
            return c[0]
        if pts:
            return float(np.mean([p[2] for p in pts]))
        return 40.0

    @staticmethod
    def _ema(d, key, x, alpha=0.1):
        c = d.setdefault(key, [x, 0])
        c[1] += 1
        a = 1.0 / c[1] if c[1] <= 5 else alpha
        c[0] = x if c[1] == 1 else (1 - a) * c[0] + a * x

    def _obs_costs(self, b, W, K, layer_m, layer_dt, wait):
        if self.cost_mode != "online":
            return
        trig = False
        for m, dt in zip(layer_m, layer_dt):
            key = (F.bucket(m), W)
            self._ema(self.lc, key, dt)
            if key in self.ref:
                f = self.fast.setdefault(key, [dt, 0])
                f[0] = 0.8 * f[0] + 0.2 * dt
                if abs(f[0] / self.ref[key] - 1) > self.drift_tol:
                    f[1] += 1
                    trig = trig or f[1] >= self.drift_run
                else:
                    f[1] = 0
        self._ema(self.vc, (b, K), wait - self._lt_est(b, W))
        if trig:
            self._cost_event()

    def _probe_queue(self):
        q = [(W, K) for W in self.w_opts for K in self.k_options] * self.probe_passes
        self.prng2.shuffle(q)
        return [tuple(int(v) for v in x) for x in q]

    def _cost_event(self):
        if self.cost_mode != "online":
            return super()._cost_event()
        self.n_cost_events += 1
        self.n_events += 1
        for d in (self.lc, self.vc):
            for v in d.values():
                v[1] = 0
        self.ref, self.fast, self.probed, self.probe_q = {}, {}, set(), {}
        for c in self.cells9.values():
            self._event(c, "cost")

    def _event(self, c, kind):
        if not self.calibrate:
            c["cal"], c["pending"] = None, None
            return
        super()._event(c, kind)

    # ---------------- W choice: probes first ----------------
    def _choose(self, world, b, cls, Wtab, warm):
        self._kforce = None
        if warm:
            return Wtab, "normal"
        key = (Q.bkey(b), cls)
        c = self.cells9.get(key)
        if c is None:
            c = self.cells9[key] = dict(cur=Wtab, cal=None, pending=None)
            self._event(c, "start")
        bk = Q.bkey(b)
        if self.cost_mode == "online" and bk not in self.probed:
            self.probe_q[bk] = self._probe_queue()
            self.probed.add(bk)
        q = self.probe_q.get(bk)
        if q:
            W, K = q.pop(0)
            self._kforce = K
            self.n_probe_rounds += 1
            if not q:
                for k, v in self.lc.items():
                    if v[1] >= 1 and k not in self.ref:
                        self.ref[k] = v[0]
            return W, "probe"
        return super()._choose(world, b, cls, Wtab, warm)

    # ---------------- diagnostics ----------------
    def cost_errors(self, world):
        """Layer cells: mean |estimate / truth - 1|. Verify: mean relative error of the K = 32 minus K = 8 difference per b
        (the absolute wait also holds the RTT, which cancels across K). Truth = the current world costs."""
        le = [abs(v[0] / world.c_d(k[0], k[1]) - 1) for k, v in self.lc.items() if v[1] >= 1]
        # verify time: compare the K = 32 minus K = 8 difference (the round wait also holds the RTT, a constant)
        ve = []
        for bb in {k[0] for k in self.vc}:
            lo, hi = self.vc.get((bb, 8)), self.vc.get((bb, 32))
            if lo and hi and lo[1] >= 1 and hi[1] >= 1:
                true = world.v_tok * bb * (32 - 8)
                ve.append(abs((hi[0] - lo[0]) - true) / max(true, 1e-9))
        return (float(np.mean(le)) if le else None, float(np.mean(ve)) if ve else None, len(le), len(ve))
