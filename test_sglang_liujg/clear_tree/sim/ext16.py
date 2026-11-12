"""Round 19: event-triggered calibration of the layer width W under one calibration budget (BatchAccount9).

Built on ext15.BatchAccount8 (v10 rule, measured costs known to the policy). W stays at the cell's current value except
inside a calibration window, which opens when a cell (b bucket, acceptance class) first appears after warm-up ("start")
and when the measured cost table changes by > 10% ("cost"). A window lasts at most `budget` rounds of that cell; there
is no exploration outside windows. Arms: W in {2, 4, 8}. Per round the comparison value is
    Y = committed tokens - Theta * round time,   Theta frozen at the window start,
for an arm that ran, or the same quantity from the committed-text replay (narrowed tree, K' chosen by the policy's
own value, time = residual + model time) for an arm narrower than the tree that was built. eps = 1% of the mean
committed tokens per round in the window.

method
  fixed30    build every round at W = 8 for 30 rounds; paired replay differences D(a) = Y(a) - Y(cur);
             switch to the passing arm with the largest mean if mean - 1.645 sd / sqrt(n) > eps (the earlier gate)
  ipw        each window round runs a uniformly random arm (p = 1/3); unpaired per-arm means; looks at 50, 100,
             150, 200 rounds with z = 2.63 (about Pocock for 4 looks, two-sided 5%, split over 2 alternatives);
             switch if diff - z se > eps; an arm is settled if diff + z se < eps; all settled -> stop, keep
  paired_cs  build at the widest arm still in contention (>= cur); narrower arms come free from replay on every
             round; paired differences per arm; plug-in normal-mixture confidence sequence (two-sided, alpha 5% split
             over the alternatives, rho = 50 sd^2), checked every round after 10 pairs: LCB > eps -> switch; UCB < eps
             -> drop the arm; window ends when no arm is left, or at the budget (uncertain -> keep)
  sh         sequential halving with real runs: 3 arms x 33 rounds (random interleaving), drop the worst mean,
             2 arms x 50 rounds (means over both phases), pick the best mean; switch if it beats cur by > eps (or cur
             was dropped)
Multiplicity is only split over the arms of one window, not over cells or events.

Round 19b fixes for paired_cs (all off by default, so run19 is reproducible):
  open_delay  windows (start and cost events) open only after this many further rounds of the cell (Theta_b, c_i settle)
  min_pairs   pairs per arm before any decision (run19: 10)
  cs_type     "nm" plug-in normal mixture (run19) or "eb": predictable plug-in empirical-Bernstein CS on pairs clipped to
              [-R, R]; R = clip_sd * sd + |mean| of the first min_pairs pairs of that arm, which are then left out of the CS
"""
import math
import numpy as np
import ext15 as G
import ext9 as Q


class BatchAccount9(G.BatchAccount8):
    def __init__(self, wk, table, lt, vt_core, method="paired_cs", budget=200, eps_rel=0.01, alpha=0.05, rho_n=50,
                 cal_seed=4711, open_delay=0, min_pairs=10, cs_type="nm", clip_sd=4.0, **kw):
        kw["wmode"] = "cal"
        super().__init__(wk, table, lt, vt_core, **kw)
        self.method, self.budget, self.eps_rel, self.alpha, self.rho_n = method, budget, eps_rel, alpha, rho_n
        self.open_delay, self.min_pairs, self.cs_type, self.clip_sd = open_delay, min_pairs, cs_type, clip_sd
        self.crng = np.random.default_rng(cal_seed)
        self.cells9 = {}
        self.events = []          # (round, b, cls, kind, cur_before, chosen, window rounds, outcome)
        self.cal_rounds = 0

    # ---------------- windows ----------------
    def _event(self, c, kind):
        if self.open_delay > 0:
            c["cal"], c["pending"] = None, dict(kind=kind, wait=self.open_delay)
        else:
            self._open(c, kind)

    def _open(self, c, kind):
        c["pending"] = None
        arms = list(self.w_opts)
        cal = dict(kind=kind, n=0, theta=None, A=[], cur0=c["cur"], data={a: [] for a in arms}, active=set(arms) - {c["cur"]},
                   plan=[], phase=0, sh_arms=list(arms), R={})
        if self.method == "sh":
            q = [a for a in arms for _ in range(33)]
            self.crng.shuffle(q)
            cal["plan"] = q
        c["cal"] = cal

    def _close(self, c, b, cls, chosen, outcome):
        cal = c["cal"]
        self.events.append((self.rounds, b, cls, cal["kind"], cal["cur0"], chosen, cal["n"], outcome))
        if chosen != c["cur"]:
            c["cur"] = chosen
            self.n_switch += 1
            self.switch_log.append((self.rounds, b, cal["cur0"], chosen))
        c["cal"] = None

    def _cost_event(self):
        self.n_events += 1
        for c in self.cells9.values():
            self._event(c, "cost")

    def _choose(self, world, b, cls, Wtab, warm):
        if warm:
            return Wtab, "normal"
        key = (Q.bkey(b), cls)
        c = self.cells9.get(key)
        if c is None:
            c = self.cells9[key] = dict(cur=Wtab, cal=None, pending=None)
            self._event(c, "start")
        if c.get("pending"):
            c["pending"]["wait"] -= 1
            if c["pending"]["wait"] <= 0:
                self._open(c, c["pending"]["kind"])
            else:
                return c["cur"], "normal"
        cal = c["cal"]
        if cal is None:
            return c["cur"], "normal"
        self.cal_rounds += 1
        m = self.method
        if m == "fixed30":
            return self.w_opts[-1], "cal"
        if m == "ipw":
            return int(self.crng.choice(self.w_opts)), "cal"
        if m == "paired_cs":
            return max(cal["active"] | {c["cur"]}), "cal"
        # sh
        return cal["plan"][0], "cal"

    # ---------------- statistics ----------------
    def _Yrep(self, val, tok, theta, b, Wx, resid, layer_m):
        k = self._pickK(val[Wx], theta, b)
        return tok[Wx][k] - theta * (resid + self._model(b, Wx, k, layer_m))

    def _cs(self, x):
        n = len(x)
        mu, sd = float(np.mean(x)), float(np.std(x, ddof=1)) if n > 1 else 0.0
        v = n * sd * sd
        rho = max(1e-12, self.rho_n * sd * sd)
        a = self.alpha / max(1, len(self.w_opts) - 1)
        r = math.sqrt((v + rho) * (math.log((v + rho) / rho) + 2 * math.log(2 / a))) / n
        return mu - r, mu + r

    def _eb(self, d, R):
        """Predictable plug-in empirical-Bernstein CS (Waudby-Smith and Ramdas) for the mean of clip(d) in [-R, R].
        Returns (lcb, ucb) on the d scale. Valid for bounded data with a constant conditional mean, for any predictable
        lambda; the prior variance 1/64 only tunes lambda (it replaces 1/4, which assumes the widest [0, 1] spread)."""
        a = self.alpha / max(1, len(self.w_opts) - 1)
        la = math.log(2 / a)
        mu_prev, var_prev, sx, ssq = 0.5, 1 / 64, 0.0, 0.0
        S_l = S_lx = S_vp = 0.0
        for i, di in enumerate(d, 1):
            x = min(1.0, max(0.0, (di + R) / (2 * R)))
            lam = min(math.sqrt(2 * la / (var_prev * i * math.log(1 + i))), 0.75)
            v = 4 * (x - mu_prev) ** 2
            S_l += lam; S_lx += lam * x; S_vp += v * (-math.log(1 - lam) - lam) / 4
            sx += x
            mu_prev = (0.5 + sx) / (i + 1)
            ssq += (x - mu_prev) ** 2
            var_prev = (1 / 64 + ssq) / (i + 1)
        c, r = S_lx / S_l, (la + S_vp) / S_l
        return 2 * R * (c - r) - R, 2 * R * (c + r) - R

    def _post_cal(self, b, cls, W, tag, trees, cs, rem, layer_m, theta, K, com, t_round):
        c = self.cells9.get((Q.bkey(b), cls))
        if c is None or c["cal"] is None or tag != "cal":
            # paired_cs also learns narrower arms from normal rounds of an open window (none here: window rounds are
            # all tagged "cal"); nothing to do outside windows
            return
        cal = c["cal"]
        if cal["theta"] is None:
            cal["theta"] = theta
        th = cal["theta"]
        cal["n"] += 1
        cal["A"].append(float(sum(com)))
        eps = self.eps_rel * float(np.mean(cal["A"]))
        y_act = float(sum(com)) - th * t_round
        cur = c["cur"]
        m = self.method
        if m in ("fixed30", "paired_cs"):
            val, tok = self._replay_tables(trees, cs, rem, W)
            resid = t_round - self._model(b, W, K, layer_m)
            ys = {a: self._Yrep(val, tok, th, b, a, resid, layer_m) for a in self.w_opts if a <= W}
            if cur in ys:
                for a in ys:
                    if a != cur:
                        cal["data"][a].append(ys[a] - ys[cur])
            if m == "fixed30":
                if cal["n"] >= 30:
                    best, bm = cur, None
                    for a in self.w_opts:
                        d = cal["data"][a]
                        if a == cur or len(d) < 5:
                            continue
                        mu, sd = float(np.mean(d)), float(np.std(d, ddof=1))
                        if mu - 1.645 * sd / math.sqrt(len(d)) > eps and (bm is None or mu > bm):
                            best, bm = a, mu
                    self._close(c, b, cls, best, "switch" if best != cur else "keep")
                return
            # paired_cs
            winner, wmu = None, None
            for a in sorted(cal["active"]):
                d = cal["data"][a]
                if len(d) < self.min_pairs:
                    continue
                if self.cs_type == "eb":
                    # the first min_pairs pairs only fix the clipping bound; the CS uses the later pairs
                    if a not in cal["R"]:
                        pil = np.array(d[:self.min_pairs])
                        cal["R"][a] = max(1e-9, self.clip_sd * float(pil.std(ddof=1)) + abs(float(pil.mean())))
                    rest = d[self.min_pairs:]
                    if len(rest) < 2:
                        continue
                    lo, hi = self._eb(rest, cal["R"][a])
                else:
                    lo, hi = self._cs(d)
                if lo > eps and (wmu is None or np.mean(d) > wmu):
                    winner, wmu = a, float(np.mean(d))
                elif hi < eps:
                    cal["active"].discard(a)
            if winner is not None:
                self._close(c, b, cls, winner, "switch")
            elif not cal["active"]:
                self._close(c, b, cls, cur, "keep-resolved")
            elif cal["n"] >= self.budget:
                self._close(c, b, cls, cur, "keep-uncertain")
            return
        # unpaired methods: record the actual Y of the arm that ran
        cal["data"][W].append(y_act)
        if m == "ipw":
            if cal["n"] in (50, 100, 150, 200) or cal["n"] >= self.budget:
                z = 2.63
                yc = cal["data"][cur]
                winner, wmu, settled = None, None, 0
                alts = [a for a in self.w_opts if a != cur]
                for a in alts:
                    ya = cal["data"][a]
                    if len(ya) < 5 or len(yc) < 5:
                        continue
                    diff = float(np.mean(ya) - np.mean(yc))
                    se = math.sqrt(np.var(ya, ddof=1) / len(ya) + np.var(yc, ddof=1) / len(yc))
                    if diff - z * se > eps and (wmu is None or diff > wmu):
                        winner, wmu = a, diff
                    elif diff + z * se < eps:
                        settled += 1
                if winner is not None:
                    self._close(c, b, cls, winner, "switch")
                elif settled == len(alts):
                    self._close(c, b, cls, cur, "keep-resolved")
                elif cal["n"] >= self.budget:
                    self._close(c, b, cls, cur, "keep-uncertain")
            return
        # sh
        cal["plan"].pop(0)
        if cal["plan"]:
            return
        means = {a: float(np.mean(cal["data"][a])) for a in cal["sh_arms"] if cal["data"][a]}
        if cal["phase"] == 0:
            worst = min(means, key=means.get)
            cal["sh_arms"] = [a for a in cal["sh_arms"] if a != worst]
            cal["phase"] = 1
            q = [a for a in cal["sh_arms"] for _ in range(50)]
            self.crng.shuffle(q)
            cal["plan"] = q
            return
        best = max(means, key=means.get)
        if best != cur and (cur not in means or means[best] - means[cur] > eps):
            self._close(c, b, cls, best, "switch")
        else:
            self._close(c, b, cls, cur, "keep")
