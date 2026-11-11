"""Adjusted (v1 core) ClearTree for simulation round 3.

CostModel3: like ext2.CostModel but the conservatism factor on the Draft layer cost (fd) and on the verify
cost (ft) also applies to learned costs.
CT3: ext2.CT2 plus a 'shadow' prediction every `shadow_every` rounds: for each width arm, rebuild layer 1
with that width, run the normal planner (mu-hat + DP for every K) and walk the planned layers to get the
predicted tokens and time of the round. The prediction feeds GuidedSlowLoop.
GuidedSlowLoop: per batch-size context, score(arm) = EMA(predicted rate) x calibration(arm), where
calibration = EMA(measured rate / predicted rate) on windows actually run with that arm (1.0 until measured).
Switch when another arm's score beats the current one by `margin`.
"""
import math
import numpy as np
import feasibility as F
import ext2 as X
from feasibility import KS
from ext import best_sub


class CostModel3(X.CostModel):
    """fd_b1: Draft-cost factor used when the batch has a single request (cur_b set by CT3 each round)."""
    fd_b1 = None
    cur_b = 2

    def _fd(self):
        return self.fd_b1 if (self.fd_b1 is not None and self.cur_b <= 1) else self.fd

    def T(self, b, K):
        if self.mode == "learned":
            return self.tw.get((b, K), self.prior_wait) * self.ft
        return self.w.T_t(b, K) * self.ft

    def c_d(self, n, W):
        if self.mode == "learned":
            return self.cd.get((F.bucket(n), W), self.prior_layer) * self._fd()
        return self.w.c_d(n, W) * self._fd()

    def raw_c_d(self, n, W):
        if self.mode == "learned":
            return self.cd.get((F.bucket(n), W), self.prior_layer)
        return self.w.c_d(n, W)

    def raw_T(self, b, K):
        if self.mode == "learned":
            return self.tw.get((b, K), self.prior_wait)
        return self.w.T_t(b, K)


class GuidedSlowLoop:
    def __init__(self, arms=(4, 8, 16), win=30, margin=0.01, start=8):
        self.arms, self.win, self.margin, self.start = arms, win, margin, start
        self.ctx = {}
        self.switches = 0

    def _st(self, b):
        k = 1 if b <= 1 else (4 if b <= 4 else 8)
        if k not in self.ctx:
            self.ctx[k] = dict(cur=self.start, pred={}, cal={a: 1.0 for a in self.arms}, tok=0.0, t=0.0, n=0, psum=0.0, pn=0)
        return self.ctx[k]

    def W(self, b):
        return self._st(b)["cur"]

    def predicted(self, b, W, rate):
        s = self._st(b)
        s["pred"][W] = rate if W not in s["pred"] else 0.8 * s["pred"][W] + 0.2 * rate
        if W == s["cur"]:
            s["psum"] += rate; s["pn"] += 1

    def feedback(self, b, tok, t):
        s = self._st(b)
        s["tok"] += tok; s["t"] += t; s["n"] += 1
        if s["n"] < self.win:
            return
        meas = s["tok"] / s["t"]
        if s["pn"]:
            ratio = meas / (s["psum"] / s["pn"])
            s["cal"][s["cur"]] = 0.5 * s["cal"][s["cur"]] + 0.5 * ratio if s["cal"][s["cur"]] != 1.0 else ratio
        s["tok"] = s["t"] = 0.0; s["n"] = 0; s["psum"] = 0.0; s["pn"] = 0
        score = {a: s["pred"][a] * s["cal"][a] for a in self.arms if a in s["pred"]}
        if not score:
            return
        best = max(score, key=score.get)
        cur = s["cur"]
        if best != cur and (cur not in score or score[best] > (1 + self.margin) * score[cur]):
            s["cur"] = best; self.switches += 1


class CT3(X.CT2):
    def __init__(self, *a, shadow_every=5, **k):
        super().__init__(*a, **k)
        self.shadow_every = shadow_every
        self._rounds = 0

    def predict_rate(self, world, slots, W, Theta, rtt_est):
        n = len(slots)
        cm = self.cm
        trees = [X.Tree2(world.reqs[r], j, world) for r, j in slots]
        prm = [self.params(world, r) for r, _ in slots]
        for t, p in zip(trees, prm):
            t.expand(W, self.prng, p)
        rem = [world.reqs[r]["L"] - j for r, j in slots]
        Dr = [world.DMAX if self.ignore_r else min(world.DMAX, max(0, x - 1)) for x in rem]
        first_ms, later_ms = X.jit_cost(n, W)
        cost_pop = lambda k: cm.c_d(k, W) + later_ms
        allr = list(range(n))
        mus, ses = self.mu_se(world, slots, trees, prm, allr, W, self.prng)
        best = None
        for K in KS:
            G0 = [t.G(K) for t in trees]
            mu_vec = [float(mus[K][r]) for r in range(n)]
            u, el = self.build_u(mu_vec, G0[:], 1, Dr, allr, n, world)
            xs = self.DP(u, el, (1 << n) - 1, cost_pop, Theta)
            # walk the plan
            A = sum(G0) + n
            T = rtt_est + cm.raw_T(n, K) + first_ms
            mask = (1 << n) - 1
            for k in range(u.shape[1]):
                mask = best_sub(xs[k], mask)[0]
                if not mask:
                    break
                S = [i for i in range(n) if mask >> i & 1]
                A += sum(u[i, k] for i in S)
                T += cm.raw_c_d(len(S), W) + later_ms
            J = A - Theta * T
            if best is None or J > best[0]:
                best = (J, A / T)
        return best[1]

    def round(self, world, slots, rng, rtt, rnd):
        n = len(slots)
        self.cm.cur_b = n
        if self.slow is not None and hasattr(self.slow, "predicted") and not (self.mode == "learned" and rnd < self.cold):
            self._rounds += 1
            if self._rounds % self.shadow_every == 1:
                Theta = self.theta(world, n)
                rtt_est = 0.0 if self.cm.mode == "learned" else rtt   # learned wait already contains RTT
                for W in self.slow.arms:
                    self.slow.predicted(n, W, self.predict_rate(world, slots, W, Theta, rtt_est))
        return super().round(world, slots, rng, rtt, rnd)


class GuidedSlowLoop2:
    """Model ranking + measurement check: switch to the model's best arm if its predicted rate beats the
    current arm by `margin`; if the new arm's measured rate over 2 windows is worse than the previous arm's
    last measured rate by more than `revert`, switch back and ban the arm for `ban` windows."""

    def __init__(self, arms=(4, 8, 16), win=30, margin=0.03, revert=0.02, ban=20, start=8):
        self.arms, self.win, self.margin, self.revert, self.ban, self.start = arms, win, margin, revert, ban, start
        self.ctx = {}
        self.switches = 0

    def _st(self, b):
        k = 1 if b <= 1 else (4 if b <= 4 else 8)
        if k not in self.ctx:
            self.ctx[k] = dict(cur=self.start, prev=None, pred={}, meas={a: [] for a in self.arms}, banned={},
                               since=0, tok=0.0, t=0.0, n=0, windows=0)
        return self.ctx[k]

    def W(self, b):
        return self._st(b)["cur"]

    def predicted(self, b, W, rate):
        s = self._st(b)
        s["pred"][W] = rate if W not in s["pred"] else 0.8 * s["pred"][W] + 0.2 * rate

    def feedback(self, b, tok, t):
        s = self._st(b)
        s["tok"] += tok; s["t"] += t; s["n"] += 1
        if s["n"] < self.win:
            return
        cur = s["cur"]
        s["meas"][cur].append(s["tok"] / s["t"])
        s["tok"] = s["t"] = 0.0; s["n"] = 0; s["windows"] += 1; s["since"] += 1
        # measurement check after a switch
        if s["prev"] is not None and s["since"] == 2 and s["meas"][s["prev"]]:
            new = np.mean(s["meas"][cur][-2:]); old = s["meas"][s["prev"]][-1]
            if new < (1 - self.revert) * old:
                s["banned"][cur] = s["windows"] + self.ban
                s["cur"], s["prev"], s["since"] = s["prev"], None, 0
                self.switches += 1
                return
        cand = {a: v for a, v in s["pred"].items() if s["banned"].get(a, -1) < s["windows"]}
        if not cand or cur not in s["pred"]:
            return
        best = max(cand, key=cand.get)
        if best != cur and cand[best] > (1 + self.margin) * s["pred"][cur]:
            s["prev"], s["cur"], s["since"] = cur, best, 0
            self.switches += 1
