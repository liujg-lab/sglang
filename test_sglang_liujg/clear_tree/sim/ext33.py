"""Round 42 (exploratory, not pre-registered): slow W from the beam-constraint signals, on the round-40 mirror of
the current code (ext31.CodeMirror). Synthetic world; not a measurement.

SigW replaces the code's W machinery (probe_w, explore every 20, replay arms) with:
  every ordinary round at the served width W, on the built trees and the chosen K:
    up_i   = topK mass(tree + the candidates ranked W+1..2W that each layer cut) - topK mass(tree)
             (first order: descendants of the cut candidates are not counted)
    down_i = topK mass(tree) - topK mass(tree cut to W/2, descendants of dropped nodes removed)
    U_up   = r_up * sum_i c_i up_i,  U_down = sum_i c_i down_i            (expected tokens)
    dC     = Theta_b * sum over (first layer + this round's layer rows) of layer_fit(W') - layer_fit(W)
    net_up = U_up - dC(W, 2W),  net_down = dC(W/2, W) - U_down
  per b bucket: EMA with step max(1/200, 1/n) of net_up, net_down and round tokens.
  every 20 eligible rounds, once 200 rounds have passed since the last switch:
    widen  when EMA(net_up)   > 3% of EMA(tokens)
    narrow when EMA(net_down) > 3% of EMA(tokens)
  calibration rounds (1 in CAL_EVERY eligible rounds) build 2W; on that tree,
    r_up = (sum actual + 0.5) / (sum first-order + 0.5), actual = topK(2W tree) - topK(W cut).
  narrow timing rounds: until W/2 has 4 layer samples, then 2 rounds every 300 eligible rounds.
Diagnostics (sim only, uses the true text): predicted vs true token change for both signals.
FixW: one width for every round, no W machinery. Reference for the best fixed W in hindsight.
"""
import numpy as np

import ext2 as X
import feasibility as F
from ext31 import (CodeMirror, WS, bbucket, narrower, wider, parents_of, truncate, kept_scores, mass,
                   replay_accept, FORGET)

C = F.C
EMA_N = 200
DWELL = 200
JUDGE_EVERY = 20
GAIN = 0.03
CAL_EVERY = 100
NARROW_IDLE = 300


class RecTree2(X.Tree2):
    """Tree2 that also records the candidates each layer cut (ranks W+1..2W). Same choices, same rng use."""

    def __init__(self, req, j, world):
        super().__init__(req, j, world)
        self.excl = []

    def expand(self, W, rng, params):
        F_ = self.frontier
        Q, H = self.req["Q"], self.req["H"]
        rows = []
        for v in F_:
            p = self.pos[v]
            if p >= 0:
                rows.append(Q[p])
            else:
                rows.append(F.gen_dists(rng, self.world.s_at(self.req, self.j + self.dep[v]), 1)[0])
        D = np.stack(rows)
        A = F.ahat(D, *params)
        fs = np.fromiter((self.score[v] for v in F_), float, len(F_))
        fp = np.fromiter((self.pi[v] for v in F_), float, len(F_))
        sc = (fs[:, None] * D).ravel(); pv = (fp[:, None] * A).ravel()
        sel = np.argpartition(-sc, W - 1)[:W] if sc.size > W else np.arange(sc.size)
        if sc.size > W:
            chosen = set(int(i) for i in sel)
            rest = [i for i in np.argsort(-sc) if int(i) not in chosen][:W]
            self.excl.append([float(sc[i]) for i in rest])
        else:
            self.excl.append([])
        new = []
        for flat in sel:
            fi, c = divmod(int(flat), C)
            v = F_[fi]
            nid = len(self.score)
            self.score.append(float(sc[flat])); self.pi.append(float(pv[flat]))
            p = self.pos[v]
            self.pos.append(p + 1 if (p >= 0 and H[p] == c) else -1)
            self.kids.append(dict()); self.kids[v][c] = nid
            self.dep.append(self.dep[v] + 1); self.phi.append(float(D[fi, 0]))
            self.pis.append(float(pv[flat])); self.scores.append(float(sc[flat])); self.ids.append(nid)
            new.append(nid)
        expanded = list(F_)
        self.frontier = new
        return D, expanded


X.Tree2 = RecTree2          # ext31 builds trees through ext2.Tree2; choices are unchanged for every arm

BIG = 10 ** 9


class FixW(CodeMirror):
    def __init__(self, fixed=4, **kw):
        super().__init__(**kw)
        self.fixed = int(fixed)

    def _persist(self, b):
        return self.fixed

    def _choose_w(self, b, eligible, suml):
        return self.fixed, "ordinary"

    def _replay(self, pv):
        return


class SigW(CodeMirror):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.st = {}
        self.cal = {}
        self.n_cal = 0
        self.n_timing = 0
        self.diag = {}
        self._suml = None

    # ------------------------------------------------------------ width per round
    def _choose_w(self, b, eligible, suml):
        base = self._persist(b)
        if not eligible:
            return base, "ordinary"
        bk = bbucket(b)
        n = self.elig_n.get(bk, 0) + 1
        self.elig_n[bk] = n
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
        if wid is not None and n % CAL_EVERY == 0:
            self.n_cal += 1
            return wid, "cal"
        return base, "ordinary"

    def _replay(self, pv):
        return

    # ------------------------------------------------------------ signals
    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        self._suml = float(sum(world.reqs[r]["p0"] + j for r, j in slots))
        out = super().round(world, slots, rng, rtt, rnd, trees_out)
        pv = self.prev
        if pv is not None and self.ledger:
            pv["suml"] = self._suml
            try:
                self._signals(pv)
            except Exception:
                raise
        return out

    def _dc(self, b, suml, rows, lo, hi):
        theta = self._theta(b)
        if theta is None:
            return None
        s = 0.0
        for m, sl in [(b, suml)] + list(rows):
            ph, pl = self._pred_layer(hi, m, sl), self._pred_layer(lo, m, sl)
            if ph is None or pl is None:
                return None
            s += ph - pl
        return theta * s

    def _r_up(self, bk):
        a, p = self.cal.get(bk, (0.0, 0.0))
        return (a + 0.5) / (p + 0.5)

    def _signals(self, pv):
        plan = pv["plan"]
        b = pv["b"]
        bk = bbucket(b)
        W = plan["W"]
        K = pv["K"]
        if pv["snap"] is None or plan["price"] is None:
            return
        cs = pv["snap"]["ratios"]
        served = self._persist(b)
        kind = plan["kind"]
        if kind == "cal" and W > served:
            self._calibrate(pv, served, K)
            return
        if kind != "ordinary" or W != served:
            return
        up = down = 0.0
        t_down = 0.0
        nar = narrower(W)
        for i, v in enumerate(pv["views"]):
            L = len(v.layers)
            allsc = v.scores_upto(L)
            base = mass(allsc, K)
            exc = [s for lay in v.t.excl[:L] for s in lay]
            merged = sorted(allsc + exc, reverse=True)
            up += cs[i] * (mass(merged, K) - base)
            if nar is not None:
                par = parents_of(v.t)
                kept = truncate(v, par, nar)
                down += cs[i] * (base - mass(kept_scores(v, kept, L), K))
                full = [list(l) for l in v.layers]
                t_down += replay_accept(v, full, L, K, BIG) - replay_accept(v, kept, L, K, BIG)
        up *= self._r_up(bk)
        tok = float(sum(pv["com"]))
        s = self.st.setdefault(bk, dict(n=0, nu=0, nd=0, up=0.0, down=0.0, tok=0.0, since=0))
        s["n"] += 1
        s["since"] += 1
        step = max(1.0 / EMA_N, 1.0 / s["n"])
        s["tok"] += step * (tok - s["tok"])
        wid = wider(W)
        dcu = None if wid is None else self._dc(b, pv["suml"], pv["rows"], W, wid)
        if dcu is not None:
            s["nu"] += 1
            st_u = max(1.0 / EMA_N, 1.0 / s["nu"])
            s["up"] += st_u * ((up - dcu) - s["up"])
        dcd = None if nar is None else self._dc(b, pv["suml"], pv["rows"], nar, W)
        if dcd is not None:
            s["nd"] += 1
            st_d = max(1.0 / EMA_N, 1.0 / s["nd"])
            s["down"] += st_d * ((dcd - down) - s["down"])
        dg = self.diag.setdefault(f"{bk}:{nar}<{W}", [0.0, 0.0, 0])
        if nar is not None:
            dg[0] += down; dg[1] += t_down; dg[2] += 1
        if s["n"] % JUDGE_EVERY == 0 and s["since"] >= DWELL:
            tgt = None
            if wid is not None and s["nu"] >= EMA_N // 2 and s["up"] > GAIN * s["tok"]:
                tgt, g = wid, s["up"] / s["tok"]
            elif nar is not None and s["nd"] >= EMA_N // 2 and s["down"] > GAIN * s["tok"]:
                tgt, g = nar, s["down"] / s["tok"]
            if tgt is not None:
                self.persist[bk] = tgt
                self.switch_log.append(("sig", self.rounds, bk, W, tgt, round(float(g), 4)))
                self.st[bk] = dict(n=0, nu=0, nd=0, up=0.0, down=0.0, tok=s["tok"], since=0)

    def _calibrate(self, pv, served, K):
        """Built at 2*served: first-order gain of the served-width cut vs the real gain, both in Draft mass."""
        bk = bbucket(pv["b"])
        cs = pv["snap"]["ratios"]
        act = fo = 0.0
        t_act = p_tok = 0.0
        for i, v in enumerate(pv["views"]):
            L = len(v.layers)
            par = parents_of(v.t)
            kept = truncate(v, par, served)
            ks = kept_scores(v, kept, L)
            base = mass(ks, K)
            full = mass(v.scores_upto(L), K)
            sc = v.t.score
            cut = []
            prev = {0}
            for d, lay in enumerate(v.layers[:len(kept)]):
                kd = set(kept[d])
                cand = sorted((sc[n] for n in lay if par.get(n) in prev and n not in kd), reverse=True)
                cut.extend(cand[:served])
                prev = kd
            fo_i = mass(sorted(ks + cut, reverse=True), K) - base
            act += cs[i] * (full - base)
            fo += cs[i] * fo_i
            fl = [list(l) for l in v.layers]
            t_act += replay_accept(v, fl, L, K, BIG) - replay_accept(v, kept, L, K, BIG)
        a, p = self.cal.get(bk, (0.0, 0.0))
        self.cal[bk] = (a * FORGET + act, p * FORGET + fo)
        p_tok = fo * self._r_up(bk)
        dg = self.diag.setdefault(f"{bk}:{served}>{2 * served}", [0.0, 0.0, 0])
        dg[0] += p_tok; dg[1] += t_act; dg[2] += 1


ARMS = {
    "code": (CodeMirror, dict()),
    "sig": (SigW, dict()),
    "fix2": (FixW, dict(fixed=2)),
    "fix4": (FixW, dict(fixed=4)),
    "fix8": (FixW, dict(fixed=8)),
}
