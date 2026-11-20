"""Round 43 (exploratory, not pre-registered): measured Draft layer cost, the per-(W, g) layer fit, and a
saturation-rate W arm. Synthetic world; not a measurement.

World ("measured"): the Draft layer time comes from the 2026-11-19-a table (RTX 3090, 2B INT8, CUDA event median,
_run_batch). Per W, a linear fit in the per-request prefix p:
  g = 1 (bs=1 graphs, cascade attention):  t = a1(W) + s1(W) * p
  g >= 2 (plain extend attention):         t = a_g(W) + s4(W) * mean_p * waves(m)
  a_2 = a1 + (a4 - a1) / 3, a_8 = a4 + (a4 - a1) * 4 / 3, waves(m) = ceil(16 m / 82) (16 q heads, 82 SMs).
The b = 2 and b = 8 rows and the wave rule are assumptions; only b = 1 and b = 4 were measured.
The existing cost-change hooks scale d0, d_req, d_tok together; the measured time is scaled by d0 / d0_start.

Controller layer fit (GFit): one forgetting regression per (W, g) on [1, sum_ell], ridge (0, 128^2), at least
4 samples, span check per cell. kappa(m) is the sum_ell slope of the cell of m. keep_by_net_benefit ranks the
requests once per distinct slope. The old fit ([1, pow2(m), sum_ell] per W) is the "code" arm.

SatW: served W per b bucket. On every ordinary round at the served width, with the round's K:
  per tree, occupancy of layer d = verified nodes in layer d; saturated when some layer uses all W slots;
  binding when, at a saturated layer, the best cut candidate (rank W+1) scores >= theta_K;
  half when every layer uses at most W / 2 slots (then W / 2 gives the same verified set).
  bind = share of trees that bind; half = 1 when every tree is half.
EMA (step max(1/200, 1/n)), judged every 20 eligible rounds after a 200-round dwell and 100 samples:
  widen  (W < 16) when EMA(bind) > BIND_UP   [satm: and EMA(net_up) > 1% of round tokens]
  narrow           when EMA(half) > HALF_DOWN [satm: and EMA(net_down) > 0]
"""
import math

import numpy as np

import ext31 as M31
from ext31 import FORGET, LAYER_MIN, bbucket, gbucket, narrower, wider, mass
from ext33 import SigW, FixW, EMA_N, DWELL, JUDGE_EVERY, CodeMirror

# ------------------------------------------------------------------ measured world
P = [256, 1024, 2048, 4096, 8192]
B1 = {2: [4000, 4137, 4300, 4581, 5114], 4: [4086, 4226, 4392, 4675, 5219], 8: [4195, 4339, 4479, 4784, 5285],
      16: [4348, 4464, 4628, 4917, 5448], 32: [4991, 5220, 5442, 5884, 6770], 64: [6488, 6856, 7138, 8018, 9856]}
B4 = {2: [5405, 7700, 10696, 16654, 29187], 4: [5470, 7716, 10649, 16500, 29389], 8: [5607, 7877, 10792, 17327, 29442],
      16: [6017, 8319, 11250, 17943, 30289], 32: [7740, 10078, 13308, 20283, 33221],
      64: [11252, 13885, 18768, 25076, 38876]}


def _lin(ys):
    x = np.array(P, float)
    A = np.stack([np.ones_like(x), x], 1)
    a, s = np.linalg.lstsq(A, np.array(ys, float) / 1000.0, rcond=None)[0]
    return float(a), float(s)


FIT1 = {w: _lin(v) for w, v in B1.items()}
FIT4 = {w: _lin(v) for w, v in B4.items()}


def measured_ms(m, W, suml):
    m = max(int(m), 1)
    a1, s1 = FIT1[W]
    if m == 1:
        return a1 + s1 * float(suml)
    a4, s4 = FIT4[W]
    g = gbucket(m)
    if g == 2:
        a = a1 + (a4 - a1) / 3.0
    elif g == 4:
        a = a4
    else:
        a = a4 + (a4 - a1) * (g - 4) / 3.0
    waves = math.ceil(16 * m / 82.0)
    return a + s4 * (float(suml) / m) * waves


def measure_world(w):
    base = float(w.d0)
    w.c_d_l = lambda n, W, suml: measured_ms(n, W, suml) * (float(w.d0) / base)


# ------------------------------------------------------------------ keep_by_net_benefit with kappa(m)
def keep_by_net_benefit(rows, theta, times, layer_cost, kappa):
    growing = [r for r in rows if r["growing"]]
    if not growing:
        return []
    if callable(kappa):
        slope_of = kappa
    else:
        const = 0.0 if kappa is None else float(kappa)
        slope_of = lambda _m: const
    joint, gains = {}, {}
    for k, spent in times.items():
        if spent is None:
            continue
        ben = 0.0
        per = {}
        for idx, r in enumerate(rows):
            ben += r["c"] * r["e"].get(k, 0.0)
            per[idx] = r["c"] * r["g"].get(k, 0.0)
        joint[k] = ben - theta * spent
        gains[k] = per
    if not joint:
        return None
    stop_value = max(joint.values())
    best = None
    any_cost = False
    gidx = [i for i, r in enumerate(rows) if r["growing"]]
    for k, base in joint.items():
        orders = {}
        for m in range(1, len(gidx) + 1):
            s = slope_of(m)
            s = 0.0 if s is None else float(s)
            if s not in orders:
                ranked = []
                for i in gidx:
                    g = gains[k][i]
                    ell = rows[i]["ell"]
                    ranked.append((g - theta * s * ell, g, ell, rows[i]["item"]))
                ranked.sort(key=lambda x: x[0], reverse=True)
                rg, re_ = [0.0], [0]
                for _n, g, ell, _it in ranked:
                    rg.append(rg[-1] + g)
                    re_.append(re_[-1] + ell)
                orders[s] = (ranked, rg, re_)
            ranked, rg, re_ = orders[s]
            cost = layer_cost(m, re_[m])
            if cost is None:
                continue
            any_cost = True
            val = base + rg[m] - theta * cost
            choice = (val, k, m)
            if best is None or choice > best[0]:
                best = (choice, [x[3] for x in ranked[:m]])
    if not any_cost or best is None:
        return None
    if best[0][0] > stop_value:
        return list(best[1])
    return []


M31.keep_by_net_benefit = keep_by_net_benefit     # identical for a numeric kappa, so the code arm is unchanged


# ------------------------------------------------------------------ per-(W, g) layer fit
class Reg2:
    def __init__(self):
        self.A = np.zeros((2, 2))
        self.y = np.zeros(2)
        self.n = 0
        self.lo = self.hi = None
        self._c = None

    def add(self, sl, t):
        x = np.array([1.0, float(sl)])
        self.A = FORGET * self.A + np.outer(x, x)
        self.y = FORGET * self.y + x * float(t)
        self.n += 1
        self._c = None
        self.lo = sl if self.lo is None else min(self.lo, sl)
        self.hi = sl if self.hi is None else max(self.hi, sl)

    def coef(self):
        if self.n < LAYER_MIN:
            return None
        if self._c is None:
            S = self.A.copy()
            S[1, 1] += 128.0 ** 2
            try:
                self._c = np.linalg.solve(S, self.y)
            except np.linalg.LinAlgError:
                return None
        return self._c

    def pred(self, sl):
        c = self.coef()
        if c is None:
            return None
        pad = max(128.0, self.hi - self.lo)
        if sl < self.lo - pad or sl > self.hi + pad:
            return None
        v = float(c[0] + c[1] * sl)
        return v if v > 0 else None

    def slope(self):
        c = self.coef()
        return None if c is None else float(c[1])


class GFit:
    """Mixin: the layer model is per (W, g); kappa is per g."""

    def _cur_g(self):
        return gbucket(getattr(self, "_round_b", 1))

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        self._round_b = len(slots)
        return super().round(world, slots, rng, rtt, rnd, trees_out)

    def _layer_add(self, w, m, sl, dt):
        self.reg.setdefault((w, gbucket(m)), Reg2()).add(sl, dt)

    def _pred_layer(self, w, m, sl):
        r = self.reg.get((w, gbucket(m)))
        return None if r is None else r.pred(sl)

    def _slope(self, w):
        def kappa(m, w=w):
            r = self.reg.get((w, gbucket(m)))
            return None if r is None else r.slope()
        return kappa

    def _layer_n(self, w):
        r = self.reg.get((w, self._cur_g()))
        return 0 if r is None else r.n


# ------------------------------------------------------------------ saturation arm
class SatW(SigW):
    def __init__(self, bind_up=0.25, half_down=0.9, magnitude=False, **kw):
        super().__init__(**kw)
        self.bind_up, self.half_down, self.magnitude = float(bind_up), float(half_down), bool(magnitude)
        self.sat_st = {}
        self.sat_log = {}

    def _signals(self, pv):
        plan = pv["plan"]
        b = pv["b"]
        bk = bbucket(b)
        W = plan["W"]
        K = pv["K"]
        if pv["snap"] is None or plan["price"] is None:
            return
        served = self._persist(b)
        if plan["kind"] == "cal" and W > served:
            self._calibrate(pv, served, K)
            return
        if plan["kind"] != "ordinary" or W != served:
            return
        cs = pv["snap"]["ratios"]
        n_bind = 0
        all_half = True
        up = down = 0.0
        nar = narrower(W)
        for i, v in enumerate(pv["views"]):
            L = len(v.layers)
            sc = v.t.score
            nodes = [n for lay in v.layers[:L] for n in lay]
            order = sorted(nodes, key=lambda n: sc[n], reverse=True)
            kset = set(order[:K])
            theta_k = sc[order[K - 1]] if len(order) >= K else 0.0
            bind = False
            for d, lay in enumerate(v.layers[:L]):
                occ = sum(1 for n in lay if n in kset)
                if occ > W // 2:
                    all_half = False
                if occ >= W and len(lay) >= W:
                    exc = v.t.excl[d] if d < len(v.t.excl) else []
                    if exc and exc[0] >= theta_k:
                        bind = True
            n_bind += int(bind)
            if self.magnitude:
                allsc = v.scores_upto(L)
                base = mass(allsc, K)
                exc_all = [s for lay in v.t.excl[:L] for s in lay]
                up += cs[i] * (mass(sorted(allsc + exc_all, reverse=True), K) - base)
                if nar is not None:
                    par = M31.parents_of(v.t)
                    kept = M31.truncate(v, par, nar)
                    down += cs[i] * (base - mass(M31.kept_scores(v, kept, L), K))
        bind_share = n_bind / max(len(pv["views"]), 1)
        s = self.sat_st.setdefault(bk, dict(n=0, bind=0.0, half=0.0, tok=0.0, up=0.0, down=0.0, nu=0, nd=0,
                                            since=0))
        s["n"] += 1
        s["since"] += 1
        step = max(1.0 / EMA_N, 1.0 / s["n"])
        s["bind"] += step * (bind_share - s["bind"])
        s["half"] += step * (float(all_half) - s["half"])
        s["tok"] += step * (float(sum(pv["com"])) - s["tok"])
        lg = self.sat_log.setdefault(f"{bk}:{W}", [0, 0.0, 0.0])
        lg[0] += 1; lg[1] += bind_share; lg[2] += float(all_half)
        wid = wider(W)
        if self.magnitude:
            up *= self._r_up(bk)
            dcu = None if wid is None else self._dc(b, pv["suml"], pv["rows"], W, wid)
            if dcu is not None:
                s["nu"] += 1
                s["up"] += max(1.0 / EMA_N, 1.0 / s["nu"]) * ((up - dcu) - s["up"])
            dcd = None if nar is None else self._dc(b, pv["suml"], pv["rows"], nar, W)
            if dcd is not None:
                s["nd"] += 1
                s["down"] += max(1.0 / EMA_N, 1.0 / s["nd"]) * ((dcd - down) - s["down"])
        if s["n"] % JUDGE_EVERY or s["since"] < DWELL or s["n"] < EMA_N // 2:
            return
        tgt = None
        if wid is not None and wid <= 16 and s["bind"] > self.bind_up:
            if not self.magnitude or (s["nu"] >= EMA_N // 2 and s["up"] > 0.01 * s["tok"]):
                tgt = wid
        if tgt is None and nar is not None and s["half"] > self.half_down:
            if not self.magnitude or (s["nd"] >= EMA_N // 2 and s["down"] > 0.0):
                tgt = nar
        if tgt is not None:
            self.persist[bk] = tgt
            self.switch_log.append(("sat", self.rounds, bk, W, tgt, round(s["bind"], 3), round(s["half"], 3)))
            self.sat_st[bk] = dict(n=0, bind=0.0, half=0.0, tok=s["tok"], up=0.0, down=0.0, nu=0, nd=0, since=0)


class CodeG(GFit, CodeMirror):
    pass


class SigG(GFit, SigW):
    pass


class SatG(GFit, SatW):
    pass


class FixG(GFit, FixW):
    pass


ARMS = {
    "code": (CodeMirror, dict()),
    "code_g": (CodeG, dict()),
    "sig_g": (SigG, dict()),
    "sat_g": (SatG, dict()),
    "sat10_g": (SatG, dict(bind_up=0.10)),
    "satm_g": (SatG, dict(magnitude=True)),
    "fix2": (FixG, dict(fixed=2)),
    "fix4": (FixG, dict(fixed=4)),
    "fix8": (FixG, dict(fixed=8)),
    "fix16": (FixG, dict(fixed=16)),
}
