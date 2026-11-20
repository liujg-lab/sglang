"""Round 44 (exploratory, not pre-registered): symmetric W rule from benefit and cost, on the measured-cost world
and the per-(W, g) layer fit of round 43. Synthetic world; not a measurement.

On every ordinary round at the served width W, with the round's K (all masses are c_i-weighted Draft path mass):
  loss_down = topK(tree) - topK(tree cut to W/2, descendants removed)              (exact cut)
  bind_i    = some layer uses all W slots and its best cut candidate (rank W+1) >= theta_K
  gain_up   = (number of binding trees) * g_hat[bk, W]                             (g_hat from real 2W rounds)
              before 5 calibrated binding trees: first-order U_up (no descendants), a lower bound
  V_up   = gain_up  - Theta_b * dT(W -> 2W)        V_down = Theta_b * dT(W/2 -> W) - loss_down
  dT from the per-(W, g) layer fit over the first layer and this round's layer rows.
EMA (step max(1/200, 1/n)); judged every 20 eligible rounds after a 200-round dwell and 100 samples:
  move to the side with the larger V when it exceeds MARGIN * EMA(round tokens); else stay.
Calibration rounds (1 in 100 eligible) build 2W: g_hat = sum(topK(2W tree) - topK(W cut)) / sum(binding trees at
W on the cut), forgetting 0.999.
"""
import ext31 as M31
from ext31 import FORGET, bbucket, narrower, wider, mass
from ext33 import SigW, EMA_N, DWELL, JUDGE_EVERY
from ext34 import GFit

G_MIN_BIND = 5


def _bind_and_mass(v, kept, L, K, W):
    """On the tree restricted to `kept` columns (or the whole tree when kept is None):
    (binding, topK mass, theta_K). Cut candidates come from the next-ranked nodes the 2W tree holds, or from
    the recorded rank W+1..2W candidates of a tree built at W."""
    sc = v.t.score
    layers = v.layers[:L] if kept is None else kept[:L]
    nodes = [n for lay in layers for n in lay]
    order = sorted(nodes, key=lambda n: sc[n], reverse=True)
    kset = set(order[:K])
    theta_k = sc[order[K - 1]] if len(order) >= K else 0.0
    top = mass([sc[n] for n in order], K)
    bind = False
    if kept is None:
        for d, lay in enumerate(layers):
            if len(lay) >= W and sum(1 for n in lay if n in kset) >= W:
                exc = v.t.excl[d] if d < len(v.t.excl) else []
                if exc and exc[0] >= theta_k:
                    bind = True
                    break
    else:
        par = M31.parents_of(v.t)
        prev = {0}
        for d, lay in enumerate(layers):
            ks = set(lay)
            if len(lay) >= W and sum(1 for n in lay if n in kset) >= W:
                cut = [sc[n] for n in v.layers[d] if par.get(n) in prev and n not in ks]
                if cut and max(cut) >= theta_k:
                    bind = True
                    break
            prev = ks
    return bind, top, theta_k


class SatD(SigW):
    def __init__(self, margin=0.005, **kw):
        super().__init__(**kw)
        self.margin = float(margin)
        self.d_st = {}
        self.ghat = {}
        self.d_log = {}

    def _g(self, bk, W):
        a, nb = self.ghat.get((bk, W), (0.0, 0.0))
        if nb < G_MIN_BIND:
            return None
        return a / nb

    def _calibrate(self, pv, served, K):
        bk = bbucket(pv["b"])
        cs = pv["snap"]["ratios"]
        act = 0.0
        nb = 0
        zero_gain_bind_free = 0.0
        for i, v in enumerate(pv["views"]):
            L = len(v.layers)
            par = M31.parents_of(v.t)
            kept = M31.truncate(v, par, served)
            bind, base, _ = _bind_and_mass(v, kept, L, K, served)
            full = mass(v.scores_upto(L), K)
            gain = cs[i] * (full - base)
            act += gain
            nb += int(bind)
            if not bind:
                zero_gain_bind_free += gain
        a, n = self.ghat.get((bk, served), (0.0, 0.0))
        self.ghat[(bk, served)] = (a * FORGET + act, n * FORGET + nb)
        lg = self.d_log.setdefault(f"cal {bk}:{served}", [0, 0.0, 0])
        lg[0] += 1; lg[1] += zero_gain_bind_free; lg[2] += nb

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
        nar, wid = narrower(W), wider(W)
        n_bind = 0
        loss = 0.0
        fo_up = 0.0
        for i, v in enumerate(pv["views"]):
            L = len(v.layers)
            bind, base, _ = _bind_and_mass(v, None, L, K, W)
            n_bind += int(bind)
            if nar is not None:
                par = M31.parents_of(v.t)
                kept = M31.truncate(v, par, nar)
                loss += cs[i] * (base - mass(M31.kept_scores(v, kept, L), K))
            allsc = v.scores_upto(L)
            exc = [s for lay in v.t.excl[:L] for s in lay]
            fo_up += cs[i] * (mass(sorted(allsc + exc, reverse=True), K) - base)
        g = self._g(bk, W)
        gain_up = n_bind * g if g is not None else fo_up
        s = self.d_st.setdefault(bk, dict(n=0, nu=0, nd=0, up=0.0, down=0.0, tok=0.0, since=0))
        s["n"] += 1
        s["since"] += 1
        s["tok"] += max(1.0 / EMA_N, 1.0 / s["n"]) * (float(sum(pv["com"])) - s["tok"])
        dcu = None if wid is None else self._dc(b, pv["suml"], pv["rows"], W, wid)
        if dcu is not None:
            s["nu"] += 1
            s["up"] += max(1.0 / EMA_N, 1.0 / s["nu"]) * ((gain_up - dcu) - s["up"])
        dcd = None if nar is None else self._dc(b, pv["suml"], pv["rows"], nar, W)
        if dcd is not None:
            s["nd"] += 1
            s["down"] += max(1.0 / EMA_N, 1.0 / s["nd"]) * ((dcd - loss) - s["down"])
        lg = self.d_log.setdefault(f"{bk}:{W}", [0, 0.0, 0.0, 0.0])
        lg[0] += 1; lg[1] += n_bind / max(len(pv["views"]), 1); lg[2] += gain_up; lg[3] += loss
        if s["n"] % JUDGE_EVERY or s["since"] < DWELL or s["n"] < EMA_N // 2:
            return
        cands = []
        if wid is not None and s["nu"] >= EMA_N // 2:
            cands.append((s["up"], wid))
        if nar is not None and s["nd"] >= EMA_N // 2:
            cands.append((s["down"], nar))
        if not cands:
            return
        v_best, tgt = max(cands)
        if v_best > self.margin * s["tok"]:
            self.persist[bk] = tgt
            self.switch_log.append(("satd", self.rounds, bk, W, tgt, round(s["up"], 3), round(s["down"], 3),
                                    round(s["tok"], 2)))
            self.d_st[bk] = dict(n=0, nu=0, nd=0, up=0.0, down=0.0, tok=s["tok"], since=0)


class SatDG(GFit, SatD):
    def _sat_out(self):
        out = {}
        for k, v in self.d_log.items():
            if k.startswith("cal"):
                out[k] = [v[0], round(v[1], 3), v[2]]
            else:
                out[k] = [v[0], round(v[1] / max(v[0], 1), 3), round(v[2] / max(v[0], 1), 3),
                          round(v[3] / max(v[0], 1), 3)]
        out["ghat"] = {f"{bk}:{w}": (round(a / nb, 3) if nb else None, round(nb, 1))
                       for (bk, w), (a, nb) in self.ghat.items()}
        return out


ARMS = {
    "satd_g": (SatDG, dict(margin=0.005)),
    "satd15_g": (SatDG, dict(margin=0.015)),
}
