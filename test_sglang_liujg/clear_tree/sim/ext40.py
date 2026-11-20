"""Round 48 (exploratory, not pre-registered): bias of counterfactual W estimates. Synthetic world; not a measurement.

Policy: fixed W, ledger on, with the world's true (noise-free) layer cost as its layer model for every width, so
cost-model error is out of the picture. Truth: rate(W_lo) / rate(W_hi) - 1 from two such fixed runs.
In the W_hi run, on every value-rule round, W_lo = W_hi / 2 is estimated three ways (expected tokens = sum c_i
topK mass + 1 per request; model time = layer cost of the first layer and every grown layer + verify table T(K)):
  a   cut the real tree to W_lo (descendants removed), same depth, the round's K
  b   cut the real tree to W_lo, same depth, K chosen by the policy's own _select_k for W_lo
  c   shadow superset: a separate tree set built at W_hi to the full depth (MAXD - 1) for every request (own rng);
      replay the policy's W_lo growth (_choose_keep with its lagged forecasts, Theta, c_i) and _select_k on the cut
      of that tree; reference = the same replay at W_hi on the uncut superset (paired).
For a and b the reference is the real W_hi round (expected tokens at its K, model time).
Estimate of the ratio = (sum E_lo / sum T_lo) / (sum E_hi / sum T_hi) - 1 over the run.
"""
from collections import defaultdict

import numpy as np

import ext2 as X
import ext31 as M31
from ext31 import MAXD, mass, TreeView
from ext33 import FixW


class CutView(TreeView):
    """Reveals the layers of a source view cut to width w (descendants of dropped nodes removed)."""

    def __init__(self, src, w):
        super().__init__(src.t)
        self.src, self.w = src, int(w)
        self.par = M31.parents_of(src.t)
        self.prev = {0}

    def can_reveal(self):
        return len(self.layers) < len(self.src.layers)

    def reveal(self):
        d = len(self.layers)
        sc = self.t.score
        cand = [n for n in self.src.layers[d] if self.par.get(n) in self.prev]
        cand.sort(key=lambda n: sc[n], reverse=True)
        lay = cand[: self.w]
        self.layers.append(lay)
        self.prev = set(lay)


class TCFix(FixW):
    def __init__(self, est=True, **kw):
        super().__init__(**kw)
        self.est = bool(est) and self.fixed > 2
        self.acc = defaultdict(float)
        self._rng2 = np.random.default_rng(987654321)
        self._world = None

    # true layer cost as the policy's layer model
    def _pred_layer(self, w, m, sl):
        return float(self._world.c_d_l(m, w, sl)) if self._world is not None else None

    def _layer_add(self, w, m, sl, dt):
        return

    def _layer_n(self, w):
        return 99

    def _slope(self, w):
        def kappa(m, w=w):
            if self._world is None:
                return None
            return (self._world.c_d_l(m, w, 2000.0) - self._world.c_d_l(m, w, 1000.0)) / 1000.0
        return kappa

    def _plan(self, b, max_ell, suml):
        self._suml_cur = float(suml)
        return super()._plan(b, max_ell, suml)

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        self._world = world
        out = super().round(world, slots, rng, rtt, rnd, trees_out)
        pv = self.prev
        if self.est and pv is not None and self.ledger and not pv["plan"]["use_tau"] and pv["snap"] is not None \
                and pv["plan"]["price"] is not None:
            self._estimate(pv, world, slots)
        return out

    # ------------------------------------------------------------ helpers
    def _model_time(self, W, b, suml, rows, K, max_ell):
        tv = self._time(b, max_ell, K)
        if tv is None:
            return None
        return float(self._world.c_d_l(b, W, suml)) + sum(float(self._world.c_d_l(m, W, sl)) for m, sl in rows) + tv

    def _plan_for(self, plan, W, b, max_ell, suml):
        fitting = self._fitting(max_ell)
        pl = dict(plan)
        pl["W"] = W
        pl["cover"] = self._cover(W, fitting)
        pl["layer_ms"] = self._pred_layer(W, b, suml)
        pl["locked"] = None
        return pl

    def _cf(self, src_views, Wc, plan, cs, ell, b, max_ell, suml):
        pl = self._plan_for(plan, Wc, b, max_ell, suml)
        n = len(src_views)
        cv = [CutView(sv, Wc) for sv in src_views]
        for v in cv:
            v.reveal()
        depth = [1] * n
        active = list(range(n))
        rows = []
        while active:
            elig = [i for i in active if depth[i] < MAXD - 1 and cv[i].can_reveal()]
            keep = self._choose_keep(cv, depth, elig, pl, cs, ell, b, max_ell) if elig else []
            keep = [i for i in keep if cv[i].can_reveal()]
            if not keep:
                break
            rows.append((len(keep), float(sum(ell[i] for i in keep))))
            for i in keep:
                cv[i].reveal()
                depth[i] += 1
            active = keep
        K = self._select_k(cv, depth, pl, cs, b, max_ell)
        E = sum(cs[i] * mass(cv[i].scores_upto(depth[i]), K) for i in range(n)) + n
        T = self._model_time(Wc, b, suml, rows, K, max_ell)
        return E, T, K, sum(depth) / n

    def _estimate(self, pv, world, slots):
        plan = pv["plan"]
        b = pv["b"]
        W = plan["W"]
        lo = W // 2
        K = pv["K"]
        cs = pv["snap"]["ratios"]
        max_ell = pv["max_ell"]
        ell = [world.reqs[r]["p0"] + j for r, j in slots]
        suml = self._suml_cur
        n = len(pv["views"])
        a = self.acc
        # reference: the real round
        Th = self._model_time(W, b, suml, pv["rows"], K, max_ell)
        if Th is None:
            return
        Eh = sum(c * e for c, e in zip(cs, pv["e_ver"])) + n
        # a: cut, same depth, same K
        Ea = 0.0
        for i, v in enumerate(pv["views"]):
            L = len(v.layers)
            kept = M31.truncate(v, M31.parents_of(v.t), lo)
            Ea += cs[i] * mass(M31.kept_scores(v, kept, L), K)
        Ea += n
        Ta = self._model_time(lo, b, suml, pv["rows"], K, max_ell)
        # b: cut, same depth, own K
        pl = self._plan_for(plan, lo, b, max_ell, suml)
        cvs = []
        for v in pv["views"]:
            c = CutView(v, lo)
            while c.can_reveal():
                c.reveal()
            cvs.append(c)
        dep = [len(c.layers) for c in cvs]
        Kb = self._select_k(cvs, dep, pl, cs, b, max_ell)
        Eb = sum(cs[i] * mass(cvs[i].scores_upto(dep[i]), Kb) for i in range(n)) + n
        Tb = self._model_time(lo, b, suml, pv["rows"], Kb, max_ell)
        # c: shadow superset at W to the full depth, replay W_lo and W
        params = (world.a, world.g, world.beta, 0.0)
        sv = []
        for r, j in slots:
            t = X.Tree2(world.reqs[r], j, world)
            v = TreeView(t)
            for _ in range(MAXD - 1):
                t.expand(W, self._rng2, params)
                v.add_layer()
            sv.append(v)
        Ec, Tc, Kc, dc = self._cf(sv, lo, plan, cs, ell, b, max_ell, suml)
        Ech, Tch, Kch, dch = self._cf(sv, W, plan, cs, ell, b, max_ell, suml)
        if None in (Ta, Tb, Tc, Tch):
            return
        a["n"] += 1
        a["tok"] += float(sum(pv["com"]))
        for k, v in (("Eh", Eh), ("Th", Th), ("Ea", Ea), ("Ta", Ta), ("Eb", Eb), ("Tb", Tb), ("Ec", Ec), ("Tc", Tc),
                     ("Ech", Ech), ("Tch", Tch)):
            a[k] += v
        a["Kh"] += K; a["Kb"] += Kb; a["Kc"] += Kc; a["Kch"] += Kch
        a["dh"] += sum(len(v.layers) for v in pv["views"]) / n
        a["dc"] += dc; a["dch"] += dch
        a["kb_ne"] += float(Kb != K)
        a["kc_ne"] += float(Kc != Kch)

    def out(self):
        return dict(self.acc)


ARMS = {f"tc{w}": (TCFix, dict(fixed=w)) for w in (2, 4, 8, 16)}
