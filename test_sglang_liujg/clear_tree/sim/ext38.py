"""Round 46 (exploratory, not pre-registered): W indicator screening. Synthetic world; not a measurement.

Fixed-W arms (same rules as FixG) that also log candidate W indicators. Per b bucket sums, two sets:
"<bk>a" over all rounds and "<bk>v" over value-rule rounds only (tau rounds stop by the tau rule):
  bind     trees with a full layer whose best cut candidate (rank W+1) >= theta_K          (sat_g's signal)
  half_r   rounds where every tree uses <= W/2 verified nodes in every layer                (sat_g's down signal)
  need     per tree: max over layers of verified nodes in the layer (W+1 if bind) -> histogram
  util     sum min(K, nodes) / sum nodes                                                    (build utilisation)
  rho      Target token at the first position == Draft top-1                                (acceptance class)
  stop     why the accepted chain stopped (per tree):
             vocab  Target token outside the Draft's C candidates
             kmiss  true child built but not in the verify set
             depth  stop node was on the last built layer
             wk     true child cut by the beam with score >= theta_K (would have been verified)  [width miss]
             wcut   true child cut by the beam, score < theta_K
  rank_hi  accepted nodes ranked > W/2 in their layer (by score)
  rup, rdn cost of widening / saving of narrowing, Theta_b * true layer time difference / round tokens
  fo_up    first-order up gain (cut candidates added, c_i weighted) / round tokens
  loss_dn  cut-to-W/2 loss (descendants removed, c_i weighted) / round tokens
"""
from collections import defaultdict

import ext31 as M31
from ext31 import bbucket, narrower, wider, mass
from ext33 import FixW
from ext34 import GFit


class IndFix(GFit, FixW):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.ind = {}
        self.ind_tau = defaultdict(int)

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        self._world = world
        out = super().round(world, slots, rng, rtt, rnd, trees_out)
        pv = self.prev
        if pv is not None and self.ledger:
            self._log(pv, slots, world)
        return out

    def _acc(self, bk):
        a = self.ind.get(bk)
        if a is None:
            a = self.ind[bk] = defaultdict(float)
            a["need"] = defaultdict(int)
        return a

    def _log(self, pv, slots, world):
        plan = pv["plan"]
        b = pv["b"]
        bk = bbucket(b)
        if plan["use_tau"]:
            self.ind_tau[bk] += 1
        keys = [f"{bk}a"] + ([] if plan["use_tau"] else [f"{bk}v"])
        for key in keys:
            self._log1(pv, world, self._acc(key))

    def _log1(self, pv, world, a):
        plan = pv["plan"]
        b = pv["b"]
        W = plan["W"]
        K = pv["K"]
        tok = float(sum(pv["com"]))
        a["rounds"] += 1
        a["tok"] += tok
        a["ms"] += pv["t_round"]
        a["trees"] += len(pv["views"])
        cs = pv["snap"]["ratios"] if pv["snap"] is not None else [1.0] * len(pv["views"])
        all_half = True
        nar = narrower(W)
        fo_up = loss_dn = 0.0
        for i, v in enumerate(pv["views"]):
            t = v.t
            L = len(v.layers)
            sc = t.score
            nodes = [n for lay in v.layers[:L] for n in lay]
            order = sorted(nodes, key=lambda n: sc[n], reverse=True)
            kset = set(order[:K])
            theta = sc[order[K - 1]] if len(order) >= K else 0.0
            bind = False
            need = 0
            rank = {}
            for d, lay in enumerate(v.layers[:L]):
                occ = sum(1 for n in lay if n in kset)
                need = max(need, occ)
                if occ > W // 2:
                    all_half = False
                if occ >= W and len(lay) >= W:
                    exc = t.excl[d] if d < len(t.excl) else []
                    if exc and exc[0] >= theta:
                        bind = True
                for r_, n in enumerate(sorted(lay, key=lambda n: sc[n], reverse=True)):
                    rank[n] = r_ + 1
            a["bind"] += bind
            a["need"][W + 1 if bind else need] += 1
            a["nodes"] += len(nodes)
            a["used"] += min(K, len(nodes))
            # first position
            H = t.req["H"]
            p0 = t.pos[0]
            if p0 >= 0:
                a["rho_n"] += 1
                a["rho"] += float(H[p0] == 0)
            # accepted chain and why it stopped
            ver = t.topk(K, False)
            v_ = 0
            n_acc = 0
            while True:
                p = t.pos[v_]
                h = H[p] if p >= 0 else -1
                if h < 0:
                    cause = "vocab"
                    break
                c = t.kids[v_].get(int(h))
                if c is not None and c in ver:
                    n_acc += 1
                    if rank.get(c, 0) > W // 2:
                        a["rank_hi"] += 1
                    v_ = c
                    continue
                if c is not None:
                    cause = "kmiss"
                    break
                d = t.dep[v_]
                if d >= L:
                    cause = "depth"
                    break
                q = world.reqs[pv["rids"][i]]["Q"][p]
                s_true = sc[v_] * float(q[int(h)])
                cause = "wk" if s_true >= theta and theta > 0 else "wcut"
                break
            a["acc"] += n_acc
            a["stop_" + cause] += 1
            # first-order up gain and cut-to-W/2 loss (Draft mass, c_i weighted)
            allsc = v.scores_upto(L)
            base = mass(allsc, K)
            exc_all = [s for lay in t.excl[:L] for s in lay]
            fo_up += cs[i] * (mass(sorted(allsc + exc_all, reverse=True), K) - base)
            if nar is not None:
                par = M31.parents_of(t)
                kept = M31.truncate(v, par, nar)
                loss_dn += cs[i] * (base - mass(M31.kept_scores(v, kept, L), K))
        a["half_r"] += all_half
        if tok > 0:
            a["fo_up"] += fo_up / tok
            a["loss_dn"] += loss_dn / tok
        theta_b = self._theta(b)
        if theta_b is not None and tok > 0:
            rows = [(b, self._first_suml)] + list(pv["rows"])
            wid = wider(W)
            if wid is not None:
                d_up = sum(world.c_d_l(m, wid, sl) - world.c_d_l(m, W, sl) for m, sl in rows)
                a["rup"] += theta_b * d_up / tok
                a["rup_n"] += 1
            if nar is not None:
                d_dn = sum(world.c_d_l(m, W, sl) - world.c_d_l(m, nar, sl) for m, sl in rows)
                a["rdn"] += theta_b * d_dn / tok
                a["rdn_n"] += 1

    def _plan(self, b, max_ell, suml):
        self._first_suml = float(suml)
        return super()._plan(b, max_ell, suml)

    def out(self):
        res = {}
        for bk, a in self.ind.items():
            d = {k: v for k, v in a.items() if k != "need"}
            d["need"] = {str(k): v for k, v in sorted(a["need"].items())}
            d["tau_rounds"] = self.ind_tau.get(int(bk[:-1]), 0)
            res[str(bk)] = d
        return res


ARMS = {f"ind{w}": (IndFix, dict(fixed=w)) for w in (2, 4, 8, 16)}
