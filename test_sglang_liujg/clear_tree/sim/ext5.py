"""Simulation round 6 (review 13): cost-table context, K_g cache scope, delta decay, switch timing basis,
stop-boundary prediction error.

CostModelLE   wait time = common level L(t) + effect g(b, K[, prefix regime]). L is updated every round (fast),
              g from the residual against the current L (slow), so an RTT change moves L and is not charged to
              whichever K happened to be measured at the time. Only differences of g enter the K choice.
              Optionally keyed by the prefix regime (Draft layer cost too) and flags level jumps.
CT5           CT3 (core v1) with: K_g re-selection also on a context-key change (b, regime) and/or a level jump;
              audit of the K choice against the true cost; audit of every stop/grow decision (predicted mu-hat vs
              the actual next layer under the fixed calibration vs the actual acceptance gain); delta decay.
WorldOH       a Target per-iteration overhead `oh` that both modes pay; T_ar_kernel() leaves it out (what a
              decode-step-time table would give).
SpecSwitch3   the D42 switch with the Target-only rate from the full cycle ("full"), from the kernel time as a
              prior until measured ("kernel_prior"), or from the kernel time throughout ("kernel_always").
simulate4     simulate3 plus RTT as a function of time and short/long prefix phases.
"""
import math
import numpy as np
import feasibility as F
import ext2 as X
import ext3 as Y
import ext4 as Z
from feasibility import KS
from ext2 import Tree2, OPT2, jit_cost


def clone_tree(t):
    c = type(t).__new__(type(t))
    c.req, c.world, c.j = t.req, t.world, t.j
    c.score, c.pi, c.pos = list(t.score), list(t.pi), list(t.pos)
    c.kids = [dict(k) for k in t.kids]
    c.frontier, c.pis, c.scores, c.ids = list(t.frontier), list(t.pis), list(t.scores), list(t.ids)
    c.dep, c.phi = list(t.dep), list(t.phi)
    return c


class CostModelLE(Y.CostModel3):
    def __init__(self, world, mode="learned", keyed=False, aL=0.2, ag=0.1, recenter=True, **kw):
        super().__init__(world, mode, **kw)
        self.keyed, self.aL, self.ag, self.recenter = keyed, aL, ag, recenter
        self.L = self.Ls = None
        self.g = {}
        self.gbar = {}          # (b, regime) -> EMA of the effect of the K actually used (re-centring anchor)
        self.jump = False

    def _rk(self):
        return getattr(self.w, "regime", 0) if self.keyed else 0

    def _g(self, b, K):
        k = (b, K, self._rk())
        if k in self.g:
            return self.g[k]
        seen = [v for (bb, _, rr), v in self.g.items() if bb == b and rr == k[2]]
        return float(np.mean(seen)) if seen else 0.0

    def T(self, b, K):
        if self.mode != "learned":
            return super().T(b, K)
        if self.L is None:
            return self.prior_wait * self.ft
        return (self.L + self._g(b, K)) * self.ft

    def raw_T(self, b, K):
        return self.T(b, K) / self.ft

    def obs_wait(self, b, K, t):
        if self.mode != "learned":
            return
        k = (b, K, self._rk())
        if self.L is None:
            self.L = self.Ls = t; self.g[k] = 0.0
            return
        if k not in self.g:
            self.g[k] = t - self.L
        else:
            self.g[k] += self.ag * (t - self.L - self.g[k])
        self.L += self.aL * (t - self.g[k] - self.L)
        if self.recenter:
            # L and the effect of the K in use are identified only as a sum; without an anchor they random-walk
            # along that line and the effects of rarely used K drift relative to it. Shift all effects of this
            # (b, regime) so that the recently used K has effect ~0; differences between K are unchanged.
            ctx = (b, k[2])
            m = self.gbar.get(ctx, self.g[k])
            m += 0.1 * (self.g[k] - m)
            self.gbar[ctx] = m
            for kk in self.g:
                if kk[0] == b and kk[2] == k[2]:
                    self.g[kk] -= m
            self.L += m
            self.gbar[ctx] = 0.0
        self.Ls += 0.02 * (t - self.g[k] - self.Ls)
        if abs(self.L - self.Ls) > max(2.0, 0.2 * self.Ls):
            self.jump, self.Ls = True, self.L

    # Draft layer cost keyed by the prefix regime when keyed
    def c_d(self, n, W):
        if self.mode == "learned" and self.keyed:
            return self.cd.get((F.bucket(n), W, self._rk()), self.prior_layer) * self._fd()
        return super().c_d(n, W)

    def raw_c_d(self, n, W):
        if self.mode == "learned" and self.keyed:
            return self.cd.get((F.bucket(n), W, self._rk()), self.prior_layer)
        return super().raw_c_d(n, W)

    def obs_layer(self, n, W, t):
        if self.mode == "learned" and self.keyed:
            k = (F.bucket(n), W, self._rk())
            self.cd[k] = t if k not in self.cd else 0.9 * self.cd[k] + 0.1 * t
            return
        super().obs_layer(n, W, t)


class CostModelAnchor(CostModelLE):
    """Anchored level + difference. Per (b, regime): the anchor is the K used most in recent rounds. A sample at
    the anchor updates the level L (fast); a sample at another K updates only d_K = EMA(t - L) (slow). Each sample
    moves exactly one quantity, so level and differences cannot random-walk against each other. When the anchor
    changes, the differences are re-based so that predictions are unchanged."""

    def __init__(self, world, mode="learned", keyed=False, aL=0.2, ad=0.1, **kw):
        super().__init__(world, mode, keyed=keyed, aL=aL, ag=ad, recenter=False, **kw)
        self.ctx = {}           # (b, regime) -> dict(L, anchor, d={K: diff}, use={K: EMA count})

    def T(self, b, K):
        if self.mode != "learned":
            return Y.CostModel3.T(self, b, K)
        c = self.ctx.get((b, self._rk()))
        if c is None:
            return self.prior_wait * self.ft
        d = c["d"].get(K)
        if d is None:
            d = float(np.mean(list(c["d"].values())))
        return (c["L"] + d) * self.ft

    def obs_wait(self, b, K, t):
        if self.mode != "learned":
            return
        key = (b, self._rk())
        c = self.ctx.get(key)
        if c is None:
            self.ctx[key] = dict(L=t, anchor=K, d={K: 0.0}, use={K: 1.0})
            return
        for k in c["use"]:
            c["use"][k] *= 0.95
        c["use"][K] = c["use"].get(K, 0.0) + 0.05
        top = max(c["use"], key=c["use"].get)
        if top != c["anchor"] and top in c["d"]:
            shift = c["d"][top]
            c["L"] += shift
            c["d"] = {k: v - shift for k, v in c["d"].items()}
            c["anchor"] = top
        if K == c["anchor"]:
            c["L"] += self.aL * (t - c["L"])
        elif K not in c["d"]:
            c["d"][K] = t - c["L"]
        else:
            c["d"][K] += self.ag * (t - c["L"] - c["d"][K])
        lvl = c["L"]
        if not hasattr(self, "_Ls"):
            self._Ls = {}
        Ls = self._Ls.get(key, lvl)
        Ls += 0.02 * (lvl - Ls)
        if abs(lvl - Ls) > max(2.0, 0.2 * Ls):
            self.jump, Ls = True, lvl
        self._Ls[key] = Ls


class CostModelEMAK(CostModelLE):
    """The v1 EMA table (one cell per (b, K)), optionally also keyed by the prefix regime, plus a level-jump flag:
    EMA of the residual (sample - its cell before the update); |EMA| > max(2 ms, 20% of the cell) sets `jump`."""

    def __init__(self, world, mode="learned", keyed=False, **kw):
        super().__init__(world, mode, keyed=keyed, recenter=False, **kw)
        self.rbar = 0.0

    def T(self, b, K):
        if self.mode != "learned":
            return Y.CostModel3.T(self, b, K)
        return self.tw.get((b, K, self._rk()), self.prior_wait) * self.ft

    def obs_wait(self, b, K, t):
        if self.mode != "learned":
            return
        k = (b, K, self._rk())
        if k in self.tw:
            r = t - self.tw[k]
            self.rbar += 0.2 * (r - self.rbar)
            if abs(self.rbar) > max(2.0, 0.2 * self.tw[k]):
                self.jump, self.rbar = True, 0.0
            self.tw[k] = 0.9 * self.tw[k] + 0.1 * t
        else:
            self.tw[k] = t


class CT5(Y.CT3):
    def __init__(self, *a, kg_key=False, kg_jump=False, audit_k=False, audit_stop=False, decay=0.9, **k):
        super().__init__(*a, **k)
        self.kg_key, self.kg_jump, self.audit_k, self.audit_stop = kg_key, kg_jump, audit_k, audit_stop
        self.dyn_decay = decay
        self.Kg_key = None
        self.krec = dict(kv_n=0, kv_dis=0, kv_loss=0.0, kg_n=0, kg_dis=0, kg_loss=0.0, reevals=0, rounds=0)
        self.srec = []          # (kind 0 stop / 1 grow, depth, b, mu, sum_u, margin, dG_model, dA_real)
        self.arng = np.random.default_rng(4242)

    TreeCls = Tree2

    def _new_round(self):
        pass

    def _true_T(self, world, n, K):
        return world.T_t(n, K)

    def _audit_stop(self, trees, prm, W, Kg, idxs, mus_K, xs0, u, mask, Dr, depth, n, pending):
        for r, i in enumerate(idxs):
            if depth + 1 > Dr[i]:
                continue
            bit = 1 << i
            mu = float(mus_K[r])
            su = float(u[i].sum())
            if mask & bit:
                rest = mask & ~bit
                margin = float(xs0[mask]) - (float(xs0[rest]) if rest else 0.0)
                pending[i] = (depth, n, mu, su, margin, trees[i].G(Kg), trees[i].accepted(trees[i].topk(Kg, True)))
            else:
                margin = float(xs0[mask | bit]) - (float(xs0[mask]) if mask else 0.0)
                c = clone_tree(trees[i])
                g0, a0 = c.G(Kg), c.accepted(c.topk(Kg, True))
                c.expand(W, self.arng, prm[i])
                self.srec.append((0, depth, n, mu, su, margin, c.G(Kg) - g0, c.accepted(c.topk(Kg, True)) - a0))

    def round(self, world, slots, rng, rtt, rnd):
        n = len(slots)
        self.cm.cur_b = n
        if self.mode == "learned" and rnd < self.cold:
            trees = []
            com, t = OPT2(8, 0.4).round(world, slots, rng, rtt, rnd, trees_out=trees)
            self.learn(world, slots, trees, com)
            self.ema(sum(com), t)
            return com, t
        W = self.W
        Theta = self.theta(world, n)
        cm = self.cm
        self._extra_ms = 0.0
        self._new_round()
        trees = [self.TreeCls(world.reqs[r], j, world) for r, j in slots]
        prm = [self.params(world, r) for r, _ in slots]
        for t, p in zip(trees, prm):
            D, ex = t.expand(W, rng, p)
            self.store(slots, trees, t, D)
        rem = [world.reqs[r]["L"] - j for r, j in slots]
        Dr = [min(world.DMAX, max(0, x - 1)) for x in rem]
        first_ms, later_ms = jit_cost(n, W)
        cost_pop = lambda k: cm.c_d(k, W) + later_ms
        depth, time_ = 1, 0.0
        allr = list(range(n))
        mus, ses = self.mu_se(world, slots, trees, prm, allr, W, self.prng)
        Gs = {K: [t.G(K) for t in trees] for K in KS}
        key = (n, getattr(world, "regime", 0))
        reeval = (self.Kg_last is None or self.since_K >= self.kreuse
                  or (self.kg_key and key != self.Kg_key)
                  or (self.kg_jump and getattr(cm, "jump", False)))
        Ks = KS if reeval else (self.Kg_last,)
        best, Jt = None, {}
        for K in Ks:
            mask, xs, u, _ = self.plan(world, slots, trees, prm, allr, (1 << n) - 1, K, mus, ses, Gs[K][:], depth, Dr, n, W, cost_pop, Theta, stat=False)
            val = float(xs[0][mask]) if mask else 0.0
            J = sum(Gs[K]) - Theta * cm.T(n, K) + val
            Jt[K] = sum(Gs[K]) - Theta * self._true_T(world, n, K) + val
            if best is None or J > best[0]:
                best = (J, K, mask, xs, u)
        _, Kg, mask, xs, u = best
        if reeval:
            self.krec["reevals"] += 1
            if self.audit_k and len(Ks) > 1:
                Kt = max(Jt, key=Jt.get)
                self.krec["kg_n"] += 1; self.krec["kg_dis"] += int(Kt != Kg); self.krec["kg_loss"] += Jt[Kt] - Jt[Kg]
            self.Kg_last, self.since_K, self.Kg_key = Kg, 0, key
            if hasattr(cm, "jump"):
                cm.jump = False
        self.since_K += 1
        time_ += first_ms if reeval else later_ms
        lastG = Gs[Kg][:]; dG = Gs[Kg][:]
        pending = {}
        if self.audit_stop:
            self._audit_stop(trees, prm, W, Kg, allr, mus[Kg], xs[0], u, mask, Dr, depth, n, pending)
        while depth < world.DMAX:
            if not mask:
                break
            S = [i for i in range(n) if mask >> i & 1]
            for i in S:
                D, ex = trees[i].expand(W, rng, prm[i])
                self.store(slots, trees, trees[i], D)
                g = trees[i].G(Kg); dG[i] = g - lastG[i]; lastG[i] = g
                if i in pending:
                    d0, n0, mu, su, margin, g0, a0 = pending.pop(i)
                    self.srec.append((1, d0, n0, mu, su, margin, g - g0, trees[i].accepted(trees[i].topk(Kg, True)) - a0))
            pending = {}
            t_layer = world.real_cd(rng, len(S), W)
            cm.obs_layer(len(S), W, t_layer)
            time_ += t_layer
            depth += 1
            if depth >= world.DMAX:
                break
            m_, s_ = self.mu_se(world, slots, trees, prm, S, W, self.prng)
            mask, xs, u, _ = self.plan(world, slots, trees, prm, S, mask, Kg, m_, s_, dG, depth, Dr, n, W, cost_pop, Theta)
            if self.audit_stop:
                self._audit_stop(trees, prm, W, Kg, S, m_[Kg], xs[0], u, mask, Dr, depth, n, pending)
            time_ += later_ms
            self.stats["decisions"] += 1
        Kv = max(KS, key=lambda K: sum(t.G(K) for t in trees) - Theta * cm.T(n, K))
        if self.audit_k:
            jt = {K: sum(t.G(K) for t in trees) - Theta * self._true_T(world, n, K) for K in KS}
            Kt = max(jt, key=jt.get)
            self.krec["kv_n"] += 1; self.krec["kv_dis"] += int(Kt != Kv); self.krec["kv_loss"] += jt[Kt] - jt[Kv]
        if cm.p_force and self.prng.random() < cm.p_force:
            Kv = int(self.prng.choice([k for k in KS if k != Kv])); self.stats["forced"] += 1
        self.Kv_hist[Kv] += 1
        com = [min(t.accepted(t.topk(Kv, True)) + 1, rem[i]) for i, t in enumerate(trees)]
        wait = world.real_rtt(rng, rtt) + world.real_T(rng, n, Kv)
        cm.obs_wait(n, Kv, wait)
        time_ += wait + self._extra_ms
        self.stats["plan_ms"] += (first_ms if reeval else later_ms) + self._extra_ms
        self.stats["rounds"] += 1
        self.krec["rounds"] += 1
        if self.mode == "learned":
            self.learn(world, slots, trees, com)
        self.ema(sum(com), time_)
        return com, time_


class WorldOH(X.World2):
    def __init__(self, *a, oh=2.0, **k):
        super().__init__(*a, **k)
        self.oh = oh

    def T_t(self, b, K):
        return super().T_t(b, K) + self.oh

    def T_ar(self, b):
        return super().T_ar(b) + self.oh

    def T_ar_kernel(self, b):
        return super().T_ar(b)


class SpecSwitch3(Z.SpecSwitch2):
    def __init__(self, inner, ar_source="full", **kw):
        super().__init__(inner, **kw)
        self.ar_source = ar_source

    def _ar_rate(self, world, s, n):
        kern = n / world.T_ar_kernel(n) if hasattr(world, "T_ar_kernel") else n / world.T_ar(n)
        if self.ar_source == "kernel_always":
            return kern
        if len(s["aw"]) >= 5:
            return self.rate(s["aw"][-self.win:])
        return kern if self.ar_source == "kernel_prior" else n / world.T_ar(n)


def simulate4(world, policy, conc=None, rtt_fn=None, regime_fn=None, long_factor=3.0):
    rng = np.random.default_rng(1)
    slots, nxt = [], 0
    tok = tt = 0.0
    rnd = 0
    base = (world.t1, world.cD1)
    world.regime = 0
    def target(t):
        if not conc:
            return world.b
        for until, b in conc:
            if t < until:
                return b
        return conc[-1][1]
    while True:
        while len(slots) < target(tt) and nxt < len(world.reqs):
            slots.append([nxt, 0]); nxt += 1
        if not slots:
            break
        if regime_fn is not None:
            r = regime_fn(tt)
            if r != world.regime:
                f = long_factor if r else 1.0
                world.t1, world.cD1, world.regime = base[0] * f, base[1] * f, r
        rtt = rtt_fn(tt) if rtt_fn else world.rtt
        com, t = policy.round(world, slots, rng, rtt, rnd)
        tok += sum(com); tt += t; rnd += 1
        keep = []
        for s, c in zip(slots, com):
            s[1] += c
            if s[1] < world.reqs[s[0]]["L"]:
                keep.append(s)
        slots = keep
        if conc and tt > conc[-1][0]:
            break
    world.t1, world.cD1, world.regime = base[0], base[1], 0
    return 1000.0 * tok / tt


class TreeP(Tree2):
    """Tree whose node probabilities are kept per posterior draw of delta (shape (Np,)); pis/G/topk use the mean
    over draws (pi-bar), multiplied along the path first and averaged last (design doc section 3.4)."""

    def __init__(self, req, j, world):
        super().__init__(req, j, world)
        self.pi = [None]                      # root: set on first expand to ones(Np)

    def expand(self, W, rng, params):
        a, g, beta, dl = params
        dl = np.atleast_1d(np.asarray(dl, float))
        if self.pi[0] is None:
            self.pi[0] = np.ones(dl.size)
        F_ = self.frontier
        Q, H = self.req["Q"], self.req["H"]
        rows = []
        for v in F_:
            p = self.pos[v]
            rows.append(Q[p] if p >= 0 else F.gen_dists(rng, self.world.s_at(self.req, self.j + self.dep[v]), 1)[0])
        D = np.stack(rows)
        hit = F.sig(a + g * F.logit(D[:, 0])[None, :] + dl[:, None])               # (Np, nF)
        w = D ** beta; w = w / w.sum(-1, keepdims=True)                             # (nF, C)
        fp = np.stack([self.pi[v] for v in F_], 1)                                  # (Np, nF)
        pv = (fp[:, :, None] * hit[:, :, None] * w[None]).reshape(dl.size, -1)      # (Np, nF*C)
        fs = np.fromiter((self.score[v] for v in F_), float, len(F_))
        sc = (fs[:, None] * D).ravel()
        sel = np.argpartition(-sc, W - 1)[:W] if sc.size > W else np.arange(sc.size)
        new = []
        for flat in sel:
            fi, c = divmod(int(flat), F.C)
            v = F_[fi]
            nid = len(self.score)
            arr = pv[:, flat]
            self.score.append(float(sc[flat])); self.pi.append(arr)
            p = self.pos[v]
            self.pos.append(p + 1 if (p >= 0 and H[p] == c) else -1)
            self.kids.append(dict()); self.kids[v][c] = nid
            self.dep.append(self.dep[v] + 1); self.phi.append(float(D[fi, 0]))
            self.pis.append(float(arr.mean())); self.scores.append(float(sc[flat])); self.ids.append(nid)
            new.append(nid)
        expanded = list(F_)
        self.frontier = new
        return D, expanded


class CT6(CT5):
    """CT5 with the joint integration over the delta posterior (Laplace: N(delta-hat, 1/Hessian)), Np draws with
    common random numbers per round."""
    TreeCls = TreeP

    def __init__(self, *a, n_param=8, **k):
        super().__init__(*a, **k)
        self.n_param = n_param
        self.zrng = np.random.default_rng(777)
        self.z = np.zeros(n_param)

    def _new_round(self):
        self.z = self.zrng.standard_normal(self.n_param)

    def params(self, world, r):
        a, g, b, d = super().params(world, r)
        if self.mode != "learned":
            return (a, g, b, np.full(self.n_param, d))
        sd = math.sqrt(getattr(self, "dvar", {}).get(r, world.sd ** 2))
        return (a, g, b, d + sd * self.z)

    def mu_se(self, world, slots, trees, prm, idxs, W, rng_):
        nR = len(idxs)
        nF = max(len(trees[i].frontier) for i in idxs)
        nT = max(len(trees[i].pis) for i in idxs)
        Np = self.n_param
        fs = np.zeros((nR, nF)); fp = np.zeros((nR, nF, Np)); cur = np.zeros((nR, nT)); dl = np.zeros((nR, Np))
        for r, i in enumerate(idxs):
            t = trees[i]; fr = t.frontier
            fs[r, :len(fr)] = [t.score[v] for v in fr]
            for f, v in enumerate(fr):
                fp[r, f] = t.pi[v]
            cur[r, :len(t.pis)] = t.pis
            dl[r] = np.atleast_1d(prm[i][3])
        a, g, beta = prm[idxs[0]][:3]
        N = self.n_snap or F.N_SNAP
        D = self._draws(world, slots, trees, idxs, nF, N, rng_)                     # (nR, N, nF, C)
        hit = F.sig(a + g * F.logit(D[..., 0])[..., None] + dl[:, None, None, :])  # (nR, N, nF, Np)
        w = D if beta == 1.0 else D ** beta
        w = w / w.sum(-1, keepdims=True)
        pbar = (fp[:, None, :, None, :] * hit[:, :, :, None, :] * w[..., None]).mean(-1)   # path product, then mean
        sc = (fs[:, None, :, None] * D).reshape(nR, N, nF * C_)
        pv = pbar.reshape(nR, N, nF * C_)
        if nF * C_ > W:
            sel = np.argpartition(-sc, W - 1, axis=-1)[..., :W]
            pv = np.take_along_axis(pv, sel, -1)
        top = -np.sort(-np.concatenate([np.broadcast_to(cur[:, None, :], (nR, N, nT)), pv], -1), -1)[..., :max(KS)]
        cum = np.cumsum(top, -1)
        curc = np.cumsum(-np.sort(-cur, -1)[:, :max(KS)], -1)
        mu, se = {}, {}
        for K in KS:
            g_ = cum[..., min(K, cum.shape[-1]) - 1] - curc[:, [min(K, curc.shape[1]) - 1]]
            mu[K] = g_.mean(1); se[K] = g_.std(1) / math.sqrt(N)
        return mu, se


C_ = F.C


class TrueCal(CT5):
    """CT5 given the true acceptance parameters (a, g, beta, delta_i); snapshots and costs still learned."""

    def params(self, world, r):
        return (world.a, world.g, world.beta_true, world.reqs[r]["delta"])


def audit_summary(srec, near=0.1):
    """Stop-boundary audit summary. Records: (kind 0 stop/1 grow, depth, b, mu, sum_u, margin, dG_model, dA_real).
    ratios are mean actual / mean predicted; flips: decisions whose sign changes when the planned value of the
    request is rescaled by the boundary ratio of dA; regret: sum over flipped decisions of |corrected margin|
    (tokens), per 1000 decisions. Everything here is computable from real logs: dG from the post-reply forward of
    the stopped request's leaves, dA from the bonus token when the path reaches a leaf."""
    A = np.asarray(srec, float)
    kind, mu, su, mg, dG, dA = A[:, 0], A[:, 3], A[:, 4], A[:, 5], A[:, 6], A[:, 7]
    r = lambda a, b: float(a.sum() / b.sum()) if b.sum() > 0 else float("nan")
    st, gr, nb = kind == 0, kind == 1, np.abs(mg) <= near
    rb = r(dA[nb], mu[nb])
    shifted = mg + (rb - 1.0) * su
    flip = np.sign(mg) != np.sign(shifted)
    return dict(n=int(len(A)), all_dA_mu=r(dA, mu), stop_dG_mu=r(dG[st], mu[st]), stop_dA_dG=r(dA[st], dG[st]),
                stop_dA_mu=r(dA[st], mu[st]), grow_dA_mu=r(dA[gr], mu[gr]),
                bnd_n=int(nb.sum()), bnd_dA_mu=rb, bnd_stop_dA_mu=r(dA[st & nb], mu[st & nb]),
                bnd_grow_dA_mu=r(dA[gr & nb], mu[gr & nb]),
                flips_bnd=float(flip[nb].mean()) if nb.any() else float("nan"), flips_all=float(flip.mean()),
                regret_per_1000=float(1000.0 * np.abs(shifted[flip]).sum() / len(A)))
