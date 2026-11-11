"""Second batch of simulation tests (stat layer, cost learning, slow loop, K reuse, switch backoff,
unknown budget, snapshot-library selection bias). Builds on feasibility.py / ext.py.

World2 adds: noisy realised costs (Target and Draft times x (1+N(0,s)), RTT + Exp(jitter)), per-request
output length L_i, and optionally an AR(1) per-position draft ease z_p (the Draft's confidence at position
p is sigmoid(logit(s_i) + z_p); off-path nodes use the ease of their nominal position).
OPT2 / CT2 use per-request L_i and the realised costs. CT2 charges the V8-JIT planner times measured
earlier (first decision of a round / later decisions).
"""
import math, time
import numpy as np
import feasibility as F
import ext as E
from feasibility import C, KS, sig, logit, bucket
from ext import nested_dp_fast, best_sub

JIT = {(1, 4): (0.011, 0.009), (1, 8): (0.020, 0.017), (1, 16): (0.047, 0.043),
       (4, 4): (0.053, 0.042), (4, 8): (0.089, 0.079), (4, 16): (0.188, 0.179),
       (8, 4): (0.228, 0.121), (8, 8): (0.307, 0.196), (8, 16): (0.521, 0.413)}


def jit_cost(n, W):
    return JIT[(1 if n <= 1 else (4 if n <= 4 else 8), W)]


class World2(F.World):
    def __init__(self, b, rtt, misspec=False, seed=7, noise=0.0, jitter=0.0, L_range=None, draft_ar=False, n_req=None, **over):
        self.noise, self.jitter, self.L_range, self.draft_ar = noise, jitter, L_range, draft_ar
        L = L_range[1] if L_range else over.pop("L", 256)
        super().__init__(b, rtt, misspec, seed=seed, L=L, n_req=n_req, **over)
        rng = np.random.default_rng(seed + 1000)
        for r in self.reqs:
            r["L"] = int(math.exp(rng.uniform(math.log(L_range[0]), math.log(L_range[1])))) if L_range else self.L

    def make_req(self, rng):
        if not getattr(self, "draft_ar", False):
            return super().make_req(rng)
        s, d = rng.uniform(*getattr(self, "s_range", (0.35, 0.85))), rng.normal(0, self.sd)
        n = self.L + self.DMAX + 2
        z = np.empty(n); z[0] = rng.normal()
        e = rng.normal(0, math.sqrt(1 - 0.81), n)
        for k in range(1, n):
            z[k] = 0.9 * z[k - 1] + e[k]
        sp = sig(logit(np.full(n, s)) + 1.0 * z)
        Q = np.stack([F.gen_dists(rng, float(x), 1)[0] for x in sp])
        hit = rng.random(n) < sig(self.a + self.g * logit(Q[:, 0]) + d)
        w = Q ** self.beta_true if hasattr(self, "beta_true") else Q
        w = w / w.sum(1, keepdims=True)
        idx = np.minimum((np.cumsum(w, 1) < rng.random(n)[:, None]).sum(1), C - 1)
        return dict(s=s, delta=d, Q=Q, H=np.where(hit, idx, -1), sp=sp)

    def s_at(self, req, p):
        if "sp" in req:
            return float(req["sp"][min(p, len(req["sp"]) - 1)])
        return req["s"]

    def real_T(self, rng, b, K):
        return self.T_t(b, K) * max(0.5, 1 + rng.normal(0, self.noise)) if self.noise else self.T_t(b, K)

    def real_cd(self, rng, n, W):
        return self.c_d(n, W) * max(0.5, 1 + rng.normal(0, self.noise)) if self.noise else self.c_d(n, W)

    def real_rtt(self, rng, base):
        return base + (rng.exponential(self.jitter) if self.jitter else 0.0)


class Tree2(F.Tree):
    """Tree that knows each node's nominal text position and its 'phi' (q1 of the distribution it came from)."""

    def __init__(self, req, j, world):
        super().__init__(req, j)
        self.world, self.j = world, j
        self.dep, self.phi = [0], [float(req["Q"][j][0])]

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


class OPT2:
    def __init__(self, N, tau):
        self.N, self.tau = N, tau

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        trees = [Tree2(world.reqs[r], j, world) for r, j in slots]
        if trees_out is not None:
            trees_out.extend(trees)
        params = (world.a, world.g, world.beta, 0.0)
        for t in trees:
            t.expand(self.N, rng, params)
        rem = [world.reqs[r]["L"] - j for r, j in slots]
        active, depth, time_ = list(range(len(trees))), 1, 0.0
        while active and depth < world.DMAX:
            active = [i for i in active
                      if max(trees[i].score[v] for v in trees[i].frontier) >= self.tau and depth + 1 <= max(1, rem[i] - 1)]
            if not active:
                break
            for i in active:
                trees[i].expand(self.N, rng, params)
            time_ += world.real_cd(rng, len(active), self.N)
            depth += 1
        com = [min(t.accepted(t.topk(self.N, False)) + 1, rem[i]) for i, t in enumerate(trees)]
        return com, time_ + world.real_rtt(rng, rtt) + world.real_T(rng, len(slots), self.N)


class CostModel:
    def __init__(self, world, mode="known", ft=1.0, fd=1.0, p_force=0.0, prior_wait=25.0, prior_layer=2.0):
        self.w, self.mode, self.ft, self.fd, self.p_force = world, mode, ft, fd, p_force
        self.tw, self.cd = {}, {}
        self.prior_wait, self.prior_layer = prior_wait, prior_layer

    def T(self, b, K):
        if self.mode == "learned":
            return self.tw.get((b, K), self.prior_wait)
        return self.w.T_t(b, K) * self.ft

    def c_d(self, n, W):
        if self.mode == "learned":
            return self.cd.get((bucket(n), W), self.prior_layer)
        return self.w.c_d(n, W) * self.fd

    def obs_layer(self, n, W, t):
        if self.mode == "learned":
            k = (bucket(n), W); self.cd[k] = t if k not in self.cd else 0.9 * self.cd[k] + 0.1 * t

    def obs_wait(self, b, K, t):
        if self.mode == "learned":
            k = (b, K); self.tw[k] = t if k not in self.tw else 0.9 * self.tw[k] + 0.1 * t


class SlowLoop:
    """UCB over window throughputs, one bandit per batch-size context."""

    def __init__(self, arms=(4, 8, 16), win=40, c=1.0):
        self.arms, self.win, self.c = arms, win, c
        self.ctx = {}

    def _st(self, b):
        k = 1 if b <= 1 else (4 if b <= 4 else 8)
        if k not in self.ctx:
            self.ctx[k] = {"cur": 8, "tok": 0.0, "t": 0.0, "n": 0, "hist": {a: [] for a in self.arms}}
        return self.ctx[k]

    def W(self, b):
        return self._st(b)["cur"]

    def feedback(self, b, tok, t):
        s = self._st(b)
        s["tok"] += tok; s["t"] += t; s["n"] += 1
        if s["n"] < self.win:
            return
        s["hist"][s["cur"]].append(s["tok"] / s["t"])
        s["tok"] = s["t"] = 0.0; s["n"] = 0
        h = s["hist"]
        untried = [a for a in self.arms if len(h[a]) < 2]
        if untried:
            s["cur"] = untried[0]; return
        tot = sum(len(v) for v in h.values())
        def ucb(a):
            v = np.asarray(h[a][-20:])
            return v.mean() + self.c * (v.std() + 1e-9) * math.sqrt(math.log(tot) / len(v))
        s["cur"] = max(self.arms, key=ucb)


class CT2(E.ClearTreeFast):
    def __init__(self, W=4, mode="oracle", cost=None, bias_corr=False, cdec=False, kappa=1.0, kreuse=1,
                 ignore_r=False, library=None, p_band=0.05, p_floor=0.01, slow=None, learn_beta=False, dyn_delta=False,
                 n_snap=None):
        super().__init__(W, mode, cP=0.0, cost="fixed", learn_beta=learn_beta, dyn_delta=dyn_delta)
        self.cm, self.bias_corr, self.cdec, self.kappa, self.kreuse = cost, bias_corr, cdec, kappa, kreuse
        self.ignore_r, self.library, self.p_band, self.p_floor, self.slow = ignore_r, library, p_band, p_floor, slow
        self.n_snap = n_snap
        self.bias_tab = {}              # (pop bucket, zone bin) -> (sum, n)
        self.Kg_last, self.since_K = None, 10 ** 9
        self.lib = {}                   # phi bin -> list of (row, weight)
        self.stats = {"cdec_calls": 0, "decisions": 0, "plan_ms": 0.0, "rounds": 0, "forced": 0, "probes": 0}

    # ---------- snapshot draws ----------
    def _draws(self, world, slots, trees, idxs, nF, N, rng_):
        if self.library in ("knn_nat", "knn_ipw", "oracle_ar"):
            D = np.empty((len(idxs), N, nF, C))
            for r, i in enumerate(idxs):
                t = trees[i]
                for f in range(nF):
                    if f >= len(t.frontier):
                        D[r, :, f] = t.req["Q"][0]
                        continue
                    v = t.frontier[f]
                    if self.library == "oracle_ar":
                        D[r, :, f] = F.gen_dists(rng_, world.s_at(t.req, t.j + t.dep[v]), N)
                    else:
                        D[r, :, f] = self._lib_sample(t.phi[v], N, rng_)
            return D
        if self.mode == "oracle":
            if self.poolarr is None:
                self.poolarr = np.stack([F.gen_dists(self.prng, r["s"], self.POOL) for r in world.reqs])
            ridx = np.array([slots[i][0] for i in idxs])
            j = rng_.integers(0, self.POOL, (len(idxs), N, nF))
            return self.poolarr[ridx[:, None, None], j]
        D = np.empty((len(idxs), N, nF, C))
        for r, i in enumerate(idxs):
            src = self.buf.get(slots[i][0])
            src = np.asarray(src) if src is not None and len(src) >= 16 else np.asarray(self.gbuf[-4000:])
            D[r] = src[rng_.integers(0, len(src), N * nF)].reshape(N, nF, C)
        return D

    def _lib_sample(self, phi, N, rng_):
        b = min(11, int(phi * 12))
        for off in (0, 1, -1, 2, -2, 3, -3):
            ent = self.lib.get(b + off)
            if ent and len(ent) >= 8:
                break
        else:
            return F.gen_dists(rng_, 0.7, N)
        rows = np.asarray([e[0] for e in ent[-1500:]])
        if self.library == "knn_ipw":
            w = np.minimum(np.asarray([e[1] for e in ent[-1500:]]), 100.0)
            idx = rng_.choice(len(rows), N, p=w / w.sum())
        else:
            idx = rng_.integers(0, len(rows), N)
        return rows[idx]

    def _lib_add(self, tree, expanded, D, weight):
        if self.library not in ("knn_nat", "knn_ipw"):
            return
        for v, row in zip(expanded, D):
            b = min(11, int(tree.phi[v] * 12))
            self.lib.setdefault(b, []).append((row, weight))
            if len(self.lib[b]) > 3000:
                del self.lib[b][:-1500]

    def mu_se(self, world, slots, trees, prm, idxs, W, rng_):
        nR = len(idxs)
        nF = max(len(trees[i].frontier) for i in idxs)
        nT = max(len(trees[i].pis) for i in idxs)
        fs = np.zeros((nR, nF)); fp = np.zeros((nR, nF)); cur = np.zeros((nR, nT)); dl = np.zeros(nR)
        for r, i in enumerate(idxs):
            t = trees[i]; fr = t.frontier
            fs[r, :len(fr)] = [t.score[v] for v in fr]
            fp[r, :len(fr)] = [t.pi[v] for v in fr]
            cur[r, :len(t.pis)] = t.pis
            dl[r] = prm[i][3]
        a, g, beta = prm[idxs[0]][:3]
        N = self.n_snap or F.N_SNAP
        D = self._draws(world, slots, trees, idxs, nF, N, rng_)
        p = sig(a + g * logit(D[..., 0]) + dl[:, None, None])
        w = D if beta == 1.0 else D ** beta
        A = p[..., None] * w / w.sum(-1, keepdims=True)
        sc = (fs[:, None, :, None] * D).reshape(nR, N, nF * C)
        pv = (fp[:, None, :, None] * A).reshape(nR, N, nF * C)
        if nF * C > W:
            sel = np.argpartition(-sc, W - 1, axis=-1)[..., :W]
            pv = np.take_along_axis(pv, sel, -1)
        top = -np.sort(-np.concatenate([np.broadcast_to(cur[:, None, :], (nR, N, nT)), pv], -1), -1)[..., :max(KS)]
        cum = np.cumsum(top, -1)                                   # (nR, N, L)
        curc = np.cumsum(-np.sort(-cur, -1)[:, :max(KS)], -1)
        mu, se = {}, {}
        for K in KS:
            g_ = cum[..., min(K, cum.shape[-1]) - 1] - curc[:, [min(K, curc.shape[1]) - 1]]
            mu[K] = g_.mean(1); se[K] = g_.std(1) / math.sqrt(N)
        return mu, se

    # ---------- one planning step ----------
    def plan(self, world, slots, trees, prm, act_list, act_mask, K, mus, ses, dG, depth, Dr, n, W, cost_pop, Theta, stat=True):
        mu_vec = [0.0] * n; se_vec = [0.0] * n
        for r, i in enumerate(act_list):
            mu_vec[i] = float(mus[K][r]); se_vec[i] = float(ses[K][r])
        u, el = self.build_u(mu_vec, dG, depth, Dr, act_list, n, world)
        xs = self.DP(u, el, act_mask, cost_pop, Theta)
        S = np.arange(xs[0].size)
        ok = ((S & ~act_mask) == 0) & (S > 0)
        cand = np.where(ok, xs[0], -np.inf)
        bm = int(np.argmax(cand)); vp = float(cand[bm])
        if not np.isfinite(vp):
            return 0, xs, u, 0.0
        if not stat or not (self.bias_corr or self.cdec):
            return (bm if vp > 0 else 0), xs, u, 0.0
        # Monte Carlo SE of the chosen plan's value (future layers scale with mu)
        H = u.shape[1]
        se_v = 0.0
        for i in range(n):
            if bm >> i & 1 and mu_vec[i] > 0:
                se_v += (se_vec[i] * u[i].sum() / mu_vec[i]) ** 2
        se_v = math.sqrt(se_v) + 1e-9
        popb = bucket(bin(bm).count("1"))
        zb = min(3, int(abs(vp) / se_v))
        vt = vp
        extra_ms = 0.0
        if self.bias_corr:
            s_, k_ = self.bias_tab.get((popb, zb), (0.0, 0))
            bh = s_ / k_ if k_ >= 10 else 0.0
            vt = vp - bh
            if self.prng.random() < 0.1:                      # C_cal: fresh evaluation of the same fixed first step
                vc = self._fresh_value(world, slots, trees, prm, bm, K, dG, depth, Dr, n, W, cost_pop, Theta)
                s_, k_ = self.bias_tab.get((popb, zb), (0.0, 0)); self.bias_tab[(popb, zb)] = (s_ + vp - vc, k_ + 1)
                extra_ms += jit_cost(n, W)[1]
        if self.cdec and abs(vt) <= self.kappa * se_v:
            self.stats["cdec_calls"] += 1
            vals = []
            for _ in range(2):
                vals.append(self._fresh_value(world, slots, trees, prm, bm, K, dG, depth, Dr, n, W, cost_pop, Theta))
                extra_ms += jit_cost(n, W)[1]
            vt = (vt + sum(vals)) / (1 + len(vals))
        self._extra_ms += extra_ms
        return (bm if vt > 0 else 0), xs, u, 0.0

    def _fresh_value(self, world, slots, trees, prm, bm, K, dG, depth, Dr, n, W, cost_pop, Theta):
        act = [i for i in range(n) if bm >> i & 1]
        mus, _ = self.mu_se(world, slots, trees, prm, act, W, self.prng)
        mu_vec = [0.0] * n
        for r, i in enumerate(act):
            mu_vec[i] = float(mus[K][r])
        u, el = self.build_u(mu_vec, dG, depth, Dr, act, n, world)
        xs = self.DP(u, el, bm, cost_pop, Theta)
        return float(xs[0][bm])

    def round(self, world, slots, rng, rtt, rnd):
        n = len(slots)
        if self.mode == "learned" and rnd < self.cold:
            trees = []
            com, t = OPT2(8, 0.4).round(world, slots, rng, rtt, rnd, trees_out=trees)
            self.learn(world, slots, trees, com)
            self.ema(sum(com), t)
            return com, t
        W = self.slow.W(n) if self.slow else self.W
        Theta = self.theta(world, n)
        cm = self.cm
        self._extra_ms = 0.0
        trees = [Tree2(world.reqs[r], j, world) for r, j in slots]
        prm = [self.params(world, r) for r, _ in slots]
        for t, p in zip(trees, prm):
            D, ex = t.expand(W, rng, p)
            self.store(slots, trees, t, D)
            self._lib_add(t, ex, D, 1.0)
        rem = [world.reqs[r]["L"] - j for r, j in slots]
        Dr = [world.DMAX if self.ignore_r else min(world.DMAX, max(0, x - 1)) for x in rem]
        first_ms, later_ms = jit_cost(n, W)
        cost_pop = lambda k: cm.c_d(k, W) + later_ms
        depth, time_ = 1, 0.0
        allr = list(range(n))
        mus, ses = self.mu_se(world, slots, trees, prm, allr, W, self.prng)
        Gs = {K: [t.G(K) for t in trees] for K in KS}
        reeval = self.Kg_last is None or self.since_K >= self.kreuse
        Ks = KS if reeval else (self.Kg_last,)
        best = None
        for K in Ks:
            mask, xs, u, _ = self.plan(world, slots, trees, prm, allr, (1 << n) - 1, K, mus, ses, Gs[K][:], depth, Dr, n, W, cost_pop, Theta, stat=False)
            val = float(xs[0][mask]) if mask else 0.0
            J = sum(Gs[K]) - Theta * cm.T(n, K) + val
            if best is None or J > best[0]:
                best = (J, K, mask)
        _, Kg, mask = best
        if self.bias_corr or self.cdec:
            mask, _, _, _ = self.plan(world, slots, trees, prm, allr, (1 << n) - 1, Kg, mus, ses, Gs[Kg][:], depth, Dr, n, W, cost_pop, Theta)
        if reeval:
            self.Kg_last, self.since_K = Kg, 0
        self.since_K += 1
        time_ += first_ms if reeval else later_ms
        lastG = Gs[Kg][:]; dG = Gs[Kg][:]
        active_mask = (1 << n) - 1
        while depth < world.DMAX:
            # random expansion probes for requests that stop here
            if self.library in ("knn_ipw",):
                stopped = [i for i in range(n) if (active_mask >> i & 1) and not (mask >> i & 1) and depth + 1 <= Dr[i]]
                probes = [i for i in stopped if self.prng.random() < self.p_floor]
                for i in probes:
                    D, ex = trees[i].expand(W, rng, prm[i])
                    self._lib_add(trees[i], ex, D, 1.0 / self.p_floor)
                    self.stats["probes"] += 1
                if probes and not mask:
                    time_ += world.real_cd(rng, len(probes), W)
            if not mask:
                break
            S = [i for i in range(n) if mask >> i & 1]
            for i in S:
                D, ex = trees[i].expand(W, rng, prm[i])
                self.store(slots, trees, trees[i], D)
                self._lib_add(trees[i], ex, D, 1.0)
                g = trees[i].G(Kg); dG[i] = g - lastG[i]; lastG[i] = g
            t_layer = world.real_cd(rng, len(S), W)
            cm.obs_layer(len(S), W, t_layer)
            time_ += t_layer
            depth += 1
            active_mask = mask
            if depth >= world.DMAX:
                break
            m_, s_ = self.mu_se(world, slots, trees, prm, S, W, self.prng)
            mask, _, _, _ = self.plan(world, slots, trees, prm, S, mask, Kg, m_, s_, dG, depth, Dr, n, W, cost_pop, Theta)
            time_ += later_ms
            self.stats["decisions"] += 1
        Kv = max(KS, key=lambda K: sum(t.G(K) for t in trees) - Theta * cm.T(n, K))
        if cm.p_force and self.prng.random() < cm.p_force:
            Kv = int(self.prng.choice([k for k in KS if k != Kv])); self.stats["forced"] += 1
        self.Kv_hist[Kv] += 1
        com = [min(t.accepted(t.topk(Kv, True)) + 1, rem[i]) for i, t in enumerate(trees)]
        wait = world.real_rtt(rng, rtt) + world.real_T(rng, n, Kv)
        cm.obs_wait(n, Kv, wait)
        time_ += wait + self._extra_ms
        self.stats["plan_ms"] += (first_ms if reeval else later_ms) + self._extra_ms
        self.stats["rounds"] += 1
        if self.mode == "learned":
            self.learn(world, slots, trees, com)
        self.ema(sum(com), time_)
        if self.slow:
            self.slow.feedback(n, sum(com), time_)
        return com, time_


class SpecSwitchBackoff(E.SpecSwitch):
    def __init__(self, inner, max_period_ms=48000.0, **kw):
        super().__init__(inner, **kw)
        self.base_pe, self.max_pe = self.pe, max_period_ms

    def round(self, world, slots, rng, rtt, rnd):
        was_probe = self.probe_left
        mode_before = self.mode
        com, t = super().round(world, slots, rng, rtt, rnd)
        if was_probe == 1 and self.probe_left == 0:            # a probe just finished
            self.pe = self.base_pe if self.mode == "spec" else min(self.max_pe, self.pe * 2)
        if mode_before == "spec" and self.mode == "ar":
            self.pe = self.base_pe
        return com, t


def simulate3(world, policy, seed=1, conc=None, period_ms=None):
    """conc: list of (until_ms, b) pieces for the target concurrency; RTT may alternate every period_ms."""
    rng = np.random.default_rng(seed)
    slots, nxt = [], 0
    tok = tt = 0.0
    rnd = 0
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
        rtt = world.rtt if not period_ms else (world.rtt[0] if int(tt // period_ms) % 2 == 0 else world.rtt[1])
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
    return 1000.0 * tok / tt
