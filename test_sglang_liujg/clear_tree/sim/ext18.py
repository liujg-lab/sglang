"""Round 21: prefix-length-aware costs. Can the policy learn layer and verify costs with a prefix term online, and does
it help? (BatchAccount11, World21)

World21 (C1 costs plus a prefix term; every request has its own prompt length p0, its prefix is p0 + generated tokens):
  Draft layer   c_d(n, W) + kd(W) * sum of the active requests' prefixes,    kd(W) = k_d (0.5 + 0.125 W)
  Target verify c0 + c_b b + v_tok b (K + 1) + kt(K) * sum of all prefixes + first layer (with its own prefix term),
                kt(K) = k_t (0.5 + (K + 1) / 34)
BatchAccount11: v10 growth (W from the table, K online, consistent growth/verify value, exits by graph tier), no W
calibration. cost_mode:
  known          the true cost functions (prefix term included); exits ordered by net gain g_i - Theta kd(W) l_i
  blind          r20 online estimates that ignore the prefix: EMA per (tier, W) and per (b, K); ordered by g_i
  aware          online regression with forgetting (0.999): per W, layer time ~ [1, tier, sum l]; round wait minus the
                 estimated first layer ~ [1{K=8}, 1{K=16}, 1{K=32}, b, sum l, (K+1) sum l / 17] pooled over K (a first
                 version with one regression per K could not identify the prefix slope of the K's it rarely ran);
                 ordered by net gain g_i - Theta kd_hat(W) l_i
  aware_noorder  the same estimates, ordered by g_i only
Online modes run the timing probe of r20 (every (W, K) twice per b bucket after warm-up) and a drift detector: a fast
EMA (0.2) of the relative residual (observed / predicted - 1) per (tier, W) beyond 15% for 10 observations in a row
triggers a re-probe; for "blind" the prediction is its own EMA reference after the probe (as in r20).
Diagnostics: mean |predicted / true - 1| of the layer time and of the verify time for what actually ran (true = the
world's noise-free cost), after the probes.
"""
import copy
import math
import numpy as np
import ext2 as X
import ext7 as S
import ext9 as Q
import ext10 as P
import ext14 as E
import feasibility as F
import run7 as R7
import run13 as U


class World21(R7.World7):
    def __init__(self, theta, a0, a_b, mix="mixed", k_d=0.001, k_t=0.002, p_seed=99, **kw):
        super().__init__(theta, a0, a_b, **kw)
        self.k_d, self.k_t, self.n_T = k_d, k_t, 0
        rng = np.random.default_rng(p_seed)
        for r in self.reqs:
            long_ = mix == "long" or (mix == "mixed" and rng.random() < 0.5)
            r["p0"] = int(rng.uniform(3000, 5000)) if long_ else int(rng.uniform(100, 400))

    def kd(self, W):
        return self.k_d * (0.5 + 0.125 * W)

    def kt(self, K):
        return self.k_t * (0.5 + (K + 1) / 34.0)

    def c_d_l(self, n, W, suml):
        return self.c_d(n, W) + self.kd(W) * suml

    def T_core(self, b, K, suml):
        return self.c0 + self.c_b * b + self.v_tok * b * (K + 1) + self.kt(K) * suml

    def real_cd_l(self, rng, n, W, suml):
        c = self.c_d_l(n, W, suml)
        return c * max(0.5, 1 + rng.normal(0, self.noise)) if self.noise else c

    def real_T_l(self, rng, b, K, suml):
        self.n_T += 1
        c = self.T_core(b, K, suml) + self.c_d_l(b, self.first_W, suml)
        return c * max(0.5, 1 + rng.normal(0, self.noise)) if self.noise else c


def make_world(mix, kind, b, seed, strength, req_mult=3):
    a, s, k = U.WL[kind]
    S.set_conc(k)
    fit = copy.deepcopy(U.FIT)
    th = fit["fits"]["full"]["theta"]
    kd, kt = {"mid": (0.00025, 0.0005), "strong": (0.001, 0.002)}[strength]
    return World21(th, fit["a0"], fit["a_b"], mix=mix, k_d=kd, k_t=kt, seed=seed, b=b, rtt=4.0,
                   n_req=req_mult * max(16, 8 * b), **R7.NOISY, **S.acc_kw(a, *s))


class _RLS:
    """Least squares with exponential forgetting on a few features (solved on demand, tiny ridge)."""

    def __init__(self, d, lam=0.999):
        self.A, self.y, self.n, self.lam = np.zeros((d, d)), np.zeros(d), 0, lam

    def add(self, x, t):
        x = np.asarray(x, float)
        self.A = self.lam * self.A + np.outer(x, x)
        self.y = self.lam * self.y + x * t
        self.n += 1
        self._w = None

    def coef(self):
        if getattr(self, "_w", None) is None:
            d = len(self.y)
            scale = np.maximum(np.diag(self.A), 1e-9)
            self._w = np.linalg.solve(self.A + 1e-6 * np.diag(scale) + 1e-12 * np.eye(d), self.y)
        return self._w

    def pred(self, x):
        return float(np.dot(self.coef(), x))


class BatchAccount11(E.BatchAccount7):
    def __init__(self, wk, cost_mode="aware", probe_passes=2, drift_tol=0.15, drift_run=10, **kw):
        super().__init__(wk, lambda b, K: 0.0, **kw)
        self.cost_mode, self.probe_passes, self.drift_tol, self.drift_run = cost_mode, probe_passes, drift_tol, drift_run
        self.w_opts = (2, 4, 8)
        self.prng2 = np.random.default_rng(2718)
        self.lc, self.vc = {}, {}                       # blind EMAs
        self.rl = {W: _RLS(3) for W in self.w_opts}     # aware: per W [1, tier, suml/1000]
        self.rv = _RLS(6)                               # aware: pooled over K [1{K=8}, 1{K=16}, 1{K=32}, b, s, (K+1) s / 17]
        self.ref, self.fast = {}, {}
        self.probed, self.probe_q = set(), {}
        self.n_probe_rounds = self.n_cost_events = 0
        self.err_l, self.err_v = [], []
        self.world = None

    # ---------------- cost estimates ----------------
    def _lt(self, m, W, suml):
        tier = F.bucket(m)
        if self.cost_mode == "known":
            return self.world.c_d_l(m, W, suml)
        if self.cost_mode == "blind":
            c = self.lc.get((tier, W))
            if c and c[1] >= 2:
                return c[0]
            pts = [(k[0], k[1], v[0]) for k, v in self.lc.items() if v[1] >= 2]
            if len({(p[0], p[1]) for p in pts}) >= 3:
                A = np.array([[1.0, p[0], p[0] * p[1]] for p in pts]); y = np.array([p[2] for p in pts])
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                est = float(coef[0] + coef[1] * tier + coef[2] * tier * W)
                if est > 0:
                    return est
            return c[0] if c else 5.0
        r = self.rl[W]
        if r.n >= 4:
            return max(0.1, r.pred([1.0, tier, suml / 1000.0]))
        return 5.0

    def _kd_hat(self, W):
        if self.cost_mode == "known":
            return self.world.kd(W)
        if self.cost_mode in ("aware", "aware_noorder") and self.rl[W].n >= 4:
            return max(0.0, self.rl[W].coef()[2] / 1000.0)
        return 0.0

    def _vt(self, b, K, suml):
        if self.cost_mode == "known":
            return self.world.T_core(b, K, suml)
        if self.cost_mode == "blind":
            c = self.vc.get((b, K))
            if c and c[1] >= 2:
                return c[0]
            same = [(k[1], v[0]) for k, v in self.vc.items() if k[0] == b and v[1] >= 2]
            if same:
                return min(same, key=lambda p: abs(p[0] - K))[1]
            return 40.0
        if self.rv.n >= 6:
            return self.rv.pred(self._vx(b, K, suml))
        return 40.0

    def _vx(self, b, K, suml):
        s_ = suml / 1000.0
        return [float(K == 8), float(K == 16), float(K == 32), float(b), s_, (K + 1) * s_ / 17.0]

    @staticmethod
    def _ema(d, key, x, alpha=0.1):
        c = d.setdefault(key, [x, 0])
        c[1] += 1
        a = 1.0 / c[1] if c[1] <= 5 else alpha
        c[0] = x if c[1] == 1 else (1 - a) * c[0] + a * x

    def _observe(self, b, W, K, layers, wait, suml_all):
        if self.cost_mode == "known":
            return
        trig = False
        for m, suml, dt in layers:
            tier = F.bucket(m)
            pred = self._lt(m, W, suml)
            if self.cost_mode == "blind":
                self._ema(self.lc, (tier, W), dt)
            else:
                self.rl[W].add([1.0, tier, suml / 1000.0], dt)
            key = (tier, W)
            if key in self.ref:
                if self.cost_mode == "blind":
                    dev = dt / self.ref[key] - 1
                else:
                    dev = dt / max(pred, 1e-6) - 1
                f = self.fast.setdefault(key, [dev, 0])
                f[0] = 0.8 * f[0] + 0.2 * dev
                if abs(f[0]) > self.drift_tol:
                    f[1] += 1
                    trig = trig or f[1] >= self.drift_run
                else:
                    f[1] = 0
        core = wait - self._lt(b, W, suml_all)
        if self.cost_mode == "blind":
            self._ema(self.vc, (b, K), core)
        else:
            self.rv.add(self._vx(b, K, suml_all), core)
        if trig:
            self.n_cost_events += 1
            for v in self.lc.values():
                v[1] = 0
            for r in list(self.rl.values()) + [self.rv]:
                r.A *= 0.0; r.y *= 0.0; r.n = 0; r._w = None
            self.ref, self.fast, self.probed, self.probe_q = {}, {}, set(), {}

    # ---------------- one round ----------------
    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        self.world = world
        b = len(slots)
        h1 = sum(int(world.reqs[r]["H"][j] == 0) for r, j in slots)
        if self.t1_n is None:
            self.t1_hit, self.t1_n = float(h1), float(b)
        else:
            self.t1_hit = 0.95 * self.t1_hit + 0.05 * h1
            self.t1_n = 0.95 * self.t1_n + 0.05 * b
        cls = self.cls()
        W, Ktab = self.wk(b, cls)
        theta = self._theta_b(b)
        warm = self.rounds < self.warm or theta is None
        kforce, probe = None, False
        if not warm and self.cost_mode != "known":
            bk = Q.bkey(b)
            if bk not in self.probed:
                q = [(w_, k_) for w_ in self.w_opts for k_ in self.k_options] * self.probe_passes
                self.prng2.shuffle(q)
                self.probe_q[bk] = [tuple(int(v) for v in x) for x in q]
                self.probed.add(bk)
            q = self.probe_q.get(bk)
            if q:
                W, kforce = q.pop(0)
                probe = True
                self.n_probe_rounds += 1
                if not q:
                    for key, v in self.lc.items():
                        if v[1] >= 1 and key not in self.ref:
                            self.ref[key] = v[0]
                    if self.cost_mode != "blind":
                        # aware modes compare with their own prediction; the reference only marks "probed"
                        for w_ in self.w_opts:
                            for t_ in (1, 2, 4, 8):
                                self.ref.setdefault((t_, w_), 1.0)
        Ks = self.k_options
        Kmax = max(Ks)
        world.first_W = W
        trees = [X.Tree2(world.reqs[r], j, world) for r, j in slots]
        rids = [r for r, _ in slots]
        ell = [world.reqs[r]["p0"] + j for r, j in slots]
        suml_all = float(sum(ell))
        n = len(trees)
        params = (world.a, world.g, world.beta, 0.0)
        for t in trees:
            t.expand(W, rng, params)
        rem = [world.reqs[r]["L"] - j for r, j in slots]
        cs = [self._c(r) for r in rids]
        e_now = [self._es(t) for t in trees]
        e_prev = [{K: 0.0 for K in Ks} for _ in range(n)]
        v_now = [self._vals(cs[i], e_now[i]) for i in range(n)]
        v_prev = [{K: 0.0 for K in Ks} for _ in range(n)]
        order_net = self.cost_mode in ("known", "aware")
        kd = self._kd_hat(W)
        active, depth, time_ = list(range(n)), 1, 0.0
        layers = []
        while active:
            cand = [i for i in active if depth < world.DMAX - 1 and depth + 1 <= max(1, rem[i] - 1)]
            if not cand:
                break
            if warm:
                keep = [i for i in cand if e_now[i][Kmax] - e_prev[i][Kmax] > 0.4]
            else:
                Jk = {K: sum(v_now[i][K] for i in range(n)) - theta * self._vt(b, K, suml_all) for K in Ks}
                stop = max(Jk.values())
                best, keep = 0.0, []
                for K in Ks:
                    g = {i: v_now[i][K] - v_prev[i][K] for i in cand}
                    sc = {i: g[i] - (theta * kd * ell[i] if order_net else 0.0) for i in cand}
                    order = sorted(cand, key=lambda i: sc[i], reverse=True)
                    for m in self._mopts(len(order)):
                        sel = order[:m]
                        cost = self._lt(m, W, sum(ell[i] for i in sel))
                        v = Jk[K] + sum(g[i] for i in sel) - theta * cost - stop
                        if v > best + 1e-12:
                            best, keep = v, sel
            if not keep:
                break
            for i in keep:
                trees[i].expand(W, rng, params)
                e_prev[i], e_now[i] = e_now[i], self._es(trees[i])
                v_prev[i], v_now[i] = v_now[i], self._vals(cs[i], e_now[i])
            sl = float(sum(ell[i] for i in keep))
            dt = world.real_cd_l(rng, len(keep), W, sl)
            if not warm and not probe and self.cost_mode != "known":
                self.err_l.append(abs(self._lt(len(keep), W, sl) / world.c_d_l(len(keep), W, sl) - 1))
            time_ += dt
            layers.append((len(keep), sl, dt))
            active = keep
            depth += 1
        if warm:
            K = Ktab
        else:
            Jk = {K_: sum(v_now[i][K_] for i in range(n)) - theta * self._vt(b, K_, suml_all) for K_ in Ks}
            K = max(Jk, key=Jk.get)
        if kforce is not None:
            K = kforce
        self.k_log[K] = self.k_log.get(K, 0) + 1
        acc = [P.last_node(t, t.topk(K, False))[0] for t in trees]
        com = [min(a + 1, rem[i]) for i, a in enumerate(acc)]
        wait = world.real_rtt(rng, rtt) + world.real_T_l(rng, b, K, suml_all)
        t_round = time_ + wait
        if not warm and not probe and self.cost_mode != "known":
            # verify: compare K = 32 minus K = 8 predictions with the truth (the RTT is a constant offset)
            pd = self._vt(b, 32, suml_all) - self._vt(b, 8, suml_all)
            td = world.T_core(b, 32, suml_all) - world.T_core(b, 8, suml_all)
            self.err_v.append(abs(pd - td) / max(td, 1e-9))
        if not warm:
            self._observe(b, W, K, layers, wait, suml_all)
        tok = float(sum(com))
        self.tok_total += tok
        fx = t_round - time_
        self.fixed[b] = fx if b not in self.fixed else 0.9 * self.fixed[b] + 0.1 * fx
        a_ = self.alpha
        if self.g_ms is None:
            self.g_tok, self.g_ms = tok, t_round
        else:
            self.g_tok = (1 - a_) * self.g_tok + a_ * tok
            self.g_ms = (1 - a_) * self.g_ms + a_ * t_round
        v = self.tb.setdefault(b, [tok, t_round, 0])
        v[2] += 1
        ab_ = max(a_, 1.0 / v[2])
        v[0] = (1 - ab_) * v[0] + ab_ * tok
        v[1] = (1 - ab_) * v[1] + ab_ * t_round
        for i, t in enumerate(trees):
            e = e_now[i][K]
            r = rids[i]
            self.req_a[r] = self.req_a.get(r, 0.0) + acc[i]
            self.req_e[r] = self.req_e.get(r, 0.0) + e
            self.c_a += acc[i]; self.c_e += e
        self.rounds += 1
        return com, t_round
