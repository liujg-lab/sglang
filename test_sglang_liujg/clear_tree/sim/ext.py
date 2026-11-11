"""Extensions to feasibility.py (remaining simulation tests).

ClearTreeV: same algorithm as feasibility.ClearTree, but the next-layer increment of all requests is
computed in one vectorised pass (requests padded with zeros), so planner time no longer grows with a
Python loop over requests. Options:
  cost="fixed"     charge cP per planning call (as before)
  cost="measured"  charge the measured wall time of each planning call
  lag=True         planning overlaps the GPU layer forward: the decision for the step after next is made
                   while the next forward runs (information one layer stale); only the part of the
                   planning time that exceeds that forward is charged; the first decision is charged fully
  dyn_delta=True   (learned) request effect estimated with exponentially decaying weights (lambda 0.9),
                   so it can follow local difficulty
SpecSwitch: wraps any policy; switches to Target-only when the measured speculative throughput falls below
(1-m) x the Target-only throughput, probes speculation periodically, switches back above (1+m).
StaticSwitch: the current OPT rule, Target-only whenever b >= b_min.
simulate_t: like feasibility.simulate, but RTT can alternate on simulated time.
"""
import math, time
import numpy as np
import feasibility as F
from feasibility import C, KS, sig, logit, ahat, Tree, nested_dp, dp_tables


def gen_dists_v(rng, s, shape):
    """Vectorised gen_dists: s has shape (R,), output (R, *shape, C)."""
    full = (len(s),) + tuple(shape)
    sb = s.reshape((-1,) + (1,) * len(shape))
    q1 = rng.beta(np.broadcast_to(sb * 6.0, full), np.broadcast_to((1 - sb) * 6.0, full))
    rho = rng.uniform(0.5, 0.95, full)
    g = rng.uniform(0.4, 0.8, full)
    pw = g[..., None] ** np.arange(C - 1)
    rest = pw / pw.sum(-1, keepdims=True) * ((1 - q1) * rho)[..., None]
    Q = np.concatenate([q1[..., None], rest], -1)
    return -np.sort(-Q, -1)


def nested_dp_full(u, elig, act_mask, cost_of_pop, Theta):
    """Like feasibility.nested_dp but also returns the per-layer x arrays (for the second step)."""
    n, H = u.shape
    S, B, pop, drop = dp_tables(n)
    cost = np.array([cost_of_pop(int(k)) if k else 0.0 for k in range(n + 1)])[pop.astype(int)]
    V = np.zeros(S.size)
    xs = [None] * H
    for k in range(H - 1, -1, -1):
        x = B @ u[:, k] - Theta * cost + V - 1e-9 * pop
        bad = (B @ (~elig[:, k]).astype(float)) > 0
        x[bad] = -np.inf
        x[0] = 0.0
        xs[k] = x
        Fm = x.copy()
        for i in range(n):
            Fm = np.maximum(Fm, Fm[drop[i]])
        V = Fm
    return xs


def best_sub(x, mask):
    S = np.arange(x.size)
    xs = np.where((S & ~mask) == 0, x, -np.inf)
    b = int(np.argmax(xs))
    return (b, float(xs[b])) if xs[b] > 0 else (0, 0.0)


class ClearTreeV(F.ClearTree):
    def __init__(self, W, mode="oracle", cP=0.1, cost="fixed", lag=False, dyn_delta=False, learn_beta=False, cold=40):
        super().__init__(W, mode, cP=cP, cold=cold, learn_beta=learn_beta)
        self.cost, self.lag, self.dyn = cost, lag, dyn_delta
        self.DP = nested_dp_full
        self.charged = []

    def mu_vec(self, world, slots, trees, prm, idxs, W):
        nR = len(idxs)
        nF = max(len(trees[i].frontier) for i in idxs)
        nT = max(len(trees[i].pis) for i in idxs)
        fs = np.zeros((nR, nF)); fp = np.zeros((nR, nF)); cur = np.zeros((nR, nT)); dl = np.zeros(nR)
        for r, i in enumerate(idxs):
            t = trees[i]
            fr = t.frontier
            fs[r, :len(fr)] = [t.score[v] for v in fr]
            fp[r, :len(fr)] = [t.pi[v] for v in fr]
            cur[r, :len(t.pis)] = t.pis
            dl[r] = prm[i][3]
        a, g, beta = prm[idxs[0]][:3]
        N = F.N_SNAP
        if self.mode == "oracle":
            s = np.array([world.reqs[slots[i][0]]["s"] for i in idxs])
            D = gen_dists_v(self.prng, s, (N, nF))
        else:
            D = np.empty((nR, N, nF, C))
            for r, i in enumerate(idxs):
                D[r] = self.draws(world, slots[i][0], nF)
        p = sig(a + g * logit(D[..., 0]) + dl[:, None, None])
        w = D ** beta
        A = p[..., None] * w / w.sum(-1, keepdims=True)
        sc = (fs[:, None, :, None] * D).reshape(nR, N, nF * C)
        pv = (fp[:, None, :, None] * A).reshape(nR, N, nF * C)
        if nF * C > W:
            sel = np.argpartition(-sc, W - 1, axis=-1)[..., :W]
            pv = np.take_along_axis(pv, sel, -1)
        allv = np.concatenate([np.broadcast_to(cur[:, None, :], (nR, N, nT)), pv], -1)
        cum = np.cumsum(-np.sort(-allv, -1), -1).mean(1)          # (nR, L)
        curc = np.cumsum(-np.sort(-cur, -1), -1)                   # (nR, nT)
        return {K: cum[:, min(K, cum.shape[1]) - 1] - curc[:, min(K, nT) - 1] for K in KS}

    def charge(self, ms, overlap_ms=0.0):
        if self.cost == "fixed":
            c = self.cP if overlap_ms == 0.0 else max(0.0, self.cP - overlap_ms)
        else:
            c = ms if overlap_ms == 0.0 else max(0.0, ms - overlap_ms)
        self.charged.append(c)
        return c

    def round(self, world, slots, rng, rtt, rnd):
        n = len(slots)
        if self.mode == "learned" and rnd < self.cold:
            trees = []
            com, t = self.boot.round(world, slots, rng, rtt, rnd, trees_out=trees)
            self.learn(world, slots, trees, com)
            self.ema(sum(com), t)
            return com, t
        W, Theta = self.W, self.theta(world, n)
        trees = [Tree(world.reqs[r], j) for r, j in slots]
        prm = [self.params(world, r) for r, _ in slots]
        for t, p in zip(trees, prm):
            self.store(slots, trees, t, t.expand(W, rng, p))
        rem = [world.L - j for _, j in slots]
        Dr = [min(world.DMAX, max(0, x - 1)) for x in rem]
        depth, time_ = 1, 0.0
        cost_pop = lambda k: world.c_d(k, W) + (self.cP if not self.lag else 0.0)
        # first decision: choose K_g and the first step (charged in full)
        t0 = time.perf_counter()
        mus = self.mu_vec(world, slots, trees, prm, list(range(n)), W)
        Gs = {K: [t.G(K) for t in trees] for K in KS}
        best = None
        for K in KS:
            u, el = self.build_u(list(mus[K]), Gs[K], depth, Dr, range(n), n, world)
            xs = self.DP(u, el, (1 << n) - 1, cost_pop, Theta)
            first, val = best_sub(xs[0], (1 << n) - 1)
            J = sum(Gs[K]) - Theta * world.T_t(n, K) + val
            if best is None or J > best[0]:
                best = (J, K, first, xs)
        _, Kg, cur_mask, xs = best
        ms = (time.perf_counter() - t0) * 1000
        self.plan_times.append(ms / 1000 / len(KS))
        time_ += self.charge(ms)
        nxt = best_sub(xs[1], cur_mask)[0] if (self.lag and cur_mask and len(xs) > 1) else None
        lastG = Gs[Kg][:]
        dG = Gs[Kg][:]
        while cur_mask and depth < world.DMAX:
            S = [i for i in range(n) if cur_mask >> i & 1]
            for i in S:
                self.store(slots, trees, trees[i], trees[i].expand(W, rng, prm[i]))
                g = trees[i].G(Kg); dG[i] = g - lastG[i]; lastG[i] = g
            fwd = world.c_d(len(S), W)
            time_ += fwd
            depth += 1
            if depth >= world.DMAX:
                break
            if self.lag:
                # the forward for `nxt` starts now; meanwhile plan the step after it
                if not nxt:
                    break
                t0 = time.perf_counter()
                act = [i for i in range(n) if cur_mask >> i & 1]
                m_ = self.mu_vec(world, slots, trees, prm, act, W)[Kg]
                mu_vec = [0.0] * n
                for r, i in enumerate(act):
                    mu_vec[i] = float(m_[r])
                u, el = self.build_u(mu_vec, dG, depth, Dr, act, n, world)
                xs = self.DP(u, el, cur_mask, cost_pop, Theta)
                after = best_sub(xs[1], nxt)[0] if len(xs) > 1 else 0
                ms = (time.perf_counter() - t0) * 1000
                self.plan_times.append(ms / 1000)
                time_ += self.charge(ms, overlap_ms=world.c_d(bin(nxt).count("1"), W))
                cur_mask, nxt = nxt, after
            else:
                t0 = time.perf_counter()
                m_ = self.mu_vec(world, slots, trees, prm, S, W)[Kg]
                mu_vec = [0.0] * n
                for r, i in enumerate(S):
                    mu_vec[i] = float(m_[r])
                u, el = self.build_u(mu_vec, dG, depth, Dr, S, n, world)
                xs = self.DP(u, el, cur_mask, cost_pop, Theta)
                cur_mask = best_sub(xs[0], cur_mask)[0]
                ms = (time.perf_counter() - t0) * 1000
                self.plan_times.append(ms / 1000)
                time_ += self.charge(ms)
        Kv = max(KS, key=lambda K: sum(t.G(K) for t in trees) - Theta * world.T_t(n, K))
        self.Kv_hist[Kv] += 1
        com = [min(t.accepted(t.topk(Kv, True)) + 1, rem[i]) for i, t in enumerate(trees)]
        time_ += rtt + world.T_t(n, Kv)
        if self.mode == "learned":
            self.learn(world, slots, trees, com)
        self.ema(sum(com), time_)
        return com, time_

    def learn(self, world, slots, trees, com):
        if not self.dyn:
            return super().learn(world, slots, trees, com)
        saved = {}
        super().learn(world, slots, trees, com)           # updates a, g, beta, labels (and a static delta)
        sd2 = world.sd ** 2
        for (r, j), c in zip(slots, com):
            xs, ys = self.lab.get(r, ([], []))
            if not xs:
                continue
            X, Y = np.asarray(xs[-60:]), np.asarray(ys[-60:])
            wts = getattr(self, "dyn_decay", 0.9) ** np.arange(len(X))[::-1]   # per label = per committed position (M1)
            d = self.dhat.get(r, 0.0)
            for _ in range(3):
                p = sig(self.ahat_ + self.ghat * X + d)
                grad = (wts * (Y - p)).sum() - d / sd2
                hess = (wts * p * (1 - p)).sum() + 1 / sd2
                d += grad / hess
            self.dhat[r] = float(np.clip(d, -3.0, 3.0))
            if not hasattr(self, "dvar"):
                self.dvar = {}
            self.dvar[r] = float(1.0 / hess)


class SpecSwitch:
    def __init__(self, inner, margin=0.03, win=20, probe_every_ms=3000.0, probe_rounds=15):
        self.inner, self.m, self.win, self.pe, self.pr = inner, margin, win, probe_every_ms, probe_rounds
        self.mode, self.clock, self.last_probe, self.probe_left = "spec", 0.0, 0.0, 0
        self.sw, self.aw = [], []
        self.spec_share = [0, 0]

    @staticmethod
    def rate(wnd):
        return sum(a for a, _ in wnd) / max(1e-9, sum(t for _, t in wnd))

    def round(self, world, slots, rng, rtt, rnd):
        n = len(slots)
        ar_rate = self.rate(self.aw[-self.win:]) if len(self.aw) >= 5 else n / world.T_ar(n)
        if self.mode == "ar" and self.probe_left == 0 and self.clock - self.last_probe >= self.pe:
            self.probe_left, self.sw = self.pr, []
        spec = self.mode == "spec" or self.probe_left > 0
        if spec:
            com, t = self.inner.round(world, slots, rng, rtt, rnd)
            self.sw.append((sum(com), t)); self.sw = self.sw[-self.win:]
        else:
            com, t = [1] * n, world.T_ar(n)
            self.aw.append((n, t)); self.aw = self.aw[-self.win:]
        self.clock += t
        self.spec_share[0] += t if spec else 0.0; self.spec_share[1] += t
        if self.mode == "spec" and len(self.sw) >= self.win and self.rate(self.sw) < (1 - self.m) * ar_rate:
            self.mode, self.last_probe = "ar", self.clock
        elif self.probe_left > 0:
            self.probe_left -= 1
            if self.probe_left == 0:
                self.mode = "spec" if self.rate(self.sw) > (1 + self.m) * ar_rate else "ar"
                self.last_probe = self.clock
        return com, t


class StaticSwitch:
    def __init__(self, inner, b_min=7):
        self.inner, self.b_min = inner, b_min

    def round(self, world, slots, rng, rtt, rnd):
        n = len(slots)
        if n >= self.b_min:
            return [1] * n, world.T_ar(n)
        return self.inner.round(world, slots, rng, rtt, rnd)


def simulate_t(world, policy, seed=1, period_ms=None):
    """feasibility.simulate with RTT alternating every period_ms of simulated time (world.rtt is a pair)."""
    if period_ms is None:
        return F.simulate(world, policy, seed)
    rng = np.random.default_rng(seed)
    b = world.b
    slots, nxt = [], 0
    while len(slots) < b and nxt < len(world.reqs):
        slots.append([nxt, 0]); nxt += 1
    tok = tt = 0.0
    rnd = 0
    while slots:
        rtt = world.rtt[0] if int(tt // period_ms) % 2 == 0 else world.rtt[1]
        com, t = policy.round(world, slots, rng, rtt, rnd)
        tok += sum(com); tt += t; rnd += 1
        keep = []
        for s, c in zip(slots, com):
            s[1] += c
            if s[1] < world.L:
                keep.append(s)
            elif nxt < len(world.reqs):
                keep.append([nxt, 0]); nxt += 1
        slots = keep
    return 1000.0 * tok / tt, {"rounds": rnd}


# ---------------- fast planner (library pool + one-shot subset-max) ----------------
_FAST = {}


def fast_tables(n):
    if n not in _FAST:
        S = np.arange(1 << n)
        subs, offs = [], []
        for s in range(1 << n):
            offs.append(len(subs))
            t = s
            while True:
                subs.append(t)
                if t == 0:
                    break
                t = (t - 1) & s
        B = ((S[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
        _FAST[n] = (S, B, B.sum(1), np.asarray(subs), np.asarray(offs))
    return _FAST[n]


def nested_dp_fast(u, elig, act_mask, cost_of_pop, Theta):
    n, H = u.shape
    S, B, pop, subs, offs = fast_tables(n)
    base = (Theta * np.array([cost_of_pop(int(k)) if k else 0.0 for k in range(n + 1)]))[pop.astype(int)] + 1e-9 * pop
    eints = (elig * (1 << np.arange(n))[:, None]).sum(0)
    U = B @ u                                   # (2^n, H) all layers at once
    V = np.zeros(S.size)
    xs = [None] * H
    for k in range(H - 1, -1, -1):
        x = U[:, k] - base + V
        x[(S & ~int(eints[k])) != 0] = -np.inf
        x[0] = 0.0
        xs[k] = x
        V = np.maximum.reduceat(x[subs], offs)
    return xs


class ClearTreeFast(ClearTreeV):
    POOL = 512

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.poolarr = None
        self.DP = nested_dp_fast

    def mu_vec(self, world, slots, trees, prm, idxs, W):
        if self.mode == "oracle":
            if self.poolarr is None:
                self.poolarr = np.stack([F.gen_dists(self.prng, r["s"], self.POOL) for r in world.reqs])
            self._ridx = np.array([slots[i][0] for i in idxs])
        return super().mu_vec(world, slots, trees, prm, idxs, W)


def _draw_pool(self, world, slots, idxs, nF):
    N = F.N_SNAP
    j = self.prng.integers(0, self.POOL, (len(idxs), N, nF))
    return self.poolarr[self._ridx[:, None, None], j]


_orig_mu = ClearTreeV.mu_vec


def _mu_vec_fast(self, world, slots, trees, prm, idxs, W):
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
    N = F.N_SNAP
    if self.mode == "oracle":
        D = _draw_pool(self, world, slots, idxs, nF)
    else:
        D = np.empty((nR, N, nF, C))
        for r, i in enumerate(idxs):
            D[r] = self.draws(world, slots[i][0], nF)
    p = sig(a + g * logit(D[..., 0]) + dl[:, None, None])
    w = D if beta == 1.0 else D ** beta
    A = p[..., None] * w / w.sum(-1, keepdims=True)
    sc = (fs[:, None, :, None] * D).reshape(nR, N, nF * C)
    pv = (fp[:, None, :, None] * A).reshape(nR, N, nF * C)
    if nF * C > W:
        sel = np.argpartition(-sc, W - 1, axis=-1)[..., :W]
        pv = np.take_along_axis(pv, sel, -1)
    kmax = max(KS)
    top = -np.sort(-np.concatenate([np.broadcast_to(cur[:, None, :], (nR, N, nT)), pv], -1), -1)[..., :kmax]
    cum = np.cumsum(top, -1).mean(1)
    curc = np.cumsum(-np.sort(-cur, -1)[:, :kmax], -1)
    return {K: cum[:, min(K, cum.shape[1]) - 1] - curc[:, min(K, curc.shape[1]) - 1] for K in KS}


ClearTreeFast.mu_vec = lambda self, world, slots, trees, prm, idxs, W: (
    setattr(self, "poolarr", self.poolarr if self.poolarr is not None or self.mode != "oracle"
            else np.stack([F.gen_dists(self.prng, r["s"], self.POOL) for r in world.reqs])),
    setattr(self, "_ridx", np.array([slots[i][0] for i in idxs])),
    _mu_vec_fast(self, world, slots, trees, prm, idxs, W))[2]
