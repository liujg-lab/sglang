"""ClearTree feasibility simulation (simplified, synthetic).

World (greedy semantics): every request has a fixed "true" continuation. At text position p the Draft
proposes C candidates with probabilities Q[p] (sorted); the Target's greedy token is candidate H[p]
(or -1 = not in top-C). P(H >= 0) = sigmoid(a + g*logit(q1) + delta_i [+ eps_p]), and given a hit the
candidate is chosen with prob ~ q^beta. eps_p (per-position difficulty) exists only in the misspecified
world and is unknown to every planner. Off-path nodes get fresh random Draft distributions.

Costs (ms): Target verify t0 + t1*b*(K+1); Draft layer cD0 + cD1*bucket(n)*W; RTT per round;
layer 1 is free (post-reply forward overlaps the wait); ClearTree pays cP per planning call.
Misspecified world: eps_p is AR(1) (rho 0.9, sd 1.0) and the Target's choice among candidates follows
q^2 while every planner model assumes q^1.

Policies: AR (Target only), OPT(N, tau) (width N, verify N by Draft score, grow while frontier max
score >= tau), ClearTree (fixed-K planning + adaptive K verification, nested subset DP with
snapshot-sampled next-layer increments; 'oracle' knows a, g, beta, delta_i and the Draft generator,
'learned' fits them online from accepted-parent labels and samples a per-request library).
Simplifications: no C_dec, bias correction, probes, slow loop, KV limits, stop tokens.
"""
import math, os, sys, time
import numpy as np

C = 8
KS = (4, 8, 16)
BUCKETS = (1, 2, 4, 8)
N_SNAP = 16


def bucket(n):
    for x in BUCKETS:
        if x >= n:
            return x
    return n


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def gen_dists(rng, s, n):
    q1 = rng.beta(s * 6.0, (1 - s) * 6.0, n)
    rho = rng.uniform(0.5, 0.95, n)
    g = rng.uniform(0.4, 0.8, n)
    pw = g[:, None] ** np.arange(C - 1)[None, :]
    rest = pw / pw.sum(1, keepdims=True) * ((1 - q1) * rho)[:, None]
    Q = np.concatenate([q1[:, None], rest], 1)
    return -np.sort(-Q, 1)


def ahat(D, a, g, beta, delta):
    p = sig(a + g * logit(D[..., 0]) + delta)
    w = D ** beta
    return p[..., None] * w / w.sum(-1, keepdims=True)


class World:
    def __init__(self, b, rtt, misspec, seed=0, L=256, n_req=None, rtt_switch=0, **over):
        self.b, self.rtt, self.misspec, self.rtt_switch = b, rtt, misspec, rtt_switch
        self.t0, self.t1 = 12.0, 0.05
        self.cD0, self.cD1 = 1.5, 0.02
        self.DMAX = 10
        self.a, self.g, self.beta, self.sd = 0.5, 1.0, 1.0, 0.6
        self.eps = 1.0 if misspec else 0.0        # AR(1) per-position difficulty, unknown to planners
        self.eps_rho = 0.9
        self.beta_true = 2.0 if misspec else 1.0  # Target concentrates on top candidates more than the model
        self.L = L
        for k, v in over.items():
            setattr(self, k, v)
        rng = np.random.default_rng(seed)
        self.reqs = [self.make_req(rng) for _ in range(n_req or max(16, 8 * b))]

    def make_req(self, rng):
        s, d = rng.uniform(*getattr(self, "s_range", (0.35, 0.85))), rng.normal(0, self.sd)
        n = self.L + self.DMAX + 2
        Q = gen_dists(rng, s, n)
        if self.eps:
            e = np.empty(n); e[0] = rng.normal(0, self.eps)
            z = rng.normal(0, self.eps * math.sqrt(1 - self.eps_rho ** 2), n)
            for k in range(1, n):
                e[k] = self.eps_rho * e[k - 1] + z[k]
        else:
            e = 0.0
        hit = rng.random(n) < sig(self.a + self.g * logit(Q[:, 0]) + d + e)
        w = Q ** self.beta_true
        w = w / w.sum(1, keepdims=True)
        idx = np.minimum((np.cumsum(w, 1) < rng.random(n)[:, None]).sum(1), C - 1)
        return dict(s=s, delta=d, Q=Q, H=np.where(hit, idx, -1))

    def rtt_at(self, rnd):
        if not self.rtt_switch:
            return self.rtt
        return self.rtt[0] if (rnd // self.rtt_switch) % 2 == 0 else self.rtt[1]

    def T_t(self, b, K):
        return self.t0 + self.t1 * b * (K + 1)

    def c_d(self, n, W):
        return self.cD0 + self.cD1 * bucket(n) * W

    def T_ar(self, b):
        return self.t0 + self.t1 * b


class Tree:
    def __init__(self, req, j):
        self.req = req
        self.score, self.pi, self.pos = [1.0], [1.0], [j]
        self.kids = [dict()]
        self.frontier = [0]
        self.pis, self.scores, self.ids = [], [], []

    def expand(self, W, rng, params):
        F = self.frontier
        Q, H = self.req["Q"], self.req["H"]
        rows, off = [], 0
        for v in F:
            p = self.pos[v]
            if p >= 0:
                rows.append(Q[p])
            else:
                rows.append(None); off += 1
        if off:
            gen = gen_dists(rng, self.req["s"], off)
            k = 0
            for t in range(len(rows)):
                if rows[t] is None:
                    rows[t] = gen[k]; k += 1
        D = np.stack(rows)
        A = ahat(D, *params)
        fs = np.fromiter((self.score[v] for v in F), float, len(F))
        fp = np.fromiter((self.pi[v] for v in F), float, len(F))
        sc = (fs[:, None] * D).ravel()
        pv = (fp[:, None] * A).ravel()
        sel = np.argpartition(-sc, W - 1)[:W] if sc.size > W else np.arange(sc.size)
        new = []
        for flat in sel:
            fi, c = divmod(int(flat), C)
            v = F[fi]
            nid = len(self.score)
            self.score.append(float(sc[flat])); self.pi.append(float(pv[flat]))
            p = self.pos[v]
            self.pos.append(p + 1 if (p >= 0 and H[p] == c) else -1)
            self.kids.append(dict()); self.kids[v][c] = nid
            self.pis.append(float(pv[flat])); self.scores.append(float(sc[flat])); self.ids.append(nid)
            new.append(nid)
        self.frontier = new
        return D

    def topk(self, K, by_pi):
        vals = self.pis if by_pi else self.scores
        if len(vals) <= K:
            return set(self.ids)
        sel = np.argpartition(-np.asarray(vals), K - 1)[:K]
        return {self.ids[i] for i in sel}

    def G(self, K):
        if not self.pis:
            return 0.0
        arr = np.sort(np.asarray(self.pis))[::-1]
        return float(arr[:K].sum())

    def accepted(self, ver):
        v, a, H = 0, 0, self.req["H"]
        while True:
            h = H[self.pos[v]]
            if h < 0:
                break
            c = self.kids[v].get(int(h))
            if c is None or c not in ver:
                break
            a += 1; v = c
        return a


def simulate(world, policy, seed=1):
    rng = np.random.default_rng(seed)
    b = world.b
    slots, nxt = [], 0
    while len(slots) < b and nxt < len(world.reqs):
        slots.append([nxt, 0]); nxt += 1
    tok = tt = 0.0
    rnd = 0
    stats = {"acc": 0, "rounds": 0}
    while slots:
        com, t = policy.round(world, slots, rng, world.rtt_at(rnd), rnd)
        tok += sum(com); tt += t; rnd += 1
        stats["rounds"] += 1
        keep = []
        for s, c in zip(slots, com):
            s[1] += c
            if s[1] < world.L:
                keep.append(s)
            elif nxt < len(world.reqs):
                keep.append([nxt, 0]); nxt += 1
        slots = keep
    return 1000.0 * tok / tt, stats


class AR:
    def round(self, world, slots, rng, rtt, rnd):
        return [1] * len(slots), world.T_ar(len(slots))


class OPT:
    def __init__(self, N, tau):
        self.N, self.tau = N, tau

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        trees = [Tree(world.reqs[r], j) for r, j in slots]
        params = (world.a, world.g, world.beta, 0.0)
        for t in trees:
            t.expand(self.N, rng, params)
        rem = [world.L - j for _, j in slots]
        active, depth, time_ = list(range(len(trees))), 1, 0.0
        while active and depth < world.DMAX:
            active = [i for i in active
                      if max(trees[i].score[v] for v in trees[i].frontier) >= self.tau and depth + 1 <= max(1, rem[i] - 1)]
            if not active:
                break
            for i in active:
                trees[i].expand(self.N, rng, params)
            time_ += world.c_d(len(active), self.N)
            depth += 1
        com = []
        for i, t in enumerate(trees):
            com.append(min(t.accepted(t.topk(self.N, False)) + 1, rem[i]))
        if trees_out is not None:
            trees_out.extend(trees)
        return com, time_ + rtt + world.T_t(len(slots), self.N)


_DP_CACHE = {}


def dp_tables(n):
    if n not in _DP_CACHE:
        S = np.arange(1 << n)
        B = ((S[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
        pop = B.sum(1)
        drop = [S & ~(1 << i) for i in range(n)]
        _DP_CACHE[n] = (S, B, pop, drop)
    return _DP_CACHE[n]


def nested_dp(u, elig, act_mask, cost_of_pop, Theta):
    """u, elig: (n, H). Returns (best first-step mask, its value) over subsets of act_mask."""
    n, H = u.shape
    S, B, pop, drop = dp_tables(n)
    cost = np.array([cost_of_pop(int(k)) if k else 0.0 for k in range(n + 1)])[pop.astype(int)]
    V = np.zeros(S.size)
    for k in range(H - 1, -1, -1):
        x = B @ u[:, k] - Theta * cost + V - 1e-9 * pop
        bad = (B @ (~elig[:, k]).astype(float)) > 0
        x[bad] = -np.inf
        x[0] = 0.0
        if k == 0:
            x0 = x
        F = x.copy()
        for i in range(n):
            F = np.maximum(F, F[drop[i]])
        V = F
    ok = (S & ~act_mask) == 0
    xs = np.where(ok, x0, -np.inf)
    best = int(np.argmax(xs))
    return (best, float(xs[best])) if xs[best] > 0 else (0, 0.0)


class ClearTree:
    def __init__(self, W, mode="oracle", cP=0.1, cold=40, learn_beta=False):
        self.W, self.mode, self.cP, self.cold, self.learn_beta = W, mode, cP, cold, learn_beta
        self.bhat, self.hits, self.nlearn = 1.0, [], 0
        self.tokE = self.timeE = None
        self.plan_times = []
        self.Kv_hist = {K: 0 for K in KS}
        self.prng = np.random.default_rng(12345)
        if mode == "learned":
            self.ahat_, self.ghat = 0.0, 1.0
            self.lab = {}       # req -> (xs, ys)
            self.dhat = {}
            self.buf = {}       # req -> list of rows
            self.gbuf = []
            self.boot = OPT(8, 0.4)

    def theta(self, world, n):
        if self.tokE is None:
            return n / world.T_ar(n)
        return self.tokE / self.timeE

    def params(self, world, r):
        if self.mode == "oracle":
            return (world.a, world.g, world.beta, world.reqs[r]["delta"])
        return (self.ahat_, self.ghat, self.bhat, self.dhat.get(r, 0.0))

    def draws(self, world, r, nF):
        if self.mode == "oracle":
            return gen_dists(self.prng, world.reqs[r]["s"], N_SNAP * nF).reshape(N_SNAP, nF, C)
        rows = self.buf.get(r)
        src = np.asarray(rows) if rows is not None and len(rows) >= 16 else np.asarray(self.gbuf[-4000:])
        idx = self.prng.integers(0, len(src), N_SNAP * nF)
        return src[idx].reshape(N_SNAP, nF, C)

    def mu_all(self, world, r, tree, params, W):
        F = tree.frontier
        nF = len(F)
        D = self.draws(world, r, nF)
        A = ahat(D, *params)
        fs = np.fromiter((tree.score[v] for v in F), float, nF)
        fp = np.fromiter((tree.pi[v] for v in F), float, nF)
        sc = (fs[None, :, None] * D).reshape(N_SNAP, -1)
        pv = (fp[None, :, None] * A).reshape(N_SNAP, -1)
        if sc.shape[1] > W:
            sel = np.argpartition(-sc, W - 1, axis=1)[:, :W]
            pv = np.take_along_axis(pv, sel, 1)
        cur = np.asarray(tree.pis)
        allv = np.concatenate([np.broadcast_to(cur, (N_SNAP, cur.size)), pv], 1)
        cum = np.cumsum(-np.sort(-allv, 1), 1)
        curc = np.cumsum(np.sort(cur)[::-1])
        return {K: float(cum[:, min(K, cum.shape[1]) - 1].mean() - curc[min(K, curc.size) - 1]) for K in KS}

    def build_u(self, mus, dGs, depth, Dr, active, n, world):
        H = max(1, world.DMAX - depth)
        u = np.zeros((n, H)); el = np.zeros((n, H), bool)
        for i in active:
            r = min(0.9, max(0.0, mus[i] / dGs[i])) if dGs[i] > 1e-12 else 0.0
            for k in range(H):
                if depth + 1 + k <= Dr[i]:
                    u[i, k] = mus[i] * (r ** k); el[i, k] = True
        return u, el

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
            D = t.expand(W, rng, p)
            self.store(slots, trees, t, D)
        rem = [world.L - j for _, j in slots]
        Dr = [min(world.DMAX, max(0, x - 1)) for x in rem]
        depth, time_ = 1, 2 * self.cP
        cost_pop = lambda k: world.c_d(k, W) + self.cP
        t0 = time.perf_counter()
        mus = [self.mu_all(world, slots[i][0], trees[i], prm[i], W) for i in range(n)]
        Gs = {K: [t.G(K) for t in trees] for K in KS}
        best = None
        for K in KS:
            u, el = self.build_u([m[K] for m in mus], Gs[K], depth, Dr, range(n), n, world)
            mask, val = nested_dp(u, el, (1 << n) - 1, cost_pop, Theta)
            J = sum(Gs[K]) - Theta * world.T_t(n, K) + val
            if best is None or J > best[0]:
                best = (J, K, mask)
        _, Kg, mask = best
        self.plan_times.append((time.perf_counter() - t0) / len(KS))
        lastG = Gs[Kg][:]
        dG = Gs[Kg][:]
        while mask and depth < world.DMAX:
            S = [i for i in range(n) if mask >> i & 1]
            for i in S:
                D = trees[i].expand(W, rng, prm[i])
                self.store(slots, trees, trees[i], D)
                g = trees[i].G(Kg); dG[i] = g - lastG[i]; lastG[i] = g
            time_ += world.c_d(len(S), W) + self.cP
            depth += 1
            if depth >= world.DMAX:
                break
            t0 = time.perf_counter()
            mus = {i: self.mu_all(world, slots[i][0], trees[i], prm[i], W)[Kg] for i in S}
            mu_vec = [mus.get(i, 0.0) for i in range(n)]
            u, el = self.build_u(mu_vec, dG, depth, Dr, S, n, world)
            mask, _ = nested_dp(u, el, mask, cost_pop, Theta)
            self.plan_times.append(time.perf_counter() - t0)
        Kv = max(KS, key=lambda K: sum(t.G(K) for t in trees) - Theta * world.T_t(n, K))
        self.Kv_hist[Kv] += 1
        com = [min(t.accepted(t.topk(Kv, True)) + 1, rem[i]) for i, t in enumerate(trees)]
        time_ += rtt + world.T_t(n, Kv)
        if self.mode == "learned":
            self.learn(world, slots, trees, com)
        self.ema(sum(com), time_)
        return com, time_

    def ema(self, tok, t, alpha=0.05):
        if self.tokE is None:
            self.tokE, self.timeE = tok, t
        else:
            self.tokE = (1 - alpha) * self.tokE + alpha * tok
            self.timeE = (1 - alpha) * self.timeE + alpha * t

    def store(self, slots, trees, tree, D):
        if self.mode != "learned":
            return
        r = None
        for (rr, _), t in zip(slots, trees):
            if t is tree:
                r = rr
        rows = self.buf.setdefault(r, [])
        for row in D:
            rows.append(row); self.gbuf.append(row)
        if len(rows) > 256:
            del rows[:-256]
        if len(self.gbuf) > 8000:
            del self.gbuf[:-4000]

    def learn(self, world, slots, trees, com):
        if self.mode != "learned":
            return
        sd2 = world.sd ** 2   # prior sd of delta assumed known (could be fitted)
        for (r, j), t, c in zip(slots, trees, com):
            if not t.pis and not self.gbuf:
                pass
            if r not in self.buf or not self.buf[r]:
                for row in world.reqs[r]["Q"][j:j + 1]:
                    self.buf.setdefault(r, []).append(row); self.gbuf.append(row)
            Q, H = world.reqs[r]["Q"], world.reqs[r]["H"]
            ps = range(j, j + c)        # accepted parents: root and accepted nodes
            xs, ys = self.lab.setdefault(r, ([], []))
            for p in ps:
                x = float(logit(Q[p][0])); y = float(H[p] >= 0)
                d = self.dhat.get(r, 0.0)
                e = y - float(sig(self.ahat_ + self.ghat * x + d))
                self.ahat_ += 0.01 * e; self.ghat = max(0.0, self.ghat + 0.01 * e * x)
                xs.append(x); ys.append(y)
                if self.learn_beta and H[p] >= 0:
                    self.hits.append((np.log(np.clip(Q[p], 1e-9, 1)), int(H[p])))
            X, Y = np.asarray(xs[-200:]), np.asarray(ys[-200:])
            d = self.dhat.get(r, 0.0)
            for _ in range(3):
                p = sig(self.ahat_ + self.ghat * X + d)
                grad = (Y - p).sum() - d / sd2
                hess = (p * (1 - p)).sum() + 1 / sd2
                d += grad / hess
            self.dhat[r] = float(np.clip(d, -2.5, 2.5))
        self.nlearn += 1
        if self.learn_beta and self.nlearn % 10 == 0 and len(self.hits) > 50:
            hs = self.hits[-3000:]
            LQ = np.stack([h[0] for h in hs]); cidx = np.array([h[1] for h in hs])
            lc = LQ[np.arange(len(hs)), cidx]
            b_ = self.bhat
            for _ in range(3):
                w = np.exp(b_ * LQ); w /= w.sum(1, keepdims=True)
                m1 = (w * LQ).sum(1); m2 = (w * LQ * LQ).sum(1)
                g1 = (lc - m1).sum(); h1 = -(m2 - m1 * m1).sum()
                if h1 < -1e-9:
                    b_ = float(np.clip(b_ - g1 / h1, 0.2, 5.0))
            self.bhat = b_


TAUS = (0.9, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02)


def scenario(name, world, cps=(0.1, 1.0)):
    out = {"name": name}
    out["AR"] = simulate(world, AR())[0]
    grid = {}
    for N in KS:
        for tau in TAUS:
            grid[(N, tau)] = simulate(world, OPT(N, tau))[0]
    out["OPT(8,0.4)"] = grid[(8, 0.4)]
    bk = max(grid, key=grid.get)
    out["OPT best"] = (grid[bk], bk)
    ct = {}
    pts = {}
    for W in KS:
        pol = ClearTree(W, "oracle", cP=cps[0])
        ct[W] = simulate(world, pol)[0]
        pts[W] = pol.plan_times
        if W == 8:
            out["Kv share"] = {K: round(v / max(1, sum(pol.Kv_hist.values())), 2) for K, v in pol.Kv_hist.items()}
    bw = max(ct, key=ct.get)
    out["CT oracle"] = {W: ct[W] for W in KS}
    out["CT oracle best"] = (ct[bw], bw)
    out["CT learned"] = simulate(world, ClearTree(bw, "learned", cP=cps[0]))[0]
    out[f"CT oracle cP={cps[1]}"] = simulate(world, ClearTree(bw, "oracle", cP=cps[1]))[0]
    allpt = np.asarray(pts[bw]) * 1000
    out["plan ms/call (mean, p99)"] = (float(allpt.mean()), float(np.percentile(allpt, 99)))
    return out


def fmt(o):
    ob, (N, tau) = o["OPT best"][0], o["OPT best"][1]
    cb, W = o["CT oracle best"]
    s = (f"{o['name']}: AR {o['AR']:.0f} | OPT(8,.4) {o['OPT(8,0.4)']:.0f} | OPT best {ob:.0f} (N={N},tau={tau}) | "
         f"CT oracle best {cb:.0f} (W={W}, {100*(cb/ob-1):+.1f}% vs best static) | CT learned {o['CT learned']:.0f} "
         f"({100*(o['CT learned']/ob-1):+.1f}%) | CT cP=1ms {[v for k, v in o.items() if k.startswith('CT oracle cP')][0]:.0f} | "
         f"plan {o['plan ms/call (mean, p99)'][0]:.2f}/{o['plan ms/call (mean, p99)'][1]:.2f} ms | Kv {o.get('Kv share')}")
    s += " | CT by W " + ", ".join(f"{w}:{v:.0f}" for w, v in o["CT oracle"].items())
    return s


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    specs = []
    for mis in (False, True):
        for b in (1, 4, 8):
            for rtt in (2.0, 40.0):
                specs.append((f"{'misspec' if mis else 'wellspec'} b={b} RTT={rtt:g}ms", dict(b=b, rtt=rtt, misspec=mis)))
        specs.append((f"{'misspec' if mis else 'wellspec'} b=4 RTT 2<->40ms every 150 rounds",
                      dict(b=4, rtt=(2.0, 40.0), misspec=mis, rtt_switch=150)))
    for k, (name, kw) in enumerate(specs):
        if which != "all" and str(k % 2) != which:
            continue
        t0 = time.time()
        o = scenario(name, World(seed=7, **kw))
        print(fmt(o) + f" | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
