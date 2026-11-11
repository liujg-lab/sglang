"""Round 9 policies on decoupled trees (layer width W, verify the top K nodes by Draft path score).

BatchAccount  "batch accounting" stop rule (design adjustment 2): after each layer, the gain of the next layer
              for request i is predicted as c_i * dE_i, where dE_i is the increase of its top-K Draft path mass
              from the layer just added (the same signal the real OPT rule thresholds) and c_i is an online
              calibration (accepted draft tokens / predicted top-K mass, per request, shrunk to the pooled value).
              The batch keeps the best m requests, with m restricted to the active count or a smaller graph
              bucket (8, 4, 2, 1), when sum of their gains - Theta * layer_cost(m) > 0; otherwise it stops.
              Theta is the EMA of committed tokens per ms over whole rounds; layer costs are learned per bucket.
              Exited requests do not come back in the same round. No per-workload tuning.
OPTRhoTau     acceptance -> tau rule (design adjustment 4): a static decoupled OPT whose tau is looked up from a
              table (fitted offline on the round-8 grid) by b and by an online estimate of the per-depth
              acceptance rho = sum(a) / (sum(a) + #rounds that stopped inside the tree), which depends little
              on tau. Batch-level tau (one value for the whole batch).
"""
import numpy as np
import feasibility as F
import ext2 as X
import ext7 as S


def _mass(t, K):
    return float(np.sum(np.sort(t.scores)[-K:])) if t.scores else 0.0


def _tree_depth(t):
    return max(t.dep)


class _Stats:
    """Per-depth acceptance estimate shared by both rules."""

    def __init__(self):
        self.num = self.den = 0.0

    def update(self, trees, ver_acc):
        for t, a in zip(trees, ver_acc):
            d = _tree_depth(t)
            self.num += a
            self.den += a + (1 if a < d else 0)

    def rho(self):
        return self.num / self.den if self.den > 0 else float("nan")


class OPTRhoTau(X.OPT2):
    def __init__(self, W, K, table, thresholds, warm=20, alpha=0.05, fixed_tau=None):
        super().__init__(W, 0.4)
        self.W, self.K, self.table, self.th, self.warm, self.alpha, self.fixed = W, K, table, thresholds, warm, alpha, fixed_tau
        self.ema_num = self.ema_den = None
        self.rounds = 0
        self.cum = _Stats()
        self.taus = []

    def _rho_ema(self):
        return self.ema_num / self.ema_den if self.ema_den else float("nan")

    def _tau(self, b):
        if self.fixed is not None:
            return self.fixed
        if self.rounds < self.warm or self.ema_den is None:
            return 0.4
        r = self._rho_ema()
        cls = "low" if r < self.th[0] else ("mid" if r < self.th[1] else "high")
        bk = min(self.table, key=lambda x: abs(x - b))
        return self.table[bk][cls]

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        tau = self._tau(len(slots))
        self.taus.append(tau)
        trees = [X.Tree2(world.reqs[r], j, world) for r, j in slots]
        params = (world.a, world.g, world.beta, 0.0)
        for t in trees:
            t.expand(self.W, rng, params)
        rem = [world.reqs[r]["L"] - j for r, j in slots]
        layers = S.grow_real_k(world, trees, self.W, self.K, tau, rem, rng)
        time_ = sum(world.real_cd(rng, n, self.W) for n in layers)
        acc = [t.accepted(t.topk(self.K, False)) for t in trees]
        com = [min(a + 1, rem[i]) for i, a in enumerate(acc)]
        num = sum(acc); den = sum(a + (1 if a < _tree_depth(t) else 0) for t, a in zip(trees, acc))
        if self.ema_den is None:
            self.ema_num, self.ema_den = float(num), float(den)
        else:
            self.ema_num = (1 - self.alpha) * self.ema_num + self.alpha * num
            self.ema_den = (1 - self.alpha) * self.ema_den + self.alpha * den
        self.cum.update(trees, acc)
        self.rounds += 1
        return com, time_ + world.real_rtt(rng, rtt) + world.real_T(rng, len(slots), self.K)


class BatchAccount(X.OPT2):
    def __init__(self, W, K, warm=10, alpha=0.05, prior=3.0, buckets=(1, 2, 4, 8)):
        super().__init__(W, 0.4)
        self.W, self.K, self.warm, self.alpha, self.prior, self.buckets = W, K, warm, alpha, prior, buckets
        self.tok = self.ms = None
        self.cost = {}
        self.req_a, self.req_e = {}, {}
        self.g_a = self.g_e = 0.0
        self.rounds = 0
        self.stats = dict(layers=0, exits=0, stops=0)

    def _theta(self):
        return self.tok / self.ms if self.ms else None

    def _c(self, r):
        cg = self.g_a / self.g_e if self.g_e > 0 else 1.0
        return (self.req_a.get(r, 0.0) + self.prior * cg) / (self.req_e.get(r, 0.0) + self.prior)

    def _layer_cost(self, m):
        bm = F.bucket(m)
        if bm in self.cost:
            return self.cost[bm]
        pts = sorted(self.cost.items())
        if len(pts) >= 2:
            (x0, y0), (x1, y1) = pts[0], pts[-1]
            slope = max(0.0, (y1 - y0) / (x1 - x0))
            return max(1e-3, y0 + slope * (bm - x0))
        if pts:
            x0, y0 = pts[0]
            return y0 * (0.5 + 0.5 * bm / x0)
        return 1.0

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        trees = [X.Tree2(world.reqs[r], j, world) for r, j in slots]
        rids = [r for r, _ in slots]
        params = (world.a, world.g, world.beta, 0.0)
        for t in trees:
            t.expand(self.W, rng, params)
        rem = [world.reqs[r]["L"] - j for r, j in slots]
        K = self.K
        e_prev = [0.0] * len(trees)
        e_now = [_mass(t, K) for t in trees]
        active, depth, time_ = list(range(len(trees))), 1, 0.0
        theta = self._theta()
        while active:
            cand = [i for i in active if depth < world.DMAX - 1 and depth + 1 <= max(1, rem[i] - 1)]
            if not cand:
                break
            if self.rounds < self.warm or theta is None:
                keep = [i for i in cand if e_now[i] - e_prev[i] > 0.4]
            else:
                g = {i: self._c(rids[i]) * (e_now[i] - e_prev[i]) for i in cand}
                order = sorted(cand, key=lambda i: g[i], reverse=True)
                opts = sorted({len(order)} | {x for x in self.buckets if x < len(order)})
                best_m, best_v = 0, 0.0
                for m in opts:
                    v = sum(g[i] for i in order[:m]) - theta * self._layer_cost(m)
                    if v > best_v:
                        best_m, best_v = m, v
                keep = order[:best_m]
                self.stats["exits"] += len(cand) - len(keep)
            if not keep:
                self.stats["stops"] += 1
                break
            for i in keep:
                trees[i].expand(self.W, rng, params)
                e_prev[i] = e_now[i]
                e_now[i] = _mass(trees[i], K)
            dt = world.real_cd(rng, len(keep), self.W)
            bm = F.bucket(len(keep))
            self.cost[bm] = dt if bm not in self.cost else (1 - 0.1) * self.cost[bm] + 0.1 * dt
            time_ += dt
            self.stats["layers"] += 1
            active = keep
            depth += 1
        acc = [t.accepted(t.topk(K, False)) for t in trees]
        com = [min(a + 1, rem[i]) for i, a in enumerate(acc)]
        t_round = time_ + world.real_rtt(rng, rtt) + world.real_T(rng, len(slots), K)
        tok = float(sum(com))
        if self.ms is None:
            self.tok, self.ms = tok, t_round
        else:
            self.tok = (1 - self.alpha) * self.tok + self.alpha * tok
            self.ms = (1 - self.alpha) * self.ms + self.alpha * t_round
        for i, t in enumerate(trees):
            e = _mass(t, K)
            self.req_a[rids[i]] = self.req_a.get(rids[i], 0.0) + acc[i]
            self.req_e[rids[i]] = self.req_e.get(rids[i], 0.0) + e
            self.g_a += acc[i]; self.g_e += e
        self.rounds += 1
        return com, t_round
