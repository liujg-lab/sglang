"""Round 11: the adjusted rule (BatchAccount2) and a switch that is off at b=1.

BatchAccount2 = ext8.BatchAccount plus two changes:
  (1) width and verify count by acceptance: an online per-depth acceptance estimate rho (EMA over rounds of
      sum(a) / (sum(a) + #rounds that stopped inside the tree)) is classified low / mid / high (thresholds from
      round 9) and (W, K) = table[b bucket][class]; the table is fitted on seed 7 (calibrate in run11.py).
  (2) state shared across b: the price Theta is kept per concurrency b and, while a b has few rounds, blended
      with a prior from a log-log fit of Theta against b over the other b values (one other b: scale by
      (b / b')^0.8). Layer cost is one linear model in bucket and bucket * W fitted over the (bucket, W) cells,
      used where a cell has fewer than 3 observations.
SwitchB1 = ext4.SpecSwitch2 (D42 settings) that never acts at b = 1.
"""
import math
import numpy as np
import feasibility as F
import ext2 as X
import ext4 as Z
import ext8 as N

RHO_TH = (0.761, 0.917)          # round 9: midpoints of the low / mid / high workloads' rho


class SwitchB1(Z.SpecSwitch2):
    def round(self, world, slots, rng, rtt, rnd):
        if len(slots) == 1:
            com, t = self.inner.round(world, slots, rng, rtt, rnd)
            self.total_time += t
            self.spec_time += t
            return com, t
        return super().round(world, slots, rng, rtt, rnd)


def bkey(b):
    return 1 if b <= 1 else (4 if b <= 5 else 8)


class BatchAccount2(X.OPT2):
    def __init__(self, wk, theta_per_b=True, alpha=0.05, n0=10, warm=10, prior=3.0, buckets=(1, 2, 4, 8),
                 rho_warm=20, rho_th=RHO_TH):
        super().__init__(4, 0.4)
        self.wk = wk                      # callable (b, cls) -> (W, K)
        self.theta_per_b, self.alpha, self.n0, self.warm, self.prior, self.buckets = theta_per_b, alpha, n0, warm, prior, buckets
        self.rho_warm, self.rho_th = rho_warm, rho_th
        self.g_tok = self.g_ms = None
        self.tb = {}                      # b -> [tok EMA, ms EMA, rounds]
        self.cells = {}                   # (bucket, W) -> [EMA ms, n]
        self.req_a, self.req_e = {}, {}
        self.c_a = self.c_e = 0.0
        self.rho_num = self.rho_den = None
        self.rounds = 0
        self.wk_log = {}
        self.stats = dict(layers=0, exits=0, stops=0)

    # ----- acceptance class -----
    def cls(self):
        if self.rounds < self.rho_warm or not self.rho_den:
            return "mid"
        r = self.rho_num / self.rho_den
        return "low" if r < self.rho_th[0] else ("mid" if r < self.rho_th[1] else "high")

    # ----- price -----
    def _theta_b(self, b):
        if not self.theta_per_b:
            return self.g_tok / self.g_ms if self.g_ms else None
        if self.g_ms is None:
            return None
        own = self.tb.get(b)
        pts = [(bb, v[0] / v[1], v[2]) for bb, v in self.tb.items() if bb != b and v[2] >= 3 and v[1] > 0]
        if len(pts) >= 2 and len({p[0] for p in pts}) >= 2:
            x = np.log([p[0] for p in pts]); y = np.log([p[1] for p in pts]); w = np.sqrt([p[2] for p in pts])
            A = np.stack([np.ones_like(x), x], 1) * w[:, None]
            coef, *_ = np.linalg.lstsq(A, y * w, rcond=None)
            prior = float(math.exp(coef[0] + coef[1] * math.log(b)))
        elif pts:
            bb, th, _ = max(pts, key=lambda p: p[2])
            prior = th * (b / bb) ** 0.8
        else:
            prior = self.g_tok / self.g_ms
        if not own or own[1] <= 0:
            return prior
        n = own[2]
        wgt = n / (n + self.n0)
        return wgt * own[0] / own[1] + (1 - wgt) * prior

    # ----- layer cost -----
    def _cost(self, m, W):
        bm = F.bucket(m)
        c = self.cells.get((bm, W))
        if c and c[1] >= 3:
            return c[0]
        pts = [(k[0], k[1], v[0]) for k, v in self.cells.items()]
        if len(pts) >= 3:
            A = np.array([[1.0, p[0], p[0] * p[1]] for p in pts]); y = np.array([p[2] for p in pts])
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            est = float(coef[0] + coef[1] * bm + coef[2] * bm * W)
            if est > 0:
                return est if not c else 0.5 * (est + c[0])
        if c:
            return c[0]
        if pts:
            b0, w0, v0 = min(pts, key=lambda p: abs(p[0] * p[1] - bm * W))
            return v0 * (0.5 + 0.5 * (bm * W) / (b0 * w0))
        return 5.0

    def _c(self, r):
        cg = self.c_a / self.c_e if self.c_e > 0 else 1.0
        return (self.req_a.get(r, 0.0) + self.prior * cg) / (self.req_e.get(r, 0.0) + self.prior)

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        b = len(slots)
        W, K = self.wk(b, self.cls())
        self.wk_log[(W, K)] = self.wk_log.get((W, K), 0) + 1
        world.first_W = W
        trees = [X.Tree2(world.reqs[r], j, world) for r, j in slots]
        rids = [r for r, _ in slots]
        params = (world.a, world.g, world.beta, 0.0)
        for t in trees:
            t.expand(W, rng, params)
        rem = [world.reqs[r]["L"] - j for r, j in slots]
        e_prev = [0.0] * len(trees)
        e_now = [N._mass(t, K) for t in trees]
        active, depth, time_ = list(range(len(trees))), 1, 0.0
        theta = self._theta_b(b)
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
                    v = sum(g[i] for i in order[:m]) - theta * self._cost(m, W)
                    if v > best_v:
                        best_m, best_v = m, v
                keep = order[:best_m]
                self.stats["exits"] += len(cand) - len(keep)
            if not keep:
                self.stats["stops"] += 1
                break
            for i in keep:
                trees[i].expand(W, rng, params)
                e_prev[i] = e_now[i]
                e_now[i] = N._mass(trees[i], K)
            dt = world.real_cd(rng, len(keep), W)
            key = (F.bucket(len(keep)), W)
            c = self.cells.setdefault(key, [dt, 0])
            c[1] += 1
            c[0] = dt if c[1] == 1 else 0.9 * c[0] + 0.1 * dt
            time_ += dt
            self.stats["layers"] += 1
            active = keep
            depth += 1
        acc = [t.accepted(t.topk(K, False)) for t in trees]
        com = [min(a + 1, rem[i]) for i, a in enumerate(acc)]
        t_round = time_ + world.real_rtt(rng, rtt) + world.real_T(rng, b, K)
        tok = float(sum(com))
        a_ = self.alpha
        if self.g_ms is None:
            self.g_tok, self.g_ms = tok, t_round
        else:
            self.g_tok = (1 - a_) * self.g_tok + a_ * tok
            self.g_ms = (1 - a_) * self.g_ms + a_ * t_round
        v = self.tb.setdefault(b, [tok, t_round, 0])
        v[2] += 1
        ab = max(a_, 1.0 / v[2])
        v[0] = (1 - ab) * v[0] + ab * tok
        v[1] = (1 - ab) * v[1] + ab * t_round
        num = sum(acc); den = sum(a + (1 if a < N._tree_depth(t) else 0) for t, a in zip(trees, acc))
        if self.rho_den is None:
            self.rho_num, self.rho_den = float(num), float(den)
        else:
            self.rho_num = 0.95 * self.rho_num + 0.05 * num
            self.rho_den = 0.95 * self.rho_den + 0.05 * den
        for i, t in enumerate(trees):
            e = N._mass(t, K)
            self.req_a[rids[i]] = self.req_a.get(rids[i], 0.0) + acc[i]
            self.req_e[rids[i]] = self.req_e.get(rids[i], 0.0) + e
            self.c_a += acc[i]; self.c_e += e
        self.rounds += 1
        return com, t_round
