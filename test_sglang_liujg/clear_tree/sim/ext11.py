"""Round 13: online choice of the layer width W and the verify count K (BatchAccount4), on top of the revised v10
rule (ext10.BatchAccount3 with the top-1 classifier, frozen costs, Theta per b).

online_K  The tree is grown with E over the top K_max nodes. Before verifying, K is chosen from k_options: going
          from K to the next larger option is worth it if sum_i c_i * (Draft path scores of the nodes ranked K+1..K')
          > Theta_b * (T_verify(b, K') - T_verify(b, K)). The verify time per (b, K) is taken as measured (the
          world's T_t, as V2 would give); it is not learned online.
online_W  Before the round, W is chosen from w_options by a chain model:
            p_r     = EMA share of rounds whose Target token at the committed position is the Draft's rank-r candidate
                      (fixed before any tree is built, so it does not depend on W, K or tau)
            q(W)    = p_1 + ... + p_W, the chance that the path continues through one layer of width W
            rate(W) = max_D b * (1 + q (1 - q^D) / (1 - q)) / (F_b + (D - 1) * C(b, W))
          F_b is the EMA of the round time minus the Draft layers after the first (RTT, Target, host and the
          first layer, which rides on the Target wait in this world) and C(b, W) the learned
          layer cost (one linear model over (bucket, W) cells). It ignores exits and the global top-W competition
          between branches. The width changes only if the predicted rate improves by more than `hyst`.
          The first 20 rounds use the table.
          kcap=True also limits D by the verify count: with W nodes per layer, the top-K verify set covers about
          K / W layers, so D <= min(9, ceil(K / W) + 1). Without it the model favours wide layers whose deep path
          nodes do not make it into the verify set.
"""
import numpy as np
import feasibility as F
import ext2 as X
import ext8 as N
import ext10 as P

C = F.C


class BatchAccount4(P.BatchAccount3):
    def __init__(self, wk, online_K=False, k_options=(8, 16, 32), online_W=False, w_options=(2, 4, 8), hyst=0.02,
                 verify_time=None, kcap=True, **kw):
        kw.setdefault("cls_mode", "top1")
        kw.setdefault("freeze", True)
        super().__init__(wk, **kw)
        self.online_K, self.k_options, self.online_W, self.w_options, self.hyst = online_K, tuple(sorted(k_options)), online_W, w_options, hyst
        self.verify_time = verify_time            # callable (b, K) -> ms
        self.kcap = kcap
        self.rank = np.zeros(C + 1)               # EMA counts: ranks 0..C-1, index C = not in top-C
        self.rank_n = 0.0
        self.fixed = {}                           # b -> EMA of round time minus Draft layer time
        self.W_cur = {}
        self.k_log = {}

    def _q(self, W):
        if self.rank_n <= 0:
            return None
        return float(self.rank[:W].sum() / self.rank_n)

    def _choose_W(self, b, W_table, K_for_cap=32):
        if not self.online_W or self.rounds < self.rho_warm or b not in self.fixed:
            return W_table
        Fb = self.fixed[b]
        best = {}
        for W in self.w_options:
            q = self._q(W)
            if q is None:
                return W_table
            q = min(q, 0.999)
            c = self._cost(b, W)
            r = 0.0
            dmax = min(9, -(-K_for_cap // W) + 1) if self.kcap else 9
            for D in range(1, dmax + 1):
                r = max(r, b * (1 + q * (1 - q ** D) / (1 - q)) / (Fb + (D - 1) * c))
            best[W] = r
        cur = self.W_cur.get(b, W_table)
        cand = max(best, key=best.get)
        if cur in best and best[cand] < (1 + self.hyst) * best[cur]:
            cand = cur
        self.W_cur[b] = cand
        return cand

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        b = len(slots)
        # first-position statistics (known before the tree is built)
        h1 = 0
        for r, j in slots:
            h = int(world.reqs[r]["H"][j])
            h1 += int(h == 0)
            self.rank *= 0.995 ** (1.0 / max(1, b))
            self.rank_n *= 0.995 ** (1.0 / max(1, b))
            self.rank[h if h >= 0 else C] += 1.0
            self.rank_n += 1.0
        if self.t1_n is None:
            self.t1_hit, self.t1_n = float(h1), float(b)
        else:
            self.t1_hit = 0.95 * self.t1_hit + 0.05 * h1
            self.t1_n = 0.95 * self.t1_n + 0.05 * b
        W_t, K_t = self.wk(b, self.cls())
        K_grow = max(self.k_options) if self.online_K else K_t
        W = self._choose_W(b, W_t, K_grow)
        self.wk_log[(W, K_grow)] = self.wk_log.get((W, K_grow), 0) + 1
        world.first_W = W
        trees = [X.Tree2(world.reqs[r], j, world) for r, j in slots]
        rids = [r for r, _ in slots]
        params = (world.a, world.g, world.beta, 0.0)
        for t in trees:
            t.expand(W, rng, params)
        rem = [world.reqs[r]["L"] - j for r, j in slots]
        K = K_grow
        e_prev = [0.0] * len(trees)
        e_now = [N._mass(t, K) for t in trees]
        active, depth, time_ = list(range(len(trees))), 1, 0.0
        theta = self._theta_b(b)
        pending = []
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
            if not keep:
                break
            for i in keep:
                trees[i].expand(W, rng, params)
                e_prev[i] = e_now[i]
                e_now[i] = N._mass(trees[i], K)
            dt = world.real_cd(rng, len(keep), W)
            pending.append((((F.bucket(len(keep))), W), dt))
            time_ += dt
            active = keep
            depth += 1
        for key, dt in pending:
            self._obs_cost(key, dt)
        # verify count
        if self.online_K and theta is not None and self.rounds >= self.warm:
            K = self.k_options[0]
            sc = [np.sort(np.asarray(t.scores))[::-1] for t in trees]
            for K2 in self.k_options[1:]:
                extra = sum(self._c(rids[i]) * float(s[K:K2].sum()) for i, s in enumerate(sc))
                if extra > theta * (self.verify_time(b, K2) - self.verify_time(b, K)):
                    K = K2
                else:
                    break
        elif self.online_K:
            K = K_t
        self.k_log[K] = self.k_log.get(K, 0) + 1
        acc = [P.last_node(t, t.topk(K, False))[0] for t in trees]
        com = [min(a + 1, rem[i]) for i, a in enumerate(acc)]
        t_round = time_ + world.real_rtt(rng, rtt) + world.real_T(rng, b, K)
        tok = float(sum(com))
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
        ab = max(a_, 1.0 / v[2])
        v[0] = (1 - ab) * v[0] + ab * tok
        v[1] = (1 - ab) * v[1] + ab * t_round
        for i, t in enumerate(trees):
            e = N._mass(t, K)
            r = rids[i]
            self.req_a[r] = self.req_a.get(r, 0.0) + acc[i]
            self.req_e[r] = self.req_e.get(r, 0.0) + e
            self.c_a += acc[i]; self.c_e += e
        self.rounds += 1
        return com, t_round
