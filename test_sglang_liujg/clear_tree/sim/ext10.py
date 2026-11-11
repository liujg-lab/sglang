"""Round 12: fixes from review 14 on top of ext9.BatchAccount2, a conservative b=1 switch, and an exit audit.

BatchAccount3 options (all off = BatchAccount2):
  rho_mode   "old": rho = sum(a) / (sum(a) + #{a < tree depth}) (round 11);
             "opp": a stop counts as a failure only if the last accepted node had at least one child in the verify set
                    (an edge was actually tested); otherwise it is censored. Classification is binary high / other.
  c_decay    per-round decay of the calibration sums (per request and pooled); None = no forgetting.
  freeze     layer-cost observations of a round are applied only after the round (prices, costs and c_i fixed
             within a round).
  theta_mode "per_b" (round 11) or "global_reset": one global EMA price whose step becomes 1/n again after b changes.
  lookahead  (diagnostic oracle) predict the next layer from its own Draft-score increment, obtained by growing a
             cloned tree one layer with a separate RNG; removes the one-step lag (not available in a real system).
  cls_mode   "rho" (default, rho_mode above) or "top1": classify by the EMA of the first-position top-1 hit rate
             (Target token == Draft top-1 at the committed position), threshold th_top1.
  audit_p    probability of auditing a layer decision: active trees are cloned and grown one and two more layers
             with a separate RNG (the main simulation is unchanged); true increments are the realised acceptance
             changes. Records (one-step regret of the decision vs the greedy choice under true increments,
             two-layer optimum minus greedy-with-true-increments), both in tokens net of Theta * cost.
SwitchTok: D42 for b > 1; at b = 1 a conservative switch (50-round blocks, exit only after two blocks losing > 20%).
"""
import itertools
import numpy as np
import feasibility as F
import ext2 as X
import ext4 as Z
import ext5 as V
import ext8 as N
import ext9 as Q


def last_node(t, ver):
    """Accepted count a, and whether the stop was an observed failure (an edge out of the last node was tested)."""
    v, a, H = 0, 0, t.req["H"]
    while True:
        h = H[t.pos[v]]
        c = t.kids[v].get(int(h)) if h >= 0 else None
        if c is None or c not in ver:
            break
        a += 1; v = c
    tested = any(ch in ver for ch in t.kids[v].values())
    return a, tested


class SwitchTok:
    def __init__(self, inner, sw_kw, b1_kw=None):
        self.inner = inner
        self.sw = Z.SpecSwitch2(inner, **sw_kw)
        self.sw1 = Z.SpecSwitch2(inner, **dict(sw_kw, **(b1_kw or dict(win=50, exit_blocks=2, exit_fast=None, margin=0.2))))

    @property
    def spec_time(self):
        return self.sw.spec_time + self.sw1.spec_time

    @property
    def total_time(self):
        return self.sw.total_time + self.sw1.total_time

    def round(self, world, slots, rng, rtt, rnd):
        return (self.sw1 if len(slots) == 1 else self.sw).round(world, slots, rng, rtt, rnd)


class BatchAccount3(Q.BatchAccount2):
    def __init__(self, wk, rho_mode="old", th_high=Q.RHO_TH[1], c_decay=None, freeze=False, theta_mode="per_b",
                 audit_p=0.0, audit_seed=12345, lookahead=False, cls_mode="rho", th_top1=0.84, **kw):
        super().__init__(wk, theta_per_b=(theta_mode == "per_b"), **kw)
        self.rho_mode, self.th_high, self.c_decay, self.freeze, self.theta_mode = rho_mode, th_high, c_decay, freeze, theta_mode
        self.audit_p, self.arng = audit_p, np.random.default_rng(audit_seed)
        self.b_last, self.n_since = None, 0
        self.lookahead, self.cls_mode, self.th_top1 = lookahead, cls_mode, th_top1
        self.t1_hit = self.t1_n = None
        self.audit = dict(n=0, regret_pred=0.0, gap_struct=0.0, gap_pos=0, regret_pos=0, tokens=0.0)

    def cls(self):
        if self.cls_mode == "top1":
            if self.rounds < self.rho_warm or not self.t1_n:
                return "mid"
            return "high" if self.t1_hit / self.t1_n >= self.th_top1 else "mid"
        if self.rho_mode == "old":
            return super().cls()
        if self.rounds < self.rho_warm or not self.rho_den:
            return "mid"
        return "high" if self.rho_num / self.rho_den >= self.th_high else "mid"

    def _theta_b(self, b):
        if self.theta_mode == "global_reset":
            return self.g_tok / self.g_ms if self.g_ms else None
        return super()._theta_b(b)

    # ----- audit -----
    def _audit(self, world, trees, cand, keep, W, K, theta, depth, rem, params):
        acc0 = {i: trees[i].accepted(trees[i].topk(K, False)) for i in cand}
        g1, g2 = {}, {}
        for i in cand:
            c1 = V.clone_tree(trees[i]); c1.expand(W, self.arng, params)
            a1 = c1.accepted(c1.topk(K, False))
            g1[i] = a1 - acc0[i]
            if depth + 1 < world.DMAX - 1 and depth + 2 <= max(1, rem[i] - 1):
                c2 = V.clone_tree(c1); c2.expand(W, self.arng, params)
                g2[i] = c2.accepted(c2.topk(K, False)) - a1
            else:
                g2[i] = None
        cost = lambda m: theta * self._cost(m, W) if m else 0.0
        opts = lambda n: sorted({n} | {x for x in self.buckets if x < n}) if n else []

        def best_layer(members, gains):
            el = sorted([i for i in members if gains[i] is not None], key=lambda i: gains[i], reverse=True)
            best, bset = 0.0, []
            for m in opts(len(el)):
                v = sum(gains[i] for i in el[:m]) - cost(m)
                if v > best:
                    best, bset = v, el[:m]
            return best, bset

        g1v, s1 = best_layer(cand, g1)
        greedy2 = g1v + best_layer(s1, g2)[0]
        chosen = (sum(g1[i] for i in keep) - cost(len(keep))) if keep else 0.0
        opt2 = 0.0
        for r in range(1, len(cand) + 1):
            for sub in itertools.combinations(cand, r):
                v = sum(g1[i] for i in sub) - cost(len(sub)) + best_layer(sub, g2)[0]
                opt2 = max(opt2, v)
        a = self.audit
        a["n"] += 1
        a["regret_pred"] += g1v - chosen
        a["gap_struct"] += opt2 - greedy2
        a["regret_pos"] += int(g1v - chosen > 1e-9)
        a["gap_pos"] += int(opt2 - greedy2 > 1e-9)

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        b = len(slots)
        h1 = sum(int(world.reqs[r]["H"][j] == 0) for r, j in slots)
        if self.t1_n is None:
            self.t1_hit, self.t1_n = float(h1), float(b)
        else:
            self.t1_hit = 0.95 * self.t1_hit + 0.05 * h1
            self.t1_n = 0.95 * self.t1_n + 0.05 * b
        if self.theta_mode == "global_reset" and b != self.b_last:
            self.n_since = 0
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
        pending = []
        while active:
            cand = [i for i in active if depth < world.DMAX - 1 and depth + 1 <= max(1, rem[i] - 1)]
            if not cand:
                break
            if self.rounds < self.warm or theta is None:
                keep = [i for i in cand if e_now[i] - e_prev[i] > 0.4]
            else:
                if self.lookahead:
                    g = {}
                    for i in cand:
                        c1 = V.clone_tree(trees[i]); c1.expand(W, self.arng, params)
                        g[i] = self._c(rids[i]) * (N._mass(c1, K) - e_now[i])
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
                if self.audit_p and len(cand) >= 2 and self.arng.random() < self.audit_p:
                    self._audit(world, trees, cand, keep, W, K, theta, depth, rem, params)
            if not keep:
                self.stats["stops"] += 1
                break
            for i in keep:
                trees[i].expand(W, rng, params)
                e_prev[i] = e_now[i]
                e_now[i] = N._mass(trees[i], K)
            dt = world.real_cd(rng, len(keep), W)
            key = (F.bucket(len(keep)), W)
            if self.freeze:
                pending.append((key, dt))
            else:
                self._obs_cost(key, dt)
            time_ += dt
            self.stats["layers"] += 1
            active = keep
            depth += 1
        for key, dt in pending:
            self._obs_cost(key, dt)
        vers = [t.topk(K, False) for t in trees]
        acc, tested = [], []
        for t, ver in zip(trees, vers):
            a_, te = last_node(t, ver)
            acc.append(a_); tested.append(te)
        com = [min(a + 1, rem[i]) for i, a in enumerate(acc)]
        t_round = time_ + world.real_rtt(rng, rtt) + world.real_T(rng, b, K)
        tok = float(sum(com))
        self.audit["tokens"] += tok
        if self.theta_mode == "global_reset":
            ab = max(self.alpha, 1.0 / (self.n_since + 1))
            if self.g_ms is None:
                self.g_tok, self.g_ms = tok, t_round
            else:
                self.g_tok = (1 - ab) * self.g_tok + ab * tok
                self.g_ms = (1 - ab) * self.g_ms + ab * t_round
            self.n_since += 1
            self.b_last = b
        else:
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
        if self.rho_mode == "opp":
            num = sum(acc); den = sum(acc) + sum(1 for te in tested if te)
        else:
            num = sum(acc); den = sum(a + (1 if a < N._tree_depth(t) else 0) for t, a in zip(trees, acc))
        if self.rho_den is None:
            self.rho_num, self.rho_den = float(num), float(den)
        else:
            self.rho_num = 0.95 * self.rho_num + 0.05 * num
            self.rho_den = 0.95 * self.rho_den + 0.05 * den
        if self.c_decay is not None:
            self.c_a *= self.c_decay; self.c_e *= self.c_decay
        for i, t in enumerate(trees):
            e = N._mass(t, K)
            r = rids[i]
            if self.c_decay is not None:
                self.req_a[r] = self.req_a.get(r, 0.0) * self.c_decay
                self.req_e[r] = self.req_e.get(r, 0.0) * self.c_decay
            self.req_a[r] = self.req_a.get(r, 0.0) + acc[i]
            self.req_e[r] = self.req_e.get(r, 0.0) + e
            self.c_a += acc[i]; self.c_e += e
        self.rounds += 1
        return com, t_round

    def _obs_cost(self, key, dt):
        c = self.cells.setdefault(key, [dt, 0])
        c[1] += 1
        c[0] = dt if c[1] == 1 else 0.9 * c[0] + 0.1 * dt
