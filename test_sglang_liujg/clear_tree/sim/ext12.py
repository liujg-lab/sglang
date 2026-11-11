"""Round 14: choose (W, K) by paired replay of recent trees against the committed text (BatchAccount5).

Greedy decoding makes the committed text independent of the tree, so once the text is known (here: the world's H,
i.e. the tokens later rounds commit) every built tree can be re-scored for other options:
  narrowed tree   from the tree built with width W, keep layer by layer the global top-W' (W' <= W) nodes among the
                  children of the nodes kept one layer up (the "wide tree cut down" approximation of a W' tree)
  verify set      the top-K' nodes of the narrowed tree by Draft path score
  tokens          accepted along the true path within the verify set, + 1 bonus (capped by the remaining length)
  time            the round's actual time + [first layer + every Draft layer at W' minus at W] (learned layer cost
                  model) + [verify time at K' minus at K] (verify time as measured, here the world's T_t)
Per context (b bucket) and per option o, sums of (tokens_o, time_o) and of the current option's (tokens, time) are kept
over the rounds where o could be replayed, so every comparison is paired. Options narrower than the current width are
replayable on every round; wider ones only on exploration rounds (probability p_explore), which build the tree one
width step wider and pay its real cost. Every `every` rounds of a context, the option with the best paired gain
replaces the current one if the gain exceeds `hyst` and it has at least `min_n` paired rounds; statistics then reset.
Until replay makes its first switch in a context, the current option follows the table (which can change once the
top-1 classifier has warmed up); after that replay owns the choice.
layer_time: optional measured Draft layer time per (active count, width); without it the replay uses the learned
model, which must extrapolate to widths it has not run (round 14: this biased b=1 towards narrow layers).
Audit (probability audit_p): for a narrower width, a real tree of that width is grown to each request's built depth
with a separate RNG, and its accepted count is compared with the narrowed tree's (same K).
"""
import numpy as np
import feasibility as F
import ext2 as X
import ext8 as N
import ext10 as P
import ext11 as O
import ext9 as Q


def parents_of(t):
    par = {}
    for v, kd in enumerate(t.kids):
        for c in kd.values():
            par[c] = v
    return par


def narrowed(t, Wn):
    """Node ids kept when the built tree is cut to width Wn layer by layer (root 0 excluded)."""
    par = parents_of(t)
    by_depth = {}
    for nid in t.ids:
        by_depth.setdefault(t.dep[nid], []).append(nid)
    keep, prev = [], {0}
    for d in sorted(by_depth):
        cand = [n for n in by_depth[d] if par.get(n) in prev]
        if not cand:
            break
        cand.sort(key=lambda n: t.score[n], reverse=True)
        sel = cand[:Wn]
        keep.extend(sel)
        prev = set(sel)
    return keep


def accepted_in(t, ver):
    v, a, H = 0, 0, t.req["H"]
    while True:
        h = H[t.pos[v]]
        c = t.kids[v].get(int(h)) if h >= 0 else None
        if c is None or c not in ver:
            break
        a += 1; v = c
    return a


def topk_of(t, nodes, K):
    return set(sorted(nodes, key=lambda n: t.score[n], reverse=True)[:K])


class BatchAccount5(O.BatchAccount4):
    def __init__(self, wk, w_options=(2, 4, 8), k_options=(8, 16, 32), p_explore=0.05, every=20, hyst=0.03,
                 min_n=30, min_n_explore=10, audit_p=0.1, verify_time=None, layer_time=None, **kw):
        super().__init__(wk, verify_time=verify_time, **kw)
        self.w_opts, self.k_opts = tuple(sorted(w_options)), tuple(sorted(k_options))
        self.p_explore, self.every, self.hyst2, self.min_n, self.min_n_explore = p_explore, every, hyst, min_n, min_n_explore
        self.audit_p2 = audit_p
        self.layer_time = layer_time                    # callable (m, W) -> ms, e.g. a measured table; None = learned model
        self.ctx = {}                                   # bkey -> state
        self.prng = np.random.default_rng(777)
        self.choice_log, self.explore_rounds, self.switches = {}, 0, 0
        self.naudit = dict(n=0, diff=0.0, absdiff=0.0, differ=0, real=0.0)

    def _ctx(self, b, cls):
        k = Q.bkey(b)
        if k not in self.ctx:
            W, K = self.wk(b, cls)
            self.ctx[k] = dict(cur=(W, K), stats={}, rounds=0, follow=True)
        s = self.ctx[k]
        if s["follow"]:
            t = self.wk(b, cls)
            if t != s["cur"]:
                s["cur"], s["stats"] = t, {}
        return s

    def _decide(self, s):
        best, bg = None, self.hyst2
        for o, v in s["stats"].items():
            if o == s["cur"]:
                continue
            need = self.min_n if o[0] <= s["cur"][0] else self.min_n_explore
            if v[4] < need or v[1] <= 0 or v[3] <= 0:
                continue
            g = (v[0] / v[1]) / (v[2] / v[3]) - 1
            if g > bg:
                best, bg = o, g
        if best is not None:
            s["cur"], s["follow"] = best, False
            s["stats"] = {}
            self.switches += 1

    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        b = len(slots)
        # first-position statistics (shared with the table classifier)
        h1 = sum(int(world.reqs[r]["H"][j] == 0) for r, j in slots)
        if self.t1_n is None:
            self.t1_hit, self.t1_n = float(h1), float(b)
        else:
            self.t1_hit = 0.95 * self.t1_hit + 0.05 * h1
            self.t1_n = 0.95 * self.t1_n + 0.05 * b
        s = self._ctx(b, self.cls())
        Wc, Kc = s["cur"]
        W = Wc
        explore = False
        wider = [w for w in self.w_opts if w > Wc]
        if wider and self.rounds >= self.warm and self.prng.random() < self.p_explore:
            W, explore = wider[0], True
            self.explore_rounds += 1
        K = Kc
        self.choice_log[(Wc, Kc)] = self.choice_log.get((Wc, Kc), 0) + 1
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
        pending, layer_m = [], []
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
            pending.append(((F.bucket(len(keep)), W), dt))
            layer_m.append(len(keep))
            time_ += dt
            active = keep
            depth += 1
        acc = [accepted_in(t, t.topk(K, False)) for t in trees]
        com = [min(a + 1, rem[i]) for i, a in enumerate(acc)]
        t_round = time_ + world.real_rtt(rng, rtt) + world.real_T(rng, b, K)
        # ---- paired replay (costs and prices still frozen at this round's values) ----
        if self.rounds >= self.warm and self.verify_time is not None:
            tok_cur = float(sum(com))
            narrowed_sets = {Wn: [narrowed(t, Wn) if Wn < W else list(t.ids) for t in trees] for Wn in self.w_opts if Wn <= W}
            lc = self.layer_time or self._cost
            for Wn, sets in narrowed_sets.items():
                dlayer = (lc(b, Wn) - lc(b, W)) + sum(lc(m, Wn) - lc(m, W) for m in layer_m)
                for Kn in self.k_opts:
                    o = (Wn, Kn)
                    tok = 0.0
                    for i, t in enumerate(trees):
                        a = accepted_in(t, topk_of(t, sets[i], Kn))
                        tok += min(a + 1, rem[i])
                    tm = t_round + dlayer + (self.verify_time(b, Kn) - self.verify_time(b, K))
                    if explore:
                        # compare against the current option replayed on the same (wider) round
                        cur_sets = narrowed_sets.get(Wc)
                        ctok = sum(min(accepted_in(t, topk_of(t, cur_sets[i], Kc)) + 1, rem[i]) for i, t in enumerate(trees))
                        cdl = (lc(b, Wc) - lc(b, W)) + sum(lc(m, Wc) - lc(m, W) for m in layer_m)
                        ctm = t_round + cdl + (self.verify_time(b, Kc) - self.verify_time(b, K))
                    else:
                        ctok, ctm = tok_cur, t_round
                    v = s["stats"].setdefault(o, [0.0, 0.0, 0.0, 0.0, 0])
                    v[0] += tok; v[1] += tm; v[2] += ctok; v[3] += ctm; v[4] += 1
            s["rounds"] += 1
            if s["rounds"] % self.every == 0:
                self._decide(s)
            # ---- narrowing audit ----
            if W > self.w_opts[0] and self.prng.random() < self.audit_p2:
                Wn = max(w for w in self.w_opts if w < W)
                for t in trees:
                    real = X.Tree2(t.req, t.j, world)
                    real.expand(Wn, self.arng, params)
                    for _ in range(max(t.dep) - 1):
                        real.expand(Wn, self.arng, params)
                    a_real = accepted_in(real, real.topk(K, False))
                    a_nar = accepted_in(t, topk_of(t, narrowed(t, Wn), K))
                    na = self.naudit
                    na["n"] += 1; na["diff"] += a_real - a_nar; na["absdiff"] += abs(a_real - a_nar)
                    na["differ"] += int(a_real != a_nar); na["real"] += a_real
        for key, dt in pending:
            self._obs_cost(key, dt)
        # ---- learning (same as BatchAccount4) ----
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
