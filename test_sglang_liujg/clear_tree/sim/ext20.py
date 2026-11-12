"""Round 23: W thresholds derived from the cost model (BatchAccount13), vs per-cell paired calibration, the replay
controller, a flat table and fixed W, in prefix-aware worlds (ext19.World22) with known costs.

Common to all policies: v10 growth (net-gain order, consistent growth/verify value, K online), known prefix-aware costs,
flat starting table (W = 8 at b = 1, else 4). Replay (for every W' <= the built W): the tree is cut to width W'
(ext12.narrowed); the cut tree may stop earlier than the built one (depth cut chosen by its own predicted net value
J = sum c_i * top-K mass - Theta * model time), and on "deep" rounds (two extra layers grown after the stop) also deeper;
K' is chosen by the same J; tokens come from the committed text; time is the model time of that schedule.

wmode
  flat       the flat table
  fixed      one W (reference)
  cellcal    round-19b paired calibration per cell (b bucket, class): window 50 rounds after the cell appears, all
             other W as candidates, build at the widest still in contention, per-round paired differences of
             Y = tokens - Theta * model time, plug-in normal-mixture CS, 30 pairs, eps = 1% of tokens per round,
             200-round budget, uncertain -> keep
  replay14   round-14 replay controller per b bucket: narrower W replayed every round, one step wider on 5% of the
             rounds, every 20 rounds switch if the paired gain > 3% (30 / 10 rounds), follows the table until then
  derived    the value of one width step (2->4, 4->8) per request, dA(h) = tokens(W_hi) - tokens(W_lo) of request i in
             the same round, pooled over b, binned by the request's own first-position hit EMA h_i (bins of 0.05 on
             [0.4, 1], below 0.4 one bin), made non-increasing in h (pool adjacent violators, weighted by counts);
             layers per round L(W) per b bucket from the replays. Each round, for each step:
                G = sum_i dA(h_i) (+- 2 se),  C = Theta * [L_hi C(tier, W_hi, sum l) - L_lo C(tier, W_lo, sum l)
                                                            + first-layer difference]
             widen one step if G - 2 se > C + margin, narrow if G + 2 se < C - margin (margin = 1% of tokens/round),
             at most one step per round, only with >= n_min samples in every bin the batch uses; L_lo and L_hi come
             from the same rounds (both replayable); a switch needs the same proposal 10 rounds in a row and is
             followed by 150 rounds without switching (a first version flip-flopped between 4 and 2)
             Data: narrower steps come free from replay; a window (80 rounds, 50 rounds after a b bucket first appears)
             builds one step wider and makes 1 in 4 rounds deep; while a step adjacent to the current W stays
             undecided, the window re-opens after 1000, 2000, 4000, ... rounds of that bucket.
  derived2   derived with two changes (added after smoke tests): the cost side is the EMA of the paired model-time
             difference of the two widths on the same replayed rounds (no full-tier approximation); and a window
             does not build the wider width when even the value of the step below (an upper bound, by diminishing
             returns) is under the model cost of that widening (60-round windows, deep rounds still happen)
"""
import math
import numpy as np
import ext2 as X
import ext8 as N
import ext9 as Q
import ext10 as P
import ext12 as RP
import ext18 as H
import ext19 as Z
import feasibility as F

W_OPTS = (2, 4, 8)
PAIRS = ((2, 4), (4, 8))


def flat_wk(b, cls):
    return (8, 32) if b <= 1 else ((4, 32) if b <= 4 else (4, 16))


def make_world(kind, mix, b, seed, phase=None, req_mult=3, n_req=None):
    w = Z.make_world(mix, kind, b, seed, "strong", req_mult=req_mult)
    if n_req is not None:
        pass
    if phase == "lowhighmid":
        rs = sorted(w.reqs, key=lambda r: r["s"])
        k = len(rs) // 3
        low, mid, high = rs[:k], rs[k:2 * k], rs[2 * k:]
        w.reqs = low + high + mid
    return w


def _bin(h):
    return 0 if h < 0.4 else min(12, 1 + int((h - 0.4) / 0.05))


class BatchAccount13(H.BatchAccount11):
    def __init__(self, wk=flat_wk, wmode="flat", fixed_W=None, n_min=30, win=80, deep_frac=0.25, open_delay=50,
                 seed=1234, **kw):
        kw.setdefault("cost_mode", "known")
        super().__init__(wk, **kw)
        self.wmode, self.fixed_W, self.n_min, self.win, self.deep_frac, self.open_delay = wmode, fixed_W, n_min, win, deep_frac, open_delay
        self.rng2 = np.random.default_rng(seed)
        self.h_req = {}
        self.cur = {}                      # bkey -> current W (derived, replay14 share this)
        self.w_log, self.n_switch, self.n_explore, self.n_deep = {}, 0, 0, 0
        self.switch_log = []
        # derived
        self.st = {p: np.zeros((13, 3)) for p in PAIRS}      # per bin: n, sum, sumsq
        self.Lw = {}                                         # (bkey, W) -> [EMA layers, n] (diagnostic)
        self.Lp = {}                                         # (bkey, pair) -> [EMA L_lo, EMA L_hi, n], same rounds
        self.dwell, self.persist = 150, 10
        self.dTp = {}                                        # derived2: (bkey, pair) -> [EMA dT, n] (same rounds)
        self.v2 = wmode == "derived2"
        if self.v2:
            self.win = 60
        self.bk_state = {}                                   # bkey -> dict(n, win_left, next_open, backoff)
        self.tok_ema = None
        # cellcal
        self.cells = {}
        # replay14
        self.ctx14 = {}

    # ---------------- helpers ----------------
    def _model_time(self, b, W, K, layers, suml_all):
        w = self.world
        return (sum(w.c_d_l(m, W, sl) for m, sl in layers) + w.c_d_l(b, W, suml_all)
                + w.T_core(b, K, suml_all) + 5.0)

    def _replay(self, trees, cs, rem, ell, layers, W, b, theta, deep_extra):
        """For every W' <= W: per-request tokens, model time, K', depth. layers: [(m, suml)] of the built layers."""
        w = self.world
        suml_all = float(sum(ell))
        Dbuilt = 1 + len(layers)
        out = {}
        for Wn in W_OPTS:
            if Wn > W:
                continue
            nodes = [RP.narrowed(t, Wn) if Wn < W else list(t.ids) for t in trees]
            if Wn == W and not deep_extra:
                depths = [Dbuilt]
            else:
                lo = max(1, Dbuilt - 2 - deep_extra)
                depths = list(range(lo, Dbuilt + 1))
            best = None
            for d in depths:
                sel = [[n for n in nodes[i] if t.dep[n] <= d] for i, t in enumerate(trees)]
                lt = sum(w.c_d_l(m, Wn, sl) for m, sl in layers[:d - 1]) + w.c_d_l(b, Wn, suml_all)
                for K in self.k_options:
                    v = sum(cs[i] * sum(sorted((t.score[n] for n in sel[i]), reverse=True)[:K]) for i, t in enumerate(trees))
                    J = v - theta * (lt + w.T_core(b, K, suml_all))
                    if best is None or J > best[0]:
                        best = (J, d, K, sel, lt)
            _, d, K, sel, lt = best
            tok = [min(RP.accepted_in(t, RP.topk_of(t, sel[i], K)) + 1, rem[i]) for i, t in enumerate(trees)]
            out[Wn] = dict(tok=tok, time=lt + w.T_core(b, K, suml_all) + 5.0, K=K, layers=d - 1)
        return out

    # ---------------- derived: decision ----------------
    def _curve(self, pair):
        s = self.st[pair]
        n, mu = s[:, 0], np.where(s[:, 0] > 0, s[:, 1] / np.maximum(s[:, 0], 1), 0.0)
        var = np.where(s[:, 0] > 1, (s[:, 2] - s[:, 1] ** 2 / np.maximum(s[:, 0], 1)) / np.maximum(s[:, 0] - 1, 1), 0.0)
        # pool adjacent violators for a non-increasing curve over the bins with data
        idx = [i for i in range(13) if n[i] > 0]
        blocks = [[mu[i], n[i], [i]] for i in idx]
        k = 0
        while k < len(blocks) - 1:
            if blocks[k][0] < blocks[k + 1][0]:
                a, c = blocks[k], blocks[k + 1]
                tot = a[1] + c[1]
                blocks[k] = [(a[0] * a[1] + c[0] * c[1]) / tot, tot, a[2] + c[2]]
                del blocks[k + 1]
                k = max(0, k - 1)
            else:
                k += 1
        fit = mu.copy()
        for m_, w_, ids in blocks:
            for i in ids:
                fit[i] = m_
        return n, fit, var

    def _decide(self, b, slots, ell, theta):
        bk = Q.bkey(b)
        cur = self.cur.get(bk, flat_wk(b, "mid")[0])
        w = self.world
        suml = float(sum(ell))
        tier = F.bucket(b)
        tokr = self.tok_ema or 1.0
        margin = 0.01 * tokr
        pref, undecided = {}, False
        for (lo, hi) in PAIRS:
            n, fit, var = self._curve((lo, hi))
            bins = [_bin(self.h_req.get(r, self._hglob())) for r, _ in slots]
            if any(n[i] < self.n_min for i in bins):
                pref[(lo, hi)] = None
                if lo == cur or hi == cur:
                    undecided = True
                continue
            G = float(sum(fit[i] for i in bins))
            se = math.sqrt(sum(var[i] / max(n[i], 1) for i in bins))
            lp = self.Lp.get((bk, (lo, hi)))
            Llo, Lhi = (lp[0], lp[1]) if lp and lp[2] >= 5 else (None, None)
            if Llo is None or Lhi is None:
                pref[(lo, hi)] = None
                undecided = undecided or lo == cur or hi == cur
                continue
            if self.v2:
                e = self.dTp.get((bk, (lo, hi)))
                if not e or e[1] < 5:
                    pref[(lo, hi)] = None
                    undecided = undecided or lo == cur or hi == cur
                    continue
                C = theta * e[0]
            else:
                C = theta * (Lhi * w.c_d_l(b, hi, suml) - Llo * w.c_d_l(b, lo, suml) + w.c_d_l(b, hi, suml) - w.c_d_l(b, lo, suml))
            if G - 2 * se > C + margin:
                pref[(lo, hi)] = "hi"
            elif G + 2 * se < C - margin:
                pref[(lo, hi)] = "lo"
            else:
                pref[(lo, hi)] = None
                undecided = undecided or lo == cur or hi == cur
        new = cur
        if cur == 2 and pref.get((2, 4)) == "hi":
            new = 4
        elif cur == 4:
            if pref.get((4, 8)) == "hi":
                new = 8
            elif pref.get((2, 4)) == "lo":
                new = 2
        elif cur == 8 and pref.get((4, 8)) == "lo":
            new = 4
        st = self.bk_state[bk]
        if new != cur and st["n"] >= st.get("dwell_until", 0):
            st["streak"] = st.get("streak", 0) + 1 if st.get("cand") == new else 1
            st["cand"] = new
            if st["streak"] >= self.persist:
                self.cur[bk] = new
                st["dwell_until"] = st["n"] + self.dwell
                st["streak"], st["cand"] = 0, None
                self.n_switch += 1
                if len(self.switch_log) < 30:
                    self.switch_log.append((self.rounds, b, cur, new))
        else:
            st["streak"], st["cand"] = 0, None
        return undecided

    def _prune(self, b, cur, hi, slots, ell, theta):
        """True if exploring cur -> hi cannot pay: the value of the step below (lo -> cur), an upper bound by
        diminishing returns, is already below the model cost of widening cur -> hi at the current layer count."""
        lo = max([x for x in W_OPTS if x < cur], default=None)
        if lo is None:
            return False
        n, fit, var = self._curve((lo, cur))
        bins = [_bin(self.h_req.get(r, self._hglob())) for r, _ in slots]
        if any(n[i] < self.n_min for i in bins):
            return False
        G_up = float(sum(fit[i] for i in bins)) + 2 * math.sqrt(sum(var[i] / max(n[i], 1) for i in bins))
        w = self.world
        suml = float(sum(ell))
        L = self.Lw.get((Q.bkey(b), cur), [4.0])[0]
        C = theta * ((L + 1) * (w.c_d_l(b, hi, suml) - w.c_d_l(b, cur, suml)))
        return G_up < C

    def _hglob(self):
        return (self.t1_hit / self.t1_n) if self.t1_n else 0.7

    # ---------------- W choice ----------------
    def _choose(self, b, cls, warm, slots, ell, theta):
        Wflat = flat_wk(b, cls)[0]
        if warm or self.wmode == "flat":
            return Wflat, "normal"
        if self.wmode == "fixed":
            return self.fixed_W, "normal"
        bk = Q.bkey(b)
        if self.wmode == "replay14":
            s = self.ctx14.setdefault(bk, dict(cur=Wflat, follow=True, stats={}, n=0))
            if s["follow"]:
                s["cur"] = Wflat
            wider = [x for x in W_OPTS if x > s["cur"]]
            if wider and self.rng2.random() < 0.05:
                self.n_explore += 1
                return wider[0], "explore"
            return s["cur"], "normal"
        if self.wmode == "cellcal":
            key = (bk, cls)
            c = self.cells.get(key)
            if c is None:
                c = self.cells[key] = dict(cur=Wflat, wait=self.open_delay, cal=None)
            if c["wait"] is not None:
                c["wait"] -= 1
                if c["wait"] <= 0:
                    c["wait"] = None
                    c["cal"] = dict(n=0, active=set(W_OPTS) - {c["cur"]}, data={x: [] for x in W_OPTS}, A=[], theta=None)
            if c["cal"] is None:
                return c["cur"], "normal"
            return max(c["cal"]["active"] | {c["cur"]}), "cal"
        # derived / derived2
        st = self.bk_state.setdefault(bk, dict(n=0, win_left=0, next_open=self.open_delay, backoff=1000))
        self.cur.setdefault(bk, Wflat)
        st["n"] += 1
        if st["win_left"] <= 0 and st["n"] >= st["next_open"]:
            st["win_left"] = self.win
            st["next_open"] = st["n"] + 10 ** 9
        if st["win_left"] > 0:
            st["win_left"] -= 1
            cur = self.cur[bk]
            wider = [x for x in W_OPTS if x > cur]
            if self.v2 and wider and self._prune(b, cur, wider[0], slots, ell, theta):
                wider = []
                self.n_pruned = getattr(self, "n_pruned", 0) + 1
            deep = self.rng2.random() < self.deep_frac
            if deep:
                self.n_deep += 1
            self.n_explore += 1
            return (wider[0] if wider else cur), ("deep" if deep else "explore")
        undecided = self._decide(b, slots, ell, theta)
        if undecided and st["next_open"] > st["n"] + st["backoff"]:
            st["next_open"] = st["n"] + st["backoff"]
            st["backoff"] *= 2
        return self.cur[bk], "normal"

    # ---------------- post-round learning per mode ----------------
    def _post(self, b, cls, W, tag, rep, rids, slots, com, t_round, theta, layers_built):
        bk = Q.bkey(b)
        if self.wmode in ("derived", "derived2"):
            for (lo, hi) in PAIRS:
                if lo in rep and hi in rep:
                    for i, r in enumerate(rids):
                        d = rep[hi]["tok"][i] - rep[lo]["tok"][i]
                        k = _bin(self.h_req.get(r, self._hglob()))
                        s = self.st[(lo, hi)][k]
                        s[0] += 1; s[1] += d; s[2] += d * d
            for (lo, hi) in PAIRS:
                if lo in rep and hi in rep:
                    e2 = self.dTp.setdefault((bk, (lo, hi)), [rep[hi]["time"] - rep[lo]["time"], 0])
                    e2[1] += 1
                    a2 = 1.0 / e2[1] if e2[1] <= 10 else 0.05
                    e2[0] = (1 - a2) * e2[0] + a2 * (rep[hi]["time"] - rep[lo]["time"])
                    e = self.Lp.setdefault((bk, (lo, hi)), [float(rep[lo]["layers"]), float(rep[hi]["layers"]), 0])
                    e[2] += 1
                    a = 1.0 / e[2] if e[2] <= 10 else 0.05
                    e[0] = (1 - a) * e[0] + a * rep[lo]["layers"]
                    e[1] = (1 - a) * e[1] + a * rep[hi]["layers"]
            for Wx, o in rep.items():
                e = self.Lw.setdefault((bk, Wx), [float(o["layers"]), 0])
                e[1] += 1
                a = 1.0 / e[1] if e[1] <= 10 else 0.05
                e[0] = (1 - a) * e[0] + a * o["layers"]
            return
        Y = {Wx: sum(o["tok"]) - theta * o["time"] for Wx, o in rep.items()}
        if self.wmode == "cellcal":
            c = self.cells.get((bk, cls))
            if c is None or c["cal"] is None or tag != "cal":
                return
            cal = c["cal"]
            cal["n"] += 1
            cal["A"].append(float(sum(com)))
            eps = 0.01 * float(np.mean(cal["A"]))
            cur = c["cur"]
            if cur not in Y:
                return
            for a in Y:
                if a != cur:
                    cal["data"][a].append(Y[a] - Y[cur])
            winner, wmu = None, None
            for a in sorted(cal["active"]):
                d = cal["data"][a]
                if len(d) < 30:
                    continue
                n = len(d)
                mu, sd = float(np.mean(d)), float(np.std(d, ddof=1))
                v, rho = n * sd * sd, max(1e-12, 50 * sd * sd)
                rad = math.sqrt((v + rho) * (math.log((v + rho) / rho) + 2 * math.log(2 / 0.025))) / n
                if mu - rad > eps and (wmu is None or mu > wmu):
                    winner, wmu = a, mu
                elif mu + rad < eps:
                    cal["active"].discard(a)
            if winner is not None or not cal["active"] or cal["n"] >= 200:
                if winner is not None:
                    c["cur"] = winner
                    self.n_switch += 1
                    if len(self.switch_log) < 30:
                        self.switch_log.append((self.rounds, b, cur, winner))
                c["cal"] = None
            return
        if self.wmode == "replay14":
            s = self.ctx14[bk]
            cur = s["cur"]
            if cur not in rep:
                return
            if tag == "explore":
                ctok, ctm = sum(rep[cur]["tok"]), rep[cur]["time"]
            else:
                ctok, ctm = float(sum(com)), t_round
            for o, v in rep.items():
                if o == cur:
                    continue
                st = s["stats"].setdefault(o, [0.0, 0.0, 0.0, 0.0, 0])
                st[0] += sum(v["tok"]); st[1] += v["time"]; st[2] += ctok; st[3] += ctm; st[4] += 1
            s["n"] += 1
            if s["n"] % 20 == 0:
                best, bg = None, 0.03
                for o, st in s["stats"].items():
                    need = 30 if o < cur else 10
                    if st[4] < need:
                        continue
                    g = (st[0] / st[1]) / (st[2] / st[3]) - 1
                    if g > bg:
                        best, bg = o, g
                if best is not None:
                    s["cur"], s["follow"], s["stats"] = best, False, {}
                    self.n_switch += 1
                    if len(self.switch_log) < 30:
                        self.switch_log.append((self.rounds, b, cur, best))

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
        _, Ktab = flat_wk(b, cls)
        theta = self._theta_b(b)
        warm = self.rounds < self.warm or theta is None
        rids = [r for r, _ in slots]
        ell = [world.reqs[r]["p0"] + j for r, j in slots]
        suml_all = float(sum(ell))
        W, tag = self._choose(b, cls, warm, slots, ell, theta)
        self.w_log[W] = self.w_log.get(W, 0) + 1
        Ks = self.k_options
        Kmax = max(Ks)
        world.first_W = W
        trees = [X.Tree2(world.reqs[r], j, world) for r, j in slots]
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
        kd = world.kd(W)
        active, depth, time_ = list(range(n)), 1, 0.0
        layers = []
        while active:
            cand = [i for i in active if depth < world.DMAX - 1 and depth + 1 <= max(1, rem[i] - 1)]
            if not cand:
                break
            if warm:
                keep = [i for i in cand if e_now[i][Kmax] - e_prev[i][Kmax] > 0.4]
            else:
                Jk = {K: sum(v_now[i][K] for i in range(n)) - theta * world.T_core(b, K, suml_all) for K in Ks}
                stop = max(Jk.values())
                best, keep = 0.0, []
                for K in Ks:
                    g = {i: v_now[i][K] - v_prev[i][K] for i in cand}
                    sc = {i: g[i] - theta * kd * ell[i] for i in cand}
                    order = sorted(cand, key=lambda i: sc[i], reverse=True)
                    for m in self._mopts(len(order)):
                        sel = order[:m]
                        v = Jk[K] + sum(g[i] for i in sel) - theta * world.c_d_l(m, W, sum(ell[i] for i in sel)) - stop
                        if v > best + 1e-12:
                            best, keep = v, sel
            if not keep:
                break
            for i in keep:
                trees[i].expand(W, rng, params)
                e_prev[i], e_now[i] = e_now[i], self._es(trees[i])
                v_prev[i], v_now[i] = v_now[i], self._vals(cs[i], e_now[i])
            sl = float(sum(ell[i] for i in keep))
            time_ += world.real_cd_l(rng, len(keep), W, sl)
            layers.append((len(keep), sl))
            active = keep
            depth += 1
        deep_extra = 0
        if tag == "deep":
            cand = [i for i in range(n) if depth < world.DMAX - 1 and depth + 1 <= max(1, rem[i] - 1)]
            for _ in range(2):
                cand = [i for i in cand if depth < world.DMAX - 1 and depth + 1 <= max(1, rem[i] - 1)]
                if not cand:
                    break
                for i in cand:
                    trees[i].expand(W, rng, params)
                    e_prev[i], e_now[i] = e_now[i], self._es(trees[i])
                    v_prev[i], v_now[i] = v_now[i], self._vals(cs[i], e_now[i])
                sl = float(sum(ell[i] for i in cand))
                time_ += world.real_cd_l(rng, len(cand), W, sl)
                layers.append((len(cand), sl))
                depth += 1
                deep_extra += 1
        if warm:
            K = Ktab
        else:
            Jk = {K_: sum(v_now[i][K_] for i in range(n)) - theta * world.T_core(b, K_, suml_all) for K_ in Ks}
            K = max(Jk, key=Jk.get)
        self.k_log[K] = self.k_log.get(K, 0) + 1
        acc = [P.last_node(t, t.topk(K, False))[0] for t in trees]
        com = [min(a + 1, rem[i]) for i, a in enumerate(acc)]
        t_round = time_ + world.real_rtt(rng, rtt) + world.real_T_l(rng, b, K, suml_all)
        if not warm and self.wmode in ("derived", "cellcal", "replay14"):
            rep = self._replay(trees, cs, rem, ell, layers, W, b, theta, deep_extra)
            self._post(b, cls, W, tag, rep, rids, slots, com, t_round, theta, layers)
        # per-request first-position hit EMA
        for r, j in slots:
            hit = float(world.reqs[r]["H"][j] == 0)
            self.h_req[r] = hit if r not in self.h_req else 0.9 * self.h_req[r] + 0.1 * hit
        tok = float(sum(com))
        self.tok_ema = tok if self.tok_ema is None else 0.95 * self.tok_ema + 0.05 * tok
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
