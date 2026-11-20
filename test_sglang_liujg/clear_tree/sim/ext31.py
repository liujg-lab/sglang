"""Round 40 (exploratory, not pre-registered): a mirror of the online K and W rules of the current
CLEAR_TREE_REMOTE code (branch clear-tree-remote, ba6911d670), plus switches for the closed-loop changes
proposed on 2026-11-18. Synthetic world from the round-20+ simulator; not a measurement.

Mirror (all flags off, ledger on):
  - Theta_b: EMA of tokens / round ms, step max(0.05, 1/n), log-log prior from other b, weight n/(n+10).
  - Verify table T(b, K, bucket): T_K = RTT + verify (seed layer removed), last 16 samples, one spike dropped.
    bucket = smallest of 256..4096 holding max_ell + K + 1; none -> sample dropped. fitting_ks -> [2] when no K fits.
  - Layer fit per W: forgetting LS (0.999) on [1, pow2(active), sum_ell], ridge (0, 1e-6, 128^2), >= 4 samples,
    span check on sum_ell with pad max(128, span); non-positive -> None.
  - c_i shrink (lambda 3), next-layer gain ratio r (global, calibrated at flat K), keep_by_net_benefit over (K, m).
  - K probes (nearest to the flat tier, 2 samples per cell, budget max(2*|K|, rounds/20), counted when chosen).
  - W: probe_w (300 idle / fill to 4 samples / early when unpriced), explore every 20 eligible rounds, replay of
    narrower widths (truncate, stop at depth-2..depth, K' by J, +1 bonus), counterfactual time, cumulative arms,
    judge every 20 counted rounds, 30 pairs, 3%.
  - Settlement happens when the next round starts; a round whose request set changes is excluded.
Flags (the 2026-11-18 proposal):
  k1       overflow bin for long prefixes, fitting_ks never collapses to [2], over-bucket rounds keep Theta.
  level    common level L_b in the verify table.
  refresh  re-probe a neighbour of the last value-rule K after 400 rounds; probes counted only when a sample lands.
  selfb    W arms compare self-replay of the built width instead of the real round.
  bias     per-(bucket, W) replay bias from rounds that really built W; gains corrected by it.
  win      arms are sliding windows of 60 pairs.
  confirm  confirm phase (build candidate on alternate rounds, 20 settled rounds), post-switch window with denser
           exploration of the old width, immediate rollback, 1000-round block.
  guard    flat-policy guard: guard rounds (20n+10), paired replay online vs flat, hold flat for 600 rounds
           when flat is 3% better over >= 60 pairs.
  rcell    gain ratio per (b bucket, W), calibrated at the round's flat tier.
"""
import math
from collections import deque

import numpy as np

import ext2 as X

KS = (2, 4, 8, 16, 32, 64)
WS = (2, 4, 8, 16)
KV = (256, 512, 1024, 2048, 4096)
OVER = -1
TAU = 0.5
MAXD = 8
FORGET = 0.999
LAM = 3.0
EXPLORE_EVERY = 20
PROBE_RUNS = 2
VERIFY_WINDOW = 16
LAYER_MIN = 4
PROBE_W_IDLE = 300
REPLAY_GAIN = 0.03
MIN_PAIRS = 30
# closed-loop constants (proposal initial values)
LEVEL_STEP, LEVEL_SPIKE = 0.1, 0.5
PROBE_STALE = 400
ARM_WINDOW = 60
BIAS_WINDOW, BIAS_MIN = 60, 8
CONFIRM_ROUNDS, CONFIRM_MAX = 20, 60
POST_ROUNDS, POST_EVERY, BLOCK_ROUNDS = 300, 5, 1000
GUARD_WINDOW, GUARD_MIN, GUARD_GAIN, GUARD_HOLD = 120, 60, 0.03, 600
GAIN_MIN_MASS = 20.0


def flat_w(b):
    return 8 if b == 1 else 4


def flat_k(b):
    return 16 if b == 1 else 8


def bbucket(b):
    return 1 if b == 1 else (2 if b <= 4 else 5)


def gbucket(n):
    p = 1
    while p < n:
        p *= 2
    return p


def smallest(needed):
    for x in KV:
        if needed <= x:
            return x
    return None


def wider(w):
    c = [x for x in WS if x > w]
    return min(c) if c else None


def narrower(w):
    c = [x for x in WS if x < w]
    return max(c) if c else None


def tier_for(need, fitting):
    tiers = list(fitting) or list(KS)
    enough = [k for k in tiers if k >= need]
    return min(enough) if enough else max(tiers)


def mass(sorted_desc, k):
    return float(sum(sorted_desc[:k]))


def steady_mean(rows):
    kept = list(rows)
    if len(kept) >= 2:
        big = max(kept)
        rest = list(kept)
        rest.remove(big)
        if big > float(np.median(rest)) * 1.5:
            kept = rest
    return sum(kept) / len(kept)


def raw_marginal(scores_desc, latest, k):
    if not scores_desc:
        return 0.0
    th = scores_desc[k - 1] if len(scores_desc) >= k else 0.0
    return float(sum(max(0.0, s - th) for s in latest))


def keep_by_net_benefit(rows, theta, times, layer_cost, kappa):
    """Mirror of clear_tree_policy.keep_by_net_benefit. rows: dict e, g, ell, c, item, growing."""
    growing = [r for r in rows if r["growing"]]
    if not growing:
        return []
    slope = 0.0 if kappa is None else float(kappa)
    joint, gains = {}, {}
    for k, spent in times.items():
        if spent is None:
            continue
        ben = 0.0
        per = {}
        for idx, r in enumerate(rows):
            ben += r["c"] * r["e"].get(k, 0.0)
            per[idx] = r["c"] * r["g"].get(k, 0.0)
        joint[k] = ben - theta * spent
        gains[k] = per
    if not joint:
        return None
    stop_value = max(joint.values())
    best = None
    any_cost = False
    gidx = [i for i, r in enumerate(rows) if r["growing"]]
    for k, base in joint.items():
        ranked = []
        for i in gidx:
            g = gains[k][i]
            ell = rows[i]["ell"]
            ranked.append((g - theta * slope * ell, g, ell, rows[i]["item"]))
        ranked.sort(key=lambda x: x[0], reverse=True)
        rg, re_ = 0.0, 0
        for m, (_n, g, ell, _it) in enumerate(ranked, start=1):
            rg += g
            re_ += ell
            cost = layer_cost(m, re_)
            if cost is None:
                continue
            any_cost = True
            val = base + rg - theta * cost
            choice = (val, k, m)
            if best is None or choice > best[0]:
                best = (choice, [x[3] for x in ranked[:m]])
    if not any_cost or best is None:
        return None
    if best[0][0] > stop_value:
        return list(best[1])
    return []


class Reg:
    def __init__(self):
        self.A = np.zeros((3, 3))
        self.y = np.zeros(3)
        self.n = 0
        self.lo = self.hi = None
        self._c = None

    def add(self, m, sl, t):
        x = np.array([1.0, float(gbucket(m)), float(sl)])
        self.A = FORGET * self.A + np.outer(x, x)
        self.y = FORGET * self.y + x * float(t)
        self.n += 1
        self._c = None
        self.lo = sl if self.lo is None else min(self.lo, sl)
        self.hi = sl if self.hi is None else max(self.hi, sl)

    def coef(self):
        if self.n < LAYER_MIN:
            return None
        if self._c is None:
            S = self.A.copy()
            S[1, 1] += 1e-6
            S[2, 2] += 128.0 ** 2
            try:
                self._c = np.linalg.solve(S, self.y)
            except np.linalg.LinAlgError:
                return None
        return self._c

    def pred(self, m, sl):
        c = self.coef()
        if c is None:
            return None
        pad = max(128.0, self.hi - self.lo)
        if sl < self.lo - pad or sl > self.hi + pad:
            return None
        v = float(c[0] + c[1] * gbucket(m) + c[2] * sl)
        return v if v > 0 else None

    def slope(self):
        c = self.coef()
        return None if c is None else float(c[2])


class TreeView:
    """Per-layer node ids and the true-path walk of one sim tree."""

    def __init__(self, t):
        self.t = t
        self.layers = []          # list of node-id lists, layer 0 = first expansion

    def add_layer(self):
        self.layers.append(list(self.t.frontier))

    def scores_upto(self, d):
        sc = self.t.score
        out = []
        for lay in self.layers[:d]:
            out.extend(sc[n] for n in lay)
        out.sort(reverse=True)
        return out

    def latest(self, d):
        sc = self.t.score
        return [sc[n] for n in self.layers[d - 1]]


def parents_of(t):
    par = {}
    for v, kd in enumerate(t.kids):
        for c in kd.values():
            par[c] = v
    return par


def truncate(view, par, w):
    """kept[d] = node ids kept at layer d when the tree is cut to width w."""
    sc = view.t.score
    kept, prev = [], {0}
    for lay in view.layers:
        cand = [n for n in lay if par.get(n) in prev]
        cand.sort(key=lambda n: sc[n], reverse=True)
        sel = cand[:w]
        if not sel:
            break
        kept.append(sel)
        prev = set(sel)
    return kept


def kept_scores(view, kept, end):
    sc = view.t.score
    out = []
    for d in range(min(end, len(kept))):
        out.extend(sc[n] for n in kept[d])
    out.sort(reverse=True)
    return out


def replay_accept(view, kept, stop, kp, committed):
    """Accepted drafts on the top kp kept nodes before stop, plus one when the committed text is longer."""
    t = view.t
    sc = t.score
    nodes = []
    for d in range(min(stop, len(kept))):
        for n in kept[d]:
            nodes.append((-sc[n], d, n))
    nodes.sort()
    chosen = {n for _s, _d, n in nodes[:max(kp, 0)]}
    v, a, H = 0, 0, t.req["H"]
    while True:
        h = H[t.pos[v]]
        c = t.kids[v].get(int(h)) if h >= 0 else None
        if c is None or c not in chosen:
            break
        a += 1
        v = c
    a = min(a, committed)
    return a + 1 if committed > a else a


class CodeMirror:
    def __init__(self, ledger=True, k1=False, level=False, refresh=False, selfb=False, bias=False, win=False,
                 confirm=False, guard=False, rcell=False, llevel=False, guard_rounds=True):
        self.ledger, self.k1, self.levelf, self.refresh = ledger, k1, level, refresh
        self.llevel = llevel
        self.guard_rounds = guard_rounds
        self.lscale, self.lscale_ref = 1.0, 1.0
        self.lratios = deque(maxlen=3)
        self.selfb, self.biasf, self.win, self.confirm, self.guard, self.rcell = selfb, bias, win, confirm, guard, rcell
        self.rounds = 0
        self.probe_rounds = 0
        self.clock, self.clock_round, self.level, self.last_k = {}, {}, {}, {}
        self.pn, self.ptok, self.pms = {}, {}, {}
        self.sa, self.se = {}, {}
        self.ta = self.te = 0.0
        self.gp = self.gact = 0.0
        self.gcell = {}
        self.reg = {}
        self.persist = {}
        self.elig_n, self.rep_n, self.bk_rounds = {}, {}, {}
        self.arms = {}
        self.narrow_idle, self.pw_left, self.pw_fill = {}, {}, {}
        self.selfbias, self.confirm_st, self.post, self.block = {}, {}, {}, {}
        self.gwin, self.gnew, self.hold_until = {}, {}, {}
        self.prev = None
        self.prev_rids = None
        # logs
        self.w_log, self.k_log, self.kind_log, self.src_log = {}, {}, {}, {}
        self.switch_log, self.events = [], []
        self.n_tau = self.n_value = self.n_over = self.n_excl = 0
        self.n_hold_rounds = 0

    # ------------------------------------------------------------ learned quantities
    def _bin(self, needed):
        s = smallest(needed)
        if s is None and self.k1:
            return OVER
        return s

    def _fitting(self, max_ell):
        found = [k for k in KS if smallest(max_ell + k + 1) is not None]
        if found:
            return found
        return list(KS) if self.k1 else [KS[0]]

    def _cover(self, w, fitting):
        return tier_for(w * (MAXD - 1), fitting)

    def _covered_flat(self, b, max_ell, cover, fitting):
        pref = flat_k(b)
        if pref not in fitting:
            pref = max(fitting)
        fit = [k for k in fitting if k <= cover]
        if pref in fit:
            return pref
        if fit:
            return max(fit)
        return pref

    def _T(self, b, k, binv):
        if binv is None:
            return None
        rows = self.clock.get((b, k, binv))
        if not rows:
            return None
        lv = self.level.get(b, 0.0) if self.levelf else 0.0
        return steady_mean(rows) + lv

    def _time(self, b, max_ell, k):
        return self._T(b, k, self._bin(max_ell + k + 1))

    def _observe_verify(self, b, k, binv, ms, src):
        if binv is None:
            return
        key = (b, k, binv)
        rows = self.clock.setdefault(key, [])
        lv = self.level.get(b, 0.0) if self.levelf else 0.0
        if self.levelf and rows:
            pred = steady_mean(rows) + lv
            resid = ms - pred
            if pred > 0 and abs(resid) <= LEVEL_SPIKE * pred:
                lv += LEVEL_STEP * resid
                self.level[b] = lv
        rows.append(ms - lv)
        if len(rows) > VERIFY_WINDOW:
            del rows[:-VERIFY_WINDOW]
        self.clock_round[key] = self.rounds
        if src == "value":
            self.last_k[b] = k
        if self.refresh and src in ("fill", "refresh"):
            self.probe_rounds += 1

    def _probe_limit(self):
        return max(PROBE_RUNS * len(KS), self.rounds // EXPLORE_EVERY)

    def _fill_probe(self, max_ell, b, cover, flat):
        ranked = sorted((k for k in KS if k <= cover and k != flat),
                        key=lambda k: (abs(math.log2(k) - math.log2(flat)), k))
        for k in ranked:
            binv = self._bin(max_ell + k + 1)
            if binv is None:
                continue
            if len(self.clock.get((b, k, binv), ())) >= PROBE_RUNS:
                continue
            return k
        return None

    def _refresh_probe(self, max_ell, b, cover, fitting):
        last = self.last_k.get(b)
        if last is None:
            return None
        i = KS.index(last)
        for j in (i - 1, i + 1):
            if not 0 <= j < len(KS):
                continue
            k = KS[j]
            if k > cover or k not in fitting:
                continue
            binv = self._bin(max_ell + k + 1)
            if binv is None:
                continue
            key = (b, k, binv)
            if key in self.clock and self.rounds - self.clock_round.get(key, 0) > PROBE_STALE:
                return k
        return None

    def _raw_theta(self, b):
        n = self.pn.get(b, 0)
        ms = self.pms.get(b, 0.0)
        if n <= 0 or ms <= 0:
            return None
        return self.ptok[b] / ms

    def _theta(self, b):
        raw = self._raw_theta(b)
        usable = [(k, self._raw_theta(k)) for k in self.pn if k != b]
        usable = [(k, v) for k, v in usable if v is not None and v > 0]
        prior = None
        if len(usable) == 1:
            prior = usable[0][1] * (b / usable[0][0])
        elif len(usable) > 1:
            xs = [math.log(k) for k, _ in usable]
            ys = [math.log(v) for _, v in usable]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            vx = sum((x - mx) ** 2 for x in xs)
            if vx > 0:
                sl = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / vx
                prior = math.exp(my - sl * mx + sl * math.log(b))
        if raw is None:
            return prior
        if prior is None:
            return raw
        n = float(self.pn.get(b, 0))
        wgt = n / (n + 10.0)
        return wgt * raw + (1 - wgt) * prior

    def _observe_price(self, b, tok, ms):
        if ms <= 0:
            return
        n = self.pn.get(b, 0) + 1
        st = max(0.05, 1.0 / n)
        self.pn[b] = n
        self.ptok[b] = (1 - st) * self.ptok.get(b, 0.0) + st * tok
        self.pms[b] = (1 - st) * self.pms.get(b, 0.0) + st * ms

    def _c(self, rid):
        pooled = self.ta / self.te if self.te > 0 else 1.0
        return (self.sa.get(rid, 0.0) + LAM * pooled) / (self.se.get(rid, 0.0) + LAM)

    def _gain_ratio(self, cell=None):
        if self.rcell and cell is not None:
            st = self.gcell.get(cell)
            if st is not None and st[0] >= GAIN_MIN_MASS:
                return min(1.0, (st[1] + LAM) / (st[0] + LAM))
        if self.gp == 0.0 and self.gact == 0.0:
            return 1.0
        return min(1.0, (self.gact + LAM) / (self.gp + LAM))

    def _observe_gain(self, pred, act, cell):
        self.gp = self.gp * FORGET + pred
        self.gact = self.gact * FORGET + act
        if self.rcell:
            st = self.gcell.setdefault(cell, [0.0, 0.0])
            st[0] = st[0] * FORGET + pred
            st[1] = st[1] * FORGET + act

    def _pred_layer(self, w, m, sl):
        r = self.reg.get(w)
        p = None if r is None else r.pred(m, sl)
        if p is not None and self.llevel:
            p *= self.lscale
        return p

    def _slope(self, w):
        r = self.reg.get(w)
        s_ = None if r is None else r.slope()
        if s_ is not None and self.llevel:
            s_ *= self.lscale
        return s_

    def _layer_add(self, w, m, sl, dt):
        reg = self.reg.setdefault(w, Reg())
        if not self.llevel:
            reg.add(m, sl, dt)
            return
        p = reg.pred(m, sl)
        if p is not None:
            self.lratios.append(dt / (p * self.lscale))
            if len(self.lratios) == self.lratios.maxlen:
                med = sorted(self.lratios)[len(self.lratios) // 2]
                f = min(2.0, max(0.5, 1.0 + 0.3 * (med - 1.0)))
                self.lscale *= f
                self.lratios = deque((r / f for r in self.lratios), maxlen=3)
                if abs(self.lscale / self.lscale_ref - 1.0) > 0.3:
                    self.lscale_ref = self.lscale
                    self.arms.clear()
                    self.rep_n.clear()
                    self.events.append(("layer_level", self.rounds, round(self.lscale, 3)))
        reg.add(m, sl, dt / self.lscale)

    def _layer_n(self, w):
        r = self.reg.get(w)
        return 0 if r is None else r.n

    def _persist(self, b):
        return self.persist.get(bbucket(b), flat_w(b))

    # ------------------------------------------------------------ W choice
    def _mark_ran(self, bk, w):
        self.narrow_idle[(bk, w)] = 0
        self.pw_left[(bk, w)] = 0

    def _held(self, bk):
        return self.guard and self.bk_rounds.get(bk, 0) < self.hold_until.get(bk, -1)

    def _choose_w(self, b, eligible, suml):
        base = self._persist(b)
        if not eligible:
            return base, "ordinary"
        bk = bbucket(b)
        n = self.elig_n.get(bk, 0) + 1
        self.elig_n[bk] = n
        if self.confirm and bk in self.confirm_st:
            c = self.confirm_st[bk]
            if n - c["start"] > CONFIRM_MAX:
                self.block[(bk, c["cand"])] = n + BLOCK_ROUNDS // 4
                self.events.append(("confirm_abort", self.rounds, bk, c["cand"]))
                del self.confirm_st[bk]
            else:
                c["n"] += 1
                if c["n"] % 2 == 1:
                    return c["cand"], "confirm"
                self._mark_ran(bk, base)
                return base, "ordinary"
        nar = narrower(base)
        if nar is not None:
            key = (bk, nar)
            left = self.pw_left.get(key, 0)
            idle = self.narrow_idle.get(key, PROBE_W_IDLE)
            samples = self._layer_n(nar)
            if left > 0 and self.pw_fill.get(key, False) and samples >= LAYER_MIN:
                self.pw_left[key] = 0
                self.pw_fill[key] = False
                left = 0
            early = (left <= 0 and samples >= LAYER_MIN and EXPLORE_EVERY <= idle < PROBE_W_IDLE
                     and self._pred_layer(nar, b, suml) is None)
            if left > 0 or idle >= PROBE_W_IDLE or early:
                if left <= 0:
                    if early:
                        left = 1
                        self.pw_fill[key] = False
                    else:
                        fill = samples < LAYER_MIN
                        left = LAYER_MIN if fill else PROBE_RUNS
                        self.pw_fill[key] = fill
                self.pw_left[key] = left - 1
                self.narrow_idle[key] = 0
                return nar, "probe_w"
            self.narrow_idle[key] = idle + 1
        if self.guard and self.guard_rounds and n % EXPLORE_EVERY == EXPLORE_EVERY // 2:
            return flat_w(b), "guard"
        every = EXPLORE_EVERY
        post = self.post.get(bk)
        if self.confirm and post is not None and n < post["until"] and post["prev"] > base:
            every = POST_EVERY
        if n % every == 0:
            nx = wider(base)
            if nx is not None:
                self._mark_ran(bk, nx)
                return nx, "explore"
        self._mark_ran(bk, base)
        return base, "ordinary"

    # ------------------------------------------------------------ round plan
    def _plan(self, b, max_ell, suml):
        self.rounds += 1
        bk = bbucket(b)
        self.bk_rounds[bk] = self.bk_rounds.get(bk, 0) + 1
        fitting = self._fitting(max_ell)
        if not self.ledger:
            w = flat_w(b)
            k = self._covered_flat(b, max_ell, self._cover(w, fitting), fitting)
            return dict(W=w, kind="off", use_tau=True, locked=k, k=k, cover=self._cover(w, fitting), price=None,
                        repl=False, bias_ok=False, src="tau", flat=k, served=w)
        served = self._persist(b)
        pcover = self._cover(served, fitting)
        sflat = self._covered_flat(b, max_ell, pcover, fitting)
        flat_ready = self._time(b, max_ell, sflat) is not None
        held = self._held(bk)
        probe, pkind = None, None
        if not held and flat_ready and self.probe_rounds < self._probe_limit():
            probe = self._fill_probe(max_ell, b, pcover, sflat)
            pkind = "fill" if probe is not None else None
            if probe is None and self.refresh:
                probe = self._refresh_probe(max_ell, b, pcover, fitting)
                pkind = "refresh" if probe is not None else None
            if probe is not None and not self.refresh:
                self.probe_rounds += 1
        price = self._theta(b)
        eligible = probe is None and not held
        if held:
            w, kind = flat_w(b), "hold"
            self.n_hold_rounds += 1
        else:
            w, kind = self._choose_w(b, eligible, suml)
        if kind in ("guard", "hold"):
            cover = self._cover(w, fitting)
            k = self._covered_flat(b, max_ell, cover, fitting)
            return dict(W=w, kind=kind, use_tau=True, locked=k, k=k, cover=cover, price=price, repl=False,
                        bias_ok=False, src="tau", flat=k, served=served)
        layer_ms = self._pred_layer(w, b, suml)
        cover = self._cover(w, fitting)
        ftier = self._covered_flat(b, max_ell, cover, fitting)
        ready = self._time(b, max_ell, ftier) is not None
        use_tau = probe is not None or price is None or layer_ms is None or not ready
        if probe is not None:
            chosen = locked = probe
        else:
            chosen = ftier
            locked = chosen if use_tau else None
        return dict(W=w, kind=kind, use_tau=use_tau, locked=locked, k=chosen, cover=cover, price=price,
                    layer_ms=layer_ms, repl=eligible and not use_tau and kind != "probe_w",
                    bias_ok=eligible and not use_tau, src=pkind or ("tau" if use_tau else "value"),
                    flat=ftier, served=served)

    # ------------------------------------------------------------ growth decisions
    def _times(self, b, max_ell, cap):
        out = {}
        for k in KS:
            if k > cap:
                continue
            t = self._time(b, max_ell, k)
            if t is not None:
                out[k] = t
        return out

    def _choose_keep(self, views, depth, elig, plan, cs, ell, b, max_ell):
        W = plan["W"]
        times = self._times(b, max_ell, plan["cover"])
        ratio = self._gain_ratio((bbucket(b), W))
        rows = []
        eset = set(elig)
        for i, v in enumerate(views):
            sd = v.scores_upto(depth[i])
            lat = v.latest(depth[i])
            e, g = {}, {}
            for k in times:
                e[k] = mass(sd, k)
                g[k] = raw_marginal(sd, lat, k) * ratio
            rows.append(dict(e=e, g=g, ell=ell[i], c=cs[i], item=i, growing=i in eset))
        theta = plan["price"]
        kept = keep_by_net_benefit(rows, theta, times, lambda m, sl: self._pred_layer(W, m, sl), self._slope(W))
        if kept is not None:
            return kept
        # _continue fallback over the eligible requests only
        if plan.get("layer_ms") is None:
            return []
        cur, nxt = {}, {}
        for k, t in times.items():
            bc = bn = 0.0
            for i in elig:
                sd = views[i].scores_upto(depth[i])
                bc += cs[i] * mass(sd, k)
                bn += cs[i] * (mass(sd, k) + raw_marginal(sd, views[i].latest(depth[i]), k) * ratio)
            cur[k] = bc - theta * t
            nxt[k] = bn - theta * t
        if not cur:
            return []
        if max(nxt.values()) - theta * plan["layer_ms"] > max(cur.values()):
            return list(elig)
        return []

    def _select_k(self, views, depth, plan, cs, b, max_ell):
        nodes = max(depth[i] * plan["W"] for i in range(len(views)))
        fit = [k for k in KS if k <= plan["cover"]]
        cap = tier_for(nodes, fit) if nodes > 0 and fit else plan["cover"]
        vals = {}
        for k in KS:
            if k > cap:
                continue
            t = self._time(b, max_ell, k)
            if t is None:
                continue
            ben = sum(cs[i] * mass(views[i].scores_upto(depth[i]), k) for i in range(len(views)))
            vals[k] = ben - plan["price"] * t
        if not vals:
            return plan["k"]
        return max(vals, key=lambda k: (vals[k], k))

    # ------------------------------------------------------------ round
    def round(self, world, slots, rng, rtt, rnd, trees_out=None):
        b = len(slots)
        rids = [r for r, _ in slots]
        if self.prev is not None:
            self._settle(self.prev, rids)
            self.prev = None
        ell = [world.reqs[r]["p0"] + j for r, j in slots]
        max_ell = max(ell)
        suml = float(sum(ell))
        plan = self._plan(b, max_ell, suml)
        W = plan["W"]
        world.first_W = W
        self.w_log[W] = self.w_log.get(W, 0) + 1
        self.kind_log[plan["kind"]] = self.kind_log.get(plan["kind"], 0) + 1
        self.src_log[plan["src"]] = self.src_log.get(plan["src"], 0) + 1
        if plan["use_tau"]:
            self.n_tau += 1
        else:
            self.n_value += 1
        if smallest(max_ell + plan["k"] + 1) is None or smallest(max_ell + 3) is None:
            self.n_over += 1
        params = (world.a, world.g, world.beta, 0.0)
        trees = [X.Tree2(world.reqs[r], j, world) for r, j in slots]
        views = [TreeView(t) for t in trees]
        n = len(trees)
        for t, v in zip(trees, views):
            t.expand(W, rng, params)
            v.add_layer()
        rem = [world.reqs[r]["L"] - j for r, j in slots]
        cs = [self._c(r) for r in rids]
        depth = [1] * n
        prev_e = [0.0] * n
        raw_gain = {}
        cal_k = plan["flat"] if self.rcell else flat_k(b)
        cell = (bbucket(b), W)
        self._begin_layer(views, list(range(n)), depth, raw_gain, cal_k, cell)
        active = list(range(n))
        time_ = 0.0
        rows, samples = [], []
        while active:
            if plan["use_tau"]:
                keep = []
                for i in active:
                    if depth[i] >= MAXD - 1:
                        continue
                    e = mass(views[i].scores_upto(depth[i]), plan["k"])
                    if e - prev_e[i] > TAU:
                        prev_e[i] = e
                        keep.append(i)
            else:
                elig = [i for i in active if depth[i] < MAXD - 1]
                keep = self._choose_keep(views, depth, elig, plan, cs, ell, b, max_ell) if elig else []
            if not keep:
                break
            m = len(keep)
            sl = float(sum(ell[i] for i in keep))
            dt = world.real_cd_l(rng, m, W, sl)
            time_ += dt
            rows.append((m, sl))
            samples.append((m, sl, dt))
            for i in keep:
                trees[i].expand(W, rng, params)
                views[i].add_layer()
                depth[i] += 1
            self._begin_layer(views, keep, depth, raw_gain, cal_k, cell)
            active = keep
        if plan["locked"] is not None:
            K = plan["locked"]
        else:
            K = self._select_k(views, depth, plan, cs, b, max_ell)
        self.k_log[K] = self.k_log.get(K, 0) + 1
        acc = [_accepted(t, t.topk(K, False)) for t in trees]
        com = [min(a + 1, rem[i]) for i, a in enumerate(acc)]
        rt = world.real_rtt(rng, rtt)
        tv = world.real_T_l(rng, b, K, suml)
        t_round = time_ + rt + tv
        t_k = rt + tv - world.c_d_l(b, W, suml)
        e_ver = [mass(views[i].scores_upto(depth[i]), K) for i in range(n)]
        snap = None
        if self.ledger:
            snap = dict(verify={k: self._time(b, max_ell, k) for k in KS},
                        preds={w: [self._pred_layer(w, m, sl) for (m, sl) in rows] for w in WS},
                        ratios=list(cs), gain=self._gain_ratio(cell))
        self.prev = dict(plan=plan, b=b, rids=rids, max_ell=max_ell, K=K, com=com, acc=acc, views=views,
                         depth=depth, rows=rows, samples=samples, t_round=t_round, t_k=t_k, e_ver=e_ver, snap=snap,
                         first=self.prev_rids is None)
        self.prev_rids = rids
        return com, t_round

    def _begin_layer(self, views, touched, depth, raw_gain, k, cell):
        for i in touched:
            d = depth[i]
            v = views[i]
            sd = v.scores_upto(d)
            if d >= 2 and i in raw_gain:
                prev = v.scores_upto(d - 1)
                inc = mass(sd, k) - mass(prev, k)
                self._observe_gain(raw_gain[i], inc, cell)
            raw_gain[i] = raw_marginal(sd, v.latest(d), k)

    # ------------------------------------------------------------ settlement
    def _settle(self, pv, next_rids):
        plan = pv["plan"]
        b = pv["b"]
        nxt = set(next_rids)
        excluded = pv["first"] or nxt != set(pv["rids"])
        if excluded:
            self.n_excl += 1
        over = smallest(pv["max_ell"] + pv["K"] + 1) is None
        # c_i (settles even on excluded rounds, only for requests still present)
        for i, r in enumerate(pv["rids"]):
            if r in nxt:
                a = max(pv["com"][i] - 1, 0)
                e = pv["e_ver"][i]
                if a or e:
                    self.sa[r] = self.sa.get(r, 0.0) + a
                    self.se[r] = self.se.get(r, 0.0) + e
                    self.ta += a
                    self.te += e
            else:
                self.sa.pop(r, None)
                self.se.pop(r, None)
        if not self.ledger:
            return
        if not excluded:
            binv = self._bin(pv["max_ell"] + pv["K"] + 1)
            self._observe_verify(b, pv["K"], binv, pv["t_k"], plan["src"])
            for (m, sl, dt) in pv["samples"]:
                self._layer_add(plan["W"], m, sl, dt)
        price_kept = not excluded and (self.k1 or not over)
        if price_kept:
            self._observe_price(b, float(sum(pv["com"])), pv["t_round"])
        if not price_kept:
            return
        if pv["snap"] is None or plan["price"] is None:
            return
        if plan["served"] != self._persist(b):
            return
        self._replay(pv)

    # ------------------------------------------------------------ replay and W
    def _replay_width(self, pv, w, pars, value=True):
        """(tokens, ms or None) of width w cut from the built trees, under the value rule (or tau if not value)."""
        plan = pv["plan"]
        snap = pv["snap"]
        views = pv["views"]
        kept = [truncate(v, par, w) if w < plan["W"] else [list(l) for l in v.layers] for v, par in zip(views, pars)]
        verify = {k: t for k, t in snap["verify"].items() if t is not None}
        theta = plan["price"]
        preds = snap["preds"][w]
        if value:
            stop = self._batch_stop(views, kept, theta, verify, preds, snap["gain"], snap["ratios"])
            stops = [min(stop, len(kp)) for kp in kept]
            retained = max(sum(len(kp[d]) for d in range(s)) for kp, s in zip(kept, stops))
            cap = tier_for(retained, KS) if retained > 0 else None
            vals = {}
            for k, t in verify.items():
                if cap is not None and k > cap:
                    continue
                vals[k] = sum(c * mass(kept_scores(v, kp, s), k)
                              for v, kp, s, c in zip(views, kept, stops, snap["ratios"])) - theta * t
            kp_ = max(vals, key=lambda k: (vals[k], k)) if vals else plan["flat"]
            nlay = max(stop - 1, 0)
        else:
            kf = pv["guard_k"]
            stops = []
            for v, kp in zip(views, kept):
                prev = 0.0
                s = len(kp)
                for d in range(1, len(kp) + 1):
                    m_ = mass(kept_scores(v, kp, d), kf)
                    if not (m_ - prev > TAU and d < MAXD - 1):
                        s = d
                        break
                    prev = m_
                stops.append(s)
            kp_ = kf
            nlay = max(max(stops) - 1, 0)
        tok = float(sum(replay_accept(v, kp, s, kp_, c) for v, kp, s, c in zip(views, kept, stops, pv["com"])))
        built_preds = snap["preds"][plan["W"]]
        if any(p is None for p in built_preds) or snap["verify"].get(pv["K"]) is None:
            return tok, None
        model_actual = sum(built_preds) + snap["verify"][pv["K"]]
        cand = preds[:nlay]
        if len(cand) < nlay or any(p is None for p in cand) or verify.get(kp_) is None:
            return tok, None
        cf = pv["t_round"] - model_actual + sum(cand) + verify[kp_]
        return tok, (cf if cf > 0 else None)

    def _batch_stop(self, views, kept, theta, verify, preds, gain, ratios):
        depth = max(len(kp) for kp in kept)
        if depth < 2:
            return depth
        for stop in range(max(1, depth - 2), depth):
            pred = preds[stop - 1] if 0 <= stop - 1 < len(preds) else None
            if pred is None:
                return depth
            rows = []
            for idx, (v, kp, c) in enumerate(zip(views, kept, ratios)):
                end = min(stop, len(kp))
                if end < 1:
                    continue
                sd = kept_scores(v, kp, end)
                sc = v.t.score
                lat = [sc[nid] for nid in kp[end - 1]]
                rows.append(dict(e={k: mass(sd, k) for k in KS},
                                 g={k: raw_marginal(sd, lat, k) * gain for k in KS},
                                 ell=0, c=c, item=idx, growing=len(kp) > stop))
            if not rows:
                return depth
            kept_items = keep_by_net_benefit(rows, theta, verify, lambda _m, _l, p=pred: p, 0.0)
            if kept_items is None:
                return depth
            if not kept_items:
                return stop
        return depth

    def _arm_add(self, bk, w, item):
        row = self.arms.setdefault(bk, {})
        if self.win:
            row.setdefault(w, deque(maxlen=ARM_WINDOW)).append(item)
        else:
            st = row.setdefault(w, [0.0, 0.0, 0.0, 0.0, 0])
            st[0] += item[0]; st[1] += item[1]; st[2] += item[2]; st[3] += item[3]; st[4] += 1

    def _arm_sum(self, bk, w):
        st = self.arms.get(bk, {}).get(w)
        if st is None:
            return None
        if self.win:
            if not st:
                return None
            a = [sum(x[i] for x in st) for i in range(4)]
            return a + [len(st)]
        return st

    def _bias(self, bk, w):
        rows = self.selfbias.get((bk, w))
        if not rows or len(rows) < BIAS_MIN:
            return None
        rt, rm, at, am = (sum(x[i] for x in rows) for i in range(4))
        if min(rm, at, am) <= 0:
            return None
        return (rt / rm) / (at / am) - 1.0

    def _gain_of(self, bk, cur, cand):
        st = self._arm_sum(bk, cand)
        if st is None or st[4] < MIN_PAIRS or min(st[0], st[1], st[2], st[3]) <= 0:
            return None, False
        g = (st[0] / st[1]) / (st[2] / st[3]) - 1.0
        if self.biasf:
            bc, bw = self._bias(bk, cur), self._bias(bk, cand)
            if bc is not None and bw is not None:
                return (1 + g) * (1 + bc) / (1 + bw) - 1.0, True
            return g, False
        return g, True

    def _assign(self, bk, w, ev, g):
        old = self.persist.get(bk)
        self.persist[bk] = w
        self.arms.pop(bk, None)
        self.rep_n[bk] = 0
        self.switch_log.append((ev, self.rounds, bk, old, w, round(float(g), 4)))

    def _judge(self, bk, cur):
        best, bg, bcorr = cur, 0.0, False
        adj = narrower(cur)
        n = self.elig_n.get(bk, 0)
        for cand in WS:
            if cand == cur:
                continue
            if self.block.get((bk, cand), -1) > n:
                continue
            g, corr = self._gain_of(bk, cur, cand)
            if g is None:
                continue
            if cand < cur and cand != adj:
                continue
            if g > bg and g > REPLAY_GAIN:
                best, bg, bcorr = cand, g, corr
        if best == cur:
            return
        if not self.confirm:
            self._assign(bk, best, "apply", bg)
            return
        post = self.post.get(bk)
        if post is not None and n < post["until"] and best == post["prev"]:
            self.block[(bk, cur)] = n + BLOCK_ROUNDS
            self._assign(bk, best, "rollback", bg)
            self.post.pop(bk, None)
            return
        if bk not in self.confirm_st:
            self.confirm_st[bk] = dict(cand=best, cur=cur, n=0, settled=0, start=n)
            self.events.append(("confirm_start", self.rounds, bk, cur, best, round(bg, 4)))

    def _finish_confirm(self, bk):
        c = self.confirm_st.pop(bk)
        cur = self._persist_bk(bk)
        g, corr = self._gain_of(bk, cur, c["cand"])
        n = self.elig_n.get(bk, 0)
        if g is not None and corr and g > REPLAY_GAIN:
            self._assign(bk, c["cand"], "apply", g)
            self.post[bk] = dict(prev=cur, until=n + POST_ROUNDS)
        else:
            self.block[(bk, c["cand"])] = n + BLOCK_ROUNDS
            self.events.append(("reject", self.rounds, bk, cur, c["cand"], None if g is None else round(g, 4)))

    def _persist_bk(self, bk):
        return self.persist.get(bk, 8 if bk == 1 else 4)

    def _replay(self, pv):
        plan = pv["plan"]
        b = pv["b"]
        bk = bbucket(b)
        built = plan["W"]
        kind = plan["kind"]
        cur = self._persist(b)
        actual = (float(sum(pv["com"])), pv["t_round"])
        need_self = self.selfb or self.biasf or self.guard
        do_arms = plan["repl"]
        do_bias = self.biasf and plan["bias_ok"]
        do_guard = self.guard and (kind in ("guard", "hold") or (plan["repl"] and kind == "ordinary"))
        if not (do_arms or do_bias or do_guard):
            return
        pars = [parents_of(v.t) for v in pv["views"]]
        pairs = {}
        if kind in ("guard", "hold"):
            if do_guard and plan["served"] <= built:
                fl = self._replay_flat_self(pv, pars)
                on = self._replay_width(pv, plan["served"], pars, value=True)
                if fl[1] is not None and on[1] is not None:
                    self._guard_add(bk, on, fl)
            return
        if do_arms or do_bias:
            for w in WS:
                if w < built or (w == built and need_self):
                    pairs[w] = self._replay_width(pv, w, pars, value=True)
        selfp = pairs.get(built)
        if do_bias and selfp is not None and selfp[1] is not None:
            self.selfbias.setdefault((bk, built), deque(maxlen=BIAS_WINDOW)).append(
                (selfp[0], selfp[1], actual[0], actual[1]))
        if do_guard and selfp is not None and selfp[1] is not None and flat_w(b) <= built:
            pv["guard_k"] = self._covered_flat(b, pv["max_ell"], self._cover(flat_w(b), self._fitting(pv["max_ell"])),
                                               self._fitting(pv["max_ell"]))
            fl = self._replay_width(pv, flat_w(b), pars, value=False)
            if fl[1] is not None:
                self._guard_add(bk, selfp, fl)
        if not do_arms:
            return
        ref = selfp if self.selfb else actual
        if ref is None or ref[1] is None:
            return
        arms = {}
        if built == cur:
            arms[cur] = ref
            for w, p in pairs.items():
                if w < cur and p[1] is not None:
                    arms[w] = p
        elif built > cur:
            arms[built] = ref
            for w, p in pairs.items():
                if w < built and p[1] is not None:
                    arms[w] = p
        if cur in arms and arms[cur][1] > 0:
            ct, cm = arms[cur]
            for w, p in arms.items():
                if w == cur or p[1] <= 0:
                    continue
                self._arm_add(bk, w, (p[0], p[1], ct, cm))
        self.rep_n[bk] = self.rep_n.get(bk, 0) + 1
        if self.confirm and bk in self.confirm_st and kind in ("confirm", "ordinary"):
            self.confirm_st[bk]["settled"] += 1
            if self.confirm_st[bk]["settled"] >= CONFIRM_ROUNDS:
                self._finish_confirm(bk)
                return
        if self.rep_n[bk] % EXPLORE_EVERY == 0:
            self._judge(bk, cur)

    def _replay_flat_self(self, pv, pars):
        pv["guard_k"] = pv["plan"]["k"]
        return self._replay_width(pv, pv["plan"]["W"], pars, value=False)

    def _guard_add(self, bk, on, fl):
        d = self.gwin.setdefault(bk, deque(maxlen=GUARD_WINDOW))
        d.append((on[0], on[1], fl[0], fl[1]))
        self.gnew[bk] = self.gnew.get(bk, 0) + 1
        if self.gnew[bk] % 20 == 0 and len(d) >= GUARD_MIN and not self._held(bk):
            ot, om, ft, fm = (sum(x[i] for x in d) for i in range(4))
            if min(ot, om, ft, fm) > 0:
                g = (ft / fm) / (ot / om) - 1.0
                if g > GUARD_GAIN:
                    self.hold_until[bk] = self.bk_rounds.get(bk, 0) + GUARD_HOLD
                    self.events.append(("guard_trip", self.rounds, bk, round(g, 4)))


def _accepted(t, ver):
    v, a, H = 0, 0, t.req["H"]
    while True:
        h = H[t.pos[v]]
        c = t.kids[v].get(int(h)) if h >= 0 else None
        if c is None or c not in ver:
            break
        a += 1
        v = c
    return a


ARMS = {
    "off_code": dict(ledger=False),
    "off_k1": dict(ledger=False, k1=True),
    "code": dict(),
    "k1": dict(k1=True),
    "kloop_nl": dict(k1=True, level=True, refresh=True),
    "wloop_nl": dict(k1=True, level=True, refresh=True, selfb=True, bias=True, win=True, confirm=True),
    "kloop": dict(k1=True, level=True, refresh=True, llevel=True),
    "wloop": dict(k1=True, level=True, refresh=True, llevel=True, selfb=True, bias=True, win=True, confirm=True),
    "full": dict(k1=True, level=True, refresh=True, llevel=True, selfb=True, bias=True, win=True, confirm=True,
                 guard=True),
    "full_r": dict(k1=True, level=True, refresh=True, llevel=True, selfb=True, bias=True, win=True, confirm=True,
                   guard=True, guard_rounds=False),
    "k1_lv": dict(k1=True, level=True),
    "k1_rf": dict(k1=True, refresh=True),
    "k1_ll": dict(k1=True, llevel=True),
    "kl_selfb": dict(k1=True, level=True, refresh=True, llevel=True, selfb=True),
    "kl_bias": dict(k1=True, level=True, refresh=True, llevel=True, bias=True),
    "kl_win": dict(k1=True, level=True, refresh=True, llevel=True, win=True),
    "kl_conf": dict(k1=True, level=True, refresh=True, llevel=True, confirm=True),
    "kl_sbw": dict(k1=True, level=True, refresh=True, llevel=True, selfb=True, bias=True, win=True),
    "rec": dict(k1=True, llevel=True, win=True),
    "rec_lr": dict(k1=True, level=True, refresh=True, llevel=True, win=True),
}
