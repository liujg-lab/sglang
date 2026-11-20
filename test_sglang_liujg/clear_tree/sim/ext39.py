"""Round 47 (exploratory, not pre-registered): indicator W arms from the round-46 screen, with the post-switch
verify / rollback of satpv_g. Synthetic world; not a measurement.

Per b bucket, on every ordinary round at the served width W (value rule), EMA step max(1/200, 1/n):
  bind    sat_g's saturation share
  tok     round tokens
  dcu     Theta_b * (layer_fit(2W) - layer_fit(W)) over this round's layers   (cost of widening, tokens)
  dcd     Theta_b * (layer_fit(W) - layer_fit(W/2)) over this round's layers  (saving of narrowing, tokens)
  fo_up   sum c_i (topK(tree + cut candidates) - topK(tree))                   (first-order gain of widening)
  loss    sum c_i (topK(tree) - topK(tree cut to W/2, descendants removed))   (loss of narrowing)
Costs come from the online per-(W, g) layer fit (ext34.GFit), never from the world's true cost.
Up-width timing rounds: while the 2W fit has < 4 samples, one 2W round every 25 eligible rounds; then 2 rounds
every 300 (as the narrow probe_w rounds).
Judged every 20 base rounds after a 200-round dwell, >= 100 samples:
  satc   widen  when EMA(bind) > 0.25 and EMA(dcu) / EMA(tok) < R_MAX (0.02)
         narrow when EMA(dcd - loss) > 0
  net0   widen  when EMA(fo_up - dcu) > 0
         narrow when EMA(dcd - loss) > 0
After a switch: verify (old width at 25 per 100 rounds, roll back when g_old - se > 0, rolled-back widening
blocks that width for 3000 rounds, doubling). An indicator narrowing blocks widening back for 3000 rounds
(verify can still roll it back). No steady real probes and no real-data moves.
"""
import ext31 as M31
from ext31 import bbucket, narrower, wider, mass
from ext33 import NARROW_IDLE, EMA_N, DWELL
from ext37 import SatP, W_MAX

UP_FILL_EVERY = 25


class IndW(SatP):
    def __init__(self, rule="satc", r_max=0.02, **kw):
        super().__init__(up_max=0.0, down_max=0.0, **kw)
        self.rule, self.r_max = rule, float(r_max)
        self.ist = {}
        self.up_idle = {}
        self.up_left = {}
        self.n_up_timing = 0

    def _is(self, bk):
        s = self.ist.get(bk)
        if s is None:
            s = self.ist[bk] = dict(n=0, nu=0, nd=0, tok=0.0, dcu=0.0, dcd=0.0, fo=0.0, net_up=0.0, net_dn=0.0)
        return s

    # ------------------------------------------------------------ up-width timing rounds
    def _choose_w(self, b, eligible, suml):
        base = self._persist(b)
        if eligible:
            bk = bbucket(b)
            wid = wider(base)
            if wid is not None and wid <= W_MAX:
                key = (bk, wid)
                left = self.up_left.get(key, 0)
                idle = self.up_idle.get(key, 0) + 1
                self.up_idle[key] = idle
                if left <= 0:
                    if self._layer_n(wid) < 4 and idle >= UP_FILL_EVERY:
                        left = 1
                    elif idle >= NARROW_IDLE:
                        left = 2
                if left > 0:
                    self.up_left[key] = left - 1
                    self.up_idle[key] = 0
                    self.n_up_timing += 1
                    self.elig_n[bk] = self.elig_n.get(bk, 0) + 1
                    return wid, "probe_wu"
        return super()._choose_w(b, eligible, suml)

    # ------------------------------------------------------------ indicators
    def _signals(self, pv):
        super()._signals(pv)
        plan = pv["plan"]
        b = pv["b"]
        W = plan["W"]
        if pv["snap"] is None or plan["price"] is None or plan["kind"] != "ordinary" or W != self._persist(b):
            return
        bk = bbucket(b)
        K = pv["K"]
        cs = pv["snap"]["ratios"]
        nar, wid = narrower(W), wider(W)
        if wid is not None and wid > W_MAX:
            wid = None
        fo = loss = 0.0
        for i, v in enumerate(pv["views"]):
            L = len(v.layers)
            allsc = v.scores_upto(L)
            base = mass(allsc, K)
            if wid is not None:
                exc = [x for lay in v.t.excl[:L] for x in lay]
                fo += cs[i] * (mass(sorted(allsc + exc, reverse=True), K) - base)
            if nar is not None:
                par = M31.parents_of(v.t)
                kept = M31.truncate(v, par, nar)
                loss += cs[i] * (base - mass(M31.kept_scores(v, kept, L), K))
        s = self._is(bk)
        s["n"] += 1
        st = max(1.0 / EMA_N, 1.0 / s["n"])
        s["tok"] += st * (float(sum(pv["com"])) - s["tok"])
        dcu = None if wid is None else self._dc(b, pv["suml"], pv["rows"], W, wid)
        if dcu is not None:
            s["nu"] += 1
            su = max(1.0 / EMA_N, 1.0 / s["nu"])
            s["dcu"] += su * (dcu - s["dcu"])
            s["fo"] += su * (fo - s["fo"])
            s["net_up"] += su * ((fo - dcu) - s["net_up"])
        dcd = None if nar is None else self._dc(b, pv["suml"], pv["rows"], nar, W)
        if dcd is not None:
            s["nd"] += 1
            sd = max(1.0 / EMA_N, 1.0 / s["nd"])
            s["dcd"] += sd * (dcd - s["dcd"])
            s["net_dn"] += sd * ((dcd - loss) - s["net_dn"])

    # ------------------------------------------------------------ decisions
    def _judge(self, bk, s, W):
        if s["mode"] == "verify":
            return super()._judge(bk, s, W)
        if s["since"] < DWELL:
            return
        now = self.elig_n.get(bk, 0)
        sat = self.sat_st.get(bk) or {}
        i = self._is(bk)
        wid = wider(W)
        if wid is not None and wid > W_MAX:
            wid = None
        nar = narrower(W)
        up_ok = wid is not None and s["block"].get(wid, -1) <= now and i["nu"] >= EMA_N // 2
        if up_ok:
            if self.rule == "satc":
                r_up = i["dcu"] / i["tok"] if i["tok"] > 0 else 1.0
                go = sat.get("n", 0) >= EMA_N // 2 and sat.get("bind", 0.0) > self.p_bind and r_up < self.r_max
                a_, b_ = sat.get("bind", 0.0), r_up
            else:
                go = i["net_up"] > 0.0
                a_, b_ = i["net_up"], i["tok"]
            if go:
                self._ev("ind_widen")
                self._switch(bk, W, wid, "ind_up", a_, b_)
                return
        if nar is not None and i["nd"] >= EMA_N // 2 and i["net_dn"] > 0.0:
            self._ev("ind_narrow")
            self._switch(bk, W, nar, "ind_down", i["net_dn"], i["tok"])

    def _switch(self, bk, cur, tgt, why, a, b_):
        now = self.elig_n.get(bk, 0)
        super()._switch(bk, cur, tgt, why, a, b_)
        if why == "ind_down":
            self.ps[bk]["block"][cur] = now + self.block0
        old = self.ist.get(bk)
        self.ist[bk] = None
        self.ist.pop(bk)
        if old is not None:
            self._is(bk)["tok"] = old["tok"]


ARMS = {
    "satc_g": (IndW, dict(rule="satc")),
    "net0_g": (IndW, dict(rule="net0")),
}
