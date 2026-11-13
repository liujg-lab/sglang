"""Round 35: does the tight next-layer gain's loss on high acceptance b = 8 come from c_i understating the
later rank bands, and does the round-17 online f_band put that back?

usage (conda sglang env): python run35.py <proc 0|1>
  scope  settled policy (tight on). Written together with the other two arms to r35.
  loose  scope with tight off (the previous one-step increment).
  fband  scope with fband="online" (prior 5). f multiplies value next round; the tight gain is not multiplied.
Six runs: high acceptance, short prefixes, b = 8, and the same world with Draft layer cost x0.3 at round 400.
Seeds 8 and 9 short (3x), seed 8 long (10x).
"""
import json, sys, time
import ext26 as P
import run32 as R

SCOPE = ("scope", dict(cooldown=False, rob=True, rob_scope=True))
LOOSE = ("loose", dict(cooldown=False, rob=True, rob_scope=True, tight=False))
FBAND = ("fband", dict(cooldown=False, rob=True, rob_scope=True, fband="online"))


class _Tree:
    """A chain that accepts token 0, plus a frontier the tight gain can read."""

    def __init__(self):
        self.req = {"H": [0] * 40}
        self.pos = list(range(40))
        self.kids = [{0: i + 1} for i in range(39)] + [{}]
        self.scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
        self.score = {0: 0.5, 1: 0.2}
        self.frontier = [0, 1]

    def topk(self, K, by_pi):
        return set(range(K))


def check():
    es = {8: 1.0, 16: 1.4, 32: 1.5}
    plain = P.BatchAccount19(cooldown=False, rob=True, rob_scope=True)
    assert plain.fband is None and all(plain.f[K] == 1.0 for K in plain.k_options)
    got = plain._vals(0.8, es)
    assert abs(got[8] - 0.8 * 1.0) < 1e-9 and abs(got[32] - 0.8 * 1.5) < 1e-9
    plain._fband_obs([_Tree()], [0.8], [es])
    assert all(plain.f[K] == 1.0 for K in plain.k_options)

    band = P.BatchAccount19(cooldown=False, rob=True, rob_scope=True, fband="online")
    before = band._vals(1.0, es)
    assert abs(before[32] - 1.5) < 1e-9
    t = _Tree()
    g0 = band._g(t, 1.0, 8, {8: 3.0}, {8: 1.0})
    band._fband_obs([t], [1.0], [es])
    assert band.f[32] > 1.0 and band.f[16] > 1.0
    after = band._vals(1.0, es)
    assert after[32] > before[32]
    g1 = band._g(t, 1.0, 8, {8: 3.0}, {8: 1.0})
    assert g0 == g1
    band.f[32] = 5.0
    assert band._g(t, 1.0, 8, {8: 3.0}, {8: 1.0}) == g0
    loose = P.BatchAccount19(cooldown=False, rob=True, rob_scope=True, tight=False)
    assert loose.tight is False
    assert loose._g(t, 1.0, 8, {8: 3.0}, {8: 1.0}) == 2.0


def jobs():
    out = []
    pols = [SCOPE, LOOSE, FBAND]
    for (s, mu) in R.RUNS:
        out.append((("S", ("high", "high", "short", 8, None), s, mu, None), pols))
        out.append((("S", ("chg x0.3", "high", "short", 8, None), s, mu, 0.3), pols))
    assert len(out) == 6
    return out


def one(pt, name, kind, mix, b, phase, seed, mult, cm, kw):
    made = []
    orig = P.BatchAccount19.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        made.append(self)

    P.BatchAccount19.__init__ = init
    try:
        out = R.one(pt, name, kind, mix, b, phase, seed, mult, cm, kw)
    finally:
        P.BatchAccount19.__init__ = orig
    p = made[-1]
    out["depth"] = p.sum_depth / max(p.n_depth, 1)
    out["f"] = {str(k): round(float(p.f[k]), 4) for k in p.k_options}
    return out


def main():
    check()
    print("check ok", file=sys.stderr)
    proc = int(sys.argv[1])
    for i, ((pt, (name, kind, mix, b, phase), seed, mult, cm), pols) in enumerate(jobs()):
        if i % 2 != proc:
            continue
        t0 = time.time()
        res = {lab: one(pt, name, kind, mix, b, phase, seed, mult, cm, kw) for lab, kw in pols}
        print(json.dumps(dict(part=pt, run="r35", world=name, kind=kind, mix=mix, b=b, phase=phase, seed=seed,
                              req_mult=mult, cmult=cm, res=res, sec=round(time.time() - t0, 1),
                              python=sys.version.split()[0])), flush=True)


if __name__ == "__main__":
    main()
