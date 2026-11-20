"""Round 46: W indicator screening. usage: python analyze46.py [a|v]   (a = all rounds, v = value-rule rounds)

Labels from the same jobs' fixed-W runs: up(W) = rate(2W)/rate(W) - 1, down(W) = rate(W/2)/rate(W) - 1.
1. AUC of each indicator for sign(up) and sign(down); 2. decision value of rules (share of the hindsight-available
gain captured, losses counted); 3. hill-climb on the indicator values from the start width (8 at b=1, else 4):
final W and loss vs best fixed W. Synthetic world; not a measurement.
"""
import glob, json, math, sys
import os
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
from collections import defaultdict

import numpy as np

SET = sys.argv[1] if len(sys.argv) > 1 else "a"
PRE = sys.argv[2] if len(sys.argv) > 2 else "r46"
WS = (2, 4, 8, 16)
jobs = []
for f in sorted(glob.glob(os.path.join(RES, PRE + ".*.jsonl"))):
    for l in open(f):
        r = json.loads(l)
        if r["world"] == "bvar":
            continue
        jobs.append(r)
print(f"{len(jobs)} jobs (b varies excluded), indicator set '{SET}'")


def bbucket(b):
    return 1 if b == 1 else (2 if b <= 4 else 5)


def feats(a, W):
    n, R = max(a["trees"], 1), max(a["rounds"], 1)
    need = {int(k): v for k, v in a["need"].items()}
    tot = sum(need.values()) or 1
    cum, q90 = 0, W + 1
    for k in sorted(need):
        cum += need[k]
        if cum >= 0.9 * tot:
            q90 = k
            break
    st = lambda k: a.get("stop_" + k, 0.0) / n
    f = dict(
        bind=a["bind"] / n, half=a["half_r"] / R, util=a["used"] / max(a["nodes"], 1),
        rho=a["rho"] / max(a["rho_n"], 1), wk=st("wk"), wcut=st("wcut"), wall=st("wk") + st("wcut"),
        depth=st("depth"), kmiss=st("kmiss"), vocab=st("vocab"), rank_hi=a["rank_hi"] / max(a["acc"], 1),
        acc=a["acc"] / n, q90=q90 / W, need_hi=sum(v for k, v in need.items() if k > W // 2) / tot,
        need_full=sum(v for k, v in need.items() if k >= W) / tot,
        rup=a["rup"] / a["rup_n"] if a.get("rup_n") else float("nan"),
        rdn=a["rdn"] / a["rdn_n"] if a.get("rdn_n") else float("nan"),
        fo_up=a["fo_up"] / R, loss_dn=a["loss_dn"] / R)
    f["net_up"] = f["fo_up"] - f["rup"]
    f["net_dn"] = f["rdn"] - f["loss_dn"]
    f["wk_c"] = f["wk"] - f["rup"]
    return f


rows = []           # one per (job, W)
J = []
for j, r in enumerate(jobs):
    bk = bbucket(r["b"])
    rate = {w: r["res"][f"ind{w}"]["rate"] for w in WS}
    F = {}
    for w in WS:
        a = r["res"][f"ind{w}"]["ind"].get(f"{bk}{SET}")
        if a is None or a["rounds"] < 20:
            continue
        F[w] = feats(a, w)
    grp = f"b={r['b']}" + (" mixed" if r["mix"] == "mixed" else "")
    J.append(dict(r=r, rate=rate, F=F, grp=grp, b=r["b"]))
    for w in F:
        up = rate[2 * w] / rate[w] - 1 if 2 * w in rate else None
        dn = rate[w // 2] / rate[w] - 1 if w // 2 in rate else None
        rows.append(dict(j=j, W=w, grp=grp, b=r["b"], up=up, dn=dn, **F[w]))
NAMES = [k for k in rows[0] if k not in ("j", "W", "grp", "b", "up", "dn")]


def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, bool)
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    p, n = x[y], x[~y]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    s = 0.0
    for v in p:
        s += np.sum(v > n) + 0.5 * np.sum(v == n)
    return s / (len(p) * len(n))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


for lab in ("up", "dn"):
    R_ = [r for r in rows if r[lab] is not None]
    y = [r[lab] > 0 for r in R_]
    print(f"\n== {lab}: {len(R_)} (job, W) points, {sum(y)} where one step {'wider' if lab == 'up' else 'narrower'} is better ==")
    print(f"  {'indicator':10s} {'AUC':>5s} {'rho_s':>6s}   AUC by group (b=1 / b=4 / b=8)")
    res = []
    for k in NAMES:
        x = [r[k] for r in R_]
        A = auc(x, y)
        res.append((abs(A - 0.5), k, A, spearman(x, [r[lab] for r in R_])))
    for _, k, A, s in sorted(res, reverse=True):
        by = []
        for g in ("b=1", "b=4", "b=8"):
            Rg = [r for r in R_ if r["b"] == int(g[2:])]
            by.append(auc([r[k] for r in Rg], [r[lab] > 0 for r in Rg]))
        print(f"  {k:10s} {A:5.2f} {s:+6.2f}   " + " / ".join(f"{v:4.2f}" for v in by))


# ------------------------------------------------------------ decision value of rules
def value(rule_up, rule_dn):
    """Mean realized gain per decision point and the oracle mean, in %, over all (job, W) points."""
    got = ora = 0.0
    n = 0
    for r in rows:
        u = r["up"] if r["up"] is not None else None
        d = r["dn"] if r["dn"] is not None else None
        o = max([0.0] + [x for x in (u, d) if x is not None])
        g = 0.0
        if u is not None and rule_up(r):
            g = u
        elif d is not None and rule_dn(r):
            g = d
        got += g
        ora += o
        n += 1
    return 100 * got / n, 100 * ora / n


def climb(jb, rule_up, rule_dn):
    w0 = 8 if jb["b"] == 1 else 4
    w, seen = w0, [w0]
    for _ in range(6):
        F = jb["F"].get(w)
        if F is None:
            break
        r = dict(F, W=w)
        if 2 * w in jb["rate"] and rule_up(r):
            w = 2 * w
        elif w // 2 in jb["rate"] and rule_dn(r):
            w = w // 2
        else:
            break
        if w in seen:
            break
        seen.append(w)
    best = max(jb["rate"].values())
    return w, 100 * (jb["rate"][w] / best - 1)


NEVER = lambda r: False
RULES = {
    "start width (no change)": (NEVER, NEVER),
    "sat: bind>.25 / half>.9": (lambda r: r["bind"] > 0.25, lambda r: r["half"] > 0.9),
    "sat, up only": (lambda r: r["bind"] > 0.25, NEVER),
    "satc: bind>.25 & rup<.02": (lambda r: r["bind"] > 0.25 and r["rup"] < 0.02, lambda r: r["half"] > 0.9),
    "satc: bind>.25 & rup<.03": (lambda r: r["bind"] > 0.25 and r["rup"] < 0.03, lambda r: r["half"] > 0.9),
    "wk>.05": (lambda r: r["wk"] > 0.05, NEVER),
    "wk>.05 & rup<.03": (lambda r: r["wk"] > 0.05 and r["rup"] < 0.03, NEVER),
    "wk>rup (miss vs cost)": (lambda r: r["wk"] > r["rup"], NEVER),
    "q90>W/2 up if need_full>.5": (lambda r: r["need_full"] > 0.5, lambda r: r["q90"] <= 0.5),
    "net (first-order, true cost)": (lambda r: r["net_up"] > 0, lambda r: r["net_dn"] > 0),
    "util<.3 -> down, bind up": (lambda r: r["bind"] > 0.25, lambda r: r["util"] < 0.3),
}
print("\n== rules: decision value (mean realized gain per (job, W) point vs oracle) and hill-climb from the start width ==")
print(f"  {'rule':32s} {'value':>6s} {'oracle':>6s}   climb: mean loss vs best fixed, worst, final==best, by group")
for name, (ru, rd) in RULES.items():
    v, o = value(ru, rd)
    cl = [climb(jb, ru, rd) for jb in J]
    loss = [c[1] for c in cl]
    eq = sum(1 for jb, c in zip(J, cl) if jb["rate"][c[0]] == max(jb["rate"].values()))
    by = defaultdict(list)
    for jb, c in zip(J, cl):
        by[jb["grp"]].append(c[1])
    print(f"  {name:32s} {v:+6.2f} {o:+6.2f}   {sum(loss) / len(loss):+6.2f}  {min(loss):+6.2f}  {eq:2d}/{len(J)}   " +
          "  ".join(f"{g} {sum(x) / len(x):+.2f}" for g, x in sorted(by.items())))


# ------------------------------------------------------------ logistic, leave one job out
def fit_logit(X, y, lam=1.0):
    X = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + lam * np.r_[0, w[1:]]
        Hm = (X * (p * (1 - p))[:, None]).T @ X + lam * np.diag(np.r_[0, np.ones(len(w) - 1)])
        w -= np.linalg.pinv(Hm) @ g
    return w


def lojo(lab, keys):
    R_ = [r for r in rows if r[lab] is not None and not any(math.isnan(r[k]) for k in keys)]
    X = np.array([[r[k] for k in keys] for r in R_], float)
    y = np.array([r[lab] > 0 for r in R_], float)
    js = np.array([r["j"] for r in R_])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    pred = np.zeros(len(R_))
    for j in set(js):
        tr = js != j
        w = fit_logit(Xs[tr], y[tr])
        pred[~tr] = 1 / (1 + np.exp(-(np.hstack([np.ones(((~tr).sum(), 1)), Xs[~tr]]) @ w)))
    gains = np.array([r[lab] for r in R_])
    got = float(np.sum(gains * (pred > 0.5)))
    ora = float(np.sum(np.maximum(gains, 0)))
    return auc(pred, y), 100 * got / len(R_), 100 * ora / len(R_), len(R_)


print("\n== logistic, leave one job out: AUC, realized gain per point if acting at p>.5, oracle ==")
for lab, sets in (("up", [["bind"], ["bind", "rup"], ["wk"], ["wk", "rup"], ["bind", "wk", "rup"], ["net_up"],
                          ["need_full", "rup"], ["bind", "wk", "rup", "rho", "util"]]),
                  ("dn", [["half"], ["util"], ["loss_dn", "rdn"], ["net_dn"], ["need_hi"], ["rank_hi"],
                          ["half", "util", "rdn"], ["need_hi", "rank_hi", "rdn", "util"]])):
    for keys in sets:
        A, g, o, n = lojo(lab, keys)
        print(f"  {lab} {'+'.join(keys):28s} AUC {A:4.2f}  value {g:+5.2f}%  oracle {o:+5.2f}%  n {n}")

print("\n== indicator means by group and W (bind / wk / rup / util / half / rank_hi), best fixed W per job ==")
agg = defaultdict(lambda: defaultdict(list))
for r in rows:
    for k in ("bind", "wk", "rup", "util", "half", "rank_hi", "up", "dn"):
        if r[k] is not None and not (isinstance(r[k], float) and math.isnan(r[k])):
            agg[(r["grp"], r["W"])][k].append(r[k])
for key in sorted(agg):
    m = {k: sum(v) / len(v) for k, v in agg[key].items()}
    print(f"  {key[0]:10s} W={key[1]:2d}  bind {m['bind']:.2f}  wk {m['wk']:.3f}  rup {m.get('rup', float('nan')):.3f}  "
          f"util {m['util']:.2f}  half {m['half']:.2f}  rank_hi {m['rank_hi']:.2f}  "
          f"up {100 * m['up']:+5.2f}%" if 'up' in m else f"  {key[0]:10s} W={key[1]:2d}  (W=16)  bind {m['bind']:.2f}  util {m['util']:.2f}  dn {100 * m.get('dn', 0):+5.2f}%")
