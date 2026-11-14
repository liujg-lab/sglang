"""Read r38*.jsonl and apply the pre-registered lines in results.md 7.41.

Does not write r38_param.json or r38_adopt.json. Those are written once, by hand, from this printout.
"""
import json, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def load(tag):
    rows = []
    for i in (0, 1):
        path = RES / f"{tag}.{i}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def se(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return var ** 0.5 / n ** 0.5


def pct(num, den):
    return (num / den - 1.0) * 100.0


def k32(res, rnd=500):
    for pair in res.get("k_snap") or []:
        if pair[0] == rnd:
            return int(pair[1].get("32", 0))
    return None


def show_p():
    rows = load("r38p")
    print(f"r38p rows {len(rows)}")
    acc = {2: [], 6: [], 20: []}
    for r in rows:
        base = r["res"]["n2"]["rate"]
        bits = []
        for n0 in (2, 6, 20):
            rel = pct(r["res"][f"n{n0}"]["rate"], base)
            acc[n0].append(rel)
            bits.append(f"n{n0}={rel:+.4f}")
        print(r["world"], "b", r["b"], " ".join(bits))
    means = {n0: mean(v) for n0, v in acc.items()}
    top = max(means.values())
    pick = min(n0 for n0 in (2, 6, 20) if top - means[n0] < 0.10)
    print("means", {k: round(v, 4) for k, v in means.items()}, "pick", pick)
    return pick, means


def _bias_gap(rows, W, Wn):
    rt = rm = st = sm = n = 0
    for row in rows:
        if row["W"] == W and row["Wn"] == Wn:
            rt += row["rep_tok"]
            rm += row["rep_mt"]
            st += row["sh_tok"]
            sm += row["sh_mt"]
            n += row["n"]
    if n == 0 or rm <= 0 or sm <= 0:
        return None, n
    return pct(rt / rm, st / sm), n


def show_b():
    rows = load("r38b")
    print(f"r38b rows {len(rows)}")
    for r in rows:
        bias = r["res"]["scope"].get("bias") or []
        g4, n4 = _bias_gap(bias, 8, 4)
        g2, n2 = _bias_gap(bias, 8, 2)
        rates = {lab: r["res"][lab]["rate"] for lab in ("f2", "f4", "f8")}
        best = max(rates, key=rates.get)
        scope = r["res"]["scope"]["rate"]
        final = r["res"]["scope"].get("final", {}).get("1")
        print(r["world"], "replay8to4", None if g4 is None else round(g4, 3), "n", n4,
              "replay8to2", None if g2 is None else round(g2, 3), "n", n2,
              "final", final, "best", best, "gap", round(pct(scope, rates[best]), 3),
              "rates", {k: round(v, 3) for k, v in rates.items()})


def _pool_w(rows):
    free = defaultdict(lambda: [0.0, 0.0, 0, 0.0, 0.0, 0])
    sh = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])
    for r in rows:
        for row in r["res"]["scope"].get("ex") or []:
            st = free[row["b"]]
            st[0] += row["e_tok"]
            st[1] += row["e_mt"]
            st[2] += row["e_n"]
            st[3] += row["n_tok"]
            st[4] += row["n_time"]
            st[5] += row["n_n"]
        for row in r["res"]["scope"].get("bias") or []:
            if row["W"] <= row["Wn"]:
                continue
            st = sh[row["b"]]
            st[0] += row["rep_tok"]
            st[1] += row["rep_mt"]
            st[2] += row["sh_tok"]
            st[3] += row["sh_mt"]
            st[4] += row["n"]
    return free, sh


def show_w():
    rows = load("r38w")
    print(f"r38w rows {len(rows)}")
    free, sh = _pool_w(rows)
    ok = True
    judged = 0
    for b in sorted(set(free) | set(sh)):
        e = free[b]
        s = sh[b]
        g_free = pct(e[0] / e[1], e[3] / e[4]) if e[1] > 0 and e[4] > 0 else None
        g_sh = pct(s[0] / s[1], s[2] / s[3]) if s[1] > 0 and s[3] > 0 else None
        use = e[2] >= 30 and e[5] >= 30 and s[4] >= 30 and g_free is not None and g_sh is not None
        gap = None if g_free is None or g_sh is None else g_free - g_sh
        if use:
            judged += 1
            if abs(gap) > 2.0:
                ok = False
        print("b", b, "g_free", None if g_free is None else round(g_free, 3),
              "g_sh", None if g_sh is None else round(g_sh, 3),
              "gap_pp", None if gap is None else round(gap, 3),
              "e_n", e[2], "n_n", e[5], "sh_n", s[4], "judge", use)
    print("r38w pass" if judged and ok else "r38w fail", "buckets", judged)
    return bool(judged) and ok


def show_g():
    rows = load("r38g")
    print(f"r38g rows {len(rows)}")
    hits = []
    for r in rows:
        rates = {lab: r["res"][lab]["rate"] for lab in ("f2", "f4", "f8")}
        gap = pct(rates["f8"], rates["f4"])
        best = max(rates, key=rates.get)
        flag = gap > 3.0
        if flag:
            hits.append((r["b"], r["lmult"], round(gap, 3)))
        print("b", r["b"], "lmult", r["lmult"], "gap8vs4", round(gap, 3), "best", best,
              {k: round(v, 3) for k, v in rates.items()})
    print("wider-best", hits if hits else "none")
    return hits


def _arm_rows(tags, arm):
    rows = []
    for tag in tags:
        rows.extend(load(tag))
    out = []
    for r in rows:
        if arm not in r["res"] or "scope" not in r["res"]:
            continue
        out.append((r, pct(r["res"][arm]["rate"], r["res"]["scope"]["rate"])))
    return out


def _judge(tags, arm):
    paired = _arm_rows(tags, arm)
    xs = [p for _, p in paired]
    worlds = defaultdict(list)
    for r, p in paired:
        worlds[(r["world"], r["kind"], r["b"])].append(p)
    wmean = {k: mean(v) for k, v in worlds.items()}
    worst_k = min(wmean, key=wmean.get) if wmean else None
    het = []
    k_ok = True
    for r, _p in paired:
        if r["world"] == "het" and r["kind"] == "het" and r["b"] == 4:
            a = k32(r["res"][arm])
            s = k32(r["res"]["scope"])
            het.append((r["seed"], r["req_mult"], a, s))
            if a is None or s is None or a > 0.5 * s:
                k_ok = False
    m = mean(xs)
    if m <= 0.10:
        verdict = "reject"
    elif m >= 0.30 and wmean and min(wmean.values()) >= -0.30 and k_ok and het:
        verdict = "adopt"
    else:
        verdict = "inconclusive"
    print(arm, "n", len(xs), "mean", round(m, 4), "se", round(se(xs), 4),
          "min", round(min(xs), 4) if xs else None, "max", round(max(xs), 4) if xs else None,
          "worst_world", None if worst_k is None else (worst_k, round(wmean[worst_k], 4)),
          "k32", het, "k_ok", k_ok, "verdict", verdict)
    events = []
    for r, _p in paired:
        ev = r["res"][arm]
        events.append((ev.get("cost_events"), ev.get("checks"), r["res"]["scope"].get("cost_events"),
                       r["res"]["scope"].get("checks")))
    if events:
        print(arm, "cost_events", sum(e[0] or 0 for e in events), "checks", sum(e[1] or 0 for e in events),
              "scope_events", sum(e[2] or 0 for e in events), "scope_checks", sum(e[3] or 0 for e in events))
    return verdict, m


def show_eval(extra=False):
    tags = ["r38e"] + (["r38s"] if extra else [])
    v_nn, m_nn = _judge(tags, "vnn")
    v_pr, m_pr = _judge(tags, "vprior")
    _judge(tags, "both")
    adopted = None
    if v_nn == "adopt" and v_pr == "adopt":
        if abs(m_nn - m_pr) < 0.10:
            adopted = "vprior"
        else:
            adopted = "vnn" if m_nn > m_pr else "vprior"
    elif v_nn == "adopt":
        adopted = "vnn"
    elif v_pr == "adopt":
        adopted = "vprior"
    print("adopted", adopted)
    return adopted


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("p", "all"):
        show_p()
    if part in ("b", "all"):
        show_b()
    if part in ("w", "all"):
        show_w()
    if part in ("g", "all"):
        show_g()
    if part in ("eval", "all"):
        show_eval(False)
    if part == "seed16":
        show_eval(True)


if __name__ == "__main__":
    main()
