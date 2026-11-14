"""Apply the pre-registered lines in results.md 7.42. Does not write r39_adopt.json.

usage: python analyze39.py [eval|seed20|guard]
  eval    r39e, 60 runs. Prints each arm's verdict.
  seed20  r39e plus r39s. An arm already at or below +0.10% on the 60 stays rejected.
          An inconclusive arm is rejudged on the 75. An arm already adopted on the 60 stays adopted.
  guard   r39g, recorded only.
"""
import json, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
ARMS = ("b3", "b4")


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


def bkey(b):
    return 1 if b <= 1 else (4 if b <= 5 else 8)


def k32(res, rnd=500):
    for pair in res.get("k_snap") or []:
        if pair[0] == rnd:
            return int(pair[1].get("32", 0))
    return None


def first_to(log, bucket, new):
    for s in log:
        if bkey(s[1]) == bucket and s[3] == new and s[2] != new:
            return s[0]
    return None


def left_after(log, bucket, new):
    arrived = False
    for s in log:
        if bkey(s[1]) != bucket:
            continue
        if s[3] == new and s[2] != new:
            arrived = True
        elif arrived and s[2] == new and s[3] != new:
            return True
    return False


def widened(res):
    for s in res["switch_log"]:
        if bkey(s[1]) != 1 and s[3] == 8 and s[3] > s[2]:
            return True
    return False


def paired(rows, arm):
    out = []
    for r in rows:
        if arm not in r["res"] or "scope" not in r["res"]:
            continue
        out.append((r, pct(r["res"][arm]["rate"], r["res"]["scope"]["rate"])))
    return out


def phased_ok(rows, arm):
    """Every phased b=8 run ends at W=2 on bucket 8, does not leave it, and is not later than scope."""
    runs = [r for r in rows if r["world"] == "phased" and r["b"] == 8]
    detail = []
    ok = bool(runs)
    for r in runs:
        a, s = r["res"][arm], r["res"]["scope"]
        end = a["final"].get("8", a["final"].get(8))
        at, scope_at = first_to(a["switch_log"], 8, 2), first_to(s["switch_log"], 8, 2)
        left = left_after(a["switch_log"], 8, 2)
        full = a["switches"] <= len(a["switch_log"]) and s["switches"] <= len(s["switch_log"])
        timing = scope_at is not None and at is not None and at <= scope_at
        good = end == 2 and not left and timing and full
        ok = ok and good
        detail.append((r["seed"], r["req_mult"], end, at, scope_at, left, full, good))
    return ok, detail


def judge(rows, arm):
    got = paired(rows, arm)
    xs = [p for _, p in got]
    worlds = defaultdict(list)
    for r, p in got:
        worlds[(r["world"], r["kind"], r["b"])].append(p)
    wmean = {k: mean(v) for k, v in worlds.items()}
    worst_k = min(wmean, key=wmean.get) if wmean else None
    worst = wmean[worst_k] if worst_k is not None else float("nan")
    m = mean(xs)
    ph_ok, ph = phased_ok(rows, arm)
    if m <= 0.10:
        verdict = "reject"
    elif m >= 0.30 and wmean and worst >= -0.30 and ph_ok:
        verdict = "adopt"
    else:
        verdict = "inconclusive"
    below = [(r, p) for r, p in got if p < -2.0]
    wide = sum(1 for r, _p in got if widened(r["res"][arm]))
    wide_s = sum(1 for r, _p in got if widened(r["res"]["scope"]))
    het = []
    for r, _p in got:
        if r["world"] == "het" and r["kind"] == "het" and r["b"] == 4:
            het.append((r["seed"], r["req_mult"], k32(r["res"][arm]), k32(r["res"]["scope"])))
    ev = sum((r["res"][arm].get("cost_events") or 0) for r, _p in got)
    ck = sum((r["res"][arm].get("checks") or 0) for r, _p in got)
    evs = sum((r["res"]["scope"].get("cost_events") or 0) for r, _p in got)
    cks = sum((r["res"]["scope"].get("checks") or 0) for r, _p in got)
    neg = sum(1 for p in xs if p < -1e-9)
    print(arm, "n", len(xs), "mean", round(m, 4), "se", round(se(xs), 4),
          "min", None if not xs else round(min(xs), 4),
          "max", None if not xs else round(max(xs), 4),
          "neg", neg, "below-2", len(below),
          "worst", None if worst_k is None else (worst_k, round(worst, 4)),
          "phased_ok", ph_ok, "verdict", verdict)
    print(" ", arm, "phased", ph)
    print(" ", arm, "wide8", wide, "scope_wide8", wide_s, "k32", het,
          "events", ev, "checks", ck, "scope_events", evs, "scope_checks", cks)
    if below:
        for r, p in below:
            print(" ", arm, "below-2", r["world"], r["kind"], "b", r["b"], "s", r["seed"],
                  "x", r["req_mult"], round(p, 4))
    for key in sorted(wmean):
        print(" ", arm, "world", key, round(wmean[key], 4), "n", len(worlds[key]))
    return verdict, m


def choose(verdicts, means):
    adopted = [a for a in ARMS if verdicts[a] == "adopt"]
    if len(adopted) == 2 and abs(means["b3"] - means["b4"]) < 0.10:
        return "b3"
    if len(adopted) == 2:
        return "b3" if means["b3"] >= means["b4"] else "b4"
    return adopted[0] if adopted else None


def show_eval(tags, expect):
    rows = []
    for tag in tags:
        rows.extend(load(tag))
    print("rows", len(rows), "expect", expect)
    if len(rows) != expect:
        print("incomplete")
        return None
    verdicts, means = {}, {}
    for arm in ARMS:
        verdicts[arm], means[arm] = judge(rows, arm)
    print("adopted", choose(verdicts, means))
    return verdicts, means


def show_seed20():
    base = show_eval(["r39e"], 60)
    if base is None:
        return
    verdicts, means = base
    extra = load("r39s")
    print("seed20 rows", len(extra))
    if len(extra) != 15:
        print("seed20 incomplete")
        return
    final_v, final_m = {}, {}
    both = load("r39e") + extra
    for arm in ARMS:
        if verdicts[arm] == "reject":
            print(arm, "stays reject", round(means[arm], 4))
            final_v[arm], final_m[arm] = "reject", means[arm]
        elif verdicts[arm] == "adopt":
            print(arm, "stays adopt on the 60", round(means[arm], 4))
            final_v[arm], final_m[arm] = "adopt", means[arm]
        else:
            print(arm, "rejudge on 75")
            final_v[arm], final_m[arm] = judge(both, arm)
    print("adopted", choose(final_v, final_m))


def show_guard():
    rows = load("r39g")
    print("guard rows", len(rows), "expect 28")
    if not rows:
        return
    arms = [a for a in rows[0]["res"] if a != "scope"]
    for arm in arms:
        judge(rows, arm)


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if part == "eval":
        show_eval(["r39e"], 60)
    elif part == "seed20":
        show_seed20()
    elif part == "guard":
        show_guard()
    else:
        raise SystemExit("usage: analyze39.py [eval|seed20|guard]")


if __name__ == "__main__":
    main()
