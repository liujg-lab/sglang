"""Summarize round 18 (run18.py): online update of W. usage: analyze18.py"""
import json, os
from collections import defaultdict

R = "results"
rows = [json.loads(l) for f in ("r18A.0.jsonl", "r18A.1.jsonl") if os.path.exists(os.path.join(R, f))
        for l in open(os.path.join(R, f))]
brows = [json.loads(l) for f in ("r18B.0.jsonl", "r18B.1.jsonl") if os.path.exists(os.path.join(R, f))
         for l in open(os.path.join(R, f))]
_bix = {(r["world"], r["kind"], r["b"], r["seed"]): r for r in brows}
for r in rows:
    x = _bix.get((r["world"], r["kind"], r["b"], r["seed"]))
    if x:
        r["res"].update(x["res"])
orows = [json.loads(l) for f in ("r18O.0.jsonl", "r18O.1.jsonl") if os.path.exists(os.path.join(R, f))
         for l in open(os.path.join(R, f))]
ONLINE = ["table", "replay14", "rescore", "rescore no-confirm"]
if brows:
    ONLINE += ["rescore start no-confirm", "rescore start long-trial", "rescore step no-confirm", "rescore step long-trial"]
WORLD_ORDER = ("base", "tok x3", "tok x0.3", "change")

cell = defaultdict(list)
for r in rows:
    cell[(r["world"], r["kind"], r["b"])].append(r)
orc = defaultdict(list)
for r in orows:
    orc[(r["kind"], r["b"])].append(r)


def mean(x):
    return sum(x) / len(x) if x else float("nan")


print("== Round 18: online update of W. Rates are means over seeds; % vs the hindsight best fixed W of the scenario")
print("   (K online in every variant; all variants know the true current costs). [W use %] switches / explore / trials ok")
group = defaultdict(lambda: defaultdict(list))
for wd in WORLD_ORDER:
    for kind in ("mid", "high"):
        for b in (1, 4, 8):
            rs = cell.get((wd, kind, b))
            if not rs:
                continue
            fx = {W: mean([r["res"][f"fixed {W}"]["rate"] for r in rs]) for W in (2, 4, 8)}
            bw = max(fx, key=fx.get)
            ref = fx[bw]
            parts = [f"{wd:8s} {kind:4s} b={b} n={len(rs)} | best fixed W={bw} {ref:6.1f} (2:{fx[2]:.1f} 4:{fx[4]:.1f} 8:{fx[8]:.1f})"]
            for v in ONLINE:
                if any(v not in r["res"] for r in rs):
                    continue
                rate = mean([r["res"][v]["rate"] for r in rs])
                d = 100 * (rate / ref - 1)
                group[wd][v].append(d)
                wu = defaultdict(int)
                for r in rs:
                    for k, n in r["res"][v]["w"].items():
                        wu[k] += n
                tot = sum(wu.values()) or 1
                use = ",".join(f"{k}:{100 * n / tot:.0f}" for k, n in sorted(wu.items(), key=lambda x: int(x[0])))
                extra = ""
                if v != "table":
                    sw = sum(r["res"][v]["switches"] for r in rs)
                    ex = sum(r["res"][v]["explore"] for r in rs)
                    tr = sum(r["res"][v]["trials"] for r in rs)
                    ok = sum(r["res"][v]["trial_ok"] for r in rs)
                    extra = f" {sw}/{ex}" + (f"/{ok}of{tr}" if tr else "")
                parts.append(f"{v} {d:+.1f}% [{use}]{extra}")
            if wd == "change" and orc.get((kind, b)):
                o = orc[(kind, b)]
                orate = mean([r["res"]["oracle switch"]["rate"] for r in o])
                d = 100 * (orate / ref - 1)
                group[wd]["oracle switch"].append(d)
                parts.append(f"oracle switch ({o[0]['res']['w_before']}->{o[0]['res']['w_after']}) {d:+.1f}%")
            print(" | ".join(parts))
print()
print("  group means (min) vs the hindsight best fixed W:")
for wd in WORLD_ORDER:
    if wd not in group:
        continue
    s = "; ".join(f"{v} {mean(x):+.2f}% ({min(x):+.1f})" for v, x in group[wd].items())
    print(f"   {wd:8s}: {s}")
print()
print("  recovered share of the table's loss (cells where the table loses > 1% to the best fixed W):")
for v in ONLINE[1:] + ["oracle switch"]:
    rec = []
    for wd in WORLD_ORDER:
        if v not in group[wd]:
            continue
        for t, x in zip(group[wd]["table"], group[wd][v]):
            if t < -1.0:
                rec.append((x - t) / -t)
    if rec:
        print(f"   {v}: {100 * mean(rec):.0f}% over {len(rec)} cells")
print()
print("  cost where the table is right (base world, cells where the table is within 1% of the best fixed W), vs table:")
for v in ONLINE[1:]:
    if v not in group["base"]:
        continue
    ds = [x - t for t, x in zip(group["base"]["table"], group["base"][v]) if t > -1.0]
    if ds:
        print(f"   {v}: mean {mean(ds):+.2f} pt, worst {min(ds):+.2f} pt over {len(ds)} cells")
