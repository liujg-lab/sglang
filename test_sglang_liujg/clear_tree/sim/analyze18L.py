"""Round 18 parts L / L2: 10x longer runs vs the short runs of the same seed (8). usage: analyze18L.py
(L2 = rescore with exploration records kept in a separate buffer; long runs only)

Per scenario and variant: rate vs the table (and, in tok x0.3, vs the best fixed W), the share of rounds run at a width
above the table's W (exploration, trials and switches to wider widths), and the switch count, short (r18A/B, 3x the
usual requests) next to long (r18L, 30x)."""
import json, os

R = "results"


def load(names):
    out = []
    for f in names:
        p = os.path.join(R, f)
        if os.path.exists(p):
            out += [json.loads(l) for l in open(p)]
    return out


short = {}
for r in load(["r18A.0.jsonl", "r18A.1.jsonl", "r18B.0.jsonl", "r18B.1.jsonl"]):
    if r["seed"] != 8:
        continue
    short.setdefault((r["world"], r["kind"], r["b"]), {}).update(r["res"])
long_ = {}
for r in load(["r18L.0.jsonl", "r18L.1.jsonl", "r18L2.0.jsonl", "r18L2.1.jsonl"]):
    long_.setdefault((r["world"], r["kind"], r["b"]), {})[r["variant"]] = r["res"]

TABLE_W = {4: lambda kind: 4, 8: lambda kind: 2 if kind == "high" else 4}
VARS = ("replay14", "rescore no-confirm", "rescore step no-confirm", "rescore step no-confirm keep-wide")


def wide_share(res, wt):
    tot = sum(res["w"].values()) or 1
    return 100.0 * sum(n for w, n in res["w"].items() if int(w) > wt) / tot


print("== Round 18 part L: short (3x requests) vs long (30x), seed 8. % vs table; [share of rounds above the table W]; switches")
summ = {}
for key in sorted(long_, key=lambda k: (k[0] != "base", k[1], k[2])):
    wd, kind, b = key
    L, S = long_[key], short.get(key, {})
    if "table" not in L:
        continue
    wt = TABLE_W[b](kind) if wd in ("base", "tok x0.3") else None
    head = f"{wd:8s} {kind:4s} b={b} | spec rounds short {S.get('table', {}).get('spec_rounds', '-')} long {L['table']['spec_rounds']}"
    fx = [v for v in L if v.startswith("fixed")]
    if fx:
        f = fx[0]
        head += f" | {f}: short {100 * (S[f]['rate'] / S['table']['rate'] - 1):+.2f}%, long {100 * (L[f]['rate'] / L['table']['rate'] - 1):+.2f}% vs table"
    print(head)
    for v in VARS:
        if v not in L:
            continue
        line = f"   {v:24s}"
        for tag, D in (("short", S), ("long", L)):
            if v not in D or "table" not in D:
                line += f" | {tag} -"
                continue
            d = 100 * (D[v]["rate"] / D["table"]["rate"] - 1)
            ws = wide_share(D[v], wt)
            line += f" | {tag} {d:+.2f}% [{ws:.1f}%] sw {D[v]['switches']}"
            summ.setdefault((wd, v, tag), []).append(d)
            if fx and fx[0] in D:
                summ.setdefault((wd, v, tag + " vs best"), []).append(100 * (D[v]["rate"] / D[fx[0]]["rate"] - 1))
        print(line)
print()
print("  means over the four scenarios of each world:")
for wd in ("base", "tok x0.3"):
    for v in VARS:
        if not any((wd, v, t) in summ for t in ("short", "long")):
            continue
        parts = []
        for tag in ("short", "long", "short vs best", "long vs best"):
            x = summ.get((wd, v, tag))
            if x:
                parts.append(f"{tag} {sum(x) / len(x):+.2f}%")
        print(f"   {wd:8s} {v:24s} " + "; ".join(parts))
