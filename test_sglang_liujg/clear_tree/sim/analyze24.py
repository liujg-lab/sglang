"""Summarize the round-24 diagnosis (diag24.py). usage: python analyze24.py

Differences are mean per-round Y differences (Y = tokens - Theta * model time) in % of the built width's committed
tokens per round, +- 1 se. "need" = rounds until the round-19b confidence sequence (plug-in normal mixture, rho = 50 sd^2,
alpha 5% split over 2 alternatives) would put the replay estimate above eps = 1%, if its mean and sd stay as measured
(inf: the mean is not above eps)."""
import json, math, os

rows = []
for f in ("results/r24.0.jsonl", "results/r24.1.jsonl"):
    if os.path.exists(f):
        rows += [json.loads(l) for l in open(f)]
order = {("ext20", "high"): 0, ("ext20", "mid"): 1, ("ext20", "het"): 2, ("ext15", "high"): 3, ("ext15", "mid"): 4}
rows.sort(key=lambda r: (order[(r["setting"], r["kind"])], -r["b"]))


def need(mu, se, n, eps=1.0, a=0.025):
    if mu <= eps:
        return math.inf
    sd = se * math.sqrt(n)
    for k in range(10, 100000):
        r = math.sqrt((k + 50) * sd * sd * (math.log((k + 50) / 50) + 2 * math.log(2 / a))) / k
        if mu - r > eps:
            return k
    return math.inf


print("== Round 24: is the paired replay biased against narrower W? (seed 8; one fixed built width per run)")
print("   replay est vs TRUE (own trees) in % of tokens/round; bias = est - TRUE; components of TRUE - est on the Wn side:")
print("   narrowing (ext12.narrowed vs a real Wn tree), exits/stop (own rule vs the built schedule), depth (no cap)")
summary = []
for r in rows:
    res = r["res"]
    fx = {W: res[f"fixed {W}"]["rate"] for W in (2, 4, 8)}
    bw = max(fx, key=fx.get)
    head = (f"{r['setting']} {r['kind']:4s} {r['mix']:5s} b={r['b']} | fixed rate 2:{fx[2]:.1f} 4:{fx[4]:.1f} "
            f"8:{fx[8]:.1f} (best {bw}; 2 vs 4 {100 * (fx[2] / fx[4] - 1):+.1f}%)")
    print(head)
    for Wb in (4, 8):
        s = res[f"built {Wb}"]
        d, n = s["diff_pct_tok"], s["n"]
        Wns = (2,) if Wb == 4 else (4, 2)
        for Wn in Wns:
            p = f"{Wb}->{Wn}"
            est, tru = d[f"{p} replay est"], d[f"{p} TRUE"]
            extra = ""
            for lab in ("deep-round est", "cut-rule est"):
                if f"{p} {lab}" in d:
                    extra = f" | {lab.split(' ')[0]} est {d[f'{p} {lab}'][0]:+.2f}%"
            comp = "; ".join(f"{c} {d[f'{p}   {c} ' + k][0]:+.2f}" for c, k in
                             (("narrowing", "R1-R0"), ("exits/stop", "R2-R1"), ("depth", "R3-R2")))
            print(f"   built {Wb}, {Wn} vs {Wb}: est {est[0]:+6.2f}% (+-{est[1]:.2f}) TRUE {tru[0]:+6.2f}% (+-{tru[1]:.2f}) "
                  f"bias {est[0] - tru[0]:+.2f}{extra} | {comp} | deeper rounds {100 * s['deeper_share'][str(Wn)]:.0f}%")
        if Wb == 8:
            est, tru = d["2 vs 4 (both from 8) replay est"], d["2 vs 4 TRUE (own trees)"]
            extra = ""
            for lab in ("deep-round est", "cut-rule est"):
                if f"2 vs 4 (both from 8) {lab}" in d:
                    extra = f" | {lab.split(' ')[0]} est {d[f'2 vs 4 (both from 8) {lab}'][0]:+.2f}%"
            nd = need(est[0], est[1], n)
            print(f"   built 8, 2 vs 4 (both replayed): est {est[0]:+6.2f}% (+-{est[1]:.2f}) TRUE {tru[0]:+6.2f}% "
                  f"(+-{tru[1]:.2f}) bias {est[0] - tru[0]:+.2f}{extra} | per-round sd {est[1] * math.sqrt(n):.1f}% | "
                  f"need {nd} rounds")
            summary.append((r["setting"], r["kind"], r["b"], est[0], tru[0], nd, 100 * (fx[2] / fx[4] - 1)))
        s4 = res["built 4"]
        if Wb == 4:
            e = s4["diff_pct_tok"]["4->2 replay est"]
            summary_42 = (e[0], s4["diff_pct_tok"]["4->2 TRUE"][0], need(e[0], e[1], s4["n"]))
    print(f"   built 4: 2 vs 4 need {summary_42[2]} rounds;  k mismatch (replay K vs built K) "
          f"{res['built 4']['k_mismatch']}/{res['built 4']['n']}, {res['built 8']['k_mismatch']}/{res['built 8']['n']}")
print()
print("  2 vs 4 from W=8 trees (what a window built at 8 compares): est / TRUE / need / long-run rate 2 vs 4")
for st, k, b, e, t, nd, rr in summary:
    print(f"   {st} {k:4s} b={b}: est {e:+.2f}%  TRUE {t:+.2f}%  bias {e - t:+.2f}  need {nd}  rate {rr:+.1f}%")
