"""Round-4 diagnostic: are the switch's exits in the long-prefix runs noise or real local losses?
Runs ClearTree core without a switch, records every round, and compares the 20-round window rate (what the
switch sees) with a centred 200-round rate (a proxy for the batch's true speculative rate) and with the
Target-only rate of the same b.
usage: diag4b.py <b> <seed>"""
import json, sys
import numpy as np
import ext2 as X
import ext4 as Z
import run4 as R4
from calibrated import CAL


class Rec:
    def __init__(self, inner):
        self.inner, self.rows = inner, []

    def round(self, world, slots, rng, rtt, rnd):
        com, t = self.inner.round(world, slots, rng, rtt, rnd)
        self.rows.append((len(slots), sum(com), t, tuple(r for r, _ in slots),
                          float(np.mean([world.reqs[r]["delta"] for r, _ in slots]))))
        return com, t


def main():
    b, seed = int(sys.argv[1]), int(sys.argv[2])
    w = X.World2(seed=seed, b=b, rtt=4.0, **dict(CAL, **Z.long_kw(R4.LONG_FACTOR)))
    rec = Rec(R4.core(w))
    rate = X.simulate3(w, rec)
    rows = [r for r in rec.rows if r[0] == b]
    tok = np.array([r[1] for r in rows], float); t = np.array([r[2] for r in rows])
    ar = 1000.0 * b / w.T_ar(b)
    def win(n):
        k = np.ones(n)
        return 1000.0 * np.convolve(tok, k, "valid") / np.convolve(t, k, "valid")
    w20, w200 = win(20), win(200)
    # align: centred 200 window for each 20-window end
    off = 100 - 19
    pairs = [(w20[i], w200[i - off]) for i in range(len(w20)) if 0 <= i - off < len(w200)]
    below20 = np.mean([a < 0.97 * ar for a, _ in pairs])
    below200 = np.mean([c < 0.97 * ar for _, c in pairs])
    noise = np.mean([a < 0.97 * ar and c >= ar for a, c in pairs])
    # does mean request delta of the batch explain the 200-round rate?
    dl = np.array([r[4] for r in rows])
    d200 = np.convolve(dl, np.ones(200) / 200, "valid")
    corr = float(np.corrcoef(w200, d200)[0, 1])
    sd20 = float(np.std(w20 / np.repeat(w200, 1)[:len(w20)] if len(w200) >= len(w20) else 0))
    print(json.dumps(dict(scen=f"long prefix b={b}", seed=seed, rate=round(rate, 1), ar=round(ar, 1),
                          rounds=len(rows),
                          frac_w20_below=round(float(below20), 3), frac_w200_below=round(float(below200), 3),
                          frac_w20_below_while_w200_above_ar=round(float(noise), 3),
                          w20_rel_sd=round(float(np.std([a / c for a, c in pairs])), 3),
                          w200_min_max=[round(float(w200.min()), 1), round(float(w200.max()), 1)],
                          corr_w200_vs_batch_delta=round(corr, 2))), flush=True)


if __name__ == "__main__":
    main()
