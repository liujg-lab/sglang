"""Round 9 follow-up: batch accounting with width 8 at b=1 (the b=1 optimum is W=8 in the low and mid workloads).
Pure workloads on seed 7, the mixed load on seed 8."""
import json
import ext8 as N
import run8 as R8
import run9 as R9

out = {}
for K in (16, 32):
    for kind in R8.WORKLOADS:
        out[f"pure {kind} K={K}"] = R9.run(R8.make_world(R9.FIT, kind, 1, 7), N.BatchAccount(8, K), 8)["rate"]
    out[f"mixed K={K}"] = R9.run(R8.mixed_world(R9.FIT, 1, 8), N.BatchAccount(8, K), 8)["rate"]
print(json.dumps(out))
