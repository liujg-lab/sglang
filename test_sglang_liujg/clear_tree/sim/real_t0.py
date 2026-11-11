"""Real T0 throughput (tok/s) used to recalibrate the simulator (round 7).

Source: clear_tree/progress/results.md section 2, natural short prompts, lookahead off, KV reuse off,
temperature 0, max_new_tokens 256, Target tp=2 on NPU, Draft on one GPU, RTT 3-5 ms. One run per cell
except where results.md says otherwise. The N=8 cells of the N x tau table are the same runs as the
tau sweep, so they appear once here.
"""

TAUS_SWEEP = (0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)

# (b, N, tau) -> tok/s
SPEC = {}
for b, row in {1: (47.31, 49.13, 50.21, 50.91, 51.43, 51.02, 35.58),
               4: (124.26, 126.69, 130.65, 136.07, 140.35, 139.97, 116.32),
               8: (185.13, 189.62, 194.10, 203.42, 209.29, 212.41, 194.75)}.items():
    for tau, v in zip(TAUS_SWEEP, row):
        SPEC[(b, 8, tau)] = v
for b, table in {1: {2: (37.17, 37.24, 36.05), 4: (44.41, 44.94, 44.04), 16: (52.07, 54.73, 51.92)},
                 8: {2: (189.81, 194.16, 191.93), 4: (200.99, 206.22, 209.35), 16: (174.06, 181.23, 182.05)}}.items():
    for N, row in table.items():
        for tau, v in zip((0.2, 0.4, 0.8), row):
            SPEC[(b, N, tau)] = v

TARGET_ONLY = {1: 30.74, 4: 114.31, 8: 221.39}

# Accept length (tokens per round per request) that results.md reports for N=8 natural prompts:
# 3.82 / 3.80 in the preflight (tau not recorded); 3.491 at tau=0.8 with lookahead and KV reuse on.
ALEN_N8 = {"preflight": 3.81, "tau0.8": 3.491}
