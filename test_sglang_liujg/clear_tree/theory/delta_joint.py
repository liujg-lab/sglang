"""Shared request effect delta_i: joint path integral vs per-edge predictive vs MAP plug-in.

logit P(edge accepted | delta) = eta + delta, delta ~ N(0, s^2) shared by every depth of one request.
A depth-d chain is accepted with probability E_delta[ sigmoid(eta + delta)^d ].
  per-edge : (E_delta[sigmoid(eta + delta)])^d      (predictive per edge, then multiplied)
  MAP-plug : sigmoid(eta)^d                          (delta at its mode 0)
Prints each approximation's error relative to the joint integral.
"""
import math
import random


def sig(x):
    return 1 / (1 + math.exp(-x))


def main(n=200000, seed=0):
    rng = random.Random(seed)
    for s in (0.25, 0.5, 1.0):
        deltas = [rng.gauss(0, s) for _ in range(n)]
        print(f"posterior sd of delta = {s}")
        for eta in (2.0, 1.0):
            for d in (2, 4, 8):
                joint = sum(sig(eta + x) ** d for x in deltas) / n
                per_edge = (sum(sig(eta + x) for x in deltas) / n) ** d
                plug = sig(eta) ** d
                print(f"  eta={eta} d={d}: joint {joint:.4f}  per-edge {per_edge:.4f} "
                      f"({per_edge / joint - 1:+.1%})  MAP-plug {plug:.4f} ({plug / joint - 1:+.1%})")


if __name__ == "__main__":
    main()
