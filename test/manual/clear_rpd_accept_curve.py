"""RPD accepts per K tier against path mass, from ClearTree Draft logs.

Reads ``[CLEAR ledger d_round]`` lines. Each request row with ``accept_k``
(only when the Target verified with RPD and sent masks) gives the exact RPD
accepts A(K') of the top-K' window and its path mass ``mass_k``.

Two tables per batch bucket:

* ``all``: r(K') = sum A / sum mass over every row that has tier K'. Tiers
  past a row's window are null unless the window was the whole tree, so the
  large tiers only see shallow trees. Do not compare tiers in this table.
* ``matched``: the verdict. Tiers are added from the smallest up while at
  least --min-n rows, and at least --min-share of all rows, have every chosen
  tier; r(K') is then summed over those same rows, so the tiers are compared
  on one set of requests. The share keeps the set from shrinking to shallow
  whole-tree rows only.

``served_c`` is sum e_pred / sum e_mass of the rounds used. With the default
pooled accept ratio every request shares one P, so this is exactly that P
whatever subset of requests is kept. Under SGLANG_CLEAR_ACCEPT_PER_REQUEST=1
it is a round mix and r/c is only indicative.

If matched r(K') stays within --spread, one pooled c can absorb RPD;
otherwise the controller needs a per-tier c_K.

Usage:
    python test/manual/clear_rpd_accept_curve.py draft.log [more.log ...]
"""

import argparse
import json
import random
import sys
from collections import defaultdict

MARK = "[CLEAR ledger d_round] "
BUCKETS = ("1", "2-4", ">=5")


def bucket_of(b):
    if b <= 1:
        return "1"
    if b <= 4:
        return "2-4"
    return ">=5"


def read_rounds(paths):
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                at = line.find(MARK)
                if at < 0:
                    continue
                try:
                    yield json.loads(line[at + len(MARK) :])
                except ValueError:
                    continue


def collect(paths):
    """Per bucket: rounds, each a list of rid rows {tier: (A, mass)}."""
    rounds = defaultdict(list)
    served = defaultdict(lambda: [0.0, 0.0])
    stats = defaultdict(int)
    for record in read_rounds(paths):
        stats["rounds"] += 1
        if record.get("excluded"):
            stats["excluded"] += 1
            continue
        if record.get("rpd_masks") is None:
            stats["no_masks"] += 1
            continue
        rows = []
        for row in (record.get("e_rid") or {}).values():
            if "accept_k" not in row:
                continue
            stats["rids"] += 1
            if not row.get("rpd_ok") or not row.get("accept_k"):
                stats["rpd_not_ok"] += 1
                continue
            mass = row.get("mass_k") or {}
            tiers = {
                int(tier): (float(value), float(mass[tier]))
                for tier, value in row["accept_k"].items()
                if value is not None and tier in mass
            }
            if tiers:
                rows.append(tiers)
        if not rows:
            continue
        bucket = bucket_of(int(record.get("b") or 1))
        rounds[bucket].append(rows)
        if record.get("e_pred") is not None and record.get("e_mass"):
            served[bucket][0] += float(record["e_pred"])
            served[bucket][1] += float(record["e_mass"])
    return rounds, served, stats


def keep_rows(rounds, need):
    """Rounds restricted to rows that have every tier in ``need``."""
    return [[row for row in rows if need <= row.keys()] for rows in rounds]


def ratio(rounds, tier):
    a = m = 0.0
    for rows in rounds:
        for row in rows:
            if tier in row:
                a += row[tier][0]
                m += row[tier][1]
    return (a / m) if m > 0 else None


def count(rounds, tier):
    return sum(1 for rows in rounds for row in rows if tier in row)


def bootstrap_se(rounds, tier, draws, rng):
    """Resample whole rounds, so rows of one round stay together."""
    rounds = [rows for rows in rounds if any(tier in row for row in rows)]
    if len(rounds) < 2:
        return None
    values = []
    for _ in range(draws):
        sample = [rounds[rng.randrange(len(rounds))] for _ in rounds]
        value = ratio(sample, tier)
        if value is not None:
            values.append(value)
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def matched_tiers(rounds, min_n, min_share):
    """Smallest-first tiers that enough rows all have."""
    tiers = sorted({tier for rows in rounds for row in rows for tier in row})
    total = sum(len(rows) for rows in rounds)
    chosen = set()
    for tier in tiers:
        trial = chosen | {tier}
        rows = sum(len(kept) for kept in keep_rows(rounds, trial))
        if rows < min_n or rows < min_share * total:
            break
        chosen = trial
    return sorted(chosen)


def print_table(title, rounds, tiers, c, draws, rng):
    print(f"  {title}")
    print(f"  {'K':>4} {'n':>6} {'r(K)':>8} {'se':>8} {'r/c':>7}")
    values = {}
    for tier in tiers:
        r = ratio(rounds, tier)
        se = bootstrap_se(rounds, tier, draws, rng)
        rel = (r / c) if (r is not None and c) else None
        values[tier] = r
        print(
            f"  {tier:>4} {count(rounds, tier):>6} "
            f"{'-' if r is None else f'{r:.4f}':>8} "
            f"{'-' if se is None else f'{se:.4f}':>8} "
            f"{'-' if rel is None else f'{rel:.3f}':>7}"
        )
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--min-n", type=int, default=30, help="rows in the matched set")
    parser.add_argument(
        "--min-share", type=float, default=0.5, help="share of rows in the matched set"
    )
    parser.add_argument("--boot", type=int, default=200)
    parser.add_argument("--spread", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rounds, served, stats = collect(args.logs)
    rng = random.Random(args.seed)
    keys = ("rounds", "excluded", "no_masks", "rids", "rpd_not_ok")
    print(" ".join(f"{key}={stats.get(key, 0)}" for key in keys))
    if not rounds:
        print("no RPD rows with accept_k; is the Target on --speculative-verify-mode rpd?")
        return 1
    for bucket in sorted(rounds, key=BUCKETS.index):
        per_round = rounds[bucket]
        e_pred, e_mass = served[bucket]
        c = (e_pred / e_mass) if e_mass > 0 else None
        rows = sum(len(kept) for kept in per_round)
        print(
            f"\nb={bucket} rounds={len(per_round)} rows={rows} "
            f"served_c={'-' if c is None else f'{c:.4f}'}"
        )
        all_tiers = sorted({t for kept in per_round for row in kept for t in row})
        print_table("all (tiers see different rows; do not compare)", per_round,
                    all_tiers, c, args.boot, rng)
        tiers = matched_tiers(per_round, args.min_n, args.min_share)
        if len(tiers) < 2:
            print(
                f"  fewer than two tiers shared by {args.min_n} rows and "
                f"{args.min_share:.0%} of rows; no verdict"
            )
            continue
        matched = keep_rows(per_round, set(tiers))
        values = print_table(
            f"matched ({sum(len(k) for k in matched)} rows with every tier {tiers})",
            matched, tiers, c, args.boot, rng,
        )
        kept = [values[tier] for tier in tiers if values[tier] is not None]
        low = min(kept)
        spread = (max(kept) - low) / low if low > 0 else float("inf")
        verdict = "flat: one c absorbs RPD" if spread < args.spread else "varies: needs c_K"
        print(f"  matched spread: {spread:.1%} -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
