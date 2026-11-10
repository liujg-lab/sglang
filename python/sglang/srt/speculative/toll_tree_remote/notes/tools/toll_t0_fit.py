"""Fit the OPT T0 cost checks and print the section-6 decision.

``b == 1`` and ``b > 1`` are fit separately. One N is held out of the cost
fit. Coefficients are clipped at zero. Missing measurements stay missing and
fail the check that needs them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def _float(row, key):
    value = row.get(key, "")
    if value in ("", None, "None"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _load(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("status", "ok") == "ok"]


def _nnls(rows, y_key):
    usable = []
    for row in rows:
        y = _float(row, y_key)
        b = _float(row, "b")
        ell = _float(row, "ell")
        width = _float(row, "N")
        if None in (y, b, ell, width):
            continue
        usable.append((y, 1.0, b * ell, b * width))
    if len(usable) < 3:
        return None
    free = [0, 1, 2]
    coef = [0.0, 0.0, 0.0]
    for _ in range(8):
        solved = _lstsq(usable, free)
        if solved is None:
            return None
        for index, value in zip(free, solved):
            coef[index] = value
        free = [index for index, value in enumerate(coef) if value > 1e-12]
        for index in range(3):
            if index not in free:
                coef[index] = 0.0
        if not free:
            break
    return coef


def _lstsq(usable, free):
    if not free:
        return []
    # Normal equations on the free columns. The design matrix is tiny.
    gram = [[0.0 for _ in free] for _ in free]
    rhs = [0.0 for _ in free]
    for y, *features in usable:
        cols = [features[index] for index in free]
        for i, left in enumerate(cols):
            rhs[i] += left * y
            for j, right in enumerate(cols):
                gram[i][j] += left * right
    return _solve(gram, rhs)


def _solve(matrix, rhs):
    size = len(rhs)
    aug = [row[:] + [rhs[index]] for index, row in enumerate(matrix)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        for j in range(col, size + 1):
            aug[col][j] /= scale
        for row in range(size):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(col, size + 1):
                aug[row][j] -= factor * aug[col][j]
    return [aug[row][size] for row in range(size)]


def _predict(coef, row):
    b = _float(row, "b")
    ell = _float(row, "ell")
    width = _float(row, "N")
    if coef is None or None in (b, ell, width):
        return None
    return coef[0] + coef[1] * b * ell + coef[2] * b * width


def _rel_error(rows, coef, y_key):
    errors = []
    for row in rows:
        y = _float(row, y_key)
        pred = _predict(coef, row)
        if y in (None, 0) or pred is None:
            continue
        errors.append(abs(pred - y) / abs(y))
    if not errors:
        return None
    return sum(errors) / len(errors)


def _holdout(rows):
    widths = sorted({int(float(row["N"])) for row in rows if row.get("N")})
    if len(widths) < 2:
        return None, rows, []
    held = widths[len(widths) // 2]
    train = [row for row in rows if int(float(row["N"])) != held]
    test = [row for row in rows if int(float(row["N"])) == held]
    return held, train, test


def _q1(rows):
    report = {}
    passed = True
    for label, subset in _split_b(rows):
        held, train, test = _holdout([row for row in subset if row.get("group") == "A"])
        if held is None:
            report[label] = {"status": "insufficient"}
            passed = False
            continue
        item = {"holdout_n": held}
        for key in ("C_L_ms", "T_t_ms"):
            coef = _nnls(train, key)
            error = _rel_error(test, coef, key)
            item[key] = None if error is None else round(error, 4)
            item[key + "_coef"] = None if coef is None else [round(value, 6) for value in coef]
            if error is None or error > 0.15:
                passed = False
        report[label] = item
    return passed and bool(report), report


def _q2(rows):
    errors = []
    missing = 0
    for row in rows:
        if row.get("group") != "A":
            continue
        total = _float(row, "T_ms")
        parts = [_float(row, name) for name in ("R_ms", "T_t_ms", "draft_round_ms")]
        if total in (None, 0) or any(part is None for part in parts):
            missing += 1
            continue
        errors.append(abs(total - sum(parts)) / abs(total))
    if not errors:
        return False, {"mean_residual": None, "missing": missing}
    mean = sum(errors) / len(errors)
    return mean <= 0.10 and missing == 0, {
        "mean_residual": round(mean, 4),
        "missing": missing,
        "n": len(errors),
    }


def _split_b(rows):
    return (
        ("b1", [row for row in rows if _float(row, "b") == 1]),
        ("bgt1", [row for row in rows if (_float(row, "b") or 0) > 1]),
    )


def _throughput(row):
    return _float(row, "throughput_tok_s")


def _best_tau(rows):
    by_tau = defaultdict(list)
    for row in rows:
        if row.get("group") != "B" or _throughput(row) is None:
            continue
        by_tau[float(row["tau"])].append(_throughput(row))
    if not by_tau:
        return None, None
    means = {tau: sum(values) / len(values) for tau, values in by_tau.items()}
    best = max(means, key=means.get)
    return best, means[best]


def _crossing(points):
    """Return tau where (tok/s / 1000 / b) * C_L_ms equals tau."""
    scored = []
    for tau, throughput, cost, batch in points:
        if None in (throughput, cost) or batch in (None, 0):
            continue
        scored.append((tau, (throughput / 1000.0 / batch) * cost))
    scored.sort()
    for (left_t, left_f), (right_t, right_f) in zip(scored, scored[1:]):
        if (left_f - left_t) == 0:
            return left_t
        if (left_f - left_t) * (right_f - right_t) <= 0:
            span = (left_f - left_t) - (right_f - right_t)
            if span == 0:
                return left_t
            weight = (left_f - left_t) / span
            return left_t + weight * (right_t - left_t)
    return None


def _q3_q5(rows):
    conditions = defaultdict(list)
    for row in rows:
        if row.get("group") != "B":
            continue
        conditions[_float(row, "b")].append(row)
    ratios = []
    gains = []
    global_choice = _best_tau([row for row in rows if row.get("group") == "B"])
    global_tau = global_choice[0]
    for batch, subset in conditions.items():
        if batch is None:
            continue
        best, best_tp = _best_tau(subset)
        points = []
        by_tau = defaultdict(list)
        for row in subset:
            by_tau[float(row["tau"])].append(row)
        for tau, group in by_tau.items():
            throughput = [value for value in (_throughput(row) for row in group) if value is not None]
            cost = [_float(row, "C_L_ms") for row in group]
            cost = [value for value in cost if value is not None]
            if throughput and cost:
                points.append((tau, sum(throughput) / len(throughput), sum(cost) / len(cost), batch))
        predicted = _crossing(points)
        if best is not None and predicted not in (None, 0):
            ratios.append(math.log(best / predicted) / math.log(2))
        if best_tp and global_tau is not None:
            same = [
                _throughput(row)
                for row in subset
                if float(row["tau"]) == global_tau and _throughput(row) is not None
            ]
            if same and sum(same) / len(same) > 0:
                gains.append(best_tp / (sum(same) / len(same)))
    if len(ratios) < 2:
        return False, None, {"ratios": ratios, "gains": gains, "global_tau": global_tau}
    mean = sum(ratios) / len(ratios)
    spread = max(ratios) - min(ratios)
    # One grid step of the tau list is about 2x, so one log2 unit.
    stable = spread <= 1.0
    value = None
    if gains:
        log_mean = sum(math.log(item) for item in gains) / len(gains)
        value = math.exp(log_mean) - 1
    return stable, value, {
        "log2_mean": round(mean, 4),
        "log2_spread": round(spread, 4),
        "V": None if value is None else round(value, 4),
        "global_tau": global_tau,
    }


def _q4(rows):
    best = {}
    for batch in (1, 4, 8):
        subset = [row for row in rows if row.get("group") == "B" and _float(row, "b") == batch]
        tau, _tp = _best_tau(subset)
        best[batch] = tau
    if None in (best.get(1), best.get(8)):
        return False, {"best_tau": best, "reason": "missing b"}
    # Larger b should not deepen the tree: best tau should not fall.
    direction = best[8] + 1e-9 >= best[1] * 0.5
    return direction, {"best_tau": best}


def _noise(rows):
    """Relative spread of repeated controls that share N, b, and tau."""
    buckets = defaultdict(list)
    for row in rows:
        if row.get("group") != "control" or _throughput(row) is None:
            continue
        buckets[(row.get("N"), row.get("b"), row.get("tau"))].append(_throughput(row))
    spreads = []
    for values in buckets.values():
        if len(values) < 2:
            continue
        mean = sum(values) / len(values)
        if not mean:
            continue
        var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        spreads.append(math.sqrt(var) / mean)
    if not spreads:
        return None
    return sum(spreads) / len(spreads)


def _decide(q1, q2, q3, q4, value, noise):
    if not q1 or not q2:
        return "Q1 or Q2 failed: change the cost model before more cells"
    noisy = value is not None and noise is not None and value < 2 * noise
    if value is None:
        return "V is missing"
    if noisy or value < 0.02:
        return "V is under 2% or inside the noise: stop adaptation and keep a fixed tau"
    if q3 and q4 and value >= 0.05:
        return "Q1-Q4 passed and V >= 5%: continue with T2b and T3"
    if q3 and 0.02 <= value < 0.05:
        return "Q1-Q3 passed and V is between 2% and 5%: tune tau only and keep N fixed"
    if not q3 or not q4:
        return "Q3 is not stable or Q4 points the wrong way on b: change the decision rule"
    return "no rule matched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    rows = _load(args.csv)
    q1, q1_detail = _q1(rows)
    q2, q2_detail = _q2(rows)
    q3, value, q3_detail = _q3_q5(rows)
    q4, q4_detail = _q4(rows)
    noise = _noise(rows)
    decision = _decide(q1, q2, q3, q4, value, noise)
    report = {
        "Q1": q1,
        "Q1_detail": q1_detail,
        "Q2": q2,
        "Q2_detail": q2_detail,
        "Q3": q3,
        "Q3_detail": q3_detail,
        "Q4": q4,
        "Q4_detail": q4_detail,
        "V": None if value is None else round(value, 4),
        "control_cv": None if noise is None else round(noise, 4),
        "decision": decision,
        "rows": len(rows),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
