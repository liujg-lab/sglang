"""Compare three layer-cost fits on the ledger's own d_layer stream.

Usage:
    python layer_fit_compare.py draft.log [more.log ...]
    python layer_fit_compare.py --synthetic

Each ``[CLEAR ledger d_layer]`` row is replayed in log order: predict first,
then update when the row has ``fit: true``. This is the same order the Draft
ledger uses while serving, so it evaluates the fitting rule, not an offline
table. A larger ``session_id`` drops every fit, like ``on_external_event``.
Rows with ``post_reply: true`` are skipped, as in the ledger.

Fits, all with the ledger's ``ForgettingRegressor`` (forget 0.999):
    A  current: key (W, path), features [1, g, sum_ell]
    B  one-hot: key (W, path), features [1{g=..}, ..., sum_ell], one slope
    C  split:   key (W, path, g), features [1, sum_ell]

``g`` is ``graph_bucket``. Output per (W, path, g): rows, how many a fit could
price (and how many also pass the sum_ell span check and give > 0), median
residual and median absolute residual. Residual is measured / predicted - 1,
the same sign as ``residual`` in the log.

Standard library only. The ledger module is loaded from this repository by
file path, so ``sglang`` does not need to import.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import re
import sys
import types
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PKG = os.path.join(REPO, "python", "sglang", "srt", "speculative", "clear_tree_remote")
ROW = re.compile(r"\[CLEAR ledger d_layer\]\s*(\{.*\})")


def _load_ledger():
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.speculative",
        "sglang.srt.speculative.clear_tree_remote",
    ):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []
            sys.modules[name] = mod
    full = "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger"
    spec = importlib.util.spec_from_file_location(
        full, os.path.join(PKG, "clear_tree_ledger.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


L = _load_ledger()
SLOPE_RIDGE = L.LAYER_SPAN_PAD**2


class Fit:
    """One variant. ``query`` returns (raw prediction, usable prediction)."""

    def __init__(self, kind, buckets):
        self.kind = kind
        self.buckets = sorted(buckets)
        self.models = {}
        self.cell_n = defaultdict(int)

    def reset(self):
        self.models.clear()
        self.cell_n.clear()

    def _key(self, w, path, g):
        return (w, path, g) if self.kind == "C" else (w, path)

    def _new(self):
        if self.kind == "A":
            return L._new_layer_model()
        if self.kind == "B":
            n = len(self.buckets) + 1
            ridge = [L.RIDGE] * len(self.buckets) + [SLOPE_RIDGE]
            return L.ForgettingRegressor(
                n, L.LAYER_MIN_SAMPLES, ridge=ridge, span_feature=n - 1
            )
        return L.ForgettingRegressor(
            2, L.LAYER_MIN_SAMPLES, ridge=(0.0, SLOPE_RIDGE), span_feature=1
        )

    def _features(self, g, ell):
        if self.kind == "A":
            return [1.0, float(g), float(ell)]
        if self.kind == "B":
            return [1.0 if g == b else 0.0 for b in self.buckets] + [float(ell)]
        return [1.0, float(ell)]

    def query(self, w, path, g, ell):
        model = self.models.get(self._key(w, path, g))
        if model is None or model.n < L.LAYER_MIN_SAMPLES:
            return None, None
        if self.kind == "B" and self.cell_n[(w, path, g)] == 0:
            # This bucket's intercept has no data, only the ridge.
            return None, None
        feats = self._features(g, ell)
        raw = model.predict(feats)
        if raw is None:
            return None, None
        usable = raw if (model.span_allows(feats) and raw > 0.0) else None
        return raw, usable

    def update(self, w, path, g, ell, ms):
        key = self._key(w, path, g)
        model = self.models.get(key)
        if model is None:
            model = self._new()
            self.models[key] = model
        model.update(self._features(g, ell), ms)
        self.cell_n[(w, path, g)] += 1


def _measured(row):
    if row.get("wall_us") is not None:
        return int(row["wall_us"]) / 1000.0
    if row.get("host_us") is not None and row.get("gpu_us") is not None:
        return (int(row["host_us"]) + int(row["gpu_us"])) / 1000.0
    return None


def _session(value):
    """Numeric when it parses, so "10" sorts after "9"."""
    text = "" if value is None else str(value)
    try:
        return (0, int(text), "")
    except ValueError:
        return (1, 0, text)


def read_rows(paths):
    rows = []
    for path in paths:
        with open(path, errors="replace") as handle:
            for line in handle:
                found = ROW.search(line)
                if not found:
                    continue
                try:
                    rows.append(json.loads(found.group(1)))
                except ValueError:
                    continue
    return rows


def evaluate(rows):
    rows = [r for r in rows if not r.get("post_reply") and _measured(r) is not None]
    buckets = {int(r["graph_bucket"]) for r in rows}
    fits = {kind: Fit(kind, buckets) for kind in "ABC"}
    stats = {kind: defaultdict(lambda: {"n": 0, "raw": [], "use": 0}) for kind in "ABC"}
    session = None
    for row in rows:
        sid = _session(row.get("session_id"))
        if session is not None and sid > session:
            for fit in fits.values():
                fit.reset()
        if session is None or sid > session:
            session = sid
        w, path = int(row["w"]), str(row["path"])
        g, ell, ms = int(row["graph_bucket"]), int(row["sum_ell"]), _measured(row)
        for kind, fit in fits.items():
            cell = stats[kind][(w, path, g)]
            cell["n"] += 1
            raw, usable = fit.query(w, path, g, ell)
            if raw not in (None, 0.0):
                cell["raw"].append(ms / raw - 1.0)
            if usable is not None:
                cell["use"] += 1
            if row.get("fit", True):
                fit.update(w, path, g, ell, ms)
    return stats


def _median(values):
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _pct(value):
    return "    -" if value is None else f"{value * 100:+6.1f}%"


def report(stats):
    cells = sorted(set().union(*(set(s) for s in stats.values())))
    print("W  path   g      rows | fit  priced  usable   med res  med |res|")
    for cell in cells:
        w, path, g = cell
        for kind in "ABC":
            item = stats[kind][cell]
            raw = item["raw"]
            med = _median(raw)
            mabs = _median([abs(x) for x in raw])
            print(
                f"{w:<3}{path:<6}{g:>3} {item['n']:>9} |  {kind}  {len(raw):>6}  "
                f"{item['use']:>6}  {_pct(med)}  {_pct(mabs).strip():>7}"
            )
        print()


def synthetic(seed=0):
    """A made-up c1 -> c4 stream. Not a measurement; it shows the mechanism."""
    rng = random.Random(seed)
    # Concave in g, slope per summed prefix token shared inside one W.
    base = {
        (4, 1): 3.4, (4, 2): 4.9, (4, 4): 5.6,
        (8, 1): 4.6, (8, 2): 6.3, (8, 4): 7.4,
        (16, 1): 6.0, (16, 2): 8.6, (16, 4): 10.5,
    }
    kappa = {4: 0.0006, 8: 0.0009, 16: 0.0014}
    rows = []

    def emit(w, m, ell):
        g = L.graph_bucket(m)
        ms = (base[(w, g)] + kappa[w] * ell) * math.exp(rng.gauss(0.0, 0.06))
        rows.append(
            {"w": w, "path": "graph", "graph_bucket": g, "sum_ell": ell,
             "wall_us": int(ms * 1000), "fit": True, "session_id": "s"}
        )

    # c1: one request, served W 8, explore W 16 every 20th, W 4 burst every 300.
    prefix = 200
    for rnd in range(15000):
        if rnd % 160 == 0:
            prefix = rng.randint(150, 400)
        prefix += 3
        w = 16 if rnd % 20 == 19 else (4 if rnd % 300 < 2 else 8)
        for _layer in range(rng.randint(3, 6)):
            emit(w, 1, prefix)
    # c4: four requests, served W 4, explore W 8. Late layers keep fewer requests.
    prefixes = [rng.randint(150, 900) for _ in range(4)]
    for rnd in range(6000):
        prefixes = [p + 3 if p < 1300 else rng.randint(150, 400) for p in prefixes]
        w = 8 if rnd % 20 == 19 else 4
        active = list(range(4))
        for layer in range(rng.randint(3, 6)):
            if layer >= 2:
                active = [i for i in active if rng.random() < 0.83]
                if rng.random() < 0.012:
                    active = active[:1]
            if not active:
                break
            emit(w, len(active), sum(prefixes[i] for i in active))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="*")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if args.synthetic:
        rows = synthetic()
    elif args.logs:
        rows = read_rows(args.logs)
    else:
        parser.error("give Draft log files or --synthetic")
    print(f"{len(rows)} d_layer rows")
    report(evaluate(rows))


if __name__ == "__main__":
    main()
