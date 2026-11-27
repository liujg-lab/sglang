#!/usr/bin/env python3
"""Review this run's archived evidence only; never contact or start a service."""

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

RUN_ID = "2026-11-27-c"
COMMIT = "4b5de83245a134624fdd88acd610c7175ad06f64"
BATCHES = [1, 2, 4]
TIERS = [2, 4, 8, 16, 32, 64]
BUCKETS = [256, 512, 1024, 2048, 4096]
N_ITEMS = 100
TAU = 0.4
MODES = {"greedy": "clear_tree_fit.json", "rpd": "clear_tree_fit.rpd.json"}
TRIALS = [
    {"id": trial, "mode": mode, "dataset": "RSTeller", "concurrency": concurrency,
     **({"rpd_tau": TAU} if mode == "rpd" else {})}
    for trial, mode, concurrency in (("01", "greedy", 1), ("02", "greedy", 4),
                                     ("03", "rpd", 1), ("04", "rpd", 4))
]
PLAN = dict(run_id=RUN_ID, experiment="e2e", commit=COMMIT, fit_schema=4,
            refit_each_mode=True, ledger=True, kv_reuse=True, bonus_prediction=True,
            draft_lookahead=True, target_lookahead=False, accept_half_life=1000,
            graph_bs=BATCHES, max_running_requests=4, widths=TIERS, ks=TIERS,
            initial_width=8, max_depth=8, clear_tree_tau=0.5, draft_capacity=64,
            target_capacity=65, max_items=N_ITEMS, max_new_tokens=1024, temperature=0,
            request_rate="inf", timeout_s=600, metrics=["bleu", "rouge"], trials=TRIALS)
FIT_INPUTS = dict(batches=BATCHES, ks=TIERS, widths=TIERS, buckets=BUCKETS, max_depth=8)
FIT_FINGERPRINT = hashlib.sha1(json.dumps(
    dict(FIT_INPUTS, verify_budget_version=1, layer_sample_version=1, layer_samples=50,
         width_sample_target=60, width_min_depth=4, width_rounds_per_width=4),
    separators=(",", ":"), sort_keys=True,
).encode()).hexdigest()[:8]
TIMINGS = ("verify_wall_us", "accept_ready_us", "rpd_build_us", "rpd_stats_us",
           "rpd_select_us", "rpd_mask_us")
METRICS = {"Avg BLEU": "bleu", "Avg ROUGE-1": "rouge1", "Avg ROUGE-2": "rouge2", "Avg ROUGE-L": "rougeL"}
EVIDENCE_ERRORS = (OSError, ValueError, KeyError, TypeError, SyntaxError, csv.Error)


def number(value):
    return type(value) is int or type(value) is float and math.isfinite(value)


def nonnegative(value):
    return number(value) and value >= 0


def near(a, b):
    return number(a) and number(b) and math.isclose(a, b, rel_tol=2e-5, abs_tol=1e-7)


def obj(value):
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def matches(actual, expected):
    """Compare required values without treating True as 1."""
    return all(type(actual.get(k)) is type(v) and actual[k] == v for k, v in expected.items())


def text(path):
    return path.read_text(encoding="utf-8", errors="replace")


def parse_json(value, *, allow_log_suffix=False):
    def bad_constant(token):
        raise ValueError(f"non-finite JSON value {token}")

    def finite_float(token):
        value = float(token)
        if not math.isfinite(value):
            bad_constant(token)
        return value

    decoder = json.JSONDecoder(parse_constant=bad_constant, parse_float=finite_float)
    if allow_log_suffix:
        return obj(decoder.raw_decode(value.lstrip())[0])
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return obj(decoder.decode(value))


def load(path):
    return parse_json(text(path))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def only(paths):
    paths = sorted(paths)
    if len(paths) != 1:
        raise ValueError(f"expected one artifact, found {paths}")
    return paths[0]


def record(issues, label, fn, default=None):
    try:
        return fn()
    except EVIDENCE_ERRORS as exc:
        issues.append(f"{label}: {exc}")
        return default


def jsonl(path, issues):
    with path.open(encoding="utf-8", errors="replace") as handle:
        for n, line in enumerate(handle, 1):
            if line.strip():
                row = record(issues, f"{path.name}:{n}", lambda: parse_json(line))
                if row is not None:
                    yield row


def flush_counts(accept, content):
    return Counter(accept.FLUSH.findall(content)), Counter(accept.HTTP_FLUSH.findall(content))


def passed_flush(counts, ordinal):
    return counts == ({"0": ordinal, "1": ordinal}, {"200": ordinal})


def client(accept, raw, trial, issues):
    directory = raw / trial["id"]
    errors = []
    rows = []
    files = sorted(directory.glob("client.*.jsonl"))
    path = record(errors, "client JSONL", lambda: only(files))
    if path:
        rows = list(jsonl(path, errors))
    stdout = record(errors, "client stdout", lambda: text(directory / "client.stdout"), "")
    exit_code = record(errors, "client exit", lambda: int(text(directory / "exit-code.txt")))
    manifest = record(errors, "client manifest", lambda: load(only(directory.glob("client.*.run.json"))), {})
    stats = record(errors, "client summary", lambda: obj(manifest.get("summary")), {}) if manifest else {}
    csv_path = record(errors, "client CSV", lambda: only(directory.glob("client.*.csv")))

    def csv_row():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            return only_row(csv.DictReader(handle))

    csv_data = record(errors, "client CSV row", csv_row, {}) if csv_path else {}
    metrics = {key: record(errors, key, lambda: float(csv_data[key])) if csv_data.get(key) else None
               for key in METRICS}
    metrics = {key: value if number(value) else None for key, value in metrics.items()}
    metric_rows = [value if isinstance(value, dict) else {} for value in (row.get("metrics") for row in rows)]
    item_metrics = {key: [row.get(source) for row in metric_rows] for key, source in METRICS.items()}
    scored = re.findall(r"\bscored (\d+)/(\d+) items", stdout)
    printed = re.findall(r"\bsummary items=(\d+)/(\d+)\s+toks=(\d+)\s+wall=([\d.]+)s", stdout)
    wall = record(errors, "formal wall", lambda: float(printed[0][3])) if len(printed) == 1 else None
    ok = [row for row in rows if not row.get("error")]
    rids = {row["response_meta_info"].get("id") for row in ok
            if isinstance(row.get("response_meta_info"), dict)
            and isinstance(row["response_meta_info"].get("id"), str)
            and row["response_meta_info"]["id"]}
    fields = ("completion_tokens", "spec_accept_token_num", "spec_draft_token_num", "spec_fallback_ct")
    coverage = {k: sum(nonnegative(r.get(k)) and r[k] == int(r[k]) for r in ok) for k in fields}
    totals = [sum(int(r[k]) for r in ok) if ok and coverage[k] == len(ok) else None for k in fields]
    out, accepted, drafted, fallback = totals
    tok_s = stats.get("output_tok_s")
    ratio = accepted / drafted if drafted and accepted is not None else None
    client_id = path.name.removeprefix("client.").removesuffix(".jsonl") if path else None
    snapshot = manifest.get("server") or {}
    settings = (snapshot.get("settings") or {}) if isinstance(snapshot, dict) else {}
    expected_target = dict(speculative_algorithm="CLEAR_TREE_REMOTE", device="npu", tp_size=2,
                           page_size=128, disable_cuda_graph=False, cuda_graph_bs=BATCHES,
                           attention_backend="ascend", speculative_verify_mode=trial["mode"],
                           speculative_num_steps=8, speculative_num_draft_tokens=65)
    if trial["mode"] == "rpd":
        expected_target["speculative_rpd_tau"] = TAU
    checked_target = {key: settings.get(key) for key in expected_target} if isinstance(settings, dict) else {}
    items = [(row.get("image"), row.get("question"), row.get("ground_truth")) for row in rows]
    traffic = re.findall(r"\btraffic=open\s+request_rate=(\S+)\s+max_concurrency=(\S+)\s+max_new_tokens=(\d+)", stdout)
    warm_flush = re.findall(r"=== warmup \(first item, discarded\) ===|warmup ok|flush_cache ->|=== main run ===", stdout)
    flush = record(errors, "trial flush", lambda: text(directory / "target-flush.txt"))
    counts = flush_counts(accept, flush) if accept is not None and flush is not None else None
    gates = dict(
        artifacts=bool(not errors and manifest.get("run_id") == client_id and csv_path and path
                       and csv_path.name == path.with_suffix(".csv").name
                       and Path(str(manifest.get("results_jsonl"))).name == path.name
                       and Path(str(manifest.get("summary_csv"))).name == csv_path.name),
        exit=exit_code == 0,
        requests=len(rows) == len(ok) == len(rids) == stats.get("n_ok") == stats.get("n_total") == N_ITEMS,
        ordered_items=[row.get("global_index") for row in rows] == list(range(N_ITEMS))
                      and all(isinstance(image, str) and image and truth for image, _, truth in items),
        traffic=traffic == [("inf", str(trial["concurrency"]), "1024")]
                and csv_data.get("request_rate") == "inf" and csv_data.get("traffic") == "open"
                and csv_data.get("max_concurrency") == str(trial["concurrency"]),
        target_config=isinstance(snapshot, dict) and snapshot.get("status") == "available"
                      and matches(checked_target, expected_target),
        metrics=scored == [("100", "100")] and all(
            len(values) == N_ITEMS and all(number(v) and 0 <= v <= 1 for v in values)
            and number(metrics[key]) and abs(sum(values) / N_ITEMS - metrics[key]) <= 0.00005001
            for key, values in item_metrics.items()),
        valid_timing=bool(out and number(tok_s) and tok_s > 0 and number(wall) and wall > 0
                          and stats.get("total_tokens") == out and csv_data.get("Total Generated Tokens") == str(out)
                          and printed[0][:3] == (str(len(ok)), str(len(rows)), str(out))
                          and abs(out / tok_s - wall) <= 0.000501),
        flush_client=warm_flush == ["=== warmup (first item, discarded) ===", "warmup ok",
                                    "flush_cache ->", "=== main run ==="] and "flush_cache failed:" not in stdout,
        flush_target=passed_flush(counts, 1 if trial["concurrency"] == 1 else 2),
        spec=bool(out and drafted and ratio is not None and 0 <= ratio <= 1 and fallback == 0
                  and all(r[fields[1]] <= min(r[fields[0]], r[fields[2]]) and 0 < r[fields[0]] <= 1024 for r in ok)),
    )
    issues.extend(f"trial {trial['id']}: {error}" for error in errors)
    return dict(trial, requests=len(rows), ok=len(ok), exit_code=exit_code, output_tokens=out, wall_s=wall,
                tok_s=tok_s if number(tok_s) else None, total_accepted=accepted, total_draft=drafted,
                accept_ratio=ratio, fallback=fallback, actual_client_concurrency=stats.get("concurrency"),
                metrics=metrics, metrics_scored=list(map(int, scored[-1])) if scored else None,
                metric_coverage=coverage, response_ids_unique=len(rids) == len(ok),
                target_flush_cumulative=dict(tp=counts[0], http=counts[1]) if counts is not None else None,
                ordered_items_sha256=sha(json.dumps(items, ensure_ascii=False, sort_keys=True).encode()) if rows else None,
                target_config=checked_target), rids, gates


def only_row(reader):
    rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"expected one row, found {len(rows)}")
    return rows[0]


def round_key(row):
    session, seq = row.get("session_id"), row.get("rpc_seq")
    if not isinstance(session, str) or not session or type(seq) is not int or seq < 0:
        raise ValueError("invalid (session_id, rpc_seq)")
    return session, seq


def scan_log(accept, path, mode, role, issues):
    """Keep good records even if a bad/truncated event occurs in this block."""
    out = dict(captures=set(), flush_tp=Counter(), flush_http=Counter(), windows=[],
               fit=dict(states={}, commits=[], refills=[], started=0, driver=[], writes=[], absent=[],
                        rpd_ack=[], flat=0), events=defaultdict(list), errors=0, present=False)
    markers = {"[CLEAR fit_state]": "states", "[CLEAR prefit_commit]": "commits",
               "[CLEAR prefit_refill_group]": "refills"}
    # The hosts have different clocks. Use each log's own transition, then
    # let client response IDs select formal Draft rounds within that generation.
    active = False
    marker = re.compile(r"\[CLEAR ledger (d_round|d_layer|d_rpd|t_verify)\]")
    name = MODES[mode]
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            out["present"] = True
            for n, line in enumerate(handle, 1):
                try:
                    match = accept.STAMP.search(line)
                    rank = (match.group(2) if match else None) or "0"
                    capture = accept.CAPTURE.search(line)
                    if capture:
                        out["captures"].add(tuple(map(int, capture.groups())))
                    out["flush_tp"].update(accept.FLUSH.findall(line))
                    out["flush_http"].update(accept.HTTP_FLUSH.findall(line))
                    fit = out["fit"]
                    if "[CLEAR] no fit file at " in line:
                        fit["absent"].append((n, Path(line.split("[CLEAR] no fit file at ", 1)[1].strip()).name))
                    write = re.search(r"\[CLEAR\] wrote fit file (.+) cells=(\d+)", line)
                    if write:
                        fit["writes"].append((n, Path(write[1]).name, int(write[2])))
                        if role == "draft":
                            active = Path(write[1]).name == name
                            out["events"].clear()
                    elif "[CLEAR] wrote fit file" in line:
                        raise ValueError("malformed fit write")
                    driver = re.search(r"\[CLEAR\] prefit driver finished state=(\w+)", line)
                    if role == "target" and driver and rank == "0":
                        fit["driver"].append(driver[1])
                        active = driver[1] == "ready"
                    fit["started"] += "[CLEAR] prefit started" in line and rank == "0"
                    ack = re.search(r"\[CLEAR\] target verifies with RPD \(tau=([^)]*)\).*fit file (.+)", line)
                    if ack:
                        fit["rpd_ack"].append((ack[1], Path(ack[2].strip()).name))
                    fit["flat"] += "rpd_flat" in line
                    for token, key in markers.items():
                        if token in line:
                            row = parse_json(line.split(token, 1)[1], allow_log_suffix=role == "target")
                            if key == "states":
                                fit[key][rank] = row
                            elif rank == "0":
                                fit[key].append(row)
                            break
                    event = marker.search(line)
                    if event and active:
                        kind = event[1]
                        row = parse_json(line[event.end():])
                        if role == "draft" and kind.startswith("d_"):
                            round_key(row)
                            if kind != "d_round" or row.get("prefit") is None:
                                out["events"][kind].append(row)
                        elif role == "target" and kind == "t_verify":
                            out["events"][kind].append(row)
                    if role == "draft" and "[SR " in line and " round]" in line and accept.RANK.search(line):
                        reason = accept.REASON.search(line)
                        if accept.RANK.search(line)[1] == "0" and reason:
                            counters = obj(accept.parse_counters(line))
                            if any(type(counters.get(k, 0)) is not int or counters.get(k, 0) < 0
                                   for k in accept.BONUS_KEYS):
                                raise ValueError("invalid bonus counters")
                            out["windows"].append((reason[1], counters))
                except EVIDENCE_ERRORS as exc:
                    out["errors"] += 1
                    if out["errors"] <= 5:
                        issues.append(f"{mode}/{path.name}:{n}: {exc}")
    except OSError as exc:
        issues.append(f"{mode}/{path.name}: {exc}")
    return out


def fit_review(path, mode):
    data = path.read_bytes()
    fit = parse_json(data)
    header = obj(fit["header"])
    draft, target = obj(header["draft"]), obj(header["target"])
    plan, coverage = obj(fit["plan"]), obj(fit["coverage"])
    # For this fixed plan every graph (W, batch) cell is reachable and has
    # planned yield >= 50; the 5 KV buckets also cover all 90 H cells.
    expected_h = set(itertools.product(BATCHES, TIERS, BUCKETS))
    expected_layers = set(itertools.product(TIERS, ("graph",), BATCHES))
    violations = Counter()

    def keys(rows, name, valid):
        result = set()
        for cell in rows:
            if not valid(cell):
                violations[f"{name}_key"] += 1
            elif tuple(cell) in result:
                violations[f"{name}_duplicate"] += 1
            else:
                result.add(tuple(cell))
        return result

    def h_key(cell):
        return isinstance(cell, (tuple, list)) and len(cell) == 3 and all(type(x) is int for x in cell)

    def layer_key(cell):
        return (isinstance(cell, (tuple, list)) and len(cell) == 3 and type(cell[0]) is int
                and cell[0] in TIERS and isinstance(cell[1], str) and cell[1] in ("graph", "eager", "unknown")
                and type(cell[2]) is int and cell[2] in BATCHES)

    declared_h = keys(coverage["required_h"], "required_h", h_key)
    declared_layers = keys(coverage["required_layers"], "required_layers", layer_key)
    h_rows, layers = fit["h"], fit["layers"]
    h_counts, layer_counts = {}, {}
    null_ells = 0
    if any(not isinstance(group, list) for group in (h_rows, layers, coverage["required_h"], coverage["required_layers"])):
        raise ValueError("fit rows and coverage cells must be lists")
    for row in map(obj, h_rows):
        key = (row.get("b"), row.get("k"), row.get("bucket"))
        if not h_key(key) or key not in expected_h or key in h_counts:
            violations["h_key_or_duplicate"] += 1
            continue
        samples, ells = row.get("samples_ms"), row.get("ells")
        if (not isinstance(samples, list) or not samples or not all(map(nonnegative, samples))
                or not isinstance(ells, list) or len(ells) != len(samples)
                or any(ell is not None and (type(ell) is not int or ell < 0) for ell in ells)
                or any(k.lower().startswith("q") for k in row)):
            violations["h_samples_or_ells"] += 1
            continue
        null_ells += ells.count(None)
        h_counts[key] = len(samples)
    for row in map(obj, layers):
        key = (row.get("w"), row.get("path"), row.get("g"))
        if not layer_key(key) or key in layer_counts:
            violations["layer_key_or_duplicate"] += 1
            continue
        n, matrix, vector = (row.get(k) for k in ("n", "xtx", "xty"))
        lo, hi = row.get("span_min"), row.get("span_max")
        if (type(n) is not int or n < 1 or not isinstance(matrix, list) or len(matrix) != 2
                or not all(isinstance(values, list) and len(values) == 2 and all(map(nonnegative, values)) for values in matrix)
                or not isinstance(vector, list) or len(vector) != 2 or not all(map(nonnegative, vector))
                or not (nonnegative(lo) and nonnegative(hi) and lo <= hi)):
            violations["layer_regression_fields"] += 1
            continue
        # The installed model has two features and ridge (0, 128**2).
        a, b = matrix[0]
        c, d = matrix[1]
        det = a * (d + 128**2) - b * c
        if n >= 4 and not (a > 0 and near(b, c) and number(det) and det > 0):
            violations["layer_regression_singular"] += 1
            continue
        layer_counts[key] = n
    missing_h = sorted(key for key in expected_h if h_counts.get(key, 0) < 4)
    missing_layers = sorted(key for key in expected_layers if layer_counts.get(key, 0) < 50)
    run = fit.get("run")
    gates = dict(
        fit_schema=type(fit.get("schema")) is int and fit["schema"] == 4 and matches(header, {"schema": 4}),
        fit_identity=matches(draft, dict(tp=1, page_size=1, capture_bs=BATCHES, bonus=True, lookahead=True,
                                        attention_backend="triton", disable_cuda_graph=False))
                     and matches(target, dict(backend="npu", tp=2, page_size=128, cuda_graph_bs=BATCHES,
                                              kv_buckets=BUCKETS, verify_capture=[65, 33, 17, 9, 5, 3, 1],
                                              attention_backend="ascend", disable_cuda_graph=False))
                     and ("verify_mode" not in target if mode == "greedy" else target.get("verify_mode") == "rpd"),
        fit_plan=matches(obj(plan["inputs"]), FIT_INPUTS) and plan.get("fingerprint") == FIT_FINGERPRINT
                 and isinstance(run, str) and FIT_FINGERPRINT in run.split("_")
                 and matches(header, {k: FIT_INPUTS[k] for k in ("ks", "widths", "buckets", "max_depth")}),
        fit_coverage=declared_h == expected_h and declared_layers == expected_layers and coverage.get("skipped") == []
                     and not missing_h and not missing_layers and not violations
                     and not any(k.lower().startswith("q") for k in fit),
    )
    return {"name": path.name, "sha256": sha(data), "schema": fit.get("schema"), "cells": len(h_rows),
            "layer_cells": len(layers), "run": run, "plan": plan, "target_mode": target.get("verify_mode"),
            "draft_lookahead": draft.get("lookahead"), "required_h": len(declared_h),
            "required_layers": len(declared_layers), "skipped": coverage.get("skipped"),
            "missing_h": missing_h, "missing_layers": missing_layers, "h_null_ells": null_ells,
            "h_min_samples": min(h_counts.values(), default=None),
            "layer_min_samples": min((layer_counts.get(k, 0) for k in expected_layers), default=None),
            "layer_samples_by_cell": [[*key, n] for key, n in sorted(layer_counts.items())],
            "violations": violations}, gates


def layer_review(group, layers):
    counts, violations, paths = Counter(), Counter(), Counter()
    by_depth = defaultdict(Counter)
    for rnd in group:
        rows = layers.get(round_key(rnd), ())
        counts["matched_rounds" if rows else "rounds_without_layers"] += 1
        for row in rows:
            counts["rows"] += 1
            layer, fitted = row.get("layer"), row.get("fit")
            if type(layer) is not int or layer < 1 or type(fitted) is not bool:
                violations["layer_or_fit_type"] += 1
            if type(layer) is int and layer >= 1:
                bucket = by_depth[layer]
                bucket["rows"] += 1
                bucket["fitted"] += fitted is True
                for key in ("pred_ms", "residual"):
                    bucket[f"{key}_null"] += key in row and row[key] is None
                if layer == 1 and fitted is not False:
                    violations["layer1_fit"] += 1
                if layer >= 2 and fitted is True:
                    counts["fitted_layer_ge_2"] += 1
            for key in ("pred_ms", "residual"):
                if key not in row or row[key] is not None and not number(row[key]):
                    violations[f"{key}_invalid"] += 1
            for key in ("depth_ratio", "depth_src"):
                if key in row:
                    violations[key] += 1
            path = row.get("path")
            paths[path if isinstance(path, str) else "missing_or_invalid"] += 1
            if path not in ("graph", "eager", "unknown"):
                violations["path_invalid"] += 1
            for key in ("last", "lookahead", "lookahead_mode", "post_reply"):
                if type(row.get(key)) is not bool:
                    violations[f"{key}_invalid"] += 1
            if fitted is True and (rnd.get("excluded") or row.get("post_reply")
                                   or row.get("lookahead_mode") and not row.get("lookahead")
                                   or row.get("lookahead") and row.get("last")):
                violations["ineligible_fit"] += 1
    return dict(counts=counts, by_layer=dict(sorted(by_depth.items())), paths=paths, violations=violations)


def round_review(group, late, mode):
    counts, violations, k_modes = Counter(), Counter(), Counter()
    depth = defaultdict(lambda: {"rows": 0, "observed": 0, "expected": 0.0})
    rids_seen = set()
    for row in group:
        rids_seen.update(row["rids"])
        counts["rounds"] += 1
        counts["hold_invalid"] += bool(row.get("hold_invalid"))
        excluded = row.get("excluded")
        if type(excluded) is not bool:
            violations["excluded_type"] += 1
        pieces = [row.get(f"policy_{part}_ms") for part in ("recv", "plan", "grow", "final_k")]
        if not (nonnegative(row.get("close_ms")) and type(row.get("close_deferred")) is bool
                and all(map(nonnegative, pieces)) and near(sum(pieces), row.get("policy_ms"))):
            violations["policy_or_close_timing"] += 1
        report = late.get(round_key(row))
        if mode == "greedy" and row.get("rpd_masks") is not None:
            violations["greedy_inline_mask"] += 1
        if excluded:
            counts["excluded"] += 1
            continue
        hold, queued = row.get("hold_ms"), row.get("q_ms")
        if not (nonnegative(hold) and nonnegative(queued)):
            counts["hold_invalid"] += 1
            continue
        if not number(row.get("t_k_us")) or not near(hold + queued, row["t_k_us"] / 1000):
            violations["hold_q_total"] += 1
        counts["scored"] += 1
        k_modes[str(row.get("k_mode"))] += 1
        details = row.get("e_rid")
        if not isinstance(details, dict) or not details:
            counts["scored_missing_e_rid"] += 1
            continue
        counts["diagnostic_rows"] += len(details)
        if not set(details) <= set(row["rids"]):
            violations["e_rid_identity"] += 1
        masses, expected, accepts = [], [], []
        for detail in details.values():
            if not isinstance(detail, dict):
                violations["invalid_e_rid"] += 1
                continue
            need = ("c", "mass_d", "gain_last", "gain_next", "depth_cap")
            for key in need:
                counts[f"{key}_present"] += key in detail
                counts[f"{key}_null"] += key in detail and detail[key] is None
            if any(key not in detail for key in need):
                violations["missing_field"] += 1
            c, mass_d, last, nxt, cap = (detail.get(key) for key in need)
            dep, nacc = detail.get("depth"), detail.get("accepted")
            if not (type(dep) is int and type(cap) is int and 1 <= dep <= cap and type(nacc) is int and 0 <= nacc <= dep):
                violations["depth_or_accepted"] += 1
                continue
            if not (nonnegative(c) and isinstance(mass_d, list) and len(mass_d) == dep and all(map(nonnegative, mass_d))):
                violations["mass_or_c"] += 1
                continue
            if not ((dep == 1 and last is None) or (dep > 1 and nonnegative(last))):
                violations["gain_last"] += 1
            if not nonnegative(nxt):
                violations["gain_next"] += 1
            counts["depth_capped"] += dep == cap
            counts["path_ok"] += detail.get("path_ok") is True
            mass = sum(mass_d)
            masses.append(mass)
            expected.append(c * mass)
            accepts.append(nacc)
            kmass = detail.get("mass_k")
            if isinstance(kmass, dict) and number(kmass.get(str(row.get("k")))):
                counts["mass_k_comparable"] += 1
                counts["mass_k_not_equal_round_k"] += not near(mass, kmass[str(row["k"])])
            for j, value in enumerate(mass_d, start=1):
                agg = depth[j]
                agg["rows"] += 1
                agg["observed"] += int(nacc >= j)
                agg["expected"] += c * value
        if type(row.get("e_n")) is int and len(expected) == len(details) == row["e_n"]:
            counts["mass_and_prediction_comparable"] += 1
            if not (near(sum(masses), row.get("e_mass")) and near(sum(expected), row.get("e_pred"))
                    and sum(accepts) == row.get("accepted")):
                violations["round_mass_or_e_pred_or_accepted"] += 1
        else:
            violations["e_n_coverage"] += 1
        inline = {rid: value for rid, value in details.items()
                  if isinstance(value, dict) and any(k in value for k in ("rpd_ok", "accept_k", "rpd_valid"))}
        counts["rpd_inline_mask_rows"] += len(inline)
        if mode == "greedy" and inline:
            violations["greedy_inline_mask"] += 1
        if mode == "rpd":
            # Late reports can be absent even away from a block's tail. Do
            # not infer a zero mask or count it as validated in that case.
            counts["rpd_scored_late_matched" if report is not None else "rpd_scored_late_unmatched"] += 1
        if report is not None:
            masks = report.get("e_rid")
            if (report.get("tau") != TAU or any(report.get(k) != row.get(k) for k in ("b", "k", "w", "excluded", "e_pred", "e_mass"))
                    or not isinstance(masks, dict) or not masks or not set(masks) <= set(details)):
                violations["rpd_late_alignment"] += 1
            if isinstance(masks, dict):
                used = sum(isinstance(value, dict) and value.get("accept_k") is not None for value in masks.values())
                counts["rpd_late_mask_rows"] += used
                if type(report.get("rpd_masks")) is not int or used != report["rpd_masks"]:
                    violations["rpd_late_mask_count"] += 1
                for rid, mask in masks.items():
                    detail = details.get(rid)
                    if not isinstance(mask, dict) or not isinstance(detail, dict) or any(mask.get(k) != detail.get(k) for k in ("depth", "accepted")):
                        violations["rpd_late_detail"] += 1
                    counts["rpd_ok_false"] += isinstance(mask, dict) and mask.get("rpd_ok") is False
                    if not isinstance(mask, dict) or mask.get("rpd_ok") is not True:
                        violations["rpd_late_ok"] += 1
        counts["rpd_ok_false"] += sum(value.get("rpd_ok") is False for value in inline.values())
        if any(value.get("rpd_ok") is not True for value in inline.values()):
            violations["rpd_inline_ok"] += 1
    for values in depth.values():
        denom = values["expected"]
        values["observed_over_expected"] = values["observed"] / denom if denom > 0 else None
    counts["unique_rids"] = len(rids_seen)
    return dict(counts=counts, violations=violations, k_modes=k_modes, depth_calibration=dict(sorted(depth.items())))


def verify_review(rows):
    paths, timings, invalid = Counter(), defaultdict(list), 0
    for row in rows:
        path = row.get("path")
        paths[path if isinstance(path, str) else "missing_or_invalid"] += 1
        for key in TIMINGS:
            value = row.get(key)
            if type(value) is int and value >= 0:
                timings[key].append(value)
            elif value is not None or (key in TIMINGS[:2] and path != "fallback") or key not in row:
                invalid += 1
        wall, ready = (row.get(key) for key in TIMINGS[:2])
        invalid += number(wall) and number(ready) and wall < ready
    return dict(paths=paths, invalid=invalid,
                metrics={key: dict(n=len(timings[key]), median_us=statistics.median(timings[key]) if timings[key] else None)
                         for key in TIMINGS})


def cleanup_review(raw, mode, issues):
    def decimal(value):
        return isinstance(value, str) and re.fullmatch(r"[0-9]+", value) is not None

    def has_identity(value):
        return (isinstance(value, dict)
                and all(decimal(value.get(key)) for key in ("pid", "sid", "container_nspid", "uid", "start_ticks"))
                and isinstance(value.get("nspid"), str)
                and isinstance(value.get("boot_id"), str) and bool(value["boot_id"]))

    directory = raw / mode
    roles = {}
    for role in ("draft", "target"):
        owner = record(issues, f"{mode} {role} identity", lambda: load(directory / f"{role}-identity.json"), {})
        receipt = record(issues, f"{mode} {role} cleanup", lambda: load(directory / f"{role}-cleanup.json"), {})
        baseline = record(issues, f"{role} precheck baseline", lambda: (raw / "sync" / f"precheck-{role}.txt").read_bytes())
        pre = record(issues, f"{mode} {role} precheck", lambda: (directory / f"precheck-{role}.txt").read_bytes())
        post = record(issues, f"{mode} {role} postcheck", lambda: (directory / f"postcheck-{role}.txt").read_bytes())
        signals = receipt.get("signals")
        verified = (has_identity(owner) and owner["pid"] == owner["sid"]
                    and isinstance(signals, list) and all(
                        has_identity(member) and matches(member, {k: owner[k] for k in ("sid", "boot_id")})
                        and int(member["start_ticks"]) >= int(owner["start_ticks"])
                        and type(member.get("term_exit")) is int and member["term_exit"] in (0, 4)
                        for member in signals))
        roles[role] = dict(live_session_members=receipt.get("live_session_members"),
                           term_attempts=len(signals) if isinstance(signals, list) else None,
                           term_receipts_valid=verified,
                           precheck_matches=bool(baseline and pre == baseline),
                           postcheck_matches=bool(baseline and post == baseline))
    error_path = directory / "cleanup-errors.txt"
    error = record(issues, f"{mode} cleanup errors", lambda: text(error_path).strip()) if error_path.exists() else ""
    passed = error == "" and all(data["live_session_members"] == [] and data["term_receipts_valid"]
                                and data["precheck_matches"] and data["postcheck_matches"] for data in roles.values())
    return {"roles": roles, "errors": error, "passed": passed}


def hash_receipt(path, filename):
    parts = text(path).split()
    if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]) or parts[1] != filename:
        raise ValueError(f"invalid checksum receipt {path.name}")
    return parts[0]


def launch_review(accept, directory, draft, target, fit, issues):
    parsed, capture_recorded = {}, {}
    config_ok = True
    for role, data, capacity, attention, mm, page in (
        ("draft", draft, "64", "triton", "triton_attn", "1"),
        ("target", target, "65", "ascend", "ascend_attn", "128"),
    ):
        path = directory / f"parse-{role}.txt"
        values = record(issues, f"{directory.name} {path.name}",
                        lambda: dict(line.split(maxsplit=1) for line in text(path).splitlines()), {})
        parsed[role] = values
        expected = dict(role=role, capacity=capacity, widths=",".join(map(str, TIERS)),
                        ks=",".join(map(str, TIERS)), depth="8", steps="8", cuda_graph_bs="1,2,4",
                        page=page, attention_backend=attention, mm_attention_backend=mm)
        config_ok &= values == expected
        path = directory / f"{role}-captures.txt"
        saved = record(issues, f"{directory.name} {path.name}", lambda: text(path), "")
        captures = {tuple(map(int, cell)) for cell in accept.CAPTURE.findall(saved)}
        capture_recorded[role] = sorted(captures)
    name = MODES[directory.name]
    digest = record(issues, f"{directory.name} fit checksum", lambda: hash_receipt(directory / f"{name}.sha256", name))
    receipt = record(issues, f"{directory.name} fit coverage receipt", lambda: load(directory / "fit-coverage.json"), {})
    complete = record(issues, f"{directory.name} block completion", lambda: bool(text(directory / "block-complete.txt").strip()), False)
    return dict(parsed=parsed, capture_records=capture_recorded, fit_checksum=digest, fit_receipt=receipt,
                checks=dict(launch_config=config_ok,
                            capture_records=all(capture_recorded[role] and set(map(tuple, capture_recorded[role])) == data["captures"]
                                                for role, data in (("draft", draft), ("target", target))),
                            fit_receipt=bool(digest and digest == fit.get("sha256") and matches(
                                receipt, dict(schema=4, run=fit["run"], fingerprint=FIT_FINGERPRINT, sha256=digest,
                                              draft_health=200, target_health=200, required_h=90, required_layers=18,
                                              draft_captures=len(draft["captures"]), target_captures=len(target["captures"])))),
                            block_complete=complete))


def mode_review(accept, raw, mode, ids, issues):
    directory = raw / mode
    name = MODES[mode]
    draft = scan_log(accept, directory / "draft.log", mode, "draft", issues)
    target = scan_log(accept, directory / "target.log", mode, "target", issues)
    dfit, tfit = draft["fit"], target["fit"]
    fit, gates = record(issues, f"{mode} fit", lambda: fit_review(directory / name, mode), ({}, {}))
    for key in ("fit_schema", "fit_identity", "fit_plan", "fit_coverage"):
        gates.setdefault(key, False)
    extras = sorted(path.name for path in directory.glob("clear_tree_fit*.json") if path.name != name)
    gates["isolated_fit_file"] = bool(fit and not extras)
    run = fit.get("run")
    states = tfit["states"]
    session = states.get("0", {}).get("session")
    session_prefix = "".join(ch for ch in session if ch.isalnum() or ch == "_") if isinstance(session, str) else ""
    commits = [row for row in tfit["commits"] if row.get("ready") is True]
    gates["prefit"] = bool(isinstance(run, str) and session_prefix
                           and run.startswith(f"{session_prefix}_{FIT_FINGERPRINT}_") and run.rsplit("_", 1)[-1].isdigit()
                           and len(dfit["writes"]) == 1 and dfit["writes"][0][1:] == (name, fit.get("cells"))
                           and sum(file == name and n < dfit["writes"][0][0] for n, file in dfit["absent"]) == 1
                           and tfit["driver"] == ["ready"] and tfit["started"] == 1 and not (dfit["flat"] or tfit["flat"])
                           and set(states) == {"0", "1"} and states["0"].get("session")
                           and all((r.get("state"), r.get("reason"), r.get("run"), r.get("session"))
                                   == ("ready", "prefit", run, states["0"]["session"]) for r in states.values())
                           and len(commits) == 1 and matches(commits[0], dict(run=run, reason="ready", missing_h=[], missing_layers=[]))
                           and (dfit["rpd_ack"] == [(str(TAU), name)] if mode == "rpd" else not dfit["rpd_ack"]))
    target_ntpb = {1} | {k + 1 for k in TIERS}
    expected_target = set(itertools.product(BATCHES, target_ntpb))
    expected_draft = set(itertools.product(BATCHES, target_ntpb | set(TIERS))) - {(4, 65)}
    missing_capture = {"draft": sorted(expected_draft - draft["captures"]),
                       "target": sorted(expected_target - target["captures"])}
    gates["capture"] = (not any(missing_capture.values()) and (4, 65) not in draft["captures"] and
                        all({b for b, _ in data["captures"]} == set(BATCHES) for data in (draft, target)))
    gates["flush"] = passed_flush((target["flush_tp"], target["flush_http"]), 2)
    launch = launch_review(accept, directory, draft, target, fit, issues)
    gates.update(launch.pop("checks"))
    verify = verify_review(target["events"]["t_verify"])
    paths = verify["paths"]
    gates["graph_verify"] = bool(paths.get("graph") and set(paths) <= {"graph", "eager"})
    gates["timing"] = bool(paths) and not verify["invalid"] and all(
        (verify["metrics"][k]["n"] > 0) == (mode == "rpd") for k in TIMINGS[2:])
    rounds = draft["events"]["d_round"]
    groups, assignment = accept.assign_trial_rounds([(None, row) for row in rounds], ids)
    keys = Counter(round_key(row) for row in rounds)
    duplicate_keys = sum(n - 1 for n in keys.values())
    gates["round_assignment"] = not duplicate_keys and not any(assignment.get(k) for k in ("ambiguous_id", "mixed_trial", "mixed_untracked"))
    layers = defaultdict(list)
    for row in draft["events"]["d_layer"]:
        layers[round_key(row)].append(row)
    late_rows = draft["events"]["d_rpd"]
    late = {round_key(row): row for row in late_rows}
    gates["mask_evidence"] = not late if mode == "greedy" else bool(late and len(late) == len(late_rows))
    trial_results = {}
    for trial, tagged in groups.items():
        group = [row for _, row in tagged]
        ledger = record(issues, f"trial {trial} ledger", lambda: round_review(group, late, mode), {})
        ledger["layers"] = record(issues, f"trial {trial} layers", lambda: layer_review(group, layers), {})
        trial_results[trial] = ledger
    formal_sessions = sorted({row["session_id"] for group in groups.values() for _, row in group})
    windows = draft["windows"]
    bonus = dict(reasons=Counter(reason for reason, _ in windows),
                 tail="flushed" if windows and windows[-1][0] == "idle" else "missing",
                 **{k: sum(counters.get(k, 0) for _, counters in windows) if windows else None for k in accept.BONUS_KEYS})
    gates["bonus"] = bool(bonus["tail"] == "flushed" and bonus["opt_bonus_graph"] and bonus["opt_bonus_requests"]
                          and bonus["opt_bonus_graph"] == bonus["opt_bonus_forwards"] and bonus["opt_bonus_eager"] == 0)
    cleanup = cleanup_review(raw, mode, issues)
    gates["cleanup"] = cleanup["passed"]
    gates["parse"] = all(data["present"] and not data["errors"] for data in (draft, target))
    retention = {}
    gates["greedy_fit_retained"] = True
    if mode == "rpd":
        before = raw / "greedy" / MODES["greedy"]
        during = directory / "greedy-fit-during-rpd.json"
        retention = {
            "before": record(issues, "greedy archived fit digest", lambda: sha(before.read_bytes())),
            "during": record(issues, "rpd retained greedy fit digest", lambda: sha(during.read_bytes())),
            "during_receipt": record(issues, "rpd retained greedy fit receipt",
                                     lambda: hash_receipt(during.with_name(during.name + ".sha256"), during.name)),
            "after": record(issues, "rpd post-cleanup greedy fit digest",
                            lambda: hash_receipt(directory / "greedy-fit-after-rpd.sha256", before.name)),
        }
        gates["greedy_fit_retained"] = bool(retention["before"] and len(set(retention.values())) == 1)
    return dict(
        draft_captures=sorted(draft["captures"]), target_captures=sorted(target["captures"]),
        missing_capture_shapes=missing_capture, allowed_missing_draft_shape=[4, 65],
        flush_tp=target["flush_tp"], flush_http=target["flush_http"], draft_fit=dfit, target_fit=tfit,
        fit_file=fit, fit_sha256=fit.get("sha256"), extra_fit_files=extras,
        target_verify_paths=paths, target_timing=verify["metrics"],
        target_timing_invalid=verify["invalid"], late_rpd_rows=len(late_rows),
        late_rpd_duplicate_keys=len(late_rows) - len(late), round_assignment=assignment,
        formal_sessions=formal_sessions, duplicate_round_keys=duplicate_keys, unjoined_layer_rows=sum(len(rows) for key, rows in layers.items() if key not in keys),
        unjoined_late_rpd_rows=sum(key not in keys for key in late),
        parse_errors={"draft": draft["errors"], "target": target["errors"]},
        logs_present={"draft": draft["present"], "target": target["present"]}, bonus_windows=bonus,
        launch=launch, cleanup=cleanup, greedy_fit_retention=retention, trials=trial_results,
    ), gates


def sync_review(raw, issues):
    sync = raw / "sync"
    manifests = {role: record(issues, f"sync {role} manifest", lambda: (sync / f"{role}.manifest").read_bytes())
                 for role in ("head", "draft", "npu")}
    data = record(issues, "sync RSTeller identity", lambda: obj(load(sync / "data-identity.json")["RSTeller"]))
    rpd_data = record(issues, "RPD RSTeller identity", lambda: obj(load(sync / "data-identity-rpd.json")["RSTeller"]))
    before_package = sync / "npu-package-before.tar.gz"
    before_digest = record(issues, "NPU pre-sync package digest", lambda: sha(before_package.read_bytes()))
    before_receipt = record(issues, "NPU pre-sync package receipt", lambda: text(sync / "npu-package-before.remote-sha256").split())

    def check_hash(value):
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None

    def old_fits(role):
        receipt = sync / f"{role}-old-fit.sha256"
        rows = [line.split() for line in text(receipt).splitlines() if line.strip()]
        if any(len(pair) != 2 for pair in rows):
            raise ValueError(f"malformed old fit receipt {receipt.name}")
        names = {*MODES.values(), "clear_tree_fit.bak.json", "clear_tree_fit.bak2.json"}
        if len(rows) != len(names) or {name for _, name in rows} != names:
            raise ValueError(f"unexpected old fit names in {receipt.name}")
        checks = {}
        for digest, name in rows:
            backup = sync / f"{role}-old-fit" / name
            checks[name] = (not backup.exists() if digest == "ABSENT" else check_hash(digest)
                            and sha(backup.read_bytes()) == digest)
        return checks

    backups = {role: record(issues, f"sync {role} old fits", lambda: old_fits(role), {}) for role in ("draft", "npu")}
    gates = dict(package_manifests=bool(manifests["head"] and manifests["head"] == manifests["draft"] == manifests["npu"]),
                 old_fit_backups=all(values and all(values.values()) for values in backups.values()),
                 pre_sync_package=bool(before_digest and before_receipt and len(before_receipt) == 2
                                       and before_receipt[0] == before_digest and Path(before_receipt[1]).name == before_package.name),
                 data_identity=isinstance(data, dict) and data == rpd_data and data.get("selected") == N_ITEMS
                 and check_hash(data.get("file_sha256")) and check_hash(data.get("selected_rows_sha256"))
                 and isinstance(data.get("ordered_image_sha256"), list) and len(data["ordered_image_sha256"]) == N_ITEMS
                 and all(map(check_hash, data["ordered_image_sha256"])))
    return dict(manifest_sha256={role: sha(value) if value else None for role, value in manifests.items()},
                old_fit_backups=backups, npu_pre_sync_sha256=before_digest,
                data_identity=data, rpd_data_identity=rpd_data), gates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--accept", type=Path, required=True, help="path to the offline accept_logs.py helper")
    args = parser.parse_args()
    raw = args.run / "raw"
    issues = []
    result = dict(run_id=RUN_ID, commit=COMMIT, issues=issues, trials={}, modes={})
    checks = dict(issues=issues, run={}, trials={}, modes={})
    meta = record(issues, "meta.json", lambda: load(args.run / "meta.json"), {})
    result["plan_mismatches"] = sorted(key for key, value in PLAN.items() if not matches(meta, {key: value}))
    checks["run"]["plan"] = not result["plan_mismatches"] and not meta.get("voided")
    result["extra_trial_dirs"] = sorted(p.name for p in raw.glob("*") if p.is_dir()
                                        and p.name not in {trial["id"] for trial in TRIALS}
                                        and (p.name.isdecimal() or list(p.glob("client.*.jsonl"))))
    checks["run"]["no_extra_trials"] = not result["extra_trial_dirs"]
    result["sync"], sync_gates = sync_review(raw, issues)
    checks["run"].update(sync_gates)
    def load_helper():
        spec = importlib.util.spec_from_file_location("accept_logs", args.accept)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot import {args.accept}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    accept = record(issues, "offline log parser", load_helper)
    ids = defaultdict(dict)
    for spec in TRIALS:
        trial = spec["id"]
        row, ids[spec["mode"]][trial], gates = record(
            issues, f"trial {trial} evidence", lambda: client(accept, raw, spec, issues),
            (dict(spec, ordered_items_sha256=None), set(), {"review_available": False}),
        )
        result["trials"][trial] = row
        checks["trials"][trial] = gates
    fingerprints = [row["ordered_items_sha256"] for row in result["trials"].values()]
    checks["run"]["same_ordered_items"] = bool(all(fingerprints) and len(set(fingerprints)) == 1)
    all_ids = [rids for group in ids.values() for rids in group.values()]
    checks["run"]["no_reused_response_ids"] = len(set().union(*all_ids)) == N_ITEMS * len(TRIALS)

    for mode in MODES:
        result["modes"][mode], checks["modes"][mode] = record(
            issues, f"{mode} evidence", lambda: mode_review(accept, raw, mode, ids[mode], issues),
            ({}, {"review_available": False}),
        ) if accept is not None else ({}, {"review_available": False})
    for spec in TRIALS:
        trial, mode = spec["id"], spec["mode"]
        ledger = result["modes"][mode].get("trials", {}).get(trial, {})
        counts, layers = ledger.get("counts", {}), ledger.get("layers", {})
        checks["trials"][trial].update(
            ledger=bool(counts.get("scored") and counts["scored"] == counts.get("mass_and_prediction_comparable")
                        and counts.get("unique_rids") == N_ITEMS and not ledger.get("violations")
                        and not counts.get("hold_invalid") and not counts.get("scored_missing_e_rid")),
            layers=bool(layers.get("by_layer", {}).get(1, {}).get("rows")
                        and layers.get("counts", {}).get("fitted_layer_ge_2") and not layers.get("violations")),
            rpd_evidence=bool(counts.get("scored") and not counts.get("rpd_ok_false") and
                              (counts.get("rpd_scored_late_matched") and counts.get("rpd_late_mask_rows")
                               if mode == "rpd" else not counts.get("rpd_inline_mask_rows"))),
        )
    passed = not issues and all(checks["run"].values()) and all(
        all(v.values()) for group in ("trials", "modes") for v in checks[group].values())
    checks["passed"] = passed
    checks["status"] = "passed" if passed else "blocked" if issues else "failed"
    result["status"] = checks["status"]
    for name, data in (("summary", result), ("checks", checks)):
        (args.run / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
