#!/usr/bin/env python3
"""Review 28-d's saved artifacts only; never launch services, alter raw/, or seal.

Outputs are exclusive-create. Use --output-dir with an existing empty directory
for a second review. Exit 0 means verified, 1 means incomplete/unverified, and 2
means the report could not be written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
REFERENCE = RUN.with_name("2026-11-28-c")
WATERCAPTION_REFERENCE = RUN.with_name("2026-11-27-b")
RUN_ID = "2026-11-28-d"
COMMIT = "1a15ad1ca5331e15712fca337bf5fb7232ba3922"
FIT_FILES = {"greedy": "clear_tree_fit.json", "rpd": "clear_tree_fit.rpd.json"}
ROLES = ("draft", "target")
DATASETS = ("RSTeller", "WaterCaption")
N = 100
TAU = 0.4
BATCHES = (1, 2, 4)
TARGET_SHAPES = {(b, k) for b in BATCHES for k in (1, 3, 5, 9, 17, 33, 65)}
DRAFT_SHAPES = (TARGET_SHAPES | {(b, w) for b in BATCHES for w in (2, 4, 8, 16, 32, 64)}) - {(4, 65)}

# Match the saved log formats used by the earlier review and accept_logs.py.
TP = re.compile(r" TP(\d+)\]")
FLUSH = re.compile(r"TP(\d+)\].*Cache flushed successfully")
HTTP_FLUSH = re.compile(r'POST /flush_cache HTTP/1\.1" (\d+)')
CAPTURE = re.compile(r"Capturing: bs=(\d+), ntpb=(\d+)")
LEDGER = re.compile(r"\[CLEAR ledger (d_round|d_layer|d_rpd|t_verify)\]")
REFIT = re.compile(r"\[CLEAR\] (?:prefit started|prefit driver finished|prefit finished|wrote fit file)|\[CLEAR prefit_commit\]")
BAD_FIT = re.compile(r"no fit file|prefit required|fit file.*unreadable|rpd_flat|W/K use the flat table")
FIT_STASH = re.compile(r"\[CLEAR\] stashed fit file (.+) for the Target header check")
FIT_CAP = re.compile(r"\[CLEAR\] prefit layer weight capped cap=(\d+) cells=(\d+)")
FIT_INSTALL = re.compile(r"\[CLEAR\] installed (\d+) hold cells from the fit file")
RPD_ACK = re.compile(r"\[CLEAR\] target verifies with RPD \(tau=([^)]*)\).*fit file (.+)")
RPD_PATH = re.compile(r"Speculative RPD verify path: (\S+)")
RPD_TIMINGS = ("rpd_build_us", "rpd_stats_us", "rpd_select_us", "rpd_mask_us")
COUNTERS = ("completion_tokens", "spec_accept_token_num", "spec_draft_token_num", "spec_fallback_ct")


def obj(value):
    return value if isinstance(value, dict) else {}


def finite(value):
    return type(value) in (float, int) and (type(value) is int or math.isfinite(value))


def count(value):
    return finite(value) and value >= 0 and int(value) == value


def sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def parse_object(text, *, suffix=False):
    def reject(token):
        raise ValueError(f"non-finite JSON constant: {token}")

    def parse_float(value):
        result = float(value)
        return result if math.isfinite(result) else reject(value)

    decoder = json.JSONDecoder(parse_constant=reject, parse_float=parse_float)
    value = decoder.raw_decode(text.lstrip())[0] if suffix else decoder.decode(text.strip())
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


class Evidence:
    def __init__(self, run):
        self.run = run
        self.errors = []

    def issue(self, path, message):
        self.errors.append(f"{path.relative_to(self.run) if path.is_relative_to(self.run) else path.name}: {message}")

    def read(self, path, reader, default):
        try:
            return reader(path)
        except (OSError, ValueError, csv.Error) as exc:
            self.issue(path, exc)
            return default

    def text(self, path):
        return self.read(path, lambda p: p.read_text(encoding="utf-8"), "")

    def json(self, path):
        data = self.text(path)
        try:
            return parse_object(data) if data else {}
        except ValueError as exc:
            self.issue(path, exc)
            return {}


def flush_counts(text):
    return {"tp": dict(Counter(FLUSH.findall(text))), "http": dict(Counter(HTTP_FLUSH.findall(text)))}


def expected_flush(ordinal):
    return {"tp": {"0": ordinal, "1": ordinal}, "http": {"200": ordinal}}


def client(evidence, directory, mode, concurrency, ordinal):
    files = sorted(directory.glob("client.*.jsonl"))
    path = files[0] if len(files) == 1 else None
    rows, bad_rows = [], 0
    if path is not None:
        for n, line in enumerate(evidence.text(path).splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(parse_object(line))
            except ValueError as exc:
                bad_rows += 1
                if bad_rows <= 3:
                    evidence.issue(path, f"line {n}: {exc}")
    else:
        evidence.issue(directory, f"expected one client JSONL, found {len(files)}")
    stdout = evidence.text(directory / "client.stdout")
    exit_code = evidence.text(directory / "exit-code.txt").strip()
    receipt = flush_counts(evidence.text(directory / "target-flush.txt"))
    manifest = evidence.json(path.with_suffix(".run.json")) if path else {}

    def csv_rows(p):
        with p.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, strict=True))

    sheets = evidence.read(path.with_suffix(".csv"), csv_rows, []) if path else []
    sheet = sheets[0] if len(sheets) == 1 else {}
    ok = [r for r in rows if not r.get("error")]
    ids = [obj(r.get("response_meta_info")).get("id") for r in ok]
    ids = [rid for rid in ids if isinstance(rid, str) and rid]
    coverage = {key: sum(count(row.get(key)) for row in ok) for key in COUNTERS}
    tokens, accepted, drafted, fallback = (
        sum(int(row[key]) for row in ok) if ok and coverage[key] == len(ok) else None for key in COUNTERS
    )
    printed = re.findall(r"\bsummary items=(\d+)/(\d+)\s+toks=(\d+)\s+wall=([\d.]+)s", stdout)
    wall = evidence.read(directory / "client.stdout", lambda _: float(printed[0][3]), None) if len(printed) == 1 else None
    stats = obj(manifest.get("summary"))
    snapshot = obj(manifest.get("server"))
    settings = obj(snapshot.get("settings"))
    expected_settings = dict(speculative_algorithm="CLEAR_TREE_REMOTE", device="npu", tp_size=2, page_size=128,
                             disable_cuda_graph=False, cuda_graph_bs=list(BATCHES), attention_backend="ascend",
                             speculative_verify_mode=mode, speculative_num_steps=8, speculative_num_draft_tokens=65)
    if mode == "rpd":
        expected_settings["speculative_rpd_tau"] = TAU
    items = [(r.get("image"), r.get("question"), r.get("ground_truth")) for r in rows]
    items_hash = hashlib.sha256(json.dumps(items, ensure_ascii=False, sort_keys=True).encode()).hexdigest() if rows else None
    warmup_flush = re.findall(r"=== warmup \(first item, discarded\) ===|warmup ok|flush_cache ->|=== main run ===", stdout)
    rate = stats.get("output_tok_s")
    gates = dict(
        artifacts=bool(path and len(sheets) == 1 and not bad_rows
                       and manifest.get("run_id") == path.name.removeprefix("client.").removesuffix(".jsonl")
                       and Path(str(manifest.get("results_jsonl"))).name == path.name
                       and Path(str(manifest.get("summary_csv"))).name == path.with_suffix(".csv").name),
        exit=exit_code == "0",
        requests=len(rows) == len(ok) == len(ids) == len(set(ids)) == stats.get("n_ok") == stats.get("n_total") == N,
        ordered_items=[r.get("global_index") for r in rows] == list(range(N))
                      and all(isinstance(image, str) and image and truth for image, _, truth in items),
        traffic=(re.findall(r"traffic=open\s+request_rate=(\S+)\s+max_concurrency=(\d+)\s+max_new_tokens=(\d+)", stdout)
                 == [("inf", str(concurrency), "1024")]
                 and (sheet.get("traffic"), sheet.get("request_rate"), sheet.get("max_concurrency")) == ("open", "inf", str(concurrency))),
        target_config=snapshot.get("status") == "available" and all(settings.get(k) == v for k, v in expected_settings.items()),
        timing=bool(tokens and finite(wall) and wall > 0 and finite(rate) and rate > 0
                    and stats.get("total_tokens") == tokens and sheet.get("Total Generated Tokens") == str(tokens)
                    and printed[0][:3] == (str(len(ok)), str(len(rows)), str(tokens))
                    and abs(tokens / rate - wall) <= 0.000501),
        speculation=bool(tokens and drafted and accepted is not None and fallback == 0
                         and all(row[COUNTERS[1]] <= min(row[COUNTERS[0]], row[COUNTERS[2]])
                                 and 0 < row[COUNTERS[0]] <= 1024 for row in ok)),
        flush_client=(warmup_flush == ["=== warmup (first item, discarded) ===", "warmup ok", "flush_cache ->", "=== main run ==="]
                      and "flush_cache failed:" not in stdout),
        flush_receipt=receipt == expected_flush(ordinal),
    )
    report = dict(requests=len(rows), ok=len(ok), exit_code=exit_code or None, output_tokens=tokens,
                  wall_s=wall if finite(wall) else None, tok_s=tokens / wall if tokens and finite(wall) and wall > 0 else None,
                  manifest_tok_s=rate if finite(rate) else None, total_accepted=accepted, total_draft=drafted,
                  accept_ratio=accepted / drafted if drafted and accepted is not None else None, fallback=fallback,
                  token_field_coverage=coverage, actual_client_concurrency=stats.get("concurrency"),
                  response_ids=len(ids), unique_response_ids=len(set(ids)), ordered_items_sha256=items_hash,
                  target_config={key: settings.get(key) for key in expected_settings}, target_flush_cumulative=receipt)
    return report, gates, ids


def fit_events(line):
    if (match := FIT_STASH.search(line)):
        yield "stash", Path(match[1]).name
    if (match := FIT_CAP.search(line)):
        yield "cap", tuple(map(int, match.groups()))
    if (match := FIT_INSTALL.search(line)):
        yield "install", int(match[1])


def scan(evidence, path, role):
    result = dict(size=0, events=defaultdict(list), errors=0, captures=set(), flush=[], states={},
                  refit=0, bad_fit=0, fit_sequence=[], ack=[], rpd_paths=set(), present=False)
    allowed = {"d_round", "d_layer", "d_rpd"} if role == "draft" else {"t_verify"}
    try:
        with path.open("rb") as handle:
            result["present"] = True
            for raw in handle:
                start = result["size"]
                end = result["size"] = start + len(raw)
                line = raw.decode("utf-8", errors="replace")
                if (match := CAPTURE.search(line)):
                    result["captures"].add(tuple(map(int, match.groups())))
                for key, pattern in (("tp", FLUSH), ("http", HTTP_FLUSH)):
                    result["flush"].extend((start, end, key, val) for val in pattern.findall(line))
                result["refit"] += bool(REFIT.search(line))
                result["bad_fit"] += bool(BAD_FIT.search(line))
                result["fit_sequence"].extend(fit_events(line))
                if (match := RPD_ACK.search(line)):
                    result["ack"].append((match[1], Path(match[2].strip()).name))
                if (match := RPD_PATH.search(line)):
                    result["rpd_paths"].add(match[1])
                try:
                    if role == "target" and "[CLEAR fit_state]" in line:
                        rank = TP.search(line)
                        if rank:
                            result["states"][rank[1]] = parse_object(line.split("[CLEAR fit_state]", 1)[1], suffix=True)
                    if (match := LEDGER.search(line)) and match[1] in allowed:
                        row = parse_object(line[match.end():], suffix=role == "target")
                        result["events"][match[1]].append((start, end, row))
                except ValueError as exc:
                    result["errors"] += 1
                    if result["errors"] <= 4:
                        evidence.issue(path, f"byte {start}: {exc}")
    except OSError as exc:
        evidence.issue(path, exc)
    return result


def bounds(evidence, path, logs):
    data = evidence.json(path)
    result = {}
    for role in ROLES:
        lo, hi = (obj(data.get(edge)).get(role) for edge in ("before", "after"))
        if type(lo) is int and type(hi) is int and 0 <= lo < hi <= logs[role]["size"]:
            result[role] = (lo, hi)
        else:
            evidence.issue(path, f"invalid or missing {role} byte range")
    return result


def in_range(event, interval):
    return interval is not None and interval[0] <= event[0] and event[1] <= interval[1]


def indexed_flush(lines, predicate):
    selected = [event for event in lines if predicate(event)]
    return {kind: dict(Counter(val for _, _, key, val in selected if key == kind)) for kind in ("tp", "http")}, selected


def ids_in(row):
    values = set(obj(row.get("e_rid")))
    for key in ("rid", "request_id"):
        if isinstance(row.get(key), str) and row[key]:
            values.add(row[key])
    for key in ("rids", "request_ids"):
        if isinstance(row.get(key), list):
            values.update(rid for rid in row[key] if isinstance(rid, str) and rid)
    return values


def select(events, own, all_formal, interval, *, warmup_excluded_by_offset=False):
    # ID-bearing records are never assigned to the wrong trial by offsets.
    exact, offset, covered = [], [], set()
    ambiguous = outside = offset_with_unknown_ids = offset_with_other_trial_ids = 0
    for event in events:
        rids = ids_in(event[2])
        if rids & own:
            if rids <= own:
                exact.append(event)
                covered.update(rids)
                outside += interval is not None and not in_range(event, interval)
            else:
                ambiguous += 1
        elif in_range(event, interval):
            if rids & all_formal:
                offset_with_other_trial_ids += 1
            else:
                offset_with_unknown_ids += bool(rids)
                if not rids or not warmup_excluded_by_offset:
                    offset.append(event)
    selected = exact or offset
    source = "request_ids" if exact else "same_log_byte_offsets" if offset else "unobserved"
    return selected, dict(source=source, rows=len(selected), ids_matched=len(covered), ids_expected=len(own),
                          ids_missing=len(own - covered) if exact else None, ambiguous=ambiguous,
                          offset_candidates=len(offset), offset_candidates_with_unknown_ids=offset_with_unknown_ids,
                          offset_rows_with_other_trial_ids=offset_with_other_trial_ids, id_rows_outside_byte_range=outside,
                          warmup_exclusion=("by_request_id" if exact else "by_target_flush"
                                            if offset and warmup_excluded_by_offset else "uncertain"))


def round_key(row):
    session, seq = row.get("session_id"), row.get("rpc_seq")
    if isinstance(session, str) and session and type(seq) is int and seq >= 0:
        return session, seq
    return None


def layer_review(rows, *, formal):
    totals = Counter()
    per_layer = defaultdict(Counter)
    per_cell = defaultdict(Counter)
    for _, _, row in rows:
        layer = row.get("layer")
        fit, clipped = row.get("fit"), row.get("clipped")
        pred, wall, host, gpu = (row.get(k) for k in ("pred_ms", "wall_us", "host_us", "gpu_us"))
        raw_ms = (wall / 1000 if count(wall) else None) if wall is not None else (
            (host + gpu) / 1000 if count(host) and count(gpu) else None)
        comparable = fit is True and raw_ms is not None and finite(pred) and pred > 0
        above = comparable and raw_ms > 1.5 * pred
        outcome = dict(rows=1, fitted=fit is True, comparable=comparable, raw_over_1_5_pred=above,
                       clipped=clipped is True, eligible_unclipped=above and clipped is False,
                       clipped_without_predicate=clipped is True and not above,
                       invalid_flags=type(fit) is not bool or type(clipped) is not bool,
                       invalid_layer=type(layer) is not int or layer < 1,
                       layer_one_fit=layer == 1 and fit is True,
                       layer_two_plus_fit=type(layer) is int and layer >= 2 and fit is True)
        for bucket in (totals, per_layer[str(layer)], per_cell[f"W={row.get('w')},path={row.get('path')},g={row.get('graph_bucket')}"]):
            bucket.update(outcome)
    if not totals["rows"]:
        status = "unobserved"
    elif not formal:
        status = "formal_warmup_exclusion_uncertain"
    elif totals["clipped_without_predicate"] or totals["invalid_flags"]:
        status = "invalid_evidence"
    elif totals["clipped"]:
        status = "clipping_observed"
    elif totals["raw_over_1_5_pred"]:
        status = "above_threshold_but_no_clip_logged"
    elif totals["comparable"]:
        status = "no_threshold_trigger_observed"
    else:
        status = "no_comparable_fitted_samples"
    return dict(status=status, formal=formal, counts=dict(totals), by_layer=dict(per_layer), by_cell=dict(per_cell))


def draft_review(log, own, all_formal, interval, mode):
    events = log["events"]
    selected, attribution = select((x for x in events["d_round"] if x[2].get("prefit") is None), own, all_formal, interval)
    keys = Counter(round_key(row) for _, _, row in selected)
    keyset = keys.keys() - {None}
    round_ids = {round_key(row): ids_in(row) for _, _, row in selected}
    layers = [event for event in events["d_layer"] if round_key(event[2]) in keyset]
    by_id = attribution["source"] == "request_ids"
    if not by_id:
        layers = [event for event in layers if in_range(event, interval)]
    layer = layer_review(layers, formal=by_id)
    observed_layer = layer["counts"]
    hold_invalid = fallback = 0
    diagnostics = Counter()
    rpd_ids = set()
    for _, _, row in selected:
        hold_invalid += bool(row.get("hold_invalid")) or (row.get("excluded") is False and not all(
            finite(row.get(key)) and row[key] >= 0 for key in ("hold_ms", "q_ms")))
        reasons = row.get("exclude_reason")
        fallback += isinstance(reasons, list) and any(reason in reasons for reason in ("fallback", "target_fallback"))
    for source, group in (("inline", selected), ("late", events["d_rpd"])):
        for event in group:
            row = event[2]
            if source == "late" and (round_key(row) not in keyset or not (by_id or in_range(event, interval))):
                continue
            diagnostics[f"{source}_mask_rounds"] += bool(row.get("rpd_masks"))
            if source == "late":
                diagnostics["late_tau_invalid"] += row.get("tau") != TAU
            for rid, raw_detail in obj(row.get("e_rid")).items():
                detail = obj(raw_detail)
                if not any(k in detail for k in ("accept_k", "rpd_ok", "rpd_valid")):
                    continue
                windows = obj(detail.get("accept_k"))
                valid = (detail.get("rpd_ok") is True and rid in round_ids[round_key(row)]
                         and any(count(v) for v in windows.values()) and all(v is None or count(v) for v in windows.values()))
                if valid:
                    diagnostics[f"{source}_valid_mask_rows"] += 1
                    rpd_ids.add(rid)
                else:
                    diagnostics["invalid_mask_rows"] += 1
    valid_masks = diagnostics["inline_valid_mask_rows"] + diagnostics["late_valid_mask_rows"]
    gates = dict(draft_attribution=bool(selected and not attribution["ambiguous"]
                                       and (not by_id or attribution["ids_missing"] == 0)),
                 draft_round_keys=bool(keys and None not in keys and all(n == 1 for n in keys.values())),
                 draft_hold=bool(selected) and hold_invalid == 0, draft_no_fallback=bool(selected) and fallback == 0,
                 layer_evidence=bool(observed_layer.get("layer_two_plus_fit") and observed_layer.get("comparable")
                                     and layer["by_layer"].get("1") and not any(observed_layer.get(key) for key in
                                                 ("invalid_flags", "invalid_layer", "layer_one_fit", "clipped_without_predicate"))),
                 draft_rpd=(not diagnostics["invalid_mask_rows"] and not diagnostics["late_tau_invalid"]
                            and (bool(valid_masks) if mode == "rpd" else not valid_masks and not any(
                                diagnostics[f"{source}_mask_rounds"] for source in ("inline", "late")))))
    return dict(attribution=attribution, hold_invalid=hold_invalid, fallback_rounds=fallback,
                rpd=dict(diagnostics, unique_ids_with_valid_masks=len(rpd_ids))), layer, gates


def target_review(log, own, all_formal, interval, mode, ordinal):
    cumulative, block = indexed_flush(log["flush"], lambda e: interval is not None and e[1] <= interval[1])
    current, matches = indexed_flush(block, lambda e: in_range(e, interval))
    verified_flush = current == expected_flush(1) and cumulative == expected_flush(ordinal)
    events = log["events"]["t_verify"]
    # The latest of TP0, TP1 and the HTTP 200 is a Target-local warmup fence.
    # Never reuse a Target byte range or timestamp for the Draft log.
    offset_range = (max(event[1] for event in matches), interval[1]) if verified_flush else interval
    selected, attribution = select(events, own, all_formal, offset_range, warmup_excluded_by_offset=verified_flush)
    paths, timings = Counter(), Counter()
    invalid = fallback = 0
    for _, _, row in selected:
        path = row.get("path")
        paths[path if isinstance(path, str) else "missing_or_invalid"] += 1
        fallback += path == "fallback" or row.get("fallback") is True
        wall, ready = (row.get(key) for key in ("verify_wall_us", "accept_ready_us"))
        invalid += path not in ("graph", "eager") or not (count(wall) and count(ready) and wall >= ready)
        for key in RPD_TIMINGS:
            value = row.get(key)
            timings[key] += count(value)
            invalid += value is not None and not count(value)
    by_id = attribution["source"] == "request_ids"
    return dict(attribution=attribution, paths=paths, fallback_rows=fallback, invalid=invalid, rpd_timing_rows=timings,
                target_flush_cumulative=cumulative, target_flush_in_range=current), dict(
                    flush_log=verified_flush,
                    target_attribution=bool(selected and not attribution["ambiguous"]
                                            and (attribution["ids_missing"] == 0 if by_id else verified_flush)),
                    target_verify=bool(selected) and invalid == 0 and fallback == 0,
                    target_rpd=(all(timings[k] > 0 for k in RPD_TIMINGS) if mode == "rpd" else not any(timings.values())))


def same_ready(states):
    left, right = (obj(states.get(rank)) for rank in ("0", "1"))
    return bool(isinstance(left.get("session"), str) and left["session"] and left["session"] == right.get("session")
                and all((row.get("state"), row.get("reason"), row.get("run")) == ("ready", "ready", None)
                        for row in (left, right)))


def cleanup(evidence, raw, root):
    report = {}
    for role in ROLES:
        before = evidence.text(raw / "sync" / f"precheck-{role}.txt")
        after = evidence.text(root / f"postcheck-{role}.txt")
        receipt = evidence.json(root / f"{role}-cleanup.json")
        signals = receipt.get("signals")
        report[role] = dict(postcheck_matches=bool(before and before == after and f"{role}_ports_and_service_precheck OK" in before),
                            live_session_members=receipt.get("live_session_members"),
                            signals_valid=isinstance(signals, list) and all(isinstance(row, dict)
                                and type(row.get("term_exit")) is int and row["term_exit"] in (0, 4) for row in signals))
    errors_path = root / "cleanup-errors.txt"
    cleanup_errors = evidence.text(errors_path).strip() if errors_path.is_file() else ""
    passed = not cleanup_errors and all(r["postcheck_matches"] and r["live_session_members"] == [] and r["signals_valid"]
                                        for r in report.values())
    return dict(roles=report, errors_reported=bool(cleanup_errors)), passed


def fit_baseline(evidence):
    receipt = evidence.text(REFERENCE / "raw" / "SHA256SUMS")
    entries = {match[2]: match[1] for match in re.finditer(r"^([a-f0-9]{64})  (\S+)$", receipt, re.M)}
    return {mode: entries.get(f"{mode}/{name}") for mode, name in FIT_FILES.items()}


def mode_review(evidence, raw, mode, logs, reference_hashes):
    root = raw / mode
    draft, target = (logs[role] for role in ROLES)
    hashes = {when: evidence.json(root / f"fit-hashes-{when}.json") for when in ("before", "gate", "after")}
    coverage = evidence.json(root / "fit-coverage.json")
    gate_receipt = evidence.text(root / "fit-gate.txt")
    gate_states = {}
    for line in gate_receipt.splitlines():
        if "[CLEAR fit_state]" in line and (rank := TP.search(line)):
            try:
                gate_states[rank[1]] = parse_object(line.split("[CLEAR fit_state]", 1)[1], suffix=True)
            except ValueError as exc:
                evidence.issue(root / "fit-gate.txt", exc)
    clean, clean_ok = cleanup(evidence, raw, root)
    cumulative, _ = indexed_flush(target["flush"], lambda _: True)
    captures = {role: sorted(logs[role]["captures"]) for role in ROLES}
    coverage_ok = (coverage.get("schema") == 5 and coverage.get("fingerprint") == "babb423d"
                   and coverage.get("required_h") == 90 and coverage.get("required_layers") == 18
                   and count(coverage.get("min_h_samples")) and coverage["min_h_samples"] >= 4
                   and count(coverage.get("min_layer_samples")) and coverage["min_layer_samples"] >= 50
                   and coverage.get("reused") is True and coverage.get("sha256") == hashes["gate"].get(mode)
                   and coverage.get("session") == obj(gate_states.get("0")).get("session")
                   and all(coverage.get(f"{role}_health") == 200 and coverage.get(f"{role}_captures") == len(captures[role])
                           for role in ROLES))
    install = [("stash", FIT_FILES[mode]), ("cap", (64, 4)), ("install", 90)]

    def active_install(sequence):
        return [event for event in sequence if event[0] != "stash" or event[1] == FIT_FILES[mode]]

    gate_events = [event for line in gate_receipt.splitlines() for event in fit_events(line)]
    gates = dict(
        block_complete=bool(evidence.text(root / "block-complete.txt").strip()),
        clean=clean_ok,
        logs=all(logs[role]["present"] and not logs[role]["errors"] for role in ROLES),
        fit_coverage=coverage_ok,
        fit_hashes_stable=all(all(sha(snapshot.get(m)) for m in FIT_FILES) and snapshot == hashes["before"]
                              for snapshot in hashes.values()),
        fit_hashes_match_reference=all(sha(reference_hashes[m]) and hashes["before"].get(m) == reference_hashes[m] for m in FIT_FILES),
        fit_gate=(same_ready(target["states"]) and same_ready(gate_states)
                  and all(gate_states[rank] == target["states"][rank] for rank in ("0", "1"))
                  and active_install(gate_events) == active_install(draft["fit_sequence"]) == install),
        no_refit=not (draft["refit"] or target["refit"] or REFIT.search(gate_receipt)
                      or draft["bad_fit"] or target["bad_fit"] or BAD_FIT.search(gate_receipt)),
        captures=(TARGET_SHAPES <= target["captures"] and DRAFT_SHAPES <= draft["captures"]
                  and (4, 65) not in draft["captures"]
                  and all({b for b, _ in captures[role]} == set(BATCHES) for role in ROLES)),
        flush_total=cumulative == expected_flush(4),
        mode_handshake=(draft["ack"] == [(str(TAU), FIT_FILES[mode])] if mode == "rpd" else not draft["ack"]),
        rpd_path=(bool(target["rpd_paths"]) and target["rpd_paths"] <= {"npu_sr_host_v1", "npu_sr_host_plan"}
                  if mode == "rpd" else not target["rpd_paths"]),
    )
    return dict(fit=coverage, fit_hashes=hashes, reference_fit_hashes=reference_hashes, fit_states=target["states"],
                fit_gate_states=gate_states, fit_install_sequence=draft["fit_sequence"],
                fit_refit_markers={role: logs[role]["refit"] for role in ROLES}, captures=captures,
                target_rpd_paths=sorted(target["rpd_paths"]), draft_rpd_ack=draft["ack"], cleanup=clean,
                log_sizes={role: logs[role]["size"] for role in ROLES}, target_flush_total=cumulative), gates


def dataset_receipt(entry):
    images = entry.get("ordered_image_sha256")
    return (entry.get("selected") == N and count(entry.get("available")) and entry["available"] >= N
            and sha(entry.get("file_sha256")) and sha(entry.get("selected_rows_sha256"))
            and isinstance(images, list) and len(images) == N and all(map(sha, images)))


def review(run):
    evidence = Evidence(run)
    raw = run / "raw"
    modes, mode_gates, trials, trial_gates, trial_ids = {}, {}, [], {}, {}
    sync = raw / "sync"
    ref_hashes = fit_baseline(evidence)
    manifests = [evidence.text(sync / f"{name}.manifest") for name in ("head", "draft", "npu")]
    identity = evidence.json(sync / "data-identity.json")
    reference_identity = {
        name: obj(evidence.json(reference / "raw" / "sync" / "data-identity.json").get(name))
        for name, reference in zip(DATASETS, (REFERENCE, WATERCAPTION_REFERENCE))
    }
    reference_trials = evidence.json(REFERENCE / "summary.json").get("trials")
    ref_order = {obj(row).get("ordered_items_sha256") for row in reference_trials
                 if sha(obj(row).get("ordered_items_sha256"))} if isinstance(reference_trials, list) else set()
    for i, mode in enumerate(FIT_FILES):
        for j, (dataset, concurrency) in enumerate(((d, c) for d in DATASETS for c in (1, 4)), 1):
            trial_id = f"{4 * i + j:02d}"
            result, gates, ids = client(evidence, raw / trial_id, mode, concurrency, j)
            trials.append(dict(id=trial_id, mode=mode, dataset=dataset, concurrency=concurrency,
                               flush_ordinal=j, rpd_tau=TAU if mode == "rpd" else None, **result))
            trial_gates[trial_id], trial_ids[trial_id] = gates, ids
    ids = Counter(rid for values in trial_ids.values() for rid in values)
    all_formal = set(ids)
    ledger_ids = {role: set() for role in ROLES}
    for mode in FIT_FILES:
        logs = {role: scan(evidence, raw / mode / f"{role}.log", role) for role in ROLES}
        for role, kind in (("draft", "d_round"), ("target", "t_verify")):
            for _, _, row in logs[role]["events"][kind]:
                if row.get("prefit") is None:
                    ledger_ids[role].update(ids_in(row) & all_formal)
        modes[mode], mode_gates[mode] = mode_review(evidence, raw, mode, logs, ref_hashes)
        old_end = dict.fromkeys(ROLES, 0)
        for trial in (t for t in trials if t["mode"] == mode):
            trial_id = trial["id"]
            interval = bounds(evidence, raw / trial_id / "log-boundaries.json", logs)
            nonoverlap = all(role in interval and interval[role][0] >= old_end[role] for role in ROLES)
            for role, (_, end) in interval.items():
                old_end[role] = max(old_end[role], end)
            own = set(trial_ids[trial_id])
            draft, layers, dg = draft_review(logs["draft"], own, all_formal, interval.get("draft"), mode)
            target, tg = target_review(logs["target"], own, all_formal, interval.get("target"), mode, trial["flush_ordinal"])
            gates = trial_gates[trial_id]
            gates.update(log_boundaries=nonoverlap, **dg, **tg)
            trial.update(log_boundaries=interval, draft_rounds=draft, draft_layers=layers, target_verify=target,
                         status="verified" if all(gates.values()) else "incomplete_or_unverified")
    orders = {dataset: {trial["ordered_items_sha256"] for trial in trials if trial["dataset"] == dataset} for dataset in DATASETS}
    cross = dict(packages_match=bool(manifests[0].strip()) and len(set(manifests)) == 1,
                 data_identity=all(dataset_receipt(obj(identity.get(name))) for name in DATASETS),
                 rsteller_reference_identity=bool(reference_identity["RSTeller"])
                     and identity.get("RSTeller") == reference_identity["RSTeller"],
                 watercaption_reference_identity=bool(reference_identity["WaterCaption"])
                     and identity.get("WaterCaption") == reference_identity["WaterCaption"],
                 same_order_within_dataset=all(len(order) == 1 and None not in order for order in orders.values()),
                 rsteller_reference_order=bool(ref_order) and orders["RSTeller"] == ref_order,
                 global_unique_response_ids=len(ids) == 8 * N and all(n == 1 for n in ids.values()))
    passed = not evidence.errors and all(cross.values()) and all(
        all(gates.values()) for gates in (*trial_gates.values(), *mode_gates.values()))
    checks = dict(passed=passed, cross=cross, modes=mode_gates, trials=trial_gates, errors=evidence.errors)
    credits = dict(expected=8 * N, observed=sum(ids.values()), unique=len(ids),
                   duplicate_ids=sum(n > 1 for n in ids.values()), missing=max(0, 8 * N - len(ids)),
                   draft_ids_matched=len(ledger_ids["draft"]), target_ids_matched=len(ledger_ids["target"]))
    summary = dict(run_id=RUN_ID, commit=COMMIT, status="passed" if passed else "incomplete_or_unverified",
                   trials=trials, modes=modes, response_id_coverage=credits,
                   fit_hash_reference="2026-11-28-c/raw/SHA256SUMS",
                   layer_predicate="fit is true and raw_ms > 1.5 * positive pred_ms; raw_ms uses wall_us, otherwise host_us + gpu_us",
                   layer_note="Above-threshold is necessary, not sufficient: span/readiness can suppress an update clip. "
                              "Offset-only Draft counts can include warmup; do not treat them as formal clipping evidence.",
                   data_note="Dataset and ordered image receipts are compared to 28-c (RSTeller) and 27-b (WaterCaption); "
                             "client item order is compared across all four trials for each dataset.")
    return checks, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN, help="directory containing raw/")
    parser.add_argument("--output-dir", type=Path, help="existing output directory (defaults to --run-dir)")
    args = parser.parse_args(argv)
    run = args.run_dir.resolve()
    output = args.output_dir.resolve() if args.output_dir else run
    paths = [output / name for name in ("checks.json", "summary.json")]
    try:
        if not output.is_dir():
            raise OSError(f"output directory does not exist: {output}")
        if any(path.exists() or path.is_symlink() for path in paths):
            raise FileExistsError("refusing to overwrite checks.json or summary.json; select a new --output-dir")
        checks, summary = review(run)
        serialized = [json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n" for report in (checks, summary)]
        for path, text in zip(paths, serialized):
            with path.open("x", encoding="utf-8") as handle:
                handle.write(text)
    except (OSError, ValueError) as exc:
        print(f"review: {exc}", file=sys.stderr)
        return 2
    print(f"review: {summary['status']} ({sum(t['status'] == 'verified' for t in summary['trials'])}/8 trials verified)")
    return 0 if checks["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
