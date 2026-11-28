#!/usr/bin/env python3
"""Review the four formal RSTeller trials, including partial or blocked runs."""
import ast
import csv
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path

from supervisor import RUN, RAW, SYNC, PREVIOUS, SKILL, base, fit_coverage, load

accept = load("ct_accept", SKILL / "accept_logs.py")
prior = load("ct_review_previous", PREVIOUS / "tools/review.py")
meta = json.loads((RUN / "meta.json").read_text())


def read(path):
    return path.read_text(errors="replace") if path.is_file() else ""


def read_json(path):
    return json.loads(read(path)) if path.is_file() else None


def matching(pattern, line):
    parts = re.search(pattern, line)
    return parts.groups() if parts else None


def check_trial(trial):
    name = trial["id"]
    out = RAW / name
    artifacts = list(out.glob("client.*.jsonl"))
    if len(artifacts) != 1:
        result = dict(trial, status="missing_client_jsonl")
        return result, set(), {"jsonl": False}
    path = artifacts[0]
    rows = [json.loads(line) for line in read(path).splitlines() if line.strip()]
    ok = [row for row in rows if not row.get("error")]
    rids = {row.get("response_meta_info", {}).get("id") for row in ok if isinstance(row.get("response_meta_info"), dict)} - {None, ""}
    stdout = read(out / "client.stdout")
    reported = matching(r"summary items=(\d+)/(\d+)\s+toks=(\d+)\s+wall=([\d.]+)s", stdout)
    wall = float(reported[3]) if reported else None
    manifest = read_json(path.with_suffix(".run.json")) or {}
    stats = manifest.get("summary", {})
    csv_rows = list(csv.DictReader(StringIO(read(path.with_suffix(".csv")))))
    csv_data = csv_rows[0] if len(csv_rows) == 1 else {}
    keys = ["completion_tokens", "spec_accept_token_num", "spec_draft_token_num", "spec_fallback_ct"]
    counts = {key: sum(int(row[key]) for row in ok) if ok and all(
              type(row.get(key)) in (int, float) and math.isfinite(row[key]) and row[key] >= 0 and int(row[key]) == row[key]
              for row in ok) else None for key in keys}
    tokens, accepted, drafted, fallback = (counts[key] for key in keys)
    flow = matching(r"traffic=open\s+request_rate=(\S+)\s+max_concurrency=(\d+)\s+max_new_tokens=(\d+)", stdout)
    config = (manifest.get("server") or {}).get("settings") or {}
    expected = dict(speculative_algorithm="CLEAR_TREE_REMOTE", device="npu", tp_size=2, page_size=128,
                    disable_cuda_graph=False, cuda_graph_bs=[1, 2, 4], attention_backend="ascend",
                    speculative_verify_mode=trial["mode"], speculative_num_steps=8, speculative_num_draft_tokens=65)
    if trial["mode"] == "rpd":
        expected["speculative_rpd_tau"] = trial["rpd_tau"]
    flush = prior.flush_counts(accept, read(out / "target-flush.txt"))
    order = [(row.get("image"), row.get("question"), row.get("ground_truth")) for row in rows]
    items_hash = prior.sha(json.dumps(order, ensure_ascii=False, sort_keys=True).encode()) if rows else None
    tok_s = tokens / wall if tokens and wall and wall > 0 else None
    evidence = dict(
        artifacts=bool(len(csv_rows) == 1 and manifest.get("run_id") == path.name.removeprefix("client.").removesuffix(".jsonl")),
        exit=read(out / "exit-code.txt").strip() == "0",
        complete=len(rows) == len(ok) == len(rids) == stats.get("n_ok") == stats.get("n_total") == 100,
        ordered_items=[row.get("global_index") for row in rows] == list(range(100)),
        traffic=flow == ("inf", str(trial["concurrency"]), "1024"),
        active_target=all(config.get(k) == v for k, v in expected.items()),
        timing=bool(tok_s and reported[:3] == (str(len(ok)), str(len(rows)), str(tokens))
                    and stats.get("total_tokens") == tokens and csv_data.get("Total Generated Tokens") == str(tokens)
                    and type(stats.get("output_tok_s")) in (int, float) and stats["output_tok_s"] > 0
                    and abs(tokens / stats["output_tok_s"] - wall) <= 0.000501),
        speculation=bool(drafted and accepted is not None and 0 <= accepted <= drafted and fallback == 0),
        flush_client=bool(re.search(r"=== warmup \(first item, discarded\) ===[\s\S]*warmup ok[\s\S]*flush_cache ->[\s\S]*=== main run ===", stdout)
                          and "flush_cache failed:" not in stdout),
        flush_target=prior.passed_flush(flush, 1 if trial["concurrency"] == 1 else 2),
    )
    return dict(trial, status="completed" if all(evidence.values()) else "unverified",
                requests=len(rows), ok=len(ok), output_tokens=tokens, wall_s=wall, tok_s=tok_s,
                total_accepted=accepted, total_draft=drafted, accept_ratio=accepted / drafted if drafted and accepted is not None else None,
                fallback=fallback, ordered_items_sha256=items_hash, cumulative_flush={"tp": flush[0], "http": flush[1]}), rids, evidence


def prefit_layers(path):
    counts = Counter()
    for line in read(path).splitlines():
        if "[CLEAR] wrote fit file" in line:
            break
        marker = "[CLEAR ledger d_layer] "
        if marker in line:
            row = json.JSONDecoder().raw_decode(line.split(marker, 1)[1].lstrip())[0]
            layer = "layer_1" if row["layer"] == 1 else "layer_2_plus"
            counts[f"{layer}_fit_{str(row['fit']).lower()}"] += 1
    return dict(counts)


def prefit_width_groups(path):
    report = defaultdict(list)
    for line in read(path).splitlines():
        marker = "[CLEAR] prefit group "
        if marker in line:
            row = ast.literal_eval(line.split(marker, 1)[1])
            if row["kind"] == "w":
                report[row["b"]].append({key: row[key] for key in ("steps", "step_target_met", "retries")})
    return dict(report)


def review_mode(mode, trials, rids, notes):
    root = RAW / mode
    if not root.is_dir():
        return {"status": "not_started"}, {"started": False}
    draft = prior.scan_log(accept, root / "draft.log", mode, "draft", notes)
    target = prior.scan_log(accept, root / "target.log", mode, "target", notes)
    draft_fit, target_fit = draft["fit"], target["fit"]
    name = base.NAMES[mode]
    fit = read_json(root / name)
    cleanup = prior.cleanup_review(RAW, mode, notes)
    coverage, session, fit_error = None, None, None
    if fit is not None:
        try:
            coverage = fit_coverage(fit, mode, draft["captures"])
            session = target_fit["states"]["0"]["session"]
        except (RuntimeError, KeyError, ValueError, TypeError) as exc:
            fit_error = str(exc)
    actual_write = draft_fit["writes"]
    single_write = len(actual_write) == 1 and actual_write[0][1] == name
    cache_miss = single_write and any(filename == name and n < actual_write[0][0] for n, filename in draft_fit["absent"])
    active_fit = bool(session and fit and all(
        target_fit["states"].get(rank, {}).get("session") == session
        and tuple(target_fit["states"][rank].get(key) for key in ("state", "reason", "run")) == ("ready", "prefit", fit["run"])
        for rank in ("0", "1")))
    merge = defaultdict(list)
    for row in draft["events"]["d_layer"]:
        merge[prior.round_key(row)].append(row)
    rounds = draft["events"]["d_round"]
    late = {prior.round_key(row): row for row in draft["events"]["d_rpd"]}
    tagged = [(None, row) for row in rounds]
    groups, assignment = accept.assign_trial_rounds(tagged, {t["id"]: rids[t["id"]] for t in trials})
    checks_by_trial = {}
    for trial in trials:
        key = trial["id"]
        group = [row for _, row in groups[key]]
        layer_report = prior.layer_review(group, merge)
        round_report = prior.round_review(group, late, mode)
        accepted_k = Counter()
        for row in group:
            report = late.get(prior.round_key(row)) or row
            for detail in (report.get("e_rid") or {}).values():
                if detail.get("rpd_ok") is True and isinstance(detail.get("accept_k"), dict):
                    accepted_k.update(k for k, value in detail["accept_k"].items() if value is not None)
        trial["draft_rounds"] = round_report
        trial["draft_layers"] = layer_report
        trial["rpd_accept_k_observed"] = dict(accepted_k) if mode == "rpd" else None
        checks_by_trial[key] = dict(rounds_observed=bool(group), layer_one_excluded=(
            bool(layer_report["by_layer"].get(1, {}).get("rows")) and not layer_report["violations"].get("layer1_fit")),
            layer_two_plus_fit=bool(layer_report["counts"].get("fitted_layer_ge_2")),
            layer_valid=not layer_report["violations"], missing_or_invalid_hold=round_report["counts"].get("hold_invalid", 0),
            rpd_mask_valid=not any(k.startswith("rpd_") for k in round_report["violations"]),
            rpd_masks_observed=(bool(accepted_k) if mode == "rpd" else None))
    windows = draft["windows"]
    bonus = dict(windows=len(windows), tail=("flushed" if windows[-1][0] == "idle" else "missing") if windows else "missing",
                 reasons=Counter(reason for reason, _ in windows), totals={key: sum(w.get(key, 0) for _, w in windows) if windows else None
                                                                          for key in accept.BONUS_KEYS})
    verify = prior.verify_review(target["events"]["t_verify"])
    host_rpd = [verify["metrics"][key]["n"] for key in prior.TIMINGS[2:]]
    commit = [row for row in target_fit["commits"] if row.get("ready") is True]
    expected_target = {(b, n) for b in base.BATCHES for n in (1, 3, 5, 9, 17, 33, 65)}
    expected_draft = (expected_target | {(b, n) for b in base.BATCHES for n in base.TIERS}) - {(4, 65)}
    gates = dict(
        block_complete=(root / "block-complete.txt").is_file(),
        clean=cleanup["passed"],
        refit=bool(coverage and active_fit and cache_miss and target_fit["started"] == 1
                   and target_fit["driver"] == ["ready"] and commit and commit[-1].get("run") == fit["run"]
                   and not draft_fit["flat"] and not target_fit["flat"]),
        mode_handshake=draft_fit["rpd_ack"] == ([("0.4", name)] if mode == "rpd" else []),
        captures=expected_target <= target["captures"] and expected_draft <= draft["captures"] and (4, 65) not in draft["captures"],
        target_verify=bool(verify["paths"]) and not verify["invalid"] and (all(host_rpd) if mode == "rpd" else not any(host_rpd)),
        ledger=bool(trials) and all(v["rounds_observed"] and v["layer_one_excluded"] and v["layer_two_plus_fit"]
                                    and v["layer_valid"] and v["rpd_mask_valid"] and not v["missing_or_invalid_hold"]
                                    and v["rpd_masks_observed"] is not False for v in checks_by_trial.values()),
    )
    if mode == "rpd":
        during = root / "greedy-fit-during-rpd.json"
        after = read(root / "greedy-fit-after-rpd.sha256").split()
        earlier = RAW / "greedy" / base.NAMES["greedy"]
        gates["greedy_preserved"] = bool(during.is_file() and earlier.is_file() and after
                                         and base.sha_bytes(earlier) == base.sha_bytes(during) == after[0])
    return dict(status="completed" if gates["block_complete"] else "incomplete", fit=coverage, fit_error=fit_error,
                prefit_layers=prefit_layers(root / "draft.log"), prefit_width_groups=prefit_width_groups(root / "target.log"),
                commits=target_fit["commits"], verify=verify, bonus=bonus, cleanup=cleanup, round_assignment=assignment,
                captures={"draft": sorted(draft["captures"]), "target": sorted(target["captures"]), "draft_over_capacity": [4, 65]}), dict(gates, trials=checks_by_trial)


def accept_block(mode, ids):
    root = RAW / mode
    if not root.is_dir():
        return
    view = root / "accept-input"
    view.mkdir(exist_ok=True)
    for local, source in [(f"{role}.log", root / f"{role}.log") for role in ("draft", "target")] + [(trial, RAW / trial) for trial in ids]:
        path = view / local
        if source.exists() and not path.is_symlink():
            path.symlink_to(Path("../..") / source.relative_to(RAW), target_is_directory=source.is_dir())
    result = subprocess.run([sys.executable, "-B", str(SKILL / "accept_logs.py"), "--raw", str(view)],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (root / "accept-logs.txt").write_text(result.stdout)
    return result.returncode


def main():
    reviews, rids, checks, notes = [], {}, {}, []
    for trial in meta["trials"]:
        result, identities, gates = check_trial(trial)
        reviews.append(result)
        rids[trial["id"]] = identities
        checks[trial["id"]] = gates
    modes, mode_checks = {}, {}
    for mode in base.TRIALS:
        selected = [trial for trial in reviews if trial["mode"] == mode]
        modes[mode], mode_checks[mode] = review_mode(mode, selected, rids, notes)
        mode_checks[mode]["accept_logs_exit"] = accept_block(mode, [trial["id"] for trial in selected])
    manifests = [read(SYNC / f"{name}.manifest") for name in ("head", "draft", "npu")]
    expected = read_json(PREVIOUS / "raw/sync/data-identity.json")

    def data_match(suffix):
        observed = read_json(SYNC / f"data-identity{suffix}.json")
        return None if expected is None or observed is None else observed == expected

    all_ids = [rid for ids in rids.values() for rid in ids]
    item_hashes = [trial.get("ordered_items_sha256") for trial in reviews]
    cross = dict(packages_match=bool(manifests[0]) and len(set(manifests)) == 1,
                 data_match_before_greedy=data_match(""), data_match_before_rpd=data_match("-rpd"),
                 same_ordered_items=len(set(item_hashes)) == 1 if all(item_hashes) else None,
                 distinct_response_ids=(len(set(all_ids)) == 400 if len(all_ids) == 400 else None))
    for mode in base.TRIALS:
        if read(RAW / mode / "driver-failure.txt"):
            notes.append(f"{mode}: see raw/{mode}/driver-failure.txt.")
    passed = all(cross.values()) and all(all(x.values()) for x in checks.values()) and all(
        all(value for key, value in row.items() if key not in ("trials", "accept_logs_exit")) and row["accept_logs_exit"] == 0
        for row in mode_checks.values())
    base.write_json(RUN / "summary.json", dict(run_id=meta["run_id"], commit=meta["commit"],
                   status="passed" if passed else "incomplete_or_unverified", trials=reviews, modes=modes))
    base.write_json(RUN / "checks.json", dict(passed=passed, cross=cross, trials=checks, modes=mode_checks, errors=notes))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
