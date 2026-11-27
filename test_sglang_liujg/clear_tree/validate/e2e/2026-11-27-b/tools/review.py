#!/usr/bin/env python3
"""Summarize only the two modes and formal request IDs recorded in this run."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def only(paths):
    paths = list(paths)
    if len(paths) != 1:
        raise ValueError(f'expected one path, found {paths}')
    return paths[0]


def number(x):
    return not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x)


def near(a, b):
    return math.isclose(a, b, rel_tol=2e-5, abs_tol=1e-7)


def client(raw, trial, spec):
    d = raw / trial
    path = only(d.glob('client.*.jsonl'))
    rows = read_jsonl(path)
    ok = [row for row in rows if not row.get('error')]
    output = (d / 'client.stdout').read_text(errors='replace')
    stats = load(path.with_suffix('.run.json'))['summary']
    csvrow = only(csv.DictReader(only(d.glob('client.*.csv')).open(newline='')))
    metrics = {k: float(csvrow[k]) if csvrow.get(k) else None for k in ('Avg BLEU', 'Avg ROUGE-1', 'Avg ROUGE-2', 'Avg ROUGE-L')}
    scored = re.findall(r'\bscored (\d+)/(\d+) items', output)
    fields = ('completion_tokens', 'spec_accept_token_num', 'spec_draft_token_num', 'spec_fallback_ct')
    coverage = {k: sum(number(r.get(k)) for r in ok) for k in fields}
    totals = {k: int(sum(r[k] for r in ok)) if coverage[k] == len(ok) and ok else None for k in fields}
    out = totals['completion_tokens']
    accept = totals['spec_accept_token_num']
    draft = totals['spec_draft_token_num']
    wall_printed = float(only(re.findall(r'\btoks=\d+\s+wall=([0-9.]+)s', output)))
    tok_s = stats['output_tok_s']
    rid = {row.get('response_meta_info', {}).get('id') for row in ok}
    rid.discard(None)
    result = dict(spec, requests=len(rows), ok=len(ok), output_tokens=out, wall_s=wall_printed,
                  tok_s=tok_s, total_accepted=accept, total_draft=draft, accept_ratio=(accept/draft if draft and accept is not None else None),
                  fallback=totals['spec_fallback_ct'], actual_client_concurrency=stats['concurrency'], metrics=metrics,
                  metrics_scored=list(map(int, scored[-1])) if scored else None,
                  flush_client_ok='flush_cache failed:' not in output and '=== main run ===' in output and 'flush_cache ->' in output,
                  counts_ok=len(rows) == len(ok) == len(rid) == stats['n_ok'] == stats['n_total'] == 100,
                  metric_coverage=coverage,
                  timing_consistent=bool(out and tok_s and stats['total_tokens'] == out and abs(out / tok_s - wall_printed) <= 0.000501),
                  response_ids_unique=len(rid) == len(ok))
    return result, rid


def event(line, name):
    return json.loads(line.split(f'[CLEAR ledger {name}]', 1)[1])


def mode_review(accept, raw, mode, rids):
    d = raw / mode
    draft = d / 'draft.log'
    target = d / 'target.log'
    rounds, windows, _, draft_caps, _, _, dfit, de = accept.load_server_log(draft)
    _, _, _, target_caps, flushes, statuses, tfit, te = accept.load_server_log(target)
    groups, assignment = accept.assign_trial_rounds(rounds, rids)
    seen_fit = False
    masks = {}
    for line in draft.open(errors='replace'):
        if '[CLEAR] wrote fit file' in line:
            masks.clear()
            seen_fit = True
        if seen_fit and '[CLEAR ledger d_rpd]' in line:
            row = event(line, 'd_rpd')
            masks[(row['session_id'], row['rpc_seq'])] = row
    verify = defaultdict(list)
    active = False
    timing_bad = 0
    paths = Counter()
    metrics = ('verify_wall_us', 'accept_ready_us', 'rpd_build_us', 'rpd_stats_us', 'rpd_select_us', 'rpd_mask_us')
    for line in target.open(errors='replace'):
        if 'prefit driver finished state=ready' in line:
            active = True
        if active and '[CLEAR ledger t_verify]' in line:
            row = event(line, 't_verify')
            paths[row.get('path')] += 1
            for key in metrics:
                val = row.get(key)
                if type(val) is int and val >= 0:
                    verify[key].append(val)
                elif val is not None or key in metrics[:2] and row.get('path') != 'fallback':
                    timing_bad += 1
            wall, ready = (row.get(k) for k in metrics[:2])
            timing_bad += int(number(wall) and number(ready) and wall < ready)
    exported_rounds = {}
    for trial, group in groups.items():
        counts = Counter()
        violations = Counter()
        depth = defaultdict(lambda: {'rows': 0, 'observed': 0, 'expected': 0.0})
        scored = [(when, row) for when, row in group if not row.get('excluded') and row.get('hold_ms') is not None and row.get('q_ms') is not None]
        k_modes = Counter(str(row.get('k_mode')) for _, row in scored)
        counts.update(rounds=len(group), scored=len(scored), excluded=sum(bool(r.get('excluded')) for _, r in group), hold_invalid=sum(bool(r.get('hold_invalid')) for _, r in group))
        for _, row in scored:
            rpd = masks.get((row['session_id'], row['rpc_seq']))
            details = row.get('e_rid') or {}
            k = row.get('k')
            if mode == 'rpd':
                counts['rpd_scored_late_matched' if rpd else 'rpd_scored_late_unmatched'] += 1
            if mode == 'rpd' and row.get('rpd_masks'):
                counts['rpd_inline_mask_rows'] += row['rpd_masks']
            mask_details = {}
            if rpd:
                if rpd.get('tau') != 0.4:
                    violations['rpd_tau'] += 1
                counts['rpd_late_mask_rows'] += rpd.get('rpd_masks') or 0
                mask_details.update(rpd.get('e_rid') or {})
            mask_details.update({rid: value for rid, value in details.items() if 'rpd_ok' in value})
            counts['rpd_ok_false'] += sum(value.get('rpd_ok') is False for value in mask_details.values())
            if not details:
                counts['scored_missing_e_rid'] += 1
                continue
            counts['diagnostic_rows'] += len(details)
            masses = []
            expected = []
            for detail in details.values():
                need = ('c', 'mass_d', 'gain_last', 'gain_next', 'depth_cap')
                for key in need:
                    counts[f'{key}_present'] += key in detail
                    counts[f'{key}_null'] += detail.get(key) is None
                c, mass_d, last, nxt, cap = (detail.get(key) for key in need)
                if any(key not in detail for key in need):
                    violations['missing_field'] += 1
                dep, nacc = detail.get('depth'), detail.get('accepted')
                if not (type(dep) is int and type(cap) is int and 1 <= dep <= cap and type(nacc) is int and 0 <= nacc <= dep):
                    violations['depth_or_accepted'] += 1
                    continue
                if not (number(c) and c >= 0 and isinstance(mass_d, list) and len(mass_d) == dep and all(number(x) and x >= 0 for x in mass_d)):
                    violations['mass_or_c'] += 1
                    continue
                if not ((dep == 1 and last is None) or (dep > 1 and number(last) and last >= 0)):
                    violations['gain_last'] += 1
                if not (number(nxt) and nxt >= 0):
                    violations['gain_next'] += 1
                counts['depth_capped'] += dep == cap
                counts['path_ok'] += detail.get('path_ok') is True
                mass = sum(mass_d)
                masses.append(mass)
                expected.append(c * mass)
                kmass = (detail.get('mass_k') or {}).get(str(k))
                if number(kmass):
                    counts['mass_k_comparable'] += 1
                    if not near(mass, kmass):
                        counts['mass_k_not_equal_round_k'] += 1
                for j, value in enumerate(mass_d, start=1):
                    agg = depth[j]
                    agg['rows'] += 1
                    agg['observed'] += int(nacc >= j)
                    agg['expected'] += c * value
            if len(expected) == row.get('e_n'):
                counts['mass_and_prediction_comparable'] += 1
                if not (near(sum(masses), row['e_mass']) and near(sum(expected), row['e_pred'])):
                    violations['round_mass_or_e_pred'] += 1
        for values in depth.values():
            denom = values['expected']
            values['observed_over_expected'] = values['observed'] / denom if denom > 0 else None
        exported_rounds[trial] = {'counts': counts, 'violations': violations, 'k_modes': k_modes, 'depth_calibration': dict(sorted(depth.items()))}
    results = {'draft_captures': sorted(set((int(a), int(b)) for a, b, _ in draft_caps)),
               'target_captures': sorted(set((int(a), int(b)) for a, b, _ in target_caps)),
               'flush_tp': dict(flushes), 'flush_http': dict(statuses), 'target_fit': tfit, 'draft_fit': dfit,
               'target_verify_paths': dict(paths), 'target_timing_invalid': timing_bad,
               'target_timing': {key: {'n': len(verify[key]), 'median_us': statistics.median(verify[key]) if verify[key] else None} for key in metrics},
               'late_rpd_rows': len(masks), 'round_assignment': dict(assignment), 'parse_errors': {'draft': de, 'target': te},
               'bonus_windows': {'reasons': dict(Counter(reason for _, reason, _ in windows)),
                                 'tail': 'flushed' if windows and windows[-1][1] == 'idle' else 'missing',
                                 **{k: sum(int(counter.get(k, 0)) for _, _, counter in windows) if windows else None for k in accept.BONUS_KEYS}},
               'cleanup': all((d / f'{role}-cleanup.json').exists() and not load(d / f'{role}-cleanup.json')['live_session_members']
                              and (d / f'postcheck-{role}.txt').read_bytes() == (raw / 'sync' / f'precheck-{role}.txt').read_bytes() for role in ('draft', 'target'))}
    fit_path = only(d.glob('clear_tree_fit*.json'))
    fit = load(fit_path)
    results['fit_sha256'] = hashlib.sha256(fit_path.read_bytes()).hexdigest()
    results['fit_file'] = {'name': fit_path.name, 'schema': fit['schema'], 'cells': len(fit['h']),
                           'target_mode': fit['header']['target'].get('verify_mode'),
                           'draft_lookahead': fit['header']['draft'].get('lookahead')}
    results['trials'] = exported_rounds
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--accept', type=Path, required=True)
    args = p.parse_args()
    run = args.run
    raw = run / 'raw'
    meta = load(run / 'meta.json')
    spec = importlib.util.spec_from_file_location('accept_logs', args.accept)
    accept = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(accept)
    result = {'run_id': meta['run_id'], 'commit': meta['commit'], 'trials': {}, 'modes': {}, 'issues': []}
    ids = defaultdict(dict)
    for trial in meta['trials']:
        key = trial['id']
        try:
            info, rids = client(raw, key, trial)
            result['trials'][key] = info
            ids[trial['mode']][key] = rids
        except (ValueError, OSError, KeyError, TypeError) as exc:
            result['issues'].append(f'{key}: {exc}')
    for mode in ('greedy', 'rpd'):
        try:
            result['modes'][mode] = mode_review(accept, raw, mode, ids[mode])
        except (ValueError, OSError, KeyError, TypeError) as exc:
            result['issues'].append(f'{mode}: {exc}')
    checks = {'issues': result['issues'], 'trials': {}, 'modes': {}}
    for key, row in result['trials'].items():
        ledger = result['modes'].get(row['mode'], {}).get('trials', {}).get(key, {})
        counts = ledger.get('counts', {})
        checks['trials'][key] = dict(requests=row['counts_ok'], metrics=row['metrics_scored'] == [100,100] and all(number(v) for v in row['metrics'].values()),
            valid_timing=row['timing_consistent'], flush_client=row['flush_client_ok'], spec=bool(row['total_draft'] and row['fallback'] == 0 and row['accept_ratio'] is not None),
            ledger=bool(counts.get('diagnostic_rows') and counts.get('mass_and_prediction_comparable')
                        and not ledger.get('violations') and not counts.get('hold_invalid')
                        and not counts.get('rpd_ok_false') and not counts.get('scored_missing_e_rid')))
    for mode, row in result['modes'].items():
        paths = row['target_verify_paths']
        counts = [data['counts'] for data in row['trials'].values()]
        checks['modes'][mode] = dict(cleanup=row['cleanup'], prefit=(row['draft_fit']['written'] and row['target_fit']['driver'] == 'ready'
            and all((state.get('state'), state.get('reason')) == ('ready', 'prefit') for state in row['target_fit']['states'].values())
            and set(row['target_fit']['states']) == {'0','1'} and row['fit_file']['schema'] == 3 and row['fit_file']['draft_lookahead'] is True
            and row['fit_file']['target_mode'] == (None if mode == 'greedy' else 'rpd')),
            capture=all({x[0] for x in row[f'{role}_captures']} == {1,2,4} for role in ('draft','target')),
            flush=row['flush_http'] == {'200':4} and row['flush_tp'] == {'0':4,'1':4},
            graph_verify=bool(paths.get('graph')) and not paths.get('fallback'),
            mask_evidence=(not row['late_rpd_rows'] if mode == 'greedy' else all(x.get('rpd_scored_late_matched', 0) > 0 for x in counts)),
            timing=not row['target_timing_invalid'] and all((entry['n'] == 0 if mode == 'greedy' else entry['n'] > 0)
                for key, entry in row['target_timing'].items() if key.startswith('rpd_')),
            parse=not any(row['parse_errors'].values()),
            greedy_fit_retained=(mode != 'rpd' or (raw/'rpd'/'greedy-fit-during-rpd.json').read_bytes() == (raw/'greedy'/'clear_tree_fit.json').read_bytes()))
    checks['passed'] = (not result['issues'] and len(checks['trials']) == 8 and len(checks['modes']) == 2
                        and all(all(v.values()) for group in ('trials','modes') for v in checks[group].values()))
    for name, data in (('summary', result), ('checks', checks)):
        (run / f'{name}.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    for key, row in result['trials'].items():
        print(key, row['mode'], row['dataset'], row['concurrency'], row['ok'], row['output_tokens'], row['wall_s'], row['tok_s'], row['accept_ratio'], row['metrics'])
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
