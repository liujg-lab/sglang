#!/usr/bin/env python3
"""Supervise one fresh ClearTree mode of the 2026-11-27-c RSTeller run."""

import argparse
import base64
import hashlib
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'raw'
SYNC = RAW / 'sync'
ROLES = ('draft', 'target')
TRIALS = {'greedy': (('01', 1), ('02', 4)), 'rpd': (('03', 1), ('04', 4))}
NAMES = {'greedy': 'clear_tree_fit.json', 'rpd': 'clear_tree_fit.rpd.json'}
BATCHES = (1, 2, 4)
TIERS = (2, 4, 8, 16, 32, 64)
BUCKETS = (256, 512, 1024, 2048, 4096)
IDENTITY_KEYS = ('pid', 'nspid', 'container_nspid', 'sid', 'start_ticks', 'uid', 'boot_id')

# Compare the identity from inside the container immediately before signaling.
# Use a pidfd where the actual Python/kernel provide it to bind the signal to
# that process instance. Treat the exit of a concurrently disappearing member
# as normal; identity/permission errors are not silently treated as exits.
SIGNAL_MEMBER = '''
import errno, os, signal, sys
pid, sid, stamp, uid = map(int, sys.argv[1:])
fd = None
try:
    if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
        try:
            fd = os.pidfd_open(pid)
        except OSError as exc:
            if exc.errno not in (errno.ENOSYS, errno.EPERM):
                raise
    path = f'/proc/{pid}'
    with open(path + '/stat') as f:
        fields = f.read().rsplit(')', 1)[1].split()
    if fields[0] == 'Z':
        sys.exit(4)
    if (int(fields[3]), int(fields[19]), os.stat(path).st_uid) != (sid, stamp, uid):
        sys.exit(5)
    if fd is not None:
        signal.pidfd_send_signal(fd, signal.SIGTERM)
    else:
        os.kill(pid, signal.SIGTERM)
except (FileNotFoundError, ProcessLookupError):
    sys.exit(4)
finally:
    if fd is not None:
        os.close(fd)
'''


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def env(key):
    value = os.environ.get(key)
    require(value, f'missing environment variable {key}')
    return value


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha_bytes(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def persistent(state):
    return {key: state[key] for key in IDENTITY_KEYS}


def fit_coverage(fit, mode, draft_captures):
    """Validate the persisted shape and coverage of this fixed HEAD plan."""
    header = fit['header']
    require(fit['schema'] == header['schema'] == 4, 'expected schema 4 fit/header')
    require(header['draft']['lookahead'] is True, 'Draft lookahead absent from fit')
    target = header['target']
    require((target.get('verify_mode') == 'rpd') if mode == 'rpd' else ('verify_mode' not in target),
            'fit Target identity does not match selected mode')
    # rpd_tau is not persisted in the Target identity (check the handshake log).
    inputs = dict(batches=list(BATCHES), ks=list(TIERS), widths=list(TIERS),
                  buckets=list(BUCKETS), max_depth=8)
    plan = fit['plan']
    require(plan['inputs'] == inputs and all(header[k] == inputs[k] for k in inputs if k != 'batches'),
            'FIT plan does not match this run')
    require(target['kv_buckets'] == inputs['buckets'], 'FIT Target buckets differ')
    fp_values = dict(inputs, verify_budget_version=1, layer_sample_version=1, layer_samples=50,
                     width_sample_target=60, width_min_depth=4, width_rounds_per_width=4)
    fingerprint = hashlib.sha1(json.dumps(fp_values, separators=(',', ':'), sort_keys=True).encode()).hexdigest()[:8]
    require(plan['fingerprint'] == fingerprint, 'FIT plan fingerprint differs')

    # For these buckets every verify group fits, including K=64 at bucket 256.
    h_expected = {(b, k, bucket) for b in BATCHES for k in TIERS for bucket in BUCKETS}
    # Width sweeps plan >=60 layer-2+ samples for each non-flat captured width;
    # the verify groups independently exceed 50 for each flat width. Thus the
    # only possible skips in this fixed plan are uncaptured graph layer cells.
    layer_candidates = {(w, 'graph', b) for b in BATCHES for w in TIERS}
    layer_expected = {cell for cell in layer_candidates if (cell[2], cell[0]) in draft_captures}
    coverage = fit['coverage']
    for key, expected in (('required_h', h_expected), ('required_layers', layer_expected)):
        keys = [tuple(cell) for cell in coverage[key]]
        require(len(keys) == len(expected) and set(keys) == expected, f'FIT {key} differs from the capture/plan')
    skipped = [(tuple(row['cell']), row['reason']) for row in coverage['skipped']]
    expected_skipped = {(cell, 'no_graph') for cell in layer_candidates - layer_expected}
    require(len(skipped) == len(expected_skipped) and set(skipped) == expected_skipped,
            'unexpected FIT skipped cells (none of the captured widths are low_yield in this plan)')

    hs, layers = {}, {}
    for row in fit['h']:
        key = (row['b'], row['k'], row['bucket'])
        ms, ells = row['samples_ms'], row['ells']
        require(key not in hs and len(ms) == len(ells) and
                all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in ms) and
                all(v is None or (type(v) is int and v >= 0) for v in ells), f'invalid H row {key}')
        hs[key] = len(ms)
    for row in fit['layers']:
        key = (row['w'], row['path'], row['g'])
        n, xtx, xty = row['n'], row['xtx'], row['xty']
        require(key not in layers and type(n) is int and n >= 0 and
                len(xtx) == len(xty) == 2 and all(len(r) == 2 for r in xtx) and
                all(type(v) in (int, float) and math.isfinite(v) for v in [*xtx[0], *xtx[1], *xty]),
                f'invalid layer regressor {key}')
        require(all(row[k] is None or (type(row[k]) in (float, int) and math.isfinite(row[k]))
                    for k in ('span_min', 'span_max')), f'invalid layer span {key}')
        if key in layer_expected:
            # Two-feature ridge system as stored by _new_layer_model (ridge
            # on the prefix-sum coefficient only, not the intercept).
            a, b = xtx[0]
            c, d = xtx[1]
            require(a > 0 and d + 128**2 - (c / a) * b > 0, f'unsolvable layer regressor {key}')
        layers[key] = n
    require(all(hs.get(key, 0) >= 4 for key in h_expected), 'required H cell has fewer than 4 samples')
    require(all(layers.get(key, 0) >= 50 for key in layer_expected), 'required layer cell has fewer than 50 samples')
    return dict(schema=4, fingerprint=fingerprint, run=fit['run'], required_h=len(h_expected),
                min_h_samples=min(hs[key] for key in h_expected), required_layers=len(layer_expected),
                min_layer_samples=min(layers[key] for key in layer_expected), skipped=coverage['skipped'],
                low_yield=[row for row in coverage['skipped'] if row['reason'] == 'low_yield'])


class Driver:
    def __init__(self):
        self.work = env('CT_WORK')
        require(re.fullmatch(r'/[A-Za-z0-9_./-]+', self.work) and
                str(PurePosixPath(self.work)) == self.work and '..' not in PurePosixPath(self.work).parts,
                'CT_WORK must be a normalized absolute path without spaces')
        self.remote = env('CT_REMOTE')
        self.ssh = ['-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10',
                    '-o', 'ServerAliveInterval=30', '-J', env('CT_JUMP')]
        self.shared = {key: env('CT_' + key) for key in ('RPC_PORT', 'RPC_ADDR', 'HTTP_ADDR')}
        self.cfg = {role: {key: env(f'CT_{role.upper()}_{key}')
                           for key in ('CONTAINER', 'ROOT', 'MODEL', 'DEVICE', 'HTTP_PORT', 'TUNNEL')}
                    for role in ROLES}
        self.identity_path = {'draft': env('CT_IDENTITY'), 'target': f'{self.work}/process_identity.py'}
        self.precheck_path = {
            'draft': os.environ.get('CT_PRECHECK') or str(ROOT / 'tools' / 'precheck.sh'),
            'target': os.environ.get('CT_TARGET_PRECHECK') or f'{self.work}/precheck.sh',
        }
        self.data, self.images = env('CT_RSTELLER_DATA'), env('CT_RSTELLER_IMG')

    @staticmethod
    def run(argv, *, label, capture=True, check=True):
        # Do not echo argv or configuration values in our diagnostics.
        result = subprocess.run(argv, stdout=subprocess.PIPE if capture else None, text=True)
        require(not check or result.returncode == 0, f'{label} exited {result.returncode}')
        return result

    def host(self, role, argv, **kwargs):
        argv = [str(x) for x in argv]
        if role == 'target':
            encoded = base64.b64encode(shlex.join(argv).encode()).decode()
            argv = ['ssh', '-n', *self.ssh, self.remote, f'printf %s {encoded} | base64 -d | bash']
        return self.run(argv, **kwargs)

    def docker(self, role, *args, **kwargs):
        return self.host(role, ['docker', *args], label=f'{role} docker {args[0]}', **kwargs)

    def container(self, role, *argv, **kwargs):
        return self.docker(role, 'exec', self.cfg[role]['CONTAINER'], *argv, **kwargs)

    def launch_argv(self, role, mode, action):
        vals = {key: self.cfg[role][key] for key in ('ROOT', 'MODEL', 'DEVICE', 'HTTP_PORT')}
        vals.update(self.shared, PARSER=f'{self.work}/parse_argv.py')
        return ['env', *(f'CT_{k}={v}' for k, v in vals.items()), 'bash',
                f'{self.work}/server.sh', role, mode, action]

    def validate_argv(self, role, mode, dest):
        result = self.container(role, *self.launch_argv(role, mode, 'parse'), check=False)
        dest.write_text(result.stdout)
        require(result.returncode == 0, f'{role} server settings parser failed')
        backend = 'triton' if role == 'draft' else 'ascend'
        tiers = ','.join(str(n) for n in TIERS)
        expected = dict(role=role, capacity='64' if role == 'draft' else '65',
                        widths=tiers, ks=tiers, depth='8', steps='8', cuda_graph_bs='1,2,4',
                        page='1' if role == 'draft' else '128', attention_backend=backend,
                        mm_attention_backend=f'{backend}_attn')
        require(set(f'{key} {value}' for key, value in expected.items()) <= set(result.stdout.splitlines()),
                f'{role} resolved settings differ from the fixed matrix')

    def precheck(self, role, dest):
        baseline = (SYNC / f'precheck-{role}.txt').read_text()
        result = self.host(role, ['bash', self.precheck_path[role], role, self.cfg[role]['TUNNEL'],
                                  self.identity_path[role]], label=f'{role} precheck', check=False)
        dest.write_text(result.stdout)
        require(result.returncode == 0 and result.stdout == baseline,
                f'{role} precheck differs from baseline: {dest.name}')

    def live(self, role):
        rows = self.docker(role, 'top', self.cfg[role]['CONTAINER'], '-eo', 'pid,sid,stat').stdout.splitlines()
        require(rows and rows[0].split() == ['PID', 'SID', 'STAT'], f'{role} unexpected process listing')
        return [(int(pid), int(sid)) for pid, sid, state in (line.split() for line in rows[1:])
                if not state.startswith('Z')]

    def identity(self, role, pid):
        result = self.host(role, ['python3', '-B', self.identity_path[role], pid],
                           label=f'{role} process identity', check=False)
        if result.returncode == 0:
            return dict(line.split(maxsplit=1) for line in result.stdout.splitlines())
        # Establish actual absence before mistaking a transport or helper error for exit.
        present = self.host(role, ['test', '-d', f'/proc/{pid}'], label=f'{role} pid presence', check=False)
        require(present.returncode == 1, f'{role} process identity could not be read')
        return None

    def discover_owner(self, role, mode):
        for _ in range(20):
            result = self.container(role, 'cat', f'{self.work}/{mode}/leader.pid', check=False)
            parts = result.stdout.split()
            if result.returncode == 0 and len(parts) == 2 and all(x.isdigit() for x in parts):
                inner, start_ticks = parts
                found = []
                for pid, sid in self.live(role):
                    if pid == sid:
                        state = self.identity(role, pid)
                        if (state and state['container_nspid'] == inner and state['state'] != 'Z'
                                and state['start_ticks'] == start_ticks):
                            found.append(persistent(state))
                require(len(found) <= 1, f'{role} launcher identity is ambiguous')
                if found:
                    return found[0]
            time.sleep(1)
        raise RuntimeError(f'cannot establish {role} launcher identity; refusing blind signals')

    def start(self, role, mode, out, attempted, owned):
        self.precheck(role, out / f'precheck-{role}.txt')
        work = f'{self.work}/{mode}'
        self.container(role, 'mkdir', work)
        command = shlex.join(self.launch_argv(role, mode, 'serve'))
        shell = ('read -r stat < "/proc/$$/stat" && fields=${stat##*) } && '
                 'read -r -a proc <<< "$fields" && '
                 f'printf "%s %s\\n" "$$" "${{proc[19]}}" > {shlex.quote(work + "/leader.pid")} && '
                 f'exec {command} > {shlex.quote(work + "/" + role + ".log")} 2>&1')
        attempted.add(role)
        self.docker(role, 'exec', '-d', self.cfg[role]['CONTAINER'], 'setsid', '--wait', 'bash', '-c', shell)
        owned[role] = self.discover_owner(role, mode)
        write_json(out / f'{role}-identity.json', owned[role])
        print(f'{mode} {role} launched; own host pid {owned[role]["pid"]}', flush=True)

    def assert_owner(self, role, owner):
        state = self.identity(role, owner['pid'])
        require(state and state['state'] != 'Z' and persistent(state) == owner,
                f'{role} is no longer the live owned process')

    def logpart(self, role, mode, pattern):
        return self.container(role, 'awk', '-v', f'p={pattern}', '$0 ~ p {print}',
                              f'{self.work}/{mode}/{role}.log').stdout

    def ready_http(self, role):
        host = self.shared['HTTP_ADDR']
        port = self.cfg[role]['HTTP_PORT']
        result = self.container(role, 'curl', '-s', '--max-time', '5', '-o', '/dev/null',
                                '-w', '%{http_code}', f'http://{host}:{port}/health', check=False)
        return result.returncode == 0 and result.stdout == '200'

    def wait_ready(self, role, mode, owned):
        last_progress = None
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            for peer, owner in owned.items():
                self.assert_owner(peer, owner)
            if role == 'draft':
                if self.ready_http(role):
                    return
            else:
                text = self.logpart(role, mode, 'prefit driver finished state=|CLEAR fit_state')
                if re.search(r'prefit driver finished state=(?!ready\b)\S+', text):
                    raise RuntimeError('Target prefit driver did not finish ready')
                states = self.fit_states(text)
                if any(s.get('state') in ('failed', 'disabled', 'legacy', 'breaker') for s in states.values()):
                    raise RuntimeError('Target entered a terminal non-ready FIT state')
                # The driver emits its finish line before TP0 logs ready;
                # TP1 mirrors the terminal state on a subsequent tick.
                if ('prefit driver finished state=ready' in text and
                        all(states.get(rank, {}).get('state') == 'ready' for rank in ('0', '1')) and
                        self.ready_http(role)):
                    return
            size = self.container(role, 'stat', '-c', '%s', f'{self.work}/{mode}/{role}.log').stdout.strip()
            if size != last_progress:
                print(f'{mode} {role} startup log {size} bytes', flush=True)
                last_progress = size
            time.sleep(20)
        raise RuntimeError(f'{role} readiness deadline exceeded')

    @staticmethod
    def fit_states(text):
        states = {}
        for line in text.splitlines():
            prefix, sep, payload = line.partition('[CLEAR fit_state]')
            rank = re.search(r'\bTP(\d+)\]', prefix)
            if sep and rank:
                # Scheduler stdout from different ranks can share the line;
                # only the leading complete JSON object is the FIT record.
                states[rank[1]] = json.JSONDecoder().raw_decode(payload.lstrip())[0]
        return states

    def fit_path(self, role, mode):
        return f"{self.cfg[role]['ROOT']}/python/sglang/srt/speculative/clear_tree_remote/{NAMES[mode]}"

    def require_absent(self, role, mode):
        path = self.fit_path(role, mode)
        self.container(role, 'bash', '-c', '[[ ! -e $1 && ! -L $1 && ! -e $1.tmp && ! -L $1.tmp ]]',
                       'fit-absence', path)

    def copyout(self, role, source, dest):
        container = self.cfg[role]['CONTAINER']
        if role == 'draft':
            self.docker(role, 'cp', f'{container}:{source}', dest)
        else:
            self.docker(role, 'cp', f'{container}:{source}', f'{self.work}/')
            self.run(['scp', '-r', *self.ssh, f'{self.remote}:{self.work}/{Path(source).name}', str(dest)],
                     label='Target artifact copy', capture=False)

    def draft_digest(self, mode):
        text = self.container('draft', 'sha256sum', self.fit_path('draft', mode)).stdout
        fields = text.split()
        require(fields and re.fullmatch(r'[0-9a-f]{64}', fields[0]), f'invalid Draft {mode} fit digest')
        return fields[0]

    def fit_gate(self, mode, out, owned):
        source = self.fit_path('draft', mode)
        dlog = self.logpart('draft', mode, 'no fit file|wrote fit file|target verifies with RPD|rpd_flat')
        tlog = self.logpart('target', mode, 'prefit started|prefit driver finished state=|CLEAR fit_state|rpd_flat')
        (out / 'fit-gate.txt').write_text(dlog + tlog)
        require(sum(line.endswith(f'no fit file at {source}') for line in dlog.splitlines()) == 1,
                'active fit absence was not logged exactly once')
        writes = [line for line in dlog.splitlines() if 'wrote fit file' in line]
        require(len(writes) == 1 and writes[0].endswith(f'wrote fit file {source} cells=90'),
                'expected exactly one active-mode fit write for the 90 H cells')
        require(tlog.count('[CLEAR] prefit started') == 1 and
                re.findall(r'prefit driver finished state=(\w+)', tlog) == ['ready'],
                'expected one successful Target prefit driver')
        require('rpd_flat' not in dlog + tlog, 'legacy RPD FIT bypass appeared')
        ack = [line for line in dlog.splitlines() if 'target verifies with RPD' in line]
        if mode == 'rpd':
            require(len(ack) == 1 and ack[0].endswith(f'target verifies with RPD (tau=0.4); W/K run online on fit file {source}'),
                    'missing RPD tau=0.4 acknowledgement for the RPD file')
        else:
            require(not ack, 'Greedy run negotiated RPD')
        local = out / NAMES[mode]
        self.copyout('draft', source, local)
        digest = sha_bytes(local)
        require(digest == self.draft_digest(mode), 'Draft FIT artifact digest differs from copied bytes')
        (out / f'{NAMES[mode]}.sha256').write_text(f'{digest}  {NAMES[mode]}\n')
        fit = json.loads(local.read_text())
        states = self.fit_states(tlog)
        for rank in ('0', '1'):
            record = states.get(rank, {})
            require((record.get('state'), record.get('reason'), record.get('run')) ==
                    ('ready', 'prefit', fit['run']) and record.get('session'),
                    f'Target TP{rank} did not report ready/prefit for the saved FIT generation')
        session = states['0']['session']
        require(states['1']['session'] == session, 'Target ranks disagree on the FIT session')
        session_prefix = ''.join(ch for ch in session if ch.isalnum() or ch == '_')
        require(fit['run'].startswith(f"{session_prefix}_{fit['plan']['fingerprint']}_") and
                fit['run'].rsplit('_', 1)[-1].isdigit(), 'FIT generation does not match session and fingerprint')

        captures = {}
        for role in ROLES:
            text = self.logpart(role, mode, 'Capturing: bs=')
            (out / f'{role}-captures.txt').write_text(text)
            captures[role] = {(int(a), int(b)) for a, b in re.findall(r'Capturing: bs=(\d+), ntpb=(\d+)', text)}
            require({a for a, _ in captures[role]} == set(BATCHES), f'{role} capture batches differ')
        target_shapes = {(a, n) for a in BATCHES for n in (1, 3, 5, 9, 17, 33, 65)}
        draft_shapes = target_shapes | {(a, n) for a in BATCHES for n in TIERS}
        require(target_shapes <= captures['target'] and
                draft_shapes - {(4, 65)} <= captures['draft'] and (4, 65) not in captures['draft'],
                'required graph captures missing or Draft captured an over-capacity shape')
        evidence = fit_coverage(fit, mode, captures['draft'])
        for role in ROLES:
            self.assert_owner(role, owned[role])
            require(self.ready_http(role), f'{role} HTTP not 200 at FIT gate')
        write_json(out / 'fit-coverage.json', dict(evidence, session=session, sha256=digest,
                   draft_health=200, target_health=200, draft_captures=len(captures['draft']),
                   target_captures=len(captures['target']), draft_capacity_skip=[4, 65]))

    def stop(self, role, owner, out):
        sid = int(owner['sid'])
        evidence = {'signals': [], 'live_session_members': None}
        visited = set()
        try:
            for _ in range(30):
                current = self.identity(role, owner['pid'])
                require(not current or persistent(current) == owner,
                        f'{role} leader identity changed; refusing signals')
                members = [pid for pid, sess in self.live(role) if sess == sid]
                evidence['live_session_members'] = members
                if not members:
                    return
                for pid in members:
                    member = self.identity(role, pid)
                    if not member or member['state'] == 'Z':
                        continue
                    require(member['sid'] == str(sid) and member['boot_id'] == owner['boot_id'] and
                            int(member['start_ticks']) >= int(owner['start_ticks']),
                            f'{role} session/boot/start identity changed; refusing signals')
                    token = (pid, member['start_ticks'])
                    if token in visited:
                        continue
                    again = self.identity(role, pid)
                    if not again or again['state'] == 'Z' or persistent(again) != persistent(member):
                        continue
                    result = self.container(role, 'python3', '-B', '-c', SIGNAL_MEMBER,
                                            member['container_nspid'], owner['container_nspid'],
                                            member['start_ticks'], member['uid'], check=False)
                    evidence['signals'].append(dict(persistent(member), term_exit=result.returncode))
                    require(result.returncode in (0, 4), f'{role} verified TERM failed ({result.returncode})')
                    visited.add(token)
                time.sleep(2)
            raise RuntimeError(f'{role} owned session still live after TERM')
        finally:
            write_json(out / f'{role}-cleanup.json', evidence)

    def client(self, mode, trial, concurrency, ordinal):
        out = RAW / trial
        out.mkdir()
        dest = f'{self.work}/{trial}'
        self.container('target', 'mkdir', dest)
        args = ['env', f"CT_HTTP_ADDR={self.shared['HTTP_ADDR']}",
                f"CT_HTTP_PORT={self.cfg['target']['HTTP_PORT']}", f'CT_CLIENT_DIR={self.work}/client',
                'bash', f'{self.work}/run-client.sh', trial, 'rsteller', str(concurrency),
                self.data, self.images, dest]
        print(f'formal {trial} rsteller concurrency={concurrency} started', flush=True)
        result = self.container('target', 'bash', '-c',
                                f'{shlex.join(args)} > {shlex.quote(dest + "/client.stdout")} 2>&1',
                                capture=False, check=False)
        (out / 'exit-code.txt').write_text(f'{result.returncode}\n')
        self.copyout('target', dest, RAW)
        paths = list(out.glob('client.*.jsonl'))
        require(result.returncode == 0 and len(paths) == 1, f'formal {trial} failed or has no unique JSONL')
        with paths[0].open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        require(len(rows) == 100 and all(not row.get('error') for row in rows),
                f'formal {trial} contained failed or missing requests')
        require(paths[0].with_suffix('.csv').stat().st_size and paths[0].with_suffix('.run.json').stat().st_size,
                f'formal {trial} has empty CSV or manifest')
        metrics = ('bleu', 'rouge1', 'rouge2', 'rougeL')
        require(all(all(type(row.get('metrics', {}).get(key)) in (int, float) for key in metrics)
                    for row in rows), f'formal {trial} did not score 100/100 rows')
        require(all(row.get('spec_fallback_ct') == 0 for row in rows),
                f'formal {trial} detected fallback or missing fallback counters')
        flush = self.logpart('target', mode, 'Cache flushed successfully|POST /flush_cache')
        (out / 'target-flush.txt').write_text(flush)
        require(re.findall(r'"POST /flush_cache HTTP/\S+" (\d+)', flush) == ['200'] * ordinal and
                all(len(re.findall(rf'\bTP{rank}\].*Cache flushed successfully', flush)) == ordinal
                    for rank in (0, 1)), f'formal {trial} did not get one HTTP 200 flush confirmed by both ranks')
        print(f'formal {trial} completed and scored 100/100; artifacts copied', flush=True)

    def block(self, mode):
        for role in ROLES:
            require((SYNC / f'precheck-{role}.txt').is_file(), f'missing {role} precheck baseline')
        require(not any((RAW / trial).exists() for trial, _ in TRIALS[mode]), 'formal output already exists; no retry')
        previous = None
        if mode == 'rpd':
            greedy = RAW / 'greedy'
            require((greedy / 'block-complete.txt').is_file(), 'Greedy must finish and clean up before RPD')
            previous = sha_bytes(greedy / NAMES['greedy'])
        out = RAW / mode
        out.mkdir()  # Reserve this mode permanently; no adopt or retry path.
        owned, attempted = {}, set()
        failure = None
        try:
            for role in ROLES:
                self.precheck(role, out / f'before-parse-{role}.txt')
                self.validate_argv(role, mode, out / f'parse-{role}.txt')
                self.require_absent(role, mode)
                if mode == 'greedy' or role == 'target':
                    self.require_absent(role, 'rpd' if mode == 'greedy' else 'greedy')
            if previous is not None:
                require(self.draft_digest('greedy') == previous, 'Draft Greedy fit differs before RPD launch')
            for role in ROLES:
                self.start(role, mode, out, attempted, owned)
                self.wait_ready(role, mode, owned)
                print(f'{mode} {role} HTTP ready', flush=True)
            self.fit_gate(mode, out, owned)
            if previous is not None:
                saved = out / 'greedy-fit-during-rpd.json'
                self.copyout('draft', self.fit_path('draft', 'greedy'), saved)
                require(sha_bytes(saved) == previous, 'RPD prefit overwrote Greedy fit')
                (out / 'greedy-fit-during-rpd.json.sha256').write_text(f'{previous}  {saved.name}\n')
            print(f'{mode} FIT generation, coverage and capture gate passed', flush=True)
            for ordinal, (trial, concurrency) in enumerate(TRIALS[mode], 1):
                for role in ROLES:
                    self.assert_owner(role, owned[role])
                self.client(mode, trial, concurrency, ordinal)
        except BaseException as exc:
            failure = exc
            raise
        finally:
            errors = []
            if attempted:
                for role in ('target', 'draft'):
                    if role not in attempted:
                        continue
                    try:
                        if role not in owned:
                            owned[role] = self.discover_owner(role, mode)
                            write_json(out / f'{role}-identity.json', owned[role])
                        self.stop(role, owned[role], out)
                    except Exception as exc:
                        errors.append(f'{role} stop: {exc}')
                for role in ('target', 'draft'):
                    if role in attempted:
                        try:
                            self.copyout(role, f'{self.work}/{mode}/{role}.log', out / f'{role}.log')
                        except Exception as exc:
                            errors.append(f'{role} log copy: {exc}')
                    try:
                        self.precheck(role, out / f'postcheck-{role}.txt')
                    except Exception as exc:
                        errors.append(f'{role} postcheck: {exc}')
                if previous is not None:
                    try:
                        digest = self.draft_digest('greedy')
                        (out / 'greedy-fit-after-rpd.sha256').write_text(f'{digest}  {NAMES["greedy"]}\n')
                        require(digest == previous, 'RPD block overwrote Greedy fit')
                    except Exception as exc:
                        errors.append(f'Greedy FIT preservation: {exc}')
            if errors:
                (out / 'cleanup-errors.txt').write_text('\n'.join(errors) + '\n')
                print('\n'.join(errors), file=sys.stderr, flush=True)
                if failure is None:
                    raise RuntimeError('cleanup evidence incomplete')
        (out / 'block-complete.txt').write_text('two formal trials completed; owned sessions stopped and postchecks matched\n')
        print(f'{mode} cleaned up: ports, devices and tunnel identities match baseline', flush=True)


def interrupted(signum, _frame):
    # First supervisor signal enters cleanup; a second cannot interrupt it.
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, signal.SIG_IGN)
    raise KeyboardInterrupt(f'interrupted by signal {signum}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=tuple(TRIALS))
    args = parser.parse_args()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        Driver().block(args.mode)
    except (RuntimeError, OSError, ValueError, KeyError, KeyboardInterrupt) as exc:
        print(f'ClearTree {args.mode}: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
