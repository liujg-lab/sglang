#!/usr/bin/env python3
"""Run the two recorded ClearTree blocks; deployment addresses are env inputs."""
import argparse
import base64
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'raw'
SYNC = RAW / 'sync'
TOOL = ROOT / 'tools'
ENV = os.environ
WORK = ENV['CT_WORK']
IDENTITY = Path(ENV['CT_IDENTITY'])
REMOTE = ENV['CT_REMOTE']
SSH = ['-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=30', '-J', ENV['CT_JUMP']]
CONFIG = {
    role: {key: ENV[f'CT_{role.upper()}_{key}'] for key in ('CONTAINER', 'ROOT', 'MODEL', 'DEVICE', 'HTTP_PORT', 'TUNNEL')}
    for role in ('draft', 'target')
}


def q(value):
    return shlex.quote(str(value))


def run(argv, *, capture=True, check=True):
    return subprocess.run(argv, check=check, stdout=subprocess.PIPE if capture else None, text=True)


def host(role, argv, **kwargs):
    if role == 'target':
        encoded = base64.b64encode(shlex.join([str(x) for x in argv]).encode()).decode()
        return run(['ssh', '-n', *SSH, REMOTE, f'printf %s {encoded} | base64 -d | bash'], **kwargs)
    return run([str(x) for x in argv], **kwargs)


def docker(role, args, **kwargs):
    return host(role, ['docker', *args], **kwargs)


def container(role, *argv, **kwargs):
    return docker(role, ['exec', CONFIG[role]['CONTAINER'], *argv], **kwargs)


def launch_argv(role, mode, action):
    cfg = CONFIG[role]
    vals = {key: cfg[key] for key in ('ROOT', 'MODEL', 'DEVICE', 'HTTP_PORT')}
    vals |= {'RPC_PORT': ENV['CT_RPC_PORT'], 'RPC_ADDR': ENV['CT_RPC_ADDR'],
             'HTTP_ADDR': ENV['CT_HTTP_ADDR'], 'PARSER': f'{WORK}/parse_argv.py'}
    return ['env', *(f'CT_{k}={v}' for k, v in vals.items()), 'bash', f'{WORK}/server.sh', role, mode, action]


def precheck(role, destination):
    cfg = CONFIG[role]
    where = [str(TOOL / 'precheck.sh'), str(IDENTITY)] if role == 'draft' else [f'{WORK}/precheck.sh', f'{WORK}/process_identity.py']
    result = host(role, ['bash', where[0], role, cfg['TUNNEL'], where[1]], check=False)
    destination.write_text(result.stdout)
    if result.returncode != 0 or result.stdout != (SYNC / f'precheck-{role}.txt').read_text():
        raise RuntimeError(f'{role} resource or tunnel precheck differs from baseline: {destination.name}')


def live(role):
    data = docker(role, ['top', CONFIG[role]['CONTAINER'], '-eo', 'pid,sid,stat']).stdout.splitlines()
    return [(int(pid), int(sid)) for pid, sid, state in (line.split() for line in data[1:]) if not state.startswith('Z')]


def identity(role, pid):
    script = str(IDENTITY) if role == 'draft' else f'{WORK}/process_identity.py'
    result = host(role, ['python3', script, str(pid)], check=False)
    return dict(line.split(maxsplit=1) for line in result.stdout.splitlines()) if result.returncode == 0 else None


def persistent_fields(state):
    return {key: state[key] for key in ('pid', 'nspid', 'container_nspid', 'sid', 'start_ticks', 'uid', 'boot_id')}


def container_identity(role, inner):
    script = 'import os,sys;p=int(sys.argv[1]);s=open(f"/proc/{p}/stat").read().rsplit(")",1)[1].split();print(s[3],s[19],os.stat(f"/proc/{p}").st_uid)'
    result = container(role, 'python3', '-B', '-c', script, str(inner), check=False)
    if result.returncode != 0:
        return None
    sid, stamp, uid = result.stdout.strip().split()
    return sid, stamp, uid


def logpart(role, block, pattern):
    result = container(role, 'awk', '-v', f'p={pattern}', '$0 ~ p {print}', f'{WORK}/{block}/{role}.log', check=False)
    return result.stdout


def ready_http(role):
    return container(role, 'curl', '-fsS', '--max-time', '5', '-o', '/dev/null', f"http://{ENV['CT_HTTP_ADDR']}:{CONFIG[role]['HTTP_PORT']}/health", check=False).returncode == 0


def copyout(role, source, dest):
    c = CONFIG[role]['CONTAINER']
    if role == 'draft':
        docker(role, ['cp', f'{c}:{source}', str(dest)])
        return
    docker(role, ['cp', f'{c}:{source}', f'{WORK}/'])
    run(['scp', '-r', *SSH, f'{REMOTE}:{WORK}/{Path(source).name}', str(dest)], capture=False)


def fit_gate(mode, out):
    role = 'draft'
    pkg = CONFIG[role]['ROOT'] + '/python/sglang/srt/speculative/clear_tree_remote'
    name = 'clear_tree_fit.json' if mode == 'greedy' else 'clear_tree_fit.rpd.json'
    source = f'{pkg}/{name}'
    dlog = logpart('draft', mode, 'no fit file|wrote fit file|target verifies with RPD|rpd_flat')
    tlog = logpart('target', mode, 'prefit started|prefit driver finished state=|CLEAR fit_state')
    (out / 'fit-gate.txt').write_text(dlog + tlog)
    assert any('no fit file' in line and source in line for line in dlog.splitlines()), 'fit absence not logged'
    writes = [line for line in dlog.splitlines() if 'wrote fit file' in line]
    assert len(writes) == 1 and source in writes[0], 'expected one fit write for this mode'
    states = {}
    for line in tlog.splitlines():
        if '[CLEAR fit_state]' in line:
            rank = re.search(r' TP(\d+)\]', line)
            if rank:
                states[rank[1]] = json.loads(line.split('[CLEAR fit_state]', 1)[1])
    assert 'prefit started' in tlog and 'prefit driver finished state=ready' in tlog, 'Target prefit driver not ready'
    assert all(states.get(rank, {}).get('state') == 'ready' and states[rank].get('reason') == 'prefit' for rank in ('0', '1')), 'Target TP prefit state not established'
    assert 'rpd_flat' not in dlog + tlog
    if mode == 'rpd':
        assert re.search(r'target verifies with RPD \(tau=0\.4\)', dlog), 'missing RPD tau acknowledgement'
    fit = json.loads(container(role, 'python3', '-B', '-c', 'import sys;print(open(sys.argv[1]).read())', source).stdout)
    assert fit.get('schema') == 3 and fit['header']['draft'].get('lookahead') is True, 'fit schema or lookahead differs'
    assert fit['header']['target'].get('verify_mode') == ('rpd' if mode == 'rpd' else None), 'fit verification mode differs'
    copyout(role, source, out / name)
    captures = {}
    for role in ('draft', 'target'):
        lines = logpart(role, mode, 'Capturing: bs=')
        (out / f'{role}-captures.txt').write_text(lines)
        captures[role] = {(int(a), int(b)) for a, b in re.findall(r'Capturing: bs=(\d+), ntpb=(\d+)', lines)}
        assert {a for a, _ in captures[role]} == {1, 2, 4}, f'{role} capture batch differs'
    expected_t = {(a, b) for a in (1, 2, 4) for b in (1, 3, 5, 9, 17, 33, 65)}
    expected_d = {(a, b) for a in (1, 2, 4) for b in (1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 64, 65)} - {(4, 65)}
    assert expected_t <= captures['target'] and expected_d <= captures['draft'], 'required capture shape missing'


def start(role, block, out, owned):
    precheck(role, out / f'precheck-{role}.txt')
    parsed = container(role, *launch_argv(role, block, 'parse')).stdout
    (out / f'parse-{role}.txt').write_text(parsed)
    work = f'{WORK}/{block}'
    container(role, 'mkdir', work)
    command = shlex.join(launch_argv(role, block, 'serve'))
    shell = f'printf "%s\\n" "$$" > {q(work + "/leader.pid")}; exec {command} > {q(work + "/" + role + ".log")} 2>&1'
    docker(role, ['exec', '-d', CONFIG[role]['CONTAINER'], 'setsid', '--wait', 'bash', '-c', shell])
    for _ in range(20):
        result = container(role, 'awk', 'NR==1 {print $1}', f'{work}/leader.pid', check=False)
        if result.returncode == 0 and result.stdout.strip().isdigit():
            inner = result.stdout.strip()
            for pid, sid in live(role):
                if pid != sid:
                    continue
                state = identity(role, pid)
                if state and state['container_nspid'] == inner and state['state'] != 'Z':
                    owned[role] = persistent_fields(state)
                    (out / f'{role}-identity.json').write_text(json.dumps(owned[role], indent=2) + '\n')
                    print(f'{block} {role} launched; own host pid {pid}', flush=True)
                    return
        time.sleep(1)
    raise RuntimeError(f'cannot establish identity of {role} launcher; no blind signaling')


def wait_ready(role, block, owner, prefit=False):
    last_progress = None
    while True:
        state = identity(role, owner['pid'])
        if not state or state['state'] == 'Z' or persistent_fields(state) != owner:
            raise RuntimeError(f'{role} process exited during start')
        if prefit:
            status = [line.split('prefit driver finished state=', 1)[1].split()[0] for line in logpart(role, block, 'prefit driver finished state=').splitlines()]
            if status and status[-1] != 'ready':
                raise RuntimeError(f'{role} prefit status: {status[-1]}')
            if status and ready_http(role):
                return
        elif ready_http(role):
            return
        stat = container(role, 'stat', '-c', '%s', f'{WORK}/{block}/{role}.log', check=False).stdout.strip()
        if stat != last_progress:
            print(f'{block} {role} startup log {stat} bytes', flush=True)
            last_progress = stat
        time.sleep(20)


def stop(role, owner, out):
    sid = int(owner['sid'])
    current = identity(role, owner['pid'])
    if current and persistent_fields(current) != owner:
        raise RuntimeError(f'{role} owner identity changed; refusing signals')
    evidence = []
    visited = set()
    for _ in range(30):
        members = [pid for pid, sess in live(role) if sess == sid]
        if not members:
            (out / f'{role}-cleanup.json').write_text(json.dumps({'signals': evidence, 'live_session_members': []}, indent=2) + '\n')
            return
        for pid in members:
            member = identity(role, pid)
            if not member or member['state'] == 'Z':
                continue
            if member['sid'] != str(sid) or member['boot_id'] != owner['boot_id']:
                raise RuntimeError(f'{role} session or boot identity changed')
            token = (str(pid), member['start_ticks'])
            if token in visited:
                continue
            inner = member['container_nspid']
            within = container_identity(role, inner)
            if within is None:
                continue
            if within != (owner['container_nspid'], member['start_ticks'], member['uid']):
                raise RuntimeError(f'{role} container identity changed')
            again = identity(role, pid)
            if not again or persistent_fields(again) != persistent_fields(member):
                continue
            result = container(role, 'kill', '-TERM', '--', inner, check=False)
            evidence.append(dict(persistent_fields(member), term_exit=result.returncode))
            visited.add(token)
        time.sleep(2)
    raise RuntimeError(f'{role} owned session still live after TERM')


def client(trial, dataset, conc):
    out = RAW / trial
    out.mkdir()
    dest = f'{WORK}/{trial}'
    container('target', 'mkdir', dest)
    data = ENV[f'CT_{dataset.upper()}_DATA']
    img = ENV[f'CT_{dataset.upper()}_IMG']
    args = ['env', f'CT_HTTP_ADDR={ENV["CT_HTTP_ADDR"]}', f'CT_HTTP_PORT={CONFIG["target"]["HTTP_PORT"]}', f'CT_CLIENT_DIR={WORK}/client', 'bash', f'{WORK}/run-client.sh', trial, dataset, str(conc), data, img, dest]
    print(f'formal {trial} {dataset} concurrency={conc} started', flush=True)
    result = container('target', 'bash', '-c', f'{shlex.join(args)} > {q(dest + "/client.stdout")} 2>&1', capture=False, check=False)
    copyout('target', dest, RAW)
    (out / 'exit-code.txt').write_text(f'{result.returncode}\n')
    files = list(out.glob('client.*.jsonl'))
    if result.returncode or len(files) != 1:
        raise RuntimeError(f'formal {trial} client exit {result.returncode} or output missing')
    rows = [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]
    if len(rows) != 100 or any(row.get('error') for row in rows):
        raise RuntimeError(f'formal {trial} contained failed or missing requests')
    print(f'formal {trial} completed 100/100; artifacts copied', flush=True)


def block(mode, adopt=False):
    out = RAW / mode
    owned = {}
    if adopt:
        for role in ('draft', 'target'):
            saved = json.loads((out / f'{role}-identity.json').read_text())
            current = identity(role, saved['pid'])
            if not current or current['state'] == 'Z' or persistent_fields(current) != saved:
                raise RuntimeError(f'{role} no longer matches the recorded launch identity')
            owned[role] = saved
        first = 1 if mode == 'greedy' else 5
        if any((RAW / f'{n:02}').exists() for n in range(first, first + 4)):
            raise RuntimeError('cannot adopt a block after any client was started')
        print(f'{mode} resumed supervisor on the same verified Draft/Target sessions; no service restart', flush=True)
    else:
        out.mkdir()
    failure = False
    try:
        if mode == 'rpd':
            expected = (RAW / 'greedy' / 'clear_tree_fit.json').read_bytes()
        if not adopt:
            start('draft', mode, out, owned)
            wait_ready('draft', mode, owned['draft'])
            print(f'{mode} Draft HTTP ready', flush=True)
            start('target', mode, out, owned)
        wait_ready('target', mode, owned['target'], prefit=True)
        fit_gate(mode, out)
        print(f'{mode} Target HTTP and refit ready; capture gate passed', flush=True)
        if mode == 'rpd':
            fit = CONFIG['draft']['ROOT'] + '/python/sglang/srt/speculative/clear_tree_remote/clear_tree_fit.json'
            copyout('draft', fit, out / 'greedy-fit-during-rpd.json')
            assert (out / 'greedy-fit-during-rpd.json').read_bytes() == expected, 'Greedy fit overwritten'
        first = 1 if mode == 'greedy' else 5
        for n, (dataset, conc) in enumerate((('rsteller', 1), ('rsteller', 4), ('watercaption', 1), ('watercaption', 4)), first):
            client(f'{n:02}', dataset, conc)
    except BaseException:
        failure = True
        raise
    finally:
        errors = []
        for role in ('target', 'draft'):
            if role in owned:
                try:
                    stop(role, owned[role], out)
                except Exception as exc:
                    errors.append(f'{role} stop: {exc}')
            if (role in owned or container(role, 'test', '-f', f'{WORK}/{mode}/{role}.log', check=False).returncode == 0):
                try:
                    copyout(role, f'{WORK}/{mode}/{role}.log', out / f'{role}.log')
                except Exception as exc:
                    errors.append(f'{role} log copy: {exc}')
            try:
                precheck(role, out / f'postcheck-{role}.txt')
            except Exception as exc:
                errors.append(f'{role} postcheck: {exc}')
        if errors:
            (out / 'cleanup-errors.txt').write_text('\n'.join(errors) + '\n')
            print('\n'.join(errors), file=sys.stderr, flush=True)
            if not failure:
                raise RuntimeError('cleanup evidence incomplete')
        else:
            print(f'{mode} cleaned up: ports, devices and tunnel identities match baseline', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('greedy', 'rpd'))
    parser.add_argument('--adopt', action='store_true', help='Resume before formal traffic on the same recorded sessions')
    args = parser.parse_args()
    block(args.mode, args.adopt)
