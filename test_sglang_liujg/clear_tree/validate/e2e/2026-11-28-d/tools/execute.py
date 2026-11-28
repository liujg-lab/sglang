#!/usr/bin/env python3
"""Run a fixed two-mode ClearTree matrix without modifying the saved fits."""
import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from staging import META, RAW, REPO, SYNC, StagingDriver, base, fit_coverage, require

FIT_LOG = (r"no fit file|stashed fit file|wrote fit file|layer slope priors installed|"
           r"prefit layer weight capped|installed [0-9]+ hold cells from the fit file|"
           r"prefit required|fit file.*unreadable|target verifies with RPD|prefit started|"
           r"prefit driver finished state=|CLEAR fit_state|rpd_flat")
BAD_FIT = re.compile(r"no fit file|wrote fit file|prefit required|fit file.*unreadable|"
                     r"prefit started|prefit driver finished state=|rpd_flat")
TERMINAL_BAD = {"failed", "disabled", "legacy", "breaker", "prefitting"}
MODES = ("greedy", "rpd")


def validate_file(path, mode, shapes):
    fit = json.loads(path.read_text())
    report = fit_coverage(fit, mode, shapes)
    require((report["schema"], report["fingerprint"], report["required_h"], report["required_layers"])
            == (META["fit_schema"], META["fit_fingerprint"], 90, 18), "unexpected saved FIT coverage")
    return report


class Driver(StagingDriver):
    def check_hashes(self, output):
        result = {mode: self.draft_digest(mode) for mode in MODES}
        base.write_json(output, result)
        require(result == META["fit_sha256"], f"saved Draft fit hashes changed: {output.name}")
        return result

    def coverage(self, mode, captures):
        root = self.cfg["draft"]["ROOT"]
        result = self.container("draft", "env", f"PYTHONPATH={root}/python", "python3", "-B",
                                f"{root}/{Path(__file__).resolve().relative_to(REPO)}", "check-file",
                                self.fit_path("draft", mode), mode, json.dumps(sorted(captures)))
        return json.loads(result.stdout)

    def validate_saved(self):
        self.check_hashes(SYNC / "fit-hashes-preflight.json")
        shapes = {(batch, width) for batch in base.BATCHES for width in base.TIERS}
        for mode in MODES:
            base.write_json(SYNC / f"{mode}-fit-coverage-preflight.json", self.coverage(mode, shapes))

    def fit_logs(self, mode):
        return {role: self.logpart(role, mode, FIT_LOG) for role in base.ROLES}

    def fit_states_ready(self, text):
        states = self.fit_states(text)
        require(not any(s.get("state") in TERMINAL_BAD for s in states.values()),
                "Target entered a terminal non-ready FIT state")
        state = [states.get(rank, {}) for rank in ("0", "1")]
        if not all((s.get("state"), s.get("reason"), s.get("run")) == ("ready", "ready", None) for s in state):
            return None
        require(state[0].get("session") and state[0]["session"] == state[1].get("session"),
                "Target ranks disagree on FIT session")
        return state[0]["session"]

    def wait_ready(self, role, mode, owned):
        if role == "draft":
            return super().wait_ready(role, mode, owned)
        deadline = time.monotonic() + 3600
        last_size = None
        while time.monotonic() < deadline:
            for peer, owner in owned.items():
                self.assert_owner(peer, owner)
            logs = self.fit_logs(mode)
            require(not any(BAD_FIT.search(text) for text in logs.values()), "fit reuse refused; no refit allowed")
            if self.fit_states_ready(logs["target"]) and self.ready_http("target"):
                return
            size = self.container(role, "stat", "-c", "%s", f"{self.work}/{mode}/{role}.log").stdout.strip()
            if size != last_size:
                print(f"{mode} {role} startup log {size} bytes", flush=True)
                last_size = size
            time.sleep(20)
        raise RuntimeError("Target HTTP / same-session saved FIT readiness deadline exceeded")

    def check_fit_logs(self, mode, dest):
        logs = self.fit_logs(mode)
        dest.write_text(logs["draft"] + logs["target"])
        require(not any(BAD_FIT.search(text) for text in logs.values()), "fit reuse failed or FIT write/scan detected")
        session = self.fit_states_ready(logs["target"])
        require(session, "Target is not ready on both ranks for reused FIT")
        draft = logs["draft"].splitlines()
        source = self.fit_path("draft", mode)
        stashed = [i for i, line in enumerate(draft) if line.endswith(f"stashed fit file {source} for the Target header check")]
        capped = [i for i, line in enumerate(draft) if line.endswith("prefit layer weight capped cap=64 cells=4")]
        installed = [i for i, line in enumerate(draft) if line.endswith("installed 90 hold cells from the fit file")]
        require(len(stashed) == len(capped) == len(installed) == 1 and stashed[0] < capped[0] < installed[0],
                "expected one saved FIT install with cap=64 cells=4 and 90 H cells")
        ack = [line for line in draft if "target verifies with RPD" in line]
        require((len(ack) == 1 and ack[0].endswith(f"target verifies with RPD (tau=0.4); W/K run online on fit file {source}"))
                if mode == "rpd" else not ack, "wrong RPD mode, tau, or FIT file")
        return session

    def fit_gate(self, mode, out, owned):
        session = self.check_fit_logs(mode, out / "fit-gate.txt")
        hashes = self.check_hashes(out / "fit-hashes-gate.json")
        captures = {}
        for role in base.ROLES:
            self.assert_owner(role, owned[role])
            require(self.ready_http(role), f"{role} HTTP not 200 at FIT gate")
            text = self.logpart(role, mode, "Capturing: bs=")
            (out / f"{role}-captures.txt").write_text(text)
            captures[role] = {(int(a), int(n)) for a, n in re.findall(r"Capturing: bs=(\d+), ntpb=(\d+)", text)}
            require({a for a, _ in captures[role]} == set(base.BATCHES), f"{role} capture batches differ")
        target_shapes = {(a, n) for a in base.BATCHES for n in (1, 3, 5, 9, 17, 33, 65)}
        draft_shapes = target_shapes | {(a, n) for a in base.BATCHES for n in base.TIERS}
        require(target_shapes <= captures["target"] and draft_shapes - {(4, 65)} <= captures["draft"]
                and (4, 65) not in captures["draft"], "missing or over-capacity graph shapes")
        report = self.coverage(mode, captures["draft"])
        base.write_json(out / "fit-coverage.json", dict(report, session=session, sha256=hashes[mode],
                        reused=True, draft_health=200, target_health=200, draft_captures=len(captures["draft"]),
                        target_captures=len(captures["target"]), draft_capacity_skip=[4, 65]))
        return session

    def log_sizes(self, mode):
        return {role: int(self.container(role, "stat", "-c", "%s", f"{self.work}/{mode}/{role}.log").stdout)
                for role in base.ROLES}

    def client(self, mode, trial, ordinal):
        number, dataset, concurrency = (trial[key] for key in ("trial", "dataset", "concurrency"))
        out = RAW / number
        out.mkdir()
        dest = f"{self.work}/{number}"
        self.container("target", "mkdir", dest)
        before = self.log_sizes(mode)
        data, images = self.datasets[dataset]
        args = ["env", f"CT_HTTP_ADDR={self.shared['HTTP_ADDR']}", f"CT_HTTP_PORT={self.cfg['target']['HTTP_PORT']}",
                f"CT_CLIENT_DIR={self.work}/client", "bash", f"{self.work}/run-client.sh", number, dataset,
                str(concurrency), data, images, dest]
        print(f"formal {number} {mode} {dataset} concurrency={concurrency} started", flush=True)
        result = self.container("target", "bash", "-c", f"{shlex.join(args)} > {shlex.quote(dest + '/client.stdout')} 2>&1",
                                capture=False, check=False)
        (out / "exit-code.txt").write_text(f"{result.returncode}\n")
        self.copyout("target", dest, RAW)
        # Let the existing idle/log flush complete; no requests are sent here and
        # this interval is outside the client's measured wall time.
        time.sleep(3)
        base.write_json(out / "log-boundaries.json", {"before": before, "after": self.log_sizes(mode)})
        paths = list(out.glob("client.*.jsonl"))
        require(result.returncode == 0 and len(paths) == 1, f"formal {number} failed or JSONL missing")
        rows = [json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]
        require(len(rows) == META["max_items"] and all(not row.get("error") for row in rows), f"formal {number}: not 100/100")
        require(all(row.get("spec_fallback_ct") == 0 for row in rows), f"formal {number}: fallback or missing counter")
        require(paths[0].with_suffix(".csv").stat().st_size and paths[0].with_suffix(".run.json").stat().st_size,
                f"formal {number}: missing CSV or manifest")
        flush = self.logpart("target", mode, "Cache flushed successfully|POST /flush_cache")
        (out / "target-flush.txt").write_text(flush)
        require(re.findall(r'"POST /flush_cache HTTP/\S+" (\d+)', flush) == ["200"] * ordinal and
                all(len(re.findall(rf"\bTP{rank}\].*Cache flushed successfully", flush)) == ordinal for rank in (0, 1)),
                f"formal {number}: flush not confirmed by HTTP 200 and both ranks")
        print(f"formal {number} complete (100/100, no fallback, flush confirmed)", flush=True)

    def block(self, mode):
        self.verify_synced()
        trials = [row for row in META["trials"] if row["mode"] == mode]
        require(len(trials) == 4 and not any((RAW / trial["trial"]).exists() for trial in trials), "bad matrix or existing trial output")
        if mode == "rpd":
            require((RAW / "greedy/block-complete.txt").is_file(), "Greedy must complete and clean up before RPD")
        out = RAW / mode
        out.mkdir()
        owned, attempted, errors = {}, set(), []
        failed = False
        try:
            self.check_hashes(out / "fit-hashes-before.json")
            for role in base.ROLES:
                self.precheck(role, out / f"before-parse-{role}.txt")
                self.validate_argv(role, mode, out / f"parse-{role}.txt")
            for role in base.ROLES:
                self.start(role, mode, out, attempted, owned)
                self.wait_ready(role, mode, owned)
                print(f"{mode} {role} HTTP ready", flush=True)
                if role == "draft":
                    draft_fit = self.logpart(role, mode, FIT_LOG)
                    (out / "draft-fit-before-target.txt").write_text(draft_fit)
                    require(not BAD_FIT.search(draft_fit) and all(
                        f"stashed fit file {self.fit_path(role, candidate)} for the Target header check" in draft_fit
                        for candidate in MODES), "Draft did not read both saved fits; refusing Target launch")
            session = self.fit_gate(mode, out, owned)
            print(f"{mode} saved FIT reuse and graph gate passed", flush=True)
            for ordinal, trial in enumerate(trials, 1):
                for role in base.ROLES:
                    self.assert_owner(role, owned[role])
                self.client(mode, trial, ordinal)
                require(self.check_fit_logs(mode, out / "fit-gate-final.txt") == session, "FIT session changed after launch")
        except BaseException:
            failed = True
            raise
        finally:
            for role in reversed(base.ROLES):
                if role not in attempted:
                    continue
                try:
                    if role not in owned:
                        owned[role] = self.discover_owner(role, mode)
                        base.write_json(out / f"{role}-identity.json", owned[role])
                    self.stop(role, owned[role], out)
                except Exception as exc:
                    errors.append(f"{role} stop: {exc}")
            for role in reversed(base.ROLES):
                if role in attempted:
                    try:
                        self.copyout(role, f"{self.work}/{mode}/{role}.log", out / f"{role}.log")
                    except Exception as exc:
                        errors.append(f"{role} log copy: {exc}")
                try:
                    self.precheck(role, out / f"postcheck-{role}.txt")
                except Exception as exc:
                    errors.append(f"{role} postcheck: {exc}")
            try:
                self.check_hashes(out / "fit-hashes-after.json")
            except Exception as exc:
                errors.append(f"FIT preservation: {exc}")
            if errors:
                (out / "cleanup-errors.txt").write_text("\n".join(errors) + "\n")
                print("\n".join(errors), file=sys.stderr, flush=True)
                require(failed, "cleanup evidence incomplete")
        (out / "block-complete.txt").write_text("four formal trials completed; owned sessions stopped and postchecks matched\n")
        print(f"{mode} cleaned up: ports, devices, tunnel identities, and both fit hashes match", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("check-file", "setup", "sync", *MODES))
    parser.add_argument("parameters", nargs="*")
    args = parser.parse_args()
    if args.step == "check-file":
        path, mode, shapes = args.parameters
        print(json.dumps(validate_file(Path(path), mode, set(map(tuple, json.loads(shapes))))))
        return 0
    if args.parameters:
        parser.error("unexpected parameters")
    for sig in (base.signal.SIGTERM, base.signal.SIGINT):
        base.signal.signal(sig, base.interrupted)
    try:
        driver = Driver()
        if args.step == "setup":
            driver.setup()
            driver.validate_saved()
        elif args.step == "sync":
            driver.sync()
            driver.verify_synced()
            driver.check_hashes(SYNC / "fit-hashes-post-sync.json")
        else:
            driver.block(args.step)
    except (RuntimeError, OSError, ValueError, KeyError, KeyboardInterrupt, subprocess.SubprocessError) as exc:
        failure = f"ClearTree {args.step}: {exc}"
        dest = RAW / args.step if args.step in MODES else SYNC
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"{args.step}-failure.txt").write_text(failure + "\n")
        print(failure, file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
