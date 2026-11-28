#!/usr/bin/env python3
"""Run the two fresh-fit blocks; SGLang-dependent checks execute in Draft."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from supervisor import META, RAW, REFERENCE, REPO, SYNC, StagingDriver, base, fit_coverage, require

CAP_LOG = 'wrote fit file|layer slope priors installed|prefit layer weight capped'


class Driver(StagingDriver):
    def in_draft(self, step, *paths_and_args):
        root = self.cfg["draft"]["ROOT"]
        def visible(path):
            return f"{root}/{path.relative_to(REPO)}" if isinstance(path, Path) else path
        result = self.container("draft", "env", f"PYTHONPATH={root}/python", "python3", "-B",
                                visible(Path(__file__)), step, *map(visible, paths_and_args))
        return json.loads(result.stdout)

    def coverage(self, path, mode, captures):
        report = self.in_draft("check-fit", path, mode, json.dumps(sorted(captures)))
        require((report["schema"], report["fingerprint"], report["required_h"], report["required_layers"])
                == (META["fit_schema"], META["fit_fingerprint"], 90, 18), "unexpected fit schema, fingerprint, or coverage")
        return report

    def coverage_preflight(self):
        prior = REFERENCE / "raw/greedy"
        captures = {(int(b), int(n)) for b, n in re.findall(
            r"Capturing: bs=(\d+), ntpb=(\d+)", (prior / "draft-captures.txt").read_text())}
        return self.coverage(prior / base.NAMES["greedy"], "greedy", captures)

    def live_coverage(self, fit, mode, captures):
        report = self.coverage(RAW / mode / base.NAMES[mode], mode, captures)
        require(report["run"] == fit["run"], "coverage checked a different fit generation")
        return report

    def fit_gate(self, mode, out, owned):
        super().fit_gate(mode, out, owned)
        cap_log = out / "fit-cap-log.txt"
        cap_log.write_text(self.logpart("draft", mode, CAP_LOG))
        report = self.in_draft("check-cap", out / base.NAMES[mode], cap_log)
        base.write_json(out / "fit-cap.json", report)
        require(report["passed"], "fresh-fit Draft cap log differs from saved eligible cells")
        print(f"{mode} layer weight cap={report['expected_cap']} cells={report['expected_cells']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("check-fit", "check-cap", "setup", "sync", "greedy", "rpd"))
    parser.add_argument("parameters", nargs="*")
    args = parser.parse_args()
    if args.step == "check-fit":
        path, mode, shapes = args.parameters
        print(json.dumps(fit_coverage(json.loads(Path(path).read_text()), mode, set(map(tuple, json.loads(shapes))))))
        return 0
    if args.step == "check-cap":
        from fit_cap import check
        path, log = map(Path, args.parameters)
        print(json.dumps(check(json.loads(path.read_text()), log.read_text())))
        return 0
    if args.parameters:
        parser.error("unexpected parameters")
    for sig in (base.signal.SIGTERM, base.signal.SIGINT):
        base.signal.signal(sig, base.interrupted)
    try:
        driver = Driver()
        base.fit_coverage = driver.live_coverage
        if args.step == "setup":
            report = driver.coverage_preflight()
            driver.setup()
            base.write_json(SYNC / "draft-container-coverage-preflight.json", report)
            print("Draft-container coverage preflight passed", flush=True)
        elif args.step == "sync":
            require((SYNC / "draft-container-coverage-preflight.json").is_file(), "missing container coverage preflight")
            driver.sync()
        else:
            driver.verify_synced()
            if args.step == "rpd":
                driver.data_check("-rpd")
            driver.block(args.step)
    except (RuntimeError, OSError, ValueError, KeyError, KeyboardInterrupt, subprocess.SubprocessError) as exc:
        failure = f"ClearTree {args.step}: {exc}"
        evidence = RAW / args.step
        if evidence.is_dir():
            (evidence / "driver-failure.txt").write_text(failure + "\n")
        print(failure, file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
