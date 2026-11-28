#!/usr/bin/env python3
"""Run the four-trial retry; the SGLang-dependent gate runs inside Draft."""
import argparse
import json
import re
import sys
from pathlib import Path

from supervisor import RAW, REPO, SYNC, StagingDriver, base, fit_coverage


class Driver(StagingDriver):
    def coverage_in_draft(self, path, mode, captures):
        root = self.cfg["draft"]["ROOT"]
        tool = f"{root}/{Path(__file__).relative_to(REPO)}"
        visible_fit = f"{root}/{path.relative_to(REPO)}"
        result = self.container("draft", "env", f"PYTHONPATH={root}/python", "python3", "-B", tool,
                                "check-fit", str(visible_fit), mode, json.dumps(sorted(captures)))
        return json.loads(result.stdout)

    def coverage_preflight(self):
        # Exercise the same coverage gate against this run's successful Greedy snapshot.
        archived = RAW / "greedy"
        captures = {(int(b), int(n)) for b, n in re.findall(
            r"Capturing: bs=(\d+), ntpb=(\d+)", (archived / "draft-captures.txt").read_text())}
        report = self.coverage_in_draft(archived / base.NAMES["greedy"], "greedy", captures)
        base.require(report["schema"] == 5 and report["required_h"] == 90 and report["required_layers"] == 18,
                     "Draft container coverage preflight returned an unexpected result")
        return report

    def live_coverage(self, fit, mode, captures):
        snapshot = RAW / mode / base.NAMES[mode]
        report = self.coverage_in_draft(snapshot, mode, captures)
        base.require(report["run"] == fit["run"], "container coverage checked a different fit generation")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("check-fit", "setup", "sync", "greedy", "rpd"))
    parser.add_argument("parameters", nargs="*")
    args = parser.parse_args()
    if args.step == "check-fit":
        if len(args.parameters) != 3:
            parser.error("check-fit requires snapshot, mode, and capture shapes")
        path, mode, shapes = args.parameters
        report = fit_coverage(json.loads(Path(path).read_text()), mode, set(map(tuple, json.loads(shapes))))
        print(json.dumps(report))
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
            base.require((SYNC / "draft-container-coverage-preflight.json").is_file(), "missing container coverage preflight")
            driver.sync()
        else:
            driver.coverage_preflight()
            if args.step == "rpd":
                driver.data_check("-rpd")
            driver.block(args.step)
    except (RuntimeError, OSError, ValueError, KeyError, KeyboardInterrupt) as exc:
        print(f"ClearTree {args.step}: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
