#!/usr/bin/env python3
"""Run the shared log utility separately per mode, with read-only views of raw logs."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


run = Path(__file__).resolve().parents[1]
repo = run.parents[4]
raw = run / "raw"
trials = json.loads((run / "meta.json").read_text())["trials"]
accept = repo / ".cursor/skills/clear-tree-e2e-validation/scripts/accept_logs.py"
for mode in ("greedy", "rpd"):
    with tempfile.TemporaryDirectory(prefix=f"ct-accept-{mode}-") as temp:
        view = Path(temp)
        for role in ("draft", "target"):
            (view / f"{role}.log").symlink_to(raw / mode / f"{role}.log")
        for row in trials:
            if row["mode"] == mode:
                name = row["trial"]
                (view / name).symlink_to(raw / name, target_is_directory=True)
        with (raw / mode / "accept-logs.txt").open("x") as output:
            subprocess.run([sys.executable, "-B", str(accept), "--raw", temp], stdout=output, check=True)
