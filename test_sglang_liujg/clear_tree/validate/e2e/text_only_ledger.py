#!/usr/bin/env python3
"""Open-loop text_only client for the ClearTree end-to-end comparison.

Timing stays in ``eval_spectre_text_serving``. This file only sets the
comparison defaults and forwards every other flag.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EVALUATION = Path(__file__).resolve().parents[3] / "evaluation"
if str(_EVALUATION) not in sys.path:
    sys.path.insert(0, str(_EVALUATION))

_DEFAULTS = (
    ("--max-items", "100"),
    ("--max-concurrency", "4"),
    ("--request-rate", "inf"),
    ("--max-new-tokens", "1024"),
    ("--temperature", "0"),
    ("--skip-metrics", None),
)


def _present(argv: list[str]) -> set[str]:
    found = set()
    for arg in argv:
        if arg.startswith("--"):
            found.add(arg.split("=", 1)[0])
    return found


def main() -> int:
    from eval_spectre_text_serving import main as run_text

    argv = sys.argv[1:]
    have = _present(argv)
    extra: list[str] = []
    for flag, value in _DEFAULTS:
        if flag in have:
            continue
        extra.append(flag)
        if value is not None:
            extra.append(value)
    sys.argv = [sys.argv[0], *extra, *argv]
    return run_text()


if __name__ == "__main__":
    raise SystemExit(main())
