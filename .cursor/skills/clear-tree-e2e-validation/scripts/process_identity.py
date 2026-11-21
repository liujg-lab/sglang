#!/usr/bin/env python3
"""Print the host identity of one PID. Read-only. Stdlib only.

The last NSpid is the pid inside the innermost namespace. Compare this output
again before signaling that pid from inside the container.
"""

from __future__ import annotations

import sys
from pathlib import Path


def status_value(text: str, key: str) -> str:
    prefix = key + ":"
    for line in text.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    raise SystemExit(f"/proc status has no {key}")


def stat_fields(pid: int) -> tuple[str, str, str, str]:
    text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    end = text.rfind(")")
    if end < 0:
        raise SystemExit(f"/proc/{pid}/stat has no command field")
    fields = text[end + 2 :].split()
    # After comm: state, ppid, pgrp, session, ..., starttime at field 22.
    return fields[0], fields[1], fields[3], fields[19]


def main() -> None:
    if len(sys.argv) != 2 or not sys.argv[1].isdigit():
        raise SystemExit("usage: process_identity.py <host-pid>")
    pid = int(sys.argv[1])
    status_path = Path(f"/proc/{pid}/status")
    if not status_path.is_file():
        raise SystemExit(f"missing /proc/{pid}")
    status = status_path.read_text(encoding="utf-8")
    nspid = status_value(status, "NSpid").split()
    uid = status_value(status, "Uid").split()[0]
    state, ppid, sid, start_ticks = stat_fields(pid)
    comm = Path(f"/proc/{pid}/comm").read_text(encoding="utf-8").strip()
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    rows = (
        ("pid", pid),
        ("comm", comm),
        ("state", state),
        ("nspid", " ".join(nspid)),
        ("container_nspid", nspid[-1]),
        ("ppid", ppid),
        ("sid", sid),
        ("start_ticks", start_ticks),
        ("uid", uid),
        ("boot_id", boot_id),
    )
    sys.stdout.write("".join(f"{key} {value}\n" for key, value in rows))


if __name__ == "__main__":
    main()
