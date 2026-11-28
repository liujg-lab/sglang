#!/usr/bin/env python3
"""Seal regular raw artifacts, not links used by the log reviewer."""
import hashlib
from pathlib import Path


raw = Path(__file__).resolve().parents[1] / "raw"
manifest = raw / "SHA256SUMS"
entries = []
for path in sorted(raw.rglob("*")):
    if path == manifest:
        continue
    if path.is_symlink():
        path.resolve(strict=True).relative_to(raw)
        continue
    if not path.is_file():
        continue
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    entries.append(f"{digest.hexdigest()}  {path.relative_to(raw).as_posix()}\n")
with manifest.open("x") as stream:
    stream.writelines(entries)
print(f"Sealed {len(entries)} raw artifacts")
