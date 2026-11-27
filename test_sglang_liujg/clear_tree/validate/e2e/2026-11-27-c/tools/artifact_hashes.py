#!/usr/bin/env python3
"""Seal the raw evidence and list paths relative to the run's raw directory."""
import hashlib
from pathlib import Path
import sys


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(raw):
    destination = raw / "SHA256SUMS"
    paths = sorted(
        path for path in raw.rglob("*")
        if path.is_file() and not path.is_symlink() and path != destination
    )
    sums = "".join(f"{sha256(path)}  {path.relative_to(raw)}\n" for path in paths)
    destination.write_text(sums, encoding="utf-8")
    print(f"Hashed {len(paths)} raw artifacts")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
