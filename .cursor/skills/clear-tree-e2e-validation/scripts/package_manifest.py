#!/usr/bin/env python3
"""SHA256 of the ClearTree package, excluding caches and the fit file.

Prints ``<digest>  ./relative-path`` sorted by path. Optional ``--tar`` packs
the same files. Stdlib only. No addresses, accounts, or model paths.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
from pathlib import Path


def default_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "python/sglang/srt/speculative/clear_tree_remote"
        if candidate.is_dir():
            return candidate
    raise SystemExit("pass --root; clear_tree_remote was not found from this script")


def skipped(rel: Path) -> bool:
    name = rel.name
    if "__pycache__" in rel.parts:
        return True
    if name.endswith(".pyc") or name.endswith(".tmp"):
        return True
    return name == "clear_tree_fit.json"


def listed_files(root: Path) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root)
        if skipped(rel):
            continue
        found.append((rel.as_posix(), path))
    found.sort(key=lambda item: item[0])
    return found


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="clear_tree_remote directory")
    parser.add_argument("--tar", type=Path, help="optional archive of the same files")
    args = parser.parse_args()
    root = (args.root or default_root()).resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    files = listed_files(root)
    lines = [f"{digest(path)}  ./{rel}" for rel, path in files]
    sys.stdout.write("\n".join(lines))
    if lines:
        sys.stdout.write("\n")
    if args.tar is None:
        return
    mode = "w:gz" if args.tar.suffix in {".gz", ".tgz"} else "w"
    with tarfile.open(args.tar, mode) as archive:
        for rel, path in files:
            archive.add(path, arcname=f"./{rel}")


if __name__ == "__main__":
    main()
