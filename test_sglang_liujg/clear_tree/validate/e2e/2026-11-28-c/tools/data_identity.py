#!/usr/bin/env python3
"""Hash the same ordered RSTeller inputs as the reference run, without scoring."""
import hashlib
import json
import sys
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main(dataset, image_root):
    import aiohttp  # noqa: F401 -- required by the formal client

    data = Path(dataset).read_bytes()
    rows = json.loads(data)
    if not isinstance(rows, list) or len(rows) < 100:
        raise ValueError("Expected at least 100 RSTeller rows")
    selected = rows[:100]
    root = Path(image_root)
    result = dict(
        file_sha256=digest(data),
        selected_rows_sha256=digest(json.dumps(
            selected, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()),
        ordered_image_sha256=[digest((root / row["image"]).read_bytes()) for row in selected],
        available=len(rows),
        selected=len(selected),
    )
    print(json.dumps({"RSTeller": result}, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
