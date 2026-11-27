#!/usr/bin/env python3
import hashlib
import json
import sys
from pathlib import Path

# Check the same imports needed by the formal client and scorer.
import aiohttp  # noqa: F401
import nltk  # noqa: F401
import rouge_score  # noqa: F401


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main(dataset, image_root):
    raw = Path(dataset).read_bytes()
    rows = json.loads(raw)
    if not isinstance(rows, list) or len(rows) < 100:
        raise ValueError("RSTeller needs at least 100 rows")
    batch = rows[:100]
    if not all(row.get("ground_truth") for row in batch):
        raise ValueError("missing ground truth in formal batch")
    root = Path(image_root)
    hashes = {
        "file_sha256": sha(raw),
        "selected_rows_sha256": sha(
            json.dumps(batch, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ),
        "ordered_image_sha256": [sha((root / row["image"]).read_bytes()) for row in batch],
        "available": len(rows),
        "selected": len(batch),
    }
    print(json.dumps({"RSTeller": hashes}, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
