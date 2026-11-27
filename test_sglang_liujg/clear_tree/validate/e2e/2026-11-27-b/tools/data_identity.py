#!/usr/bin/env python3
import hashlib
import json
import sys
from pathlib import Path

import aiohttp
import nltk
import rouge_score


def sha(data):
    return hashlib.sha256(data).hexdigest()


result = {}
for label, dataset, img in zip(sys.argv[1::3], sys.argv[2::3], sys.argv[3::3]):
    raw = Path(dataset).read_bytes()
    rows = json.loads(raw)
    assert isinstance(rows, list) and len(rows) >= 100, label
    batch = rows[:100]
    image_hashes = []
    for row in batch:
        assert row.get("ground_truth"), label
        image_hashes.append(sha((Path(img) / row["image"]).read_bytes()))
    result[label] = {
        "file_sha256": sha(raw),
        "selected_rows_sha256": sha(json.dumps(batch, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()),
        "ordered_image_sha256": image_hashes,
        "available": len(rows),
        "selected": len(batch),
    }
assert len(result) == 2
if result["RSTeller"]["file_sha256"] != "f9a74112455c63c67e566e68d7b7c520769c67bcb27a7f960054ca6e1beabbbc":
    raise SystemExit("RSTeller does not match reference")
print(json.dumps(result, indent=2))
