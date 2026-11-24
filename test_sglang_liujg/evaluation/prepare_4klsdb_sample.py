#!/usr/bin/env python3
import argparse
import json
import re
import tarfile
from pathlib import Path

MAX_SAMPLES = 100
MEMBER_NAME = re.compile(r"(sample_[0-9]+)\.(jpg|txt)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the first 100 valid pairs in a 4KLSDB TAR shard."
    )
    parser.add_argument("--tar", type=Path, required=True, help="Input TAR shard")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New directory for the sample"
    )
    args = parser.parse_args()

    images_dir = args.output_dir / "images"
    samples = {}
    with tarfile.open(args.tar, "r|*") as archive:
        current_key = None
        image = None
        caption = None
        for member in archive:
            match = MEMBER_NAME.fullmatch(member.name)
            if match is None or not member.isfile():
                continue
            key, extension = match.groups()
            if key in samples:
                continue
            if key != current_key:
                current_key = key
                image = caption = None
            data = archive.extractfile(member).read()
            if extension == "jpg":
                image = data
            else:
                try:
                    caption = data.decode("utf-8").strip()
                except UnicodeDecodeError:
                    caption = None
            if not image or not caption:
                continue
            if not samples:
                try:
                    args.output_dir.mkdir()
                except FileExistsError:
                    parser.error(f"output directory already exists: {args.output_dir}")
                images_dir.mkdir()
            filename = f"{key}.jpg"
            (images_dir / filename).write_bytes(image)
            samples[key] = {
                "image": filename,
                "question": "Describe this image in English.",
                "ground_truth": caption,
            }
            if len(samples) == MAX_SAMPLES:
                break

    if not samples:
        parser.error(f"no valid image/caption pairs in {args.tar}")
    manifest = args.output_dir / "samples.json"
    manifest.write_text(
        json.dumps(list(samples.values()), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(samples)} samples to {manifest} (limit: {MAX_SAMPLES})")


if __name__ == "__main__":
    main()
