#!/usr/bin/env python3
"""Extract DOCCI caption samples and original images from a local Parquet shard."""

import argparse
import io
import json
from pathlib import Path

DATASET_ROOT = Path("/home_18T/liujg/Hugging_Face/gokaygokay/random_instruct_docci")
DEFAULT_PARQUET = DATASET_ROOT / "data" / "test-00000-of-00001.parquet"
DEFAULT_OUTPUT_DIR = DATASET_ROOT.with_name("random_instruct_docci_sample_120")
COLUMNS = ["image", "description", "example_id"]
IMAGE_EXTENSIONS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-items", type=int, default=120)
    args = parser.parse_args()
    if args.max_items < 1:
        parser.error("--max-items must be >= 1")
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    if not args.parquet.is_file():
        parser.error(f"Parquet file not found: {args.parquet}")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from PIL import Image
    except ImportError as exc:
        parser.error(f"conversion requires PyArrow and Pillow: {exc}")

    images_dir = args.output_dir / "images"
    samples = []
    skipped = 0
    with pq.ParquetFile(args.parquet) as shard:
        schema = shard.schema_arrow
        missing = [column for column in COLUMNS if column not in schema.names]
        if missing:
            parser.error(f"missing Parquet columns: {', '.join(missing)}")
        image_type = schema.field("image").type
        if not pa.types.is_struct(image_type) or image_type.get_field_index("bytes") < 0:
            parser.error(f"expected image.bytes in Parquet, got image: {image_type}")

        for batch in shard.iter_batches(batch_size=16, columns=COLUMNS):
            for row in batch.to_pylist():
                caption = (row["description"] or "").strip()
                raw = (row["image"] or {}).get("bytes")
                if not caption or not raw:
                    skipped += 1
                    continue
                try:
                    with Image.open(io.BytesIO(raw)) as image:
                        extension = IMAGE_EXTENSIONS.get(image.format)
                        image.verify()
                except (OSError, SyntaxError, ValueError, Image.DecompressionBombError):
                    extension = None
                if extension is None:
                    skipped += 1
                    continue

                if not samples:
                    try:
                        args.output_dir.mkdir()
                    except FileExistsError:
                        parser.error(f"output directory already exists: {args.output_dir}")
                    images_dir.mkdir()
                filename = f"docci_{len(samples):06d}.{extension}"
                (images_dir / filename).write_bytes(raw)
                samples.append(
                    {
                        "image": filename,
                        "question": "Describe this image in English.",
                        "ground_truth": caption,
                        "example_id": row["example_id"],
                    }
                )
                if len(samples) == args.max_items:
                    break
            if len(samples) == args.max_items:
                break

    if not samples:
        parser.error(f"no valid image/caption pairs in {args.parquet} ({skipped} skipped)")
    manifest = args.output_dir / "samples.json"
    manifest.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(samples)}/{args.max_items} samples to {manifest} ({skipped} skipped)")


if __name__ == "__main__":
    main()
