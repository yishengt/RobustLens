#!/usr/bin/env python3
"""Extract SID_Set parquet shards into image files on disk.

The shards store images inline, which the inference CLI cannot read. This
unpacks them into class folders that ``--input-dir`` accepts directly, plus a
``labels.json`` manifest carrying ground truth for evaluation.

Images are written **byte-identical** to the dataset: re-encoding would resample
the compression artefacts AI-image detection relies on.

Layout::

    data/extracted/sid_set/
    |-- real/            label 0
    |-- ai_generated/    labels 1 (full synthetic) and 2 (tampered)
    `-- labels.json

Examples::

    # everything in the downloaded shards
    python scripts/extract_dataset.py

    # a balanced 200-image sample
    python scripts/extract_dataset.py --per-class-limit 100
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BINARY_DIRS = {0: "real", 1: "ai_generated"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unpack SID_Set parquet shards into image files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="data/sid_set", help="Directory holding the shards")
    parser.add_argument("--split", default="validation", help="Shard split prefix to extract")
    parser.add_argument(
        "--output", default="data/extracted/sid_set", help="Where to write the image folders"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Stop after N images total (0 = no limit)"
    )
    parser.add_argument(
        "--per-class-limit",
        type=int,
        default=None,
        help="Cap images taken from each of the three source classes (balances the sample)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rewrite files that already exist")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-image progress")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from src.evaluation.sid_set import CLASS_NAMES, find_shards, iter_raw_records

    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    output_dir = Path(args.output).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    try:
        shards = find_shards(data_dir, args.split)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 4

    print(f"Shards : {len(shards)} ({args.split})")
    print(f"Output : {output_dir}")

    for name in BINARY_DIRS.values():
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    counts: Counter[str] = Counter()
    skipped = 0
    limit = None if args.limit in (0, None) else int(args.limit)

    for index, record in enumerate(
        iter_raw_records(shards, limit=limit, per_class_limit=args.per_class_limit), start=1
    ):
        folder = BINARY_DIRS[record.binary_label]
        # Prefix with the source class so tampered and synthetic stay tellable
        # apart inside the shared ai_generated/ folder -- but SID_Set already
        # names those rows "full_synthetic_..."/"tampered_...", so only add the
        # prefix where it is missing (real images use bare hash ids).
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in record.img_id)
        filename = (
            f"{safe_id}{record.extension}"
            if safe_id.startswith(record.class_name)
            else f"{record.class_name}_{safe_id}{record.extension}"
        )
        destination = output_dir / folder / filename

        if destination.exists() and not args.overwrite:
            skipped += 1
        else:
            # Bytes are written verbatim - no decode, no re-encode.
            destination.write_bytes(record.data)

        counts[record.class_name] += 1
        manifest.append(
            {
                "image_path": str(destination.relative_to(output_dir)),
                "img_id": record.img_id,
                "label": record.binary_label,
                "source_label": record.label,
                "class_name": record.class_name,
                "format": record.image_format,
                "shard": record.source,
            }
        )

        if not args.quiet and index % 200 == 0:
            print(f"  [{index}] {counts.most_common()}", file=sys.stderr)

    if not manifest:
        print("No images were extracted.", file=sys.stderr)
        return 1

    manifest_path = output_dir / "labels.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": "SID_Set",
                "split": args.split,
                "source": "https://huggingface.co/datasets/saberzl/SID_Set",
                "label_mapping": {
                    "0": "real",
                    "1": "ai_generated (full_synthetic or tampered)",
                },
                "class_distribution": dict(counts),
                "count": len(manifest),
                "images": manifest,
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    total_bytes = sum((output_dir / entry["image_path"]).stat().st_size for entry in manifest)
    print(f"\nExtracted {len(manifest)} images ({total_bytes / 1e9:.2f} GB)")
    for name in CLASS_NAMES.values():
        if counts.get(name):
            print(f"  {name:16} {counts[name]}")
    print(f"  -> real         {sum(1 for e in manifest if e['label'] == 0)}")
    print(f"  -> ai_generated {sum(1 for e in manifest if e['label'] == 1)}")
    if skipped:
        print(f"  ({skipped} already existed; use --overwrite to rewrite)")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
