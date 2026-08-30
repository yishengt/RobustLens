#!/usr/bin/env python3
"""Convert Image-editing/escher-vismin Parquet relations to RobustLens folders.

The source and every edit for one original are assigned to one split. Randomly
splitting source/edit images would leak the original scene into evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image


def _image_bytes(value: Any) -> Optional[bytes]:
    if isinstance(value, bytes):
        return value
    if isinstance(value, dict):
        raw = value.get("bytes")
        if isinstance(raw, bytes):
            return raw
        path = value.get("path")
        if path:
            return Path(str(path)).read_bytes()
    if isinstance(value, Image.Image):
        buffer = io.BytesIO()
        value.save(buffer, format="PNG")
        return buffer.getvalue()
    return None


def _save_image(raw: bytes, path: Path) -> None:
    with Image.open(io.BytesIO(raw)) as image:
        image.convert("RGB").save(path, format="JPEG", quality=95)


def _split(group_id: str, validation_fraction: float, test_fraction: float) -> str:
    bucket = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) / 2**32
    if bucket < test_fraction:
        return "test"
    if bucket < test_fraction + validation_fraction:
        return "validation"
    return "train"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Downloaded dataset repo or its data directory")
    parser.add_argument("--output", default=Path("data/local_edits"), type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Keep at most this many original-image groups; all edits in a kept group are retained",
    )
    args = parser.parse_args(argv)
    if args.validation_fraction < 0 or args.test_fraction < 0 or args.validation_fraction + args.test_fraction >= 1:
        parser.error("validation and test fractions must be non-negative and sum to less than 1")
    if args.max_groups is not None and args.max_groups <= 0:
        parser.error("--max-groups must be positive")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install the dataset dependency first: pip install datasets") from exc

    data_root = args.input / "data"
    parquet_files = sorted((data_root if data_root.is_dir() else args.input).glob("**/*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No Parquet relation files found under {args.input}")
    # ``all`` is reserved by datasets as a union-of-splits keyword.  Use a
    # regular split name and stream the Parquet rows to avoid a large cache and
    # unnecessary memory use for this dataset.
    dataset = load_dataset(
        "parquet",
        data_files={"train": [str(path) for path in parquet_files]},
        split="train",
        streaming=True,
    )
    written: Dict[str, int] = {"authentic": 0, "ai_edited": 0}
    seen_sources: Dict[str, Path] = {}
    for index, row in enumerate(dataset):
        source = _image_bytes(row.get("source_image"))
        edited = _image_bytes(row.get("edited_image"))
        if source is None or edited is None:
            continue
        group_id = hashlib.sha256(source).hexdigest()[:20]
        if group_id not in seen_sources and args.max_groups is not None and len(seen_sources) >= args.max_groups:
            continue
        split = _split(group_id, args.validation_fraction, args.test_fraction)
        source_path = args.output / split / "authentic" / group_id / "source.jpg"
        edited_path = args.output / split / "ai_edited" / group_id / f"edit_{index:06d}.jpg"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        edited_path.parent.mkdir(parents=True, exist_ok=True)
        if group_id not in seen_sources:
            _save_image(source, source_path)
            seen_sources[group_id] = source_path
            written["authentic"] += 1
        _save_image(edited, edited_path)
        written["ai_edited"] += 1
    print({"written": written, "groups": len(seen_sources), "output": str(args.output)})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
