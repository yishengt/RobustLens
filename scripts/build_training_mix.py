#!/usr/bin/env python3
"""Assemble a balanced, confound-normalised AI-vs-real training set.

Sources
-------
Defactify (MS COCOAI)   real COCO photos + SD 2.1 / SDXL / SD 3.
                        DALL-E 3 is deliberately excluded so the competition
                        benchmark stays a genuine unseen-generator test.
GenImage (arrow mirror) real ImageNet photos + ADM / glide / Midjourney.

Why every image is re-encoded
-----------------------------
In both sources every generated image is square and almost every authentic one
is not, so aspect ratio alone separates the classes. Resolution and compression
history leak the same way. Each image is therefore center-cropped to a square,
resized to one common edge and re-encoded at one JPEG quality, so a detector
cannot answer from the container instead of the pixels.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import random
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import pyarrow as pa
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENIMAGE_SHARDS = {
    "ADM": "data-00000-of-00075.arrow",
    "glide": "data-00000-of-00062.arrow",
    "Midjourney": "data-00000-of-00476.arrow",
}


def normalise(raw: bytes, edge: int, quality: int) -> bytes | None:
    """Center-crop to a square, resize to ``edge`` and re-encode as JPEG."""

    try:
        image = Image.open(io.BytesIO(raw))
        image = image.convert("RGB")
    except Exception:
        return None
    width, height = image.size
    if min(width, height) < 64:
        return None
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    image = image.resize((edge, edge), Image.BICUBIC)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def read_arrow(path: Path) -> pa.Table:
    with pa.memory_map(str(path), "rb") as source:
        try:
            return pa.ipc.open_stream(source).read_all()
        except Exception:
            source.seek(0)
            return pa.ipc.open_file(source).read_all()


def defactify_records(root: Path) -> Iterator[Tuple[bytes, int, str]]:
    """Yield ``(raw_bytes, label, generator)`` from the extracted Defactify set."""

    for label, folder in ((0, "authentic"), (1, "synthetic")):
        for path in sorted((root / folder).glob("*.jpg")):
            yield path.read_bytes(), label, path.name.split("_")[0]


def genimage_records(shard_root: Path, per_generator: int) -> Iterator[Tuple[bytes, int, str]]:
    for generator, filename in GENIMAGE_SHARDS.items():
        path = shard_root / "data/train" / generator / filename
        if not path.is_file():
            continue
        table = read_arrow(path)
        columns = table.select(["image", "label"]).to_pydict()
        taken: Dict[int, int] = collections.Counter()
        for raw, label in zip(columns["image"], columns["label"]):
            if taken[label] >= per_generator:
                continue
            taken[label] += 1
            yield bytes(raw), int(label), ("imagenet_real" if label == 0 else generator)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--defactify-dir", default="data/mix/defactify")
    parser.add_argument("--genimage-dir", default="data/raw_shards")
    parser.add_argument("--output", default="data/local_edits")
    parser.add_argument("--edge", type=int, default=384, help="Square edge length")
    parser.add_argument("--quality", type=int, default=95, help="Uniform JPEG quality")
    parser.add_argument("--per-generator", type=int, default=700)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = PROJECT_ROOT / args.output
    pool: List[Tuple[bytes, int, str]] = []
    pool.extend(defactify_records(PROJECT_ROOT / args.defactify_dir))
    pool.extend(genimage_records(PROJECT_ROOT / args.genimage_dir, args.per_generator))

    # Normalise first, then de-duplicate on the normalised bytes: two sources
    # can hold the same photograph at different sizes, and only the normalised
    # form reveals that.
    seen: Dict[str, str] = {}
    records: List[Tuple[str, bytes, int, str]] = []
    duplicates = 0
    for raw, label, generator in pool:
        encoded = normalise(raw, args.edge, args.quality)
        if encoded is None:
            continue
        digest = hashlib.sha256(encoded).hexdigest()
        if digest in seen:
            duplicates += 1
            continue
        seen[digest] = generator
        records.append((digest, encoded, label, generator))

    # Balance: equal authentic and synthetic, and equal weight per generator.
    by_generator: Dict[str, List] = collections.defaultdict(list)
    for record in records:
        by_generator[record[3]].append(record)
    rng = random.Random(args.seed)
    for items in by_generator.values():
        rng.shuffle(items)
    fake_generators = [g for g in by_generator if g != "coco" and not g.endswith("_real")]
    fake_generators = [g for g in by_generator if g not in {"coco", "imagenet_real"}]
    real_generators = [g for g in by_generator if g in {"coco", "imagenet_real"}]

    n_fake_total = sum(len(by_generator[g]) for g in fake_generators)
    n_real_total = sum(len(by_generator[g]) for g in real_generators)
    target = min(n_fake_total, n_real_total)

    def take(groups: List[str], total: int) -> List:
        per = total // len(groups)
        chosen: List = []
        for index, group in enumerate(groups):
            want = per + (1 if index < total - per * len(groups) else 0)
            chosen.extend(by_generator[group][:want])
        return chosen

    selected = take(fake_generators, target) + take(real_generators, target)
    rng.shuffle(selected)

    # Split by normalised digest so an image can never straddle two splits.
    n_validation = int(len(selected) * args.validation_fraction)
    n_test = int(len(selected) * args.test_fraction)
    splits = {
        "validation": selected[:n_validation],
        "test": selected[n_validation : n_validation + n_test],
        "train": selected[n_validation + n_test :],
    }

    manifest: List[Dict] = []
    for split, items in splits.items():
        for digest, encoded, label, generator in items:
            folder = output / split / ("authentic" if label == 0 else "synthetic")
            folder.mkdir(parents=True, exist_ok=True)
            name = f"{generator}_{digest[:16]}.jpg"
            (folder / name).write_bytes(encoded)
            manifest.append(
                {
                    "split": split,
                    "file": f"{split}/{'authentic' if label == 0 else 'synthetic'}/{name}",
                    "label": label,
                    "generator": generator,
                    "sha256": digest,
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    print(f"normalised to {args.edge}x{args.edge} JPEG q{args.quality}")
    print(f"dropped {duplicates} duplicate image(s) after normalisation")
    for split, items in splits.items():
        counts = collections.Counter(("synthetic" if i[2] else "authentic") for i in items)
        generators = collections.Counter(i[3] for i in items)
        print(f"  {split:11s} n={len(items):5d}  {dict(counts)}")
        print(f"              {dict(generators)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
