#!/usr/bin/env python3
"""Write pre-augmented copies of the training split.

``training.mode='head_only'`` caches frozen backbone features once, so the
trainer refuses on-the-fly ``official_transformations``: a random draw would be
taken a single time and then reused unchanged for every epoch. Baking the
transformations into the dataset instead gives each augmented copy its own row
and its own cached feature, which is a fixed but genuine set of views.

Only the training split is augmented. Validation and test stay clean: the
robustness sweep transforms them at evaluation time, and augmenting them here
would make the held-out numbers incomparable with the baseline.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Dict, List

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.transformations import apply_transform, build_transform_specs  # noqa: E402
from src.utils.config import load_config  # noqa: E402

# Weighted towards the aggressive end: mild transformations barely move the
# score, so they teach the head very little.
WEIGHTS = {
    "jpeg_q30": 3, "jpeg_q50": 2, "jpeg_q70": 1, "jpeg_q90": 1,
    "blur_s2": 3, "blur_s1": 2, "blur_s0.5": 1,
    "resize_0.25x": 3, "resize_0.5x": 2,
    "noise_s0.1": 3, "noise_s0.05": 2, "noise_s0.02": 1,
    "color_jitter": 2, "center_crop_80": 2,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/local_edits")
    parser.add_argument("--output", default="data/local_edits_aug")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--views", type=int, default=3, help="Augmented copies per image")
    parser.add_argument("--limit", type=int, default=0, help="Cap source training images (0 = all)")
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = PROJECT_ROOT / args.source
    output = PROJECT_ROOT / args.output
    config = load_config(PROJECT_ROOT / args.config)
    specs = {spec.name: spec for spec in build_transform_specs(config)}
    unknown = sorted(set(WEIGHTS) - set(specs))
    if unknown:
        raise SystemExit(f"Unknown transformation name(s): {', '.join(unknown)}")

    population: List[str] = []
    for name, weight in WEIGHTS.items():
        population.extend([name] * weight)

    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    train = [r for r in manifest if r["split"] == "train"]
    rng = random.Random(args.seed)
    rng.shuffle(train)
    if args.limit:
        train = train[: args.limit]

    # Validation and test are copied through unchanged so the trainer sees a
    # complete dataset directory.
    written: List[Dict] = []
    used = collections.Counter()
    for record in manifest:
        if record["split"] != "train":
            src = source / record["file"]
            dst = output / record["file"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                dst.write_bytes(src.read_bytes())
            written.append({**record, "view": "clean"})

    for record in train:
        src = source / record["file"]
        folder = output / "train" / ("authentic" if record["label"] == 0 else "synthetic")
        folder.mkdir(parents=True, exist_ok=True)
        stem = Path(record["file"]).stem

        clean = folder / f"{stem}.jpg"
        if not clean.exists():
            clean.write_bytes(src.read_bytes())
        written.append({**record, "view": "clean", "file": f"train/{folder.name}/{clean.name}"})

        image = Image.open(src).convert("RGB")
        chosen = rng.sample(population, k=min(args.views * 3, len(population)))
        seen: List[str] = []
        for name in chosen:
            if name in seen:
                continue
            seen.append(name)
            if len(seen) > args.views:
                break
        for name in seen[: args.views]:
            # The suffix keeps the source stem intact so _group_id() still maps
            # every view of one image to a single group and cannot split them
            # across folds.
            target = folder / f"{stem}__{name.replace('.', 'p')}.jpg"
            if not target.exists():
                # A per-view seed keeps the stochastic families (noise, jitter)
                # reproducible while still differing between views.
                seed = (hash((stem, name)) & 0x7FFFFFFF)
                variant = apply_transform(image, specs[name], seed=seed)
                variant.convert("RGB").save(target, format="JPEG", quality=args.quality)
            used[name] += 1
            written.append(
                {**record, "view": name, "file": f"train/{folder.name}/{target.name}"}
            )

    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(written, indent=1), encoding="utf-8")

    counts = collections.Counter((r["split"], "synthetic" if r["label"] else "authentic") for r in written)
    print(f"source training images: {len(train)}   views per image: 1 clean + {args.views}")
    for split in ("train", "validation", "test"):
        a = counts[(split, "authentic")]
        s = counts[(split, "synthetic")]
        print(f"  {split:11s} authentic={a:6d}  synthetic={s:6d}  n={a + s}")
    print("\ntransformation usage:")
    for name, n in used.most_common():
        print(f"  {name:16s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
