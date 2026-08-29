#!/usr/bin/env python3
"""Evaluate a checkpoint against labelled SID_Set shards.

Reports detection quality (accuracy, AUC, precision/recall/F1) for the fused
prediction and robustness metrics for every transformation.

Example::

    python scripts/evaluate_dataset.py \
        --data-dir data/sid_set \
        --checkpoint checkpoints/best.pt \
        --limit 300 \
        --output outputs/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the detector against labelled SID_Set images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="data/sid_set", help="Directory of parquet shards")
    parser.add_argument("--split", default="validation", help="Shard split prefix to evaluate")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt", help="Model checkpoint")
    parser.add_argument("--config", default="configs/config.yaml", help="YAML configuration")
    parser.add_argument(
        "--limit", type=int, default=300, help="Maximum images to evaluate (0 = all)"
    )
    parser.add_argument(
        "--per-class-limit",
        type=int,
        default=None,
        help="Cap images taken from each of the three source classes",
    )
    parser.add_argument("--output", default=None, help="Optional path for the JSON report")
    parser.add_argument(
        "--device", default=None, choices=["auto", "cpu", "cuda", "mps"], help="Override the device"
    )
    parser.add_argument(
        "--threshold", type=float, default=None, help="Override the binary decision threshold"
    )
    parser.add_argument(
        "--no-transformations",
        action="store_true",
        help="Score only the original image, skipping the robustness breakdown",
    )
    parser.add_argument(
        "--describe", action="store_true", help="Print shard statistics and exit"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-image progress")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from src.evaluation.benchmark import format_report, run_benchmark
    from src.evaluation.sid_set import describe_shards, find_shards, iter_labelled_images
    from src.pipeline.model_loader import ModelSetupError
    from src.pipeline.pipeline import DetectionPipeline
    from src.utils.config import load_config, resolve_config_path

    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    try:
        shards = find_shards(data_dir, args.split)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 4

    if args.describe:
        print(json.dumps(describe_shards(shards), indent=2))
        return 0

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.threshold is not None:
        config.setdefault("inference", {})["threshold"] = float(args.threshold)
    if args.no_transformations:
        config.setdefault("transformations", {})["enabled"] = False

    threshold = float(config.get("inference", {}).get("threshold", 0.5))

    try:
        pipeline = DetectionPipeline.from_checkpoint(
            resolve_config_path(config, args.checkpoint),
            config,
            device=args.device,
            explain_images=False,  # Grad-CAM is not needed for metrics
        )
    except ModelSetupError as exc:
        print(f"Model setup error: {exc}", file=sys.stderr)
        return 3

    limit = None if args.limit in (0, None) else int(args.limit)
    samples = iter_labelled_images(
        shards, limit=limit, per_class_limit=args.per_class_limit
    )

    started = time.time()

    def progress(index: int, img_id: str) -> None:
        elapsed = time.time() - started
        rate = index / elapsed if elapsed > 0 else 0.0
        total = f"/{limit}" if limit else ""
        print(f"[{index}{total}] {img_id[:40]:42} {rate:5.2f} img/s", file=sys.stderr)

    try:
        report = run_benchmark(
            pipeline, samples, threshold=threshold, progress=None if args.quiet else progress
        )
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    report.dataset.update(
        {
            "name": "SID_Set",
            "split": args.split,
            "shards": [shard.name for shard in shards],
            "source": "https://huggingface.co/datasets/saberzl/SID_Set",
            "label_mapping": "0=real -> 0; 1=full_synthetic, 2=tampered -> 1 (AI-generated)",
        }
    )

    elapsed = time.time() - started
    print(format_report(report))
    print(f"Evaluated {report.count} images in {elapsed:.1f}s "
          f"({report.count / elapsed:.2f} img/s)")

    if args.output:
        output = resolve_config_path(config, args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(report.as_dict(), handle, indent=2)
            handle.write("\n")
        print(f"Wrote {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
