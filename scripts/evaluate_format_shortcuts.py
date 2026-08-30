#!/usr/bin/env python3
"""Probe whether a checkpoint relies on class-correlated image file formats."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.shortcut_checks import evaluate_format_reencoding  # noqa: E402
from src.pipeline.pipeline import DetectionPipeline  # noqa: E402
from src.utils.config import load_config, resolve_config_path  # noqa: E402


def _images(root: Path, limit: int) -> list[Path]:
    allowed = {".jpg", ".jpeg", ".png", ".webp"}
    return [path for path in sorted(root.rglob("*")) if path.suffix.lower() in allowed][:limit]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authentic-dir", required=True)
    parser.add_argument("--synthetic-dir", required=True)
    parser.add_argument("--checkpoint", default="models/pretrained/pytorch_model.pt")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--per-class-limit", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.69)
    parser.add_argument("--output-format", choices=("PNG", "JPEG", "WEBP"), default="PNG")
    parser.add_argument("--output", default="outputs/data_quality/file_format_shortcut.json")
    args = parser.parse_args(argv)
    if args.per_class_limit <= 0:
        parser.error("--per-class-limit must be positive")

    config = load_config(args.config)
    config.setdefault("transformations", {})["enabled"] = False
    config.setdefault("patches", {})["enabled"] = False
    pipeline = DetectionPipeline.from_checkpoint(
        resolve_config_path(config, args.checkpoint),
        config,
        device=None if args.device == "auto" else args.device,
        explain_images=False,
    )
    if args.adapter_dir:
        from src.finetune.model import load_saved_adapter_into_model

        load_saved_adapter_into_model(
            pipeline.bundle.model, resolve_config_path(config, args.adapter_dir)
        )
    authentic = _images(Path(args.authentic_dir).expanduser(), args.per_class_limit)
    synthetic = _images(Path(args.synthetic_dir).expanduser(), args.per_class_limit)
    if not authentic or not synthetic:
        parser.error("Both input directories must contain supported images")
    paths = [*authentic, *synthetic]
    labels = [0] * len(authentic) + [1] * len(synthetic)
    report = evaluate_format_reencoding(
        paths,
        labels,
        lambda path: pipeline.analyse_path(path).ai_probability,
        args.threshold,
        args.output_format,
    )
    destination = resolve_config_path(config, args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_image"}, indent=2))
    print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
