#!/usr/bin/env python3
"""Command-line batch inference for the AI-generated image detector.

Example::

    python scripts/run_inference.py \
        --input-dir path/to/images \
        --checkpoint checkpoints/best.pt \
        --output outputs/predictions.json \
        --detailed-output outputs/predictions_detailed.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``python scripts/run_inference.py`` work from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI. Kept import-free so ``--help`` works without torch."""

    parser = argparse.ArgumentParser(
        description="Estimate the AI-generated probability for every image in a directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", required=True, help="Directory of images to analyse")
    # Defaults to where scripts/setup.py downloads the checkpoint, so the
    # documented command works with no extra flags. checkpoints/best.pt is
    # still accepted if you keep one there.
    parser.add_argument(
        "--checkpoint",
        default="models/pretrained/pytorch_model.pt",
        help="Model checkpoint",
    )
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="Optional RobustLens adapter directory applied after loading the base checkpoint",
    )
    parser.add_argument("--config", default="configs/config.yaml", help="YAML configuration")
    parser.add_argument("--output", default="outputs/predictions.json", help="Simple JSON output")
    parser.add_argument(
        "--detailed-output",
        default=None,
        help="Optional path for the detailed JSON report (labels, per-transform scores, errors)",
    )
    parser.add_argument(
        "--device", default=None, choices=["auto", "cpu", "cuda", "mps"], help="Override the device"
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override inference.batch_size")
    parser.add_argument(
        "--threshold", type=float, default=None, help="Override the binary decision threshold"
    )
    parser.add_argument(
        "--no-transformations",
        action="store_true",
        help="Score the original image only, skipping the robustness transforms",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Disable calibration fitted for a different checkpoint",
    )
    parser.add_argument(
        "--frequency",
        action="store_true",
        help="Enable the optional frequency/noise analysis module",
    )
    patches = parser.add_mutually_exclusive_group()
    patches.add_argument(
        "--patches",
        dest="patches",
        action="store_true",
        default=None,
        help="Enable patch-level localisation (adds one forward pass per patch)",
    )
    patches.add_argument(
        "--no-patches",
        dest="patches",
        action="store_false",
        help="Disable patch-level localisation (default in batch mode, for speed)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only process the first N images"
    )
    parser.add_argument(
        "--on-error",
        default="skip",
        choices=["skip", "fallback"],
        help="Leave failed images out of the simple output, or record --fallback-pred for them",
    )
    parser.add_argument(
        "--fallback-pred",
        type=float,
        default=0.5,
        help="Score recorded for failed images when --on-error=fallback",
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Write image_path values relative to --input-dir",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-image progress output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Imports are deferred so ``--help`` stays usable before the ML stack is
    # installed, and so a missing dependency reports itself clearly.
    from src.inference.batch_inference import run_batch
    from src.pipeline.model_loader import ModelSetupError
    from src.pipeline.validation import ImageValidationError
    from src.utils.config import load_config, resolve_config_path

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    # Apply CLI overrides on top of the config file.
    if args.batch_size is not None:
        config.setdefault("inference", {})["batch_size"] = int(args.batch_size)
    if args.threshold is not None:
        config.setdefault("inference", {})["threshold"] = float(args.threshold)
    if args.no_transformations:
        config.setdefault("transformations", {})["enabled"] = False
    if args.no_calibration:
        config.setdefault("calibration", {})["enabled"] = False
    if args.frequency:
        config.setdefault("frequency", {})["enabled"] = True
    # Patch analysis costs one forward pass per patch on every image, so batch
    # runs opt out by default. --patches turns it back on.
    config.setdefault("patches", {})["enabled"] = bool(args.patches)

    input_dir = resolve_config_path(config, args.input_dir)

    def progress(index: int, total: int, path: str) -> None:
        print(f"[{index}/{total}] {path}", file=sys.stderr)

    try:
        report = run_batch(
            image_dir=input_dir,
            checkpoint_path=resolve_config_path(config, args.checkpoint),
            adapter_dir=(resolve_config_path(config, args.adapter_dir) if args.adapter_dir else None),
            config=config,
            output_path=resolve_config_path(config, args.output),
            detailed_output_path=(
                resolve_config_path(config, args.detailed_output)
                if args.detailed_output
                else None
            ),
            device=args.device,
            limit=args.limit,
            on_error=args.on_error,
            fallback_pred=args.fallback_pred,
            relative_to=input_dir if args.relative_paths else None,
            progress=None if args.quiet else progress,
        )
    except ModelSetupError as exc:
        print(f"Model setup error: {exc}", file=sys.stderr)
        return 3
    except ImageValidationError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 4
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Inference failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(report.summary(), file=sys.stderr)
    print(f"Wrote {args.output}", file=sys.stderr)
    if args.detailed_output:
        print(f"Wrote {args.detailed_output}", file=sys.stderr)
    if report.errors:
        print(f"{len(report.errors)} error(s):", file=sys.stderr)
        for error in report.errors[:10]:
            print(f"  - {error.get('image_path')}: {error.get('error')}", file=sys.stderr)
        if len(report.errors) > 10:
            print(f"  ... and {len(report.errors) - 10} more", file=sys.stderr)

    return 0 if report.processed else 1


if __name__ == "__main__":
    raise SystemExit(main())
