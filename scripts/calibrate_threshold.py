#!/usr/bin/env python3
"""Fit calibration and freeze thresholds from clean validation data only.

The command deliberately accepts only the SID_Set ``validation`` split.  It
fits on original images, persists Platt parameters plus selected thresholds,
then evaluates that same frozen balanced threshold on clean and transformed
images without retuning per transformation.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit probability calibration and fixed thresholds on SID_Set validation data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="data/sid_set")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--output", default="outputs/calibration.json", help="Calibration parameter file"
    )
    parser.add_argument("--report", default="outputs/calibration_report.json")
    parser.add_argument("--plots-dir", default="outputs/calibration_plots")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--per-class-limit", type=int, default=None)
    parser.add_argument("--target-fpr", type=float, default=None)
    parser.add_argument(
        "--method",
        default="platt",
        choices=["platt", "temperature"],
        help="platt fits slope+intercept; temperature fits slope only (no boundary shift)",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        print("--limit must be positive", file=sys.stderr)
        return 2
    try:
        from src.evaluation.calibration import (
            ProbabilityCalibrator,
            calibration_summary,
            plot_calibration_results,
            plot_clean_vs_transformed,
            plot_roc_pr_curves,
            search_thresholds,
        )
        from src.evaluation.sid_set import find_shards, iter_labelled_images
        from src.pipeline.pipeline import DetectionPipeline
        from src.utils.config import load_config, resolve_config_path

        config = load_config(args.config)
        data_dir = Path(args.data_dir).expanduser()
        if not data_dir.is_absolute():
            data_dir = PROJECT_ROOT / data_dir
        shards = find_shards(data_dir, "validation")
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"Setup error: {exc}", file=sys.stderr)
        return 2

    if args.target_fpr is None:
        args.target_fpr = float(
            (config.get("calibration", {}) or {}).get("target_false_positive_rate", 0.01)
        )

    # Clean validation fitting: no transformations, no explanation, and no
    # calibration loaded. Only the original raw model probability is collected.
    fit_config = copy.deepcopy(config)
    fit_config.setdefault("transformations", {})["enabled"] = False
    fit_config.setdefault("calibration", {})["enabled"] = False
    try:
        fit_pipeline = DetectionPipeline.from_checkpoint(
            resolve_config_path(config, args.checkpoint),
            fit_config,
            device=args.device,
            explain_images=False,
        )
        labels = []
        raw_scores = []
        samples = iter_labelled_images(
            shards,
            limit=args.limit,
            per_class_limit=args.per_class_limit,
            batch_size=int(config.get("inference", {}).get("batch_size", 32)),
        )
        for sample in samples:
            result = fit_pipeline.analyse_image(sample.image)
            original = next(item for item in result.predictions if item.is_original)
            labels.append(int(sample.binary_label))
            raw_scores.append(
                float(
                    original.raw_probability
                    if original.raw_probability is not None
                    else original.ai_probability
                )
            )
            if not args.quiet and len(labels) % 25 == 0:
                print(f"Collected {len(labels)} clean validation images", file=sys.stderr)
        calibrator = ProbabilityCalibrator.fit(
            raw_scores, labels, input_type="probabilities", method=args.method
        )
        calibrated_scores = calibrator.transform(raw_scores)
        selection = search_thresholds(labels, calibrated_scores, args.target_fpr)
        calibrator = ProbabilityCalibrator(
            method=calibrator.method,
            input_type=calibrator.input_type,
            scale=calibrator.scale,
            bias=calibrator.bias,
            fitted_on="clean_validation",
            selected_thresholds={
                "balanced": selection.balanced,
                "f1_optimal": selection.f1_optimal,
                "low_false_positive": selection.low_false_positive,
                "high_recall": selection.high_recall,
            },
        )
        calibration_path = calibrator.save(resolve_config_path(config, args.output))
    except (FileNotFoundError, ValueError, RuntimeError, OSError, MemoryError, ImportError) as exc:
        print(f"Calibration fitting failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    clean_calibration = calibration_summary(labels, calibrated_scores)
    report = {
        "data_split": "validation",
        "fit_data": "clean validation originals only",
        "count": len(labels),
        "calibration": calibrator.as_dict(),
        "clean_validation_calibration": clean_calibration,
        "threshold_selection": selection.as_dict(),
        "plots": [],
        "robustness": None,
    }
    try:
        report["plots"] = plot_calibration_results(
            labels,
            raw_scores,
            calibrated_scores,
            selection,
            resolve_config_path(config, args.plots_dir),
        )
        report["plots"].extend(
            plot_roc_pr_curves(
                labels, calibrated_scores, resolve_config_path(config, args.plots_dir)
            )
        )
    except RuntimeError as exc:
        report["plot_error"] = str(exc)

    # Reuse the persisted calibrator and freeze the balanced threshold for all
    # transformations. A second fresh iterator is essential: no per-transform
    # threshold search occurs here.
    eval_config = copy.deepcopy(config)
    eval_config["calibration"] = {
        **(eval_config.get("calibration", {}) or {}),
        "enabled": True,
        "path": str(calibration_path),
        "use_selected_threshold": True,
    }
    try:
        from src.evaluation.benchmark import run_benchmark

        eval_pipeline = DetectionPipeline.from_checkpoint(
            resolve_config_path(config, args.checkpoint),
            eval_config,
            device=args.device,
            explain_images=False,
        )
        robustness = run_benchmark(
            eval_pipeline,
            iter_labelled_images(
                shards,
                limit=args.limit,
                per_class_limit=args.per_class_limit,
                batch_size=int(config.get("inference", {}).get("batch_size", 32)),
            ),
            threshold=selection.balanced,
        )
        report["robustness"] = robustness.as_dict()
        if report["robustness"].get("per_version"):
            report["plots"].append(
                plot_clean_vs_transformed(
                    report["robustness"]["per_version"],
                    resolve_config_path(config, args.plots_dir),
                )
            )
    except (ValueError, RuntimeError, OSError, MemoryError, ImportError) as exc:
        report["robustness_error"] = f"{type(exc).__name__}: {exc}"

    report_path = resolve_config_path(config, args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "calibration_file": str(calibration_path),
                "report_file": str(report_path),
                "images": len(labels),
                "balanced_threshold": selection.balanced,
                "low_false_positive_threshold": selection.low_false_positive,
                "high_recall_threshold": selection.high_recall,
                "target_fpr_met": selection.target_met,
                "robustness_evaluated_with_fixed_threshold": report["robustness"] is not None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
