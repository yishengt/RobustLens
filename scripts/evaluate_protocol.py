#!/usr/bin/env python3
"""Run the full Track 5 evaluation protocol and save every artefact.

One scoring pass per image records the clean score, all transformed-version
scores and the patch evidence. Everything else -- per-transformation
robustness, the four-system ablation, confusion matrices, failure grids -- is
derived from that cache, so the systems are compared on identical forward
passes and re-analysis is instant.

The decision threshold is selected once on the CLEAN scores of a validation
split and then frozen for every condition and every variant.

Examples::

    # full protocol, ~20 s/image on MPS
    ./.venv/bin/python scripts/evaluate_protocol.py \\
      --checkpoint models/pretrained/pytorch_model.pt --limit 120 --device mps

    # re-analyse a finished run without re-scoring anything
    ./.venv/bin/python scripts/evaluate_protocol.py --reuse-scores
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track 5 evaluation protocol: robustness, ablation and failure analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="data/sid_set")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--checkpoint", default="models/pretrained/pytorch_model.pt")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output-dir", default="outputs/protocol")
    parser.add_argument("--limit", type=int, default=120, help="Total images to score")
    parser.add_argument(
        "--per-class-limit", type=int, default=None, help="Cap per source class (balances)"
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.4,
        help="Share of images used ONLY for threshold selection",
    )
    parser.add_argument("--split-seed", type=int, default=1234)
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.05,
        help="Target false-positive rate for threshold selection on clean validation data",
    )
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--no-patches", action="store_true", help="Skip patch scoring (faster)")
    parser.add_argument(
        "--reuse-scores", action="store_true", help="Re-analyse a cached scoring pass"
    )
    parser.add_argument("--examples", type=int, default=8, help="Failure examples per class")
    parser.add_argument(
        "--calibration",
        default=None,
        help="Optional calibration JSON; applied to every cached score before analysis",
    )
    parser.add_argument(
        "--operating-point",
        default="f1_optimal",
        choices=["f1_optimal", "balanced", "low_false_positive", "high_recall"],
        help="Which threshold from the clean-validation search to freeze",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import torch

    from src.evaluation import protocol as P
    from src.evaluation import protocol_charts as C
    from src.evaluation.sid_set import find_shards, iter_labelled_images
    from src.pipeline.model_loader import ModelSetupError, load_model
    from src.pipeline.preprocessing import Preprocessor
    from src.utils.config import load_config
    from src.utils.device import describe_device

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    charts_dir = output_dir / "charts"
    tables_dir = output_dir / "tables"
    examples_dir = output_dir / "examples"
    for directory in (output_dir, charts_dir, tables_dir, examples_dir):
        directory.mkdir(parents=True, exist_ok=True)

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    try:
        shards = find_shards(data_dir, args.split)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 4

    scores_path = output_dir / "scores.json"
    device_description = "cached"
    model_summary = {}

    # ---------------- scoring pass (the expensive part) -------------------
    if args.reuse_scores and scores_path.is_file():
        cached = json.loads(scores_path.read_text(encoding="utf-8"))
        records = [P.ScoredImage.from_dict(row) for row in cached["records"]]
        model_summary = cached.get("model", {})
        device_description = cached.get("device", "cached")
        print(f"Reusing {len(records)} cached scores from {scores_path}", file=sys.stderr)
    else:
        try:
            bundle = load_model(args.checkpoint, config, device=args.device)
        except ModelSetupError as exc:
            print(f"Model setup error: {exc}", file=sys.stderr)
            return 3
        model_summary = bundle.summary()
        device_description = describe_device(bundle.device)
        preprocessor = Preprocessor.from_config(config)

        samples = list(
            iter_labelled_images(
                shards,
                limit=None if args.limit in (0, None) else args.limit,
                per_class_limit=args.per_class_limit,
            )
        )
        if not samples:
            print("No images were loaded from the shards.", file=sys.stderr)
            return 4

        started = time.time()

        def progress(index: int, img_id: str, seconds: float) -> None:
            rate = (time.time() - started) / index
            remaining = (len(samples) - index) * rate
            print(
                f"[{index}/{len(samples)}] {img_id[:34]:36} {seconds:5.1f}s/img  "
                f"eta {remaining / 60:5.1f} min",
                file=sys.stderr,
            )

        records = P.score_images(
            bundle,
            preprocessor,
            samples,
            config,
            with_patches=not args.no_patches,
            progress=None if args.quiet else progress,
        )
        scores_path.write_text(
            json.dumps(
                {
                    "records": [record.as_dict() for record in records],
                    "model": model_summary,
                    "device": device_description,
                    "shards": [shard.name for shard in shards],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote raw scores to {scores_path}", file=sys.stderr)

    if len(records) < 4:
        print("Too few images to evaluate; increase --limit.", file=sys.stderr)
        return 1

    # ---------------- optional calibration --------------------------------
    calibration_info = {"applied": False, "reason": "no calibration file supplied"}
    if args.calibration:
        from src.evaluation.calibration import ProbabilityCalibrator

        calibration_path = Path(args.calibration)
        if not calibration_path.is_absolute():
            calibration_path = PROJECT_ROOT / calibration_path
        try:
            calibrator = ProbabilityCalibrator.load(calibration_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Calibration error: {exc}", file=sys.stderr)
            return 2
        # Applied uniformly to every version, so no condition gets special
        # treatment and the transformation comparison stays like-for-like.
        for record in records:
            record.version_scores = {
                name: float(calibrator.transform([score])[0])
                for name, score in record.version_scores.items()
            }
            if record.patch_evidence is not None:
                record.patch_evidence = float(calibrator.transform([record.patch_evidence])[0])
        calibration_info = {
            "applied": True,
            "path": str(calibration_path),
            "method": calibrator.method,
            "fitted_on": calibrator.fitted_on,
        }

    # ---------------- leakage-safe split + frozen threshold ---------------
    validation, test = P.split_records(records, args.validation_fraction, args.split_seed)
    try:
        selection = P.select_fixed_threshold(validation, args.target_fpr)
    except ValueError as exc:
        print(f"Threshold selection failed: {exc}", file=sys.stderr)
        return 1
    threshold = float(getattr(selection, args.operating_point))

    # ---------------- analyses, all at the ONE frozen threshold -----------
    per_version = P.per_transformation_metrics(test, threshold)
    summaries = {
        metric: P.robustness_summary(per_version, metric)
        for metric in ("accuracy", "balanced_accuracy", "f1", "auc", "recall")
    }
    variants = P.variant_metrics(test, threshold, config)
    subgroups = P.subgroup_metrics(test, threshold, config)
    failures = P.failure_examples(test, threshold, config, top_k=args.examples)
    distributions = P.confidence_distributions(test, config)
    runtime = P.runtime_summary(records)
    consistency = P.consistency_distribution(test, config)

    training_dataset = model_summary.get("training_dataset") or (
        (model_summary.get("metadata") or {}).get("dataset")
    )

    report = {
        "protocol": "TikTok TechJam 2026 Track 5",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model_summary,
        "device": device_description,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "torch": torch.__version__,
        },
        "dataset": {
            "name": "SID_Set",
            "split": args.split,
            "shards": [shard.name for shard in shards],
            "images_scored": len(records),
            "validation_images": len(validation),
            "test_images": len(test),
            "validation_fraction": args.validation_fraction,
            "split_seed": args.split_seed,
        },
        "calibration": calibration_info,
        "threshold": {
            "value": threshold,
            "operating_point": args.operating_point,
            "selected_on": "clean scores of the validation split only",
            "target_false_positive_rate": args.target_fpr,
            "target_met": selection.target_met,
            "retuned_per_transformation": False,
            "alternatives": {
                "f1_optimal": selection.f1_optimal,
                "balanced": selection.balanced,
                "low_false_positive": selection.low_false_positive,
                "high_recall": selection.high_recall,
            },
        },
        "per_transformation": per_version,
        "robustness": summaries,
        "system_ablation": variants,
        "generation_families": subgroups,
        "failures": failures,
        "confidence_distributions": {
            key: {k: v for k, v in payload.items() if k != "scores"}
            for key, payload in distributions.items()
        },
        "transformation_consistency": consistency,
        "runtime": runtime,
        "claims": {
            "dataset_source_holdout": P.dataset_holdout_statement(training_dataset),
            "unseen_generator": P.generator_claim_statement(),
        },
        "configuration": {
            "transformations": config.get("transformations"),
            "patches": config.get("patches"),
            "fusion": config.get("fusion"),
            "confidence": config.get("confidence"),
            "labels": config.get("labels"),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # ---------------- CSV tables ------------------------------------------
    write_tables(tables_dir, per_version, summaries, variants, subgroups, threshold)

    # ---------------- charts ----------------------------------------------
    produced = [
        C.plot_robustness(summaries["accuracy"], charts_dir, "accuracy"),
        C.plot_severity_curves(per_version, charts_dir, "accuracy"),
        C.plot_confusion_matrices(variants, charts_dir),
        C.plot_confidence_distributions(distributions, threshold, charts_dir),
        C.plot_variant_comparison(variants, charts_dir),
        C.plot_subgroups(subgroups, charts_dir),
    ]
    produced = [path for path in produced if path]

    # ---------------- failure grids (need the pixels back) ----------------
    wanted = {row["img_id"] for row in failures["false_positives"] + failures["false_negatives"]}
    if wanted:
        lookup = {}
        for sample in iter_labelled_images(
            shards,
            limit=None if args.limit in (0, None) else args.limit,
            per_class_limit=args.per_class_limit,
        ):
            if sample.img_id in wanted:
                lookup[sample.img_id] = sample.image.convert("RGB")
        for key, title in (
            ("false_positives", "False positives: authentic images flagged as AI-generated"),
            ("false_negatives", "False negatives: AI-generated images reported as authentic"),
        ):
            path = C.save_failure_grid(
                test, failures[key], lookup, title, examples_dir / f"{key}.png"
            )
            if path:
                produced.append(path)

    print_summary(report, threshold, produced, output_dir)
    return 0


def write_tables(tables_dir, per_version, summaries, variants, subgroups, threshold) -> None:
    """Write the CSV tables that go straight into a report."""

    import csv

    with (tables_dir / "per_transformation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "condition",
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "auc",
                "fpr",
                "fnr",
                "accuracy_drop",
                "robustness_ratio",
                "threshold",
            ]
        )
        clean_accuracy = per_version.get("clean", {}).get("accuracy")
        for name, metrics in per_version.items():
            accuracy = metrics.get("accuracy")
            writer.writerow(
                [
                    name,
                    accuracy,
                    metrics.get("balanced_accuracy"),
                    metrics.get("precision"),
                    metrics.get("recall"),
                    metrics.get("f1"),
                    metrics.get("auc"),
                    metrics.get("false_positive_rate"),
                    metrics.get("false_negative_rate"),
                    None if clean_accuracy is None else round(clean_accuracy - accuracy, 6),
                    None if not clean_accuracy else round(accuracy / clean_accuracy, 6),
                    threshold,
                ]
            )

    with (tables_dir / "system_ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "system",
                "count",
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "auc",
                "fpr",
                "fnr",
            ]
        )
        for name, payload in variants.items():
            metrics = payload.get("metrics") or {}
            writer.writerow(
                [
                    payload.get("label", name),
                    payload.get("count"),
                    metrics.get("accuracy"),
                    metrics.get("balanced_accuracy"),
                    metrics.get("precision"),
                    metrics.get("recall"),
                    metrics.get("f1"),
                    metrics.get("auc"),
                    metrics.get("false_positive_rate"),
                    metrics.get("false_negative_rate"),
                ]
            )

    with (tables_dir / "robustness_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "metric",
                "clean",
                "average_transformed",
                "worst_case",
                "worst_transformation",
                "largest_drop",
                "average_ratio",
            ]
        )
        for metric, summary in summaries.items():
            writer.writerow(
                [
                    metric,
                    summary.get("clean"),
                    summary.get("average_transformed"),
                    summary.get("worst_case"),
                    summary.get("worst_transformation"),
                    summary.get("largest_drop"),
                    summary.get("average_ratio"),
                ]
            )

    with (tables_dir / "generation_families.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "family",
                "ai_images",
                "authentic_images",
                "accuracy",
                "recall",
                "fpr",
                "mean_ai_probability",
            ]
        )
        for name, payload in subgroups.items():
            metrics = payload.get("metrics") or {}
            writer.writerow(
                [
                    name,
                    payload.get("ai_images"),
                    payload.get("authentic_images"),
                    metrics.get("accuracy"),
                    metrics.get("recall"),
                    metrics.get("false_positive_rate"),
                    payload.get("mean_ai_probability"),
                ]
            )


def print_summary(report, threshold, produced, output_dir) -> None:
    """Console summary of the headline findings."""

    line = "=" * 74
    print(line)
    print("TRACK 5 EVALUATION PROTOCOL")
    print(line)
    model = report["model"]
    print(
        f"Model      : {model.get('architecture')} "
        f"({model.get('parameters_millions')} M params) on {report['device']}"
    )
    dataset = report["dataset"]
    print(
        f"Dataset    : {dataset['name']} {dataset['split']} - "
        f"{dataset['images_scored']} scored "
        f"({dataset['validation_images']} val / {dataset['test_images']} test)"
    )
    print(f"Threshold  : {threshold:.2f}  (fixed, from clean validation only; never retuned)")
    print()

    clean = report["per_transformation"].get("clean", {})
    print(
        f"Clean      : acc {clean.get('accuracy', 0):.4f}  auc {clean.get('auc') or 0:.4f}  "
        f"F1 {clean.get('f1', 0):.4f}  FPR {clean.get('false_positive_rate', 0):.4f}"
    )

    accuracy = report["robustness"]["accuracy"]
    if accuracy.get("average_transformed") is not None:
        print(
            f"Transformed: avg {accuracy['average_transformed']:.4f}  "
            f"worst {accuracy['worst_case']:.4f} ({accuracy['worst_transformation']})  "
            f"largest drop {accuracy['largest_drop']:.4f}"
        )
    print()

    print("System ablation (identical images, one frozen threshold):")
    print(f"  {'system':<26}{'acc':>8}{'bal acc':>9}{'F1':>8}{'AUC':>8}{'FPR':>8}")
    for payload in report["system_ablation"].values():
        metrics = payload.get("metrics")
        if not metrics:
            continue
        print(
            f"  {payload['label']:<26}{metrics['accuracy']:>8.4f}"
            f"{metrics['balanced_accuracy']:>9.4f}{metrics['f1']:>8.4f}"
            f"{(metrics['auc'] or 0):>8.4f}{metrics['false_positive_rate']:>8.4f}"
        )
    print()

    failures = report["failures"]
    print(
        f"Failures   : {failures['false_positive_count']} false positives, "
        f"{failures['false_negative_count']} false negatives"
    )
    runtime = report["runtime"]
    print(
        f"Runtime    : {runtime['seconds_per_image_mean']:.2f} s/image "
        f"({runtime['forward_passes_per_image']} forward passes)"
    )
    print()
    print(f"Wrote {len(produced)} figures and 4 tables to {output_dir}")
    print(line)


if __name__ == "__main__":
    raise SystemExit(main())
