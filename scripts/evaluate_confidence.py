#!/usr/bin/env python3
"""Confidence ablation, calibration and threshold selection from cached scores.

Reuses ``outputs/protocol/scores.json`` -- the raw per-image scores from the
protocol run -- so no GPU time is spent re-scoring. Everything downstream is
derived from those numbers.

Leakage discipline: calibration is fitted and thresholds are selected on the
**clean scores of the validation split only**, then frozen and applied to the
disjoint test split.

Example::

    ./.venv/bin/python scripts/evaluate_confidence.py --method temperature
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the patch-agreement confidence term, fit calibration, "
        "and select thresholds from clean validation data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scores", default="outputs/protocol/scores.json")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output-dir", default="outputs/confidence")
    parser.add_argument("--validation-fraction", type=float, default=0.4)
    parser.add_argument("--split-seed", type=int, default=1234)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--method", default="platt", choices=["platt", "temperature"])
    parser.add_argument(
        "--reliability-margin",
        type=float,
        default=0.25,
        help="Variant C distrusts patch agreement when |patch - whole| exceeds this",
    )
    parser.add_argument(
        "--patch-weight",
        type=float,
        default=0.15,
        help="Confidence weight given to patch agreement in variants B and C",
    )
    parser.add_argument(
        "--save-calibration",
        default=None,
        help="Optional path to persist the fitted calibration parameters",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from src.evaluation import confidence_eval as CE
    from src.evaluation.calibration import (
        ProbabilityCalibrator,
        calibration_summary,
        search_thresholds,
    )
    from src.evaluation.protocol import ScoredImage, split_records
    from src.utils.config import load_config

    scores_path = Path(args.scores)
    if not scores_path.is_absolute():
        scores_path = PROJECT_ROOT / scores_path
    if not scores_path.is_file():
        print(
            f"Cached scores not found: {scores_path}\n"
            f"This analysis needs labelled validation scores. Produce them with:\n"
            f"    ./.venv/bin/python scripts/evaluate_protocol.py "
            f"--checkpoint models/pretrained/pytorch_model.pt --limit 120 --device mps",
            file=sys.stderr,
        )
        return 4

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    payload = json.loads(scores_path.read_text(encoding="utf-8"))
    records = [ScoredImage.from_dict(row) for row in payload["records"]]
    validation, test = split_records(records, args.validation_fraction, args.split_seed)

    validation_labels = [r.binary_label for r in validation]
    if len(set(validation_labels)) < 2:
        print(
            "The validation split contains only one class, so calibration and "
            "threshold selection cannot run. Increase the sample size.",
            file=sys.stderr,
        )
        return 1

    # --- Task 2: calibration, fitted on CLEAN VALIDATION scores only --------
    raw_validation = [r.clean_score for r in validation]
    calibrator = ProbabilityCalibrator.fit(raw_validation, validation_labels, method=args.method)
    calibrated_validation = calibrator.transform(raw_validation)
    before = calibration_summary(validation_labels, raw_validation)
    after = calibration_summary(validation_labels, calibrated_validation)

    # --- Task 3: threshold search on calibrated clean validation scores -----
    selection = search_thresholds(validation_labels, calibrated_validation, args.target_fpr)
    threshold = float(selection.f1_optimal)

    if args.save_calibration:
        stored = ProbabilityCalibrator(
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
        path = Path(args.save_calibration)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        stored.save(path)

    # --- Task 1: confidence ablation on the TEST split ----------------------
    results = {
        variant: CE.evaluate_variant(
            test,
            variant,
            config,
            threshold,
            margin=args.reliability_margin,
            patch_weight=args.patch_weight,
        )
        for variant in CE.CONFIDENCE_VARIANTS
    }
    decision = CE.decide(results)

    report = {
        "source_scores": str(scores_path),
        "images": {"total": len(records), "validation": len(validation), "test": len(test)},
        "split": {
            "validation_fraction": args.validation_fraction,
            "seed": args.split_seed,
            "note": "stratified and disjoint; calibration and thresholds use validation only",
        },
        "calibration": {
            "method": calibrator.method,
            "fitted_on": "clean validation scores only",
            "parameters": calibrator.as_dict(),
            "before": before,
            "after": after,
            "improvement": {
                "expected_calibration_error": round(
                    before["expected_calibration_error"] - after["expected_calibration_error"], 6
                ),
                "brier_score": round(before["brier_score"] - after["brier_score"], 6),
            },
        },
        "threshold_selection": {
            "evaluated": "0.01 to 0.99 on calibrated clean validation scores",
            "chosen": threshold,
            "chosen_point": "f1_optimal",
            "target_false_positive_rate": args.target_fpr,
            "target_met": selection.target_met,
            "operating_points": selection.as_dict(),
        },
        "patch_agreement_weight_under_test": args.patch_weight,
        "confidence_ablation": {
            variant: {k: v for k, v in payload.items() if k != "rows"}
            for variant, payload in results.items()
        },
        "decision": decision,
    }
    (output_dir / "confidence_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    with (output_dir / "confidence_variants.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "variant",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "fpr",
                "fnr",
                "auc",
                "conf_auroc_vs_correct",
                "conf_gap",
                "point_biserial",
                "conf_ece",
                "uncertain_rate",
                "patch_agreement_applied",
            ]
        )
        for variant, payload in results.items():
            m = payload["probability_metrics"]
            r = payload["confidence_reliability"]
            writer.writerow(
                [
                    variant,
                    m["accuracy"],
                    m["precision"],
                    m["recall"],
                    m["f1"],
                    m["false_positive_rate"],
                    m["false_negative_rate"],
                    m["auc"],
                    r["auroc_confidence_vs_correctness"],
                    r["confidence_gap"],
                    r["point_biserial_correlation"],
                    r["expected_calibration_error"],
                    payload["uncertain_rate"],
                    payload["patch_agreement_applied"],
                ]
            )

    print_report(report, results)
    return 0


def print_report(report: dict, results: dict) -> None:
    line = "=" * 92
    print(line)
    print("CONFIDENCE ABLATION, CALIBRATION AND THRESHOLD SELECTION")
    print(line)
    images = report["images"]
    print(
        f"Images     : {images['total']} cached "
        f"({images['validation']} validation / {images['test']} test, disjoint)"
    )

    cal = report["calibration"]
    print()
    print(f"Calibration: {cal['method']} fitted on {cal['fitted_on']}")
    print(
        f"  ECE   {cal['before']['expected_calibration_error']:.4f} -> "
        f"{cal['after']['expected_calibration_error']:.4f}"
    )
    print(f"  Brier {cal['before']['brier_score']:.4f} -> {cal['after']['brier_score']:.4f}")

    sel = report["threshold_selection"]
    print()
    print("Thresholds : 0.01-0.99 on calibrated clean validation scores")
    points = sel["operating_points"]
    for name, key in (
        ("F1-optimal", "f1_optimal"),
        ("balanced", "balanced"),
        ("low-FPR", "low_false_positive"),
        ("high-recall", "high_recall"),
    ):
        metrics = points["selected_metrics"][key]
        print(
            f"  {name:<12} t={points[key + '_threshold']:.2f}  F1={metrics['f1']:.4f}  "
            f"FPR={metrics['false_positive_rate']:.4f}  recall={metrics['recall']:.4f}"
        )
    print(f"  FPR <= {sel['target_false_positive_rate']:.0%} target met: {sel['target_met']}")
    print(f"  CHOSEN: {sel['chosen']:.2f} ({sel['chosen_point']}), frozen for all conditions")

    print()
    print("Confidence variants (probability metrics are identical by design):")
    header = (
        f"  {'variant':<34}{'acc':>7}{'F1':>7}{'FPR':>7}"
        f"{'conf~correct':>14}{'gap':>8}{'ECE':>8}{'uncert':>8}{'applied':>9}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for payload in results.values():
        m = payload["probability_metrics"]
        r = payload["confidence_reliability"]
        auroc = r["auroc_confidence_vs_correctness"]
        print(
            f"  {payload['label']:<34}{m['accuracy']:>7.3f}{m['f1']:>7.3f}"
            f"{m['false_positive_rate']:>7.3f}"
            f"{(auroc if auroc is not None else 0):>14.4f}{(r['confidence_gap'] or 0):>8.4f}"
            f"{r['expected_calibration_error']:>8.4f}{payload['uncertain_rate']:>8.3f}"
            f"{payload['patch_agreement_applied']:>9}"
        )
    print()
    decision = report["decision"]
    print(f"DECISION: {decision['conclusion']}")
    print(f"  rule  : {decision.get('rule', '-')}")
    print(f"  reason: {decision['reason']}")
    print(line)


if __name__ == "__main__":
    raise SystemExit(main())
