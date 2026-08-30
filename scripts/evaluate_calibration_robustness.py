#!/usr/bin/env python3
"""Report calibration on held-out images, clean and under every transformation.

The confidence report fits a calibrator on the clean validation split and scores
its ECE on that same split. This script answers the two questions that leaves
open: how well the calibration transfers to images it never saw, and whether it
survives the transformations the detector is built to withstand.

It reads the cached scores from ``scripts/evaluate_protocol.py``, so it runs in
under a second and never re-runs the model.

    python scripts/evaluate_calibration_robustness.py \
        --scores outputs/protocol/scores.json \
        --calibration outputs/calibration.json \
        --output-dir outputs/calibration_robustness
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.calibration import ProbabilityCalibrator  # noqa: E402
from src.evaluation.calibration_robustness import calibration_robustness  # noqa: E402
from src.evaluation.protocol import ScoredImage, split_records  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", default="outputs/protocol/scores.json")
    parser.add_argument("--calibration", default="outputs/calibration.json")
    parser.add_argument("--output-dir", default="outputs/calibration_robustness")
    parser.add_argument("--validation-fraction", type=float, default=0.4)
    parser.add_argument("--split-seed", type=int, default=1234)
    parser.add_argument(
        "--operating-point",
        default="balanced",
        help="Which fitted threshold to evaluate the accuracy column at",
    )
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--quiet", action="store_true")
    return parser


def _resolve(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    scores_path = _resolve(args.scores)
    if not scores_path.is_file():
        print(
            f"No cached scores at {scores_path}.\n"
            f"Produce them first:\n"
            f"  python scripts/evaluate_protocol.py --limit 120",
            file=sys.stderr,
        )
        return 1

    payload = json.loads(scores_path.read_text(encoding="utf-8"))
    records = [ScoredImage.from_dict(item) for item in payload["records"]]
    if not records:
        print(f"{scores_path} contains no records.", file=sys.stderr)
        return 1

    validation, test = split_records(
        records, validation_fraction=args.validation_fraction, seed=args.split_seed
    )
    if not test:
        print("The split produced an empty held-out set.", file=sys.stderr)
        return 1

    calibration_path = _resolve(args.calibration)
    calibrator = None
    threshold = 0.5
    if calibration_path.is_file():
        calibrator = ProbabilityCalibrator.load(calibration_path)
        selected = calibrator.selected_thresholds or {}
        threshold = float(
            selected.get(args.operating_point, selected.get("balanced", threshold))
        )
    elif not args.quiet:
        print(
            f"No calibration at {calibration_path}; reporting UNCALIBRATED scores.",
            file=sys.stderr,
        )

    report = calibration_robustness(
        test, calibrator, threshold, fitted_on=validation, bins=args.bins
    )
    report["source_scores"] = str(scores_path)
    report["calibration_path"] = str(calibration_path) if calibrator else None
    report["split"] = {
        "validation_images": len(validation),
        "held_out_images": len(test),
        "validation_fraction": args.validation_fraction,
        "seed": args.split_seed,
        "note": (
            "The calibrator was fitted on the clean scores of the validation split "
            "only; every number below is measured on the disjoint held-out split."
        ),
    }

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "calibration_robustness.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if not args.quiet:
        print(
            f"Held-out images: {report['held_out_images']}  "
            f"threshold: {report['threshold']}\n"
        )
        header = (
            f"{'condition':<18}{'ECE':>9}{'ECE(raw)':>10}{'Brier':>9}"
            f"{'acc':>8}{'AUC':>8}{'shift':>10}"
        )
        print(header)
        print("-" * len(header))
        for item in report["conditions"]:
            print(
                f"{item['condition']:<18}{item['expected_calibration_error']:>9.4f}"
                f"{item['expected_calibration_error_uncalibrated']:>10.4f}"
                f"{item['brier_score']:>9.4f}{item['accuracy']:>8.4f}"
                f"{item['auc']:>8.4f}{item['mean_score_shift']:>+10.5f}"
            )
        print(f"\n{report['statement']}")
        print(f"\nWrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
