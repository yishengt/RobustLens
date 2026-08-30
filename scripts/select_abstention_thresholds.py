#!/usr/bin/env python3
"""Fit abstention thresholds on the validation split, then check them held out.

Abstention is a trade: every image the system declines to judge is one fewer
chance to be wrong and one fewer answer delivered. The useful setting is the one
that removes more errors than answers, so this script sweeps each rule's
threshold on the VALIDATION split only, keeps the setting whose abstentions are
enriched in errors, and reports what that setting does on the held-out split
without ever having fitted to it.

Derived from the cached per-version scores, so it costs no forward passes.

    python scripts/select_abstention_thresholds.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.calibration import ProbabilityCalibrator  # noqa: E402
from src.evaluation.protocol import CLEAN_KEY, ScoredImage, split_records  # noqa: E402
from src.pipeline.abstention import evaluate_abstention  # noqa: E402
from src.pipeline.consistency import consistency_score  # noqa: E402
from src.pipeline.prediction import LABEL_AI, LABEL_AUTHENTIC  # noqa: E402

# The sweep grid. Fixed before any result is inspected.
BORDERLINE_GRID = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15]
DRIFT_GRID = [1.0, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10]
CONSISTENCY_GRID = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
AGREEMENT_GRID = [0.0, 0.40, 0.50, 0.60, 0.70, 0.80]
# Above 1.0 the crossing rule can never fire, which is the "off" setting.
CROSSING_GRID = [2.0, 0.40, 0.30, 0.20, 0.10]

# A setting is only worth keeping if abstaining removes errors faster than it
# removes answers. Pre-registered.
MIN_ERROR_ENRICHMENT = 1.5
MAX_ABSTENTION_RATE = 0.35


def _fused(record: ScoredImage, calibrator, weights=(0.7, 0.3)) -> Dict[str, Any]:
    """Reproduce the shipped fusion and consistency for one cached record."""

    names = [n for n in record.version_scores if n != CLEAN_KEY]
    raw_clean = record.version_scores[CLEAN_KEY]
    raw_transformed = [record.version_scores[n] for n in names]
    if calibrator is not None:
        clean = float(calibrator.transform(np.array([raw_clean]))[0])
        transformed = [float(v) for v in calibrator.transform(np.array(raw_transformed))]
    else:
        clean = float(raw_clean)
        transformed = [float(v) for v in raw_transformed]
    fused = weights[0] * clean + weights[1] * float(np.mean(transformed))
    every = [clean, *transformed]
    return {
        "clean": clean,
        "transformed": transformed,
        "fused": float(fused),
        "consistency": consistency_score(every),
        "label": record.binary_label,
    }


def _evaluate(
    rows: Sequence[Dict[str, Any]], threshold: float, settings: Dict[str, Any]
) -> Dict[str, Any]:
    """Abstention rate, and how concentrated the errors are among abstentions."""

    config = {"abstention": {**settings, "enabled": True}}
    abstained = 0
    abstained_wrong = 0
    answered = 0
    answered_wrong = 0
    for row in rows:
        predicted_ai = row["fused"] >= threshold
        correct = int(predicted_ai) == row["label"]
        agreement = float(
            np.mean([(v >= threshold) == predicted_ai for v in [row["clean"], *row["transformed"]]])
        )
        decision = evaluate_abstention(
            label=LABEL_AI if predicted_ai else LABEL_AUTHENTIC,
            final_probability=row["fused"],
            threshold=threshold,
            clean_probability=row["clean"],
            transformed_probabilities=row["transformed"],
            consistency_score=row["consistency"],
            agreement=agreement,
            patch_available=False,
            config=config,
        )
        if decision.abstain:
            abstained += 1
            abstained_wrong += 0 if correct else 1
        else:
            answered += 1
            answered_wrong += 0 if correct else 1

    total = len(rows)
    abstention_rate = abstained / total if total else 0.0
    base_error = sum(1 for r in rows if (r["fused"] >= threshold) != bool(r["label"])) / total
    abstained_error = abstained_wrong / abstained if abstained else 0.0
    answered_error = answered_wrong / answered if answered else 0.0
    return {
        "abstention_rate": abstention_rate,
        "error_rate_all": base_error,
        "error_rate_among_abstained": abstained_error,
        "error_rate_among_answered": answered_error,
        "accuracy_among_answered": 1.0 - answered_error,
        # How much more error-dense the abstained set is than the population.
        "error_enrichment": (abstained_error / base_error) if base_error else 0.0,
        "abstained": abstained,
        "answered": answered,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", default="outputs/protocol/scores.json")
    parser.add_argument("--calibration", default="outputs/calibration.json")
    parser.add_argument("--output-dir", default="outputs/abstention")
    parser.add_argument("--validation-fraction", type=float, default=0.4)
    parser.add_argument("--split-seed", type=int, default=1234)
    parser.add_argument("--operating-point", default="balanced")
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
            f"No cached scores at {scores_path}. Run scripts/evaluate_protocol.py first.",
            file=sys.stderr,
        )
        return 1

    records = [
        ScoredImage.from_dict(item)
        for item in json.loads(scores_path.read_text(encoding="utf-8"))["records"]
    ]
    validation, test = split_records(
        records, validation_fraction=args.validation_fraction, seed=args.split_seed
    )

    calibration_path = _resolve(args.calibration)
    calibrator = None
    threshold = 0.5
    if calibration_path.is_file():
        calibrator = ProbabilityCalibrator.load(calibration_path)
        selected = calibrator.selected_thresholds or {}
        threshold = float(selected.get(args.operating_point, selected.get("balanced", 0.5)))

    validation_rows = [_fused(r, calibrator) for r in validation]
    test_rows = [_fused(r, calibrator) for r in test]

    # Every rule starts at a setting it can never trigger at, so the baseline
    # really is "never abstain" and each accepted threshold is a measured gain
    # over answering everything.
    base = {
        "borderline_margin": 0.0,
        "max_transformed_drift": 1.0,
        "min_consistency": 0.0,
        "min_agreement": 0.0,
        "min_patch_coverage": 0.0,
        "boundary_crossing_fraction": 2.0,
        "abstain_on_evidence_errors": False,
    }

    sweeps = {
        "borderline_margin": BORDERLINE_GRID,
        "max_transformed_drift": DRIFT_GRID,
        "min_consistency": CONSISTENCY_GRID,
        "min_agreement": AGREEMENT_GRID,
        "boundary_crossing_fraction": CROSSING_GRID,
    }

    chosen = dict(base)
    trace: List[Dict[str, Any]] = []
    # One rule at a time, each on top of the rules already accepted. Sweeping
    # them jointly on 48 validation images would fit noise.
    for name, grid in sweeps.items():
        best_value = base[name]
        best_score = -1.0
        for value in grid:
            candidate = {**chosen, name: value}
            result = _evaluate(validation_rows, threshold, candidate)
            trace.append({"rule": name, "value": value, **result})
            if result["abstention_rate"] > MAX_ABSTENTION_RATE:
                continue
            if result["error_enrichment"] < MIN_ERROR_ENRICHMENT:
                continue
            # Prefer the setting that leaves the answered set most accurate,
            # breaking ties toward abstaining less.
            score = result["accuracy_among_answered"] - 0.05 * result["abstention_rate"]
            if score > best_score:
                best_score = score
                best_value = value
        chosen[name] = best_value

    validation_result = _evaluate(validation_rows, threshold, chosen)
    held_out_result = _evaluate(test_rows, threshold, chosen)
    no_abstention = _evaluate(test_rows, threshold, base)

    report = {
        "source_scores": str(scores_path),
        "threshold": threshold,
        "operating_point": args.operating_point,
        "split": {
            "validation_images": len(validation),
            "held_out_images": len(test),
            "note": "Thresholds were fitted on the validation split only.",
        },
        "decision_rule": {
            "min_error_enrichment": MIN_ERROR_ENRICHMENT,
            "max_abstention_rate": MAX_ABSTENTION_RATE,
            "note": (
                "A setting is kept only if abstained images are at least "
                f"{MIN_ERROR_ENRICHMENT}x more error-prone than the population and "
                f"it abstains on at most {MAX_ABSTENTION_RATE:.0%} of images."
            ),
        },
        "selected": chosen,
        "validation": validation_result,
        "held_out": held_out_result,
        "held_out_without_abstention": no_abstention,
        "sweep": trace,
    }

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "abstention_thresholds.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if not args.quiet:
        print(f"threshold {threshold}   validation {len(validation)}   held out {len(test)}\n")
        print("Selected on validation only:")
        for key, value in chosen.items():
            if key in sweeps:
                print(f"  {key:<24} {value}")
        for name, result in (
            ("validation", validation_result),
            ("held out", held_out_result),
            ("held out, no abstention", no_abstention),
        ):
            print(
                f"\n{name}:\n"
                f"  abstention rate         {result['abstention_rate']:.3f}\n"
                f"  accuracy among answered {result['accuracy_among_answered']:.3f}\n"
                f"  error rate, abstained   {result['error_rate_among_abstained']:.3f}\n"
                f"  error enrichment        {result['error_enrichment']:.2f}x"
            )
        print(f"\nWrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
