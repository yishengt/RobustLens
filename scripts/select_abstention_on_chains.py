#!/usr/bin/env python3
"""Fit abstention thresholds on transformation CHAINS, held out honestly.

The single-transformation sweep selected nothing: one JPEG or one resize barely
moves a score, so the drift and consistency rules never fired at any threshold.
Compound chains are where the failure actually lives -- scores walk downward as
operations stack -- so this refits the same rules on chain data.

The pre-registered bars are unchanged and are NOT negotiable here: a rule is
kept only if abstained images are at least 1.5x more error-prone than the
population, while abstaining on at most 35% of images. If nothing passes, that
is the result, and abstention stays off.

Images are split into a chain-validation and a chain-test half by a hash of the
image id, so thresholds are fitted on one half and reported on the other.

    python scripts/select_abstention_on_chains.py --scores outputs/chains_fit/chain_scores.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.calibration import ProbabilityCalibrator  # noqa: E402
from src.evaluation.chains import CLEAN_CHAIN, ChainRecord  # noqa: E402
from src.pipeline.abstention import evaluate_abstention  # noqa: E402
from src.pipeline.consistency import consistency_score  # noqa: E402
from src.pipeline.prediction import LABEL_AI, LABEL_AUTHENTIC  # noqa: E402

# Same grids and the same pre-registered bars as the single-transform sweep.
BORDERLINE_GRID = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15]
DRIFT_GRID = [1.0, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
CONSISTENCY_GRID = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
AGREEMENT_GRID = [0.0, 0.40, 0.50, 0.60, 0.70, 0.80]
CROSSING_GRID = [2.0, 0.40, 0.30, 0.20, 0.10]

MIN_ERROR_ENRICHMENT = 1.5
MAX_ABSTENTION_RATE = 0.35

BASE = {
    "borderline_margin": 0.0,
    "max_transformed_drift": 1.0,
    "min_consistency": 0.0,
    "min_agreement": 0.0,
    "min_patch_coverage": 0.0,
    "boundary_crossing_fraction": 2.0,
    "abstain_on_evidence_errors": False,
}


def _split(records: Sequence[ChainRecord], seed: int) -> Tuple[List, List]:
    """Stratified halves by a hash of the image id, so both keep both classes."""

    by_class: Dict[int, List[ChainRecord]] = {}
    for record in records:
        by_class.setdefault(record.binary_label, []).append(record)
    validation: List[ChainRecord] = []
    test: List[ChainRecord] = []
    for _label, group in sorted(by_class.items()):
        ranked = sorted(
            group,
            key=lambda r: hashlib.sha256(f"{seed}:{r.img_id}".encode("utf-8")).hexdigest(),
        )
        cut = max(1, min(len(ranked) - 1, len(ranked) // 2)) if len(ranked) >= 2 else len(ranked)
        validation.extend(ranked[:cut])
        test.extend(ranked[cut:])
    return validation, test


def _rows(records: Sequence[ChainRecord], calibrator) -> List[Dict[str, Any]]:
    """One row per image: clean score plus every chain's score, calibrated."""

    rows: List[Dict[str, Any]] = []
    for record in records:
        names = [n for n in record.chain_scores if n != CLEAN_CHAIN]
        raw_clean = record.chain_scores[CLEAN_CHAIN]
        raw_chain = [record.chain_scores[n] for n in names]
        if calibrator is not None:
            clean = float(calibrator.transform(np.array([raw_clean]))[0])
            chained = [float(v) for v in calibrator.transform(np.array(raw_chain))]
        else:
            clean = float(raw_clean)
            chained = [float(v) for v in raw_chain]
        every = [clean, *chained]
        rows.append(
            {
                "img_id": record.img_id,
                "clean": clean,
                "chained": chained,
                # The deployed score is the clean image's; the chains are the
                # evidence about how stable that score is.
                "fused": clean,
                "consistency": consistency_score(every),
                "label": record.binary_label,
            }
        )
    return rows


def _evaluate(rows: Sequence[Dict[str, Any]], threshold: float, settings: Dict[str, Any]) -> Dict:
    config = {"abstention": {**settings, "enabled": True}}
    abstained = abstained_wrong = answered = answered_wrong = 0
    for row in rows:
        predicted_ai = row["fused"] >= threshold
        correct = int(predicted_ai) == row["label"]
        every = [row["clean"], *row["chained"]]
        decision = evaluate_abstention(
            label=LABEL_AI if predicted_ai else LABEL_AUTHENTIC,
            final_probability=row["fused"],
            threshold=threshold,
            clean_probability=row["clean"],
            transformed_probabilities=row["chained"],
            consistency_score=row["consistency"],
            agreement=float(np.mean([(v >= threshold) == predicted_ai for v in every])),
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
    base_error = sum(1 for r in rows if (r["fused"] >= threshold) != bool(r["label"])) / total
    abstained_error = abstained_wrong / abstained if abstained else 0.0
    answered_error = answered_wrong / answered if answered else 0.0
    return {
        "abstention_rate": abstained / total,
        "error_rate_all": base_error,
        "error_rate_among_abstained": abstained_error,
        "accuracy_among_answered": 1.0 - answered_error,
        "error_enrichment": (abstained_error / base_error) if base_error else 0.0,
        "abstained": abstained,
        "answered": answered,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", default="outputs/chains_fit/chain_scores.json")
    parser.add_argument("--calibration", default="outputs/calibration.json")
    parser.add_argument("--output-dir", default="outputs/abstention_chains")
    parser.add_argument("--operating-point", default="balanced")
    parser.add_argument("--split-seed", type=int, default=1234)
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
            f"No chain scores at {scores_path}. Run:\n"
            f"  python scripts/evaluate_chains.py --limit 48 --output-dir outputs/chains_fit",
            file=sys.stderr,
        )
        return 1

    payload = json.loads(scores_path.read_text(encoding="utf-8"))
    records = [ChainRecord.from_dict(item) for item in payload["records"]]
    validation, test = _split(records, args.split_seed)

    calibration_path = _resolve(args.calibration)
    calibrator = None
    threshold = 0.5
    if calibration_path.is_file():
        calibrator = ProbabilityCalibrator.load(calibration_path)
        selected = calibrator.selected_thresholds or {}
        threshold = float(selected.get(args.operating_point, selected.get("balanced", 0.5)))

    validation_rows = _rows(validation, calibrator)
    test_rows = _rows(test, calibrator)

    sweeps = {
        "borderline_margin": BORDERLINE_GRID,
        "max_transformed_drift": DRIFT_GRID,
        "min_consistency": CONSISTENCY_GRID,
        "min_agreement": AGREEMENT_GRID,
        "boundary_crossing_fraction": CROSSING_GRID,
    }

    chosen = dict(BASE)
    trace: List[Dict[str, Any]] = []
    accepted: List[str] = []
    for name, grid in sweeps.items():
        best_value = BASE[name]
        best_score = -1.0
        for value in grid:
            result = _evaluate(validation_rows, threshold, {**chosen, name: value})
            trace.append({"rule": name, "value": value, **result})
            if result["abstention_rate"] > MAX_ABSTENTION_RATE:
                continue
            if result["error_enrichment"] < MIN_ERROR_ENRICHMENT:
                continue
            score = result["accuracy_among_answered"] - 0.05 * result["abstention_rate"]
            if score > best_score:
                best_score = score
                best_value = value
        if best_value != BASE[name]:
            accepted.append(name)
        chosen[name] = best_value

    validation_result = _evaluate(validation_rows, threshold, chosen)
    held_out = _evaluate(test_rows, threshold, chosen)
    held_out_baseline = _evaluate(test_rows, threshold, BASE)

    passed = bool(accepted)
    report = {
        "source_scores": str(scores_path),
        "threshold": threshold,
        "operating_point": args.operating_point,
        "split": {
            "chain_validation_images": len(validation),
            "chain_test_images": len(test),
            "seed": args.split_seed,
            "note": "Thresholds fitted on the chain-validation half only.",
        },
        "decision_rule": {
            "min_error_enrichment": MIN_ERROR_ENRICHMENT,
            "max_abstention_rate": MAX_ABSTENTION_RATE,
            "lowered": False,
        },
        "any_rule_passed": passed,
        "accepted_rules": accepted,
        "selected": chosen,
        "chain_validation": validation_result,
        "chain_held_out": held_out,
        "chain_held_out_without_abstention": held_out_baseline,
        "recommendation": (
            "Enable abstention with the selected thresholds."
            if passed
            else "Keep abstention DISABLED. No rule met the pre-registered bar on chain "
            "data either. The bars were not lowered."
        ),
        "sweep": trace,
    }

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "abstention_on_chains.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if not args.quiet:
        print(
            f"chain-validation {len(validation)}   chain-test {len(test)}   "
            f"threshold {threshold}\n"
        )
        print(f"{'rule':<28}{'value':>7}{'abst':>8}{'enrich':>8}{'acc_ans':>9}")
        print("-" * 60)
        for row in trace:
            print(
                f"{row['rule']:<28}{row['value']:>7}{row['abstention_rate']:>8.3f}"
                f"{row['error_enrichment']:>8.2f}{row['accuracy_among_answered']:>9.3f}"
            )
        print(f"\naccepted rules: {accepted or 'NONE'}")
        for name, result in (
            ("chain validation", validation_result),
            ("chain held out", held_out),
            ("chain held out, abstention off", held_out_baseline),
        ):
            print(
                f"\n{name}:\n"
                f"  abstention rate         {result['abstention_rate']:.3f}\n"
                f"  accuracy among answered {result['accuracy_among_answered']:.3f}\n"
                f"  error enrichment        {result['error_enrichment']:.2f}x"
            )
        print(f"\n{report['recommendation']}")
        print(f"\nWrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
