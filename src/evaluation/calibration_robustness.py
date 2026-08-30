"""Does the calibration still hold once the image has been transformed?

``scripts/evaluate_confidence.py`` fits a calibrator on the CLEAN scores of the
validation split and reports its ECE on that same split. Those two numbers are
in-sample: they say the fit converged, not that the probabilities transfer. Two
questions were therefore never answered:

1. What is the calibration error on images the calibrator never saw?
2. What is it under the transformations the detector is meant to survive?

Both matter more than the in-sample number, because a probability that is only
calibrated on pristine held-in data is not a probability a moderator can act on.

Everything here is derived from the cached per-version scores produced by
``scripts/evaluate_protocol.py``, so answering these costs no forward passes and
every condition is compared on identical model outputs. The calibrator is
applied exactly as shipped -- never refitted per condition, which would hide the
very drift being measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.evaluation.calibration import ProbabilityCalibrator, calibration_summary
from src.evaluation.metrics import compute_metrics
from src.evaluation.protocol import CLEAN_KEY, ScoredImage

# A condition is flagged when its calibration is materially worse than clean.
# Fixed here rather than chosen after seeing results.
ECE_DEGRADATION_LIMIT = 0.05


@dataclass
class ConditionCalibration:
    """Calibration and accuracy for one image condition (clean or transformed)."""

    condition: str
    count: int
    expected_calibration_error: float
    expected_calibration_error_uncalibrated: float
    brier_score: float
    mean_confidence: float
    accuracy: float
    auc: float
    mean_score_shift: float
    reliability_bins: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        def r(value: float) -> float:
            return round(float(value), 6)

        return {
            "condition": self.condition,
            "count": self.count,
            "expected_calibration_error": r(self.expected_calibration_error),
            "expected_calibration_error_uncalibrated": r(
                self.expected_calibration_error_uncalibrated
            ),
            "brier_score": r(self.brier_score),
            "mean_confidence": r(self.mean_confidence),
            "accuracy": r(self.accuracy),
            "auc": r(self.auc),
            "mean_score_shift": r(self.mean_score_shift),
            "reliability_bins": self.reliability_bins,
        }


def condition_calibration(
    records: Sequence[ScoredImage],
    condition: str,
    calibrator: Optional[ProbabilityCalibrator],
    threshold: float,
    bins: int = 10,
) -> ConditionCalibration:
    """Measure calibration for one condition on the supplied records.

    ``records`` should be the held-out split. Passing the split the calibrator
    was fitted on produces an in-sample number, which is exactly the
    overstatement this module exists to expose.
    """

    if not records:
        raise ValueError("condition_calibration requires at least one record")
    missing = [r.img_id for r in records if condition not in r.version_scores]
    if missing:
        raise KeyError(
            f"{len(missing)} record(s) have no '{condition}' score, first: {missing[0]}"
        )

    labels = np.asarray([r.binary_label for r in records], dtype=np.int64)
    raw = np.asarray([r.version_scores[condition] for r in records], dtype=np.float64)
    clean = np.asarray([r.version_scores[CLEAN_KEY] for r in records], dtype=np.float64)
    calibrated = calibrator.transform(raw) if calibrator is not None else raw

    summary = calibration_summary(labels, calibrated, bins=bins)
    uncalibrated = calibration_summary(labels, raw, bins=bins)
    metrics = compute_metrics(labels, calibrated, threshold=threshold)

    return ConditionCalibration(
        condition=condition,
        count=len(records),
        expected_calibration_error=summary["expected_calibration_error"],
        expected_calibration_error_uncalibrated=uncalibrated["expected_calibration_error"],
        brier_score=summary["brier_score"],
        mean_confidence=summary["mean_confidence"],
        accuracy=metrics.accuracy,
        auc=metrics.auc,
        mean_score_shift=float(np.mean(raw - clean)),
        reliability_bins=summary["accuracy_by_confidence_bin"],
    )


def condition_names(records: Sequence[ScoredImage]) -> List[str]:
    """Every condition present in the cache, clean first."""

    if not records:
        return []
    names = [CLEAN_KEY] if CLEAN_KEY in records[0].version_scores else []
    names += [name for name in records[0].version_scores if name != CLEAN_KEY]
    return names


def calibration_robustness(
    held_out: Sequence[ScoredImage],
    calibrator: Optional[ProbabilityCalibrator],
    threshold: float,
    fitted_on: Sequence[ScoredImage] | None = None,
    bins: int = 10,
) -> Dict[str, Any]:
    """Calibration on held-out data, clean and under every cached transformation.

    ``fitted_on`` is optional and used only to report the in-sample number beside
    the held-out one, which is the comparison that shows how optimistic the
    in-sample figure is.
    """

    conditions = condition_names(held_out)
    if not conditions:
        raise ValueError("No conditions found in the supplied records")

    results = [
        condition_calibration(held_out, name, calibrator, threshold, bins=bins)
        for name in conditions
    ]
    by_name = {item.condition: item for item in results}
    clean = by_name.get(CLEAN_KEY)
    transformed = [item for item in results if item.condition != CLEAN_KEY]

    worst = (
        max(transformed, key=lambda item: item.expected_calibration_error)
        if transformed
        else None
    )
    degradation = (
        None
        if worst is None or clean is None
        else worst.expected_calibration_error - clean.expected_calibration_error
    )

    in_sample = None
    if fitted_on:
        in_sample = condition_calibration(
            fitted_on, CLEAN_KEY, calibrator, threshold, bins=bins
        ).expected_calibration_error

    payload: Dict[str, Any] = {
        "held_out_images": len(held_out),
        "threshold": round(float(threshold), 6),
        "calibrated": calibrator is not None,
        "ece_degradation_limit": ECE_DEGRADATION_LIMIT,
        "clean_held_out_ece": (
            None if clean is None else round(clean.expected_calibration_error, 6)
        ),
        "clean_in_sample_ece": None if in_sample is None else round(in_sample, 6),
        "in_sample_optimism": (
            None
            if in_sample is None or clean is None
            else round(clean.expected_calibration_error - in_sample, 6)
        ),
        "mean_transformed_ece": (
            None
            if not transformed
            else round(float(np.mean([i.expected_calibration_error for i in transformed])), 6)
        ),
        "worst_transformed_condition": None if worst is None else worst.condition,
        "worst_transformed_ece": (
            None if worst is None else round(worst.expected_calibration_error, 6)
        ),
        "ece_degradation": None if degradation is None else round(float(degradation), 6),
        "calibration_holds_under_transformation": (
            None if degradation is None else bool(degradation <= ECE_DEGRADATION_LIMIT)
        ),
        "conditions": [item.as_dict() for item in results],
    }
    payload["statement"] = _statement(payload)
    return payload


def _statement(payload: Dict[str, Any]) -> str:
    """One sentence a reader can quote, with no claim the data does not support."""

    if not payload["calibrated"]:
        return (
            "No calibrator was applied, so these are uncalibrated model scores and the "
            "reliability figures describe ranking only, not probabilities."
        )
    clean = payload["clean_held_out_ece"]
    worst = payload["worst_transformed_ece"]
    if clean is None or worst is None:
        return "Not enough conditions were cached to judge calibration robustness."
    holds = payload["calibration_holds_under_transformation"]
    verdict = (
        "calibration transfers to transformed images within the pre-registered "
        f"{ECE_DEGRADATION_LIMIT:.2f} ECE limit"
        if holds
        else "calibration degrades beyond the pre-registered "
        f"{ECE_DEGRADATION_LIMIT:.2f} ECE limit under transformation"
    )
    optimism = payload["in_sample_optimism"]
    tail = ""
    if optimism is not None:
        tail = (
            f" The in-sample ECE on the fitting split is {payload['clean_in_sample_ece']:.4f}, "
            f"{optimism:+.4f} away from the held-out value, so the in-sample figure "
            f"{'overstates' if optimism > 0 else 'understates'} calibration quality."
        )
    return (
        f"On held-out images the calibrated ECE is {clean:.4f} clean and at worst "
        f"{worst:.4f} ({payload['worst_transformed_condition']}), so {verdict}.{tail}"
    )
