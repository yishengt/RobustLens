"""Stage 7: transformation consistency.

A detector that is genuinely reading generation artefacts should keep saying
roughly the same thing after the image is compressed, blurred or resized. A
detector that latches onto fragile cues swings wildly. This module turns that
spread into a single 0-1 score, where 1.0 means "the prediction barely moved".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

from src.pipeline.prediction import Prediction, binary_threshold

DEFAULT_STD_REFERENCE = 0.20
DEFAULT_RANGE_REFERENCE = 0.60
DEFAULT_STD_WEIGHT = 0.5
DEFAULT_RANGE_WEIGHT = 0.5


@dataclass(frozen=True)
class ConsistencyReport:
    """Spread statistics across all image versions."""

    mean: float
    minimum: float
    maximum: float
    std: float
    score_range: float
    consistency_score: float
    agreement: float
    num_versions: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mean_ai_probability": round(self.mean, 6),
            "min_score": round(self.minimum, 6),
            "max_score": round(self.maximum, 6),
            "std": round(self.std, 6),
            "score_range": round(self.score_range, 6),
            "consistency_score": round(self.consistency_score, 6),
            "agreement": round(self.agreement, 6),
            "num_versions": self.num_versions,
        }


def _settings(config: Optional[Dict[str, Any]]) -> Dict[str, float]:
    section = (config or {}).get("consistency", {}) or {}
    std_reference = float(section.get("std_reference", DEFAULT_STD_REFERENCE))
    range_reference = float(section.get("range_reference", DEFAULT_RANGE_REFERENCE))
    if std_reference <= 0 or range_reference <= 0:
        raise ValueError("consistency.std_reference and range_reference must be positive")
    return {
        "std_reference": std_reference,
        "range_reference": range_reference,
        "std_weight": float(section.get("std_weight", DEFAULT_STD_WEIGHT)),
        "range_weight": float(section.get("range_weight", DEFAULT_RANGE_WEIGHT)),
    }


def consistency_score(
    probabilities: Sequence[float], config: Optional[Dict[str, Any]] = None
) -> float:
    """Return a 0-1 stability score for a set of per-version probabilities.

    Both the standard deviation and the min-max range contribute, each divided
    by a configurable reference value that represents "as unstable as we care
    to measure". A single version is trivially consistent and scores 1.0.
    """

    values = np.asarray(list(probabilities), dtype=np.float64)
    if values.size <= 1:
        return 1.0

    settings = _settings(config)
    std = float(values.std(ddof=0))
    spread = float(values.max() - values.min())

    weight_sum = settings["std_weight"] + settings["range_weight"]
    if weight_sum <= 0:
        raise ValueError("consistency std_weight + range_weight must be greater than zero")

    instability = (
        settings["std_weight"] * (std / settings["std_reference"])
        + settings["range_weight"] * (spread / settings["range_reference"])
    ) / weight_sum
    return float(np.clip(1.0 - instability, 0.0, 1.0))


def compute_consistency(
    predictions: Sequence[Prediction], config: Optional[Dict[str, Any]] = None
) -> ConsistencyReport:
    """Summarise the spread of AI probabilities across all image versions."""

    if not predictions:
        raise ValueError("Consistency requires at least one prediction")

    values = np.asarray([item.ai_probability for item in predictions], dtype=np.float64)
    threshold = binary_threshold(config)

    reference = next((item for item in predictions if item.is_original), predictions[0])
    reference_is_ai = reference.ai_probability >= threshold
    agreement = float(np.mean([(value >= threshold) == reference_is_ai for value in values]))

    return ConsistencyReport(
        mean=float(values.mean()),
        minimum=float(values.min()),
        maximum=float(values.max()),
        std=float(values.std(ddof=0)),
        score_range=float(values.max() - values.min()),
        consistency_score=consistency_score(values, config),
        agreement=agreement,
        num_versions=int(values.size),
    )


SEVERITY_LOW = "low"
SEVERITY_MEDIUM = "medium"
SEVERITY_HIGH = "high"


def estimate_manipulation_severity(
    report: ConsistencyReport, config: Optional[Dict[str, Any]] = None
) -> str:
    """Describe how sensitive this image's score was to transformation.

    This is a statement about *observable transformation sensitivity*, nothing
    more. A score that moves a lot under compression and blur is fragile
    evidence; that is what "high" means here. It is explicitly **not** a claim
    about how many times an image was edited, re-uploaded or passed through a
    platform - that history is not recoverable from a single image.
    """

    settings = (config or {}).get("consistency", {}) or {}
    medium_at = float(settings.get("severity_medium_max", 0.85))
    low_at = float(settings.get("severity_low_min", 0.95))
    score = float(report.consistency_score)
    if score >= low_at:
        return SEVERITY_LOW
    if score >= medium_at:
        return SEVERITY_MEDIUM
    return SEVERITY_HIGH
