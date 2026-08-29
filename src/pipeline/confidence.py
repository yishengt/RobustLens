"""Stage 10: confidence scoring.

Confidence answers "how much should you trust this call?", which is a separate
question from "is this AI-generated?". It blends up to four signals:

* **decisiveness**    - how far the fused probability sits from the 0.5 fence;
* **agreement**       - the share of image versions that landed on the same side;
* **consistency**     - how tightly the per-version scores clustered;
* **patch agreement** - the share of patches agreeing with the whole image,
  included only when patch analysis ran.

The result is reported as High / Medium / Low. Wording throughout stays
probabilistic: this pipeline estimates likelihood and never claims proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from src.pipeline.prediction import LABEL_AI, LABEL_AUTHENTIC, LABEL_UNCERTAIN

CONFIDENCE_HIGH = "High"
CONFIDENCE_MEDIUM = "Medium"
CONFIDENCE_LOW = "Low"

DEFAULT_DECISIVENESS_WEIGHT = 0.35
DEFAULT_AGREEMENT_WEIGHT = 0.25
DEFAULT_CONSISTENCY_WEIGHT = 0.25
DEFAULT_PATCH_AGREEMENT_WEIGHT = 0.15
DEFAULT_HIGH_MIN = 0.70
DEFAULT_MEDIUM_MIN = 0.45

DISCLAIMER = (
    "This is a probabilistic estimate from a hackathon-scale model, not proof "
    "of how the image was made."
)


@dataclass(frozen=True)
class ConfidenceReport:
    """The confidence level plus the components behind it."""

    level: str
    score: float
    decisiveness: float
    agreement: float
    consistency: float
    statement: str
    weights: Dict[str, float] = field(default_factory=dict)
    patch_agreement: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "score": round(self.score, 6),
            "decisiveness": round(self.decisiveness, 6),
            "agreement": round(self.agreement, 6),
            "consistency": round(self.consistency, 6),
            "patch_agreement": (
                None if self.patch_agreement is None else round(self.patch_agreement, 6)
            ),
            "statement": self.statement,
            "weights": {key: round(value, 6) for key, value in self.weights.items()},
        }


def decisiveness(probability: float) -> float:
    """Distance from the undecided midpoint, rescaled to ``[0, 1]``."""

    return float(np.clip(abs(float(probability) - 0.5) * 2.0, 0.0, 1.0))


def compute_confidence(
    final_probability: float,
    agreement: float,
    consistency: float,
    config: Optional[Dict[str, Any]] = None,
    label: Optional[str] = None,
    patch_agreement: Optional[float] = None,
) -> ConfidenceReport:
    """Blend decisiveness, agreement and consistency into a confidence level.

    ``patch_agreement`` is the share of patches landing on the same side of the
    threshold as the whole image. When patch analysis did not run it is
    ``None``, its weight is dropped and the remaining weights are renormalised,
    so a missing signal never silently counts as disagreement.
    """

    settings = (config or {}).get("confidence", {}) or {}
    raw_weights = {
        "decisiveness": float(settings.get("decisiveness_weight", DEFAULT_DECISIVENESS_WEIGHT)),
        "agreement": float(settings.get("agreement_weight", DEFAULT_AGREEMENT_WEIGHT)),
        "consistency": float(settings.get("consistency_weight", DEFAULT_CONSISTENCY_WEIGHT)),
    }
    if patch_agreement is not None:
        raw_weights["patch_agreement"] = float(
            settings.get("patch_agreement_weight", DEFAULT_PATCH_AGREEMENT_WEIGHT)
        )
    total = sum(raw_weights.values())
    if total <= 0:
        raise ValueError(f"Confidence weights must sum to a positive value, got {raw_weights}")
    weights = {key: value / total for key, value in raw_weights.items()}

    high_min = float(settings.get("high_min", DEFAULT_HIGH_MIN))
    medium_min = float(settings.get("medium_min", DEFAULT_MEDIUM_MIN))
    if not 0.0 <= medium_min <= high_min <= 1.0:
        raise ValueError(
            "confidence.medium_min must be <= confidence.high_min and both within [0, 1]; "
            f"got medium_min={medium_min}, high_min={high_min}"
        )

    decisive = decisiveness(final_probability)
    agreement = float(np.clip(agreement, 0.0, 1.0))
    consistency = float(np.clip(consistency, 0.0, 1.0))

    score = float(
        weights["decisiveness"] * decisive
        + weights["agreement"] * agreement
        + weights["consistency"] * consistency
    )
    patch_value: Optional[float] = None
    if patch_agreement is not None and "patch_agreement" in weights:
        patch_value = float(np.clip(patch_agreement, 0.0, 1.0))
        score += weights["patch_agreement"] * patch_value

    if score >= high_min:
        level = CONFIDENCE_HIGH
    elif score >= medium_min:
        level = CONFIDENCE_MEDIUM
    else:
        level = CONFIDENCE_LOW

    # An "Uncertain" verdict must never be reported as high confidence.
    if label == LABEL_UNCERTAIN and level == CONFIDENCE_HIGH:
        level = CONFIDENCE_MEDIUM

    return ConfidenceReport(
        level=level,
        score=float(np.clip(score, 0.0, 1.0)),
        decisiveness=decisive,
        agreement=agreement,
        consistency=consistency,
        statement=describe(label or "", level),
        weights=weights,
        patch_agreement=patch_value,
    )


def describe(label: str, level: str) -> str:
    """Return the user-facing sentence for a label and confidence level."""

    level_text = level.lower()
    if label == LABEL_AI:
        verdict = "This image is likely AI-generated"
    elif label == LABEL_AUTHENTIC:
        verdict = "This image is likely authentic"
    elif label == LABEL_UNCERTAIN:
        verdict = "The evidence is mixed and the result is uncertain"
    else:
        verdict = "The result is inconclusive"
    return f"{verdict} ({level_text} confidence). {DISCLAIMER}"
