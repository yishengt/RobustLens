"""Task 1: does the patch-agreement term make confidence more reliable?

Confidence does not change the predicted probability, so accuracy, F1 and
AUROC are identical across the variants below -- reporting them is the point,
because it demonstrates the term is not leaking into probability scoring.

What confidence *should* do is separate correct predictions from incorrect
ones. That is what this module measures: the AUROC of confidence against
correctness, the gap in mean confidence between right and wrong calls, and
whether the High/Medium/Low bands are ordered by actual accuracy.

The decision rule below is fixed before any result is inspected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.evaluation.metrics import compute_metrics, roc_auc
from src.evaluation.protocol import CLEAN_KEY, ScoredImage
from src.pipeline.confidence import compute_confidence
from src.pipeline.consistency import compute_consistency
from src.pipeline.fusion import fuse_predictions
from src.pipeline.prediction import LABEL_UNCERTAIN, Prediction, label_for_probability

VARIANT_WITHOUT = "without_patch_agreement"
VARIANT_WITH = "with_patch_agreement"
VARIANT_GATED = "patch_agreement_when_reliable"
CONFIDENCE_VARIANTS = (VARIANT_WITHOUT, VARIANT_WITH, VARIANT_GATED)

VARIANT_LABELS = {
    VARIANT_WITHOUT: "A. no patch agreement",
    VARIANT_WITH: "B. patch agreement always",
    VARIANT_GATED: "C. patch agreement when reliable",
}

# Variant C treats patch agreement as unreliable when the patch evidence
# disagrees sharply with the whole-image score. That is the pathology already
# measured in outputs/patch_ablation: on authentic images patch evidence
# averaged 0.297 against a whole-image score of 0.081.
DEFAULT_RELIABILITY_MARGIN = 0.25

# ---------------------------------------------------------------------------
# DECISION RULE -- fixed before results are examined.
# Patch agreement is retained in confidence scoring only if it makes confidence
# better at telling correct predictions from incorrect ones: the AUROC of
# confidence against correctness must improve by more than MIN_AUROC_GAIN, and
# the mean-confidence gap between correct and incorrect calls must not shrink.
# ---------------------------------------------------------------------------
MIN_AUROC_GAIN = 0.01

# The weight under test. The shipped config sets patch_agreement_weight to 0
# because this ablation rejected the term, so the experiment must supply its
# own weight -- otherwise re-running it would silently compare three identical
# variants and could never overturn the decision.
PATCH_AGREEMENT_WEIGHT_UNDER_TEST = 0.15


@dataclass
class ConfidenceRow:
    """One image evaluated under one confidence variant."""

    img_id: str
    binary_label: int
    probability: float
    predicted: int
    correct: int
    label: str
    confidence_level: str
    confidence_score: float
    patch_agreement_used: Optional[float]


def _predictions_from(record: ScoredImage, config: Dict[str, Any]) -> List[Prediction]:
    """Rebuild Prediction objects from cached scores."""

    predictions = []
    for name, score in record.version_scores.items():
        predictions.append(
            Prediction(
                name=("original" if name == CLEAN_KEY else name),
                ai_probability=float(score),
                real_probability=1.0 - float(score),
                label=label_for_probability(float(score), config),
                is_original=(name == CLEAN_KEY),
            )
        )
    predictions.sort(key=lambda item: not item.is_original)
    return predictions


def patch_agreement_for(
    record: ScoredImage, variant: str, margin: float = DEFAULT_RELIABILITY_MARGIN
) -> Optional[float]:
    """The patch-agreement value each variant feeds into confidence."""

    if variant == VARIANT_WITHOUT:
        return None
    if not record.patch_available or record.patch_agreement is None:
        return None
    if variant == VARIANT_WITH:
        return float(record.patch_agreement)
    if variant == VARIANT_GATED:
        if record.patch_evidence is None:
            return None
        if abs(float(record.patch_evidence) - record.clean_score) > float(margin):
            return None  # patch evidence contradicts the whole image; distrust it
        return float(record.patch_agreement)
    raise ValueError(f"Unknown confidence variant '{variant}'")


def evaluate_variant(
    records: Sequence[ScoredImage],
    variant: str,
    config: Dict[str, Any],
    threshold: float,
    margin: float = DEFAULT_RELIABILITY_MARGIN,
    patch_weight: float = PATCH_AGREEMENT_WEIGHT_UNDER_TEST,
) -> Dict[str, Any]:
    """Score every image under one confidence variant and summarise."""

    # Variants B and C must actually apply a weight, whatever the shipped
    # config says; variant A must not.
    config = dict(config)
    confidence_settings = dict(config.get("confidence", {}) or {})
    confidence_settings["patch_agreement_weight"] = (
        0.0 if variant == VARIANT_WITHOUT else float(patch_weight)
    )
    config["confidence"] = confidence_settings

    rows: List[ConfidenceRow] = []
    for record in records:
        predictions = _predictions_from(record, config)
        consistency = compute_consistency(predictions, config)
        fusion = fuse_predictions(
            record.clean_score,
            [p.ai_probability for p in predictions if not p.is_original],
            config,
        )
        probability = float(fusion.final_probability)
        label = label_for_probability(probability, config)
        agreement = patch_agreement_for(record, variant, margin)
        confidence = compute_confidence(
            probability,
            consistency.agreement,
            consistency.consistency_score,
            config,
            label=label,
            patch_agreement=agreement,
        )
        predicted = int(probability >= threshold)
        rows.append(
            ConfidenceRow(
                img_id=record.img_id,
                binary_label=record.binary_label,
                probability=probability,
                predicted=predicted,
                correct=int(predicted == record.binary_label),
                label=label,
                confidence_level=confidence.level,
                confidence_score=float(confidence.score),
                patch_agreement_used=agreement,
            )
        )

    labels = [row.binary_label for row in rows]
    scores = [row.probability for row in rows]
    correct = np.array([row.correct for row in rows])
    confidence_scores = np.array([row.confidence_score for row in rows])

    # The headline question: does confidence rank correct calls above wrong ones?
    confidence_auroc = roc_auc(correct.tolist(), confidence_scores.tolist())
    mean_correct = float(confidence_scores[correct == 1].mean()) if (correct == 1).any() else None
    mean_incorrect = float(confidence_scores[correct == 0].mean()) if (correct == 0).any() else None
    gap = None if mean_correct is None or mean_incorrect is None else mean_correct - mean_incorrect

    by_level: Dict[str, Any] = {}
    for level in ("High", "Medium", "Low"):
        subset = [row for row in rows if row.confidence_level == level]
        if subset:
            by_level[level] = {
                "count": len(subset),
                "accuracy": round(float(np.mean([row.correct for row in subset])), 6),
                "mean_confidence": round(
                    float(np.mean([row.confidence_score for row in subset])), 6
                ),
            }

    # Confidence should behave like P(correct); measure that directly.
    ece = _expected_calibration_error(confidence_scores, correct)

    return {
        "variant": variant,
        "label": VARIANT_LABELS[variant],
        "images": len(rows),
        "patch_agreement_applied": sum(1 for row in rows if row.patch_agreement_used is not None),
        "probability_metrics": compute_metrics(labels, scores, threshold).as_dict(),
        "confidence_reliability": {
            "auroc_confidence_vs_correctness": (
                None if not np.isfinite(confidence_auroc) else round(float(confidence_auroc), 6)
            ),
            "mean_confidence_when_correct": (
                None if mean_correct is None else round(mean_correct, 6)
            ),
            "mean_confidence_when_incorrect": (
                None if mean_incorrect is None else round(mean_incorrect, 6)
            ),
            "confidence_gap": None if gap is None else round(gap, 6),
            "point_biserial_correlation": round(
                float(np.corrcoef(confidence_scores, correct)[0, 1]), 6
            )
            if len(set(correct.tolist())) > 1
            else None,
            "expected_calibration_error": round(ece, 6),
            "accuracy_by_confidence_level": by_level,
        },
        "uncertain_rate": round(float(np.mean([row.label == LABEL_UNCERTAIN for row in rows])), 6),
        "rows": [row.__dict__ for row in rows],
    }


def _expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, bins: int = 10
) -> float:
    """ECE treating the confidence score as an estimate of P(correct)."""

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        upper_inclusive = index == bins - 1
        mask = (confidence >= edges[index]) & (
            (confidence <= edges[index + 1]) if upper_inclusive else (confidence < edges[index + 1])
        )
        if not mask.any():
            continue
        total += mask.mean() * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(total)


def decide(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Apply the pre-registered rule to the measured results."""

    baseline = results.get(VARIANT_WITHOUT)
    if baseline is None:
        return {"conclusion": "inconclusive", "reason": "variant A was not evaluated"}

    base_auroc = baseline["confidence_reliability"]["auroc_confidence_vs_correctness"]
    base_gap = baseline["confidence_reliability"]["confidence_gap"]
    if base_auroc is None or base_gap is None:
        return {
            "conclusion": "inconclusive",
            "reason": "confidence reliability could not be measured (one class only)",
        }

    improved = []
    for variant in (VARIANT_WITH, VARIANT_GATED):
        payload = results.get(variant)
        if not payload:
            continue
        reliability = payload["confidence_reliability"]
        auroc = reliability["auroc_confidence_vs_correctness"]
        gap = reliability["confidence_gap"]
        if auroc is None or gap is None:
            continue
        if auroc - base_auroc > MIN_AUROC_GAIN and gap >= base_gap:
            improved.append(
                {
                    "variant": variant,
                    "auroc_gain": round(auroc - base_auroc, 6),
                    "gap_change": round(gap - base_gap, 6),
                }
            )

    if improved:
        return {
            "conclusion": "retain_patch_agreement",
            "rule": f"AUROC(confidence vs correctness) gain > {MIN_AUROC_GAIN} and no shrink in the confidence gap",
            "variants_that_passed": improved,
            "reason": "patch agreement measurably improved confidence reliability",
        }
    return {
        "conclusion": "remove_patch_agreement",
        "rule": f"AUROC(confidence vs correctness) gain > {MIN_AUROC_GAIN} and no shrink in the confidence gap",
        "variants_that_passed": [],
        "reason": (
            "patch agreement did not improve confidence reliability by the pre-registered "
            "margin, so it should be removed from confidence scoring. Set "
            "confidence.patch_agreement_weight to 0."
        ),
    }
