"""Track 5 evaluation protocol.

Design
------
The expensive part is scoring. This module runs **one** scoring pass per image
that records the whole-image score, all 14 transformed-version scores and the
patch evidence, then caches those raw numbers. Every downstream analysis --
per-transformation robustness, the four-way system ablation, confusion
matrices, confidence distributions, failure grids -- is derived from that one
cache, so re-analysis is instant and the four system variants are compared on
*identical* forward passes rather than separate runs.

Threshold discipline
--------------------
One threshold is selected on the **clean scores of a validation split only**,
then frozen and applied unchanged to every condition and every system variant.
It is never retuned per transformation. The validation and test splits are
disjoint and assigned deterministically by image id.

What may and may not be claimed
-------------------------------
* The detector was trained on OpenFake and is evaluated here on SID_Set, so a
  **dataset-source holdout** is genuine.
* SID_Set does not publish per-generator labels, so its ``full_synthetic`` and
  ``tampered`` classes are reported as *generation-process families*, which is
  a **proxy** for a generator holdout, not a true one. See
  :func:`generator_claim_statement`.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from src.evaluation.calibration import ThresholdSelection, search_thresholds
from src.evaluation.metrics import compute_metrics
from src.evaluation.sid_set import CLASS_NAMES
from src.pipeline.consistency import consistency_score
from src.pipeline.fusion import fuse_predictions
from src.pipeline.patches import analyse_patches
from src.pipeline.prediction import predict_images
from src.pipeline.transformations import ORIGINAL_KEY, build_transform_specs, generate_variants

CLEAN_KEY = "clean"

# The four systems compared in the ablation.
VARIANT_WHOLE = "whole_only"
VARIANT_WHOLE_TRANSFORM = "whole_plus_transformations"
VARIANT_WHOLE_PATCH = "whole_plus_patches"
VARIANT_FUSED = "fused_system"
VARIANTS = (VARIANT_WHOLE, VARIANT_WHOLE_TRANSFORM, VARIANT_WHOLE_PATCH, VARIANT_FUSED)

VARIANT_LABELS = {
    VARIANT_WHOLE: "Whole-image only",
    VARIANT_WHOLE_TRANSFORM: "Whole + transformations",
    VARIANT_WHOLE_PATCH: "Whole + patch analysis",
    VARIANT_FUSED: "Complete fused system",
}


@dataclass
class ScoredImage:
    """Every raw number one image produced, cached so analysis is repeatable."""

    img_id: str
    source_label: int
    class_name: str
    binary_label: int
    version_scores: Dict[str, float] = field(default_factory=dict)
    patch_evidence: Optional[float] = None
    patch_agreement: Optional[float] = None
    patch_available: bool = False
    num_patches: int = 0
    width: int = 0
    height: int = 0
    seconds: float = 0.0

    @property
    def clean_score(self) -> float:
        return float(self.version_scores[CLEAN_KEY])

    @property
    def transformed_scores(self) -> List[float]:
        return [float(value) for name, value in self.version_scores.items() if name != CLEAN_KEY]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "img_id": self.img_id,
            "source_label": self.source_label,
            "class_name": self.class_name,
            "binary_label": self.binary_label,
            "version_scores": {k: round(float(v), 6) for k, v in self.version_scores.items()},
            "patch_evidence": (
                None if self.patch_evidence is None else round(float(self.patch_evidence), 6)
            ),
            "patch_agreement": (
                None if self.patch_agreement is None else round(float(self.patch_agreement), 6)
            ),
            "patch_available": self.patch_available,
            "num_patches": self.num_patches,
            "width": self.width,
            "height": self.height,
            "seconds": round(float(self.seconds), 4),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ScoredImage":
        return cls(
            img_id=str(payload["img_id"]),
            source_label=int(payload["source_label"]),
            class_name=str(payload["class_name"]),
            binary_label=int(payload["binary_label"]),
            version_scores={str(k): float(v) for k, v in payload["version_scores"].items()},
            patch_evidence=payload.get("patch_evidence"),
            patch_agreement=payload.get("patch_agreement"),
            patch_available=bool(payload.get("patch_available", False)),
            num_patches=int(payload.get("num_patches", 0)),
            width=int(payload.get("width", 0)),
            height=int(payload.get("height", 0)),
            seconds=float(payload.get("seconds", 0.0)),
        )


# ---------------------------------------------------------------------------
# Scoring pass
# ---------------------------------------------------------------------------


def score_images(
    bundle: Any,
    preprocessor: Any,
    samples: Sequence[Any],
    config: Dict[str, Any],
    with_patches: bool = True,
    progress: Optional[Callable[[int, str, float], None]] = None,
) -> List[ScoredImage]:
    """Score every image once: clean, all transformed versions, and patches."""

    specs = build_transform_specs(config)
    records: List[ScoredImage] = []
    batch_size = int(config.get("inference", {}).get("batch_size", 8))

    for index, sample in enumerate(samples, start=1):
        started = time.time()
        image = sample.image.convert("RGB")
        variants, _ = generate_variants(image, config, specs)

        names = list(variants)
        scores = predict_images(
            bundle, [variants[name] for name in names], preprocessor, batch_size=batch_size
        )
        version_scores = {
            (CLEAN_KEY if name == ORIGINAL_KEY else name): float(score)
            for name, score in zip(names, scores)
        }

        patch_evidence = patch_agreement = None
        patch_available, num_patches = False, 0
        if with_patches:
            report = analyse_patches(
                bundle,
                image,
                preprocessor,
                config,
                whole_image_probability=version_scores[CLEAN_KEY],
            )
            patch_available = report.available
            if report.available:
                patch_evidence = float(report.evidence)
                patch_agreement = float(report.agreement)
                num_patches = len(report.patches)

        elapsed = time.time() - started
        records.append(
            ScoredImage(
                img_id=sample.img_id,
                source_label=int(sample.label),
                class_name=sample.class_name,
                binary_label=int(sample.binary_label),
                version_scores=version_scores,
                patch_evidence=patch_evidence,
                patch_agreement=patch_agreement,
                patch_available=patch_available,
                num_patches=num_patches,
                width=image.size[0],
                height=image.size[1],
                seconds=elapsed,
            )
        )
        if progress is not None:
            progress(index, sample.img_id, elapsed)
    return records


# ---------------------------------------------------------------------------
# Leakage-safe splitting and threshold selection
# ---------------------------------------------------------------------------


def split_records(
    records: Sequence[ScoredImage], validation_fraction: float = 0.4, seed: int = 1234
) -> tuple[List[ScoredImage], List[ScoredImage]]:
    """Split into (validation, test) deterministically and stratified by class.

    Hashing the id rather than shuffling keeps the assignment stable across
    runs and independent of ordering, so a threshold fitted today applies to
    the same test images tomorrow.

    The split is stratified by the binary label: each class is ranked by its
    own hash and cut at ``validation_fraction``. An unstratified hash split can
    hand the validation side a single class at small sample sizes, which makes
    threshold selection impossible; stratifying guarantees both classes land on
    both sides whenever the data contains both.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")

    validation: List[ScoredImage] = []
    test: List[ScoredImage] = []
    by_class: Dict[int, List[ScoredImage]] = {}
    for record in records:
        by_class.setdefault(record.binary_label, []).append(record)

    for _label, group in sorted(by_class.items()):
        ranked = sorted(
            group,
            key=lambda r: hashlib.sha256(f"{seed}:{r.img_id}".encode("utf-8")).hexdigest(),
        )
        # At least one image per class on each side whenever the class has >= 2.
        cut = int(round(len(ranked) * validation_fraction))
        if len(ranked) >= 2:
            cut = max(1, min(len(ranked) - 1, cut))
        validation.extend(ranked[:cut])
        test.extend(ranked[cut:])
    return validation, test


def select_fixed_threshold(
    validation: Sequence[ScoredImage], target_false_positive_rate: float = 0.05
) -> ThresholdSelection:
    """Select one threshold from the CLEAN scores of the validation split.

    Clean-only and validation-only, by design: fitting on transformed images
    would leak the very conditions the protocol is meant to measure.
    """

    labels = [record.binary_label for record in validation]
    scores = [record.clean_score for record in validation]
    if len(set(labels)) < 2:
        raise ValueError(
            "Threshold selection needs both authentic and AI-generated images in the "
            "validation split; increase --limit or lower --validation-fraction."
        )
    return search_thresholds(labels, scores, target_false_positive_rate)


# ---------------------------------------------------------------------------
# System variants -- all derived from the same cached scores
# ---------------------------------------------------------------------------


def variant_probability(
    record: ScoredImage, variant: str, config: Dict[str, Any]
) -> Optional[float]:
    """Fused probability for one system variant, via the production fusion code."""

    clean = record.clean_score
    if variant == VARIANT_WHOLE:
        return clean

    transformed = record.transformed_scores
    patch = record.patch_evidence if record.patch_available else None

    if variant == VARIANT_WHOLE_TRANSFORM:
        settings = {"fusion": {**config.get("fusion", {}), "mode": "rgb_transform"}}
        return float(fuse_predictions(clean, transformed, settings).final_probability)

    if variant == VARIANT_WHOLE_PATCH:
        if patch is None:
            return None  # nothing to fuse; excluded from this variant's metrics
        settings = {"fusion": {**config.get("fusion", {}), "mode": "whole_patch_transform"}}
        return float(fuse_predictions(clean, [], settings, patch_evidence=patch).final_probability)

    if variant == VARIANT_FUSED:
        settings = {"fusion": {**config.get("fusion", {}), "mode": "whole_patch_transform"}}
        return float(
            fuse_predictions(clean, transformed, settings, patch_evidence=patch).final_probability
        )

    raise ValueError(f"Unknown system variant '{variant}'. Valid: {', '.join(VARIANTS)}")


# ---------------------------------------------------------------------------
# Metric assembly
# ---------------------------------------------------------------------------


def _metrics_or_none(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> Optional[Dict[str, Any]]:
    if not labels or len(set(labels)) < 2:
        return None
    return compute_metrics(labels, scores, threshold).as_dict()


def per_transformation_metrics(
    records: Sequence[ScoredImage], threshold: float
) -> Dict[str, Dict[str, Any]]:
    """Whole-image metrics under each condition, at the one frozen threshold."""

    labels = [record.binary_label for record in records]
    names: List[str] = [CLEAN_KEY]
    names += [n for n in records[0].version_scores if n != CLEAN_KEY] if records else []

    results: Dict[str, Dict[str, Any]] = {}
    for name in names:
        scores = [record.version_scores.get(name) for record in records]
        if any(score is None for score in scores):
            continue
        computed = _metrics_or_none(labels, [float(s) for s in scores], threshold)
        if computed is not None:
            results[name] = computed
    return results


def robustness_summary(
    per_version: Dict[str, Dict[str, Any]], key: str = "accuracy"
) -> Dict[str, Any]:
    """Drop, ratio and worst case for one metric, relative to clean."""

    clean = per_version.get(CLEAN_KEY, {}).get(key)
    transformed = {n: m for n, m in per_version.items() if n != CLEAN_KEY}
    if clean is None or not transformed:
        return {"metric": key, "clean": clean, "transformed": {}}

    rows = []
    for name, metrics in transformed.items():
        value = metrics.get(key)
        if value is None:
            continue
        rows.append(
            {
                "transformation": name,
                key: round(float(value), 6),
                "robustness_drop": round(float(clean) - float(value), 6),
                "robustness_ratio": (round(float(value) / float(clean), 6) if clean else None),
            }
        )
    rows.sort(key=lambda row: row["robustness_drop"], reverse=True)
    values = [row[key] for row in rows]
    worst = rows[0] if rows else None
    return {
        "metric": key,
        "clean": round(float(clean), 6),
        "average_transformed": round(float(np.mean(values)), 6) if values else None,
        "worst_case": round(float(np.min(values)), 6) if values else None,
        "worst_transformation": worst["transformation"] if worst else None,
        "largest_drop": worst["robustness_drop"] if worst else None,
        "average_ratio": (
            round(float(np.mean([r["robustness_ratio"] for r in rows if r["robustness_ratio"]])), 6)
            if rows
            else None
        ),
        "per_transformation": rows,
    }


def variant_metrics(
    records: Sequence[ScoredImage], threshold: float, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Metrics for each of the four system variants on the same images."""

    results: Dict[str, Any] = {}
    for variant in VARIANTS:
        pairs = [
            (record.binary_label, variant_probability(record, variant, config))
            for record in records
        ]
        usable = [(label, score) for label, score in pairs if score is not None]
        skipped = len(pairs) - len(usable)
        if not usable:
            results[variant] = {
                "label": VARIANT_LABELS[variant],
                "metrics": None,
                "skipped": skipped,
            }
            continue
        labels = [label for label, _ in usable]
        scores = [float(score) for _, score in usable]
        results[variant] = {
            "label": VARIANT_LABELS[variant],
            "count": len(usable),
            "skipped_no_patches": skipped,
            "metrics": _metrics_or_none(labels, scores, threshold),
        }
    return results


def subgroup_metrics(
    records: Sequence[ScoredImage], threshold: float, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Fused-system metrics split by the dataset's own source classes.

    Each AI family is scored against the shared authentic pool, so every
    subgroup keeps both classes and AUROC stays defined.
    """

    authentic = [record for record in records if record.binary_label == 0]
    groups: Dict[str, Any] = {}
    for label, name in CLASS_NAMES.items():
        if label == 0:
            continue
        family = [record for record in records if record.source_label == label]
        if not family:
            continue
        subset = authentic + family
        labels = [record.binary_label for record in subset]
        scores = [variant_probability(record, VARIANT_FUSED, config) for record in subset]
        usable = [(a, b) for a, b in zip(labels, scores) if b is not None]
        groups[name] = {
            "ai_images": len(family),
            "authentic_images": len(authentic),
            "metrics": _metrics_or_none(
                [a for a, _ in usable], [float(b) for _, b in usable], threshold
            ),
            "mean_ai_probability": (
                round(
                    float(
                        np.mean(
                            [variant_probability(r, VARIANT_FUSED, config) or 0.0 for r in family]
                        )
                    ),
                    6,
                )
            ),
        }
    return groups


def generator_claim_statement() -> str:
    """The exact claim the available labels support -- and the one they do not."""

    return (
        "SID_Set does not publish per-generator labels, so this is a "
        "generation-process family split (fully synthetic vs locally tampered), "
        "not a true unseen-generator holdout. It shows whether performance "
        "transfers across generation processes within one dataset. A genuine "
        "unseen-generator claim requires a dataset with per-generator labels "
        "where the evaluated generator is excluded from model development."
    )


def validate_generator_generalisation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Refuse a true generator-holdout run when generator labels are absent.

    SID_Set's source classes describe broad generation processes, not generator
    identities. An explicit opt-in therefore requires both a generator label
    field and named holdout values; otherwise proceeding would fabricate the
    experiment the configuration claims to run.
    """

    section = ((config.get("evaluation", {}) or {}).get("generator_generalisation", {}) or {})
    if not bool(section.get("enabled", False)):
        return {
            "enabled": False,
            "status": "not_established",
            "reason": generator_claim_statement(),
        }
    label_field = section.get("generator_label_field")
    holdout = section.get("holdout_generators") or []
    if not label_field or not holdout:
        raise ValueError(
            "Generator generalisation cannot be enabled for SID_Set: it has no "
            "per-generator labels. Configure a dataset with "
            "evaluation.generator_generalisation.generator_label_field and one or "
            "more holdout_generators; generation-process classes are not a substitute."
        )
    return {
        "enabled": True,
        "status": "configured",
        "generator_label_field": str(label_field),
        "holdout_generators": list(holdout),
    }


def dataset_holdout_statement(training_dataset: Optional[str]) -> str:
    """Whether the dataset-source holdout is genuine, based on checkpoint metadata."""

    if not training_dataset:
        return (
            "The checkpoint records no training dataset, so it cannot be verified "
            "that the evaluation data was excluded from model development. No "
            "generalisation claim is made."
        )
    return (
        f"The checkpoint records training on '{training_dataset}'. This protocol "
        f"evaluates on SID_Set, a different dataset with different collection "
        f"procedures, so the dataset-source holdout is genuine and cross-dataset "
        f"generalisation may be claimed at this sample size. Per-generator "
        f"generalisation may not."
    )


def failure_examples(
    records: Sequence[ScoredImage], threshold: float, config: Dict[str, Any], top_k: int = 8
) -> Dict[str, List[Dict[str, Any]]]:
    """The most confident mistakes, which are the informative ones."""

    scored = []
    for record in records:
        probability = variant_probability(record, VARIANT_FUSED, config)
        if probability is None:
            continue
        scored.append((record, float(probability)))

    false_positives = [
        {
            "img_id": r.img_id,
            "class_name": r.class_name,
            "score": round(p, 6),
            "clean_score": round(r.clean_score, 6),
        }
        for r, p in scored
        if r.binary_label == 0 and p >= threshold
    ]
    false_negatives = [
        {
            "img_id": r.img_id,
            "class_name": r.class_name,
            "score": round(p, 6),
            "clean_score": round(r.clean_score, 6),
        }
        for r, p in scored
        if r.binary_label == 1 and p < threshold
    ]
    false_positives.sort(key=lambda row: row["score"], reverse=True)
    false_negatives.sort(key=lambda row: row["score"])
    return {
        "false_positives": false_positives[:top_k],
        "false_negatives": false_negatives[:top_k],
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
    }


def confidence_distributions(
    records: Sequence[ScoredImage], config: Dict[str, Any]
) -> Dict[str, Any]:
    """Fused-score distributions for authentic vs AI images."""

    output: Dict[str, Any] = {}
    for label, name in ((0, "authentic"), (1, "ai_generated")):
        values = [
            variant_probability(record, VARIANT_FUSED, config)
            for record in records
            if record.binary_label == label
        ]
        values = [float(v) for v in values if v is not None]
        if not values:
            continue
        array = np.asarray(values)
        output[name] = {
            "count": int(array.size),
            "mean": round(float(array.mean()), 6),
            "median": round(float(np.median(array)), 6),
            "std": round(float(array.std(ddof=0)), 6),
            "p05": round(float(np.percentile(array, 5)), 6),
            "p95": round(float(np.percentile(array, 95)), 6),
            "scores": [round(float(v), 6) for v in values],
        }
    return output


def runtime_summary(records: Sequence[ScoredImage]) -> Dict[str, Any]:
    """Wall-clock cost per image for the full scoring pass."""

    seconds = np.asarray([record.seconds for record in records], dtype=np.float64)
    versions = len(records[0].version_scores) if records else 0
    patches = int(np.mean([r.num_patches for r in records])) if records else 0
    return {
        "images": int(seconds.size),
        "versions_per_image": versions,
        "mean_patches_per_image": patches,
        "forward_passes_per_image": versions + patches,
        "seconds_per_image_mean": round(float(seconds.mean()), 4) if seconds.size else None,
        "seconds_per_image_median": round(float(np.median(seconds)), 4) if seconds.size else None,
        "seconds_per_image_min": round(float(seconds.min()), 4) if seconds.size else None,
        "seconds_per_image_max": round(float(seconds.max()), 4) if seconds.size else None,
        "total_seconds": round(float(seconds.sum()), 2) if seconds.size else None,
    }


def consistency_distribution(
    records: Sequence[ScoredImage], config: Dict[str, Any]
) -> Dict[str, Any]:
    """Transformation-consistency score distribution across the test split."""

    values = [consistency_score(list(record.version_scores.values()), config) for record in records]
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {}
    return {
        "mean": round(float(array.mean()), 6),
        "median": round(float(np.median(array)), 6),
        "min": round(float(array.min()), 6),
        "p05": round(float(np.percentile(array, 5)), 6),
    }
