"""Benchmark the inference pipeline against a labelled dataset.

Two sets of numbers come out of a run:

* **Detection quality** - accuracy, AUC, precision/recall/F1 for the fused
  prediction, plus a breakdown by original dataset class.
* **Robustness** - the same metrics recomputed for every transformed version of
  every image, which shows how much each real-world transformation costs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from src.evaluation.metrics import compute_metrics
from src.evaluation.sid_set import CLASS_NAMES, LabelledImage
from src.pipeline.consistency import ConsistencyReport
from src.pipeline.pipeline import DetectionPipeline
from src.pipeline.transformations import ORIGINAL_KEY


@dataclass
class BenchmarkReport:
    """Aggregated results for one benchmark run."""

    count: int = 0
    failed: int = 0
    threshold: float = 0.5
    fused: Dict[str, Any] = field(default_factory=dict)
    original_only: Dict[str, Any] = field(default_factory=dict)
    per_version: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    per_class: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    consistency: Dict[str, float] = field(default_factory=dict)
    confidence_levels: Dict[str, int] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    dataset: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "failed": self.failed,
            "threshold": self.threshold,
            "model": self.model,
            "dataset": self.dataset,
            "fused": self.fused,
            "original_only": self.original_only,
            "per_version": self.per_version,
            "per_class": self.per_class,
            "consistency": self.consistency,
            "confidence_levels": self.confidence_levels,
            "errors": self.errors[:50],
        }


def _robustness_table(per_version: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort versions by accuracy drop against the original image."""

    baseline = per_version.get(ORIGINAL_KEY, {}).get("accuracy")
    rows = []
    for name, metrics in per_version.items():
        if name == ORIGINAL_KEY:
            continue
        accuracy = metrics.get("accuracy")
        rows.append(
            {
                "transformation": name,
                "accuracy": accuracy,
                "auc": metrics.get("auc"),
                "accuracy_drop": (
                    None if baseline is None or accuracy is None else round(baseline - accuracy, 6)
                ),
            }
        )
    return sorted(
        rows, key=lambda row: (row["accuracy_drop"] is None, -(row["accuracy_drop"] or 0))
    )


def run_benchmark(
    pipeline: DetectionPipeline,
    samples: Sequence[LabelledImage] | Any,
    threshold: float = 0.5,
    progress: Optional[Callable[[int, str], None]] = None,
) -> BenchmarkReport:
    """Run the pipeline over labelled samples and aggregate every metric."""

    report = BenchmarkReport(threshold=float(threshold))
    report.model = pipeline.bundle.summary()

    binary_labels: List[int] = []
    source_labels: List[int] = []
    fused_scores: List[float] = []
    original_scores: List[float] = []
    version_scores: Dict[str, List[float]] = defaultdict(list)
    version_labels: Dict[str, List[int]] = defaultdict(list)
    consistency_scores: List[float] = []
    agreement_scores: List[float] = []
    confidence_counts: Dict[str, int] = defaultdict(int)

    for index, sample in enumerate(samples, start=1):
        try:
            result = pipeline.analyse_image(sample.image, metadata=None)
        except (RuntimeError, ValueError, OSError, MemoryError) as exc:
            report.failed += 1
            report.errors.append({"img_id": sample.img_id, "error": f"{type(exc).__name__}: {exc}"})
            continue

        binary_labels.append(int(sample.binary_label))
        source_labels.append(int(sample.label))
        fused_scores.append(float(result.ai_probability))

        for prediction in result.predictions:
            version_scores[prediction.name].append(float(prediction.ai_probability))
            version_labels[prediction.name].append(int(sample.binary_label))
            if prediction.is_original:
                original_scores.append(float(prediction.ai_probability))

        consistency: ConsistencyReport = result.consistency
        consistency_scores.append(consistency.consistency_score)
        agreement_scores.append(consistency.agreement)
        confidence_counts[result.confidence.level] += 1
        report.count += 1

        if progress is not None:
            progress(index, sample.img_id)

    if report.count == 0:
        raise ValueError("No samples were successfully evaluated")

    report.fused = compute_metrics(binary_labels, fused_scores, threshold).as_dict()
    if original_scores:
        report.original_only = compute_metrics(binary_labels, original_scores, threshold).as_dict()

    for name in version_scores:
        metrics = compute_metrics(version_labels[name], version_scores[name], threshold).as_dict()
        values = np.asarray(version_scores[name], dtype=np.float64)
        metrics["average_ai_probability"] = round(float(values.mean()), 6)
        metrics["probability_std"] = round(float(values.std(ddof=0)), 6)
        report.per_version[name] = metrics

    # Accuracy broken down by the dataset's own three classes.
    scores = np.asarray(fused_scores, dtype=np.float64)
    sources = np.asarray(source_labels, dtype=np.int64)
    truths = np.asarray(binary_labels, dtype=np.int64)
    for label, name in CLASS_NAMES.items():
        mask = sources == label
        if not mask.any():
            continue
        predicted = (scores[mask] >= threshold).astype(np.int64)
        report.per_class[name] = {
            "count": int(mask.sum()),
            "accuracy": round(float((predicted == truths[mask]).mean()), 6),
            "mean_ai_probability": round(float(scores[mask].mean()), 6),
        }

    report.consistency = {
        "mean_consistency_score": round(float(np.mean(consistency_scores)), 6),
        "min_consistency_score": round(float(np.min(consistency_scores)), 6),
        "mean_agreement": round(float(np.mean(agreement_scores)), 6),
    }
    report.confidence_levels = dict(confidence_counts)
    report.dataset["robustness_ranking"] = _robustness_table(report.per_version)
    return report


def format_report(report: BenchmarkReport) -> str:
    """Render a benchmark report as a readable console summary."""

    lines: List[str] = []
    fused = report.fused
    lines.append("=" * 68)
    lines.append("BENCHMARK RESULTS")
    lines.append("=" * 68)
    model = report.model
    lines.append(
        f"Model:    {model.get('architecture')} "
        f"({model.get('parameters_millions')} M params) on {model.get('device')}"
    )
    dataset = report.dataset
    if dataset.get("name"):
        lines.append(f"Dataset:  {dataset['name']}  ({report.count} images)")
    if report.failed:
        lines.append(f"Failed:   {report.failed} image(s)")
    lines.append("")

    lines.append(f"Fused prediction (threshold {report.threshold}):")
    auc = fused.get("auc")
    lines.append(f"  accuracy   {fused.get('accuracy', 0):.4f}")
    lines.append(f"  AUC        {auc:.4f}" if auc is not None else "  AUC        n/a")
    lines.append(f"  precision  {fused.get('precision', 0):.4f}")
    lines.append(f"  recall     {fused.get('recall', 0):.4f}")
    lines.append(f"  F1         {fused.get('f1', 0):.4f}")
    matrix = fused.get("confusion_matrix", {})
    lines.append(
        f"  confusion  TP={matrix.get('true_positives')} FP={matrix.get('false_positives')} "
        f"TN={matrix.get('true_negatives')} FN={matrix.get('false_negatives')}"
    )

    if report.original_only:
        original_auc = report.original_only.get("auc")
        lines.append("")
        lines.append("Original image only (no fusion):")
        lines.append(f"  accuracy   {report.original_only.get('accuracy', 0):.4f}")
        lines.append(
            f"  AUC        {original_auc:.4f}" if original_auc is not None else "  AUC        n/a"
        )

    if report.per_class:
        lines.append("")
        lines.append("By dataset class:")
        for name, stats in report.per_class.items():
            lines.append(
                f"  {name:16} n={stats['count']:5d}  accuracy={stats['accuracy']:.4f}  "
                f"mean p(AI)={stats['mean_ai_probability']:.4f}"
            )

    ranking = report.dataset.get("robustness_ranking") or []
    if ranking:
        lines.append("")
        lines.append("Robustness by transformation (largest accuracy drop first):")
        lines.append(f"  {'transformation':<20} {'accuracy':>9} {'AUC':>8} {'drop':>8}")
        for row in ranking:
            auc_text = f"{row['auc']:.4f}" if row.get("auc") is not None else "     n/a"
            drop_text = (
                f"{row['accuracy_drop']:+.4f}" if row.get("accuracy_drop") is not None else "  n/a"
            )
            lines.append(
                f"  {row['transformation']:<20} {row['accuracy']:>9.4f} {auc_text:>8} {drop_text:>8}"
            )

    if report.consistency:
        lines.append("")
        lines.append(
            f"Transformation consistency: mean "
            f"{report.consistency['mean_consistency_score']:.4f}, "
            f"min {report.consistency['min_consistency_score']:.4f}, "
            f"mean agreement {report.consistency['mean_agreement']:.4f}"
        )
    if report.confidence_levels:
        levels = ", ".join(f"{k}={v}" for k, v in sorted(report.confidence_levels.items()))
        lines.append(f"Confidence levels: {levels}")
    lines.append("=" * 68)
    return "\n".join(lines)
