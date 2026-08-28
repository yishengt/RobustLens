"""Classification metrics implemented with NumPy only.

scikit-learn is not a project dependency, so the handful of metrics the
benchmark needs are implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    """Standard binary-classification metrics at one decision threshold."""

    count: int
    positives: int
    negatives: int
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    average_precision: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "positives": self.positives,
            "negatives": self.negatives,
            "threshold": round(self.threshold, 6),
            "accuracy": round(self.accuracy, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "auc": round(self.auc, 6) if np.isfinite(self.auc) else None,
            "average_precision": (
                round(self.average_precision, 6)
                if np.isfinite(self.average_precision)
                else None
            ),
            "confusion_matrix": {
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "true_negatives": self.true_negatives,
                "false_negatives": self.false_negatives,
            },
        }


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney U) identity.

    Ties receive averaged ranks. Returns NaN when only one class is present,
    since AUC is undefined there.
    """

    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_scores = s[order]
    index = 0
    while index < len(sorted_scores):
        end = index
        while end + 1 < len(sorted_scores) and sorted_scores[end + 1] == sorted_scores[index]:
            end += 1
        # Ranks are 1-based; tied entries share their average rank.
        ranks[order[index : end + 1]] = (index + end) / 2.0 + 1.0
        index = end + 1

    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Average precision, the step-wise area under the precision-recall curve."""

    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    positives = int((y == 1).sum())
    if positives == 0:
        return float("nan")

    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    true_positives = np.cumsum(y_sorted)
    precision = true_positives / np.arange(1, len(y_sorted) + 1)
    return float((precision * y_sorted).sum() / positives)


def compute_metrics(
    labels: Sequence[int], scores: Sequence[float], threshold: float = 0.5
) -> BinaryMetrics:
    """Compute every benchmark metric for one set of scores."""

    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape:
        raise ValueError(f"labels and scores must align, got {y.shape} and {s.shape}")
    if y.size == 0:
        raise ValueError("Cannot compute metrics for an empty set of predictions")

    predicted = (s >= float(threshold)).astype(np.int64)
    true_positives = int(((predicted == 1) & (y == 1)).sum())
    false_positives = int(((predicted == 1) & (y == 0)).sum())
    true_negatives = int(((predicted == 0) & (y == 0)).sum())
    false_negatives = int(((predicted == 0) & (y == 1)).sum())

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives)
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives)
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return BinaryMetrics(
        count=int(y.size),
        positives=int((y == 1).sum()),
        negatives=int((y == 0).sum()),
        threshold=float(threshold),
        accuracy=float((predicted == y).mean()),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        auc=roc_auc(y, s),
        average_precision=average_precision(y, s),
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
    )
