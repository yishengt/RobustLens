"""Losses and dependency-light binary classification metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn


def binary_loss() -> nn.Module:
    """The detector has one output logit: authentic=0, ai_edited=1."""

    return nn.BCEWithLogitsLoss()


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute the requested image-level metrics without sklearn."""

    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if y.size != p.size:
        raise ValueError(f"labels and probabilities differ in length: {y.size} != {p.size}")
    predicted = p >= float(threshold)
    tp = float(np.sum(predicted & (y == 1)))
    tn = float(np.sum(~predicted & (y == 0)))
    fp = float(np.sum(predicted & (y == 0)))
    fn = float(np.sum(~predicted & (y == 1)))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    auc = float("nan")
    if positives and negatives:
        # Ascending averaged ranks implement the Mann-Whitney U identity. The
        # previous descending ordinal ranks inverted AUC and mishandled ties.
        order = np.argsort(p, kind="mergesort")
        ranks = np.empty(len(p), dtype=np.float64)
        sorted_scores = p[order]
        index = 0
        while index < len(sorted_scores):
            end = index
            while end + 1 < len(sorted_scores) and sorted_scores[end + 1] == sorted_scores[index]:
                end += 1
            ranks[order[index : end + 1]] = (index + end) / 2.0 + 1.0
            index = end + 1
        auc = float(
            (np.sum(ranks[y == 1]) - positives * (positives + 1) / 2)
            / (positives * negatives)
        )
    return {
        "loss": float("nan"),
        "accuracy": _safe_div(tp + tn, len(y)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auc,
        "fpr": _safe_div(fp, tn + fp),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "threshold": float(threshold),
    }


def metrics_by_subgroup(
    labels: np.ndarray,
    probabilities: np.ndarray,
    subgroups: Sequence[str],
    threshold: float = 0.5,
    expected_subgroups: Sequence[str] = (),
) -> Dict[str, Dict[str, Any]]:
    """Compute the same fixed-threshold metrics for each provenance subgroup."""

    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    groups = np.asarray(list(subgroups), dtype=object).reshape(-1)
    if not (y.size == p.size == groups.size):
        raise ValueError(
            "labels, probabilities and subgroups must align: "
            f"{y.size}, {p.size}, {groups.size}"
        )
    names = list(dict.fromkeys([*expected_subgroups, *sorted(set(groups.tolist()))]))
    result: Dict[str, Dict[str, Any]] = {}
    for name in names:
        mask = groups == name
        if not np.any(mask):
            result[name] = {"count": 0, "metrics": None}
            continue
        metrics: Dict[str, Any] = binary_metrics(y[mask], p[mask], threshold)
        if not np.isfinite(metrics["auroc"]):
            metrics["auroc"] = None
        result[name] = {
            "count": int(np.sum(mask)),
            "positives": int(np.sum(y[mask] == 1)),
            "negatives": int(np.sum(y[mask] == 0)),
            "metrics": metrics,
        }
    return result


def attach_loss(metrics: Dict[str, float], loss: torch.Tensor | float) -> Dict[str, float]:
    result = dict(metrics)
    result["loss"] = float(loss.detach().cpu().item() if isinstance(loss, torch.Tensor) else loss)
    return result


# ---------------------------------------------------------------------------
# Transformation-consistency loss
# ---------------------------------------------------------------------------
#
# The classification loss alone gives the model no reason to score two versions
# of one photograph alike, which is exactly the property Track 5 is judged on.
# This term supplies that reason: it penalises disagreement between versions
# that share a source image, and it never compares unrelated images -- pulling
# two different pictures toward the same score would destroy the signal rather
# than stabilise it.

CONSISTENCY_MSE = "mse"
CONSISTENCY_KL = "kl"
CONSISTENCY_METHODS = (CONSISTENCY_MSE, CONSISTENCY_KL)


def _pairs_within_groups(group_ids: Sequence[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Index pairs of batch entries sharing a group id.

    Returns two empty tensors when the batch holds no related pair, which is the
    normal case for a small batch and must cost nothing.
    """

    buckets: Dict[Any, List[int]] = {}
    for index, group in enumerate(group_ids):
        buckets.setdefault(group, []).append(index)
    left: List[int] = []
    right: List[int] = []
    for members in buckets.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                left.append(members[i])
                right.append(members[j])
    return (
        torch.tensor(left, dtype=torch.long),
        torch.tensor(right, dtype=torch.long),
    )


def transformation_consistency_loss(
    logits: torch.Tensor,
    group_ids: Sequence[Any],
    method: str = CONSISTENCY_MSE,
    labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Penalise disagreement between versions of the same source image.

    ``method='mse'`` compares logits directly; ``method='kl'`` compares the
    Bernoulli distributions the logits imply, symmetrised so neither version is
    treated as the teacher.

    When ``labels`` is supplied, only pairs that share a label are used. Two
    versions of one source can legitimately carry different labels here -- the
    authentic source and its edited copy share a group -- and forcing those to
    agree would train the model to ignore the very edit it must detect.
    """

    method = str(method).strip().lower()
    if method not in CONSISTENCY_METHODS:
        raise ValueError(
            f"consistency method must be one of {', '.join(CONSISTENCY_METHODS)}, got '{method}'"
        )

    logits = logits.reshape(-1)
    if len(group_ids) != logits.shape[0]:
        raise ValueError(
            f"group_ids has {len(group_ids)} entries but logits has {logits.shape[0]}"
        )

    left, right = _pairs_within_groups(group_ids)
    if left.numel() == 0:
        return logits.new_zeros(())

    left = left.to(logits.device)
    right = right.to(logits.device)
    if labels is not None:
        labels = labels.reshape(-1).to(logits.device)
        keep = labels[left] == labels[right]
        left, right = left[keep], right[keep]
        if left.numel() == 0:
            return logits.new_zeros(())

    a = logits[left]
    b = logits[right]
    if method == CONSISTENCY_MSE:
        return torch.mean((a - b) ** 2)

    # Symmetric KL between the two Bernoulli distributions.
    pa = torch.sigmoid(a).clamp(1e-6, 1 - 1e-6)
    pb = torch.sigmoid(b).clamp(1e-6, 1 - 1e-6)
    kl_ab = pa * torch.log(pa / pb) + (1 - pa) * torch.log((1 - pa) / (1 - pb))
    kl_ba = pb * torch.log(pb / pa) + (1 - pb) * torch.log((1 - pb) / (1 - pa))
    return torch.mean(0.5 * (kl_ab + kl_ba))


def combined_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_ids: Sequence[Any],
    criterion: nn.Module,
    consistency_weight: float = 0.0,
    consistency_method: str = CONSISTENCY_MSE,
    same_label_pairs_only: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """``classification_loss + weight * consistency_loss``.

    Classification stays the primary objective: a weight of 0 makes this exactly
    the original loss, and the consistency term is never allowed to be the only
    gradient signal.
    """

    classification = criterion(logits.reshape(-1), labels.reshape(-1))
    weight = float(consistency_weight)
    if weight < 0:
        raise ValueError(f"consistency_weight must be non-negative, got {weight}")
    if weight == 0.0:
        return classification, {
            "classification_loss": float(classification.detach().cpu().item()),
            "consistency_loss": 0.0,
            "consistency_weight": 0.0,
        }

    consistency = transformation_consistency_loss(
        logits,
        group_ids,
        method=consistency_method,
        labels=labels if same_label_pairs_only else None,
    )
    total = classification + weight * consistency
    return total, {
        "classification_loss": float(classification.detach().cpu().item()),
        "consistency_loss": float(consistency.detach().cpu().item()),
        "consistency_weight": weight,
    }
