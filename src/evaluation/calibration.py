"""Probability calibration and fixed-threshold selection.

The calibrator is a small, dependency-free Platt model.  It operates on the
model's raw probability (or on logits when explicitly requested), fits only on
clean labelled validation data, and serialises its parameters as JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.evaluation.metrics import BinaryMetrics, compute_metrics

# Platt scaling fits a slope and an intercept on the logit; temperature scaling
# fits the slope only (intercept pinned at 0), which is the standard
# single-parameter calibrator for neural networks and cannot shift the
# decision boundary on its own.
CALIBRATION_METHODS = ("platt", "temperature")


def _arrays(values: Sequence[float], labels: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.shape != y.shape or x.size == 0:
        raise ValueError(
            f"scores and labels must be non-empty and aligned, got {x.shape} and {y.shape}"
        )
    if not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("labels must contain only 0 and 1")
    if np.unique(y).size < 2:
        raise ValueError("Calibration requires both real (0) and AI-generated (1) labels")
    if not np.all(np.isfinite(x)):
        raise ValueError("Calibration scores must be finite")
    return x, y


def _logit(probabilities: np.ndarray) -> np.ndarray:
    return np.log(
        np.clip(probabilities, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - probabilities, 1e-6, 1.0)
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-values))


@dataclass(frozen=True)
class ProbabilityCalibrator:
    """Platt calibration parameters for model probabilities or logits."""

    method: str = "platt"
    input_type: str = "probabilities"
    scale: float = 1.0
    bias: float = 0.0
    fitted_on: str = "clean_validation"
    selected_thresholds: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method not in CALIBRATION_METHODS:
            raise ValueError(
                f"method must be one of {', '.join(CALIBRATION_METHODS)}, got '{self.method}'"
            )
        if self.input_type not in {"probabilities", "logits"}:
            raise ValueError("input_type must be 'probabilities' or 'logits'")
        if not np.isfinite(self.scale) or not np.isfinite(self.bias):
            raise ValueError("Calibration parameters must be finite")

    def transform(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return calibrated probabilities clipped to [0, 1]."""

        values_array = np.asarray(values, dtype=np.float64)
        features = (
            values_array if self.input_type == "logits" else _logit(np.clip(values_array, 0.0, 1.0))
        )
        return np.clip(_sigmoid(self.scale * features + self.bias), 0.0, 1.0)

    @property
    def temperature(self) -> Optional[float]:
        """The temperature ``T`` for temperature scaling, where scale = 1/T."""

        if self.method != "temperature" or self.scale == 0:
            return None
        return float(1.0 / self.scale)

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "method": self.method,
            "input_type": self.input_type,
            "scale": round(float(self.scale), 12),
            "bias": round(float(self.bias), 12),
            "fitted_on": self.fitted_on,
            "selected_thresholds": self.selected_thresholds,
        }
        if self.temperature is not None:
            payload["temperature"] = round(float(self.temperature), 12)
        return payload

    def save(self, path: str | Path) -> Path:
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(self.as_dict(), handle, indent=2)
            handle.write("\n")
        return output

    @classmethod
    def load(cls, path: str | Path) -> "ProbabilityCalibrator":
        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"Calibration parameters not found: {source}")
        try:
            with source.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            return cls(
                **{key: payload[key] for key in ("method", "input_type", "scale", "bias")},
                fitted_on=payload.get("fitted_on", "unknown"),
                selected_thresholds=payload.get("selected_thresholds", {}) or {},
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Could not load calibration parameters '{source}': {exc}") from exc

    @classmethod
    def fit(
        cls,
        values: Sequence[float],
        labels: Sequence[int],
        input_type: str = "probabilities",
        max_iter: int = 100,
        method: str = "platt",
    ) -> "ProbabilityCalibrator":
        """Fit a calibrator on clean labelled validation scores.

        ``platt`` fits a slope and an intercept on the logit. ``temperature``
        fits the slope only, holding the intercept at zero: it sharpens or
        softens confidence without moving the decision boundary, which is the
        usual choice when the ranking is already good but the probabilities are
        over-confident.
        """

        if method not in CALIBRATION_METHODS:
            raise ValueError(
                f"method must be one of {', '.join(CALIBRATION_METHODS)}, got '{method}'"
            )

        x, y = _arrays(values, labels)
        if input_type == "probabilities":
            x = _logit(np.clip(x, 0.0, 1.0))
        elif input_type != "logits":
            raise ValueError("input_type must be 'probabilities' or 'logits'")

        # Platt's target smoothing. With linearly separable validation scores
        # the unsmoothed logistic optimum sits at infinity, so the fit collapses
        # to a step function of only 0.0 and 1.0 -- which looks confident but
        # makes every decision threshold equivalent. Smoothing the targets to
        # (N+ + 1)/(N+ + 2) and 1/(N- + 2) keeps the optimum finite, as in
        # Platt (1999).
        positives = float(np.sum(y == 1.0))
        negatives = float(np.sum(y == 0.0))
        high = (positives + 1.0) / (positives + 2.0)
        low = 1.0 / (negatives + 2.0)
        y = np.where(y == 1.0, high, low)

        if method == "temperature":
            return cls._fit_temperature(x, y, input_type, max_iter)

        # Start at identity scaling and optimise the logistic NLL with a small
        # ridge term. The Newton step is damped by a backtracking line search:
        # an undamped step diverges on near-separable validation data, driving
        # the slope to ~1e8 and collapsing every calibrated score to exactly 0
        # or 1. That looks like a confident calibrator but is a step function,
        # and it makes every decision threshold equivalent.
        params = np.array([1.0, 0.0], dtype=np.float64)
        ridge = 1e-6

        def negative_log_likelihood(values: np.ndarray) -> float:
            logits = np.clip(values[0] * x + values[1], -60.0, 60.0)
            loss = float(np.sum(np.logaddexp(0.0, logits) - y * logits))
            return loss + 0.5 * ridge * float(np.dot(values, values))

        current_loss = negative_log_likelihood(params)
        for _ in range(max(1, int(max_iter))):
            probabilities = _sigmoid(params[0] * x + params[1])
            residual = probabilities - y
            gradient = np.array([np.dot(residual, x), residual.sum()]) + ridge * params
            if not np.all(np.isfinite(gradient)) or np.max(np.abs(gradient)) < 1e-9:
                break
            curvature = probabilities * (1.0 - probabilities)
            hessian = np.array(
                [
                    [np.dot(curvature, x * x) + ridge, np.dot(curvature, x)],
                    [np.dot(curvature, x), curvature.sum() + ridge],
                ],
                dtype=np.float64,
            )
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(step)):
                break

            # Backtrack until the objective actually improves.
            scale_factor = 1.0
            improved = False
            for _ in range(40):
                candidate = params - scale_factor * step
                if np.all(np.isfinite(candidate)):
                    candidate_loss = negative_log_likelihood(candidate)
                    if candidate_loss <= current_loss:
                        improved = True
                        break
                scale_factor *= 0.5
            if not improved:
                break
            shift = np.max(np.abs(candidate - params))
            params, current_loss = candidate, candidate_loss
            if shift < 1e-9:
                break
        return cls(
            method="platt", input_type=input_type, scale=float(params[0]), bias=float(params[1])
        )

    @classmethod
    def _fit_temperature(
        cls, x: np.ndarray, y: np.ndarray, input_type: str, max_iter: int
    ) -> "ProbabilityCalibrator":
        """Fit the single slope w = 1/T by Newton descent on the logistic NLL."""

        weight = 1.0
        ridge = 1e-6
        for _ in range(max(1, int(max_iter))):
            probabilities = _sigmoid(weight * x)
            gradient = float(np.dot(probabilities - y, x)) + ridge * weight
            curvature = probabilities * (1.0 - probabilities)
            hessian = float(np.dot(curvature, x * x)) + ridge
            if hessian <= 0:
                break
            step = gradient / hessian
            updated = weight - step
            # A non-positive temperature would invert the ranking, so keep the
            # slope strictly positive.
            if updated <= 1e-6:
                updated = 1e-6
            if abs(updated - weight) < 1e-9:
                weight = updated
                break
            weight = updated
        return cls(method="temperature", input_type=input_type, scale=float(weight), bias=0.0)


@dataclass(frozen=True)
class ThresholdSelection:
    """Three frozen operating points selected from clean validation data."""

    balanced: float
    f1_optimal: float
    low_false_positive: float
    high_recall: float
    target_false_positive_rate: float
    target_met: bool
    metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    curve: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "balanced_threshold": self.balanced,
            "f1_optimal_threshold": self.f1_optimal,
            "low_false_positive_threshold": self.low_false_positive,
            "high_recall_threshold": self.high_recall,
            "target_false_positive_rate": self.target_false_positive_rate,
            "target_met": self.target_met,
            "selected_metrics": self.metrics,
            "threshold_curve": self.curve,
        }


def search_thresholds(
    labels: Sequence[int], scores: Sequence[float], target_false_positive_rate: float = 0.01
) -> ThresholdSelection:
    """Evaluate thresholds 0.01..0.99 and select fixed operating points."""

    if not 0.0 <= float(target_false_positive_rate) <= 1.0:
        raise ValueError("target_false_positive_rate must be within [0, 1]")
    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    curve: List[Dict[str, Any]] = []
    metric_objects: Dict[float, BinaryMetrics] = {}
    for threshold in thresholds:
        metrics = compute_metrics(labels, scores, float(threshold))
        metric_objects[float(threshold)] = metrics
        curve.append(metrics.as_dict())

    # Maximise Youden's J; ties choose the threshold nearest the conventional
    # midpoint to avoid arbitrary changes between equivalent candidates.
    balanced = max(
        thresholds, key=lambda t: (metric_objects[float(t)].youden_j, -abs(float(t) - 0.5))
    )
    # Maximise F1; ties prefer the lower threshold, which favours recall.
    f1_optimal = max(thresholds, key=lambda t: (metric_objects[float(t)].f1, -float(t)))

    eligible = [
        t
        for t in thresholds
        if metric_objects[float(t)].false_positive_rate <= target_false_positive_rate
    ]
    target_met = bool(eligible)
    if eligible:
        low_fp = min(eligible, key=float)
    else:
        low_fp = min(
            thresholds, key=lambda t: (metric_objects[float(t)].false_positive_rate, float(t))
        )

    high_recall = max(
        thresholds,
        key=lambda t: (
            metric_objects[float(t)].recall,
            -metric_objects[float(t)].false_positive_rate,
            -float(t),
        ),
    )
    selected = {
        "balanced": metric_objects[float(balanced)].as_dict(),
        "f1_optimal": metric_objects[float(f1_optimal)].as_dict(),
        "low_false_positive": metric_objects[float(low_fp)].as_dict(),
        "high_recall": metric_objects[float(high_recall)].as_dict(),
    }
    return ThresholdSelection(
        balanced=float(balanced),
        f1_optimal=float(f1_optimal),
        low_false_positive=float(low_fp),
        high_recall=float(high_recall),
        target_false_positive_rate=float(target_false_positive_rate),
        target_met=target_met,
        metrics=selected,
        curve=curve,
    )


def calibration_summary(
    labels: Sequence[int], scores: Sequence[float], bins: int = 10
) -> Dict[str, Any]:
    """Return reliability bins, ECE, Brier score, and mean confidence."""

    y = np.asarray(labels, dtype=np.float64)
    p = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("labels and scores must be non-empty and aligned")
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    rows: List[Dict[str, Any]] = []
    ece = 0.0
    for index in range(int(bins)):
        mask = (p >= edges[index]) & (
            (p < edges[index + 1]) if index < bins - 1 else (p <= edges[index + 1])
        )
        count = int(mask.sum())
        accuracy = float(y[mask].mean()) if count else None
        confidence = float(p[mask].mean()) if count else None
        if count:
            ece += count / p.size * abs(accuracy - confidence)
        rows.append(
            {
                "bin_lower": round(float(edges[index]), 1),
                "bin_upper": round(float(edges[index + 1]), 1),
                "count": count,
                "accuracy": accuracy,
                "mean_confidence": confidence,
            }
        )
    return {
        "expected_calibration_error": float(ece),
        "brier_score": float(np.mean((p - y) ** 2)),
        "mean_confidence": float(p.mean()),
        "accuracy_by_confidence_bin": rows,
    }


def plot_calibration_results(
    labels: Sequence[int],
    raw_scores: Sequence[float],
    calibrated_scores: Sequence[float],
    selection: ThresholdSelection,
    output_dir: str | Path,
) -> List[str]:
    """Save requested calibration/threshold plots when matplotlib is available."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to save calibration plots") from exc
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    y = np.asarray(labels)
    raw = np.asarray(raw_scores)
    calibrated = np.asarray(calibrated_scores)
    paths: List[str] = []

    def save(name: str) -> None:
        path = output / name
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
        paths.append(str(path))

    for scores, title, filename in (
        (raw, "Raw reliability diagram", "reliability_raw.png"),
        (calibrated, "Calibrated reliability diagram", "reliability_calibrated.png"),
    ):
        plt.figure()
        summary = calibration_summary(y, scores)
        points = [
            (row["mean_confidence"], row["accuracy"])
            for row in summary["accuracy_by_confidence_bin"]
            if row["count"]
        ]
        plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        if points:
            plt.plot(*zip(*points), "o-", label="Model")
        plt.xlabel("Mean confidence")
        plt.ylabel("Accuracy")
        plt.title(title)
        plt.legend()
        save(filename)

    curve = selection.curve
    x = [row["threshold"] for row in curve]
    for key, title, filename in (
        ("f1", "F1 versus threshold", "f1_vs_threshold.png"),
        ("false_positive_rate", "False-positive rate versus threshold", "fpr_vs_threshold.png"),
        ("recall", "Recall versus threshold", "recall_vs_threshold.png"),
    ):
        plt.figure()
        plt.plot(x, [row[key] for row in curve])
        plt.xlabel("Threshold")
        plt.ylabel(key)
        plt.title(title)
        save(filename)

    plt.figure()
    for label, value in (("Real", 0), ("AI-generated", 1)):
        subset = calibrated[y == value]
        if subset.size:
            plt.hist(subset, bins=10, alpha=0.6, label=label)
    plt.xlabel("Calibrated AI probability")
    plt.ylabel("Images")
    plt.title("Prediction distribution")
    plt.legend()
    save("prediction_distribution.png")
    return paths


def plot_roc_pr_curves(
    labels: Sequence[int], scores: Sequence[float], output_dir: str | Path
) -> List[str]:
    """Save ROC and precision-recall curves for the clean validation set."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to save evaluation plots") from exc
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(scores, dtype=np.float64)
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    thresholds = np.r_[1.0, np.linspace(0.99, 0.01, 99), 0.0]
    tprs, fprs, precisions = [], [], []
    for threshold in thresholds:
        predicted = p >= threshold
        tp = int(((predicted) & (y == 1)).sum())
        fp = int(((predicted) & (y == 0)).sum())
        tn = int(((~predicted) & (y == 0)).sum())
        fn = int(((~predicted) & (y == 1)).sum())
        tprs.append(tp / (tp + fn) if tp + fn else 0.0)
        fprs.append(fp / (fp + tn) if fp + tn else 0.0)
        precisions.append(tp / (tp + fp) if tp + fp else 1.0)

    paths: List[str] = []
    for name, x, xlabel, y_values, ylabel, title in (
        ("roc_curve.png", fprs, "False-positive rate", tprs, "Recall", "ROC curve"),
        (
            "precision_recall_curve.png",
            tprs,
            "Recall",
            precisions,
            "Precision",
            "Precision-recall curve",
        ),
    ):
        plt.figure()
        plt.plot(x, y_values)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        path = output / name
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
        paths.append(str(path))
    return paths


def plot_clean_vs_transformed(
    per_version: Dict[str, Dict[str, Any]], output_dir: str | Path
) -> str:
    """Save a compact fixed-threshold accuracy comparison chart."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to save evaluation plots") from exc
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    names = list(per_version)
    values = [per_version[name].get("accuracy", 0.0) for name in names]
    plt.figure(figsize=(max(7, len(names) * 0.55), 4))
    plt.bar(names, values)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Fixed-threshold clean versus transformed performance")
    plt.xticks(rotation=65, ha="right")
    plt.tight_layout()
    path = output / "clean_vs_transformed.png"
    plt.savefig(path, dpi=140)
    plt.close()
    return str(path)
