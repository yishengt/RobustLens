"""Leakage and file-container shortcut probes used by the evaluation protocol."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import numpy as np
from PIL import Image

from src.evaluation.metrics import compute_metrics


def evaluate_format_reencoding(
    paths: Sequence[str | Path],
    labels: Sequence[int],
    scorer: Callable[[Path], float],
    threshold: float,
    output_format: str = "PNG",
) -> Dict[str, Any]:
    """Compare scores before/after putting every image in one file container.

    A detector exploiting class-correlated extensions or container metadata will
    move sharply when all inputs are decoded to RGB and re-encoded identically.
    The probe does not fit a threshold and never exposes its temporary files to
    training.
    """

    items = [Path(path).expanduser() for path in paths]
    y = [int(label) for label in labels]
    if len(items) != len(y) or not items:
        raise ValueError("paths and labels must be aligned and non-empty")
    normalized_format = str(output_format).strip().upper()
    suffixes = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}
    if normalized_format not in suffixes:
        raise ValueError(f"output_format must be one of {', '.join(suffixes)}")

    original_scores = [float(scorer(path)) for path in items]
    with tempfile.TemporaryDirectory(prefix="robustlens-format-probe-") as tmp:
        root = Path(tmp)
        normalized_paths = []
        for index, path in enumerate(items):
            destination = root / f"{index:06d}{suffixes[normalized_format]}"
            with Image.open(path) as image:
                image.convert("RGB").save(destination, format=normalized_format)
            normalized_paths.append(destination)
        normalized_scores = [float(scorer(path)) for path in normalized_paths]

    before = compute_metrics(y, original_scores, threshold).as_dict()
    after = compute_metrics(y, normalized_scores, threshold).as_dict()
    deltas = np.asarray(normalized_scores) - np.asarray(original_scores)
    decisions_before = np.asarray(original_scores) >= float(threshold)
    decisions_after = np.asarray(normalized_scores) >= float(threshold)
    return {
        "count": len(items),
        "threshold": float(threshold),
        "threshold_note": "Fixed before re-encoding; never fitted on this probe.",
        "output_format": normalized_format,
        "original_metrics": before,
        "reencoded_metrics": after,
        "mean_score_delta": float(np.mean(deltas)),
        "mean_absolute_score_delta": float(np.mean(np.abs(deltas))),
        "max_absolute_score_delta": float(np.max(np.abs(deltas))),
        "decision_preservation_rate": float(np.mean(decisions_before == decisions_after)),
        "per_image": [
            {
                "image_path": str(path),
                "label": label,
                "original_score": original,
                "reencoded_score": normalized,
                "score_delta": normalized - original,
            }
            for path, label, original, normalized in zip(
                items, y, original_scores, normalized_scores
            )
        ],
    }


__all__ = ["evaluate_format_reencoding"]
