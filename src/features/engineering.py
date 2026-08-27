"""Lightweight pixel-derived forensic features.

These features intentionally use decoded image pixels only. They do not read
EXIF fields, filenames, watermarks, or other metadata and are used for
diagnostics/error analysis rather than as a substitute for the CNN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np
import pandas as pd

from src.data.dataset import ImageRecord, read_image


FEATURE_NAMES = [
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "rgb_std_r",
    "rgb_std_g",
    "rgb_std_b",
    "gray_mean",
    "gray_std",
    "gray_entropy",
    "edge_density",
    "laplacian_variance",
    "saturation_mean",
    "saturation_std",
]


def _entropy(gray_uint8: np.ndarray) -> float:
    histogram = cv2.calcHist([gray_uint8], [0], None, [256], [0, 256]).ravel()
    probabilities = histogram / max(float(histogram.sum()), 1.0)
    nonzero = probabilities[probabilities > 0]
    return float(-(nonzero * np.log2(nonzero)).sum())


def compute_image_features(image: np.ndarray) -> Dict[str, float]:
    """Extract compact, transformation-sensitive pixel statistics."""

    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("compute_image_features expects an HxWx3 RGB NumPy array")
    if image.size == 0:
        raise ValueError("Cannot extract features from an empty image")
    rgb = image.astype(np.float32) / 255.0
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    laplacian = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    return {
        "rgb_mean_r": float(rgb[:, :, 0].mean()),
        "rgb_mean_g": float(rgb[:, :, 1].mean()),
        "rgb_mean_b": float(rgb[:, :, 2].mean()),
        "rgb_std_r": float(rgb[:, :, 0].std()),
        "rgb_std_g": float(rgb[:, :, 1].std()),
        "rgb_std_b": float(rgb[:, :, 2].std()),
        "gray_mean": float(gray.mean() / 255.0),
        "gray_std": float(gray.std() / 255.0),
        "gray_entropy": _entropy(gray),
        "edge_density": float(np.count_nonzero(edges) / edges.size),
        "laplacian_variance": float(laplacian.var()),
        "saturation_mean": float(hsv[:, :, 1].mean() / 255.0),
        "saturation_std": float(hsv[:, :, 1].std() / 255.0),
    }


def features_from_path(path: str | Path) -> Dict[str, float]:
    """Extract pixel features from an image path with existing image errors preserved."""

    return compute_image_features(read_image(path))


def records_to_feature_frame(records: Sequence[ImageRecord]) -> pd.DataFrame:
    """Create a pandas feature table for a sequence of labelled records."""

    rows: List[Dict[str, object]] = []
    for record in records:
        row: Dict[str, object] = {"image_path": record.path, "label": int(record.label)}
        row.update(features_from_path(record.path))
        rows.append(row)
    columns = ["image_path", "label", *FEATURE_NAMES]
    return pd.DataFrame(rows, columns=columns)
