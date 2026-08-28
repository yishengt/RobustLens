"""Stages 6-7: feature extraction, classification and per-version predictions.

The backbone does feature extraction and classification in one forward pass;
this module turns raw logits into calibrated 0-1 probabilities and the
three-way label required by the problem statement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from src.pipeline.model_loader import ModelBundle
from src.pipeline.preprocessing import Preprocessor
from src.pipeline.transformations import ORIGINAL_KEY

LABEL_AUTHENTIC = "Likely authentic"
LABEL_UNCERTAIN = "Uncertain"
LABEL_AI = "Likely AI-generated"

DEFAULT_AUTHENTIC_MAX = 0.40
DEFAULT_AI_MIN = 0.60
DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class Prediction:
    """One image version's result."""

    name: str
    ai_probability: float
    real_probability: float
    label: str
    is_original: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ai_probability": round(float(self.ai_probability), 6),
            "real_probability": round(float(self.real_probability), 6),
            "label": self.label,
            "is_original": self.is_original,
        }


def label_thresholds(config: Optional[Dict[str, Any]] = None) -> Tuple[float, float]:
    """Return the (authentic_max, ai_min) decision boundaries."""

    settings = (config or {}).get("labels", {}) or {}
    authentic_max = float(settings.get("authentic_max", DEFAULT_AUTHENTIC_MAX))
    ai_min = float(settings.get("ai_min", DEFAULT_AI_MIN))
    if not 0.0 <= authentic_max <= ai_min <= 1.0:
        raise ValueError(
            "labels.authentic_max must be <= labels.ai_min and both within [0, 1]; "
            f"got authentic_max={authentic_max}, ai_min={ai_min}"
        )
    return authentic_max, ai_min


def binary_threshold(config: Optional[Dict[str, Any]] = None) -> float:
    """Return the configurable binary AI/real decision threshold."""

    return float((config or {}).get("inference", {}).get("threshold", DEFAULT_THRESHOLD))


def label_for_probability(
    probability: float, config: Optional[Dict[str, Any]] = None
) -> str:
    """Map an AI probability to the three-way label.

    Default bands: ``[0.00, 0.40)`` likely authentic, ``[0.40, 0.60)``
    uncertain, ``[0.60, 1.00]`` likely AI-generated. Both boundaries are
    configurable under ``labels`` in the config file.
    """

    authentic_max, ai_min = label_thresholds(config)
    value = float(probability)
    if value < authentic_max:
        return LABEL_AUTHENTIC
    if value >= ai_min:
        return LABEL_AI
    return LABEL_UNCERTAIN


def probabilities_from_logits(
    logits: torch.Tensor, num_classes: int, ai_class_index: int = 1
) -> np.ndarray:
    """Convert model outputs to AI-generated probabilities in ``[0, 1]``."""

    logits = logits.detach().float().cpu()
    if num_classes == 1 or logits.dim() == 1 or logits.shape[-1] == 1:
        probabilities = torch.sigmoid(logits.reshape(-1))
    else:
        index = int(ai_class_index)
        if not 0 <= index < logits.shape[-1]:
            raise ValueError(
                f"model.ai_class_index={index} is outside the {logits.shape[-1]} output columns"
            )
        probabilities = torch.softmax(logits, dim=-1)[..., index].reshape(-1)
    return np.clip(probabilities.numpy().astype(np.float64), 0.0, 1.0)


@torch.inference_mode()
def predict_tensor_batch(bundle: ModelBundle, batch: torch.Tensor) -> np.ndarray:
    """Run a preprocessed ``NxCxHxW`` batch through the model."""

    if batch.dim() == 3:
        batch = batch.unsqueeze(0)
    if batch.dim() != 4:
        raise ValueError(f"Expected a 4-D NCHW batch, got shape {tuple(batch.shape)}")
    outputs = bundle.model(batch.to(bundle.device))
    return probabilities_from_logits(outputs, bundle.num_classes, bundle.ai_class_index)


def predict_images(
    bundle: ModelBundle,
    images: Sequence[Image.Image] | Iterable[Image.Image],
    preprocessor: Preprocessor,
    batch_size: int = 32,
) -> np.ndarray:
    """Preprocess and classify a sequence of images, returning AI probabilities."""

    images = list(images)
    if not images:
        return np.empty((0,), dtype=np.float64)
    batch_size = max(1, int(batch_size))
    chunks: List[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        window = images[start : start + batch_size]
        chunks.append(predict_tensor_batch(bundle, preprocessor.batch(window)))
    return np.concatenate(chunks, axis=0)


def predict_variants(
    bundle: ModelBundle,
    variants: Dict[str, Image.Image],
    preprocessor: Preprocessor,
    config: Optional[Dict[str, Any]] = None,
    batch_size: Optional[int] = None,
) -> List[Prediction]:
    """Classify the original plus every transformed version.

    The original is always first in the returned list so downstream stages can
    weight it separately.
    """

    if not variants:
        raise ValueError("At least one image version is required for prediction")
    if batch_size is None:
        batch_size = int((config or {}).get("inference", {}).get("batch_size", 32))

    names = list(variants)
    if ORIGINAL_KEY in names:
        names.remove(ORIGINAL_KEY)
        names.insert(0, ORIGINAL_KEY)

    probabilities = predict_images(
        bundle, [variants[name] for name in names], preprocessor, batch_size
    )
    return [
        Prediction(
            name=name,
            ai_probability=float(probability),
            real_probability=float(1.0 - probability),
            label=label_for_probability(float(probability), config),
            is_original=(name == ORIGINAL_KEY),
        )
        for name, probability in zip(names, probabilities)
    ]


def split_predictions(
    predictions: Sequence[Prediction],
) -> Tuple[Optional[Prediction], List[Prediction]]:
    """Split a prediction list into (original, transformed versions)."""

    original = next((item for item in predictions if item.is_original), None)
    transformed = [item for item in predictions if not item.is_original]
    return original, transformed
