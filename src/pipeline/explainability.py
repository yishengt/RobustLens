"""Stage 11: explainability.

Provides a Grad-CAM suspicious-region heatmap plus the chart data the demo
renders. Grad-CAM needs a spatial convolutional layer; when the selected model
does not expose one, every function here degrades to a clear message and the
pipeline carries on. Nothing in this module can abort an analysis.

Chart helpers return plain dicts rather than figures so the same data can feed
Streamlit's native charts, a notebook, or a JSON report without pulling in a
plotting dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.pipeline.confidence import ConfidenceReport
from src.pipeline.consistency import ConsistencyReport
from src.pipeline.model_loader import ModelBundle
from src.pipeline.prediction import Prediction

_RESAMPLE = getattr(Image, "Resampling", Image)


class ExplainabilityUnavailable(RuntimeError):
    """Raised internally when Grad-CAM cannot run; always caught and reported."""


@dataclass
class ExplanationResult:
    """Grad-CAM output, or a reason it was not produced."""

    available: bool
    method: str
    message: str
    heatmap: Optional[np.ndarray] = None
    overlay: Optional[np.ndarray] = None
    charts: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe summary; the raw arrays are omitted."""

        return {
            "available": self.available,
            "method": self.method,
            "message": self.message,
            "has_heatmap": self.heatmap is not None,
            "charts": self.charts,
        }


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------


def find_target_layer(model: nn.Module, architecture: str = "") -> Optional[nn.Module]:
    """Pick the last spatial feature block for the known architectures."""

    named = dict(model.named_modules())
    preferred = {
        "efficientnet_b0": ["features.8", "features"],
        "convnext_tiny": ["features.7", "features"],
        "resnet18": ["layer4"],
    }.get(str(architecture).lower(), [])

    for name in preferred:
        layer = named.get(name)
        if layer is not None:
            return layer

    convolutions = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    return convolutions[-1] if convolutions else None


class GradCAM:
    """Minimal Grad-CAM: weight activations by their mean gradient."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        if target_layer is None:
            raise ExplainabilityUnavailable("No convolutional layer available for Grad-CAM.")
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._handles: List[Any] = []

    def __enter__(self) -> "GradCAM":
        self._handles = [
            self.target_layer.register_forward_hook(self._forward_hook),
            self.target_layer.register_full_backward_hook(self._backward_hook),
        ]
        return self

    def __exit__(self, *_: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _forward_hook(self, _module: nn.Module, _inputs: Any, output: Any) -> None:
        self._activations = output[0] if isinstance(output, tuple) else output

    def _backward_hook(self, _module: nn.Module, _grad_in: Any, grad_out: Any) -> None:
        self._gradients = grad_out[0]

    def __call__(self, tensor: torch.Tensor) -> np.ndarray:
        """Return a normalised ``HxW`` heatmap in ``[0, 1]`` for one image."""

        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        # Grad-CAM needs autograd, so parameters are temporarily made
        # differentiable and restored afterwards.
        original_flags = [parameter.requires_grad for parameter in self.model.parameters()]
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        try:
            with torch.enable_grad():
                self.model.zero_grad(set_to_none=True)
                # clone() detaches from any inference-mode provenance.
                inputs = tensor.clone().detach().requires_grad_(True)
                outputs = self.model(inputs)
                flat = outputs.reshape(outputs.shape[0], -1)
                # Target the AI-generated logit: column 1 for a softmax pair,
                # the single logit otherwise.
                column = 1 if flat.shape[-1] > 1 else 0
                flat[0, column].backward()

            if self._activations is None or self._gradients is None:
                raise ExplainabilityUnavailable(
                    "Grad-CAM hooks captured no activations or gradients for this model."
                )
            activations = self._activations.detach()
            gradients = self._gradients.detach()
            if activations.dim() != 4:
                raise ExplainabilityUnavailable(
                    f"Grad-CAM needs a 4-D spatial feature map, got shape "
                    f"{tuple(activations.shape)}."
                )

            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activations).sum(dim=1))[0]
        finally:
            self.model.zero_grad(set_to_none=True)
            for parameter, flag in zip(self.model.parameters(), original_flags):
                parameter.requires_grad_(flag)

        heatmap = cam.float().cpu().numpy()
        heatmap -= heatmap.min()
        peak = float(heatmap.max())
        return heatmap / peak if peak > 1e-8 else np.zeros_like(heatmap)


def _jet_colormap(values: np.ndarray) -> np.ndarray:
    """Map ``[0, 1]`` values to a JET-style RGB image without OpenCV."""

    scaled = np.clip(values, 0.0, 1.0) * 4.0
    red = np.clip(1.5 - np.abs(scaled - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(scaled - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(scaled - 1.0), 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a heatmap over the image, returning an ``HxWx3`` uint8 array."""

    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    resized = np.asarray(
        Image.fromarray(heatmap.astype(np.float32), mode="F").resize(
            (base.shape[1], base.shape[0]), _RESAMPLE.BILINEAR
        ),
        dtype=np.float32,
    )
    coloured = _jet_colormap(resized).astype(np.float32)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return np.clip((1.0 - alpha) * base + alpha * coloured, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Chart data
# ---------------------------------------------------------------------------


def prediction_comparison_chart(predictions: Sequence[Prediction]) -> Dict[str, Any]:
    """Per-version AI probabilities, ordered with the original first."""

    return {
        "title": "AI probability by image version",
        "x_label": "Image version",
        "y_label": "AI-generated probability",
        "labels": [item.name for item in predictions],
        "values": [round(float(item.ai_probability), 6) for item in predictions],
        "highlight": [item.name for item in predictions if item.is_original],
    }


def transformation_score_chart(predictions: Sequence[Prediction]) -> Dict[str, Any]:
    """Per-transformation drift relative to the original prediction."""

    original = next((item for item in predictions if item.is_original), None)
    baseline = float(original.ai_probability) if original else 0.0
    transformed = [item for item in predictions if not item.is_original]
    return {
        "title": "Drift from the original prediction",
        "x_label": "Transformation",
        "y_label": "AI probability - original probability",
        "baseline": round(baseline, 6),
        "labels": [item.name for item in transformed],
        "values": [round(float(item.ai_probability) - baseline, 6) for item in transformed],
    }


def confidence_bar_chart(
    confidence: ConfidenceReport, consistency: ConsistencyReport
) -> Dict[str, Any]:
    """The 0-1 components that make up the confidence level."""

    return {
        "title": f"Confidence: {confidence.level}",
        "x_label": "Component",
        "y_label": "Score (0-1)",
        "labels": ["Decisiveness", "Agreement", "Consistency", "Overall"],
        "values": [
            round(confidence.decisiveness, 6),
            round(confidence.agreement, 6),
            round(consistency.consistency_score, 6),
            round(confidence.score, 6),
        ],
    }


def build_charts(
    predictions: Sequence[Prediction],
    consistency: ConsistencyReport,
    confidence: ConfidenceReport,
) -> Dict[str, Any]:
    """Assemble every chart payload the demo can render."""

    return {
        "prediction_comparison": prediction_comparison_chart(predictions),
        "transformation_scores": transformation_score_chart(predictions),
        "confidence_components": confidence_bar_chart(confidence, consistency),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def explain(
    bundle: ModelBundle,
    tensor: torch.Tensor,
    image: Image.Image,
    config: Optional[Dict[str, Any]] = None,
) -> ExplanationResult:
    """Produce a Grad-CAM heatmap, or a clear reason it is unavailable.

    Never raises: an explainability failure is reported, not propagated.
    """

    settings = (config or {}).get("explainability", {}) or {}
    if not settings.get("enabled", True):
        return ExplanationResult(
            available=False,
            method="disabled",
            message="Explainability is disabled in the configuration.",
        )
    if not settings.get("gradcam", True):
        return ExplanationResult(
            available=False,
            method="disabled",
            message="Grad-CAM is disabled in the configuration.",
        )
    if bundle.input_kind == "dual":
        # Both dual-input detectors take two separately preprocessed tensors at
        # different resolutions, so a single-input Grad-CAM pass cannot attribute
        # the score correctly. For the LoRA detector there is a second problem:
        # gradients would have to flow through two adapted transformer branches
        # and be fused across differing token grids, which no single heatmap can
        # represent honestly. Report unavailable rather than draw a wrong map.
        if bundle.architecture == "bombek_siglip2_dinov2":
            reason = (
                "Grad-CAM is unavailable for the Bombek1 SigLIP2+DINOv2 LoRA detector. "
                "It scores two LoRA-adapted transformer branches on separately "
                "preprocessed inputs (SigLIP2 at 384px, DINOv2 at 392px) with "
                "different token grids, so no single attribution map can faithfully "
                "represent both. A merged heatmap would be misleading."
            )
        else:
            reason = (
                "Grad-CAM is unavailable for the dual-backbone model because it "
                "requires two processor-specific inputs."
            )
        return ExplanationResult(
            available=False,
            method="grad-cam",
            message=f"{reason} Predictions, consistency and confidence are unaffected.",
        )

    try:
        layer = find_target_layer(bundle.model, bundle.architecture)
        if layer is None:
            raise ExplainabilityUnavailable(
                f"Model '{bundle.architecture}' exposes no convolutional layer, so "
                f"Grad-CAM is not compatible with it. All other results are unaffected."
            )
        with GradCAM(bundle.model, layer) as cam:
            heatmap = cam(tensor.to(bundle.device))
        overlay = overlay_heatmap(image, heatmap, float(settings.get("overlay_alpha", 0.45)))
        return ExplanationResult(
            available=True,
            method="grad-cam",
            message=(
                "Warm regions contributed most to the AI-generated score. Grad-CAM "
                "shows where the model looked, not proof of manipulation."
            ),
            heatmap=heatmap,
            overlay=overlay,
        )
    except (ExplainabilityUnavailable, RuntimeError, ValueError, TypeError) as exc:
        return ExplanationResult(
            available=False,
            method="grad-cam",
            message=(
                f"Grad-CAM could not run for this model ({type(exc).__name__}: {exc}). "
                f"Predictions, consistency and confidence are unaffected."
            ),
        )
