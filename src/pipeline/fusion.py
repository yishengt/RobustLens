"""Stage 8: prediction fusion.

The default mode combines three views of the same image::

    final = 0.6 * whole_image
          + 0.2 * mean(transformed versions)
          + 0.2 * patch_evidence

Whole-image scoring carries the most weight because it is the signal the model
was trained to produce. The transformed mean rewards predictions that survive
redistribution, and the patch term lets a locally edited region raise the score
on an image that looks authentic overall.

Every term degrades safely. If patch analysis is unavailable (image too small,
patches disabled, or a patch failure) its weight is dropped and the remaining
weights are renormalised rather than treating the missing evidence as zero.
The same applies when no transformed versions exist.

A simpler two-term mode remains available::

    final = 0.7 * original + 0.3 * mean(transformed)

An optional frequency mode is supported for when a trained frequency model is
available::

    final = 0.5 * rgb + 0.3 * frequency + 0.2 * transformation_consistency

Frequency mode never activates on its own: if no frequency probability is
supplied, fusion falls back to the default formula and records why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

MODE_RGB_TRANSFORM = "rgb_transform"
MODE_FREQUENCY = "frequency"
MODE_WHOLE_PATCH_TRANSFORM = "whole_patch_transform"

DEFAULT_ORIGINAL_WEIGHT = 0.7
DEFAULT_TRANSFORM_WEIGHT = 0.3
DEFAULT_FREQ_RGB_WEIGHT = 0.5
DEFAULT_FREQ_FREQ_WEIGHT = 0.3
DEFAULT_FREQ_CONSISTENCY_WEIGHT = 0.2
DEFAULT_WHOLE_WEIGHT = 0.6
DEFAULT_WP_TRANSFORM_WEIGHT = 0.2
DEFAULT_PATCH_WEIGHT = 0.2


@dataclass(frozen=True)
class FusionResult:
    """The fused probability plus the inputs and weights that produced it."""

    final_probability: float
    mode: str
    weights: Dict[str, float] = field(default_factory=dict)
    components: Dict[str, float] = field(default_factory=dict)
    fallback_reason: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "final_probability": round(self.final_probability, 6),
            "mode": self.mode,
            "weights": {key: round(value, 6) for key, value in self.weights.items()},
            "components": {key: round(value, 6) for key, value in self.components.items()},
            "fallback_reason": self.fallback_reason,
        }


def _normalised(weights: Dict[str, float]) -> Dict[str, float]:
    """Scale weights so they sum to 1, so a misconfigured file cannot skew the range."""

    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError(f"Fusion weights must sum to a positive value, got {weights}")
    return {key: float(value) / total for key, value in weights.items()}


def fuse_predictions(
    original_probability: float,
    transformed_probabilities: Sequence[float],
    config: Optional[Dict[str, Any]] = None,
    consistency: Optional[float] = None,
    frequency_probability: Optional[float] = None,
    patch_evidence: Optional[float] = None,
) -> FusionResult:
    """Combine the available signals into one AI-generated probability."""

    settings = (config or {}).get("fusion", {}) or {}
    mode = str(settings.get("mode", MODE_RGB_TRANSFORM)).strip().lower()
    valid_modes = (MODE_RGB_TRANSFORM, MODE_FREQUENCY, MODE_WHOLE_PATCH_TRANSFORM)
    if mode not in valid_modes:
        raise ValueError(f"Unknown fusion.mode '{mode}'. Valid modes: {', '.join(valid_modes)}.")

    original_probability = float(np.clip(original_probability, 0.0, 1.0))
    transformed = np.clip(np.asarray(list(transformed_probabilities), dtype=np.float64), 0.0, 1.0)
    fallback_reason: Optional[str] = None

    # --- RGB + transform fusion, always computed as the base signal --------
    if transformed.size == 0:
        rgb_weights = _normalised({"original": 1.0, "transformed": 0.0})
        rgb_probability = original_probability
        if (config or {}).get("transformations", {}).get("enabled", True):
            fallback_reason = "No transformed versions available; used the original only."
    else:
        rgb_weights = _normalised(
            {
                "original": float(settings.get("original_weight", DEFAULT_ORIGINAL_WEIGHT)),
                "transformed": float(settings.get("transform_weight", DEFAULT_TRANSFORM_WEIGHT)),
            }
        )
        rgb_probability = rgb_weights["original"] * original_probability + rgb_weights[
            "transformed"
        ] * float(transformed.mean())

    components: Dict[str, float] = {
        "original": original_probability,
        "transformed_mean": float(transformed.mean()) if transformed.size else original_probability,
        "rgb": float(rgb_probability),
    }

    # --- Whole image + patch evidence + transformed mean ------------------
    if mode == MODE_WHOLE_PATCH_TRANSFORM:
        whole_patch = settings.get("whole_patch", {}) or {}
        raw_weights: Dict[str, float] = {
            "whole": float(whole_patch.get("whole_weight", DEFAULT_WHOLE_WEIGHT)),
            "transformed": float(whole_patch.get("transform_weight", DEFAULT_WP_TRANSFORM_WEIGHT)),
            "patch": float(whole_patch.get("patch_weight", DEFAULT_PATCH_WEIGHT)),
        }
        missing: list[str] = []
        if transformed.size == 0:
            raw_weights.pop("transformed", None)
            missing.append("transformed versions")
        if patch_evidence is None:
            raw_weights.pop("patch", None)
            missing.append("patch-level evidence")
        if missing:
            fallback_reason = (
                f"Fusion ran without {' and '.join(missing)}; those weights were "
                f"dropped and the remaining weights renormalised."
            )

        weights = _normalised(raw_weights)
        final = weights["whole"] * original_probability
        if "transformed" in weights:
            final += weights["transformed"] * float(transformed.mean())
        if "patch" in weights:
            patch_value = float(np.clip(patch_evidence, 0.0, 1.0))
            components["patch_evidence"] = patch_value
            final += weights["patch"] * patch_value

        return FusionResult(
            final_probability=float(np.clip(final, 0.0, 1.0)),
            mode=MODE_WHOLE_PATCH_TRANSFORM,
            weights=weights,
            components=components,
            fallback_reason=fallback_reason,
        )

    # --- Optional frequency fusion ----------------------------------------
    if mode == MODE_FREQUENCY:
        if frequency_probability is None:
            fallback_reason = (
                "Frequency fusion requested but no frequency model prediction was "
                "available; fell back to the 70/30 RGB formula."
            )
            mode = MODE_RGB_TRANSFORM
        elif consistency is None:
            fallback_reason = (
                "Frequency fusion requested but no consistency score was available; "
                "fell back to the 70/30 RGB formula."
            )
            mode = MODE_RGB_TRANSFORM
        else:
            frequency_settings = settings.get("frequency", {}) or {}
            weights = _normalised(
                {
                    "rgb": float(frequency_settings.get("rgb_weight", DEFAULT_FREQ_RGB_WEIGHT)),
                    "frequency": float(
                        frequency_settings.get("frequency_weight", DEFAULT_FREQ_FREQ_WEIGHT)
                    ),
                    "consistency": float(
                        frequency_settings.get(
                            "consistency_weight", DEFAULT_FREQ_CONSISTENCY_WEIGHT
                        )
                    ),
                }
            )
            frequency_probability = float(np.clip(frequency_probability, 0.0, 1.0))
            consistency = float(np.clip(consistency, 0.0, 1.0))
            components["frequency"] = frequency_probability
            components["consistency"] = consistency
            final = (
                weights["rgb"] * rgb_probability
                + weights["frequency"] * frequency_probability
                + weights["consistency"] * consistency
            )
            return FusionResult(
                final_probability=float(np.clip(final, 0.0, 1.0)),
                mode=MODE_FREQUENCY,
                weights=weights,
                components=components,
                fallback_reason=None,
            )

    return FusionResult(
        final_probability=float(np.clip(rgb_probability, 0.0, 1.0)),
        mode=MODE_RGB_TRANSFORM,
        weights=rgb_weights,
        components=components,
        fallback_reason=fallback_reason,
    )
