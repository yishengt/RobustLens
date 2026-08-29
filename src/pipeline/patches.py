"""Stage 4b: patch-level analysis for locally AI-edited regions.

Whole-image scoring answers "is this picture synthetic?". It is much weaker at
"was a small region of this otherwise authentic photo replaced?", because a
single edited object is averaged away across the whole frame. This module tiles
the image into overlapping patches, scores each patch through the *same* model
and preprocessing as the full image, and reconstructs the scores into a
heatmap.

Cost warning
------------
Every patch is a full forward pass. With a 740 M-parameter backbone that is
seconds per patch on CPU, so ``patches.max_patches`` caps the work: when the
configured grid produces more boxes than that, the list is evenly subsampled
rather than silently running for minutes. Patch analysis is skipped entirely
for images too small to tile.

Interpretation
--------------
A hot patch means the detector responded strongly to that region. It is
evidence about the model's attention, **not** proof that the region was edited.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from src.evaluation.calibration import ProbabilityCalibrator
from src.pipeline.model_loader import ModelBundle
from src.pipeline.prediction import binary_threshold, predict_images
from src.pipeline.preprocessing import Preprocessor

DEFAULT_PATCH_SIZE = 256
DEFAULT_STRIDE = 192
DEFAULT_MIN_PATCH_SIZE = 64
DEFAULT_TOP_K = 3
DEFAULT_HEATMAP_THRESHOLD = 0.5
DEFAULT_MAX_PATCHES = 12
DEFAULT_EVIDENCE = "top_k_mean"

EVIDENCE_STATISTICS = ("top_k_mean", "max", "mean")


@dataclass(frozen=True)
class Patch:
    """One scored patch, in original-image pixel coordinates."""

    index: int
    x: int
    y: int
    width: int
    height: int
    ai_probability: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "score": round(float(self.ai_probability), 6),
        }


@dataclass
class PatchReport:
    """Patch-level findings, or the reason they are unavailable."""

    available: bool
    message: str
    patches: List[Patch] = field(default_factory=list)
    top_patches: List[Patch] = field(default_factory=list)
    highest_risk_region: Optional[Dict[str, Any]] = None
    mean_probability: Optional[float] = None
    max_probability: Optional[float] = None
    min_probability: Optional[float] = None
    evidence: Optional[float] = None
    agreement: Optional[float] = None
    heatmap: Optional[np.ndarray] = None
    coverage: Optional[np.ndarray] = None
    settings: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """JSON-safe summary. The raw heatmap array is omitted."""

        def rounded(value: Optional[float]) -> Optional[float]:
            return None if value is None else round(float(value), 6)

        return {
            "available": self.available,
            "message": self.message,
            "num_patches": len(self.patches),
            "mean_patch_probability": rounded(self.mean_probability),
            "max_patch_probability": rounded(self.max_probability),
            "min_patch_probability": rounded(self.min_probability),
            "patch_evidence": rounded(self.evidence),
            "patch_agreement": rounded(self.agreement),
            "highest_risk_region": self.highest_risk_region,
            "top_patches": [patch.as_dict() for patch in self.top_patches],
            "patches": [patch.as_dict() for patch in self.patches],
            "has_heatmap": self.heatmap is not None,
            "heatmap_coverage": (
                None if self.coverage is None else round(float(self.coverage.mean()), 6)
            ),
            "settings": dict(self.settings),
        }


def patch_settings(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolve the configurable patch parameters."""

    section = (config or {}).get("patches", {}) or {}
    settings = {
        "enabled": bool(section.get("enabled", True)),
        "patch_size": int(section.get("patch_size", DEFAULT_PATCH_SIZE)),
        "stride": int(section.get("stride", DEFAULT_STRIDE)),
        "min_patch_size": int(section.get("min_patch_size", DEFAULT_MIN_PATCH_SIZE)),
        "top_k": int(section.get("top_k", DEFAULT_TOP_K)),
        "heatmap_threshold": float(section.get("heatmap_threshold", DEFAULT_HEATMAP_THRESHOLD)),
        "max_patches": int(section.get("max_patches", DEFAULT_MAX_PATCHES)),
        "evidence_statistic": str(section.get("evidence_statistic", DEFAULT_EVIDENCE)),
    }
    if settings["patch_size"] <= 0:
        raise ValueError(f"patches.patch_size must be positive, got {settings['patch_size']}")
    if settings["stride"] <= 0:
        raise ValueError(f"patches.stride must be positive, got {settings['stride']}")
    if settings["min_patch_size"] <= 0:
        raise ValueError("patches.min_patch_size must be positive")
    if settings["top_k"] <= 0:
        raise ValueError("patches.top_k must be positive")
    if settings["max_patches"] <= 0:
        raise ValueError("patches.max_patches must be positive")
    if not 0.0 <= settings["heatmap_threshold"] <= 1.0:
        raise ValueError("patches.heatmap_threshold must be within [0, 1]")
    if settings["evidence_statistic"] not in EVIDENCE_STATISTICS:
        raise ValueError(
            f"patches.evidence_statistic must be one of {', '.join(EVIDENCE_STATISTICS)}"
        )
    return settings


def _axis_positions(length: int, patch: int, stride: int) -> List[int]:
    """Start offsets covering one axis, always including the final edge."""

    if length <= patch:
        return [0]
    positions = list(range(0, length - patch + 1, stride))
    if positions[-1] != length - patch:
        positions.append(length - patch)
    return positions


def generate_patch_boxes(
    width: int,
    height: int,
    patch_size: int = DEFAULT_PATCH_SIZE,
    stride: int = DEFAULT_STRIDE,
    min_patch_size: int = DEFAULT_MIN_PATCH_SIZE,
    max_patches: int = DEFAULT_MAX_PATCHES,
) -> List[Tuple[int, int, int, int]]:
    """Return overlapping ``(x, y, w, h)`` boxes covering the image.

    Returns an empty list when the image is smaller than ``min_patch_size`` on
    either side. When the grid exceeds ``max_patches`` the boxes are evenly
    subsampled, which keeps spatial coverage while bounding the compute cost.
    """

    if min(width, height) < min_patch_size:
        return []

    # Shrink the window for images smaller than the configured patch size, so
    # medium images still get a real grid instead of one whole-image patch.
    effective = max(min_patch_size, min(patch_size, width, height))
    effective_stride = max(1, min(stride, effective))

    boxes = [
        (x, y, effective, effective)
        for y in _axis_positions(height, effective, effective_stride)
        for x in _axis_positions(width, effective, effective_stride)
    ]
    if len(boxes) > max_patches:
        indices = np.linspace(0, len(boxes) - 1, num=max_patches).round().astype(int)
        boxes = [boxes[index] for index in sorted(set(indices.tolist()))]
    return boxes


def build_heatmap(
    boxes: Sequence[Tuple[int, int, int, int]],
    scores: Sequence[float],
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct patch scores into ``(heatmap, coverage)`` maps.

    Overlapping patches are averaged, so a pixel covered by several boxes gets
    the mean of their scores rather than whichever box happened to be last.

    ``coverage`` is a boolean mask of pixels that any patch actually scored.
    When ``max_patches`` subsamples the grid some pixels are never scored, and
    they must not be drawn as cold: a 0.0 in an unscored region would read as
    "confidently authentic here" when nothing was measured at all.
    """

    total = np.zeros((height, width), dtype=np.float64)
    counts = np.zeros((height, width), dtype=np.float64)
    for (x, y, box_width, box_height), score in zip(boxes, scores):
        total[y : y + box_height, x : x + box_width] += float(score)
        counts[y : y + box_height, x : x + box_width] += 1.0
    covered = counts > 0
    heatmap = np.zeros((height, width), dtype=np.float32)
    heatmap[covered] = (total[covered] / counts[covered]).astype(np.float32)
    return np.clip(heatmap, 0.0, 1.0), covered


def _evidence_value(scores: np.ndarray, statistic: str, top_k: int) -> float:
    """Reduce patch scores to the single number fusion consumes."""

    if statistic == "max":
        return float(scores.max())
    if statistic == "mean":
        return float(scores.mean())
    # top_k_mean: less jumpy than a bare max, which drifts upward simply
    # because it is the maximum of many noisy scores and inflates false
    # positives on authentic images with many patches.
    k = min(int(top_k), scores.size)
    return float(np.sort(scores)[-k:].mean())


def analyse_patches(
    bundle: ModelBundle,
    image: Image.Image,
    preprocessor: Preprocessor,
    config: Optional[Dict[str, Any]] = None,
    whole_image_probability: Optional[float] = None,
    calibrator: Optional[ProbabilityCalibrator] = None,
) -> PatchReport:
    """Score overlapping patches and reconstruct a risk heatmap.

    Never raises: any failure is reported as an unavailable :class:`PatchReport`
    so the surrounding pipeline keeps its whole-image result.
    """

    try:
        settings = patch_settings(config)
    except ValueError as exc:
        return PatchReport(available=False, message=f"Invalid patch configuration: {exc}")

    if not settings["enabled"]:
        return PatchReport(
            available=False,
            message="Patch-level analysis is disabled in the configuration.",
            settings=settings,
        )

    rgb = image.convert("RGB")
    width, height = rgb.size
    boxes = generate_patch_boxes(
        width,
        height,
        patch_size=settings["patch_size"],
        stride=settings["stride"],
        min_patch_size=settings["min_patch_size"],
        max_patches=settings["max_patches"],
    )
    if not boxes:
        return PatchReport(
            available=False,
            message=(
                f"Image is {width}x{height}, smaller than the "
                f"{settings['min_patch_size']}px minimum patch size, so patch-level "
                f"analysis was skipped. The whole-image result is unaffected."
            ),
            settings=settings,
        )
    if len(boxes) < 2:
        # A single box spanning the whole image would just re-measure the
        # whole-image score. Feeding that back as independent "patch evidence"
        # would double-count one measurement and make patch agreement
        # trivially 1.0, inflating confidence for no new information.
        return PatchReport(
            available=False,
            message=(
                f"Image is {width}x{height}, which fits in a single patch, so patch-level "
                f"analysis would only repeat the whole-image score. It was skipped and the "
                f"whole-image result is unaffected."
            ),
            settings=settings,
        )

    try:
        crops = [rgb.crop((x, y, x + w, y + h)) for x, y, w, h in boxes]
        raw_scores = predict_images(
            bundle,
            crops,
            preprocessor,
            batch_size=int((config or {}).get("inference", {}).get("batch_size", 8)),
        )
        scores = (
            calibrator.transform(raw_scores) if calibrator is not None else np.asarray(raw_scores)
        )
        scores = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    except (RuntimeError, ValueError, OSError, MemoryError) as exc:
        return PatchReport(
            available=False,
            message=(
                f"Patch-level analysis failed ({type(exc).__name__}: {exc}). "
                f"The whole-image result is unaffected."
            ),
            settings=settings,
        )

    patches = [
        Patch(index=index, x=x, y=y, width=w, height=h, ai_probability=float(score))
        for index, ((x, y, w, h), score) in enumerate(zip(boxes, scores))
    ]
    ordered = sorted(patches, key=lambda patch: patch.ai_probability, reverse=True)
    top_patches = ordered[: settings["top_k"]]

    threshold = binary_threshold(config)
    reference = (
        float(whole_image_probability)
        if whole_image_probability is not None
        else float(scores.mean())
    )
    reference_is_ai = reference >= threshold
    agreement = float(np.mean([(score >= threshold) == reference_is_ai for score in scores]))

    heatmap, coverage = build_heatmap(boxes, scores.tolist(), width, height)
    highest = ordered[0]
    return PatchReport(
        available=True,
        message=(
            f"Scored {len(patches)} overlapping patches. Warmer regions produced higher "
            f"synthetic-image signals; this indicates where the detector responded, not "
            f"proof that a region was edited."
        ),
        patches=patches,
        top_patches=top_patches,
        highest_risk_region=highest.as_dict(),
        mean_probability=float(scores.mean()),
        max_probability=float(scores.max()),
        min_probability=float(scores.min()),
        evidence=_evidence_value(scores, settings["evidence_statistic"], settings["top_k"]),
        agreement=agreement,
        heatmap=heatmap,
        coverage=coverage,
        settings=settings,
    )


def overlay_patch_heatmap(
    image: Image.Image,
    report: PatchReport,
    alpha: float = 0.45,
    draw_top_boxes: bool = True,
) -> Optional[np.ndarray]:
    """Render the patch heatmap over the image as an ``HxWx3`` uint8 array.

    Regions no patch scored are left untinted rather than drawn cold, so the
    picture never implies a measurement that was not made. The highest-risk
    patches are outlined when ``draw_top_boxes`` is set.
    """

    if not report.available or report.heatmap is None:
        return None

    from src.pipeline.explainability import _jet_colormap

    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    heatmap = report.heatmap.astype(np.float32)
    if heatmap.shape != base.shape[:2]:  # pragma: no cover - defensive
        return None

    coloured = _jet_colormap(heatmap).astype(np.float32)
    weight = np.full(heatmap.shape, float(np.clip(alpha, 0.0, 1.0)), dtype=np.float32)
    if report.coverage is not None:
        weight = weight * report.coverage.astype(np.float32)
    weight = weight[..., None]

    blended = np.clip(base * (1.0 - weight) + coloured * weight, 0, 255).astype(np.uint8)

    if draw_top_boxes:
        height, width = heatmap.shape
        for patch in report.top_patches:
            x0, y0 = max(0, patch.x), max(0, patch.y)
            x1 = min(width - 1, patch.x + patch.width - 1)
            y1 = min(height - 1, patch.y + patch.height - 1)
            if x1 <= x0 or y1 <= y0:
                continue
            for thickness in range(2):
                top, bottom = min(y0 + thickness, y1), max(y1 - thickness, y0)
                left, right = min(x0 + thickness, x1), max(x1 - thickness, x0)
                blended[top, left : right + 1] = (255, 255, 255)
                blended[bottom, left : right + 1] = (255, 255, 255)
                blended[top : bottom + 1, left] = (255, 255, 255)
                blended[top : bottom + 1, right] = (255, 255, 255)
    return blended
