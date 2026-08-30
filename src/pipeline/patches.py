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

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
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
DEFAULT_COARSE_MAX_PATCHES = 4
DEFAULT_REFINE_FACTOR = 2
DEFAULT_UNCERTAIN_BAND = (0.2, 0.8)

EVIDENCE_STATISTICS = ("top_k_mean", "max", "mean")

# Analysis modes, cheapest first.
MODE_OFF = "off"  # no patch scoring at all
MODE_COARSE = "coarse"  # one small grid
MODE_FULL = "full"  # one dense grid, capped by max_patches
MODE_TOP_K = "top_k"  # coarse grid, then refine the top-k regions
MODE_UNCERTAIN_ONLY = "uncertain_only"  # run base_mode only for undecided images
PATCH_MODES = (MODE_OFF, MODE_COARSE, MODE_FULL, MODE_TOP_K, MODE_UNCERTAIN_ONLY)


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
    mode: str = MODE_FULL
    forward_passes: int = 0
    reused_scores: int = 0
    seconds: float = 0.0
    peak_memory_mb: Optional[float] = None

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
            "mode": self.mode,
            "forward_passes": self.forward_passes,
            "reused_scores": self.reused_scores,
            "seconds": round(float(self.seconds), 4),
            "peak_memory_mb": (
                None if self.peak_memory_mb is None else round(float(self.peak_memory_mb), 2)
            ),
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
        "mode": str(section.get("mode", MODE_FULL)),
        "base_mode": str(section.get("base_mode", MODE_FULL)),
        "coarse_max_patches": int(section.get("coarse_max_patches", DEFAULT_COARSE_MAX_PATCHES)),
        "refine_factor": int(section.get("refine_factor", DEFAULT_REFINE_FACTOR)),
        "uncertain_band": tuple(
            float(v) for v in section.get("uncertain_band", DEFAULT_UNCERTAIN_BAND)
        ),
        "batch_size": section.get("batch_size"),
        "device": section.get("device"),
        "patch_size": int(section.get("patch_size", DEFAULT_PATCH_SIZE)),
        "stride": int(section.get("stride", DEFAULT_STRIDE)),
        "min_patch_size": int(section.get("min_patch_size", DEFAULT_MIN_PATCH_SIZE)),
        "top_k": int(section.get("top_k", DEFAULT_TOP_K)),
        "heatmap_threshold": float(section.get("heatmap_threshold", DEFAULT_HEATMAP_THRESHOLD)),
        "grid": section.get("grid"),
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
    if settings["grid"] is not None:
        grid = int(settings["grid"])
        if grid < 1:
            raise ValueError(f"patches.grid must be at least 1, got {grid}")
        if grid * grid > settings["max_patches"]:
            raise ValueError(
                f"patches.grid={grid} needs {grid * grid} forward passes but "
                f"patches.max_patches is {settings['max_patches']}. Raise "
                f"max_patches to at least {grid * grid}, or lower the grid."
            )
        settings["grid"] = grid
    if not 0.0 <= settings["heatmap_threshold"] <= 1.0:
        raise ValueError("patches.heatmap_threshold must be within [0, 1]")
    if settings["evidence_statistic"] not in EVIDENCE_STATISTICS:
        raise ValueError(
            f"patches.evidence_statistic must be one of {', '.join(EVIDENCE_STATISTICS)}"
        )
    if settings["mode"] not in PATCH_MODES:
        raise ValueError(f"patches.mode must be one of {', '.join(PATCH_MODES)}")
    if settings["base_mode"] not in (MODE_COARSE, MODE_FULL, MODE_TOP_K):
        raise ValueError(
            f"patches.base_mode must be one of {MODE_COARSE}, {MODE_FULL}, {MODE_TOP_K}"
        )
    if settings["coarse_max_patches"] <= 0:
        raise ValueError("patches.coarse_max_patches must be positive")
    if settings["refine_factor"] < 1:
        raise ValueError("patches.refine_factor must be at least 1")
    band = settings["uncertain_band"]
    if len(band) != 2 or not 0.0 <= band[0] <= band[1] <= 1.0:
        raise ValueError("patches.uncertain_band must be an ascending pair within [0, 1]")
    if settings["batch_size"] is not None and int(settings["batch_size"]) <= 0:
        raise ValueError("patches.batch_size must be positive when set")
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


def generate_grid_boxes(
    width: int,
    height: int,
    grid: int,
    min_patch_size: int = DEFAULT_MIN_PATCH_SIZE,
) -> List[Tuple[int, int, int, int]]:
    """Split the image into exactly ``grid x grid`` tiles covering all of it.

    The alternative -- fixed-size square tiles -- makes the region count depend
    on the image's aspect ratio: a 4:3 photo tiles 4x3, a square one 4x4. That
    is fine for the model but surprising in a UI, where "4x4" should mean
    sixteen regions whatever the picture.

    Tiles here are not square on a non-square image. That costs nothing in
    practice: preprocessing resizes every crop to the model's square input
    anyway, exactly as it already does to the whole image.

    Remainder pixels go to the last row and column, so coverage is complete and
    no strip along the right or bottom edge goes unscored.
    """

    grid = int(grid)
    if grid < 1:
        raise ValueError(f"patches.grid must be at least 1, got {grid}")
    cell_width = width // grid
    cell_height = height // grid
    if min(cell_width, cell_height) < min_patch_size:
        return []

    boxes: List[Tuple[int, int, int, int]] = []
    for row in range(grid):
        y = row * cell_height
        # The final row and column absorb the remainder rather than leaving a
        # sliver of the image unmeasured.
        box_height = height - y if row == grid - 1 else cell_height
        for column in range(grid):
            x = column * cell_width
            box_width = width - x if column == grid - 1 else cell_width
            boxes.append((x, y, box_width, box_height))
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
    """Score patches and reconstruct a risk heatmap, under the configured mode.

    Never raises: any failure is reported as an unavailable :class:`PatchReport`
    so the surrounding pipeline keeps its whole-image result.
    """

    started = time.time()
    try:
        settings = patch_settings(config)
    except ValueError as exc:
        return PatchReport(available=False, message=f"Invalid patch configuration: {exc}")

    mode, skip_reason = resolve_mode(settings, whole_image_probability)
    if mode == MODE_OFF:
        return PatchReport(
            available=False,
            message=skip_reason or "Patch-level analysis is disabled in the configuration.",
            settings=settings,
            mode=MODE_OFF,
            seconds=time.time() - started,
        )

    rgb = image.convert("RGB")
    width, height = rgb.size

    grid_mode = MODE_COARSE if mode == MODE_TOP_K else mode
    boxes = _boxes_for_mode(grid_mode, settings, width, height)
    if not boxes:
        return PatchReport(
            available=False,
            message=(
                f"Image is {width}x{height}, smaller than the "
                f"{settings['min_patch_size']}px minimum patch size, so patch-level "
                f"analysis was skipped. The whole-image result is unaffected."
            ),
            settings=settings,
            mode=mode,
            seconds=time.time() - started,
        )
    if len(boxes) < 2 and mode != MODE_TOP_K:
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
            mode=mode,
            seconds=time.time() - started,
        )

    batch_size = settings["batch_size"] or int(
        (config or {}).get("inference", {}).get("batch_size", 8)
    )
    scorer = PatchScorer(bundle, preprocessor, rgb, int(batch_size), whole_image_probability)

    try:
        raw = scorer.score(boxes)

        if mode == MODE_TOP_K:
            # Second pass: refine only the most suspicious coarse regions, so
            # detail is bought where it might matter instead of everywhere.
            ranked = sorted(zip(boxes, raw), key=lambda pair: pair[1], reverse=True)
            focus = [box for box, _ in ranked[: settings["top_k"]]]
            children = refine_boxes(
                focus, settings["refine_factor"], settings["min_patch_size"], width, height
            )
            children = children[: max(0, settings["max_patches"] - len(boxes))]
            if children:
                boxes = list(boxes) + children
                raw = list(raw) + scorer.score(children)

        scores = (
            calibrator.transform(np.asarray(raw)) if calibrator is not None else np.asarray(raw)
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
            mode=mode,
            seconds=time.time() - started,
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
            f"Scored {len(patches)} overlapping patches ({mode} mode). Warmer regions are "
            f"suspicious regions that influenced the model, not proof that a region was "
            f"edited and not a reconstruction of any editing history."
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
        mode=mode,
        forward_passes=scorer.forward_passes,
        reused_scores=scorer.reused,
        seconds=time.time() - started,
        peak_memory_mb=_peak_memory_mb(bundle.device),
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


# ---------------------------------------------------------------------------
# Cost instrumentation
# ---------------------------------------------------------------------------


def _peak_memory_mb(device: Any) -> Optional[float]:
    """Best-effort peak allocation for the active device, in MB."""

    try:
        kind = getattr(device, "type", str(device))
        if kind == "cuda":
            return float(torch.cuda.max_memory_allocated()) / 1e6
        if kind == "mps" and hasattr(torch, "mps"):
            current = getattr(torch.mps, "current_allocated_memory", None)
            if current is not None:
                return float(current()) / 1e6
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes, macOS reports bytes.
        return float(usage) / (1e6 if usage > 1e9 else 1e3)
    except (RuntimeError, ValueError, ImportError, AttributeError):  # pragma: no cover
        return None


class PatchScorer:
    """Scores patch boxes with batching, caching and whole-image reuse.

    Two savings matter. A box covering the whole image is the whole-image score
    already computed upstream, so it is reused rather than recomputed. And any
    box requested twice -- which the two-stage ``top_k`` mode can do -- is
    served from the cache. Both are counted so the ablation can show the work
    actually avoided.
    """

    def __init__(
        self,
        bundle: ModelBundle,
        preprocessor: Preprocessor,
        image: Image.Image,
        batch_size: int,
        whole_image_probability: Optional[float] = None,
    ) -> None:
        self.bundle = bundle
        self.preprocessor = preprocessor
        self.image = image
        self.batch_size = max(1, int(batch_size))
        self.whole_image_probability = whole_image_probability
        self._cache: Dict[Tuple[int, int, int, int], float] = {}
        self.forward_passes = 0
        self.reused = 0

    def _covers_whole_image(self, box: Tuple[int, int, int, int]) -> bool:
        width, height = self.image.size
        x, y, w, h = box
        return x == 0 and y == 0 and w == width and h == height

    def score(self, boxes: Sequence[Tuple[int, int, int, int]]) -> List[float]:
        """Return one probability per box, running the model only where needed."""

        pending: List[Tuple[int, int, int, int]] = []
        for box in boxes:
            if box in self._cache:
                continue
            if self.whole_image_probability is not None and self._covers_whole_image(box):
                # Identical to the whole-image forward pass already performed.
                self._cache[box] = float(self.whole_image_probability)
                self.reused += 1
                continue
            pending.append(box)

        if pending:
            # predict_images batches internally and runs under inference_mode.
            crops = [self.image.crop((x, y, x + w, y + h)) for x, y, w, h in pending]
            scores = predict_images(
                self.bundle, crops, self.preprocessor, batch_size=self.batch_size
            )
            self.forward_passes += len(pending)
            for box, score in zip(pending, scores):
                self._cache[box] = float(score)

        return [self._cache[box] for box in boxes]


def refine_boxes(
    boxes: Sequence[Tuple[int, int, int, int]],
    factor: int,
    min_patch_size: int,
    width: int,
    height: int,
) -> List[Tuple[int, int, int, int]]:
    """Subdivide each box into ``factor x factor`` children.

    Used by ``top_k`` mode to spend a second, cheaper pass only where the coarse
    pass already found signal, instead of refining the whole image.
    """

    if factor < 2:
        return []
    children: List[Tuple[int, int, int, int]] = []
    for x, y, box_width, box_height in boxes:
        child_width = box_width // factor
        child_height = box_height // factor
        if min(child_width, child_height) < min_patch_size:
            continue
        for row in range(factor):
            for column in range(factor):
                cx = min(x + column * child_width, width - child_width)
                cy = min(y + row * child_height, height - child_height)
                children.append((cx, cy, child_width, child_height))
    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for box in children:
        if box not in seen:
            seen.add(box)
            unique.append(box)
    return unique


def _boxes_for_mode(
    mode: str, settings: Dict[str, Any], width: int, height: int
) -> List[Tuple[int, int, int, int]]:
    """Grid for a single-pass mode."""

    # An explicit grid wins over the sliding window: the caller asked for a
    # fixed number of regions, so neither the stride nor the cap should change
    # it. patch_settings has already checked it fits within max_patches.
    if settings.get("grid"):
        return generate_grid_boxes(width, height, settings["grid"], settings["min_patch_size"])

    max_patches = settings["coarse_max_patches"] if mode == MODE_COARSE else settings["max_patches"]
    stride = (
        settings["stride"]
        if mode != MODE_COARSE
        else max(settings["stride"], settings["patch_size"])
    )
    return generate_patch_boxes(
        width,
        height,
        patch_size=settings["patch_size"],
        stride=stride,
        min_patch_size=settings["min_patch_size"],
        max_patches=max_patches,
    )


def resolve_mode(
    settings: Dict[str, Any], whole_image_probability: Optional[float]
) -> Tuple[str, Optional[str]]:
    """Resolve the effective mode, returning ``(mode, skip_reason)``.

    ``uncertain_only`` is the early-stopping path: an image the whole-image
    model already calls confidently gains little from patch scoring, so the
    patches are skipped entirely and the forward passes are never spent.
    """

    mode = settings["mode"]
    if not settings["enabled"] or mode == MODE_OFF:
        return MODE_OFF, "Patch-level analysis is disabled in the configuration."

    if mode == MODE_UNCERTAIN_ONLY:
        low, high = settings["uncertain_band"]
        if whole_image_probability is None:
            return settings["base_mode"], None
        if not low <= float(whole_image_probability) <= high:
            return MODE_OFF, (
                f"Whole-image score {float(whole_image_probability):.3f} is outside the "
                f"uncertain band [{low:.2f}, {high:.2f}], so patch analysis was skipped "
                f"(early stop). The whole-image result is unaffected."
            )
        return settings["base_mode"], None

    return mode, None
