"""End-to-end orchestration of the inference pipeline.

Runs, in order::

    Input image -> Validation -> Preprocessing -> Transformation generation ->
    Feature extraction -> Classification -> Per-version predictions ->
    Consistency check -> Fusion -> Confidence -> Explainability -> Output

The model is loaded once and reused for every image, so batch runs pay the
checkpoint cost a single time.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from src.evaluation.calibration import ProbabilityCalibrator
from src.pipeline import frequency as frequency_module
from src.pipeline.confidence import ConfidenceReport, compute_confidence
from src.pipeline.consistency import ConsistencyReport, compute_consistency
from src.pipeline.explainability import ExplanationResult, build_charts, explain
from src.pipeline.fusion import FusionResult, fuse_predictions
from src.pipeline.model_loader import ModelBundle, load_model
from src.pipeline.prediction import (
    Prediction,
    binary_threshold,
    label_for_probability,
    predict_variants,
    split_predictions,
)
from src.pipeline.preprocessing import Preprocessor, preserve_original
from src.pipeline.transformations import (
    ORIGINAL_KEY,
    build_transform_specs,
    generate_variants,
)
from src.pipeline.validation import (
    ImageMetadata,
    ImageValidationError,
    load_validated_image,
    validate_image_bytes,
)


@dataclass
class PipelineResult:
    """Everything one image analysis produced."""

    image_path: str
    ai_probability: float
    real_probability: float
    label: str
    confidence: ConfidenceReport
    consistency: ConsistencyReport
    fusion: FusionResult
    predictions: List[Prediction]
    threshold_used: float = 0.5
    metadata: Optional[ImageMetadata] = None
    explanation: Optional[ExplanationResult] = None
    frequency_features: Optional[Dict[str, Any]] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)
    original_image: Optional[Image.Image] = None

    # -- output views -------------------------------------------------------

    def as_simple_dict(self) -> Dict[str, Any]:
        """The required submission record: path plus AI-generated probability."""

        return {"image_path": self.image_path, "pred": round(float(self.ai_probability), 6)}

    def as_detailed_dict(self) -> Dict[str, Any]:
        """The full record, including per-transformation predictions."""

        original, transformed = split_predictions(self.predictions)
        raw_probability = (
            original.raw_probability if original and original.raw_probability is not None else None
        )
        return {
            "image_path": self.image_path,
            "pred": round(float(self.ai_probability), 6),
            "raw_probability": round(float(raw_probability), 6)
            if raw_probability is not None
            else None,
            "calibrated_probability": round(float(self.ai_probability), 6),
            "label": self.label,
            "confidence": self.confidence.level,
            "threshold_used": round(float(self.threshold_used), 6),
            "real_probability": round(float(self.real_probability), 6),
            "transform_consistency": round(self.consistency.consistency_score, 6),
            "transformations": {
                item.name: round(float(item.ai_probability), 6) for item in transformed
            },
            "errors": list(self.errors),
            "original_prediction": original.as_dict() if original else None,
            "predictions": [item.as_dict() for item in self.predictions],
            "consistency_detail": self.consistency.as_dict(),
            "fusion_detail": self.fusion.as_dict(),
            "confidence_detail": self.confidence.as_dict(),
            "metadata": self.metadata.as_dict() if self.metadata else None,
            "explainability": (
                self.explanation.as_dict() if self.explanation is not None else None
            ),
            "frequency": self.frequency_features,
        }


@dataclass
class FailedResult:
    """A record for an image that could not be analysed."""

    image_path: str
    errors: List[Dict[str, Any]]

    def as_detailed_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "pred": None,
            "label": "Error",
            "confidence": "None",
            "real_probability": None,
            "transform_consistency": None,
            "transformations": {},
            "errors": list(self.errors),
        }


class DetectionPipeline:
    """Runs the full inference pipeline for one image at a time."""

    def __init__(
        self,
        bundle: ModelBundle,
        config: Optional[Dict[str, Any]] = None,
        explain_images: bool = True,
    ):
        self.bundle = bundle
        self.config = deepcopy(config or {})
        self.preprocessor = Preprocessor.from_config(self.config)
        self.transform_specs = build_transform_specs(self.config)
        self.explain_images = explain_images
        self.calibrator = self._load_calibrator(self.config)
        if self.calibrator and self.calibrator.selected_thresholds:
            calibration = self.config.get("calibration", {}) or {}
            if calibration.get("use_selected_threshold", True):
                selected = self.calibrator.selected_thresholds.get("balanced")
                if selected is not None:
                    self.config.setdefault("inference", {})["threshold"] = float(selected)
                    margin = float(calibration.get("uncertainty_margin", 0.10))
                    if not 0.0 <= margin <= 1.0:
                        raise ValueError("calibration.uncertainty_margin must be within [0, 1]")
                    self.config.setdefault("labels", {})["authentic_max"] = max(
                        0.0, float(selected) - margin
                    )
                    self.config.setdefault("labels", {})["ai_min"] = min(
                        1.0, float(selected) + margin
                    )

    @staticmethod
    def _load_calibrator(config: Dict[str, Any]) -> Optional[ProbabilityCalibrator]:
        """Load optional persisted calibration parameters for normal inference."""

        calibration = config.get("calibration", {}) or {}
        if not calibration.get("enabled", False):
            return None
        path = calibration.get("path")
        if not path:
            return None
        source = Path(str(path)).expanduser()
        if not source.is_absolute() and config.get("_project_root"):
            source = Path(config["_project_root"]) / source
        return ProbabilityCalibrator.load(source)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        explain_images: bool = True,
    ) -> "DetectionPipeline":
        """Load a checkpoint and return a ready-to-use pipeline."""

        bundle = load_model(checkpoint_path, config, device=device)
        return cls(bundle, config, explain_images=explain_images)

    # -- entry points -------------------------------------------------------

    def analyse_path(self, path: str | Path) -> PipelineResult:
        """Validate and analyse an image file on disk."""

        image, metadata = load_validated_image(path, self.config)
        return self.analyse_image(image, metadata)

    def analyse_bytes(self, data: bytes, filename: str = "upload") -> PipelineResult:
        """Validate and analyse in-memory image bytes, as from an upload."""

        image, metadata = validate_image_bytes(data, filename, self.config)
        return self.analyse_image(image, metadata)

    def analyse_image(
        self, image: Image.Image, metadata: Optional[ImageMetadata] = None
    ) -> PipelineResult:
        """Run every pipeline stage on an already-validated RGB image."""

        errors: List[Dict[str, Any]] = []

        # Stage 3: keep a pristine full-resolution copy for display and overlays.
        original_image = preserve_original(image)

        # Stage 4: build the transformed versions from the ORIGINAL image, so
        # each one is preprocessed exactly like the original afterwards.
        variants, transform_errors = generate_variants(
            original_image, self.config, self.transform_specs
        )
        errors.extend(transform_errors)

        # Stages 5-6: feature extraction and classification for every version.
        predictions = predict_variants(
            self.bundle,
            variants,
            self.preprocessor,
            self.config,
            calibrator=self.calibrator,
        )
        original_prediction, transformed_predictions = split_predictions(predictions)
        if original_prediction is None:  # pragma: no cover - variants always include it
            raise RuntimeError("Pipeline lost the original image prediction")

        # Stage 7: how stable is the model across those versions?
        consistency = compute_consistency(predictions, self.config)

        # Stage 9 (optional): frequency features, and a frequency probability
        # only if a real frequency model is configured.
        frequency_features: Optional[Dict[str, Any]] = None
        frequency_probability: Optional[float] = None
        if frequency_module.is_enabled(self.config):
            try:
                frequency_features = frequency_module.extract_features(
                    original_image, self.config
                ).as_dict()
                frequency_probability, reason = frequency_module.frequency_probability(
                    original_image, self.config
                )
                if frequency_probability is None and reason:
                    errors.append({"stage": "frequency", "error": reason})
            except (ValueError, MemoryError, RuntimeError) as exc:
                errors.append({"stage": "frequency", "error": f"{type(exc).__name__}: {exc}"})

        # Stage 8: fuse the original and transformed predictions.
        fusion = fuse_predictions(
            original_prediction.ai_probability,
            [item.ai_probability for item in transformed_predictions],
            self.config,
            consistency=consistency.consistency_score,
            frequency_probability=frequency_probability,
        )
        final_probability = float(fusion.final_probability)
        label = label_for_probability(final_probability, self.config)

        # Stage 10: confidence.
        confidence = compute_confidence(
            final_probability,
            consistency.agreement,
            consistency.consistency_score,
            self.config,
            label=label,
        )

        # Stage 11: explainability, which can never abort the analysis.
        explanation: Optional[ExplanationResult] = None
        if self.explain_images:
            tensor = self.preprocessor(variants[ORIGINAL_KEY]).unsqueeze(0)
            explanation = explain(self.bundle, tensor, original_image, self.config)
            explanation.charts = build_charts(predictions, consistency, confidence)

        return PipelineResult(
            image_path=metadata.file_path if metadata else "<in-memory image>",
            ai_probability=final_probability,
            real_probability=float(1.0 - final_probability),
            label=label,
            confidence=confidence,
            consistency=consistency,
            fusion=fusion,
            predictions=predictions,
            threshold_used=binary_threshold(self.config),
            metadata=metadata,
            explanation=explanation,
            frequency_features=frequency_features,
            errors=errors,
            original_image=original_image,
        )

    def safe_analyse_path(self, path: str | Path) -> PipelineResult | FailedResult:
        """Analyse an image, returning a failure record instead of raising.

        Used by batch inference so a single unreadable file does not end the run.
        """

        try:
            return self.analyse_path(path)
        except ImageValidationError as exc:
            return FailedResult(
                image_path=str(path), errors=[{"stage": "validation", "error": str(exc)}]
            )
        except (RuntimeError, ValueError, OSError, MemoryError, TypeError) as exc:
            return FailedResult(
                image_path=str(path),
                errors=[{"stage": "inference", "error": f"{type(exc).__name__}: {exc}"}],
            )
