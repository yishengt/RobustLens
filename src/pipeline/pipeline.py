"""End-to-end orchestration of the inference pipeline.

Runs, in order::

    Input image -> Validation -> Preprocessing -> Whole-image detection ->
    Patch-level detection -> Transformation testing -> Fusion -> Confidence ->
    Explainability -> Output

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
from src.pipeline.consistency import (
    ConsistencyReport,
    compute_consistency,
    estimate_manipulation_severity,
)
from src.pipeline.explainability import ExplanationResult, build_charts, explain
from src.pipeline.fusion import FusionResult, fuse_predictions
from src.pipeline.model_loader import ModelBundle, load_model
from src.pipeline.patches import PatchReport, analyse_patches
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
    patches: Optional[PatchReport] = None
    manipulation_severity: str = "low"
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
            # --- documented report schema -------------------------------------
            # `pred` above stays the submission field; these mirror it under the
            # names used in the README report schema, plus the patch findings.
            "final_probability": round(float(self.ai_probability), 6),
            "confidence_level": self.confidence.level.lower(),
            "transformation_consistency": round(self.consistency.consistency_score, 6),
            "estimated_manipulation_severity": self.manipulation_severity,
            "highest_risk_region": (
                self.patches.highest_risk_region if self.patches is not None else None
            ),
            "per_transformation_predictions": self.per_transformation_predictions(),
            "patch_analysis": self.patches.as_dict() if self.patches is not None else None,
        }

    def per_transformation_predictions(self) -> Dict[str, float]:
        """Per-version scores with the untransformed image reported as ``clean``."""

        scores: Dict[str, float] = {}
        for item in self.predictions:
            name = "clean" if item.is_original else item.name
            scores[name] = round(float(item.ai_probability), 6)
        return scores

    def as_report_dict(self) -> Dict[str, Any]:
        """The compact human-facing report described in the README."""

        original, _ = split_predictions(self.predictions)
        raw = original.raw_probability if original else None
        return {
            "image_path": self.image_path,
            "raw_probability": (
                round(float(raw), 6) if raw is not None else round(float(self.ai_probability), 6)
            ),
            "final_probability": round(float(self.ai_probability), 6),
            "real_probability": round(float(self.real_probability), 6),
            "label": self.label,
            "confidence": self.confidence.level.lower(),
            "transformation_consistency": round(self.consistency.consistency_score, 6),
            "estimated_manipulation_severity": self.manipulation_severity,
            "highest_risk_region": (
                self.patches.highest_risk_region if self.patches is not None else None
            ),
            "per_transformation_predictions": self.per_transformation_predictions(),
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
            "final_probability": None,
            "confidence_level": "none",
            "transformation_consistency": None,
            "estimated_manipulation_severity": None,
            "highest_risk_region": None,
            "per_transformation_predictions": {},
            "patch_analysis": None,
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
        self.calibration_error: Optional[str] = None
        self.calibrator = self._load_calibrator(self.config)
        if self.calibrator and self.calibrator.selected_thresholds:
            calibration = self.config.get("calibration", {}) or {}
            if calibration.get("use_selected_threshold", True):
                # Which fitted operating point to run at. All four are stored by
                # scripts/calibrate_threshold.py; "balanced" maximises Youden's J.
                point = str(calibration.get("operating_point", "balanced"))
                selected = self.calibrator.selected_thresholds.get(point)
                if selected is None:
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

    def _load_calibrator(self, config: Dict[str, Any]) -> Optional[ProbabilityCalibrator]:
        """Load optional persisted calibration parameters for normal inference.

        A missing or unreadable calibration file must never stop inference. The
        parameters live under outputs/ which is git-ignored, so a fresh clone
        with calibration enabled would otherwise fail to run at all. Instead the
        pipeline falls back to uncalibrated scores and records why, which
        calibration_status() then reports plainly.
        """

        self.calibration_error: Optional[str] = None
        calibration = config.get("calibration", {}) or {}
        if not calibration.get("enabled", False):
            return None
        path = calibration.get("path")
        if not path:
            self.calibration_error = (
                "calibration.enabled is true but no calibration.path is set."
            )
            return None
        source = Path(str(path)).expanduser()
        if not source.is_absolute() and config.get("_project_root"):
            source = Path(config["_project_root"]) / source
        try:
            return ProbabilityCalibrator.load(source)
        except (FileNotFoundError, ValueError) as exc:
            self.calibration_error = (
                f"{exc} Falling back to UNCALIBRATED scores. Fit calibration with "
                f"scripts/calibrate_threshold.py, or set calibration.enabled to false."
            )
            return None

    def calibration_status(self) -> Dict[str, Any]:
        """Describe exactly what the reported numbers are.

        Four states must never be blurred together: a calibrated probability
        versus a raw model score, and a threshold fitted on labelled data
        versus the interface default. Presenting an uncalibrated score beside a
        default threshold as though both were derived from data would overstate
        what the system knows.
        """

        calibration = self.config.get("calibration", {}) or {}
        default_threshold = 0.5
        threshold = float(self.config.get("inference", {}).get("threshold", default_threshold))
        labels = self.config.get("labels", {}) or {}

        if self.calibrator is None:
            return {
                "calibrated": False,
                "probability_kind": "uncalibrated model score",
                "probability_note": (
                    self.calibration_error
                    or "No calibration parameters are loaded, so this is the model's raw "
                    "score. It ranks images but is not a statistically calibrated "
                    "probability and is not comparable across checkpoints."
                ),
                "calibration_error": self.calibration_error,
                "method": None,
                "temperature": None,
                "threshold": threshold,
                "threshold_source": "interface default",
                "threshold_note": (
                    "This threshold and the label bands are interface defaults, not "
                    "values derived from labelled validation data."
                ),
                "operating_point": None,
                "calibration_path": calibration.get("path"),
                "label_bands": {
                    "authentic_max": labels.get("authentic_max"),
                    "ai_min": labels.get("ai_min"),
                },
                "label_bands_source": "interface default",
            }

        used_selection = bool(
            calibration.get("use_selected_threshold", True) and self.calibrator.selected_thresholds
        )
        point = str(calibration.get("operating_point", "balanced"))
        return {
            "calibrated": True,
            "probability_kind": "calibrated probability",
            "probability_note": (
                f"Probabilities were calibrated with {self.calibrator.method} scaling "
                f"fitted on {self.calibrator.fitted_on.replace('_', ' ')} data."
            ),
            "method": self.calibrator.method,
            "temperature": self.calibrator.temperature,
            "threshold": threshold,
            "threshold_source": (
                f"data-derived ({point} operating point)" if used_selection else "interface default"
            ),
            "threshold_note": (
                f"Selected on clean validation data at the {point} operating point and "
                f"frozen for every condition."
                if used_selection
                else "A calibrator is loaded but its selected thresholds were not applied."
            ),
            "operating_point": point if used_selection else None,
            "calibration_path": calibration.get("path"),
            "label_bands": {
                "authentic_max": labels.get("authentic_max"),
                "ai_min": labels.get("ai_min"),
            },
            "label_bands_source": (
                "data-derived (threshold +/- uncertainty margin)"
                if used_selection
                else "interface default"
            ),
        }

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

        # Stage 4b: patch-level detection. Runs on the ORIGINAL image only --
        # patching every transformed version would multiply the cost by 15 for
        # no extra localisation information. Never raises.
        patch_report = analyse_patches(
            self.bundle,
            original_image,
            self.preprocessor,
            self.config,
            whole_image_probability=original_prediction.ai_probability,
            calibrator=self.calibrator,
        )
        if not patch_report.available and patch_report.settings.get("enabled", True):
            errors.append({"stage": "patches", "error": patch_report.message})

        # Stage 7: how stable is the model across those versions?
        consistency = compute_consistency(predictions, self.config)
        manipulation_severity = estimate_manipulation_severity(consistency, self.config)

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
            patch_evidence=patch_report.evidence if patch_report.available else None,
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
            patch_agreement=patch_report.agreement if patch_report.available else None,
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
            patches=patch_report,
            manipulation_severity=manipulation_severity,
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
