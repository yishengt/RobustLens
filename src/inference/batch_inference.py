"""Batch inference over a directory of images.

Produces two JSON views:

* the **simple** submission format, ``[{"image_path": ..., "pred": ...}]``,
  where ``pred`` is the AI-generated probability;
* an optional **detailed** format that also carries the label, confidence,
  real-image probability, transformation consistency, per-transformation
  predictions and any errors.

The model is loaded once. An image that fails validation or inference is
recorded and skipped rather than ending the run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.pipeline.model_loader import ModelSetupError, load_model
from src.pipeline.pipeline import DetectionPipeline, FailedResult, PipelineResult
from src.pipeline.validation import ImageValidationError, list_supported_images

ON_ERROR_SKIP = "skip"
ON_ERROR_FALLBACK = "fallback"


@dataclass
class BatchReport:
    """The outcome of a batch run."""

    simple: List[Dict[str, Any]] = field(default_factory=list)
    detailed: List[Dict[str, Any]] = field(default_factory=list)
    processed: int = 0
    failed: int = 0
    total: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        text = f"Processed {self.processed}/{self.total} images"
        if self.failed:
            text += f"; {self.failed} failed"
        return text


def write_json(payload: Any, output_path: str | Path) -> Path:
    """Write JSON to disk, creating the parent directory if needed."""

    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


def run_batch(
    image_dir: str | Path,
    checkpoint_path: str | Path,
    config: Optional[Dict[str, Any]] = None,
    output_path: Optional[str | Path] = None,
    detailed_output_path: Optional[str | Path] = None,
    device: Optional[str] = None,
    limit: Optional[int] = None,
    on_error: str = ON_ERROR_SKIP,
    fallback_pred: float = 0.5,
    explain_images: bool = False,
    relative_to: Optional[str | Path] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> BatchReport:
    """Analyse every supported image in ``image_dir`` and write the JSON output.

    ``on_error='skip'`` (the default) leaves failed images out of the simple
    output; ``on_error='fallback'`` records ``fallback_pred`` for them so the
    file still covers every input image. Failures always appear in the detailed
    output with their error messages.
    """

    config = config or {}
    if on_error not in (ON_ERROR_SKIP, ON_ERROR_FALLBACK):
        raise ValueError(
            f"on_error must be '{ON_ERROR_SKIP}' or '{ON_ERROR_FALLBACK}', got '{on_error}'"
        )
    if not 0.0 <= float(fallback_pred) <= 1.0:
        raise ValueError(f"fallback_pred must be within [0, 1], got {fallback_pred}")

    directory = Path(image_dir).expanduser()
    image_paths: Sequence[Path] = list_supported_images(directory, config)
    if not image_paths:
        raise ImageValidationError(
            f"No supported images found in {directory}. Supported types: "
            f".jpg, .jpeg, .png, .webp"
        )
    if limit is not None and limit > 0:
        image_paths = image_paths[: int(limit)]

    # Explainability is off by default in batch mode: Grad-CAM needs a backward
    # pass per image and nothing consumes the heatmap here.
    pipeline = DetectionPipeline.from_checkpoint(
        checkpoint_path, config, device=device, explain_images=explain_images
    )

    root = Path(relative_to).expanduser() if relative_to else None
    report = BatchReport(total=len(image_paths))

    for index, path in enumerate(image_paths, start=1):
        display_path = str(path)
        if root is not None:
            try:
                display_path = str(path.relative_to(root))
            except ValueError:
                display_path = str(path)

        result = pipeline.safe_analyse_path(path)

        if isinstance(result, PipelineResult):
            result.image_path = display_path
            report.simple.append(result.as_simple_dict())
            report.detailed.append(result.as_detailed_dict())
            report.processed += 1
        else:
            failure: FailedResult = result
            failure.image_path = display_path
            report.failed += 1
            report.errors.extend(
                {"image_path": display_path, **error} for error in failure.errors
            )
            detailed = failure.as_detailed_dict()
            if on_error == ON_ERROR_FALLBACK:
                detailed["pred"] = float(fallback_pred)
                detailed["label"] = "Error (fallback score)"
                report.simple.append(
                    {"image_path": display_path, "pred": float(fallback_pred)}
                )
            report.detailed.append(detailed)

        if progress is not None:
            progress(index, report.total, display_path)

    if output_path is not None:
        write_json(report.simple, output_path)
    if detailed_output_path is not None:
        write_json(report.detailed, detailed_output_path)
    return report


__all__ = [
    "BatchReport",
    "ModelSetupError",
    "ON_ERROR_FALLBACK",
    "ON_ERROR_SKIP",
    "load_model",
    "run_batch",
    "write_json",
]
