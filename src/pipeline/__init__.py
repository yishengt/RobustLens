"""Inference pipeline for robust AI-generated image detection.

Stage order::

    Input image -> Validation -> Preprocessing -> Transformation generation ->
    Feature extraction -> Classification -> Per-version predictions ->
    Consistency check -> Fusion -> Confidence -> Explainability -> Output

Submodules are imported lazily so that light stages (for example
``validation``, which only needs Pillow) stay usable in environments where the
heavier torch stack is not installed.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DetectionPipeline",
    "PipelineResult",
    "ImageValidationError",
    "ModelSetupError",
    "load_model",
]

_LAZY_EXPORTS = {
    "DetectionPipeline": ("src.pipeline.pipeline", "DetectionPipeline"),
    "PipelineResult": ("src.pipeline.pipeline", "PipelineResult"),
    "ImageValidationError": ("src.pipeline.validation", "ImageValidationError"),
    "ModelSetupError": ("src.pipeline.model_loader", "ModelSetupError"),
    "load_model": ("src.pipeline.model_loader", "load_model"),
}


def __getattr__(name: str) -> Any:
    """Resolve public names on first use to keep import side effects small."""

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_name, attribute = _LAZY_EXPORTS[name]
    return getattr(importlib.import_module(module_name), attribute)


def __dir__() -> list[str]:
    return sorted(__all__)
