"""Shared fixtures for the test suite.

Everything here is synthetic. Tests that need a model use an **untrained**
checkpoint built on the fly: they verify the plumbing (shapes, ranges, error
handling, JSON contracts), never detection accuracy. Real inference quality can
only be measured with a real trained checkpoint via
``scripts/evaluate_dataset.py``.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ResNet-18 is used for mock checkpoints because a randomly initialised
# EfficientNet-B0 collapses to a constant output in eval mode, which would mask
# genuine pipeline bugs.
MOCK_ARCHITECTURE = "resnet18"


def requires(*modules: str) -> Any:
    """Skip a test when any of the named modules is not installed."""

    import importlib

    missing = []
    for name in modules:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    return unittest.skipIf(
        bool(missing), f"missing dependencies: {', '.join(missing)}"
    )


def make_image(
    width: int = 96,
    height: int = 64,
    seed: int = 0,
    mode: str = "RGB",
) -> Image.Image:
    """Create a deterministic synthetic image with real structure.

    Structured content matters: a flat colour survives blur and JPEG unchanged,
    which would make the transformation tests vacuous.
    """

    rng = np.random.default_rng(seed)
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    base = (
        128
        + 60 * np.sin(x_grid / 4.0 + seed)
        + 40 * np.cos(y_grid / 3.0)
        + rng.normal(0, 25, (height, width))
    )
    stacked = np.stack(
        [base, np.roll(base, 5, axis=1), np.roll(base, 3, axis=0)], axis=-1
    )
    image = Image.fromarray(np.clip(stacked, 0, 255).astype(np.uint8), mode="RGB")
    return image if mode == "RGB" else image.convert(mode)


def write_image(
    directory: Path,
    name: str = "sample.jpg",
    width: int = 96,
    height: int = 64,
    seed: int = 0,
    image_format: Optional[str] = None,
) -> Path:
    """Write a synthetic image to disk and return its path."""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    image = make_image(width=width, height=height, seed=seed)
    image.save(path, format=image_format)
    return path


def write_corrupted_image(directory: Path, name: str = "corrupt.png") -> Path:
    """Write a file with a valid PNG magic number but a garbage body."""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"not actually png data" * 8)
    return path


def write_truncated_jpeg(directory: Path, name: str = "truncated.jpg") -> Path:
    """Write a JPEG whose data stops half way through."""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    full = directory / f".full_{name}"
    make_image(seed=3).save(full, format="JPEG")
    data = full.read_bytes()
    full.unlink()
    path.write_bytes(data[: len(data) // 2])
    return path


def base_config(image_size: int = 224) -> Dict[str, Any]:
    """A minimal config mirroring configs/config.yaml, for fast tests."""

    return {
        "data": {"image_size": image_size, "extensions": [".jpg", ".jpeg", ".png", ".webp"]},
        "validation": {
            "min_side": 16,
            "max_side": 20000,
            "max_pixels": 50_000_000,
            "max_file_size_mb": 64,
        },
        "model": {"name": MOCK_ARCHITECTURE, "num_classes": 1, "ai_class_index": 1},
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "transformations": {
            "enabled": True,
            "seed": 1234,
            "jpeg_qualities": [90, 70, 50, 30],
            "blur_sigmas": [0.5, 1.0, 2.0],
            "resize_scales": [0.5, 0.25],
            "noise_sigmas": [0.02, 0.05, 0.10],
            "color_jitter": {
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "variants": 1,
            },
            "center_crop_fraction": 0.8,
        },
        "labels": {"authentic_max": 0.40, "ai_min": 0.60},
        "consistency": {
            "std_reference": 0.20,
            "range_reference": 0.60,
            "std_weight": 0.5,
            "range_weight": 0.5,
        },
        "fusion": {
            "mode": "rgb_transform",
            "original_weight": 0.7,
            "transform_weight": 0.3,
            "frequency": {
                "rgb_weight": 0.5,
                "frequency_weight": 0.3,
                "consistency_weight": 0.2,
            },
        },
        "frequency": {"enabled": False, "checkpoint": None, "fft_bins": 8},
        "confidence": {
            "decisiveness_weight": 0.4,
            "agreement_weight": 0.3,
            "consistency_weight": 0.3,
            "high_min": 0.70,
            "medium_min": 0.45,
        },
        "explainability": {"enabled": True, "gradcam": True, "overlay_alpha": 0.45},
        "inference": {"batch_size": 8, "threshold": 0.5, "device": "cpu"},
    }


def write_mock_checkpoint(
    path: Path,
    architecture: str = MOCK_ARCHITECTURE,
    num_classes: int = 1,
    wrap_key: Optional[str] = "model_state_dict",
) -> Path:
    """Save an UNTRAINED checkpoint used purely to exercise the plumbing."""

    import torch

    from src.pipeline.model_loader import build_architecture

    torch.manual_seed(0)
    model = build_architecture(architecture, num_classes=num_classes, pretrained=False)
    state_dict = model.state_dict()
    payload: Any = state_dict
    if wrap_key:
        payload = {
            wrap_key: state_dict,
            "model_name": architecture,
            "num_classes": num_classes,
            "trained": False,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path
