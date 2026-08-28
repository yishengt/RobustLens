"""Configurable image transformations used by the inference pipeline."""

from __future__ import annotations

from typing import Any, Dict, List

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2


def _jpeg_transform(quality: int) -> A.BasicTransform:
    """Build an ImageCompression transform across Albumentations API versions."""

    try:
        return A.ImageCompression(
            quality_range=(int(quality), int(quality)),
            compression_type="jpeg",
            p=1.0,
        )
    except TypeError:
        # Albumentations < 2.0 used quality_lower/quality_upper.
        return A.ImageCompression(
            quality_lower=int(quality),
            quality_upper=int(quality),
            p=1.0,
        )


def _blur_transform(sigma: float) -> A.BasicTransform:
    try:
        return A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(sigma, sigma), p=1.0)
    except TypeError:
        return A.GaussianBlur(blur_limit=(3, 3), sigma_limit=sigma, p=1.0)


def _resize_down_up(image: np.ndarray, scale: float, **_: Any) -> np.ndarray:
    """Downscale an image by ``scale`` and upscale it back to its input size."""

    height, width = image.shape[:2]
    down_width = max(1, int(round(width * scale)))
    down_height = max(1, int(round(height * scale)))
    down = cv2.resize(image, (down_width, down_height), interpolation=cv2.INTER_AREA)
    return cv2.resize(down, (width, height), interpolation=cv2.INTER_LINEAR)


def _gaussian_noise(image: np.ndarray, sigma: float, **_: Any) -> np.ndarray:
    """Add zero-mean Gaussian noise where sigma is a fraction of 0-255."""

    noise = np.random.normal(0.0, float(sigma) * 255.0, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)


def _lambda_image(function: Any) -> A.BasicTransform:
    return A.Lambda(image=function, p=1.0)


def _normalization(config: Dict[str, Any]) -> List[A.BasicTransform]:
    normalization = config.get("normalization", {})
    return [
        A.Normalize(
            mean=normalization.get("mean", [0.485, 0.456, 0.406]),
            std=normalization.get("std", [0.229, 0.224, 0.225]),
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]


def _final_pipeline(
    config: Dict[str, Any],
    transforms: List[A.BasicTransform],
    include_tensor: bool = True,
) -> A.Compose:
    image_size = int(config.get("data", {}).get("image_size", 224))
    tail = _normalization(config) if include_tensor else []
    return A.Compose(
        [A.Resize(height=image_size, width=image_size, p=1.0), *transforms, *tail]
    )


def build_eval_transform(config: Dict[str, Any]) -> A.Compose:
    """Build the clean deterministic validation/inference pipeline."""

    return _final_pipeline(config, [])


def _robustness_transforms(config: Dict[str, Any], case: str) -> List[A.BasicTransform]:
    """Return the pixel-space transforms for one configured robustness case."""

    settings = config.get("augmentations", {}).get("robustness", {})
    if case == "clean":
        return []

    transforms: List[A.BasicTransform] = []
    if case.startswith("jpeg_"):
        transforms.append(_jpeg_transform(int(case.split("_", 1)[1])))
    elif case.startswith("blur_"):
        transforms.append(_blur_transform(float(case.split("_", 1)[1])))
    elif case.startswith("resize_"):
        scale = float(case.split("_", 1)[1].rstrip("x"))
        transforms.append(
            _lambda_image(lambda image, scale=scale, **kwargs: _resize_down_up(image, scale, **kwargs))
        )
    elif case.startswith("noise_"):
        sigma = float(case.split("_", 1)[1])
        transforms.append(
            _lambda_image(lambda image, sigma=sigma, **kwargs: _gaussian_noise(image, sigma, **kwargs))
        )
    elif case == "color_jitter":
        limit = float(settings.get("color_jitter_limit", 0.2))
        transforms.append(
            A.ColorJitter(
                brightness=limit,
                contrast=limit,
                saturation=limit,
                hue=0.0,
                p=1.0,
            )
        )
    elif case == "center_crop_80":
        image_size = int(config.get("data", {}).get("image_size", 224))
        fraction = float(settings.get("center_crop_fraction", 0.8))
        crop_size = max(1, int(round(image_size * fraction)))
        transforms.extend(
            [
                A.CenterCrop(height=crop_size, width=crop_size, p=1.0),
                A.Resize(height=image_size, width=image_size, p=1.0),
            ]
        )
    else:
        valid = ", ".join(robustness_cases(config))
        raise ValueError(f"Unknown robustness case '{case}'. Valid cases: {valid}")
    return transforms


def build_robustness_transform(config: Dict[str, Any], case: str) -> A.Compose:
    """Build one robustness case for model evaluation.

    ``color_jitter`` and Gaussian noise remain stochastic; the configured seed
    controls their random draws.
    """

    if case == "clean":
        return build_eval_transform(config)
    return _final_pipeline(config, _robustness_transforms(config, case))


def build_robustness_image_transform(config: Dict[str, Any], case: str) -> A.Compose:
    """Build a robustness transform that returns uint8 pixels for saving to disk."""

    return _final_pipeline(
        config,
        _robustness_transforms(config, case),
        include_tensor=False,
    )


def robustness_cases(config: Dict[str, Any]) -> List[str]:
    """Return the configured clean and robustness case names."""

    settings = config.get("augmentations", {}).get("robustness", {})
    cases = ["clean"]
    cases.extend(f"jpeg_{int(q)}" for q in settings.get("jpeg_qualities", [90, 70, 50, 30]))
    cases.extend(f"blur_{float(sigma):g}" for sigma in settings.get("blur_sigmas", [0.5, 1.0, 2.0]))
    cases.extend(f"resize_{float(scale):g}x" for scale in settings.get("resize_scales", [0.5, 0.25]))
    cases.extend(f"noise_{float(sigma):g}" for sigma in settings.get("noise_sigmas", [0.02, 0.05, 0.10]))
    cases.extend(["color_jitter", "center_crop_80"])
    return cases
