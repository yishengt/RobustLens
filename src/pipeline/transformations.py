"""Stage 4: real-world transformation generation.

Every transform is applied to the *original*, full-resolution image, mirroring
what happens when a picture is compressed, resized or re-encoded as it travels
across the internet. The transformed image is then fed through exactly the same
preprocessing and model pipeline as the original.

Transforms are driven by ``configs/config.yaml`` and are reproducible: the
stochastic ones (noise, colour jitter) derive their random state from a seed
plus the transform name, so a given image always yields the same variants.
"""

from __future__ import annotations

import io
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ORIGINAL_KEY = "original"

_RESAMPLE = getattr(Image, "Resampling", Image)
_DOWN = _RESAMPLE.BOX
_UP = _RESAMPLE.BICUBIC


@dataclass(frozen=True)
class TransformSpec:
    """A single named transformation and the parameters it was built with."""

    name: str
    family: str
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "params": dict(self.params),
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# Individual transformations. Each takes and returns a PIL RGB image.
# ---------------------------------------------------------------------------


def jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    """Re-encode the image as JPEG at the given quality and decode it back."""

    quality = int(quality)
    if not 1 <= quality <= 100:
        raise ValueError(f"JPEG quality must be between 1 and 100, got {quality}")
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    """Apply a Gaussian blur with the given standard deviation, in pixels."""

    sigma = float(sigma)
    if sigma < 0:
        raise ValueError(f"Blur sigma must be non-negative, got {sigma}")
    if sigma == 0:
        return image.copy()
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def downscale_upscale(image: Image.Image, scale: float) -> Image.Image:
    """Shrink by ``scale`` then stretch back, losing detail like a re-upload."""

    scale = float(scale)
    if not 0 < scale <= 1:
        raise ValueError(f"Resize scale must be in (0, 1], got {scale}")
    width, height = image.size
    small = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(small, _DOWN).resize((width, height), _UP)


def gaussian_noise(
    image: Image.Image, sigma: float, rng: Optional[np.random.Generator] = None
) -> Image.Image:
    """Add zero-mean Gaussian noise; ``sigma`` is a fraction of the 0-255 range."""

    sigma = float(sigma)
    if sigma < 0:
        raise ValueError(f"Noise sigma must be non-negative, got {sigma}")
    if sigma == 0:
        return image.copy()
    generator = rng if rng is not None else np.random.default_rng()
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    noise = generator.normal(0.0, sigma * 255.0, array.shape).astype(np.float32)
    noisy = np.clip(array + noise, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(noisy, mode="RGB")


def color_jitter(
    image: Image.Image,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    rng: Optional[np.random.Generator] = None,
) -> Image.Image:
    """Randomly shift brightness, contrast and saturation within +/- the limits."""

    for name, limit in (
        ("brightness", brightness),
        ("contrast", contrast),
        ("saturation", saturation),
    ):
        if not 0 <= float(limit) <= 1:
            raise ValueError(f"Colour jitter {name} must be within [0, 1], got {limit}")
    generator = rng if rng is not None else np.random.default_rng()
    result = image.convert("RGB")
    for enhancer_cls, limit in (
        (ImageEnhance.Brightness, float(brightness)),
        (ImageEnhance.Contrast, float(contrast)),
        (ImageEnhance.Color, float(saturation)),
    ):
        if limit <= 0:
            continue
        factor = float(generator.uniform(1.0 - limit, 1.0 + limit))
        result = enhancer_cls(result).enhance(factor)
    return result


def center_crop(image: Image.Image, fraction: float) -> Image.Image:
    """Keep the central ``fraction`` of the width and height.

    The crop is *not* resized back to the original dimensions: preprocessing
    resizes everything to the model input size anyway, so the effect on the
    model is a genuine zoom-in, as with a cropped repost.
    """

    fraction = float(fraction)
    if not 0 < fraction <= 1:
        raise ValueError(f"Centre crop fraction must be in (0, 1], got {fraction}")
    width, height = image.size
    crop_width = max(1, int(round(width * fraction)))
    crop_height = max(1, int(round(height * fraction)))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height))


# ---------------------------------------------------------------------------
# Spec construction and dispatch
# ---------------------------------------------------------------------------

_FAMILY_APPLIERS: Dict[str, Callable[..., Image.Image]] = {
    "jpeg": lambda image, rng, **params: jpeg_compress(image, params["quality"]),
    "blur": lambda image, rng, **params: gaussian_blur(image, params["sigma"]),
    "resize": lambda image, rng, **params: downscale_upscale(image, params["scale"]),
    "noise": lambda image, rng, **params: gaussian_noise(image, params["sigma"], rng),
    "color_jitter": lambda image, rng, **params: color_jitter(
        image,
        brightness=params.get("brightness", 0.2),
        contrast=params.get("contrast", 0.2),
        saturation=params.get("saturation", 0.2),
        rng=rng,
    ),
    "center_crop": lambda image, rng, **params: center_crop(image, params["fraction"]),
}

STOCHASTIC_FAMILIES = frozenset({"noise", "color_jitter"})


def _format_number(value: float) -> str:
    """Format a float for a transform name: 0.5 -> '0.5', 2.0 -> '2'."""

    return f"{float(value):g}"


def build_transform_specs(config: Optional[Dict[str, Any]] = None) -> List[TransformSpec]:
    """Build the configured transformation list from ``transformations``.

    Returns an empty list when ``transformations.enabled`` is false, which makes
    the rest of the pipeline fall back to the original image alone.
    """

    settings = (config or {}).get("transformations", {}) or {}
    if not settings.get("enabled", True):
        return []

    specs: List[TransformSpec] = []

    for quality in settings.get("jpeg_qualities", [90, 70, 50, 30]) or []:
        quality = int(quality)
        specs.append(
            TransformSpec(
                name=f"jpeg_q{quality}",
                family="jpeg",
                params={"quality": quality},
                description=f"JPEG re-encoded at quality {quality}",
            )
        )

    for sigma in settings.get("blur_sigmas", [0.5, 1.0, 2.0]) or []:
        sigma = float(sigma)
        specs.append(
            TransformSpec(
                name=f"blur_s{_format_number(sigma)}",
                family="blur",
                params={"sigma": sigma},
                description=f"Gaussian blur, sigma {_format_number(sigma)}",
            )
        )

    for scale in settings.get("resize_scales", [0.5, 0.25]) or []:
        scale = float(scale)
        specs.append(
            TransformSpec(
                name=f"resize_{_format_number(scale)}x",
                family="resize",
                params={"scale": scale},
                description=f"Downscaled to {_format_number(scale)}x then upscaled back",
            )
        )

    for sigma in settings.get("noise_sigmas", [0.02, 0.05, 0.10]) or []:
        sigma = float(sigma)
        specs.append(
            TransformSpec(
                name=f"noise_s{_format_number(sigma)}",
                family="noise",
                params={"sigma": sigma},
                description=f"Gaussian noise, sigma {_format_number(sigma)} of the 0-255 range",
            )
        )

    jitter = settings.get("color_jitter", {}) or {}
    brightness = float(jitter.get("brightness", 0.2))
    contrast = float(jitter.get("contrast", 0.2))
    saturation = float(jitter.get("saturation", 0.2))
    variants = max(0, int(jitter.get("variants", 1)))
    for index in range(variants):
        suffix = "" if variants == 1 else f"_{index + 1}"
        specs.append(
            TransformSpec(
                name=f"color_jitter{suffix}",
                family="color_jitter",
                params={
                    "brightness": brightness,
                    "contrast": contrast,
                    "saturation": saturation,
                    "draw": index,
                },
                description=(
                    f"Brightness/contrast/saturation jitter up to "
                    f"+/-{max(brightness, contrast, saturation) * 100:.0f}%"
                ),
            )
        )

    fraction = settings.get("center_crop_fraction", 0.8)
    if fraction:
        fraction = float(fraction)
        specs.append(
            TransformSpec(
                name=f"center_crop_{int(round(fraction * 100))}",
                family="center_crop",
                params={"fraction": fraction},
                description=f"Centre crop keeping {fraction * 100:.0f}% of each side",
            )
        )

    return specs


def _rng_for(spec: TransformSpec, seed: Optional[int]) -> np.random.Generator:
    """Derive a per-transform generator so results do not depend on ordering."""

    base = 0 if seed is None else int(seed)
    # crc32 of the name gives a stable offset across processes and Python runs.
    offset = zlib.crc32(spec.name.encode("utf-8")) & 0xFFFFFFFF
    return np.random.default_rng((base + offset) % (2**32))


def apply_transform(
    image: Image.Image, spec: TransformSpec, seed: Optional[int] = None
) -> Image.Image:
    """Apply one transformation spec to a full-resolution RGB image."""

    applier = _FAMILY_APPLIERS.get(spec.family)
    if applier is None:
        known = ", ".join(sorted(_FAMILY_APPLIERS))
        raise ValueError(f"Unknown transformation family '{spec.family}'. Known families: {known}")
    rng = _rng_for(spec, seed) if spec.family in STOCHASTIC_FAMILIES else None
    return applier(image.convert("RGB"), rng, **spec.params)


def generate_variants(
    image: Image.Image,
    config: Optional[Dict[str, Any]] = None,
    specs: Optional[List[TransformSpec]] = None,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, Image.Image], List[Dict[str, Any]]]:
    """Build every image version to classify, keyed by transformation name.

    The returned mapping always starts with ``"original"``. A transformation
    that fails is skipped and reported in the second return value rather than
    aborting the whole analysis.
    """

    if image is None:
        raise ValueError("Transformation generation requires an image, got None")

    settings = (config or {}).get("transformations", {}) or {}
    if seed is None:
        seed = settings.get("seed", 1234)
    if specs is None:
        specs = build_transform_specs(config)

    base = image.convert("RGB")
    variants: Dict[str, Image.Image] = {ORIGINAL_KEY: base}
    errors: List[Dict[str, Any]] = []

    for spec in specs:
        try:
            variants[spec.name] = apply_transform(base, spec, seed)
        except (ValueError, OSError, MemoryError) as exc:
            errors.append(
                {
                    "stage": "transformation",
                    "transform": spec.name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return variants, errors


def describe_transforms(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Return the configured transforms as plain dicts, for docs and the UI."""

    return [spec.as_dict() for spec in build_transform_specs(config)]
