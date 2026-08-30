"""Compound transformation chains.

The robustness protocol applies one transformation at a time. Real images do not
arrive that way: they are cropped, re-encoded, resized by a platform, screenshot,
and re-encoded again, and the damage compounds. The measured failure mode only
shows up under compounding -- a single JPEG barely moves a synthetic image's
score, while ten stacked operations walk it out of the AI band entirely.

This module composes the *existing* transformation primitives from
``src.pipeline.transformations`` rather than inventing new ones, so a chain is
exactly the official transformations applied in sequence. Two families are
provided:

* **named chains** -- the fixed sequences the evaluation protocol calls for,
  identical for every image so results are comparable;
* **generation chains** -- N randomly ordered operations, seeded per image and
  per generation, which is what "repeated processing" actually looks like.

Nothing here retunes a threshold. Chains are scored at whatever fixed threshold
the caller supplies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from src.pipeline.transformations import (
    center_crop,
    color_jitter,
    downscale_upscale,
    gaussian_blur,
    gaussian_noise,
    jpeg_compress,
)

CLEAN_CHAIN = "clean"

# One operation: a name and a function of (image, rng) -> image. Every entry
# delegates to the shipped transformation primitives.
Operation = Tuple[str, Callable[[Image.Image, np.random.Generator], Image.Image]]


def _screenshot(image: Image.Image, _rng: np.random.Generator) -> Image.Image:
    """Approximate a screenshot: resample to screen scale, then re-encode.

    This is a composition of operations the project already applies, not a
    claim to reproduce any particular device's capture path.
    """

    width, height = image.size
    scaled = image.resize(
        (max(1, int(width * 0.6)), max(1, int(height * 0.6))), Image.BILINEAR
    )
    return jpeg_compress(scaled, 85)


def _sharpen(image: Image.Image, _rng: np.random.Generator) -> Image.Image:
    """Unsharp-style boost, the inverse pressure to blur."""

    from PIL import ImageFilter

    return image.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))


OPERATIONS: Dict[str, Callable[[Image.Image, np.random.Generator], Image.Image]] = {
    "jpeg_q90": lambda im, rng: jpeg_compress(im, 90),
    "jpeg_q70": lambda im, rng: jpeg_compress(im, 70),
    "jpeg_q50": lambda im, rng: jpeg_compress(im, 50),
    "jpeg_q30": lambda im, rng: jpeg_compress(im, 30),
    "blur_s1": lambda im, rng: gaussian_blur(im, 1.0),
    "blur_s0.5": lambda im, rng: gaussian_blur(im, 0.5),
    "resize_0.5x": lambda im, rng: downscale_upscale(im, 0.5),
    "resize_0.7x": lambda im, rng: downscale_upscale(im, 0.7),
    "noise_s0.02": lambda im, rng: gaussian_noise(im, 0.02, rng),
    "noise_s0.05": lambda im, rng: gaussian_noise(im, 0.05, rng),
    "color_jitter": lambda im, rng: color_jitter(im, 0.2, 0.2, 0.2, rng),
    "center_crop_80": lambda im, rng: center_crop(im, 0.8),
    "center_crop_90": lambda im, rng: center_crop(im, 0.9),
    "screenshot": _screenshot,
    "sharpen": _sharpen,
}

# The fixed sequences the protocol asks for. Order is part of the definition.
NAMED_CHAINS: Dict[str, List[str]] = {
    "resize_jpeg_crop": ["resize_0.5x", "jpeg_q70", "center_crop_80"],
    "blur_jitter_jpeg": ["blur_s1", "color_jitter", "jpeg_q70"],
    "noise_resize_blur": ["noise_s0.05", "resize_0.5x", "blur_s1"],
    "crop_jpeg_color": ["center_crop_80", "jpeg_q50", "color_jitter"],
    "screenshot": ["screenshot"],
    "screenshot_reshare": ["screenshot", "jpeg_q70", "resize_0.7x"],
    "recompress_x3": ["jpeg_q70", "jpeg_q70", "jpeg_q70"],
    "recompress_x5": ["jpeg_q70"] * 5,
    "sharpen_recompress": ["sharpen", "jpeg_q50"],
}

# Generation depths for the randomly ordered chains.
DEFAULT_GENERATIONS: Tuple[int, ...] = (1, 2, 3, 5, 10)

# Operations drawn from when building a random chain. Deliberately excludes the
# harshest settings so a 10-step chain does not reduce every image to mush.
RANDOM_POOL: Tuple[str, ...] = (
    "jpeg_q70",
    "jpeg_q50",
    "blur_s1",
    "blur_s0.5",
    "resize_0.7x",
    "noise_s0.02",
    "color_jitter",
    "center_crop_90",
    "screenshot",
    "sharpen",
)


@dataclass(frozen=True)
class ChainSpec:
    """A named sequence of operations applied in order."""

    name: str
    operations: Tuple[str, ...]
    generation: int = 0
    randomised: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "operations": list(self.operations),
            "length": len(self.operations),
            "generation": self.generation,
            "randomised": self.randomised,
        }


def build_named_chains() -> List[ChainSpec]:
    """The fixed protocol chains, identical for every image."""

    return [
        ChainSpec(name=name, operations=tuple(ops), generation=len(ops))
        for name, ops in NAMED_CHAINS.items()
    ]


def build_generation_chains(
    seed: int,
    generations: Sequence[int] = DEFAULT_GENERATIONS,
    pool: Sequence[str] = RANDOM_POOL,
) -> List[ChainSpec]:
    """Randomly ordered chains of increasing depth.

    The order is randomised but seeded, so a run is reproducible and every image
    in one run sees the same chain -- otherwise per-image differences would be
    confounded with per-chain differences.
    """

    unknown = [name for name in pool if name not in OPERATIONS]
    if unknown:
        raise ValueError(f"Unknown chain operation(s): {', '.join(unknown)}")

    chains: List[ChainSpec] = []
    for depth in generations:
        depth = int(depth)
        if depth < 1:
            raise ValueError(f"generation depth must be at least 1, got {depth}")
        rng = np.random.default_rng(int(seed) + depth * 7919)
        if depth <= len(pool):
            chosen = list(rng.choice(np.array(pool, dtype=object), size=depth, replace=False))
        else:
            chosen = list(rng.choice(np.array(pool, dtype=object), size=depth, replace=True))
        chains.append(
            ChainSpec(
                name=f"generation_{depth}",
                operations=tuple(str(item) for item in chosen),
                generation=depth,
                randomised=True,
            )
        )
    return chains


def apply_chain(
    image: Image.Image, spec: ChainSpec, seed: int = 1234
) -> Image.Image:
    """Apply every operation in order, returning the compounded result."""

    result = image.convert("RGB")
    for index, name in enumerate(spec.operations):
        operation = OPERATIONS.get(name)
        if operation is None:
            raise ValueError(
                f"Unknown chain operation '{name}'. Known: {', '.join(sorted(OPERATIONS))}"
            )
        # A distinct stream per position keeps the stochastic operations from
        # repeating the same draw when one appears twice in a chain.
        rng = np.random.default_rng(int(seed) + index * 104729)
        result = operation(result, rng)
    return result


def generate_chain_variants(
    image: Image.Image,
    chains: Sequence[ChainSpec],
    seed: int = 1234,
) -> Tuple[Dict[str, Image.Image], List[Dict[str, Any]]]:
    """Build ``{chain name: image}`` including the untouched ``clean`` entry.

    A chain that fails is skipped and reported rather than aborting the run, so
    one awkward image cannot end an evaluation.
    """

    base = image.convert("RGB")
    variants: Dict[str, Image.Image] = {CLEAN_CHAIN: base}
    errors: List[Dict[str, Any]] = []
    for spec in chains:
        try:
            variants[spec.name] = apply_chain(base, spec, seed)
        except (ValueError, OSError, MemoryError) as exc:
            errors.append(
                {
                    "stage": "chain",
                    "chain": spec.name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return variants, errors


@dataclass
class ChainRecord:
    """Scores for one image across every chain."""

    img_id: str
    binary_label: int
    class_name: str = ""
    chain_scores: Dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def clean_score(self) -> float:
        return float(self.chain_scores[CLEAN_CHAIN])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "img_id": self.img_id,
            "binary_label": self.binary_label,
            "class_name": self.class_name,
            "chain_scores": {k: round(float(v), 6) for k, v in self.chain_scores.items()},
            "seconds": round(float(self.seconds), 4),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ChainRecord":
        return cls(
            img_id=str(payload["img_id"]),
            binary_label=int(payload["binary_label"]),
            class_name=str(payload.get("class_name", "")),
            chain_scores={str(k): float(v) for k, v in payload["chain_scores"].items()},
            seconds=float(payload.get("seconds", 0.0)),
        )


def chain_drift(
    records: Sequence[ChainRecord], chain: str, label: Optional[int] = None
) -> Optional[float]:
    """Mean signed score change relative to the clean image.

    Negative means the chain pushed scores toward "authentic", which is the
    direction that manufactures false negatives on synthetic images.
    """

    subset = [
        record
        for record in records
        if chain in record.chain_scores
        and CLEAN_CHAIN in record.chain_scores
        and (label is None or record.binary_label == label)
    ]
    if not subset:
        return None
    return float(
        np.mean([record.chain_scores[chain] - record.clean_score for record in subset])
    )


def chain_names(records: Sequence[ChainRecord]) -> List[str]:
    """Every chain present, clean first."""

    if not records:
        return []
    names = [CLEAN_CHAIN] if CLEAN_CHAIN in records[0].chain_scores else []
    names += [name for name in records[0].chain_scores if name != CLEAN_CHAIN]
    return names
