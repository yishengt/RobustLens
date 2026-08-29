"""Stage 3: preprocessing shared by the original image and every transform.

Only pixel content reaches the model. Filenames, EXIF metadata and any
watermark text are deliberately dropped here: ``Image.convert("RGB")`` returns
a bare pixel buffer, so nothing downstream can key off them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

DEFAULT_IMAGE_SIZE = 224
DEFAULT_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
DEFAULT_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Pillow renamed the resampling enum in v9.1; support both spellings.
_RESAMPLE = getattr(Image, "Resampling", Image)
BILINEAR = _RESAMPLE.BILINEAR


@dataclass(frozen=True)
class Preprocessor:
    """Deterministic RGB -> 224x224 -> normalized tensor conversion."""

    image_size: int = DEFAULT_IMAGE_SIZE
    mean: Tuple[float, float, float] = DEFAULT_MEAN
    std: Tuple[float, float, float] = DEFAULT_STD

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "Preprocessor":
        config = config or {}
        normalization = config.get("normalization", {}) or {}
        image_size = int(config.get("data", {}).get("image_size", DEFAULT_IMAGE_SIZE))
        if image_size <= 0:
            raise ValueError(f"data.image_size must be positive, got {image_size}")
        mean = tuple(float(value) for value in normalization.get("mean", DEFAULT_MEAN))
        std = tuple(float(value) for value in normalization.get("std", DEFAULT_STD))
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("normalization.mean and normalization.std must each have 3 values")
        if any(value == 0 for value in std):
            raise ValueError("normalization.std entries must be non-zero")
        return cls(image_size=image_size, mean=mean, std=std)

    # -- individual steps, exposed so tests and the demo can call them -------

    @staticmethod
    def to_rgb(image: Image.Image) -> Image.Image:
        """Return an RGB copy of the image; a no-op for images already RGB."""

        return image if image.mode == "RGB" else image.convert("RGB")

    def resize(self, image: Image.Image) -> Image.Image:
        """Resize to the square model input size."""

        target = (self.image_size, self.image_size)
        return image if image.size == target else image.resize(target, BILINEAR)

    def to_array(self, image: Image.Image) -> np.ndarray:
        """Return the resized RGB image as a uint8 ``HxWx3`` array."""

        return np.asarray(self.resize(self.to_rgb(image)), dtype=np.uint8)

    def normalize(self, array: np.ndarray) -> torch.Tensor:
        """Scale uint8 pixels to 0-1 and apply per-channel normalization."""

        # Copy: arrays produced by np.asarray(PIL image) are read-only, and
        # torch.from_numpy would otherwise share that non-writable buffer.
        writable = np.array(array, dtype=np.uint8, copy=True, order="C")
        tensor = torch.from_numpy(writable).permute(2, 0, 1).float().div_(255.0)
        mean = torch.tensor(self.mean, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(self.std, dtype=torch.float32).view(3, 1, 1)
        return tensor.sub_(mean).div_(std)

    # -- full pipeline ------------------------------------------------------

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Preprocess one image into a normalized ``3xHxW`` float tensor."""

        if image is None:
            raise ValueError("Preprocessing requires an image, got None")
        return self.normalize(self.to_array(image))

    def batch(self, images: Sequence[Image.Image] | Iterable[Image.Image]) -> torch.Tensor:
        """Preprocess several images into one stacked ``NxCxHxW`` tensor."""

        tensors: List[torch.Tensor] = [self(image) for image in images]
        if not tensors:
            raise ValueError("Preprocessing requires at least one image")
        return torch.stack(tensors, dim=0)

    def denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """Invert normalization, returning a uint8 ``HxWx3`` array for display."""

        if tensor.dim() == 4:
            tensor = tensor[0]
        mean = torch.tensor(self.mean, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(self.std, dtype=torch.float32).view(3, 1, 1)
        restored = tensor.detach().cpu().float().mul(std).add(mean).clamp(0.0, 1.0)
        return (restored.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def preserve_original(image: Image.Image) -> Image.Image:
    """Return an independent full-resolution RGB copy kept for display.

    Transformations and preprocessing must never mutate the image the UI shows,
    so the pipeline holds on to this copy.
    """

    return Preprocessor.to_rgb(image).copy()
