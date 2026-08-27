"""Frozen CLIP image-encoder features for AI-generated image detection.

Why frozen CLIP rather than a fine-tuned CNN:

A CNN fine-tuned end-to-end on real-vs-fake has millions of free parameters
and will latch onto the easiest available signal -- usually the *training
generator's* fingerprint. That is a shortcut: it scores well in-distribution
and collapses on generators it has never seen.

Freezing CLIP removes that freedom. Its features were learned from ~400M
web image-text pairs for an unrelated objective, so they cannot bend to fit
one generator. A linear probe on top is then forced to separate real from
fake using structure that is *already generic* in that space -- which is
what transfers to unseen generators (Ojha et al., CVPR 2023).

The text tower is discarded entirely; only the image encoder is used.

Cost model: feature extraction is the only expensive step and its output is
cached to disk, so probe training, calibration, and threshold tuning all
re-run in CPU-seconds against the cache.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from src.utils.config import get_device, resolve_config_path

# Competition rule: submitted models must stay under 2B parameters.
# ViT-L/14's image tower is ~304M, so this is a guard against an accidental
# swap to an oversized backbone, not a tight constraint.
MAX_PARAMETERS = 2_000_000_000

DEFAULT_MODEL_NAME = "ViT-L-14"
DEFAULT_PRETRAINED = "openai"


def _clip_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("clip", {}) if config else {}


def count_clip_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Return (visual tower, full model) parameter counts.

    Reported in the README to evidence the <2B compliance claim. The visual
    count is what actually ships, since the text tower is never loaded at
    inference.
    """

    total = sum(parameter.numel() for parameter in model.parameters())
    visual = sum(parameter.numel() for parameter in model.visual.parameters())
    return visual, total


def _resolve_openai_variant(model_name: str, pretrained: str, open_clip: Any) -> str:
    """Select the QuickGELU architecture when loading OpenAI weights.

    OpenAI trained CLIP with QuickGELU, but open_clip's plain ``ViT-L-14``
    config uses standard GELU. Pairing that config with the ``openai``
    weights loads mismatched activations: it emits only a UserWarning, runs
    fine, and returns quietly degraded features -- a failure mode that would
    surface as unexplained probe accuracy rather than a crash.

    Upgrading to the explicit ``-quickgelu`` variant removes the ambiguity.
    """

    if pretrained != "openai" or model_name.endswith("-quickgelu"):
        return model_name

    candidate = f"{model_name}-quickgelu"
    available = {name for name, tag in open_clip.list_pretrained() if tag == "openai"}
    return candidate if candidate in available else model_name


def load_clip(
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: str = DEFAULT_PRETRAINED,
    device: str = "cpu",
) -> Tuple[torch.nn.Module, Any]:
    """Load a CLIP backbone, frozen and in eval mode.

    ``pretrained='openai'`` is the deliberate default: those weights come
    from the ~2020 WebImageText scrape, which predates widespread diffusion
    imagery. LAION/DataComp checkpoints were scraped later and may contain
    AI-generated images, which is an awkward property for a detector's
    backbone. Record the exact (model_name, pretrained) pair in the README --
    that string, not a committed weight file, is the reproducibility contract.

    Returns the model and CLIP's own preprocessing transform. Use that
    transform, never the project's ImageNet normalization: CLIP has different
    normalization constants and mixing them silently degrades accuracy.
    """

    try:
        import open_clip
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "open_clip is required for CLIP features. Install with: "
            "pip install open_clip_torch"
        ) from exc

    model_name = _resolve_openai_variant(model_name, pretrained, open_clip)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )

    visual_parameters, total_parameters = count_clip_parameters(model)
    if visual_parameters > MAX_PARAMETERS:
        raise ValueError(
            f"{model_name}/{pretrained} has {visual_parameters:,} visual parameters, "
            f"which exceeds the {MAX_PARAMETERS:,} limit."
        )

    # Frozen: no optimizer ever sees these, and no graph is built for them.
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    return model.to(device), preprocess


def _read_pil(path: str | Path) -> Image.Image:
    """Load an image from disk as PIL RGB, with a path-aware error.

    Deliberately duplicates the loading logic in ``src.data.dataset`` rather
    than importing it: that module pulls in the Albumentations stack, and the
    CLIP path has no need for it. Keeping this module dependency-light means
    feature extraction runs in a minimal environment (e.g. a Colab runtime
    with only torch and open_clip installed).
    """

    image_path = Path(path)
    try:
        with Image.open(image_path) as image:
            return image.convert("RGB")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Image file not found: {image_path}") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Invalid or unreadable image '{image_path}': {exc}") from exc


def _to_pil(image: Any) -> Image.Image:
    """Accept a PIL image, a NumPy RGB array, or a path; return PIL RGB."""

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    if isinstance(image, (str, Path)):
        return _read_pil(image)
    raise TypeError(f"Cannot convert {type(image).__name__} to a PIL image")


@torch.no_grad()
def encode_images(
    images: Sequence[Any],
    model: torch.nn.Module,
    preprocess: Any,
    device: str = "cpu",
    batch_size: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """Encode images into a [N, D] float32 feature matrix.

    Accepts PIL images, RGB arrays, or paths. Images must already carry any
    pixel-space augmentation: transforms belong *before* CLIP preprocessing,
    never after.

    ``normalize`` L2-normalizes each vector. This is on by default because it
    conditions the feature scale for the downstream linear probe; turn it off
    to ablate.
    """

    if len(images) == 0:
        raise ValueError("No images supplied for encoding")

    # fp16 on CUDA roughly halves extraction time; CPU stays fp32 because
    # half precision is not reliably faster there.
    use_half = device.startswith("cuda")
    features: List[np.ndarray] = []

    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        tensors = torch.stack([preprocess(_to_pil(image)) for image in batch]).to(device)
        if use_half:
            tensors = tensors.half()
        with torch.autocast(device_type="cuda", enabled=use_half):
            encoded = model.encode_image(tensors)
        encoded = encoded.float()
        if normalize:
            encoded = encoded / encoded.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        features.append(encoded.cpu().numpy())

    return np.concatenate(features, axis=0).astype(np.float32)


def encode_paths(
    paths: Sequence[str | Path],
    model: torch.nn.Module,
    preprocess: Any,
    device: str = "cpu",
    batch_size: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """Encode images from disk, loading lazily one batch at a time.

    Prefer this over ``encode_images`` for large corpora: it never holds more
    than ``batch_size`` decoded images in memory at once.
    """

    features: List[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch_images = [_to_pil(path) for path in batch_paths]
        features.append(
            encode_images(
                batch_images, model, preprocess, device, len(batch_images), normalize
            )
        )
    return np.concatenate(features, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Disk cache
#
# Extraction is the one costly step; everything downstream is cheap. Caching
# the feature matrix turns each later experiment into a CPU-seconds rerun.
# ---------------------------------------------------------------------------


def save_features(
    cache_path: str | Path,
    features: np.ndarray,
    labels: np.ndarray,
    paths: Sequence[str],
    metadata: Dict[str, Any] | None = None,
) -> Path:
    """Persist a feature matrix with the provenance needed to validate it."""

    output = Path(cache_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=np.asarray(features, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        paths=np.asarray([str(path) for path in paths], dtype=object),
        metadata=np.asarray([metadata or {}], dtype=object),
    )
    return output


def load_features(
    cache_path: str | Path,
    expected_metadata: Dict[str, Any] | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """Load cached features, refusing a cache built with different settings.

    The metadata check matters: silently training a probe on features from a
    different backbone is the kind of bug that produces plausible-looking but
    meaningless numbers.
    """

    path = Path(cache_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Feature cache not found: {path}")

    payload = np.load(path, allow_pickle=True)
    metadata = dict(payload["metadata"][0])

    if expected_metadata:
        mismatched = {
            key: (metadata.get(key), value)
            for key, value in expected_metadata.items()
            if metadata.get(key) != value
        }
        if mismatched:
            raise ValueError(
                f"Feature cache at {path} was built with different settings "
                f"{mismatched}. Delete it and re-extract."
            )

    return (
        payload["features"].astype(np.float32),
        payload["labels"].astype(np.int64),
        [str(item) for item in payload["paths"]],
        metadata,
    )


def build_extractor(config: Dict[str, Any]) -> Tuple[torch.nn.Module, Any, str, Dict[str, Any]]:
    """Load the configured CLIP backbone plus the metadata that identifies it.

    The returned metadata is stamped into the feature cache so a stale cache
    from a different backbone is rejected rather than silently used.
    """

    settings = _clip_config(config)
    model_name = str(settings.get("model_name", DEFAULT_MODEL_NAME))
    pretrained = str(settings.get("pretrained", DEFAULT_PRETRAINED))
    normalize = bool(settings.get("normalize", True))
    device = get_device(config, settings.get("device"))

    model, preprocess = load_clip(model_name, pretrained, device)
    visual_parameters, total_parameters = count_clip_parameters(model)

    metadata = {
        "model_name": model_name,
        "pretrained": pretrained,
        "normalize": normalize,
        "visual_parameters": int(visual_parameters),
        "total_parameters": int(total_parameters),
    }
    return model, preprocess, device, metadata
