"""Image discovery, validation, and RGB decoding for inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, UnidentifiedImageError

DEFAULT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _extensions(config: Optional[Dict[str, Any]] = None) -> set[str]:
    configured = config.get("data", {}).get("extensions") if config else None
    return {str(ext).lower() for ext in (configured or DEFAULT_EXTENSIONS)}


def list_image_files(directory: str | Path, extensions: Optional[set[str]] = None) -> List[Path]:
    """List supported image files recursively in a directory."""

    root = Path(directory).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    allowed = {str(extension).lower() for extension in (extensions or DEFAULT_EXTENSIONS)}
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in allowed
    )


def validate_image(path: str | Path) -> None:
    """Verify that Pillow can identify an image without decoding surprises."""

    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    try:
        with Image.open(image_path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Invalid or unreadable image '{image_path}': {exc}") from exc


def read_image(path: str | Path) -> np.ndarray:
    """Read an image as an RGB NumPy array with a path-aware error."""

    image_path = Path(path)
    try:
        with Image.open(image_path) as image:
            return np.asarray(image.convert("RGB"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Image file not found: {image_path}") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Invalid or unreadable image '{image_path}': {exc}") from exc
