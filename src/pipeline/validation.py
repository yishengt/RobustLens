"""Stage 2: input validation and image metadata capture.

Only Pillow and the standard library are required here, so an invalid upload
can be rejected before any of the torch machinery is loaded.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, UnidentifiedImageError

# Formats named in the problem statement.
SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
SUPPORTED_PIL_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})

DEFAULT_MIN_SIDE = 32
DEFAULT_MAX_SIDE = 20000
DEFAULT_MAX_PIXELS = 50_000_000
DEFAULT_MAX_FILE_SIZE_MB = 64.0


class ImageValidationError(ValueError):
    """Raised when an input image cannot be used for inference.

    The message is written to be shown directly to a user: it always names the
    file and says which specific check failed.
    """


@dataclass(frozen=True)
class ImageMetadata:
    """Everything the pipeline records about an input image."""

    filename: str
    file_path: str
    file_type: str
    file_size_bytes: int
    width: int
    height: int
    color_mode: str

    @property
    def file_size_human(self) -> str:
        """Return the file size formatted for display."""

        size = float(self.file_size_bytes)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024.0 or unit == "GB":
                return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} GB"  # pragma: no cover - unreachable

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["file_size_human"] = self.file_size_human
        return payload


@dataclass(frozen=True)
class ValidationLimits:
    """Size limits applied during validation."""

    min_side: int = DEFAULT_MIN_SIDE
    max_side: int = DEFAULT_MAX_SIDE
    max_pixels: int = DEFAULT_MAX_PIXELS
    max_file_size_mb: float = DEFAULT_MAX_FILE_SIZE_MB

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "ValidationLimits":
        settings = (config or {}).get("validation", {}) or {}
        return cls(
            min_side=int(settings.get("min_side", DEFAULT_MIN_SIDE)),
            max_side=int(settings.get("max_side", DEFAULT_MAX_SIDE)),
            max_pixels=int(settings.get("max_pixels", DEFAULT_MAX_PIXELS)),
            max_file_size_mb=float(settings.get("max_file_size_mb", DEFAULT_MAX_FILE_SIZE_MB)),
        )


def supported_extensions(config: Optional[Dict[str, Any]] = None) -> Tuple[str, ...]:
    """Return the configured extensions, always lower-case and dot-prefixed."""

    configured = (config or {}).get("data", {}).get("extensions") if config else None
    if not configured:
        return SUPPORTED_EXTENSIONS
    normalised = []
    for extension in configured:
        text = str(extension).strip().lower()
        if text and not text.startswith("."):
            text = f".{text}"
        if text:
            normalised.append(text)
    return tuple(normalised) or SUPPORTED_EXTENSIONS


def _check_dimensions(width: int, height: int, limits: ValidationLimits, label: str) -> None:
    """Raise when the decoded dimensions are outside the accepted range."""

    if width <= 0 or height <= 0:
        raise ImageValidationError(f"{label}: image reports an empty size ({width}x{height}).")
    if min(width, height) < limits.min_side:
        raise ImageValidationError(
            f"{label}: image is too small ({width}x{height}); "
            f"each side must be at least {limits.min_side}px."
        )
    if max(width, height) > limits.max_side:
        raise ImageValidationError(
            f"{label}: image is too large ({width}x{height}); "
            f"each side must be at most {limits.max_side}px."
        )
    if width * height > limits.max_pixels:
        raise ImageValidationError(
            f"{label}: image has {width * height:,} pixels, above the "
            f"{limits.max_pixels:,} pixel safety limit."
        )


def _to_rgb(image: Image.Image, label: str) -> Image.Image:
    """Convert to RGB, turning any Pillow failure into a validation error."""

    try:
        return image.convert("RGB")
    except (OSError, ValueError) as exc:
        raise ImageValidationError(f"{label}: image could not be converted to RGB ({exc}).") from exc


def validate_image_file(
    path: str | Path, config: Optional[Dict[str, Any]] = None
) -> ImageMetadata:
    """Validate an image on disk and return its metadata.

    Checks, in order: the file exists, is a regular non-empty file, has a
    supported extension, is within the size limit, can be opened by Pillow,
    is not corrupted, has an accepted format and dimensions, and can be
    converted to RGB.
    """

    limits = ValidationLimits.from_config(config)
    allowed = supported_extensions(config)
    image_path = Path(path).expanduser()
    label = image_path.name or str(image_path)

    if not image_path.exists():
        raise ImageValidationError(f"{label}: file does not exist at {image_path}.")
    if not image_path.is_file():
        raise ImageValidationError(f"{label}: path is not a regular file ({image_path}).")

    suffix = image_path.suffix.lower()
    if suffix not in allowed:
        raise ImageValidationError(
            f"{label}: unsupported file type '{suffix or 'none'}'. "
            f"Supported types are {', '.join(allowed)}."
        )

    file_size = image_path.stat().st_size
    if file_size == 0:
        raise ImageValidationError(f"{label}: file is empty (0 bytes).")
    max_bytes = int(limits.max_file_size_mb * 1024 * 1024)
    if file_size > max_bytes:
        raise ImageValidationError(
            f"{label}: file is {file_size / 1024 / 1024:.1f} MB, above the "
            f"{limits.max_file_size_mb:.0f} MB limit."
        )

    # verify() consumes the file object, so the image is opened twice: once to
    # detect corruption and once to read real pixel data.
    try:
        with Image.open(image_path) as probe:
            probe.verify()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ImageValidationError(f"{label}: file is corrupted or unreadable ({exc}).") from exc

    try:
        with Image.open(image_path) as image:
            image_format = (image.format or "").upper()
            color_mode = image.mode
            width, height = image.size
            _check_dimensions(width, height, limits, label)
            if image_format and image_format not in SUPPORTED_PIL_FORMATS:
                raise ImageValidationError(
                    f"{label}: decoded format '{image_format}' is not supported. "
                    f"Supported formats are {', '.join(sorted(SUPPORTED_PIL_FORMATS))}."
                )
            # Force a full decode: truncated files only fail at this point.
            image.load()
            _to_rgb(image, label)
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError(f"{label}: image could not be decoded ({exc}).") from exc

    return ImageMetadata(
        filename=image_path.name,
        file_path=str(image_path),
        file_type=image_format or suffix.lstrip("."),
        file_size_bytes=int(file_size),
        width=int(width),
        height=int(height),
        color_mode=color_mode,
    )


def load_validated_image(
    path: str | Path, config: Optional[Dict[str, Any]] = None
) -> Tuple[Image.Image, ImageMetadata]:
    """Validate an image and return it as RGB together with its metadata.

    The returned image is the full-resolution original; transformations are
    applied to it before preprocessing shrinks anything.
    """

    metadata = validate_image_file(path, config)
    image_path = Path(metadata.file_path)
    label = metadata.filename
    try:
        with Image.open(image_path) as image:
            image.load()
            rgb = _to_rgb(image, label)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError(f"{label}: image could not be decoded ({exc}).") from exc
    return rgb, metadata


def validate_image_bytes(
    data: bytes,
    filename: str = "upload",
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Image.Image, ImageMetadata]:
    """Validate in-memory image bytes, as produced by a Streamlit upload."""

    limits = ValidationLimits.from_config(config)
    allowed = supported_extensions(config)
    label = filename or "upload"

    if not data:
        raise ImageValidationError(f"{label}: uploaded file is empty (0 bytes).")
    max_bytes = int(limits.max_file_size_mb * 1024 * 1024)
    if len(data) > max_bytes:
        raise ImageValidationError(
            f"{label}: upload is {len(data) / 1024 / 1024:.1f} MB, above the "
            f"{limits.max_file_size_mb:.0f} MB limit."
        )

    suffix = Path(label).suffix.lower()
    if suffix and suffix not in allowed:
        raise ImageValidationError(
            f"{label}: unsupported file type '{suffix}'. "
            f"Supported types are {', '.join(allowed)}."
        )

    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ImageValidationError(f"{label}: upload is corrupted or unreadable ({exc}).") from exc

    try:
        with Image.open(io.BytesIO(data)) as image:
            image_format = (image.format or "").upper()
            color_mode = image.mode
            width, height = image.size
            _check_dimensions(width, height, limits, label)
            if image_format and image_format not in SUPPORTED_PIL_FORMATS:
                raise ImageValidationError(
                    f"{label}: decoded format '{image_format}' is not supported. "
                    f"Supported formats are {', '.join(sorted(SUPPORTED_PIL_FORMATS))}."
                )
            image.load()
            rgb = _to_rgb(image, label)
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError(f"{label}: upload could not be decoded ({exc}).") from exc

    metadata = ImageMetadata(
        filename=label,
        file_path=label,
        file_type=image_format or suffix.lstrip("."),
        file_size_bytes=len(data),
        width=int(width),
        height=int(height),
        color_mode=color_mode,
    )
    return rgb, metadata


def list_supported_images(
    directory: str | Path,
    config: Optional[Dict[str, Any]] = None,
    extensions: Optional[Iterable[str]] = None,
) -> List[Path]:
    """Recursively list supported images in a directory, sorted by path."""

    root = Path(directory).expanduser()
    if not root.exists():
        raise ImageValidationError(f"Input directory does not exist: {root}")
    if not root.is_dir():
        raise ImageValidationError(f"Input path is not a directory: {root}")

    if extensions is None:
        allowed = set(supported_extensions(config))
    else:
        allowed = {
            (ext if str(ext).startswith(".") else f".{ext}").lower() for ext in extensions
        }
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed and not path.name.startswith(".")
    )
