"""Device selection shared by every stage of the inference pipeline.

The pipeline must run unchanged on a laptop CPU, an Apple Silicon GPU, and a
CUDA machine, so device choice lives in exactly one place.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

VALID_DEVICES = ("auto", "cpu", "cuda", "mps")


def cuda_available() -> bool:
    """Return True when a usable CUDA device is present."""

    try:
        return bool(torch.cuda.is_available())
    except (AssertionError, RuntimeError):  # pragma: no cover - driver issues
        return False


def mps_available() -> bool:
    """Return True when Apple's Metal backend is present and built in."""

    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    try:
        return bool(backend.is_available() and backend.is_built())
    except (AssertionError, RuntimeError):  # pragma: no cover - driver issues
        return False


def resolve_device(preference: Optional[str] = "auto") -> torch.device:
    """Turn a device preference into a concrete ``torch.device``.

    ``auto`` prefers CUDA, then Apple MPS, then CPU. An explicit request for an
    unavailable accelerator falls back to CPU rather than crashing mid-run.
    """

    requested = str(preference or "auto").strip().lower()
    if requested in {"gpu", "auto:gpu"}:
        requested = "auto"

    if requested == "auto":
        if cuda_available():
            return torch.device("cuda")
        if mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested.startswith("cuda"):
        return torch.device(requested if cuda_available() else "cpu")
    if requested.startswith("mps"):
        return torch.device("mps" if mps_available() else "cpu")
    if requested.startswith("cpu"):
        return torch.device("cpu")

    # Unknown string: stay conservative instead of raising during inference.
    return torch.device("cpu")


def device_from_config(
    config: Optional[Dict[str, Any]] = None, override: Optional[str] = None
) -> torch.device:
    """Resolve the device from ``inference.device``, honouring a CLI override."""

    if override:
        return resolve_device(override)
    preference = "auto"
    if config:
        preference = str(config.get("inference", {}).get("device", "auto"))
    return resolve_device(preference)


def describe_device(device: torch.device | str) -> str:
    """Return a short human-readable description for logs and the demo UI."""

    device = torch.device(device)
    if device.type == "cuda" and cuda_available():
        index = device.index or 0
        try:
            return f"cuda:{index} ({torch.cuda.get_device_name(index)})"
        except (AssertionError, RuntimeError):  # pragma: no cover
            return f"cuda:{index}"
    if device.type == "mps":
        return "mps (Apple Metal)"
    return "cpu"
