"""Configuration loading and path resolution helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


ROOT_MARKERS = ("requirements.txt", "ruff.toml", "app.py", "configs")


def find_project_root(config_file: Path) -> Path:
    """Locate the project root by walking up from the config file.

    Assuming the config always sits exactly one directory below the root (the
    old ``parent.parent``) silently mis-resolves every relative path whenever it
    does not -- a config at the root, or under ``configs/experiments/`` -- and
    the calibration lookup is designed to fail quietly, so a wrong root shows up
    as uncalibrated scores rather than as an error. Fall back to the old
    assumption only when no marker is found.
    """

    for candidate in config_file.parents:
        if sum((candidate / marker).exists() for marker in ROOT_MARKERS) >= 2:
            return candidate
    return config_file.parent.parent


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration and fail with a useful message on errors."""

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration at {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")
    config["_config_path"] = str(path)
    config["_project_root"] = str(find_project_root(path))
    return config


def project_root(config: Dict[str, Any]) -> Path:
    """Return the project root inferred from the config file location."""

    return Path(config["_project_root"])


def resolve_config_path(config: Dict[str, Any], value: str | Path) -> Path:
    """Resolve a config path relative to the project root unless absolute."""

    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root(config) / path


def get_device(config: Dict[str, Any], requested: str | None = None) -> str:
    """Resolve a device preference to a concrete device string.

    Delegates to :func:`src.utils.device.resolve_device` so every entry point
    agrees: ``auto`` prefers CUDA, then Apple MPS, then CPU, and an explicit
    request for an absent accelerator degrades to CPU instead of crashing
    mid-run. This used to resolve ``auto`` to CUDA-or-CPU only, which silently
    left Apple Silicon on the CPU while the main pipeline used MPS.
    """

    from src.utils.device import resolve_device

    configured = requested or config.get("inference", {}).get("device", "auto")
    return str(resolve_device(str(configured)))
