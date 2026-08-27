"""Configuration loading and path resolution helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


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
    config["_project_root"] = str(path.parent.parent)
    return config


def project_root(config: Dict[str, Any]) -> Path:
    """Return the project root inferred from the config file location."""

    return Path(config["_project_root"])


def resolve_config_path(config: Dict[str, Any], value: str | Path) -> Path:
    """Resolve a config path relative to the project root unless absolute."""

    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root(config) / path


def get_device(config: Dict[str, Any], requested: str | None = None) -> str:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""

    import torch

    configured = requested or config.get("training", {}).get("device", "auto")
    if configured != "auto":
        return str(configured)
    return "cuda" if torch.cuda.is_available() else "cpu"
