"""Checkpoint persistence and model restoration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from src.models.classifier import build_model


def _torch_load(path: Path, device: str) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    model_name: str,
    device: str,
    pretrained: bool = False,
    num_classes: int = 1,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Build a model and load a checkpoint, with actionable file errors."""

    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    try:
        payload = _torch_load(path, device)
        state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else None
        if not isinstance(state_dict, dict):
            raise ValueError("checkpoint does not contain a model_state_dict mapping")
        model = build_model(model_name, pretrained=pretrained, num_classes=num_classes)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise ValueError(f"Could not load model checkpoint '{path}': {exc}") from exc
    metadata = payload if isinstance(payload, dict) else {}
    return model, metadata
