"""Checkpoint persistence and model restoration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from src.models.classifier import build_model


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Dict[str, Any],
    model_name: str,
    image_size: int,
) -> None:
    """Save model, optimizer, and experiment metadata."""

    checkpoint_path = Path(path).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics,
        "model_name": model_name,
        "image_size": int(image_size),
    }
    torch.save(payload, checkpoint_path)


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
