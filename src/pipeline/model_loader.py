"""Stage 5: checkpoint loading.

The pipeline is inference-only. A real trained checkpoint is required; when one
is missing or unusable this module raises :class:`ModelSetupError` with setup
instructions rather than returning an untrained model that would emit
meaningless predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models

from src.utils.device import describe_device, device_from_config

# Architectures small enough for a hackathon and far below the 2B parameter cap.
SUPPORTED_ARCHITECTURES: Tuple[str, ...] = ("efficientnet_b0", "resnet18", "convnext_tiny")
MAX_PARAMETERS = 2_000_000_000

SETUP_HINT = (
    "This project does not train models. Place a trained checkpoint at the "
    "configured path (default: checkpoints/best.pt) or pass --checkpoint. "
    "See models/README.md for the expected checkpoint format."
)


class ModelSetupError(RuntimeError):
    """Raised when no usable model checkpoint could be loaded."""


@dataclass
class ModelBundle:
    """A loaded model together with everything inference needs to know."""

    model: nn.Module
    device: torch.device
    architecture: str
    num_classes: int
    num_parameters: int
    checkpoint_path: str
    ai_class_index: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serialisable description for logs and the demo UI."""

        return {
            "architecture": self.architecture,
            "num_classes": self.num_classes,
            "num_parameters": self.num_parameters,
            "parameters_millions": round(self.num_parameters / 1e6, 2),
            "under_2b_parameter_limit": self.num_parameters < MAX_PARAMETERS,
            "checkpoint_path": self.checkpoint_path,
            "device": describe_device(self.device),
        }


def normalise_architecture(name: str) -> str:
    """Normalise user-facing architecture spellings to the canonical name."""

    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "efficientnet_b0": "efficientnet_b0",
        "efficientnetb0": "efficientnet_b0",
        "resnet18": "resnet18",
        "resnet_18": "resnet18",
        "convnext_tiny": "convnext_tiny",
        "convnexttiny": "convnext_tiny",
    }
    if key not in aliases:
        raise ModelSetupError(
            f"Unsupported model architecture '{name}'. "
            f"Supported architectures: {', '.join(SUPPORTED_ARCHITECTURES)}."
        )
    return aliases[key]


def build_architecture(
    name: str = "efficientnet_b0", num_classes: int = 1, pretrained: bool = False
) -> nn.Module:
    """Create a lightweight backbone with a fresh binary classification head.

    ``num_classes=1`` gives a single logit read through a sigmoid;
    ``num_classes=2`` gives a softmax pair. Both checkpoint styles are common,
    so both are supported.
    """

    architecture = normalise_architecture(name)
    num_classes = int(num_classes)
    if num_classes not in (1, 2):
        raise ModelSetupError(
            f"model.num_classes must be 1 (sigmoid) or 2 (softmax), got {num_classes}"
        )

    if architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:  # convnext_tiny
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    return model


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters, used to prove the 2B-parameter limit is met."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad or not trainable_only
    )


def _load_payload(path: Path, device: torch.device) -> Any:
    """Load a checkpoint file, tolerating both weights_only defaults."""

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only argument
        return torch.load(path, map_location=device)


def _extract_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    """Pull the tensor state dict out of the common checkpoint layouts."""

    candidate: Any = payload
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model", "net", "weights"):
            value = payload.get(key)
            if isinstance(value, dict) and value:
                candidate = value
                break

    if isinstance(candidate, nn.Module):  # a whole pickled model
        candidate = candidate.state_dict()

    if not isinstance(candidate, dict) or not candidate:
        raise ModelSetupError(
            "Checkpoint does not contain a state dict. Expected either a raw "
            "state_dict or a dict with a 'model_state_dict'/'state_dict' key."
        )
    if not all(isinstance(value, torch.Tensor) for value in candidate.values()):
        raise ModelSetupError(
            "Checkpoint state dict contains non-tensor values; it does not look "
            "like a model checkpoint."
        )

    # Strip wrapper prefixes left behind by DataParallel / Lightning.
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in candidate.items():
        name = str(key)
        for prefix in ("module.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        cleaned[name] = value
    return cleaned


def infer_num_classes(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    """Infer the head width from the last 1-D bias or 2-D weight tensor."""

    for key in reversed(list(state_dict)):
        tensor = state_dict[key]
        if key.endswith("bias") and tensor.dim() == 1 and tensor.numel() in (1, 2):
            return int(tensor.numel())
        if key.endswith("weight") and tensor.dim() == 2 and tensor.shape[0] in (1, 2):
            return int(tensor.shape[0])
    return None


def load_model(
    checkpoint_path: str | Path,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[str | torch.device] = None,
) -> ModelBundle:
    """Load a checkpoint into a lightweight backbone and return a ready bundle.

    Raises :class:`ModelSetupError` with an actionable message when the file is
    missing, unreadable, or does not match the configured architecture.
    """

    config = config or {}
    model_config = config.get("model", {}) or {}

    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        raise ModelSetupError(f"Model checkpoint not found: {path}\n{SETUP_HINT}")
    if not path.is_file():
        raise ModelSetupError(f"Model checkpoint path is not a file: {path}\n{SETUP_HINT}")
    if path.stat().st_size == 0:
        raise ModelSetupError(f"Model checkpoint is empty (0 bytes): {path}\n{SETUP_HINT}")

    torch_device = (
        torch.device(device) if device is not None else device_from_config(config)
    )

    try:
        payload = _load_payload(path, torch_device)
    except ModelSetupError:
        raise
    except Exception as exc:  # torch raises a wide range of unpickling errors
        raise ModelSetupError(
            f"Could not read checkpoint '{path}': {type(exc).__name__}: {exc}\n{SETUP_HINT}"
        ) from exc

    state_dict = _extract_state_dict(payload)
    metadata = {
        key: value
        for key, value in (payload.items() if isinstance(payload, dict) else [])
        if isinstance(value, (str, int, float, bool))
    }

    # A checkpoint may record what it was trained as; that wins over the config.
    architecture = str(
        metadata.get("model_name")
        or metadata.get("architecture")
        or model_config.get("name", "efficientnet_b0")
    )
    num_classes = infer_num_classes(state_dict) or int(model_config.get("num_classes", 1))

    model = build_architecture(architecture, num_classes=num_classes, pretrained=False)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ModelSetupError(
            f"Checkpoint '{path}' does not match architecture "
            f"'{normalise_architecture(architecture)}' with {num_classes} output class(es).\n"
            f"Set model.name in your config to the architecture the checkpoint was "
            f"trained with (one of {', '.join(SUPPORTED_ARCHITECTURES)}).\n"
            f"Details: {exc}"
        ) from exc

    num_parameters = count_parameters(model)
    limit = int(model_config.get("max_parameters", MAX_PARAMETERS))
    if num_parameters >= limit:
        raise ModelSetupError(
            f"Model has {num_parameters:,} parameters, at or above the "
            f"{limit:,} parameter limit for this task."
        )

    model.to(torch_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return ModelBundle(
        model=model,
        device=torch_device,
        architecture=normalise_architecture(architecture),
        num_classes=num_classes,
        num_parameters=num_parameters,
        checkpoint_path=str(path),
        ai_class_index=int(model_config.get("ai_class_index", 1)),
        metadata=metadata,
    )
