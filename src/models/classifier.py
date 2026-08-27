"""Lightweight torchvision image classifier."""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from torchvision import models


def _efficientnet(pretrained: bool) -> Any:
    try:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        return models.efficientnet_b0(weights=weights)
    except (AttributeError, TypeError):
        return models.efficientnet_b0(pretrained=pretrained)


def _convnext(pretrained: bool) -> Any:
    try:
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        return models.convnext_tiny(weights=weights)
    except (AttributeError, TypeError):
        return models.convnext_tiny(pretrained=pretrained)


def build_model(model_name: str = "efficientnet_b0", pretrained: bool = False, num_classes: int = 1):
    """Create a binary classifier with one output logit by default."""

    if int(num_classes) != 1:
        raise ValueError("This detector is binary and requires num_classes=1")
    normalized_name = model_name.lower().replace("-", "_")
    if normalized_name == "efficientnet_b0":
        model = _efficientnet(pretrained)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif normalized_name in {"convnext_tiny", "convnexttiny"}:
        model = _convnext(pretrained)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose efficientnet_b0 or convnext_tiny."
        )
    return model


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
