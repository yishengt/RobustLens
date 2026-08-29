"""Grad-CAM explanations for the AI-generated positive logit."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import torch.nn as nn


def _last_convolution(model: Any) -> Any:
    """Find a spatial convolutional layer suitable for Grad-CAM."""

    import torch.nn as nn

    layers = [layer for layer in model.modules() if isinstance(layer, nn.Conv2d)]
    if not layers:
        raise ValueError("Could not find a convolutional feature layer for Grad-CAM")
    return layers[-1]


def explain_image(
    image_path: str | Path,
    model: nn.Module,
    transform: Any,
    device: str,
    output_path: str | Path,
    target_layer: Optional[Any] = None,
) -> float:
    """Save a Grad-CAM overlay and return the AI-generated probability."""

    import cv2
    import numpy as np
    import torch
    from PIL import Image

    from src.data.dataset import read_image

    image_array = read_image(image_path)
    input_tensor = transform(image=image_array)["image"].unsqueeze(0).to(device)
    layer = target_layer or _last_convolution(model)
    activations = []
    gradients = []

    def forward_hook(_: nn.Module, __: Any, output: Any) -> None:
        activations.append(output[0] if isinstance(output, tuple) else output)

    def backward_hook(_: nn.Module, __: Any, grad_output: Any) -> None:
        gradients.append(grad_output[0])

    forward_handle = layer.register_forward_hook(forward_hook)
    backward_handle = layer.register_full_backward_hook(backward_hook)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        logit = model(input_tensor).reshape(-1)[0]
        probability = float(torch.sigmoid(logit.detach()).item())
        logit.backward()
        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")
        activation = activations[-1].detach()
        gradient = gradients[-1].detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1)).squeeze(0).cpu().numpy()
    finally:
        forward_handle.remove()
        backward_handle.remove()

    cam = cam - cam.min()
    cam = cam / max(float(cam.max()), 1e-8)
    cam = cv2.resize(cam.astype(np.float32), (image_array.shape[1], image_array.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(cam * 255.0), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.clip(0.55 * image_array.astype(np.float32) + 0.45 * heatmap, 0, 255).astype(
        np.uint8
    )
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output)
    return probability


def main() -> None:
    parser = argparse.ArgumentParser(description="Save a Grad-CAM detector explanation.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="outputs/gradcam.png")
    args = parser.parse_args()
    from src.inference.predict import load_detector
    from src.utils.config import load_config, resolve_config_path

    config = load_config(args.config)
    model, transform, device, _ = load_detector(
        config, resolve_config_path(config, args.checkpoint)
    )
    probability = explain_image(
        resolve_config_path(config, args.image),
        model,
        transform,
        device,
        resolve_config_path(config, args.output),
    )
    print(f"AI-generated probability: {probability:.6f}")


if __name__ == "__main__":
    main()
