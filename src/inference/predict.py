"""Single-image and batch inference utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.augmentations import build_eval_transform
from src.data.dataset import DEFAULT_EXTENSIONS, list_image_files, read_image, validate_image
from src.utils.checkpoint import load_model_from_checkpoint
from src.utils.config import get_device


class ImageInferenceDataset(Dataset[Tuple[torch.Tensor, str]]):
    """Unlabelled dataset used by the batch inference entry point."""

    def __init__(self, image_paths: Sequence[Path], transform: Any):
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        path = self.image_paths[index]
        try:
            tensor = self.transform(image=read_image(path))["image"]
        except (FileNotFoundError, ValueError, OSError, KeyError) as exc:
            raise RuntimeError(f"Could not load inference image '{path}': {exc}") from exc
        return tensor, str(path)


@torch.no_grad()
def predict_pil_image(
    image: Image.Image, model: torch.nn.Module, transform: Any, device: str
) -> float:
    """Return the AI-generated probability for one PIL image."""

    if image is None:
        raise ValueError("An image is required for inference")
    array = np.asarray(image.convert("RGB"))
    tensor = transform(image=array)["image"].unsqueeze(0).to(device)
    return float(torch.sigmoid(model(tensor).reshape(-1))[0].item())


def load_detector(config: Dict[str, Any], checkpoint_path: str | Path):
    """Load the configured model and its clean inference transform."""

    device = get_device(config)
    model_config = config.get("model", {})
    model, metadata = load_model_from_checkpoint(
        checkpoint_path,
        model_name=str(model_config.get("name", "efficientnet_b0")),
        device=device,
        pretrained=bool(model_config.get("pretrained", False)),
        num_classes=int(model_config.get("num_classes", 1)),
    )
    return model, build_eval_transform(config), device, metadata


def predict_directory(
    image_dir: str | Path,
    checkpoint_path: str | Path,
    config: Dict[str, Any],
    output_path: str | Path | None = None,
) -> List[Dict[str, Any]]:
    """Run batched inference over a directory and optionally write JSON."""

    directory = Path(image_dir).expanduser()
    configured_extensions = {
        str(extension).lower()
        for extension in config.get("data", {}).get("extensions", DEFAULT_EXTENSIONS)
    }
    image_paths = list_image_files(directory, configured_extensions)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in input directory: {directory}")
    invalid = []
    for path in image_paths:
        try:
            validate_image(path)
        except (FileNotFoundError, ValueError) as exc:
            invalid.append(str(exc))
    if invalid:
        preview = "\n".join(invalid[:10])
        raise ValueError(f"Input directory contains invalid images:\n{preview}")

    model, transform, device, _ = load_detector(config, checkpoint_path)
    loader = DataLoader(
        ImageInferenceDataset(image_paths, transform),
        batch_size=int(config.get("inference", {}).get("batch_size", 32)),
        shuffle=False,
        num_workers=int(config.get("data", {}).get("num_workers", 0)),
        pin_memory=bool(config.get("data", {}).get("pin_memory", True)),
    )
    results: List[Dict[str, Any]] = []
    for images, paths in loader:
        probabilities = torch.sigmoid(model(images.to(device)).reshape(-1)).cpu().tolist()
        results.extend(
            {"image_path": path, "pred": float(probability)}
            for path, probability in zip(paths, probabilities)
        )

    if output_path is not None:
        output = Path(output_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
    return results
