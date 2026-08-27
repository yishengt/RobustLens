"""Clean validation metrics and evaluation CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.dataset import make_dataloaders
from src.utils.checkpoint import load_model_from_checkpoint
from src.utils.config import get_device, load_config, resolve_config_path


def _optional_float(value: Any) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)


def compute_metrics(
    labels: Iterable[int], probabilities: Iterable[float], threshold: float = 0.5
) -> Dict[str, Any]:
    """Compute binary metrics with AI-generated as the positive class."""

    y_true = np.asarray(list(labels), dtype=np.int64)
    y_prob = np.asarray(list(probabilities), dtype=np.float64)
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics for an empty prediction set")
    if y_true.shape != y_prob.shape:
        raise ValueError("Labels and probabilities must have the same length")
    y_pred = (y_prob >= float(threshold)).astype(np.int64)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = None
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _optional_float(auc),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "threshold": float(threshold),
        "num_samples": int(y_true.size),
    }


@torch.no_grad()
def predict_loader(
    model: torch.nn.Module, loader: Iterable[Tuple[torch.Tensor, torch.Tensor]], device: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect labels and AI-generated probabilities from a DataLoader."""

    model.eval()
    labels = []
    probabilities = []
    for images, batch_labels in loader:
        logits = model(images.to(device)).reshape(-1)
        probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        labels.extend(batch_labels.cpu().numpy().tolist())
    return np.asarray(labels, dtype=np.int64), np.asarray(probabilities, dtype=np.float64)


def evaluate_checkpoint(config: Dict[str, Any], checkpoint_path: str | Path) -> Dict[str, Any]:
    """Evaluate a checkpoint on the clean validation split."""

    device = get_device(config)
    _, val_loader, _, _ = make_dataloaders(config)
    model_config = config.get("model", {})
    model, metadata = load_model_from_checkpoint(
        checkpoint_path,
        model_name=str(model_config.get("name", "efficientnet_b0")),
        device=device,
        pretrained=bool(model_config.get("pretrained", False)),
        num_classes=int(model_config.get("num_classes", 1)),
    )
    labels, probabilities = predict_loader(model, val_loader, device)
    metrics = compute_metrics(
        labels,
        probabilities,
        threshold=float(config.get("training", {}).get("threshold", 0.5)),
    )
    metrics["device"] = device
    metrics["checkpoint"] = str(Path(checkpoint_path).expanduser())
    if metadata.get("epoch") is not None:
        metrics["checkpoint_epoch"] = metadata["epoch"]
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a detector checkpoint.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--output", default=None, help="Optional metrics JSON path")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = resolve_config_path(config, args.checkpoint)
    metrics = evaluate_checkpoint(config, checkpoint)
    output = Path(args.output) if args.output else None
    if output is not None:
        output = resolve_config_path(config, output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
