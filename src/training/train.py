"""Training loop and command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
from torch import nn

from src.data.dataset import make_dataloaders
from src.evaluation.evaluate import compute_metrics
from src.models.classifier import build_model, count_parameters
from src.utils.checkpoint import save_checkpoint
from src.utils.config import get_device, load_config, resolve_config_path
from src.utils.seed import seed_everything


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    """Run one binary-logit training epoch and return mean loss."""

    model.train()
    running_loss = 0.0
    sample_count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images).reshape(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = images.shape[0]
        running_loss += float(loss.item()) * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise ValueError("Training loader yielded no images")
    return running_loss / sample_count


def validate(
    model: nn.Module,
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: str,
    threshold: float,
) -> Dict[str, Any]:
    """Run validation and return loss plus classification metrics."""

    model.eval()
    running_loss = 0.0
    sample_count = 0
    labels = []
    probabilities = []
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device, non_blocking=True)
            batch_labels_device = batch_labels.float().to(device, non_blocking=True)
            logits = model(images).reshape(-1)
            loss = criterion(logits, batch_labels_device)
            batch_size = images.shape[0]
            running_loss += float(loss.item()) * batch_size
            sample_count += batch_size
            labels.extend(batch_labels.cpu().tolist())
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
    if sample_count == 0:
        raise ValueError("Validation loader yielded no images")
    metrics = compute_metrics(labels, probabilities, threshold=threshold)
    metrics["loss"] = running_loss / sample_count
    return metrics


def train(config: Dict[str, Any], force_resplit: bool = False) -> Dict[str, Any]:
    """Train, checkpoint, and return the best validation metrics."""

    seed_everything(int(config.get("seed", 42)))
    device = get_device(config)
    train_loader, val_loader, train_records, val_records = make_dataloaders(
        config, force_resplit=force_resplit
    )
    model_config = config.get("model", {})
    model_name = str(model_config.get("name", "efficientnet_b0"))
    model = build_model(
        model_name=model_name,
        pretrained=bool(model_config.get("pretrained", False)),
        num_classes=int(model_config.get("num_classes", 1)),
    ).to(device)
    parameter_count = count_parameters(model)
    if parameter_count >= 2_000_000_000:
        raise ValueError(f"Model has {parameter_count:,} trainable parameters; limit is 2B")

    training_config = config.get("training", {})
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-4)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )
    epochs = int(training_config.get("epochs", 10))
    threshold = float(training_config.get("threshold", 0.5))
    checkpoint_dir = resolve_config_path(config, config["paths"].get("checkpoint_dir", "checkpoints"))
    image_size = int(config.get("data", {}).get("image_size", 224))
    best_auc = float("-inf")
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        validation = validate(model, val_loader, criterion, device, threshold)
        validation["train_loss"] = train_loss
        validation["epoch"] = epoch
        history.append(validation)
        auc = validation.get("roc_auc")
        score = float(auc) if auc is not None else float(validation.get("f1", 0.0))
        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            epoch,
            validation,
            model_name,
            image_size,
        )
        if score > best_auc:
            best_auc = score
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                epoch,
                validation,
                model_name,
                image_size,
            )
        print(
            f"epoch={epoch}/{epochs} train_loss={train_loss:.4f} "
            f"val_loss={validation['loss']:.4f} val_f1={validation['f1']:.4f} "
            f"val_auc={validation['roc_auc']}"
        )

    result = {
        "device": device,
        "model": model_name,
        "train_samples": len(train_records),
        "validation_samples": len(val_records),
        "parameters": parameter_count,
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
        "history": history,
    }
    history_path = checkpoint_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the robust image detector.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--force-resplit", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    result = train(config, force_resplit=args.force_resplit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
