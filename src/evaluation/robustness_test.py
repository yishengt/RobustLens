"""Clean-versus-transformed validation evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from src.data.augmentations import build_robustness_transform, robustness_cases
from src.data.dataset import load_data_splits, make_eval_loader
from src.evaluation.evaluate import compute_metrics, predict_loader
from src.utils.checkpoint import load_model_from_checkpoint
from src.utils.config import get_device, load_config, resolve_config_path
from src.utils.seed import seed_everything


def evaluate_robustness(
    config: Dict[str, Any], checkpoint_path: str | Path
) -> Dict[str, Any]:
    """Evaluate every configured transformation and report deltas from clean."""

    seed_everything(int(config.get("seed", 42)))
    device = get_device(config)
    _, validation_records = load_data_splits(config)
    model_config = config.get("model", {})
    model, metadata = load_model_from_checkpoint(
        checkpoint_path,
        model_name=str(model_config.get("name", "efficientnet_b0")),
        device=device,
        pretrained=bool(model_config.get("pretrained", False)),
        num_classes=int(model_config.get("num_classes", 1)),
    )
    threshold = float(config.get("inference", {}).get("threshold", 0.5))
    results: Dict[str, Any] = {}
    for case in robustness_cases(config):
        loader = make_eval_loader(
            validation_records,
            config,
            build_robustness_transform(config, case),
        )
        labels, probabilities = predict_loader(model, loader, device)
        results[case] = compute_metrics(labels, probabilities, threshold=threshold)

    clean = results["clean"]
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    for case, metrics in results.items():
        if case == "clean":
            continue
        metrics["delta_from_clean"] = {
            name: (
                None
                if metrics.get(name) is None or clean.get(name) is None
                else float(metrics[name] - clean[name])
            )
            for name in metric_names
        }
    return {
        "checkpoint": str(Path(checkpoint_path).expanduser()),
        "device": device,
        "validation_samples": len(validation_records),
        "results": results,
        "checkpoint_epoch": metadata.get("epoch"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate robustness to image transformations.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--output", default="outputs/robustness.json")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = resolve_config_path(config, args.checkpoint)
    report = evaluate_robustness(config, checkpoint)
    output = resolve_config_path(config, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
