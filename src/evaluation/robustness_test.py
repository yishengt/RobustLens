"""Clean-versus-transformed validation evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

def evaluate_robustness(
    config: Dict[str, Any],
    checkpoint_path: str | Path,
    materialized_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Evaluate every configured transformation and report deltas from clean.

    When ``materialized_root`` is supplied, transformed images and labels are
    loaded from the handoff manifest produced by the transformation owner.
    Without it, the local Albumentations implementation is used as a fallback.
    """

    from src.data.augmentations import (
        build_eval_transform,
        build_robustness_transform,
        robustness_cases,
    )
    from src.data.dataset import load_data_splits, make_eval_loader
    from src.evaluation.evaluate import compute_metrics, predict_loader
    from src.evaluation.materialized import load_materialized_records
    from src.utils.checkpoint import load_model_from_checkpoint
    from src.utils.config import get_device, project_root
    from src.utils.seed import seed_everything

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
    cases = robustness_cases(config)
    results: Dict[str, Any] = {}
    clean_loader = make_eval_loader(
        validation_records,
        config,
        build_eval_transform(config),
    )
    labels, probabilities = predict_loader(model, clean_loader, device)
    results["clean"] = compute_metrics(labels, probabilities, threshold=threshold)

    materialized = None
    manifest_file = None
    if materialized_root is not None:
        materialized_root_path = Path(materialized_root).expanduser().resolve()
        manifest_file = (
            Path(manifest_path).expanduser().resolve()
            if manifest_path is not None
            else materialized_root_path / "manifest.json"
        )
        materialized = load_materialized_records(
            manifest_file,
            validation_records,
            cases[1:],
            project_root(config),
        )

    for case in cases[1:]:
        if materialized is None:
            case_records = validation_records
            case_transform = build_robustness_transform(config, case)
        else:
            case_records = materialized[case]
            case_transform = build_eval_transform(config)
        loader = make_eval_loader(case_records, config, case_transform)
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
        "transformation_source": "materialized_manifest" if materialized is not None else "on_the_fly_fallback",
        "manifest": str(manifest_file) if manifest_file is not None else None,
        "results": results,
        "checkpoint_epoch": metadata.get("epoch"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate robustness to image transformations.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument(
        "--materialized-root",
        default=None,
        help="Root containing a friend-produced manifest.json and transformed images",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional explicit transformation manifest path",
    )
    parser.add_argument("--output", default="outputs/robustness.json")
    args = parser.parse_args()
    from src.utils.config import load_config, resolve_config_path

    config = load_config(args.config)
    checkpoint = resolve_config_path(config, args.checkpoint)
    materialized_root = (
        resolve_config_path(config, args.materialized_root)
        if args.materialized_root
        else None
    )
    manifest = resolve_config_path(config, args.manifest) if args.manifest else None
    report = evaluate_robustness(config, checkpoint, materialized_root, manifest)
    output = resolve_config_path(config, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
