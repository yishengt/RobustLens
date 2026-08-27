"""Per-image error analysis with pixel-derived diagnostic features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import pandas as pd

from src.data.dataset import ImageRecord, make_dataloaders
from src.evaluation.evaluate import predict_loader
from src.features.engineering import FEATURE_NAMES, records_to_feature_frame
from src.utils.checkpoint import load_model_from_checkpoint
from src.utils.config import get_device, load_config, resolve_config_path


def build_error_table(
    records: Sequence[ImageRecord],
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Join predictions and engineered pixel features into a review table."""

    if not len(records) == len(labels) == len(probabilities):
        raise ValueError("Records, labels, and probabilities must have equal lengths")
    feature_frame = records_to_feature_frame(records)
    frame = feature_frame.copy()
    frame["observed_label"] = [int(label) for label in labels]
    frame["pred"] = [float(probability) for probability in probabilities]
    frame["predicted_label"] = (frame["pred"] >= float(threshold)).astype(int)
    frame["correct"] = frame["label"] == frame["predicted_label"]
    frame["confidence"] = frame["pred"].apply(lambda value: max(float(value), 1.0 - float(value)))
    frame["error_type"] = "correct"
    frame.loc[(frame["label"] == 0) & (frame["predicted_label"] == 1), "error_type"] = "false_positive"
    frame.loc[(frame["label"] == 1) & (frame["predicted_label"] == 0), "error_type"] = "false_negative"
    return frame[
        [
            "image_path",
            "label",
            "observed_label",
            "pred",
            "predicted_label",
            "correct",
            "confidence",
            "error_type",
            *FEATURE_NAMES,
        ]
    ]


def summarize_errors(frame: pd.DataFrame) -> Dict[str, Any]:
    """Return compact counts and confidence summaries for a review report."""

    by_error = frame.groupby("error_type", dropna=False).size().to_dict()
    by_label = (
        frame.groupby("label")
        .agg(samples=("label", "size"), accuracy=("correct", "mean"), mean_confidence=("confidence", "mean"))
        .to_dict(orient="index")
    )
    return {
        "num_samples": int(len(frame)),
        "error_counts": {str(key): int(value) for key, value in by_error.items()},
        "by_label": {str(key): {metric: float(value) for metric, value in metrics.items()} for key, metrics in by_label.items()},
        "review_order": "Sort the CSV by ascending confidence, then inspect false positives and false negatives.",
    }


def run_error_analysis(config: Dict[str, Any], checkpoint_path: str | Path) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Run validation predictions and return a review table plus its summary."""

    device = get_device(config)
    _, val_loader, _, validation_records = make_dataloaders(config)
    model_config = config.get("model", {})
    model, _ = load_model_from_checkpoint(
        checkpoint_path,
        model_name=str(model_config.get("name", "efficientnet_b0")),
        device=device,
        pretrained=bool(model_config.get("pretrained", False)),
        num_classes=int(model_config.get("num_classes", 1)),
    )
    labels, probabilities = predict_loader(model, val_loader, device)
    frame = build_error_table(
        validation_records,
        labels.tolist(),
        probabilities.tolist(),
        threshold=float(config.get("inference", {}).get("threshold", 0.5)),
    )
    return frame, summarize_errors(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write per-image detector error analysis.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--output", default="outputs/error_analysis.csv")
    args = parser.parse_args()
    config = load_config(args.config)
    frame, summary = run_error_analysis(config, resolve_config_path(config, args.checkpoint))
    output = resolve_config_path(config, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["correct", "confidence"], ascending=[True, True]).to_csv(output, index=False)
    summary_path = output.with_name(f"{output.stem}_summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
