#!/usr/bin/env python3
"""Evaluate a saved RobustLens local-edit adapter on the held-out test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.finetune.dataset import local_edit_collate
from src.finetune.losses import binary_loss
from src.finetune.model import FineTuneModel
from src.finetune.train_lora import (
    FeatureDataset,
    _cache_features,
    _make_datasets,
    _run_epoch,
    _run_feature_epoch,
)
from src.utils.config import load_config, resolve_config_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lora_finetune.yaml")
    parser.add_argument("--adapter-dir", default=None, help="Defaults to model.output_adapter")
    parser.add_argument("--mode", choices=("head_only", "lora"), default="head_only")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default=None)
    parser.add_argument("--output", default="outputs/lora_finetune/test_metrics.json")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    model_config = config.get("model", {}) or {}
    checkpoint = resolve_config_path(config, model_config.get("checkpoint", "models/pretrained/pytorch_model.pt"))
    adapter_dir = resolve_config_path(config, args.adapter_dir or model_config.get("output_adapter", "models/adapters/local_edit_lora"))
    model = FineTuneModel.from_checkpoint(checkpoint, config, device=args.device, mode=args.mode)
    model.load_saved_adapter(adapter_dir)
    datasets, summary = _make_datasets(config)
    batch_size = max(1, int((config.get("training", {}) or {}).get("batch_size", 2)))
    criterion = binary_loss()
    threshold = float((config.get("evaluation", {}) or {}).get("selection_threshold", 0.5))
    if args.mode == "head_only":
        from torch.utils.data import DataLoader

        raw_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, collate_fn=local_edit_collate, num_workers=0)
        cache_dir = resolve_config_path(config, (config.get("outputs", {}) or {}).get("feature_cache_dir", "outputs/lora_finetune/feature_cache"))
        features, labels, pair_ids, subgroups = _cache_features(
            model, datasets["test"], raw_loader, cache_dir / "test.pt", config
        )
        feature_loader = DataLoader(
            FeatureDataset(features, labels, pair_ids, subgroups),
            batch_size=batch_size,
            shuffle=False,
        )
        metrics = _run_feature_epoch(model, feature_loader, criterion, None, 1, threshold)
    else:
        from torch.utils.data import DataLoader

        loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, collate_fn=local_edit_collate, num_workers=0)
        metrics = _run_epoch(model, loader, criterion, None, 1, threshold)
    result = {"mode": args.mode, "adapter_dir": str(adapter_dir), "test_summary": summary["test"], "metrics": metrics}
    output = resolve_config_path(config, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
