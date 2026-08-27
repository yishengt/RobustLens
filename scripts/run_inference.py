#!/usr/bin/env python3
"""Command-line batch inference wrapper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make ``python scripts/run_inference.py`` work from any current directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main() -> None:
    parser = argparse.ArgumentParser(description="Predict AI-generated probabilities for a directory.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", default="outputs/predictions.json")
    args = parser.parse_args()

    # Keep ``--help`` usable even before the optional ML dependencies are installed.
    from src.inference.predict import predict_directory
    from src.utils.config import load_config, resolve_config_path

    config = load_config(args.config)
    results = predict_directory(
        resolve_config_path(config, args.input_dir),
        resolve_config_path(config, args.checkpoint),
        config,
        resolve_config_path(config, args.output),
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
