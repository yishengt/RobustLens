#!/usr/bin/env python3
"""Command-line batch inference wrapper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# STEP 1: Make the project package importable.
# ---------------------------------------------------------------------------
# Make ``python scripts/run_inference.py`` work from any current directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    # -----------------------------------------------------------------------
    # STEP 2: Read command-line inputs.
    # -----------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Predict AI-generated probabilities for a directory.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", default="outputs/predictions.json")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # STEP 3: Load the configuration and resolve all paths.
    # -----------------------------------------------------------------------
    # Keep ``--help`` usable even before the optional ML dependencies are installed.
    from src.inference.predict import predict_directory
    from src.utils.config import load_config, resolve_config_path

    config = load_config(args.config)

    # -----------------------------------------------------------------------
    # STEP 4: Run inference on every supported image in the input directory.
    # -----------------------------------------------------------------------
    results = predict_directory(
        resolve_config_path(config, args.input_dir),
        resolve_config_path(config, args.checkpoint),
        config,
        resolve_config_path(config, args.output),
    )

    # -----------------------------------------------------------------------
    # STEP 5: Print the JSON predictions for the user or another process.
    # -----------------------------------------------------------------------
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
