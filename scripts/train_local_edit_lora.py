#!/usr/bin/env python3
"""Train RobustLens on authentic and minor local-edit images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.finetune.train_lora import train
from src.utils.config import load_config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lora_finetune.yaml")
    parser.add_argument("--mode", choices=("head_only", "lora"), help="Override training.mode")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), help="Override device")
    args = parser.parse_args(argv)
    result = train(load_config(args.config), mode=args.mode, device=args.device)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
