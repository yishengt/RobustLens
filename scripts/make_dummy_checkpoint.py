#!/usr/bin/env python3
"""Create a RANDOMLY INITIALISED checkpoint for testing the plumbing.

WARNING
-------
The checkpoint this writes contains untrained random weights. Its predictions
are meaningless and must never be presented as detection results. It exists so
you can exercise the CLI, the Streamlit demo and the tests before a real
trained checkpoint is available.

Usage::

    python scripts/make_dummy_checkpoint.py --output checkpoints/dummy.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write an untrained checkpoint for smoke-testing the pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", default="checkpoints/dummy.pt")
    parser.add_argument(
        "--architecture",
        default="efficientnet_b0",
        choices=["efficientnet_b0", "resnet18", "convnext_tiny"],
    )
    parser.add_argument("--num-classes", type=int, default=1, choices=[1, 2])
    args = parser.parse_args(argv)

    import torch

    from src.pipeline.model_loader import build_architecture, count_parameters

    model = build_architecture(args.architecture, num_classes=args.num_classes, pretrained=False)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": args.architecture,
            "num_classes": args.num_classes,
            "trained": False,
            "warning": "Randomly initialised weights. Predictions are meaningless.",
        },
        output,
    )

    print(f"Wrote UNTRAINED checkpoint to {output}")
    print(f"  architecture: {args.architecture}")
    print(f"  parameters:   {count_parameters(model):,}")
    print("  WARNING: random weights - predictions from this file mean nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
