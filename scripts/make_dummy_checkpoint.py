#!/usr/bin/env python3
"""Create a RANDOMLY INITIALISED checkpoint for testing the plumbing.

WARNING
-------
The checkpoint this writes contains untrained weights. Its predictions are
meaningless and must never be presented as detection results. It exists so you
can exercise the CLI, the Streamlit demo and the tests before a real trained
checkpoint is available.

Why the default is ``resnet18``
-------------------------------
A randomly initialised EfficientNet-B0 is numerically dead in eval mode: its
depthwise convolutions and squeeze-excite gates shrink the signal through 16
layers while untrained BatchNorm running statistics (mean 0, variance 1) never
renormalise it, so the final feature map collapses to a standard deviation
around 1e-14 and every image yields the identical logit. That hides real
pipeline bugs. ResNet-18's residual connections preserve the signal, so its
random weights still produce input-dependent outputs and genuinely exercise
preprocessing, transformation and batching code.

Use ``--pretrained-backbone`` for an ImageNet-trained backbone with a random
head, which varies even more realistically across images.

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
        default="resnet18",
        choices=["efficientnet_b0", "resnet18", "convnext_tiny"],
        help="resnet18 stays responsive to its input with random weights",
    )
    parser.add_argument("--num-classes", type=int, default=1, choices=[1, 2])
    parser.add_argument(
        "--pretrained-backbone",
        action="store_true",
        help="Use ImageNet weights for the backbone with a random head (downloads weights)",
    )
    args = parser.parse_args(argv)

    import torch

    from src.pipeline.model_loader import build_architecture, count_parameters

    model = build_architecture(
        args.architecture, num_classes=args.num_classes, pretrained=args.pretrained_backbone
    )
    if args.architecture == "efficientnet_b0" and not args.pretrained_backbone:
        print(
            "NOTE: a randomly initialised efficientnet_b0 collapses to a constant "
            "output in eval mode, so every image scores identically. Use resnet18 "
            "or --pretrained-backbone if you want input-dependent smoke-test scores.",
            file=sys.stderr,
        )
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": args.architecture,
            "num_classes": args.num_classes,
            "trained": False,
            "warning": "Untrained weights. Predictions are meaningless.",
        },
        output,
    )

    print(f"Wrote UNTRAINED checkpoint to {output}")
    print(f"  architecture: {args.architecture}")
    print(f"  parameters:   {count_parameters(model):,}")
    print(f"  backbone:     {'ImageNet-pretrained' if args.pretrained_backbone else 'random'}")
    print("  WARNING: not trained to detect anything - predictions mean nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
