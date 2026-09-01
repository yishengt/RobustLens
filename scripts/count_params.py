#!/usr/bin/env python3
"""Print a parameter-count report proving the <2B competition limit is met.

Counts are computed with sum(p.numel() for p in model.parameters()) on the
real module, not quoted from documentation, so the figures in the README are
reproducible by anyone who clones the repo.

Runs without downloading weights by default: models are built from their
published configs, which fixes every layer shape and therefore every
parameter count, while leaving the tensors randomly initialised.

    python scripts/count_params.py
    python scripts/count_params.py --pretrained     # download real weights
    python scripts/count_params.py --checkpoint models/pretrained/pytorch_model.pt

The third form is the one the README quotes. It counts the SHIPPED checkpoint
-- the module that is actually loaded and run -- rather than a model rebuilt
from the default configs, which is a different architecture and therefore a
different number. Use it whenever the claim being made is about what ships.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# STEP 1: Make the project package importable.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LIMIT = 2_000_000_000


def _row(label: str, value: int, total: int) -> str:
    return f"  {label:<34}{value:>15,}   {value / total:>6.1%}"


def _verdict(total: int) -> None:
    """Print the budget lines and exit non-zero if the limit is exceeded."""

    print(f"\n  Limit     : {LIMIT:>15,}")
    print(f"  Headroom  : {LIMIT - total:>15,}  ({total / LIMIT:.1%} of budget used)")
    if total > LIMIT:
        print("\n  FAIL: model exceeds the 2B parameter limit.")
        raise SystemExit(1)
    print("\n  PASS: within the 2B parameter limit.")


def report_checkpoint(checkpoint: Path) -> None:
    """Count the parameters of the checkpoint that actually ships.

    This loads the file through the same code path inference uses, so the
    number reported is the number of parameters that run -- not a figure from
    a model rebuilt out of the default configs, which need not be the same
    architecture as the checkpoint on disk.
    """

    from src.pipeline.model_loader import ModelSetupError, load_model
    from src.utils.config import load_config

    config_path = PROJECT_ROOT / "configs/config.yaml"
    config = load_config(config_path) if config_path.is_file() else {}

    print(f"Loading {checkpoint} ...")
    try:
        bundle = load_model(checkpoint, config, device="cpu")
    except ModelSetupError as exc:
        print(f"\n  Could not load the checkpoint:\n  {exc}")
        raise SystemExit(1) from exc

    total = bundle.num_parameters
    print("\nRobustLens parameter budget (shipped checkpoint)")
    print(f"  Architecture : {bundle.architecture}")
    print(f"  Checkpoint   : {checkpoint}\n")

    print("  COMPONENT                               PARAMS    OF TOTAL")
    print("  " + "-" * 58)
    counted = 0
    for name, module in bundle.model.named_children():
        value = sum(p.numel() for p in module.parameters())
        counted += value
        print(_row(name, value, total))
    if counted != total:
        # Parameters held directly on the model rather than in a child module.
        print(_row("(other)", total - counted, total))
    print("  " + "-" * 58)
    print(_row("TOTAL", total, total))

    trainable = sum(p.numel() for p in bundle.model.parameters() if p.requires_grad)
    print(f"\n  Trainable : {trainable:>15,}  (inference freezes every parameter)")
    _verdict(total)
    print(
        "\nCounted with sum(p.numel() for p in model.parameters()) on the loaded\n"
        f"checkpoint. Reproduce with: python scripts/count_params.py --checkpoint {checkpoint}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report parameter counts against the 2B competition limit."
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Download real weights (counts are identical either way)",
    )
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Count the shipped checkpoint instead of a model built from config",
    )
    args = parser.parse_args()

    import warnings

    warnings.filterwarnings("ignore")

    if args.checkpoint is not None:
        report_checkpoint(args.checkpoint)
        return

    from src.models.dual_backbone import (
        DEFAULT_DINOV2,
        DEFAULT_HIDDEN,
        DEFAULT_SIGLIP,
        DualBackboneDetector,
        count_parameters,
    )

    # -----------------------------------------------------------------------
    # STEP 2: Build the model and count every parameter.
    # -----------------------------------------------------------------------
    print("Building model", "with pretrained weights..." if args.pretrained else "from config...")
    model = DualBackboneDetector(
        hidden_dim=args.hidden_dim or DEFAULT_HIDDEN,
        pretrained=args.pretrained,
    )

    counts = count_parameters(model)
    total = counts["total"]

    # The same expression, applied per component, so the parts sum to the whole.
    siglip = sum(p.numel() for p in model.siglip.parameters())
    dinov2 = sum(p.numel() for p in model.dinov2.parameters())
    head = sum(p.numel() for p in model.head.parameters())

    # -----------------------------------------------------------------------
    # STEP 3: Report.
    # -----------------------------------------------------------------------
    print("\nRobustLens parameter budget")
    print(f"  SigLIP2 : {DEFAULT_SIGLIP}")
    print(f"  DINOv2  : {DEFAULT_DINOV2}\n")

    print("  COMPONENT                               PARAMS    OF TOTAL")
    print("  " + "-" * 58)
    print(_row("SigLIP2 vision tower (frozen)", siglip, total))
    print(_row("DINOv2 (frozen)", dinov2, total))
    print(_row("Classification head (trainable)", head, total))
    print("  " + "-" * 58)
    print(_row("TOTAL", total, total))

    assert siglip + dinov2 + head == total, "component counts must sum to the total"

    print(
        f"\n  Trainable : {counts['trainable']:>15,}  ({counts['trainable'] / total:.2%} of model)"
    )
    print(f"  Frozen    : {counts['frozen']:>15,}  ({counts['frozen'] / total:.2%} of model)")

    # -----------------------------------------------------------------------
    # STEP 4: Verdict against the competition limit.
    # -----------------------------------------------------------------------
    print(f"\n  Limit     : {LIMIT:>15,}")
    print(f"  Headroom  : {LIMIT - total:>15,}  ({total / LIMIT:.1%} of budget used)")
    if total > LIMIT:
        print("\n  FAIL: model exceeds the 2B parameter limit.")
        raise SystemExit(1)
    print("\n  PASS: within the 2B parameter limit.")

    print(
        "\nCounted with sum(p.numel() for p in model.parameters()).\n"
        "Reproduce with: python scripts/count_params.py"
    )


if __name__ == "__main__":
    main()
