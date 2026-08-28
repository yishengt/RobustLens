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
    args = parser.parse_args()

    import warnings

    warnings.filterwarnings("ignore")

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
    print(f"\nRobustLens parameter budget")
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

    print(f"\n  Trainable : {counts['trainable']:>15,}  ({counts['trainable']/total:.2%} of model)")
    print(f"  Frozen    : {counts['frozen']:>15,}  ({counts['frozen']/total:.2%} of model)")

    # -----------------------------------------------------------------------
    # STEP 4: Verdict against the competition limit.
    # -----------------------------------------------------------------------
    print(f"\n  Limit     : {LIMIT:>15,}")
    print(f"  Headroom  : {LIMIT - total:>15,}  ({total/LIMIT:.1%} of budget used)")
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
