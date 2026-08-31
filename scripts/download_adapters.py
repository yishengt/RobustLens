#!/usr/bin/env python3
"""Fetch the experimental LoRA adapters from the Hugging Face Hub.

    python scripts/download_adapters.py --list
    python scripts/download_adapters.py --adapter local_edit_smoke
    python scripts/download_adapters.py --all

None of these adapters is used by the production pipeline. Every one comes from
an experiment that was rejected on measurement, and they are published so a
collaborator can verify those rejections rather than take them on trust. To
*run* RobustLens you need only the base checkpoint, which
``scripts/setup.py --checkpoint`` fetches.

Nothing here touches the base checkpoint.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPO_ID = "Dylennnn/techjam"
REPO_URL = f"https://huggingface.co/{REPO_ID}"
DESTINATION = PROJECT_ROOT / "models" / "adapters"

# Files that make up one adapter. The safetensors holds existing LoRA tensors
# only -- never full backbone weights.
ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "classifier_head.pt",
)

ADAPTERS = {
    "robustness_head": (
        "Head-only, 6 generators. ADM recall 0.305 -> 0.863, pixel-space 0.519 -> 0.914. "
        "NOT ADOPTED as the default: no gain on the DALL-E benchmarks"
    ),
    "local_edit_smoke": "Head-only local-edit fine-tune. REJECTED: AUROC 0.510 -> 0.354",
    "consistency_classification_only": "Ablation baseline, BCE loss only",
    "consistency_consistency_mse": "+ logit-MSE consistency loss. REJECTED: no gain",
    "consistency_consistency_kl": "+ symmetric-KL consistency loss. REJECTED: no gain",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--adapter", choices=sorted(ADAPTERS), help="Download one adapter")
    group.add_argument("--all", action="store_true", help="Download every adapter")
    group.add_argument("--list", action="store_true", help="List adapters and exit")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument(
        "--destination",
        default=str(DESTINATION),
        help="Where to place them (default: models/adapters)",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def _print_catalogue() -> None:
    print(f"Adapters published at {REPO_URL}\n")
    for name, description in sorted(ADAPTERS.items()):
        print(f"  {name}\n      {description}")
    print(
        "\nNone of these is used by the production pipeline. Running RobustLens "
        "needs only the base checkpoint:\n  python3 scripts/setup.py --checkpoint"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list or not (args.adapter or args.all):
        _print_catalogue()
        if not (args.adapter or args.all):
            print("\nNothing downloaded. Pass --adapter NAME or --all.")
        return 0

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "huggingface_hub is not installed. Install requirements first:\n"
            "  ./.venv/bin/pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    wanted = sorted(ADAPTERS) if args.all else [args.adapter]
    destination = Path(args.destination).expanduser()
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination

    failures: list[str] = []
    for name in wanted:
        target = destination / name
        target.mkdir(parents=True, exist_ok=True)
        if not args.quiet:
            print(f"\n{name}")
            print(f"  {ADAPTERS[name]}")
        for filename in ADAPTER_FILES:
            remote = f"adapters/{name}/{filename}"
            # Stage into a temporary directory that is always cleaned up, so a
            # download never leaves a cache folder sitting inside models/.
            with tempfile.TemporaryDirectory(prefix="robustlens-adapter-") as staging:
                try:
                    path = hf_hub_download(
                        repo_id=args.repo_id,
                        filename=remote,
                        local_dir=staging,
                    )
                except Exception as exc:  # noqa: BLE001 - report any hub failure plainly
                    failures.append(f"{name}/{filename}: {type(exc).__name__}: {exc}")
                    if not args.quiet:
                        print(f"  FAILED {filename}: {type(exc).__name__}")
                    continue
                destination_file = target / filename
                destination_file.write_bytes(Path(path).read_bytes())
            if not args.quiet:
                size = destination_file.stat().st_size / 1e6
                print(f"  {filename}  ({size:.1f} MB)")

    if failures:
        print("\nSome files could not be downloaded:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        # The repository is public, so no sign-in is needed. A 404/401 here means
        # the repo moved, was renamed, or was switched to private.
        if any("401" in line or "Unauthorized" in line or "RepositoryNotFound" in line
               for line in failures):
            print(
                f"\n{REPO_ID} is normally a PUBLIC repository needing no sign-in, so "
                f"this usually means it was renamed, deleted, or switched to private.\n"
                f"  - check {REPO_URL} in a browser\n"
                f"  - if it is now private, sign in with:  hf auth login\n"
                f"\nEither way, you do NOT need these adapters to run RobustLens -- none "
                f"of them is used by the production pipeline. Fetch the base checkpoint "
                f"instead:\n  python3 scripts/setup.py --checkpoint",
                file=sys.stderr,
            )
        else:
            print(f"\nCheck that {REPO_URL} is reachable.", file=sys.stderr)
        return 1

    if not args.quiet:
        print(
            f"\nDone. Adapters are in {destination}\n\n"
            "These load onto a model restored from the BASE checkpoint; they are\n"
            "not standalone models. The shipped calibration and the 0.69 threshold\n"
            "were fitted for the base checkpoint and do NOT apply to an adapted\n"
            "model, so pass --no-calibration when running one:\n\n"
            "  python scripts/run_inference.py --input-dir IMAGES \\\n"
            f"      --adapter-dir models/adapters/{wanted[0]} \\\n"
            "      --no-calibration --output outputs/predictions.json"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
