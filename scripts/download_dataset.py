#!/usr/bin/env python3
"""Download parquet shards of the SID_Set benchmark from the Hugging Face Hub.

SID_Set (Social media Image Detection dataSet) is public and CC-BY-4.0, so no
Hugging Face token is required. The full dataset is ~140 GB, so this script
downloads a configurable number of shards rather than the whole thing.

Splits available in the Hub repository:

===========  =======  ========  ============
split        shards   images    size
===========  =======  ========  ============
train        249      210,000   ~123 GB
validation   34       30,000    ~17 GB
===========  =======  ========  ============

The 60,000-image test split is deliberately withheld by the authors; see
https://github.com/hzlsaber/SIDA for access.

Examples::

    # ~2 GB evaluation sample (default)
    python scripts/download_dataset.py

    # list what is available without downloading
    python scripts/download_dataset.py --list

    # ten training shards as well
    python scripts/download_dataset.py --split train --shards 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPO_ID = "saberzl/SID_Set"
SHARD_COUNTS = {"train": 249, "validation": 34}
APPROX_SHARD_GB = 0.495


def shard_name(split: str, index: int) -> str:
    """Return the Hub path of one parquet shard."""

    total = SHARD_COUNTS[split]
    if not 0 <= index < total:
        raise ValueError(f"{split} shard index must be within 0-{total - 1}, got {index}")
    return f"data/{split}-{index:05d}-of-{total:05d}.parquet"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a sample of the SID_Set dataset for evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=sorted(SHARD_COUNTS),
        help="Which split to pull shards from",
    )
    parser.add_argument(
        "--shards", type=int, default=4, help="How many shards to download (~495 MB each)"
    )
    parser.add_argument(
        "--start", type=int, default=0, help="First shard index to download"
    )
    parser.add_argument(
        "--output",
        default="data/sid_set",
        help="Directory to place the parquet shards in",
    )
    parser.add_argument(
        "--list", action="store_true", help="Show split sizes and exit without downloading"
    )
    parser.add_argument(
        "--yes", action="store_true", help="Skip the download size confirmation prompt"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        print(f"Dataset: {REPO_ID}  (public, CC-BY-4.0, no token required)")
        for split, count in sorted(SHARD_COUNTS.items()):
            print(f"  {split:11} {count:3d} shards  ~{count * APPROX_SHARD_GB:6.1f} GB")
        print("  test         withheld by the authors; see https://github.com/hzlsaber/SIDA")
        print("\nLabels: 0 = real, 1 = fully synthetic, 2 = tampered")
        return 0

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "huggingface_hub is required. Install it with:\n"
            "    pip install huggingface_hub",
            file=sys.stderr,
        )
        return 2

    if args.shards <= 0:
        print("--shards must be positive", file=sys.stderr)
        return 2

    total = SHARD_COUNTS[args.split]
    indices = list(range(args.start, min(args.start + args.shards, total)))
    if not indices:
        print(f"No shards selected: {args.split} has {total} shards.", file=sys.stderr)
        return 2

    estimated_gb = len(indices) * APPROX_SHARD_GB
    output_dir = Path(args.output).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    print(f"Dataset:   {REPO_ID}")
    print(f"Split:     {args.split} ({len(indices)} of {total} shards)")
    print(f"Estimated: ~{estimated_gb:.1f} GB")
    print(f"Output:    {output_dir}")

    if not args.yes:
        answer = input("Proceed with the download? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Cancelled.")
            return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for position, index in enumerate(indices, start=1):
        name = shard_name(args.split, index)
        print(f"[{position}/{len(indices)}] {name}", file=sys.stderr)
        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=name,
                repo_type="dataset",
                local_dir=str(output_dir),
            )
        except Exception as exc:  # network, auth and hub errors all surface here
            print(f"Download failed for {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        downloaded.append(Path(path))

    actual_gb = sum(path.stat().st_size for path in downloaded) / 1e9
    print(f"\nDownloaded {len(downloaded)} shard(s), {actual_gb:.2f} GB, to {output_dir}")
    print("Next: evaluate a checkpoint against these labelled images with")
    print(f"    python scripts/evaluate_dataset.py --data-dir {args.output} \\")
    print("        --checkpoint checkpoints/best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
