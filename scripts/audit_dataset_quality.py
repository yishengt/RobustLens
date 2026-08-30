#!/usr/bin/env python3
"""Audit the local-edit dataset and write the quarantine manifest.

Reports label conflicts, duplicates across splits, near-duplicates and the file
format distribution per class. Conflicting pairs are quarantined whole -- the
audit never guesses which side carries the correct label.

    python scripts/audit_dataset_quality.py
    python scripts/audit_dataset_quality.py --data-dir data/local_edits --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.finetune.data_quality import audit_splits  # noqa: E402
from src.finetune.dataset import discover_split  # noqa: E402

SPLITS = ("train", "validation", "test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/local_edits")
    parser.add_argument("--masks-dir", default="masks")
    parser.add_argument("--output-dir", default="outputs/data_quality")
    parser.add_argument(
        "--no-near-duplicates",
        action="store_true",
        help="Skip the near-duplicate scan, which is the slowest check",
    )
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON")
    return parser


def _resolve(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _resolve(args.data_dir)
    if not root.is_dir():
        print(f"Dataset directory not found: {root}", file=sys.stderr)
        return 1

    masks = _resolve(args.masks_dir) if args.masks_dir else None
    summaries = []
    for split in SPLITS:
        split_root = root / split
        if not split_root.is_dir():
            print(f"Skipping missing split: {split_root}", file=sys.stderr)
            continue
        summaries.append(discover_split(split_root, split, masks))
    if not summaries:
        print(f"No splits found under {root}", file=sys.stderr)
        return 1

    report = audit_splits(
        summaries, root=root, check_near_duplicates=not args.no_near_duplicates
    )

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = report.save(output_dir / "data_quality.json")
    quarantine_path = output_dir / "quarantine.json"
    quarantine_path.write_text(
        json.dumps(report.quarantine_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
        return 0

    for summary in summaries:
        print(
            f"{summary.split:<11} images={summary.valid_images:<6} "
            f"groups={summary.group_count:<6} corrupt={summary.corrupted_images:<4} "
            f"missing_masks={summary.missing_masks}"
        )

    print(f"\nfiles hashed             {report.total_files}")
    print(f"distinct file hashes     {report.distinct_file_hashes}")
    print(f"label conflicts          {len(report.conflicts)}")
    print(f"files quarantined        {len(report.quarantined_paths)}")
    print(f"cross-split duplicates   {len(report.cross_split_duplicates)}")
    print(f"near-duplicate pairs     {len(report.near_duplicate_pairs)}")
    print(f"undecodable files        {len(report.undecodable)}")

    if report.conflicts:
        print("\nConflicting labels (both sides excluded by default):")
        for conflict in report.conflicts:
            print(f"  {conflict.reason}  splits={','.join(conflict.splits)}")
            for member in conflict.members:
                relative = member.path.relative_to(root)
                print(f"    label={member.label} group={member.group_id}  {relative}")

    print("\nFile format by class:")
    for label_name, counts in sorted(report.format_counts.items()):
        formatted = "  ".join(f"{ext}={n}" for ext, n in sorted(counts.items()))
        print(f"  {label_name:<12} {formatted}")

    print(f"\nWrote {report_path}")
    print(f"Wrote {quarantine_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
