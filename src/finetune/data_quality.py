"""Deterministic data-quality audit for the local-edit dataset.

The upstream editing dataset occasionally emits an "edited" image that is
byte-identical to the source it was derived from. Both files are then kept --
the source labelled ``authentic`` and the edit labelled ``ai_edited`` -- so the
model is asked to learn that one pixel array belongs to both classes. Six such
pairs exist in ``data/local_edits``, five in train and one in test.

This module finds them and every related defect, and it deliberately does *not*
guess which side of a conflicting pair is correct. A conflict is quarantined
whole: both files are excluded, the pair is recorded with enough evidence to
adjudicate by hand, and the exclusion count is written into the dataset
manifest. Including them again requires an explicit configuration change, not a
silent default.

The audit is deterministic: results depend only on file contents, so two runs on
the same data produce byte-identical reports.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.finetune.dataset import DatasetSummary, ImageRecord

# Conflict reasons, most severe first.
REASON_IDENTICAL_BYTES = "identical_file_bytes_conflicting_labels"
REASON_IDENTICAL_PIXELS = "identical_pixels_conflicting_labels"
REASON_CROSS_SPLIT_DUPLICATE = "identical_content_in_multiple_splits"

# Hamming distance on the 64-bit difference hash below which two images are
# treated as near-duplicates. 0 means visually identical after downsampling.
NEAR_DUPLICATE_DISTANCE = 2

_LABEL_NAMES = {0: "authentic", 1: "ai_edited"}


def file_hash(path: Path) -> str:
    """SHA-256 of the raw file bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_hash(path: Path) -> Optional[str]:
    """SHA-256 of the decoded RGB pixels, or ``None`` if undecodable.

    Catches the case a raw file hash misses: the same picture re-encoded, which
    a preparation script that normalises everything to JPEG can easily produce.
    """

    try:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError):
        return None
    digest = hashlib.sha256()
    digest.update(str(pixels.shape).encode("utf-8"))
    digest.update(pixels.tobytes())
    return digest.hexdigest()


def difference_hash(path: Path) -> Optional[int]:
    """64-bit dHash for near-duplicate detection, or ``None`` if undecodable."""

    try:
        with Image.open(path) as image:
            small = image.convert("L").resize((9, 8), Image.BILINEAR)
    except (UnidentifiedImageError, OSError, ValueError):
        return None
    pixels = np.asarray(small, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(left: int, right: int) -> int:
    return int(bin(left ^ right).count("1"))


@dataclass(frozen=True)
class FileFingerprint:
    """Everything the audit knows about one file."""

    path: Path
    split: str
    label: int
    group_id: str
    sha256: str
    pixels_sha256: Optional[str]
    dhash: Optional[int]

    def as_dict(self, root: Optional[Path] = None) -> Dict[str, Any]:
        path = str(self.path.relative_to(root)) if root else str(self.path)
        return {
            "path": path,
            "split": self.split,
            "label": self.label,
            "label_name": _LABEL_NAMES.get(self.label, str(self.label)),
            "group_id": self.group_id,
            "file_hash": self.sha256,
            "image_hash": self.pixels_sha256,
        }


@dataclass(frozen=True)
class Conflict:
    """A set of files that cannot all be correctly labelled."""

    reason: str
    members: Tuple[FileFingerprint, ...]

    @property
    def paths(self) -> Tuple[Path, ...]:
        return tuple(member.path for member in self.members)

    @property
    def splits(self) -> Tuple[str, ...]:
        return tuple(sorted({member.split for member in self.members}))

    @property
    def labels(self) -> Tuple[int, ...]:
        return tuple(sorted({member.label for member in self.members}))

    @property
    def same_group(self) -> bool:
        return len({member.group_id for member in self.members}) == 1

    def as_dict(self, root: Optional[Path] = None) -> Dict[str, Any]:
        return {
            "reason": self.reason,
            "shared_file_hash": self.members[0].sha256,
            "shared_image_hash": self.members[0].pixels_sha256,
            "labels": list(self.labels),
            "label_names": [_LABEL_NAMES.get(v, str(v)) for v in self.labels],
            "splits": list(self.splits),
            "same_group": self.same_group,
            "group_ids": sorted({member.group_id for member in self.members}),
            "members": [member.as_dict(root) for member in self.members],
        }


@dataclass
class DataQualityReport:
    """The audit result, plus the quarantine it implies."""

    root: Optional[Path] = None
    total_files: int = 0
    distinct_file_hashes: int = 0
    conflicts: List[Conflict] = field(default_factory=list)
    cross_split_duplicates: List[Conflict] = field(default_factory=list)
    near_duplicate_pairs: List[Dict[str, Any]] = field(default_factory=list)
    format_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    undecodable: List[str] = field(default_factory=list)

    @property
    def quarantined_paths(self) -> Set[Path]:
        """Every file excluded by default -- both sides of each conflict.

        Both sides go, not one: the audit has no evidence about which label is
        wrong, and dropping the side that happens to sort first would be a guess
        dressed up as a fix.
        """

        paths: Set[Path] = set()
        for conflict in self.conflicts:
            paths.update(conflict.paths)
        return paths

    def as_dict(self) -> Dict[str, Any]:
        root = self.root
        return {
            "root": str(root) if root else None,
            "total_files": self.total_files,
            "distinct_file_hashes": self.distinct_file_hashes,
            "conflict_count": len(self.conflicts),
            "quarantined_file_count": len(self.quarantined_paths),
            "cross_split_duplicate_count": len(self.cross_split_duplicates),
            "near_duplicate_pair_count": len(self.near_duplicate_pairs),
            "undecodable_count": len(self.undecodable),
            "conflicts": [item.as_dict(root) for item in self.conflicts],
            "cross_split_duplicates": [item.as_dict(root) for item in self.cross_split_duplicates],
            "near_duplicate_pairs": self.near_duplicate_pairs,
            "format_counts": self.format_counts,
            "undecodable": self.undecodable,
        }

    def quarantine_manifest(self) -> Dict[str, Any]:
        """The stable list a training run reads, sorted for reproducibility."""

        root = self.root
        entries = []
        for conflict in self.conflicts:
            for member in sorted(conflict.members, key=lambda m: str(m.path)):
                entries.append(
                    {
                        **member.as_dict(root),
                        "reason": conflict.reason,
                        "shared_with": [
                            str(p.relative_to(root)) if root else str(p)
                            for p in conflict.paths
                            if p != member.path
                        ],
                    }
                )
        entries.sort(key=lambda item: item["path"])
        return {
            "root": str(root) if root else None,
            "quarantined_file_count": len(entries),
            "conflict_count": len(self.conflicts),
            "policy": (
                "Both sides of every conflicting pair are excluded from training and "
                "validation by default. The audit does not guess which label is "
                "correct. Set data.include_conflicting_labels to true only after "
                "reviewing these entries by hand."
            ),
            "entries": entries,
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return destination


def fingerprint_records(
    records: Sequence[ImageRecord],
    split: str,
    compute_pixels: bool = True,
    compute_dhash: bool = True,
) -> List[FileFingerprint]:
    """Hash every record. Deterministic and order-independent."""

    fingerprints: List[FileFingerprint] = []
    for record in sorted(records, key=lambda r: str(r.image_path)):
        fingerprints.append(
            FileFingerprint(
                path=record.image_path,
                split=split,
                label=int(record.label),
                group_id=record.group_id,
                sha256=file_hash(record.image_path),
                pixels_sha256=image_hash(record.image_path) if compute_pixels else None,
                dhash=difference_hash(record.image_path) if compute_dhash else None,
            )
        )
    return fingerprints


def audit_splits(
    summaries: Sequence[DatasetSummary],
    root: Optional[str | Path] = None,
    check_near_duplicates: bool = True,
    near_duplicate_distance: int = NEAR_DUPLICATE_DISTANCE,
    compute_pixels: bool = True,
) -> DataQualityReport:
    """Audit every split for label conflicts, duplicates and format imbalance.

    ``compute_pixels`` decodes every image to catch re-encoded duplicates that a
    raw file hash misses. It dominates the runtime, so a training run that only
    needs the byte-identical conflicts can switch it off and still get an exact
    answer for those -- see :func:`quick_conflict_audit`.
    """

    root_path = Path(root).expanduser() if root else None
    report = DataQualityReport(root=root_path)

    fingerprints: List[FileFingerprint] = []
    for summary in summaries:
        fingerprints.extend(
            fingerprint_records(
                summary.records,
                summary.split,
                compute_pixels=compute_pixels,
                compute_dhash=check_near_duplicates,
            )
        )
    report.total_files = len(fingerprints)

    by_file: Dict[str, List[FileFingerprint]] = {}
    by_pixels: Dict[str, List[FileFingerprint]] = {}
    for item in fingerprints:
        by_file.setdefault(item.sha256, []).append(item)
        if item.pixels_sha256 is not None:
            by_pixels.setdefault(item.pixels_sha256, []).append(item)
        elif compute_pixels:
            # Only a genuine decode failure counts. When pixel hashing is
            # switched off every value is None by design, not by corruption.
            report.undecodable.append(str(item.path))
    report.distinct_file_hashes = len(by_file)

    # --- label conflicts -------------------------------------------------
    seen: Set[Tuple[str, ...]] = set()
    for group in by_file.values():
        if len({item.label for item in group}) > 1:
            key = tuple(sorted(str(item.path) for item in group))
            seen.add(key)
            report.conflicts.append(
                Conflict(REASON_IDENTICAL_BYTES, tuple(sorted(group, key=lambda m: str(m.path))))
            )
    for group in by_pixels.values():
        if len({item.label for item in group}) > 1:
            key = tuple(sorted(str(item.path) for item in group))
            if key in seen:  # already reported as a byte-identical conflict
                continue
            seen.add(key)
            report.conflicts.append(
                Conflict(REASON_IDENTICAL_PIXELS, tuple(sorted(group, key=lambda m: str(m.path))))
            )
    report.conflicts.sort(key=lambda c: (c.reason, str(c.paths[0])))

    # --- identical content spanning splits -------------------------------
    for group in by_file.values():
        if len({item.split for item in group}) > 1:
            report.cross_split_duplicates.append(
                Conflict(
                    REASON_CROSS_SPLIT_DUPLICATE,
                    tuple(sorted(group, key=lambda m: str(m.path))),
                )
            )
    report.cross_split_duplicates.sort(key=lambda c: str(c.paths[0]))

    # --- near duplicates across splits -----------------------------------
    if check_near_duplicates:
        report.near_duplicate_pairs = _near_duplicates_across_splits(
            fingerprints, near_duplicate_distance, root_path
        )

    # --- file format distribution per class ------------------------------
    for item in fingerprints:
        suffix = item.path.suffix.lower().lstrip(".")
        label_name = _LABEL_NAMES.get(item.label, str(item.label))
        report.format_counts.setdefault(label_name, {})
        report.format_counts[label_name][suffix] = (
            report.format_counts[label_name].get(suffix, 0) + 1
        )

    return report


def _near_duplicates_across_splits(
    fingerprints: Sequence[FileFingerprint], distance: int, root: Optional[Path]
) -> List[Dict[str, Any]]:
    """Pairs of visually near-identical images that landed in different splits."""

    train = [f for f in fingerprints if f.split == "train" and f.dhash is not None]
    others = [f for f in fingerprints if f.split != "train" and f.dhash is not None]
    if not train or not others:
        return []

    train_hashes = np.array([f.dhash for f in train], dtype=np.uint64)
    pairs: List[Dict[str, Any]] = []
    for item in others:
        xor = np.bitwise_xor(train_hashes, np.uint64(item.dhash))
        counts = np.zeros_like(xor)
        working = xor.copy()
        while working.any():
            counts += working & np.uint64(1)
            working >>= np.uint64(1)
        close = np.flatnonzero(counts <= distance)
        for index in close:
            match = train[int(index)]
            if match.sha256 == item.sha256:
                continue  # exact duplicate, already reported separately
            pairs.append(
                {
                    "distance": int(counts[index]),
                    "split": item.split,
                    "path": str(item.path.relative_to(root)) if root else str(item.path),
                    "train_path": (
                        str(match.path.relative_to(root)) if root else str(match.path)
                    ),
                    "labels": [item.label, match.label],
                }
            )
    pairs.sort(key=lambda p: (p["distance"], p["path"]))
    return pairs


def quick_conflict_audit(summaries: Sequence[DatasetSummary]) -> DataQualityReport:
    """File-hash-only audit, fast enough to run at the start of every training run.

    Finds byte-identical conflicting labels exactly. It cannot see a conflict
    that only shows up after decoding, so ``scripts/audit_dataset_quality.py``
    remains the authority; this is the safety net for a run whose manifest is
    missing or stale.
    """

    return audit_splits(summaries, check_near_duplicates=False, compute_pixels=False)


def filter_records(
    records: Iterable[ImageRecord],
    quarantined: Set[Path],
    include_conflicts: bool = False,
) -> Tuple[List[ImageRecord], int]:
    """Drop quarantined records unless explicitly told to keep them.

    Returns ``(kept, excluded_count)`` so the caller can record the exclusion in
    the dataset manifest rather than losing images silently.
    """

    items = list(records)
    if include_conflicts or not quarantined:
        return items, 0
    kept = [record for record in items if record.image_path not in quarantined]
    return kept, len(items) - len(kept)


def load_quarantine(path: str | Path, root: Optional[str | Path] = None) -> Set[Path]:
    """Read a saved quarantine manifest back into absolute paths."""

    source = Path(path).expanduser()
    if not source.is_file():
        return set()
    payload = json.loads(source.read_text(encoding="utf-8"))
    base = Path(root).expanduser() if root else (
        Path(payload["root"]) if payload.get("root") else None
    )
    paths: Set[Path] = set()
    for entry in payload.get("entries", []):
        candidate = Path(entry["path"])
        paths.add(candidate if candidate.is_absolute() or base is None else base / candidate)
    return paths
