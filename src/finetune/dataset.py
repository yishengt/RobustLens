"""Dataset utilities for authentic images and minor local AI edits.

Images are kept as PIL objects until the training loop applies the same
branch-specific processors used by inference.  A record also carries a stable
group id: callers can reject a split where versions of one original appear in
multiple partitions before training starts.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from src.data.dataset import DEFAULT_EXTENSIONS

LABELS = {"authentic": 0, "ai_edited": 1}
SUBGROUP_LABELS = {
    "authentic": 0,
    "synthetic": 1,
    "minor_edit": 1,
    "moderate_edit": 1,
    "transformed": 1,
}
DEFAULT_SUBGROUP_DIRECTORIES = {
    "authentic": "authentic",
    "minor_edit": "ai_edited",
    "synthetic": "synthetic",
    "moderate_edit": "moderate_edit",
    "transformed": "transformed",
}
_EDIT_SUFFIX = re.compile(r"(?:[_-](?:ai[_-]?)?edited|[_-](?:authentic|real)|[_-]v\d+)$", re.I)
# Every directory name that denotes a class rather than one original's folder.
# LABELS alone is not enough: the subgroup directories add "synthetic",
# "moderate_edit" and "transformed", and treating one of those as a per-original
# folder collapses an entire class into a single group.
_CLASS_DIRECTORIES = frozenset(LABELS) | frozenset(DEFAULT_SUBGROUP_DIRECTORIES.values())
_MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


@dataclass(frozen=True)
class ImageRecord:
    """One valid image and its optional matching edit mask."""

    image_path: Path
    label: int
    split: str
    group_id: str
    mask_path: Optional[Path] = None
    subgroup: str = "authentic"


@dataclass
class DatasetSummary:
    """JSON-friendly discovery statistics."""

    split: str
    root: str
    total_images: int = 0
    valid_images: int = 0
    corrupted_images: int = 0
    missing_masks: int = 0
    class_counts: Dict[str, int] = field(default_factory=lambda: {"authentic": 0, "ai_edited": 0})
    subgroup_counts: Dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(SUBGROUP_LABELS, 0)
    )
    group_count: int = 0
    records: List[ImageRecord] = field(default_factory=list, repr=False)

    @property
    def class_balance(self) -> Dict[str, Any]:
        authentic = self.class_counts.get("authentic", 0)
        edited = self.class_counts.get("ai_edited", 0)
        total = authentic + edited
        return {
            "authentic_fraction": authentic / total if total else 0.0,
            "ai_edited_fraction": edited / total if total else 0.0,
            "imbalance_ratio": max(authentic, edited) / min(authentic, edited)
            if min(authentic, edited)
            else None,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "root": self.root,
            "total_images": self.total_images,
            "valid_images": self.valid_images,
            "corrupted_images": self.corrupted_images,
            "missing_masks": self.missing_masks,
            "class_counts": dict(self.class_counts),
            "subgroup_counts": dict(self.subgroup_counts),
            "class_balance": self.class_balance,
            "group_count": self.group_count,
        }


def _group_id(path: Path, root: Path) -> str:
    """Derive a group id without depending on a filename convention.

    The preferred layout is one folder per original.  For flat folders, common
    ``*_edited``, ``*_authentic`` and version suffixes are removed.  Projects
    with a different naming scheme can pass an explicit ``group_map`` to
    :func:`discover_dataset`.
    """

    relative = path.relative_to(root)
    parts = list(relative.parts)
    if parts and parts[0] in _CLASS_DIRECTORIES:
        parts = parts[1:]
    # The preparation script puts one source image and all of its edits in
    # one folder, so a nested parent is the strongest grouping signal.
    if len(parts) > 1:
        return Path(*parts[:-1]).as_posix()
    return _EDIT_SUFFIX.sub("", Path(parts[0]).stem if parts else relative.stem)


def _find_mask(mask_root: Optional[Path], split: str, image_path: Path, image_root: Path) -> Optional[Path]:
    if mask_root is None:
        return None
    relative = image_path.relative_to(image_root)
    candidates = [mask_root / split / relative.with_suffix(ext) for ext in _MASK_EXTENSIONS]
    candidates.extend(mask_root / split / relative.parent / f"{relative.stem}_mask{ext}" for ext in _MASK_EXTENSIONS)
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _is_readable(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def discover_split(
    root: str | Path,
    split: str,
    masks_dir: str | Path | None = None,
    extensions: Optional[Iterable[str]] = None,
    group_map: Optional[Mapping[str, str]] = None,
    subgroup_directories: Optional[Mapping[str, str]] = None,
    strict: bool = False,
) -> DatasetSummary:
    """Discover valid records under configured subgroup directories.

    Corrupt files are counted and skipped by default.  ``strict=True`` turns a
    corrupt image into an error, which is useful for data preparation CI. The
    legacy ``authentic``/``ai_edited`` layout remains the default; the latter is
    recorded as the ``minor_edit`` subgroup.
    """

    image_root = Path(root).expanduser()
    if not image_root.is_dir():
        raise FileNotFoundError(f"Dataset split directory not found: {image_root}")
    allowed = {str(ext).lower() if str(ext).startswith(".") else f".{str(ext).lower()}" for ext in (extensions or DEFAULT_EXTENSIONS)}
    mask_root = Path(masks_dir).expanduser() if masks_dir else None
    summary = DatasetSummary(split=split, root=str(image_root))

    directories = dict(DEFAULT_SUBGROUP_DIRECTORIES)
    if subgroup_directories is not None:
        directories.update(
            {
                str(subgroup): str(directory)
                for subgroup, directory in subgroup_directories.items()
                if directory is not None
            }
        )
    unknown = sorted(set(directories) - set(SUBGROUP_LABELS))
    if unknown:
        raise ValueError(
            "Unknown local-edit subgroup(s): "
            f"{', '.join(unknown)}. Expected: {', '.join(SUBGROUP_LABELS)}"
        )

    seen_directories: set[str] = set()
    for subgroup, directory in directories.items():
        if directory in seen_directories:
            raise ValueError(f"Subgroup directory '{directory}' is configured more than once")
        seen_directories.add(directory)
        label = SUBGROUP_LABELS[subgroup]
        class_root = image_root / directory
        if not class_root.is_dir():
            continue
        for path in sorted(class_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in allowed:
                continue
            summary.total_images += 1
            if not _is_readable(path):
                summary.corrupted_images += 1
                if strict:
                    raise ValueError(f"Corrupt or unreadable image: {path}")
                continue
            key = str(path.relative_to(image_root))
            group_id = str(group_map.get(key, group_map.get(path.name, _group_id(path, image_root))) if group_map else _group_id(path, image_root))
            mask_path = _find_mask(mask_root, split, path, image_root)
            summary.records.append(
                ImageRecord(path, label, split, group_id, mask_path, subgroup)
            )
            summary.valid_images += 1
            summary.class_counts["ai_edited" if label else "authentic"] += 1
            summary.subgroup_counts[subgroup] += 1
            if mask_path is None:
                summary.missing_masks += 1

    summary.group_count = len({record.group_id for record in summary.records})
    return summary


def discover_labelled_directory(
    root: str | Path,
    split: str,
    subgroup: str,
    extensions: Optional[Iterable[str]] = None,
    strict: bool = False,
    group_prefix: Optional[str] = None,
) -> DatasetSummary:
    """Treat every image below ``root`` as one explicitly labelled subgroup.

    This is used for a fully synthetic replay mixture, whose source datasets do
    not necessarily use RobustLens' ``authentic``/``ai_edited`` folder names.
    Group ids are namespaced so an unrelated synthetic filename cannot collide
    with an Escher-VisMin source id.
    """

    if subgroup not in SUBGROUP_LABELS:
        raise ValueError(
            f"Unknown subgroup '{subgroup}'. Expected: {', '.join(SUBGROUP_LABELS)}"
        )
    image_root = Path(root).expanduser()
    if not image_root.is_dir():
        raise FileNotFoundError(f"Subgroup directory not found: {image_root}")
    allowed = {
        str(ext).lower() if str(ext).startswith(".") else f".{str(ext).lower()}"
        for ext in (extensions or DEFAULT_EXTENSIONS)
    }
    summary = DatasetSummary(split=split, root=str(image_root))
    label = SUBGROUP_LABELS[subgroup]
    prefix = group_prefix or subgroup
    for path in sorted(image_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        summary.total_images += 1
        if not _is_readable(path):
            summary.corrupted_images += 1
            if strict:
                raise ValueError(f"Corrupt or unreadable image: {path}")
            continue
        relative = path.relative_to(image_root)
        local_group = relative.parent.as_posix() if relative.parent != Path(".") else path.stem
        summary.records.append(
            ImageRecord(
                image_path=path,
                label=label,
                split=split,
                group_id=f"{prefix}:{local_group}",
                mask_path=None,
                subgroup=subgroup,
            )
        )
        summary.valid_images += 1
        summary.missing_masks += 1
        summary.class_counts["ai_edited" if label else "authentic"] += 1
        summary.subgroup_counts[subgroup] += 1
    summary.group_count = len({record.group_id for record in summary.records})
    return summary


def verify_split_groups(summaries: Sequence[DatasetSummary]) -> None:
    """Raise if an original/group occurs in more than one dataset split."""

    owners: Dict[str, str] = {}
    collisions: List[str] = []
    for summary in summaries:
        for record in summary.records:
            previous = owners.setdefault(record.group_id, summary.split)
            if previous != summary.split:
                collisions.append(f"{record.group_id} ({previous}, {summary.split})")
    if collisions:
        examples = ", ".join(sorted(set(collisions))[:5])
        raise ValueError(
            "Dataset leakage: versions of the same original occur in multiple splits: "
            f"{examples}"
        )


def discover_dataset(
    data_dir: str | Path,
    masks_dir: str | Path | None = None,
    extensions: Optional[Iterable[str]] = None,
    group_map: Optional[Mapping[str, str]] = None,
    subgroup_directories: Optional[Mapping[str, str]] = None,
    strict: bool = False,
) -> Dict[str, DatasetSummary]:
    """Discover train/validation/test and enforce group-level split isolation."""

    root = Path(data_dir).expanduser()
    summaries = {
        split: discover_split(
            root / split,
            split,
            masks_dir,
            extensions,
            group_map,
            subgroup_directories,
            strict,
        )
        for split in ("train", "validation", "test")
    }
    verify_split_groups(list(summaries.values()))
    return summaries


class LocalEditDataset(Dataset):
    """PIL-backed binary dataset with optional masks and corrupt-file skipping."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        masks_dir: str | Path | None = None,
        transform: Optional[Callable[[Image.Image], Any]] = None,
        extensions: Optional[Iterable[str]] = None,
        strict: bool = False,
        records: Optional[Sequence[ImageRecord]] = None,
        views: int = 1,
    ) -> None:
        self.transform = transform
        # views > 1 returns several independently transformed copies of one
        # image, tagged with a shared pair id. That pairing is what makes a
        # transformation-consistency loss possible: it compares f(T1(x)) with
        # f(T2(x)) for the same x, never two different photographs.
        if int(views) < 1:
            raise ValueError(f"views must be at least 1, got {views}")
        self.views = int(views)
        self.summary = discover_split(root, split, masks_dir, extensions, strict=strict) if records is None else DatasetSummary(split, str(root), valid_images=len(records), records=list(records))
        self.records = list(self.summary.records if records is None else records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        try:
            with Image.open(record.image_path) as image:
                image = image.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError(f"Could not decode image '{record.image_path}': {exc}") from exc
        if self.views == 1:
            value: Any = self.transform(image) if self.transform else image
            return {
                "image": value,
                "label": torch.tensor(float(record.label), dtype=torch.float32),
                "image_path": str(record.image_path),
                "group_id": record.group_id,
                "subgroup": record.subgroup,
                "pair_id": f"{self.summary.split}:{index}",
                "mask_path": str(record.mask_path) if record.mask_path else None,
            }
        views = [self.transform(image) if self.transform else image for _ in range(self.views)]
        return {
            "views": views,
            "label": torch.tensor(float(record.label), dtype=torch.float32),
            "image_path": str(record.image_path),
            "group_id": record.group_id,
            "subgroup": record.subgroup,
            "pair_id": f"{self.summary.split}:{index}",
            "mask_path": str(record.mask_path) if record.mask_path else None,
        }


def local_edit_collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate PIL images and optional masks without forcing a tensor transform.

    Multi-view items are flattened so every view becomes its own batch entry,
    with its label, group id and pair id repeated. Downstream code therefore
    sees an ordinary batch; only ``pair_ids`` reveals which entries are views of
    one image, and that is exactly what the consistency loss needs.
    """

    images: List[Any] = []
    labels: List[torch.Tensor] = []
    paths: List[str] = []
    groups: List[str] = []
    pairs: List[str] = []
    masks: List[Optional[str]] = []
    subgroups: List[str] = []
    for item in batch:
        views = item["views"] if "views" in item else [item["image"]]
        for view in views:
            images.append(view)
            labels.append(item["label"])
            paths.append(item["image_path"])
            groups.append(item["group_id"])
            pairs.append(item["pair_id"])
            masks.append(item["mask_path"])
            subgroups.append(item["subgroup"])
    return {
        "images": images,
        "labels": torch.stack(labels),
        "image_paths": paths,
        "group_ids": groups,
        "pair_ids": pairs,
        "mask_paths": masks,
        "subgroups": subgroups,
    }


def summarize_class_balance(dataset: LocalEditDataset) -> Dict[str, Any]:
    """Return the class counts and balance ratio for logging."""

    counts = Counter(LABELS.keys())
    counts.update("ai_edited" if record.label else "authentic" for record in dataset.records)
    counts = {key: counts[key] - 1 for key in LABELS}
    total = sum(counts.values())
    return {
        "class_counts": counts,
        "subgroup_counts": dict(Counter(record.subgroup for record in dataset.records)),
        "total": total,
        "imbalance_ratio": max(counts.values()) / min(counts.values()) if min(counts.values()) else None,
    }
