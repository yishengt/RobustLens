"""Image discovery, reproducible splits, and PyTorch datasets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.data.augmentations import build_eval_transform, build_train_transform
from src.utils.config import resolve_config_path


DEFAULT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageRecord:
    path: str
    label: int


def _extensions(config: Optional[Dict[str, Any]] = None) -> set[str]:
    configured = config.get("data", {}).get("extensions") if config else None
    return {str(ext).lower() for ext in (configured or DEFAULT_EXTENSIONS)}


def list_image_files(directory: str | Path, extensions: Optional[set[str]] = None) -> List[Path]:
    """List supported image files recursively in a directory."""

    root = Path(directory).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    allowed = {str(extension).lower() for extension in (extensions or DEFAULT_EXTENSIONS)}
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in allowed
    )


def validate_image(path: str | Path) -> None:
    """Verify that Pillow can identify an image without decoding surprises."""

    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    try:
        with Image.open(image_path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Invalid or unreadable image '{image_path}': {exc}") from exc


def read_image(path: str | Path) -> np.ndarray:
    """Read an image as an RGB NumPy array with a path-aware error."""

    image_path = Path(path)
    try:
        with Image.open(image_path) as image:
            return np.asarray(image.convert("RGB"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Image file not found: {image_path}") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Invalid or unreadable image '{image_path}': {exc}") from exc


def discover_records(
    real_dir: str | Path,
    ai_dir: str | Path,
    config: Optional[Dict[str, Any]] = None,
    validate: bool = True,
) -> List[ImageRecord]:
    """Discover real=0 and AI-generated=1 records from two class roots."""

    extensions = _extensions(config)
    real_files = list_image_files(real_dir, extensions)
    ai_files = list_image_files(ai_dir, extensions)
    if not real_files:
        raise FileNotFoundError(f"No supported images found under real-image directory: {real_dir}")
    if not ai_files:
        raise FileNotFoundError(
            f"No supported images found under AI-generated directory: {ai_dir}"
        )
    paths = [(path, 0) for path in real_files] + [(path, 1) for path in ai_files]
    if validate:
        invalid: List[str] = []
        for path, _ in paths:
            try:
                validate_image(path)
            except (FileNotFoundError, ValueError) as exc:
                invalid.append(str(exc))
        if invalid:
            preview = "\n".join(invalid[:10])
            suffix = "" if len(invalid) <= 10 else f"\n... and {len(invalid) - 10} more"
            raise ValueError(f"Invalid dataset images detected:\n{preview}{suffix}")
    return [ImageRecord(path=str(path.resolve()), label=label) for path, label in paths]


def _records_from_payload(payload: Dict[str, Any], split_name: str) -> List[ImageRecord]:
    entries = payload.get(split_name)
    if not isinstance(entries, list):
        raise ValueError(f"Split file is missing a '{split_name}' list")
    records: List[ImageRecord] = []
    for entry in entries:
        if not isinstance(entry, dict) or "path" not in entry or "label" not in entry:
            raise ValueError(f"Invalid record in '{split_name}' split: {entry!r}")
        records.append(ImageRecord(path=str(entry["path"]), label=int(entry["label"])))
    return records


def create_or_load_splits(
    records: Sequence[ImageRecord],
    split_path: str | Path,
    validation_fraction: float = 0.2,
    seed: int = 42,
    force_resplit: bool = False,
) -> Tuple[List[ImageRecord], List[ImageRecord]]:
    """Create a stratified split once, then reload it on future runs."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    split_file = Path(split_path).expanduser()
    if split_file.is_file() and not force_resplit:
        try:
            with split_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            train_records = _records_from_payload(payload, "train")
            val_records = _records_from_payload(payload, "validation")
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            raise ValueError(f"Could not read split file {split_file}: {exc}") from exc
        for record in [*train_records, *val_records]:
            validate_image(record.path)
        return train_records, val_records

    if len(records) < 2:
        raise ValueError("At least two valid images are required to create a split")
    labels = [record.label for record in records]
    try:
        train, validation = train_test_split(
            list(records),
            test_size=validation_fraction,
            random_state=seed,
            stratify=labels,
        )
    except ValueError as exc:
        raise ValueError(
            "Could not create a stratified split. Provide enough real and AI-generated "
            f"images for validation: {exc}"
        ) from exc
    train_records = sorted(train, key=lambda record: record.path)
    val_records = sorted(validation, key=lambda record: record.path)
    split_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "validation_fraction": validation_fraction,
        "train": [asdict(record) for record in train_records],
        "validation": [asdict(record) for record in val_records],
    }
    with split_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return train_records, val_records


def load_data_splits(
    config: Dict[str, Any], force_resplit: bool = False
) -> Tuple[List[ImageRecord], List[ImageRecord]]:
    """Discover data and create/load the configured reproducible split."""

    real_dir = resolve_config_path(config, config["paths"]["real_dir"])
    ai_dir = resolve_config_path(config, config["paths"]["ai_dir"])
    split_path = resolve_config_path(config, config["paths"]["split_file"])
    records = discover_records(real_dir, ai_dir, config=config)
    return create_or_load_splits(
        records,
        split_path,
        validation_fraction=float(config["data"].get("validation_fraction", 0.2)),
        seed=int(config.get("seed", 42)),
        force_resplit=force_resplit,
    )


class ImageDataset(Dataset[Tuple[torch.Tensor, int]]):
    """A labelled image dataset applying an Albumentations transform."""

    def __init__(self, records: Sequence[ImageRecord], transform: Any):
        self.records = list(records)
        self.transform = transform
        if not self.records:
            raise ValueError("ImageDataset cannot be constructed with zero records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        record = self.records[index]
        try:
            image = read_image(record.path)
            tensor = self.transform(image=image)["image"]
        except (FileNotFoundError, ValueError, OSError, KeyError) as exc:
            raise RuntimeError(f"Could not load dataset image '{record.path}': {exc}") from exc
        return tensor, int(record.label)


def make_dataloaders(
    config: Dict[str, Any], force_resplit: bool = False
) -> Tuple[DataLoader, DataLoader, List[ImageRecord], List[ImageRecord]]:
    """Build train/validation DataLoaders and return their records too."""

    train_records, val_records = load_data_splits(config, force_resplit=force_resplit)
    batch_size = int(config["training"].get("batch_size", 32))
    data_config = config.get("data", {})
    common = {
        "batch_size": batch_size,
        "num_workers": int(data_config.get("num_workers", 0)),
        "pin_memory": bool(data_config.get("pin_memory", True)),
    }
    train_loader = DataLoader(
        ImageDataset(train_records, build_train_transform(config)),
        shuffle=True,
        **common,
    )
    val_loader = DataLoader(
        ImageDataset(val_records, build_eval_transform(config)),
        shuffle=False,
        **common,
    )
    return train_loader, val_loader, train_records, val_records


def make_eval_loader(
    records: Sequence[ImageRecord], config: Dict[str, Any], transform: Any
) -> DataLoader:
    """Build a non-shuffling loader for a supplied transform and records."""

    return DataLoader(
        ImageDataset(records, transform),
        batch_size=int(config.get("inference", {}).get("batch_size", 32)),
        shuffle=False,
        num_workers=int(config.get("data", {}).get("num_workers", 0)),
        pin_memory=bool(config.get("data", {}).get("pin_memory", True)),
    )
