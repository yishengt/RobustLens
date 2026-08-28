"""Reader for SID_Set parquet shards.

SID_Set stores images inline in parquet, so shards are streamed in record
batches rather than loaded whole: a single shard is ~495 MB on disk.

The dataset labels three classes::

    0 = real           1 = fully synthetic          2 = tampered

The detector is binary, so labels are mapped to ``0 = real`` and
``1 = AI-generated`` (both synthetic and tampered count as AI-generated).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from PIL import Image, UnidentifiedImageError

LABEL_REAL = 0
LABEL_FULL_SYNTHETIC = 1
LABEL_TAMPERED = 2

CLASS_NAMES = {
    LABEL_REAL: "real",
    LABEL_FULL_SYNTHETIC: "full_synthetic",
    LABEL_TAMPERED: "tampered",
}


@dataclass
class LabelledImage:
    """One dataset row: a decoded image plus its labels."""

    image: Image.Image
    label: int
    binary_label: int
    img_id: str
    source: str

    @property
    def class_name(self) -> str:
        return CLASS_NAMES.get(self.label, f"unknown({self.label})")


def to_binary_label(label: int) -> int:
    """Map the 3-class SID_Set label to the detector's binary target."""

    value = int(label)
    if value not in CLASS_NAMES:
        raise ValueError(f"Unexpected SID_Set label {label}; expected 0, 1 or 2")
    return 0 if value == LABEL_REAL else 1


def find_shards(
    data_dir: str | Path, split: Optional[str] = None
) -> List[Path]:
    """Find downloaded parquet shards, optionally filtered by split name."""

    root = Path(data_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}\n"
            f"Download shards first:  python scripts/download_dataset.py"
        )
    pattern = f"{split}-*.parquet" if split else "*.parquet"
    shards = sorted(root.rglob(pattern))
    if not shards:
        raise FileNotFoundError(
            f"No {'*.parquet' if not split else pattern} shards found under {root}.\n"
            f"Download them with:  python scripts/download_dataset.py"
        )
    return shards


def _decode_image_cell(cell: Any) -> Optional[Image.Image]:
    """Decode the HF ``image`` column, stored as a struct of bytes and path."""

    payload = cell
    if isinstance(cell, dict):
        payload = cell.get("bytes")
        if payload is None and cell.get("path"):
            path = Path(str(cell["path"]))
            if path.is_file():
                with Image.open(path) as image:
                    return image.convert("RGB")
            return None
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        return None
    with Image.open(io.BytesIO(bytes(payload))) as image:
        return image.convert("RGB")


def iter_labelled_images(
    shards: Sequence[Path],
    limit: Optional[int] = None,
    batch_size: int = 32,
    per_class_limit: Optional[int] = None,
) -> Iterator[LabelledImage]:
    """Stream decoded, labelled images from a list of parquet shards.

    ``per_class_limit`` caps how many images are taken from each of the three
    source classes, which keeps an evaluation sample balanced even when shards
    are ordered by class.
    """

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - pyarrow ships with pandas
        raise ImportError(
            "pyarrow is required to read SID_Set parquet shards. "
            "Install it with: pip install pyarrow"
        ) from exc

    emitted = 0
    per_class: Dict[int, int] = dict.fromkeys(CLASS_NAMES, 0)

    for shard in shards:
        parquet_file = pq.ParquetFile(str(shard))
        available = set(parquet_file.schema_arrow.names)
        columns = [name for name in ("img_id", "image", "label") if name in available]
        if "image" not in columns or "label" not in columns:
            raise ValueError(
                f"{shard.name} is missing the required 'image'/'label' columns; "
                f"found {sorted(available)}"
            )

        for record_batch in parquet_file.iter_batches(
            batch_size=max(1, int(batch_size)), columns=columns
        ):
            rows = record_batch.to_pylist()
            for row in rows:
                if limit is not None and emitted >= limit:
                    return
                label = int(row["label"])
                if per_class_limit is not None and per_class.get(label, 0) >= per_class_limit:
                    continue
                try:
                    image = _decode_image_cell(row["image"])
                except (UnidentifiedImageError, OSError, ValueError):
                    continue
                if image is None:
                    continue
                per_class[label] = per_class.get(label, 0) + 1
                emitted += 1
                yield LabelledImage(
                    image=image,
                    label=label,
                    binary_label=to_binary_label(label),
                    img_id=str(row.get("img_id") or f"{shard.stem}:{emitted}"),
                    source=shard.name,
                )

            # Stop early once every class has hit its cap.
            if per_class_limit is not None and all(
                per_class.get(label, 0) >= per_class_limit for label in CLASS_NAMES
            ):
                return


def describe_shards(shards: Sequence[Path]) -> Dict[str, Any]:
    """Return row counts and class distribution without decoding any images."""

    import pyarrow.parquet as pq

    total_rows = 0
    distribution: Dict[str, int] = dict.fromkeys(CLASS_NAMES.values(), 0)
    for shard in shards:
        parquet_file = pq.ParquetFile(str(shard))
        total_rows += parquet_file.metadata.num_rows
        for record_batch in parquet_file.iter_batches(batch_size=4096, columns=["label"]):
            for label in record_batch.column("label").to_pylist():
                name = CLASS_NAMES.get(int(label), f"unknown({label})")
                distribution[name] = distribution.get(name, 0) + 1
    return {
        "shards": len(shards),
        "rows": total_rows,
        "class_distribution": distribution,
        "files": [shard.name for shard in shards],
    }
