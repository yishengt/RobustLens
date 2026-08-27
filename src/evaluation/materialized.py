"""Load transformed validation images produced by a separate transformation module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from src.data.dataset import ImageRecord, validate_image


def _resolve_manifest_path(value: str, manifest_dir: Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    manifest_relative = (manifest_dir / path).resolve()
    if manifest_relative.exists():
        return manifest_relative
    return (project_root / path).resolve()


def load_materialized_records(
    manifest_path: str | Path,
    validation_records: Sequence[ImageRecord],
    expected_cases: Iterable[str],
    project_root: str | Path,
) -> Dict[str, List[ImageRecord]]:
    """Load and validate the transformation handoff manifest.

    The manifest must be a JSON list of objects with ``case``, ``source_path``,
    and ``transformed_path``. Labels come from the project's validation split,
    never from the transformation producer, so a transformation cannot change
    the ground truth.
    """

    manifest_file = Path(manifest_path).expanduser().resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Transformation manifest not found: {manifest_file}")
    try:
        with manifest_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read transformation manifest {manifest_file}: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError(
            "Transformation manifest must be a JSON list of case/source/transformed records"
        )

    root = Path(project_root).expanduser().resolve()
    labels_by_source = {
        str(Path(record.path).expanduser().resolve()): int(record.label)
        for record in validation_records
    }
    expected = list(expected_cases)
    records_by_case: Dict[str, List[ImageRecord]] = {case: [] for case in expected}
    seen = set()
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid transformation manifest entry: {entry!r}")
        missing = {"case", "source_path", "transformed_path"} - set(entry)
        if missing:
            raise ValueError(
                f"Transformation manifest entry is missing fields {sorted(missing)}: {entry!r}"
            )
        case = str(entry["case"])
        if case not in records_by_case:
            continue
        source = _resolve_manifest_path(str(entry["source_path"]), manifest_file.parent, root)
        source_key = str(source)
        if source_key not in labels_by_source:
            # The producer may have transformed the full raw dataset. Extra
            # non-validation records are safe and are ignored.
            continue
        key = (case, source_key)
        if key in seen:
            raise ValueError(f"Duplicate transformation entry for {case}: {source}")
        seen.add(key)
        transformed = _resolve_manifest_path(
            str(entry["transformed_path"]), manifest_file.parent, root
        )
        validate_image(transformed)
        records_by_case[case].append(
            ImageRecord(path=str(transformed), label=labels_by_source[source_key])
        )

    expected_sources = set(labels_by_source)
    for case in expected:
        actual_sources = {
            source
            for current_case, source in seen
            if current_case == case
        }
        missing_sources = expected_sources - actual_sources
        if missing_sources:
            preview = ", ".join(sorted(missing_sources)[:3])
            raise ValueError(
                f"Transformation case '{case}' is missing {len(missing_sources)} validation "
                f"images in {manifest_file}; examples: {preview}"
            )
        records_by_case[case].sort(key=lambda record: record.path)
    return records_by_case
