"""Leakage and file-container shortcut probes used by the evaluation protocol."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import numpy as np
from PIL import Image

from src.evaluation.metrics import compute_metrics


def evaluate_format_reencoding(
    paths: Sequence[str | Path],
    labels: Sequence[int],
    scorer: Callable[[Path], float],
    threshold: float,
    output_format: str = "PNG",
) -> Dict[str, Any]:
    """Compare scores before/after putting every image in one file container.

    A detector exploiting class-correlated extensions or container metadata will
    move sharply when all inputs are decoded to RGB and re-encoded identically.
    The probe does not fit a threshold and never exposes its temporary files to
    training.
    """

    items = [Path(path).expanduser() for path in paths]
    y = [int(label) for label in labels]
    if len(items) != len(y) or not items:
        raise ValueError("paths and labels must be aligned and non-empty")
    normalized_format = str(output_format).strip().upper()
    suffixes = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}
    if normalized_format not in suffixes:
        raise ValueError(f"output_format must be one of {', '.join(suffixes)}")

    original_scores = [float(scorer(path)) for path in items]
    with tempfile.TemporaryDirectory(prefix="robustlens-format-probe-") as tmp:
        root = Path(tmp)
        normalized_paths = []
        for index, path in enumerate(items):
            destination = root / f"{index:06d}{suffixes[normalized_format]}"
            with Image.open(path) as image:
                image.convert("RGB").save(destination, format=normalized_format)
            normalized_paths.append(destination)
        normalized_scores = [float(scorer(path)) for path in normalized_paths]

    before = compute_metrics(y, original_scores, threshold).as_dict()
    after = compute_metrics(y, normalized_scores, threshold).as_dict()
    deltas = np.asarray(normalized_scores) - np.asarray(original_scores)
    decisions_before = np.asarray(original_scores) >= float(threshold)
    decisions_after = np.asarray(normalized_scores) >= float(threshold)
    return {
        "count": len(items),
        "threshold": float(threshold),
        "threshold_note": "Fixed before re-encoding; never fitted on this probe.",
        "output_format": normalized_format,
        "original_metrics": before,
        "reencoded_metrics": after,
        "mean_score_delta": float(np.mean(deltas)),
        "mean_absolute_score_delta": float(np.mean(np.abs(deltas))),
        "max_absolute_score_delta": float(np.max(np.abs(deltas))),
        "decision_preservation_rate": float(np.mean(decisions_before == decisions_after)),
        "per_image": [
            {
                "image_path": str(path),
                "label": label,
                "original_score": original,
                "reencoded_score": normalized,
                "score_delta": normalized - original,
            }
            for path, label, original, normalized in zip(
                items, y, original_scores, normalized_scores
            )
        ],
    }


__all__ = ["evaluate_format_reencoding"]


# ---------------------------------------------------------------------------
# Dataset-level confound audits
# ---------------------------------------------------------------------------
#
# A detector can only exploit a shortcut that exists in the data. These checks
# look for one: an attribute that carries no forensic meaning but still splits
# the classes. They report separation, they do not train anything, and a clean
# result here is evidence about the DATASET, not proof about the model.

from collections import Counter  # noqa: E402
from typing import Iterable, List, Optional, Tuple  # noqa: E402

# Above this, one class is distinguishable from the other by the attribute
# alone often enough to be a real confound worth ruling out.
CONFOUND_SEPARATION_LIMIT = 0.80


def _class_split(values: Sequence[Any], labels: Sequence[int]) -> Dict[str, Any]:
    """How well a single categorical attribute separates the two classes.

    ``separation`` is the accuracy of the best possible rule that looks only at
    this attribute: for each value, predict its majority class. 0.5 means the
    attribute says nothing; 1.0 means it determines the label outright.
    """

    if len(values) != len(labels):
        raise ValueError(f"values and labels differ in length: {len(values)} != {len(labels)}")
    if not values:
        return {"count": 0, "separation": None, "by_value": {}}

    by_value: Dict[Any, Counter] = {}
    for value, label in zip(values, labels):
        by_value.setdefault(value, Counter())[int(label)] += 1
    correct = sum(max(counter.values()) for counter in by_value.values())
    return {
        "count": len(values),
        "distinct_values": len(by_value),
        "separation": correct / len(values),
        "by_value": {
            str(value): {"authentic": counter.get(0, 0), "ai": counter.get(1, 0)}
            for value, counter in sorted(by_value.items(), key=lambda item: str(item[0]))
        },
    }


def _bucket(value: float, edges: Sequence[float]) -> str:
    for edge in edges:
        if value <= edge:
            return f"<={edge:g}"
    return f">{edges[-1]:g}"


def image_attributes(path: str | Path) -> Optional[Dict[str, Any]]:
    """Non-forensic container attributes for one image, or None if unreadable."""

    from PIL import Image, UnidentifiedImageError

    file_path = Path(path)
    try:
        with Image.open(file_path) as image:
            width, height = image.size
            fmt = (image.format or "").upper()
            mode = image.mode
    except (UnidentifiedImageError, OSError, ValueError):
        return None
    size = file_path.stat().st_size if file_path.is_file() else 0
    pixels = max(1, width * height)
    return {
        "extension": file_path.suffix.lower().lstrip("."),
        "format": fmt,
        "mode": mode,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "megapixels": _bucket(pixels / 1e6, [0.25, 1.0, 4.0, 12.0]),
        "aspect_ratio": _bucket(width / max(1, height), [0.8, 1.0, 1.34, 1.78]),
        # Bytes per pixel is a proxy for compression history: a heavily
        # re-compressed JPEG carries far fewer bytes per pixel than a PNG.
        "bytes_per_pixel": _bucket(size / pixels, [0.05, 0.15, 0.5, 1.5]),
    }


CONFOUND_ATTRIBUTES = (
    "extension",
    "format",
    "mode",
    "resolution",
    "megapixels",
    "aspect_ratio",
    "bytes_per_pixel",
)


def dataset_confounds(
    paths: Sequence[str | Path],
    labels: Sequence[int],
    attributes: Sequence[str] = CONFOUND_ATTRIBUTES,
    limit: float = CONFOUND_SEPARATION_LIMIT,
) -> Dict[str, Any]:
    """Report how well each non-forensic attribute separates the classes."""

    # zip() would silently truncate to the shorter list and quietly drop
    # images from the audit, so a mismatch is an error rather than a shrug.
    if len(paths) != len(labels):
        raise ValueError(
            f"paths and labels differ in length: {len(paths)} != {len(labels)}"
        )
    rows: List[Tuple[Dict[str, Any], int]] = []
    unreadable: List[str] = []
    for path, label in zip(paths, labels):
        found = image_attributes(path)
        if found is None:
            unreadable.append(str(path))
            continue
        rows.append((found, int(label)))

    results: Dict[str, Any] = {}
    flagged: List[str] = []
    for name in attributes:
        summary = _class_split([row[name] for row, _ in rows], [label for _, label in rows])
        results[name] = summary
        if summary["separation"] is not None and summary["separation"] >= limit:
            flagged.append(name)

    return {
        "images": len(rows),
        "unreadable": unreadable,
        "separation_limit": limit,
        "attributes": results,
        "flagged": flagged,
        "note": (
            "Separation is the accuracy of the best rule using only this attribute. "
            "A flagged attribute is a confound present in the DATA; it does not by "
            "itself show the detector uses it. Rule that out by re-scoring with the "
            "attribute normalised, as evaluate_format_reencoding does for format."
        ),
    }


def filename_leakage(paths: Sequence[str | Path], labels: Sequence[int]) -> Dict[str, Any]:
    """Do filenames encode the label directly?

    The preparation scripts name files by class, so this is expected to fire on
    the raw dataset. It exists so the expectation is explicit and checked rather
    than assumed: the pipeline must never read a filename, and
    ``src/pipeline/preprocessing.py`` guarantees only pixels reach the model.
    """

    if len(paths) != len(labels):
        raise ValueError(
            f"paths and labels differ in length: {len(paths)} != {len(labels)}"
        )
    tokens_by_label: Dict[int, Counter] = {0: Counter(), 1: Counter()}
    for path, label in zip(paths, labels):
        stem = Path(path).stem.lower()
        for token in stem.replace("-", "_").split("_"):
            if token and not token.isdigit():
                tokens_by_label[int(label)][token] += 1

    exclusive = {
        0: sorted(set(tokens_by_label[0]) - set(tokens_by_label[1]))[:10],
        1: sorted(set(tokens_by_label[1]) - set(tokens_by_label[0]))[:10],
    }
    return {
        "authentic_only_tokens": exclusive[0],
        "ai_only_tokens": exclusive[1],
        "filename_encodes_label": bool(exclusive[0] and exclusive[1]),
        "note": (
            "Informational. Filenames never reach the detector: preprocessing "
            "passes pixels only, so filename structure cannot be exploited."
        ),
    }


def transformed_copy_leakage(
    group_ids_by_split: Dict[str, Iterable[str]],
) -> Dict[str, Any]:
    """Any source group appearing in more than one split.

    Every transformed or edited version of an image carries its source's group
    id, so a group spanning splits means a transformed copy crossed the
    boundary -- the exact leak that makes robustness numbers meaningless.
    """

    owners: Dict[str, str] = {}
    collisions: List[Dict[str, str]] = []
    for split, groups in group_ids_by_split.items():
        for group in groups:
            previous = owners.setdefault(group, split)
            if previous != split:
                collisions.append({"group_id": group, "splits": f"{previous},{split}"})
    return {
        "groups": len(owners),
        "leaked_groups": len(collisions),
        "examples": collisions[:10],
        "clean": not collisions,
    }
