"""Training loop for head-only or existing-LoRA local-edit fine-tuning."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.finetune.data_quality import (
    filter_records,
    load_quarantine,
    quick_conflict_audit,
)
from src.finetune.dataset import (
    LABELS,
    SUBGROUP_LABELS,
    LocalEditDataset,
    discover_labelled_directory,
    discover_split,
    local_edit_collate,
    verify_split_groups,
)
from src.finetune.losses import (
    CONSISTENCY_METHODS,
    CONSISTENCY_MSE,
    attach_loss,
    binary_loss,
    binary_metrics,
    combined_loss,
    metrics_by_subgroup,
)
from src.finetune.model import FineTuneModel, parameter_counts
from src.pipeline.model_loader import MAX_PARAMETERS
from src.pipeline.transformations import TransformSpec, apply_transform
from src.utils.config import resolve_config_path


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch, including deterministic worker setup."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # pragma: no cover - older torch
        torch.use_deterministic_algorithms(True)


def _training_transform(config: Dict[str, Any]):
    configured = (config.get("training", {}) or {}).get("official_transformations", [])
    if configured is True:
        configured = ["jpeg_90", "blur_0.5", "resize_0.5x", "color_jitter"]
    if isinstance(configured, str):
        configured = [configured]
    cases = list(configured or [])
    if not cases:
        return None
    def spec_for(case: str) -> TransformSpec:
        if case.startswith("jpeg_"):
            return TransformSpec(case, "jpeg", {"quality": int(case.split("_", 1)[1])})
        if case.startswith("blur_"):
            return TransformSpec(case, "blur", {"sigma": float(case.split("_", 1)[1])})
        if case.startswith("resize_"):
            return TransformSpec(
                case, "resize", {"scale": float(case.split("_", 1)[1].rstrip("x"))}
            )
        if case.startswith("noise_"):
            return TransformSpec(case, "noise", {"sigma": float(case.split("_", 1)[1])})
        if case == "color_jitter":
            return TransformSpec(
                case,
                "color_jitter",
                {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2},
            )
        if case.startswith("center_crop_"):
            return TransformSpec(
                case, "center_crop", {"fraction": float(case.rsplit("_", 1)[1]) / 100.0}
            )
        raise ValueError(
            f"Unknown training transformation '{case}'. Supported forms: jpeg_90, "
            "blur_0.5, resize_0.5x, noise_0.02, color_jitter, center_crop_80."
        )

    specs = [spec_for(str(case)) for case in cases]

    def transform(image: Image.Image) -> Image.Image:
        spec = random.choice(specs)
        return apply_transform(image, spec, seed=random.randrange(0, 2**32))

    return transform


def consistency_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the transformation-consistency loss configuration.

    Disabled by default. The weight must be positive for the term to exist at
    all, so an accidental ``enabled: true`` with weight 0 is reported as off
    rather than pretending a term is active that contributes nothing.
    """

    section = ((config.get("training", {}) or {}).get("consistency", {}) or {})
    weight = float(section.get("weight", 0.0))
    if weight < 0:
        raise ValueError(f"training.consistency.weight must be non-negative, got {weight}")
    method = str(section.get("method", CONSISTENCY_MSE)).strip().lower()
    if method not in CONSISTENCY_METHODS:
        raise ValueError(
            f"training.consistency.method must be one of {', '.join(CONSISTENCY_METHODS)}, "
            f"got '{method}'"
        )
    enabled = bool(section.get("enabled", False)) and weight > 0.0
    return {
        "enabled": enabled,
        "weight": weight if enabled else 0.0,
        "method": method,
        "same_label_pairs_only": bool(section.get("same_label_pairs_only", True)),
    }


def _worker_init(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def _paths(config: Dict[str, Any]) -> Tuple[Path, Path, Path, Optional[Path]]:
    data = config.get("data", {}) or {}
    train = resolve_config_path(config, data.get("train_dir", "data/local_edits/train"))
    validation = resolve_config_path(config, data.get("validation_dir", "data/local_edits/validation"))
    test = resolve_config_path(config, data.get("test_dir", "data/local_edits/test"))
    masks = data.get("masks_dir", "masks")
    return train, validation, test, resolve_config_path(config, masks) if masks else None


def _resolve_quarantine(
    config: Dict[str, Any], summaries: List[Any]
) -> Tuple[Set[Path], Dict[str, Any]]:
    """Return the files to exclude, and how that list was obtained.

    Order matters. A manifest written by scripts/audit_dataset_quality.py is the
    authority because it decodes every image and so catches re-encoded conflicts
    a file hash cannot see. When no manifest exists the run still refuses to
    train on byte-identical conflicting labels -- it falls back to the fast
    hash-only audit rather than proceeding as if the data were clean.
    """

    data = config.get("data", {}) or {}
    if bool(data.get("include_conflicting_labels", False)):
        return set(), {
            "source": "disabled",
            "note": (
                "data.include_conflicting_labels is true, so images with "
                "contradictory labels were KEPT. This was an explicit choice."
            ),
        }

    manifest = data.get("quarantine_path", "outputs/data_quality/quarantine.json")
    manifest_path = resolve_config_path(config, manifest) if manifest else None
    if manifest_path is not None and manifest_path.is_file():
        paths = load_quarantine(manifest_path)
        if paths:
            return paths, {"source": "manifest", "path": str(manifest_path)}

    report = quick_conflict_audit(summaries)
    return report.quarantined_paths, {
        "source": "inline_hash_audit",
        "note": (
            "No quarantine manifest was found, so a file-hash-only audit ran. It "
            "catches byte-identical conflicts exactly but not re-encoded ones. "
            "Run scripts/audit_dataset_quality.py for the full check."
        ),
    }


def _subgroup_directories(config: Dict[str, Any]) -> Dict[str, str]:
    """Resolve optional per-subgroup directory names from the training config."""

    configured = ((config.get("data", {}) or {}).get("subgroups", {}) or {})
    result: Dict[str, str] = {}
    for subgroup, value in configured.items():
        if subgroup not in SUBGROUP_LABELS:
            raise ValueError(
                f"Unknown data subgroup '{subgroup}'. Expected: {', '.join(SUBGROUP_LABELS)}"
            )
        if value is None or value is False:
            continue
        if isinstance(value, str):
            result[subgroup] = value
            continue
        if not isinstance(value, dict):
            raise ValueError(
                f"data.subgroups.{subgroup} must be a directory string or mapping"
            )
        if "label" in value and int(value["label"]) != SUBGROUP_LABELS[subgroup]:
            raise ValueError(
                f"data.subgroups.{subgroup}.label must be {SUBGROUP_LABELS[subgroup]}"
            )
        if value.get("enabled", True) and value.get("directory"):
            result[subgroup] = str(value["directory"])
    return result


def _deterministic_record_sample(
    records: Sequence[Any], count: int, seed: int, namespace: str
) -> List[Any]:
    """Select records reproducibly without depending on filesystem ordering."""

    if count >= len(records):
        return list(records)
    ranked = sorted(
        records,
        key=lambda record: hashlib.sha256(
            f"{seed}:{namespace}:{record.image_path}".encode("utf-8")
        ).hexdigest(),
    )
    return ranked[: max(0, int(count))]


def _synthetic_mixture_records(
    config: Dict[str, Any], base_train_count: int, extensions: Any, strict: bool
) -> Tuple[List[Any], Dict[str, Any]]:
    """Load and deterministically sample fully synthetic replay images.

    ``synthetic_mixture_fraction`` is the desired fraction of the final train
    set, not a fraction of the source directory. Only training receives these
    records; validation and test stay quarantined local-edit holdouts.
    """

    data = config.get("data", {}) or {}
    source_value = data.get("synthetic_mixture_dir")
    fraction = float(data.get("synthetic_mixture_fraction", 0.0))
    if not 0.0 <= fraction < 1.0:
        raise ValueError(
            f"data.synthetic_mixture_fraction must be in [0, 1), got {fraction}"
        )
    if not source_value or fraction == 0.0:
        return [], {
            "enabled": False,
            "requested_fraction": fraction,
            "selected": 0,
            "available": 0,
            "train_only": True,
        }
    source = resolve_config_path(config, source_value)
    if (source / "train").is_dir():
        source = source / "train"
    discovered = discover_labelled_directory(
        source,
        "train",
        "synthetic",
        extensions=extensions,
        strict=strict,
        group_prefix="synthetic_mixture",
    )
    requested = int(round(base_train_count * fraction / max(1e-12, 1.0 - fraction)))
    seed = int((config.get("training", {}) or {}).get("seed", 42))
    selected = _deterministic_record_sample(
        discovered.records, min(requested, len(discovered.records)), seed, "synthetic-mixture"
    )
    final_fraction = len(selected) / (base_train_count + len(selected)) if base_train_count else 0.0
    return selected, {
        "enabled": True,
        "source": str(source),
        "requested_fraction": fraction,
        "realized_fraction": final_fraction,
        "requested": requested,
        "selected": len(selected),
        "available": len(discovered.records),
        "train_only": True,
        "seed": seed,
    }


def _limit_summary_groups(summary: Any, setting: Any, seed: int) -> int:
    """Apply an optional deterministic group limit used by smoke experiments."""

    if setting is None:
        return 0
    limit_value = setting.get(summary.split) if isinstance(setting, dict) else setting
    if limit_value is None:
        return 0
    limit = int(limit_value)
    if limit <= 0:
        raise ValueError("data.max_groups_per_split values must be positive")
    groups = sorted({record.group_id for record in summary.records})
    chosen = {
        record.group_id
        for record in _deterministic_record_sample(
            [
                type("GroupRecord", (), {"image_path": group, "group_id": group})()
                for group in groups
            ],
            limit,
            seed,
            f"group-limit:{summary.split}",
        )
    }
    before = len(summary.records)
    summary.records = [record for record in summary.records if record.group_id in chosen]
    return before - len(summary.records)


def _refresh_summary(summary: Any) -> None:
    summary.valid_images = len(summary.records)
    summary.group_count = len({record.group_id for record in summary.records})
    summary.class_counts = {
        name: sum(1 for record in summary.records if record.label == label)
        for name, label in LABELS.items()
    }
    summary.subgroup_counts = {
        subgroup: sum(1 for record in summary.records if record.subgroup == subgroup)
        for subgroup in SUBGROUP_LABELS
    }


def _make_datasets(config: Dict[str, Any]) -> Tuple[Dict[str, LocalEditDataset], Dict[str, Any]]:
    train_root, validation_root, test_root, masks_root = _paths(config)
    extensions = (config.get("data", {}) or {}).get("extensions")
    strict = bool((config.get("data", {}) or {}).get("strict_images", False))
    subgroup_directories = _subgroup_directories(config)
    summaries = [
        discover_split(
            train_root,
            "train",
            masks_root,
            extensions,
            subgroup_directories=subgroup_directories,
            strict=strict,
        ),
        discover_split(
            validation_root,
            "validation",
            masks_root,
            extensions,
            subgroup_directories=subgroup_directories,
            strict=strict,
        ),
        discover_split(
            test_root,
            "test",
            masks_root,
            extensions,
            subgroup_directories=subgroup_directories,
            strict=strict,
        ),
    ]
    verify_split_groups(summaries)

    # Contradictory labels are removed before anything sees them, so the model
    # is never asked to learn that one pixel array is both authentic and edited.
    quarantined, quarantine_source = _resolve_quarantine(config, summaries)
    include_conflicts = bool((config.get("data", {}) or {}).get("include_conflicting_labels", False))
    excluded: Dict[str, int] = {}
    limited: Dict[str, int] = {}
    group_limit = (config.get("data", {}) or {}).get("max_groups_per_split")
    seed = int((config.get("training", {}) or {}).get("seed", 42))
    for summary in summaries:
        kept, dropped = filter_records(summary.records, quarantined, include_conflicts)
        summary.records = kept
        excluded[summary.split] = dropped
        limited[summary.split] = _limit_summary_groups(summary, group_limit, seed)
        _refresh_summary(summary)

    # The replay mixture is sized and added AFTER quarantine and the group limit
    # have settled the local-edit train set. Sizing it against the pre-limit
    # count would ask for thousands of replay images for a 24-group smoke run,
    # and adding it before the limit would let the limit -- which selects whole
    # groups, one per replay image -- discard almost all of them again.
    synthetic_records, mixture_summary = _synthetic_mixture_records(
        config, len(summaries[0].records), extensions, strict
    )
    summaries[0].records.extend(synthetic_records)
    _refresh_summary(summaries[0])
    # Re-check after the mixture lands: replay groups are train-only and carry
    # their own prefix, so this must still hold.
    verify_split_groups(summaries)

    transform = _training_transform(config)
    # Two transformed views per training image only when a consistency loss will
    # actually consume them; otherwise the extra forward passes buy nothing.
    consistency = consistency_settings(config)
    # Views default to whatever the consistency loss needs. An explicit
    # training.views_per_image overrides that so an ablation can hold the views
    # fixed across variants and change ONLY the loss term -- otherwise the
    # baseline would see different data from the variants it is compared with.
    configured_views = (config.get("training", {}) or {}).get("views_per_image")
    if configured_views is None:
        views = 2 if (consistency["enabled"] and transform is not None) else 1
    else:
        views = max(1, int(configured_views))
    datasets = {
        summary.split: LocalEditDataset(
            summary.root,
            summary.split,
            masks_root,
            transform=transform if summary.split == "train" else None,
            extensions=extensions,
            records=summary.records,
            views=views if summary.split == "train" else 1,
        )
        for summary in summaries
    }
    dataset_summary = {summary.split: summary.as_dict() for summary in summaries}
    # The exclusion is part of the manifest, not a side effect: a reader must be
    # able to see how many images were dropped and why without rerunning the audit.
    dataset_summary["views_per_training_image"] = views
    dataset_summary["synthetic_mixture"] = mixture_summary
    dataset_summary["configured_subgroups"] = {
        subgroup: {"label": label, "directory": subgroup_directories.get(subgroup)}
        for subgroup, label in SUBGROUP_LABELS.items()
    }
    dataset_summary["group_limit"] = {
        "configured": group_limit,
        "records_excluded": limited,
        "seed": seed,
    }
    dataset_summary["data_quality"] = {
        "conflicting_labels_excluded": excluded,
        "conflicting_labels_excluded_total": sum(excluded.values()),
        "include_conflicting_labels": include_conflicts,
        "quarantine": quarantine_source,
    }
    return datasets, dataset_summary


def _processor_batch(processor: Any, images: Iterable[Image.Image]) -> torch.Tensor:
    encoded = processor(images=list(images), return_tensors="pt")
    pixels = encoded.get("pixel_values") if hasattr(encoded, "get") else None
    if not isinstance(pixels, torch.Tensor):
        raise ValueError("Image processor did not return a pixel_values tensor")
    return pixels


class FeatureDataset(torch.utils.data.Dataset):
    """Tensor-backed dataset used after frozen backbone features are cached.

    Pair ids travel with the features so the consistency loss can still tell
    which rows are views of one image after the backbone has been cached away.
    """

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        pair_ids: Optional[Sequence[str]] = None,
        subgroups: Optional[Sequence[str]] = None,
    ) -> None:
        self.features = features
        self.labels = labels
        self.pair_ids = list(pair_ids) if pair_ids is not None else [
            str(index) for index in range(int(labels.shape[0]))
        ]
        self.subgroups = list(subgroups) if subgroups is not None else [
            "unknown" for _ in range(int(labels.shape[0]))
        ]
        if len(self.pair_ids) != int(labels.shape[0]):
            raise ValueError(
                f"pair_ids has {len(self.pair_ids)} entries but labels has {labels.shape[0]}"
            )
        if len(self.subgroups) != int(labels.shape[0]):
            raise ValueError(
                f"subgroups has {len(self.subgroups)} entries but labels has {labels.shape[0]}"
            )

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        return (
            self.features[index],
            self.labels[index],
            self.pair_ids[index],
            self.subgroups[index],
        )


def _batch_inputs(model: FineTuneModel, images: List[Image.Image]) -> Tuple[torch.Tensor, torch.Tensor]:
    processors = getattr(model.model, "_finetune_processors", None)
    if processors is None:
        from src.models.bombek_siglip2_dinov2 import build_bombek_processors

        processors = build_bombek_processors(model.model_config)
        model.model._finetune_processors = processors
    return _processor_batch(processors[0], images), _processor_batch(processors[1], images)


def _feature_cache_key(model: FineTuneModel, dataset: LocalEditDataset, config: Dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(str(model.checkpoint_path).encode("utf-8"))
    if model.checkpoint_path:
        checkpoint = Path(model.checkpoint_path)
        if checkpoint.exists():
            stat = checkpoint.stat()
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
    digest.update(json.dumps({"model": model.model_config, "training": config.get("training", {})}, sort_keys=True, default=str).encode("utf-8"))
    for record in dataset.records:
        stat = record.image_path.stat()
        digest.update(
            f"{record.image_path}:{record.label}:{record.subgroup}:"
            f"{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
        )
    return digest.hexdigest()


def _cache_features(
    model: FineTuneModel,
    dataset: LocalEditDataset,
    loader: DataLoader,
    cache_path: Path,
    config: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    """Extract frozen dual-backbone features once and persist them on CPU."""

    key = _feature_cache_key(model, dataset, config)
    if cache_path.is_file():
        try:
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            if isinstance(cached, dict) and cached.get("key") == key:
                return (
                    cached["features"],
                    cached["labels"],
                    cached.get("pair_ids") or [],
                    cached.get("subgroups") or [],
                )
        except (OSError, RuntimeError, ValueError, KeyError):
            pass

    print(f"Caching frozen features for {dataset.summary.split}: {len(dataset)} images")
    features: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    pair_ids: List[str] = []
    subgroups: List[str] = []
    model.train(False)
    with torch.no_grad():
        for batch in loader:
            siglip_pixels, dinov2_pixels = _batch_inputs(model, batch["images"])
            _, siglip_features, dinov2_features = model.model.forward_with_features(
                siglip_pixels.to(model.device), dinov2_pixels.to(model.device)
            )
            features.append(torch.cat([siglip_features.float(), dinov2_features.float()], dim=-1).cpu())
            labels.append(batch["labels"].cpu())
            pair_ids.extend(batch.get("pair_ids", []) or [])
            subgroups.extend(batch.get("subgroups", []) or [])
    if not features:
        raise ValueError(f"Dataset split contains no valid images: {dataset.summary.split}")
    result_features = torch.cat(features, dim=0)
    result_labels = torch.cat(labels, dim=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "key": key,
            "features": result_features,
            "labels": result_labels,
            "pair_ids": pair_ids,
            "subgroups": subgroups,
        },
        cache_path,
    )
    return result_features, result_labels, pair_ids, subgroups


def _run_feature_epoch(
    model: FineTuneModel,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    accumulation_steps: int,
    threshold: float,
    consistency: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train/evaluate only the classifier using cached frozen features."""

    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    all_labels: List[np.ndarray] = []
    all_probabilities: List[np.ndarray] = []
    losses: List[float] = []
    consistency_losses: List[float] = []
    all_subgroups: List[str] = []
    settings = consistency or {}
    batches = 0
    for batches, batch in enumerate(loader, start=1):
        features, labels, pair_ids, subgroups = batch
        logits = model.model.classifier(features.to(model.device)).reshape(-1)
        labels = labels.to(model.device)
        loss, parts = combined_loss(
            logits,
            labels,
            pair_ids,
            criterion,
            consistency_weight=float(settings.get("weight", 0.0)),
            consistency_method=str(settings.get("method", CONSISTENCY_MSE)),
            same_label_pairs_only=bool(settings.get("same_label_pairs_only", True)),
        )
        consistency_losses.append(parts["consistency_loss"])
        if training:
            (loss / max(1, accumulation_steps)).backward()
            if batches % max(1, accumulation_steps) == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu().item()))
        all_labels.append(labels.detach().cpu().numpy())
        all_probabilities.append(torch.sigmoid(logits.detach()).cpu().numpy())
        all_subgroups.extend(list(subgroups))
    if training and batches and batches % max(1, accumulation_steps) != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if not all_labels:
        raise ValueError("Cached feature dataset contains no images")
    joined_labels = np.concatenate(all_labels)
    joined_probabilities = np.concatenate(all_probabilities)
    metrics = binary_metrics(joined_labels, joined_probabilities, threshold)
    metrics = attach_loss(metrics, float(np.mean(losses)))
    metrics["consistency_loss"] = float(np.mean(consistency_losses)) if consistency_losses else 0.0
    metrics["subgroups"] = metrics_by_subgroup(
        joined_labels,
        joined_probabilities,
        all_subgroups,
        threshold,
        expected_subgroups=tuple(SUBGROUP_LABELS),
    )
    return metrics


def _run_epoch(
    model: FineTuneModel,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    accumulation_steps: int,
    threshold: float,
    consistency: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    all_labels: List[np.ndarray] = []
    all_probabilities: List[np.ndarray] = []
    losses: List[float] = []
    consistency_losses: List[float] = []
    all_subgroups: List[str] = []
    settings = consistency or {}
    batches = 0
    for batches, batch in enumerate(loader, start=1):
        siglip_pixels, dinov2_pixels = _batch_inputs(model, batch["images"])
        siglip_pixels = siglip_pixels.to(model.device)
        dinov2_pixels = dinov2_pixels.to(model.device)
        labels = batch["labels"].to(model.device)
        logits = (
            model.forward_head_only(siglip_pixels, dinov2_pixels)
            if model.mode == "head_only"
            else model(siglip_pixels, dinov2_pixels)
        )
        loss, parts = combined_loss(
            logits,
            labels,
            batch.get("pair_ids") or batch.get("group_ids") or [],
            criterion,
            consistency_weight=float(settings.get("weight", 0.0)),
            consistency_method=str(settings.get("method", CONSISTENCY_MSE)),
            same_label_pairs_only=bool(settings.get("same_label_pairs_only", True)),
        )
        consistency_losses.append(parts["consistency_loss"])
        if training:
            (loss / max(1, accumulation_steps)).backward()
            if batches % max(1, accumulation_steps) == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu().item()))
        all_labels.append(labels.detach().cpu().numpy())
        all_probabilities.append(torch.sigmoid(logits.detach()).cpu().numpy())
        all_subgroups.extend(batch.get("subgroups", []) or ["unknown"] * len(labels))
    if training and batches and batches % max(1, accumulation_steps) != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if not all_labels:
        raise ValueError("Dataset split contains no valid images")
    joined_labels = np.concatenate(all_labels)
    joined_probabilities = np.concatenate(all_probabilities)
    metrics = binary_metrics(joined_labels, joined_probabilities, threshold)
    metrics = attach_loss(metrics, float(np.mean(losses)))
    metrics["consistency_loss"] = float(np.mean(consistency_losses)) if consistency_losses else 0.0
    metrics["subgroups"] = metrics_by_subgroup(
        joined_labels,
        joined_probabilities,
        all_subgroups,
        threshold,
        expected_subgroups=tuple(SUBGROUP_LABELS),
    )
    return metrics


def _json_safe_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def train(config: Dict[str, Any], mode: Optional[str] = None, device: Optional[str] = None) -> Dict[str, Any]:
    """Train and save a best adapter; the test set is evaluated only at the end."""

    training_config = config.get("training", {}) or {}
    seed = int(training_config.get("seed", 42))
    seed_everything(seed)
    mode = mode or str(training_config.get("mode", "head_only"))
    model_config = config.get("model", {}) or {}
    checkpoint = resolve_config_path(config, model_config.get("checkpoint", "models/pretrained/pytorch_model.pt"))
    model = FineTuneModel.from_checkpoint(checkpoint, config, device=device, mode=mode)
    if parameter_counts(model.model)["total"] >= int(model_config.get("parameter_limit", MAX_PARAMETERS)):
        raise ValueError("Fine-tuned model is at or above the configured parameter limit")

    datasets, dataset_summary = _make_datasets(config)
    results_dir = resolve_config_path(config, (config.get("outputs", {}) or {}).get("directory", "outputs/lora_finetune"))
    results_dir.mkdir(parents=True, exist_ok=True)
    batch_size = max(1, int(training_config.get("batch_size", 2)))
    workers = max(0, int((config.get("data", {}) or {}).get("num_workers", 0)))
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, generator=generator, num_workers=workers, collate_fn=local_edit_collate, worker_init_fn=_worker_init, pin_memory=False),
        "validation": DataLoader(datasets["validation"], batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=local_edit_collate, worker_init_fn=_worker_init, pin_memory=False),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=local_edit_collate, worker_init_fn=_worker_init, pin_memory=False),
    }
    epoch_loaders: Dict[str, DataLoader] = loaders
    if str(mode).lower() == "head_only":
        # head_only caches the frozen backbone features once and then trains the
        # classifier on those tensors for every epoch. Any random augmentation
        # would therefore be drawn exactly once per image and frozen for the
        # whole run -- the model would see one fixed transformed copy, not a
        # fresh draw per epoch, which is not what augmentation is for. Refuse
        # the combination instead of silently training on frozen augmentations.
        if datasets["train"].transform is not None and datasets["train"].views < 2:
            raise ValueError(
                "training.mode='head_only' caches frozen backbone features once, so "
                "training.official_transformations would be applied a single time and "
                "then reused unchanged for every epoch. Use training.mode='lora' to "
                "augment per epoch, set official_transformations to [], or enable "
                "training.consistency so the cached views form fixed transformation "
                "pairs the consistency loss can still learn from."
            )
        cache_dir = resolve_config_path(
            config,
            (config.get("outputs", {}) or {}).get("feature_cache_dir", "outputs/lora_finetune/feature_cache"),
        )
        feature_loaders: Dict[str, DataLoader] = {}
        for split in ("train", "validation", "test"):
            cache_path = cache_dir / f"{split}.pt"
            features, labels, pair_ids, subgroups = _cache_features(
                model, datasets[split], loaders[split], cache_path, config
            )
            feature_loaders[split] = DataLoader(
                FeatureDataset(features, labels, pair_ids, subgroups),
                batch_size=batch_size,
                shuffle=split == "train",
                generator=generator if split == "train" else None,
                num_workers=0,
            )
        epoch_loaders = feature_loaders
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for name, p in model.named_parameters() if p.requires_grad and name.startswith("model.classifier.")], "lr": float(training_config.get("classifier_learning_rate", 0.001))},
            {"params": [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("model.classifier.")], "lr": float(training_config.get("lora_learning_rate", 0.0001))},
        ],
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    if not trainable:
        raise ValueError("No trainable parameters were enabled")
    criterion = binary_loss()
    consistency = consistency_settings(config)
    threshold = float((config.get("evaluation", {}) or {}).get("selection_threshold", 0.5))
    selection_metric = str(training_config.get("selection_metric", "f1"))
    patience = max(0, int(training_config.get("early_stopping_patience", 2)))
    best_value = float("-inf")
    best_epoch = 0
    stale = 0
    best_adapter: Optional[Dict[str, torch.Tensor]] = None
    best_classifier: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict[str, Any]] = []
    epochs = max(1, int(training_config.get("epochs", 5)))
    epoch_runner = _run_feature_epoch if str(mode).lower() == "head_only" else _run_epoch
    with torch.no_grad():
        original_validation_metrics = epoch_runner(
            model, epoch_loaders["validation"], criterion, None, 1, threshold
        )
        original_test_metrics = epoch_runner(
            model, epoch_loaders["test"], criterion, None, 1, threshold
        )
    for epoch in range(1, epochs + 1):
        train_metrics = epoch_runner(
            model,
            epoch_loaders["train"],
            criterion,
            optimizer,
            int(training_config.get("gradient_accumulation_steps", 1)),
            threshold,
            consistency,
        )
        with torch.no_grad():
            # Validation and test measure classification only. Adding the
            # consistency term to the selection metric would let a model win by
            # being self-consistent and wrong.
            validation_metrics = epoch_runner(
                model, epoch_loaders["validation"], criterion, None, 1, threshold
            )
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        print(
            f"epoch {epoch}/{epochs} | train loss={train_metrics['loss']:.4f} "
            f"f1={train_metrics['f1']:.4f} | validation loss={validation_metrics['loss']:.4f} "
            f"f1={validation_metrics['f1']:.4f}"
        )
        value = float(validation_metrics.get(selection_metric, validation_metrics["f1"]))
        if value > best_value:
            best_value = value
            best_epoch = epoch
            stale = 0
            best_adapter = copy.deepcopy(model.adapter_state_dict())
            best_classifier = copy.deepcopy(model.classifier_state_dict())
        else:
            stale += 1
            if stale > patience:
                break
    if best_adapter is None or best_classifier is None:
        raise RuntimeError("Training completed without a best validation checkpoint")
    model.model.load_state_dict(best_adapter, strict=False)
    model.model.classifier.load_state_dict(best_classifier, strict=True)
    with torch.no_grad():
        test_metrics = epoch_runner(model, epoch_loaders["test"], criterion, None, 1, threshold)
    comparison_metrics = {
        "threshold": threshold,
        "threshold_note": (
            "The original and fine-tuned checkpoints are compared at the same fixed "
            "pre-training selection threshold; no test-set threshold fitting occurred."
        ),
        "original": {
            "validation": original_validation_metrics,
            "test": original_test_metrics,
        },
        "fine_tuned": {
            "validation": history[best_epoch - 1]["validation"],
            "test": test_metrics,
        },
        "test_delta": {
            name: float(test_metrics[name]) - float(original_test_metrics[name])
            for name in ("accuracy", "precision", "recall", "f1", "auroc", "balanced_accuracy")
            if test_metrics.get(name) is not None and original_test_metrics.get(name) is not None
        },
    }
    output_dir = resolve_config_path(config, model_config.get("output_adapter", "models/adapters/local_edit_lora"))
    metadata = {
        "seed": seed,
        "mode": mode,
        "consistency": consistency,
        "best_epoch": best_epoch,
        "selection_metric": selection_metric,
        "best_validation_metrics": history[best_epoch - 1]["validation"],
        "test_metrics": test_metrics,
        "original_vs_fine_tuned": comparison_metrics,
        "dataset_summary": dataset_summary,
        "parameter_counts": parameter_counts(model.model),
    }
    saved = model.save_adapter(output_dir, metadata=metadata)
    (results_dir / "training_config.json").write_text(json.dumps(_json_safe_config(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (results_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (results_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (results_dir / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (results_dir / "original_vs_fine_tuned.json").write_text(
        json.dumps(comparison_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"saved": saved, **metadata}
