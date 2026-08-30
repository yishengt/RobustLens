"""Charts and failure grids for the Track 5 evaluation protocol.

Every figure here is meant to support one conclusion. matplotlib is optional:
if it is missing, chart generation is skipped with a message and the numeric
report is unaffected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.evaluation.protocol import (
    CLEAN_KEY,
    VARIANT_LABELS,
    ScoredImage,
)

AUTHENTIC_COLOUR = "#2b8a3e"
AI_COLOUR = "#c92a2a"
NEUTRAL_COLOUR = "#495057"


def _pyplot():
    """Return matplotlib.pyplot with a headless backend, or None."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def plot_robustness(
    summary: Dict[str, Any], output_dir: str | Path, metric: str = "accuracy"
) -> Optional[str]:
    """Clean vs per-transformation performance, sorted by how much was lost."""

    plt = _pyplot()
    rows = summary.get("per_transformation") or []
    if plt is None or not rows:
        return None

    names = [row["transformation"] for row in rows]
    values = [row[metric] for row in rows]
    clean = summary["clean"]

    figure, axes = plt.subplots(figsize=(11, 5))
    colours = [AI_COLOUR if value < clean else AUTHENTIC_COLOUR for value in values]
    axes.bar(range(len(values)), values, color=colours)
    axes.axhline(
        clean,
        color=NEUTRAL_COLOUR,
        linestyle="--",
        linewidth=1.4,
        label=f"clean {metric} = {clean:.3f}",
    )
    if summary.get("worst_case") is not None:
        axes.axhline(
            summary["worst_case"],
            color="#e8590c",
            linestyle=":",
            linewidth=1.2,
            label=f"worst case = {summary['worst_case']:.3f}",
        )
    axes.set_xticks(range(len(names)))
    axes.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes.set_ylabel(metric.replace("_", " "))
    axes.set_ylim(0, 1.02)
    axes.set_title(f"{metric.replace('_', ' ').title()} by transformation (one fixed threshold)")
    axes.legend(fontsize=8)
    figure.tight_layout()
    path = Path(output_dir) / f"robustness_{metric}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return str(path)


def plot_severity_curves(
    per_version: Dict[str, Dict[str, Any]], output_dir: str | Path, metric: str = "accuracy"
) -> Optional[str]:
    """Performance against transformation severity, one line per family."""

    plt = _pyplot()
    if plt is None or not per_version:
        return None

    families = {
        "JPEG quality": [("jpeg_q90", 90), ("jpeg_q70", 70), ("jpeg_q50", 50), ("jpeg_q30", 30)],
        "Blur sigma": [("blur_s0.5", 0.5), ("blur_s1", 1.0), ("blur_s2", 2.0)],
        "Noise sigma": [("noise_s0.02", 0.02), ("noise_s0.05", 0.05), ("noise_s0.1", 0.10)],
        "Resize scale": [("resize_0.5x", 0.5), ("resize_0.25x", 0.25)],
    }
    present = {
        title: [(name, sev) for name, sev in items if name in per_version]
        for title, items in families.items()
    }
    present = {title: items for title, items in present.items() if len(items) >= 2}
    if not present:
        return None

    clean = per_version.get(CLEAN_KEY, {}).get(metric)
    figure, axes_list = plt.subplots(
        1, len(present), figsize=(4.2 * len(present), 3.6), squeeze=False
    )
    for axes, (title, items) in zip(axes_list[0], present.items()):
        severities = [sev for _, sev in items]
        values = [per_version[name][metric] for name, _ in items]
        axes.plot(severities, values, marker="o", color=AI_COLOUR)
        if clean is not None:
            axes.axhline(clean, color=NEUTRAL_COLOUR, linestyle="--", linewidth=1.0)
        axes.set_title(title, fontsize=10)
        axes.set_ylim(0, 1.02)
        axes.set_xlabel("severity")
        axes.grid(alpha=0.25)
    axes_list[0][0].set_ylabel(metric.replace("_", " "))
    figure.suptitle("Performance against transformation severity", fontsize=11)
    figure.tight_layout()
    path = Path(output_dir) / f"severity_{metric}.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return str(path)


def plot_confusion_matrices(variants: Dict[str, Any], output_dir: str | Path) -> Optional[str]:
    """Confusion matrix per system variant."""

    plt = _pyplot()
    usable = {k: v for k, v in variants.items() if v.get("metrics")}
    if plt is None or not usable:
        return None

    figure, axes_list = plt.subplots(
        1, len(usable), figsize=(3.4 * len(usable), 3.4), squeeze=False
    )
    for axes, (name, payload) in zip(axes_list[0], usable.items()):
        matrix = payload["metrics"]["confusion_matrix"]
        grid = np.array(
            [
                [matrix["true_negatives"], matrix["false_positives"]],
                [matrix["false_negatives"], matrix["true_positives"]],
            ],
            dtype=float,
        )
        axes.imshow(grid, cmap="Blues")
        for i in range(2):
            for j in range(2):
                axes.text(
                    j,
                    i,
                    int(grid[i, j]),
                    ha="center",
                    va="center",
                    color="white" if grid[i, j] > grid.max() / 2 else "black",
                    fontsize=12,
                )
        axes.set_xticks([0, 1])
        axes.set_xticklabels(["pred real", "pred AI"], fontsize=8)
        axes.set_yticks([0, 1])
        axes.set_yticklabels(["true real", "true AI"], fontsize=8)
        axes.set_title(VARIANT_LABELS.get(name, name), fontsize=9)
    figure.suptitle("Confusion matrices at the frozen threshold", fontsize=11)
    figure.tight_layout()
    path = Path(output_dir) / "confusion_matrices.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return str(path)


def plot_confidence_distributions(
    distributions: Dict[str, Any], threshold: float, output_dir: str | Path
) -> Optional[str]:
    """Score distributions for authentic vs AI images, with the threshold marked."""

    plt = _pyplot()
    if plt is None or not distributions:
        return None

    figure, axes = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 26)
    for key, colour, label in (
        ("authentic", AUTHENTIC_COLOUR, "Authentic"),
        ("ai_generated", AI_COLOUR, "AI-generated"),
    ):
        payload = distributions.get(key)
        if not payload:
            continue
        axes.hist(
            payload["scores"],
            bins=bins,
            alpha=0.6,
            color=colour,
            label=f"{label} (n={payload['count']})",
        )
    axes.axvline(
        threshold, color=NEUTRAL_COLOUR, linestyle="--", label=f"frozen threshold = {threshold:.2f}"
    )
    axes.set_xlabel("fused AI-generated probability")
    axes.set_ylabel("images")
    axes.set_title("Confidence distributions by true class")
    axes.legend(fontsize=8)
    figure.tight_layout()
    path = Path(output_dir) / "confidence_distributions.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return str(path)


def plot_variant_comparison(variants: Dict[str, Any], output_dir: str | Path) -> Optional[str]:
    """Grouped bars comparing the four systems on the headline metrics."""

    plt = _pyplot()
    usable = {k: v for k, v in variants.items() if v.get("metrics")}
    if plt is None or not usable:
        return None

    metrics = ["accuracy", "balanced_accuracy", "f1", "auc", "false_positive_rate"]
    figure, axes = plt.subplots(figsize=(10, 4.5))
    width = 0.8 / len(usable)
    for offset, (name, payload) in enumerate(usable.items()):
        values = [payload["metrics"].get(metric) or 0.0 for metric in metrics]
        axes.bar(
            np.arange(len(metrics)) + offset * width,
            values,
            width,
            label=VARIANT_LABELS.get(name, name),
        )
    axes.set_xticks(np.arange(len(metrics)) + 0.4 - width / 2)
    axes.set_xticklabels([m.replace("_", " ") for m in metrics], fontsize=9)
    axes.set_ylim(0, 1.05)
    axes.set_title("System ablation on identical images and one frozen threshold")
    axes.legend(fontsize=8)
    figure.tight_layout()
    path = Path(output_dir) / "variant_comparison.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return str(path)


def save_failure_grid(
    records: Sequence[ScoredImage],
    entries: List[Dict[str, Any]],
    image_lookup: Dict[str, Any],
    title: str,
    output_path: str | Path,
) -> Optional[str]:
    """Save a labelled grid of the most confident mistakes."""

    plt = _pyplot()
    available = [entry for entry in entries if entry["img_id"] in image_lookup]
    if plt is None or not available:
        return None

    columns = min(4, len(available))
    rows = int(np.ceil(len(available) / columns))
    figure, axes_list = plt.subplots(
        rows, columns, figsize=(3.2 * columns, 3.5 * rows), squeeze=False
    )
    for axes in axes_list.flat:
        axes.axis("off")
    for axes, entry in zip(axes_list.flat, available):
        axes.imshow(image_lookup[entry["img_id"]])
        axes.set_title(
            f"{entry['class_name']}\nfused={entry['score']:.3f} clean={entry['clean_score']:.3f}",
            fontsize=8,
        )
        axes.axis("off")
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return str(path)


def plot_subgroups(subgroups: Dict[str, Any], output_dir: str | Path) -> Optional[str]:
    """Per-family accuracy and recall, each against the shared authentic pool."""

    plt = _pyplot()
    usable = {k: v for k, v in subgroups.items() if v.get("metrics")}
    if plt is None or not usable:
        return None

    names = list(usable)
    figure, axes = plt.subplots(figsize=(7, 4))
    for offset, metric in enumerate(("accuracy", "recall", "false_positive_rate")):
        values = [usable[name]["metrics"].get(metric) or 0.0 for name in names]
        axes.bar(
            np.arange(len(names)) + offset * 0.26, values, 0.26, label=metric.replace("_", " ")
        )
    axes.set_xticks(np.arange(len(names)) + 0.26)
    axes.set_xticklabels(names, fontsize=9)
    axes.set_ylim(0, 1.05)
    axes.set_title("Performance by generation-process family (proxy, not a generator holdout)")
    axes.legend(fontsize=8)
    figure.tight_layout()
    path = Path(output_dir) / "generation_families.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return str(path)
