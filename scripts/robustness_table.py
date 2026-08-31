#!/usr/bin/env python3
"""Summarise clean vs transformed detection performance.

Reads the detailed JSON written by ``scripts/run_inference.py`` together with a
label map, and reports AUROC, accuracy, TPR and FPR for the clean image and for
every transformation, at one threshold held fixed across all conditions.

A threshold that is re-fitted per condition would hide exactly the degradation
this table exists to measure, so the clean-image operating point is chosen once
and then frozen.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Rank-based AUROC with correct handling of tied scores."""

    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return float("nan")
    ordered = sorted(zip(scores, labels))
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index
        while end + 1 < len(ordered) and ordered[end + 1][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end + 1) / 2
        rank_sum += sum(average_rank for k in range(index, end + 1) if ordered[k][1] == 1)
        index = end + 1
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def rates(scores: Sequence[float], labels: Sequence[int], threshold: float) -> Tuple[float, float, float]:
    tp = fp = tn = fn = 0
    for score, label in zip(scores, labels):
        predicted = score >= threshold
        if label == 1 and predicted:
            tp += 1
        elif label == 1:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    tpr = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    return accuracy, tpr, fpr


def youden_threshold(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Threshold maximising TPR - FPR on the clean condition."""

    best, best_j = 0.5, -2.0
    for candidate in sorted(set(scores)):
        _, tpr, fpr = rates(scores, labels, candidate)
        if tpr - fpr > best_j:
            best_j, best = tpr - fpr, candidate
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detailed", default="outputs/baseline_detailed.json")
    parser.add_argument("--labels", default="data/baseline_subset_labels.json")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Fixed decision threshold; default fits Youden J on the clean condition")
    parser.add_argument("--output", default=None, help="Optional JSON path for the table")
    args = parser.parse_args()

    rows = json.loads((PROJECT_ROOT / args.detailed).read_text(encoding="utf-8"))
    labels_map = json.loads((PROJECT_ROOT / args.labels).read_text(encoding="utf-8"))

    per_condition: Dict[str, List[Tuple[float, int]]] = collections.defaultdict(list)
    per_generator_clean: Dict[str, List[Tuple[float, int]]] = collections.defaultdict(list)
    skipped = 0
    for row in rows:
        name = Path(row.get("image_path", "")).name
        meta = labels_map.get(name)
        if meta is None or not row.get("per_transformation_predictions"):
            skipped += 1
            continue
        label = int(meta["label"])
        for condition, score in row["per_transformation_predictions"].items():
            per_condition[condition].append((float(score), label))
        clean = row["per_transformation_predictions"].get("clean")
        if clean is not None:
            per_generator_clean[meta["generator"]].append((float(clean), label))

    clean_scores = [s for s, _ in per_condition["clean"]]
    clean_labels = [y for _, y in per_condition["clean"]]
    threshold = args.threshold if args.threshold is not None else youden_threshold(clean_scores, clean_labels)

    print(f"n={len(clean_labels)} images   frozen threshold={threshold:.4f}"
          f"   (skipped {skipped} row(s) without labels/scores)\n")
    header = f"{'condition':<18}{'AUROC':>8}{'acc':>8}{'TPR':>8}{'FPR':>8}{'ΔAUROC':>9}"
    print(header)
    print("-" * len(header))

    clean_auc = auroc(clean_scores, clean_labels)
    table = {}
    ordered = ["clean"] + sorted(k for k in per_condition if k != "clean")
    for condition in ordered:
        pairs = per_condition[condition]
        scores = [s for s, _ in pairs]
        ys = [y for _, y in pairs]
        area = auroc(scores, ys)
        accuracy, tpr, fpr = rates(scores, ys, threshold)
        delta = "" if condition == "clean" else f"{area - clean_auc:+.3f}"
        print(f"{condition:<18}{area:>8.3f}{accuracy:>8.3f}{tpr:>8.3f}{fpr:>8.3f}{delta:>9}")
        table[condition] = {"auroc": area, "accuracy": accuracy, "tpr": tpr, "fpr": fpr}

    print(f"\nPer-generator recall on the clean image (threshold {threshold:.4f}):")
    for generator in sorted(per_generator_clean):
        pairs = per_generator_clean[generator]
        ys = [y for _, y in pairs]
        if not ys or ys[0] == 0:
            accuracy, _, fpr = rates([s for s, _ in pairs], ys, threshold)
            print(f"  {generator:16s} n={len(pairs):4d}  FPR={fpr:.3f}")
        else:
            _, tpr, _ = rates([s for s, _ in pairs], ys, threshold)
            print(f"  {generator:16s} n={len(pairs):4d}  recall={tpr:.3f}")

    if args.output:
        path = PROJECT_ROOT / args.output
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"threshold": threshold, "conditions": table}, indent=1), encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
