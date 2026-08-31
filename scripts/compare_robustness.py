#!/usr/bin/env python3
"""Compare two robustness sweeps at one threshold held fixed across both.

Reads the detailed JSON from two runs of scripts/run_inference.py -- typically
the original checkpoint and a fine-tuned adapter -- and reports per-condition
AUROC and TPR/FPR deltas.

Both models are judged at the SAME threshold, fitted once on the baseline's
clean condition. Re-fitting per model would let a fine-tune claim credit for a
shift the threshold could have absorbed on its own, which is exactly the
comparison this script exists to prevent.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
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
        average = (index + 1 + end + 1) / 2
        rank_sum += sum(average for k in range(index, end + 1) if ordered[k][1] == 1)
        index = end + 1
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def rates(scores: Sequence[float], labels: Sequence[int], threshold: float) -> Tuple[float, float, float]:
    tp = fp = tn = fn = 0
    for score, label in zip(scores, labels):
        hit = score >= threshold
        if label == 1 and hit:
            tp += 1
        elif label == 1:
            fn += 1
        elif hit:
            fp += 1
        else:
            tn += 1
    total = max(1, tp + tn + fp + fn)
    return (tp + tn) / total, tp / max(1, tp + fn), fp / max(1, fp + tn)


def load(path: Path, labels_map: Dict[str, dict]) -> Dict[str, List[Tuple[float, int]]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    per: Dict[str, List[Tuple[float, int]]] = collections.defaultdict(list)
    for row in rows:
        name = Path(row.get("image_path", "")).name
        meta = labels_map.get(name)
        scores = row.get("per_transformation_predictions")
        if meta is None or not scores:
            continue
        for condition, score in scores.items():
            per[condition].append((float(score), int(meta["label"])))
    return per


def youden(scores: Sequence[float], labels: Sequence[int]) -> float:
    best, best_j = 0.5, -2.0
    for candidate in sorted(set(scores)):
        _, tpr, fpr = rates(scores, labels, candidate)
        if tpr - fpr > best_j:
            best_j, best = tpr - fpr, candidate
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="outputs/baseline_detailed.json")
    parser.add_argument("--candidate", default="outputs/finetuned_detailed.json")
    parser.add_argument("--labels", default="data/baseline_subset_labels.json")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    labels_map = json.loads((PROJECT_ROOT / args.labels).read_text(encoding="utf-8"))
    base = load(PROJECT_ROOT / args.baseline, labels_map)
    cand = load(PROJECT_ROOT / args.candidate, labels_map)

    clean = base["clean"]
    threshold = args.threshold if args.threshold is not None else youden(
        [s for s, _ in clean], [y for _, y in clean]
    )

    conditions = ["clean"] + sorted(k for k in base if k != "clean")
    header = (f"{'condition':<17}{'AUROC base':>11}{'AUROC ft':>10}{'Δ':>8}"
              f"{'TPR base':>10}{'TPR ft':>8}{'Δ':>8}{'FPR base':>10}{'FPR ft':>8}{'Δ':>8}")
    print(f"threshold held fixed at {threshold:.4f} for BOTH models "
          f"(fitted once on the baseline clean condition)\n")
    print(header)
    print("-" * len(header))

    summary = {}
    transformed_deltas: List[float] = []
    tpr_deltas: List[float] = []
    for condition in conditions:
        if condition not in cand:
            continue
        bs = [s for s, _ in base[condition]]
        by = [y for _, y in base[condition]]
        cs = [s for s, _ in cand[condition]]
        cy = [y for _, y in cand[condition]]
        ba, ca = auroc(bs, by), auroc(cs, cy)
        _, btpr, bfpr = rates(bs, by, threshold)
        _, ctpr, cfpr = rates(cs, cy, threshold)
        print(f"{condition:<17}{ba:>11.3f}{ca:>10.3f}{ca - ba:>+8.3f}"
              f"{btpr:>10.3f}{ctpr:>8.3f}{ctpr - btpr:>+8.3f}"
              f"{bfpr:>10.3f}{cfpr:>8.3f}{cfpr - bfpr:>+8.3f}")
        summary[condition] = {"auroc_base": ba, "auroc_candidate": ca,
                              "tpr_base": btpr, "tpr_candidate": ctpr,
                              "fpr_base": bfpr, "fpr_candidate": cfpr}
        if condition != "clean":
            transformed_deltas.append(ca - ba)
            tpr_deltas.append(ctpr - btpr)

    print("\nTransformed conditions only (the axis this fine-tune targeted):")
    print(f"  mean ΔAUROC {sum(transformed_deltas)/len(transformed_deltas):+.4f}"
          f"   mean ΔTPR {sum(tpr_deltas)/len(tpr_deltas):+.4f}")
    print(f"  conditions improved on TPR: {sum(1 for d in tpr_deltas if d > 0)}/{len(tpr_deltas)}")
    print(f"  conditions improved on AUROC: {sum(1 for d in transformed_deltas if d > 0)}/{len(transformed_deltas)}")

    if args.output:
        path = PROJECT_ROOT / args.output
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"threshold": threshold, "conditions": summary}, indent=1),
                        encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
