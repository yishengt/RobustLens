#!/usr/bin/env python3
"""Compare the original checkpoint against a fine-tuned adapter, held out.

Both models see exactly the same images, the same transformations and the same
chains, so any difference is the adapter and not the sample. The original
checkpoint is never modified: the adapter is applied to a second, freshly
loaded copy of it.

The retention rule is fixed here, before any result is seen: keep the adapter
only if it improves local-edit recall or F1 without an unacceptable rise in
false positives on authentic images.

    python scripts/compare_finetuned.py \
        --config configs/lora_finetune_smoke.yaml \
        --adapter models/adapters/local_edit_smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.chains import (  # noqa: E402
    build_generation_chains,
    build_named_chains,
    generate_chain_variants,
)
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.finetune.model import load_saved_adapter_into_model  # noqa: E402
from src.finetune.train_lora import _make_datasets  # noqa: E402
from src.pipeline.model_loader import load_model  # noqa: E402
from src.pipeline.prediction import predict_images  # noqa: E402
from src.pipeline.preprocessing import Preprocessor  # noqa: E402
from src.pipeline.transformations import build_transform_specs, generate_variants  # noqa: E402
from src.utils.config import load_config  # noqa: E402

# Pre-registered retention rule.
MIN_LOCAL_EDIT_GAIN = 0.01          # recall or F1 must rise by at least this
MAX_AUTHENTIC_FPR_INCREASE = 0.05   # ...without FPR rising more than this


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lora_finetune_smoke.yaml")
    parser.add_argument("--base-config", default="configs/config.yaml")
    parser.add_argument("--adapter", default="models/adapters/local_edit_smoke")
    parser.add_argument("--checkpoint", default="models/pretrained/pytorch_model.pt")
    parser.add_argument("--output-dir", default="outputs/finetune_comparison")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--skip-chains", action="store_true", help="Skip the chain evaluation (faster)"
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def _resolve(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _metrics(labels: Sequence[int], scores: Sequence[float], threshold: float) -> Optional[Dict]:
    labels = list(labels)
    if not labels or len(set(labels)) < 2:
        # AUROC is undefined with one class present; report counts, not a
        # fabricated score.
        if not labels:
            return None
        predicted = [float(s) >= threshold for s in scores]
        positive = labels[0] == 1
        correct = sum(1 for p in predicted if p == positive)
        return {
            "count": len(labels),
            "single_class": "ai_edited" if positive else "authentic",
            "accuracy": round(correct / len(labels), 6),
            "note": "Only one class present; AUROC/F1 are undefined and omitted.",
        }
    m = compute_metrics(labels, list(scores), threshold=threshold)
    return {
        "count": m.count,
        "accuracy": round(m.accuracy, 6),
        "balanced_accuracy": round(m.balanced_accuracy, 6),
        "precision": round(m.precision, 6),
        "recall": round(m.recall, 6),
        "f1": round(m.f1, 6),
        "auroc": round(m.auc, 6),
        "fpr": round(m.false_positive_rate, 6),
        "fnr": round(m.false_negative_rate, 6),
        "confusion_matrix": {
            "true_positives": m.true_positives,
            "false_positives": m.false_positives,
            "true_negatives": m.true_negatives,
            "false_negatives": m.false_negatives,
        },
    }


def _score_all(bundle, images, preprocessor, batch_size) -> np.ndarray:
    return np.asarray(predict_images(bundle, images, preprocessor, batch_size=batch_size))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    adapter_dir = _resolve(args.adapter)
    if not (adapter_dir / "adapter_model.safetensors").is_file():
        print(
            f"No adapter at {adapter_dir}. Train one first:\n"
            f"  python scripts/train_local_edit_lora.py --config {args.config}",
            file=sys.stderr,
        )
        return 1

    train_config = load_config(_resolve(args.config))
    base_config = load_config(_resolve(args.base_config))
    preprocessor = Preprocessor.from_config(base_config)

    datasets, dataset_summary = _make_datasets(train_config)
    records = list(datasets["test"].records)
    if args.max_images:
        records = records[: args.max_images]
    if not records:
        print("The held-out test split is empty.", file=sys.stderr)
        return 1

    from PIL import Image

    images: List[Any] = []
    labels: List[int] = []
    subgroups: List[str] = []
    for record in records:
        with Image.open(record.image_path) as handle:
            images.append(handle.convert("RGB"))
        labels.append(int(record.label))
        subgroups.append(str(record.subgroup))

    transform_specs = build_transform_specs(base_config)
    chains = [] if args.skip_chains else build_named_chains() + build_generation_chains(1234)

    results: Dict[str, Any] = {}
    for name, apply_adapter in (("original", False), ("fine_tuned", True)):
        bundle = load_model(_resolve(args.checkpoint), base_config, device=args.device)
        if apply_adapter:
            load_saved_adapter_into_model(bundle.model, adapter_dir)

        started = time.time()
        clean = _score_all(bundle, images, preprocessor, args.batch_size)
        clean_seconds = time.time() - started

        entry: Dict[str, Any] = {
            "clean": _metrics(labels, clean, args.threshold),
            "subgroups": {},
            "official_transformations": {},
            "chains": {},
            "runtime": {
                "clean_seconds_total": round(clean_seconds, 3),
                "clean_seconds_per_image": round(clean_seconds / len(images), 4),
            },
        }

        for subgroup in sorted(set(subgroups)):
            index = [i for i, value in enumerate(subgroups) if value == subgroup]
            entry["subgroups"][subgroup] = _metrics(
                [labels[i] for i in index], [clean[i] for i in index], args.threshold
            )
            # Every subgroup is also scored against the shared authentic pool so
            # a single-class family still yields a usable AUROC.
            authentic = [i for i, v in enumerate(subgroups) if v == "authentic"]
            if subgroup != "authentic" and authentic:
                combined = sorted(set(index) | set(authentic))
                entry["subgroups"][f"{subgroup}_vs_authentic"] = _metrics(
                    [labels[i] for i in combined],
                    [clean[i] for i in combined],
                    args.threshold,
                )

        started = time.time()
        for spec in transform_specs:
            variants = []
            for image in images:
                built, _ = generate_variants(image, base_config, [spec])
                variants.append(built[spec.name])
            scores = _score_all(bundle, variants, preprocessor, args.batch_size)
            entry["official_transformations"][spec.name] = _metrics(
                labels, scores, args.threshold
            )
        entry["runtime"]["transformations_seconds"] = round(time.time() - started, 3)

        if chains:
            started = time.time()
            for spec in chains:
                variants = []
                for image in images:
                    built, _ = generate_chain_variants(image, [spec])
                    variants.append(built[spec.name])
                scores = _score_all(bundle, variants, preprocessor, args.batch_size)
                entry["chains"][spec.name] = _metrics(labels, scores, args.threshold)
            entry["runtime"]["chains_seconds"] = round(time.time() - started, 3)

        results[name] = entry
        if not args.quiet:
            print(f"scored {name}: {len(images)} images")

    # --- pre-registered retention decision ------------------------------
    def local_edit(entry: Dict[str, Any], key: str) -> Optional[float]:
        for candidate in ("minor_edit_vs_authentic", "minor_edit"):
            block = entry["subgroups"].get(candidate)
            if isinstance(block, dict) and key in block:
                return float(block[key])
        return None

    original_recall = local_edit(results["original"], "recall")
    tuned_recall = local_edit(results["fine_tuned"], "recall")
    original_f1 = local_edit(results["original"], "f1")
    tuned_f1 = local_edit(results["fine_tuned"], "f1")
    original_fpr = (results["original"]["clean"] or {}).get("fpr")
    tuned_fpr = (results["fine_tuned"]["clean"] or {}).get("fpr")

    reasons: List[str] = []
    improved = False
    if None not in (original_recall, tuned_recall) and tuned_recall - original_recall >= MIN_LOCAL_EDIT_GAIN:
        improved = True
        reasons.append(f"local-edit recall {original_recall:.3f} -> {tuned_recall:.3f}")
    if None not in (original_f1, tuned_f1) and tuned_f1 - original_f1 >= MIN_LOCAL_EDIT_GAIN:
        improved = True
        reasons.append(f"local-edit F1 {original_f1:.3f} -> {tuned_f1:.3f}")

    fpr_acceptable = True
    if None not in (original_fpr, tuned_fpr):
        delta = tuned_fpr - original_fpr
        fpr_acceptable = delta <= MAX_AUTHENTIC_FPR_INCREASE
        reasons.append(f"authentic FPR {original_fpr:.3f} -> {tuned_fpr:.3f} ({delta:+.3f})")

    keep = bool(improved and fpr_acceptable)
    decision = {
        "keep_fine_tuned_model": keep,
        "rule": (
            f"Keep only if local-edit recall or F1 improves by >= {MIN_LOCAL_EDIT_GAIN} "
            f"and authentic FPR rises by <= {MAX_AUTHENTIC_FPR_INCREASE}."
        ),
        "improved_local_edit": improved,
        "false_positives_acceptable": fpr_acceptable,
        "evidence": reasons,
        "verdict": (
            "Adopt the fine-tuned adapter."
            if keep
            else "Do NOT adopt the fine-tuned adapter; the original checkpoint stands."
        ),
    }

    report = {
        "config": str(_resolve(args.config)),
        "adapter": str(adapter_dir),
        "checkpoint": str(_resolve(args.checkpoint)),
        "threshold": args.threshold,
        "held_out_images": len(images),
        "subgroup_counts": {s: subgroups.count(s) for s in sorted(set(subgroups))},
        "dataset_summary": dataset_summary.get("test"),
        "results": results,
        "decision": decision,
    }
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "comparison.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if not args.quiet:
        print(f"\nHeld-out images: {len(images)}   threshold {args.threshold}")
        print(f"Subgroups: {report['subgroup_counts']}\n")
        header = f"{'metric':<28}{'original':>12}{'fine-tuned':>12}{'delta':>10}"
        print(header)
        print("-" * len(header))
        for label, block, key in (
            ("clean accuracy", "clean", "accuracy"),
            ("clean balanced accuracy", "clean", "balanced_accuracy"),
            ("clean F1", "clean", "f1"),
            ("clean recall", "clean", "recall"),
            ("clean AUROC", "clean", "auroc"),
            ("clean FPR", "clean", "fpr"),
            ("clean FNR", "clean", "fnr"),
        ):
            a = (results["original"][block] or {}).get(key)
            b = (results["fine_tuned"][block] or {}).get(key)
            if a is None or b is None:
                continue
            print(f"{label:<28}{a:>12.4f}{b:>12.4f}{b - a:>+10.4f}")

        for name in ("minor_edit_vs_authentic", "moderate_edit_vs_authentic", "synthetic_vs_authentic"):
            a = results["original"]["subgroups"].get(name)
            b = results["fine_tuned"]["subgroups"].get(name)
            if not isinstance(a, dict) or not isinstance(b, dict) or "f1" not in a:
                continue
            print(f"\n{name}:")
            for key in ("recall", "f1", "auroc", "fpr"):
                print(f"  {key:<26}{a[key]:>12.4f}{b[key]:>12.4f}{b[key] - a[key]:>+10.4f}")

        print(f"\nDECISION: {decision['verdict']}")
        for line in reasons:
            print(f"  - {line}")
        print(f"\nWrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
