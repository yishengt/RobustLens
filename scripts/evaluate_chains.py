#!/usr/bin/env python3
"""Evaluate the detector under compound transformation chains.

Scores every image through fixed named chains and randomly ordered chains of
increasing depth, at ONE frozen threshold. Thresholds are never retuned per
chain -- that would measure the tuner, not the detector.

Scores are cached, so re-analysis (different abstention settings, different
operating point) costs nothing after the first run.

    python scripts/evaluate_chains.py --limit 60
    python scripts/evaluate_chains.py --reuse-scores
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

from src.evaluation.calibration import ProbabilityCalibrator  # noqa: E402
from src.evaluation.chains import (  # noqa: E402
    CLEAN_CHAIN,
    ChainRecord,
    build_generation_chains,
    build_named_chains,
    chain_names,
    generate_chain_variants,
)
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.pipeline.abstention import evaluate_abstention  # noqa: E402
from src.pipeline.consistency import consistency_score  # noqa: E402
from src.pipeline.model_loader import load_model  # noqa: E402
from src.pipeline.prediction import (  # noqa: E402
    LABEL_AI,
    LABEL_AUTHENTIC,
    predict_images,
)
from src.pipeline.preprocessing import Preprocessor  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/extracted/sid_set")
    parser.add_argument("--checkpoint", default="models/pretrained/pytorch_model.pt")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--calibration", default="outputs/calibration.json")
    parser.add_argument("--output-dir", default="outputs/chains")
    parser.add_argument("--limit", type=int, default=60, help="Total images to score")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--operating-point", default="balanced")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--reuse-scores",
        action="store_true",
        help="Re-analyse cached scores in the output directory without running the model",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def _resolve(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _collect_images(root: Path, limit: int) -> List[Dict[str, Any]]:
    """Balanced sample across the dataset's class folders, deterministic order."""

    labels_file = root / "labels.json"
    if not labels_file.is_file():
        raise SystemExit(
            f"No labels.json under {root}. Run scripts/extract_dataset.py first."
        )
    payload = json.loads(labels_file.read_text(encoding="utf-8"))
    items = payload.get("images", payload)
    by_class: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        by_class.setdefault(str(item["class_name"]), []).append(item)

    per_class = max(1, limit // max(1, len(by_class)))
    chosen: List[Dict[str, Any]] = []
    for name in sorted(by_class):
        entries = sorted(by_class[name], key=lambda i: str(i["img_id"]))[:per_class]
        chosen.extend(entries)
    return chosen[:limit]


def _score(args, chains) -> Dict[str, Any]:
    config = load_config(_resolve(args.config))
    bundle = load_model(_resolve(args.checkpoint), config, device=args.device)
    preprocessor = Preprocessor.from_config(config)
    root = _resolve(args.data_dir)
    entries = _collect_images(root, args.limit)

    records: List[ChainRecord] = []
    for index, entry in enumerate(entries, start=1):
        path = root / entry["image_path"]
        if not path.is_file():
            continue
        started = time.time()
        from PIL import Image

        with Image.open(path) as handle:
            image = handle.convert("RGB")
        variants, errors = generate_chain_variants(image, chains, seed=args.seed)
        names = list(variants)
        scores = predict_images(
            bundle, [variants[name] for name in names], preprocessor, batch_size=args.batch_size
        )
        records.append(
            ChainRecord(
                img_id=str(entry["img_id"]),
                binary_label=int(entry["label"]),
                class_name=str(entry["class_name"]),
                chain_scores={name: float(score) for name, score in zip(names, scores)},
                seconds=time.time() - started,
            )
        )
        if not args.quiet:
            print(f"[{index}/{len(entries)}] {entry['img_id']}  ({time.time()-started:.1f}s)")
        if errors:
            print(f"  chain errors: {errors}", file=sys.stderr)

    return {
        "records": [record.as_dict() for record in records],
        "model": bundle.summary(),
        "chains": [spec.as_dict() for spec in chains],
        "seed": args.seed,
    }


def _analyse(
    records: Sequence[ChainRecord],
    calibrator: Optional[ProbabilityCalibrator],
    threshold: float,
    abstention_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Per-chain metrics at one frozen threshold."""

    names = chain_names(records)
    labels = np.array([r.binary_label for r in records], dtype=int)
    clean_raw = np.array([r.clean_score for r in records], dtype=float)
    clean_cal = calibrator.transform(clean_raw) if calibrator else clean_raw

    rows: List[Dict[str, Any]] = []
    for name in names:
        raw = np.array([r.chain_scores[name] for r in records], dtype=float)
        calibrated = calibrator.transform(raw) if calibrator else raw
        metrics = compute_metrics(labels, calibrated, threshold=threshold)

        abstained = 0
        for index, record in enumerate(records):
            others = [record.chain_scores[n] for n in names if n != CLEAN_CHAIN]
            others_cal = calibrator.transform(np.array(others)) if calibrator else np.array(others)
            predicted_ai = calibrated[index] >= threshold
            every = [float(clean_cal[index]), *[float(v) for v in others_cal]]
            decision = evaluate_abstention(
                label=LABEL_AI if predicted_ai else LABEL_AUTHENTIC,
                final_probability=float(calibrated[index]),
                threshold=threshold,
                clean_probability=float(clean_cal[index]),
                transformed_probabilities=[float(v) for v in others_cal],
                consistency_score=consistency_score(every),
                agreement=float(
                    np.mean([(v >= threshold) == predicted_ai for v in every])
                ),
                patch_available=False,
                config={"abstention": abstention_config},
            )
            abstained += int(decision.abstain)

        drift = float(np.mean(calibrated - clean_cal))
        rows.append(
            {
                "chain": name,
                "score_drift_vs_clean": round(drift, 6),
                "score_drift_ai_images": round(
                    float(np.mean((calibrated - clean_cal)[labels == 1])), 6
                )
                if np.any(labels == 1)
                else None,
                "score_drift_authentic_images": round(
                    float(np.mean((calibrated - clean_cal)[labels == 0])), 6
                )
                if np.any(labels == 0)
                else None,
                "confidence_drift": round(
                    float(np.mean(np.abs(calibrated - 0.5) - np.abs(clean_cal - 0.5))), 6
                ),
                "abstention_rate": round(abstained / len(records), 6) if records else 0.0,
                "accuracy": round(metrics.accuracy, 6),
                "f1": round(metrics.f1, 6),
                "recall": round(metrics.recall, 6),
                "false_positive_rate": round(metrics.false_positive_rate, 6),
                "auc": round(metrics.auc, 6),
            }
        )

    transformed = [row for row in rows if row["chain"] != CLEAN_CHAIN]
    clean_row = next((row for row in rows if row["chain"] == CLEAN_CHAIN), None)
    worst = min(transformed, key=lambda r: r["f1"]) if transformed else None
    return {
        "threshold": round(float(threshold), 6),
        "threshold_note": "One fixed threshold. Never retuned per chain.",
        "images": len(records),
        "clean": clean_row,
        "per_chain": rows,
        "worst_case": worst,
        "mean_transformed_f1": (
            round(float(np.mean([r["f1"] for r in transformed])), 6) if transformed else None
        ),
        "largest_negative_drift": (
            min(transformed, key=lambda r: r["score_drift_vs_clean"]) if transformed else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "chain_scores.json"

    chains = build_named_chains() + build_generation_chains(args.seed)

    if args.reuse_scores:
        if not scores_path.is_file():
            print(f"No cached scores at {scores_path}", file=sys.stderr)
            return 1
        payload = json.loads(scores_path.read_text(encoding="utf-8"))
    else:
        payload = _score(args, chains)
        scores_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    records = [ChainRecord.from_dict(item) for item in payload["records"]]
    if not records:
        print("No images were scored.", file=sys.stderr)
        return 1

    calibration_path = _resolve(args.calibration)
    calibrator = None
    threshold = 0.5
    if calibration_path.is_file():
        calibrator = ProbabilityCalibrator.load(calibration_path)
        selected = calibrator.selected_thresholds or {}
        threshold = float(selected.get(args.operating_point, selected.get("balanced", 0.5)))

    config = load_config(_resolve(args.config))
    abstention_config = dict((config.get("abstention", {}) or {}))
    report = _analyse(records, calibrator, threshold, abstention_config)
    report["model"] = payload.get("model")
    report["chains"] = payload.get("chains")
    report["seed"] = payload.get("seed", args.seed)
    report["calibrated"] = calibrator is not None
    report["abstention_config"] = abstention_config

    destination = output_dir / "chain_metrics.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if not args.quiet:
        print(f"\n{len(records)} images   threshold {threshold}   (never retuned per chain)\n")
        header = (
            f"{'chain':<22}{'drift':>9}{'driftAI':>9}{'abst':>7}"
            f"{'acc':>7}{'F1':>7}{'recall':>8}{'FPR':>7}{'AUC':>7}"
        )
        print(header)
        print("-" * len(header))
        for row in report["per_chain"]:
            ai_drift = row["score_drift_ai_images"]
            print(
                f"{row['chain']:<22}{row['score_drift_vs_clean']:>+9.4f}"
                f"{(ai_drift if ai_drift is not None else float('nan')):>+9.4f}"
                f"{row['abstention_rate']:>7.3f}{row['accuracy']:>7.3f}{row['f1']:>7.3f}"
                f"{row['recall']:>8.3f}{row['false_positive_rate']:>7.3f}{row['auc']:>7.3f}"
            )
        worst = report["worst_case"]
        if worst:
            print(f"\nworst chain by F1: {worst['chain']}  F1={worst['f1']:.3f}")
        drift = report["largest_negative_drift"]
        if drift:
            print(
                f"largest downward drift: {drift['chain']} "
                f"{drift['score_drift_vs_clean']:+.4f}"
            )
        print(f"\nWrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
