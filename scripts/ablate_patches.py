#!/usr/bin/env python3
"""Ablate patch-analysis modes: does patch scoring earn its cost?

For every image this computes the whole-image score once, then runs each patch
mode against that same score, recording cost (runtime, forward passes, peak
memory, coverage) and effect (patch evidence, fused score, and whether the
final decision actually changed).

The question it answers is deliberately narrow: **does patch analysis improve
F1, recall, false-positive rate or robustness enough to justify being a scoring
component?** If not, it belongs in the interface as an explainability feature.

Example::

    ./.venv/bin/python scripts/ablate_patches.py \\
      --checkpoint models/pretrained/pytorch_model.pt \\
      --limit 60 --per-class-limit 20 --device mps
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODES = ("off", "coarse", "full", "top_k", "uncertain_only")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare patch-analysis modes on cost and measurable benefit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="data/sid_set")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--checkpoint", default="models/pretrained/pytorch_model.pt")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output-dir", default="outputs/patch_ablation")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--per-class-limit", type=int, default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.42,
        help="Frozen decision threshold; never retuned per mode",
    )
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import numpy as np

    from src.evaluation.metrics import compute_metrics
    from src.evaluation.sid_set import find_shards, iter_labelled_images
    from src.pipeline.fusion import fuse_predictions
    from src.pipeline.model_loader import ModelSetupError, load_model
    from src.pipeline.patches import analyse_patches
    from src.pipeline.prediction import predict_images
    from src.pipeline.preprocessing import Preprocessor
    from src.utils.config import load_config
    from src.utils.device import describe_device

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    if args.batch_size:
        config.setdefault("inference", {})["batch_size"] = int(args.batch_size)

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    try:
        shards = find_shards(data_dir, args.split)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 4

    try:
        bundle = load_model(args.checkpoint, config, device=args.device)
    except ModelSetupError as exc:
        print(f"Model setup error: {exc}", file=sys.stderr)
        return 3
    preprocessor = Preprocessor.from_config(config)

    samples = list(
        iter_labelled_images(
            shards,
            limit=None if args.limit in (0, None) else args.limit,
            per_class_limit=args.per_class_limit,
        )
    )
    if not samples:
        print("No images loaded.", file=sys.stderr)
        return 4

    threshold = float(args.threshold)
    rows: list[dict] = []
    started = time.time()

    for index, sample in enumerate(samples, start=1):
        image = sample.image.convert("RGB")

        # One whole-image forward pass, shared by every mode.
        whole_started = time.time()
        whole = float(predict_images(bundle, [image], preprocessor, batch_size=1)[0])
        whole_seconds = time.time() - whole_started

        for mode in args.modes:
            mode_config = dict(config)
            mode_config["patches"] = {
                **(config.get("patches") or {}),
                "mode": mode,
                "enabled": mode != "off",
            }
            report = analyse_patches(
                bundle, image, preprocessor, mode_config, whole_image_probability=whole
            )
            evidence = report.evidence if report.available else None
            fused = float(
                fuse_predictions(whole, [], mode_config, patch_evidence=evidence).final_probability
            )
            rows.append(
                {
                    "img_id": sample.img_id,
                    "class_name": sample.class_name,
                    "binary_label": int(sample.binary_label),
                    "mode": mode,
                    "whole_score": round(whole, 6),
                    "patch_evidence": None if evidence is None else round(float(evidence), 6),
                    "fused_score": round(fused, 6),
                    "patch_available": report.available,
                    "num_patches": len(report.patches),
                    "forward_passes": report.forward_passes,
                    "reused_scores": report.reused_scores,
                    "coverage": (
                        None if report.coverage is None else round(float(report.coverage.mean()), 6)
                    ),
                    "patch_seconds": round(report.seconds, 4),
                    "whole_seconds": round(whole_seconds, 4),
                    "total_seconds": round(whole_seconds + report.seconds, 4),
                    "peak_memory_mb": report.peak_memory_mb,
                    "decision_whole": int(whole >= threshold),
                    "decision_fused": int(fused >= threshold),
                    "decision_changed": int((whole >= threshold) != (fused >= threshold)),
                }
            )

        if not args.quiet:
            rate = (time.time() - started) / index
            print(
                f"[{index}/{len(samples)}] {sample.img_id[:30]:32} "
                f"eta {(len(samples) - index) * rate / 60:5.1f} min",
                file=sys.stderr,
            )

    # ---------------- aggregate per mode ----------------------------------
    summary = {}
    baseline = None
    for mode in args.modes:
        subset = [row for row in rows if row["mode"] == mode]
        labels = [row["binary_label"] for row in subset]
        scores = [row["fused_score"] for row in subset]
        metrics = compute_metrics(labels, scores, threshold).as_dict()
        passes = np.array([row["forward_passes"] for row in subset], dtype=float)
        seconds = np.array([row["total_seconds"] for row in subset], dtype=float)
        coverage = [row["coverage"] for row in subset if row["coverage"] is not None]
        memory = [row["peak_memory_mb"] for row in subset if row["peak_memory_mb"]]
        summary[mode] = {
            "metrics": metrics,
            "images": len(subset),
            "patch_available": sum(1 for row in subset if row["patch_available"]),
            "mean_forward_passes": round(float(passes.mean()), 3),
            "total_forward_passes": int(passes.sum()),
            "mean_seconds_per_image": round(float(seconds.mean()), 4),
            "mean_coverage": round(float(np.mean(coverage)), 4) if coverage else None,
            "peak_memory_mb": round(float(np.max(memory)), 2) if memory else None,
            "decisions_changed": sum(row["decision_changed"] for row in subset),
            "reused_scores": int(sum(row["reused_scores"] for row in subset)),
        }
        if mode == "off":
            baseline = summary[mode]["metrics"]

    if baseline is not None:
        for payload in summary.values():
            metrics = payload["metrics"]
            payload["delta_vs_whole_image_only"] = {
                key: round(float(metrics[key] or 0) - float(baseline[key] or 0), 6)
                for key in (
                    "accuracy",
                    "balanced_accuracy",
                    "f1",
                    "recall",
                    "false_positive_rate",
                    "auc",
                )
            }

    verdict = build_verdict(summary)
    report = {
        "question": "Does patch analysis improve F1, recall, FPR or robustness enough to be a scoring component?",
        "model": bundle.summary(),
        "device": describe_device(bundle.device),
        "threshold": threshold,
        "threshold_note": "Frozen across every mode; never retuned per mode.",
        "images": len(samples),
        "patch_settings": config.get("patches"),
        "modes": summary,
        "verdict": verdict,
    }
    (output_dir / "ablation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    with (output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "modes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "mode",
                "accuracy",
                "balanced_accuracy",
                "f1",
                "recall",
                "fpr",
                "auc",
                "mean_forward_passes",
                "mean_seconds",
                "mean_coverage",
                "decisions_changed",
                "peak_memory_mb",
            ]
        )
        for mode, payload in summary.items():
            m = payload["metrics"]
            writer.writerow(
                [
                    mode,
                    m["accuracy"],
                    m["balanced_accuracy"],
                    m["f1"],
                    m["recall"],
                    m["false_positive_rate"],
                    m["auc"],
                    payload["mean_forward_passes"],
                    payload["mean_seconds_per_image"],
                    payload["mean_coverage"],
                    payload["decisions_changed"],
                    payload["peak_memory_mb"],
                ]
            )

    print_report(report)
    return 0


def build_verdict(summary: dict) -> dict:
    """Decide, from the numbers alone, whether patches should score or explain."""

    baseline = summary.get("off", {}).get("metrics")
    if not baseline:
        return {"conclusion": "inconclusive", "reason": "no whole-image-only baseline was run"}

    improved = []
    for mode, payload in summary.items():
        if mode == "off":
            continue
        delta = payload.get("delta_vs_whole_image_only", {})
        # A mode earns scoring status only if it improves a headline metric
        # without making false positives worse.
        if (delta.get("f1", 0) > 0.005 or delta.get("recall", 0) > 0.005) and delta.get(
            "false_positive_rate", 0
        ) <= 0.0:
            improved.append(mode)

    if improved:
        return {
            "conclusion": "keep_as_scoring_component",
            "modes_that_improved": improved,
            "reason": "at least one mode improved F1 or recall without raising the false-positive rate",
        }
    return {
        "conclusion": "keep_as_explainability_only",
        "modes_that_improved": [],
        "reason": (
            "no patch mode improved F1 or recall without raising the false-positive rate, "
            "so patch evidence does not earn a weight in fusion. Keep patch analysis as an "
            "optional explainability feature: set fusion.mode to rgb_transform and leave "
            "patches enabled for the heatmap only."
        ),
    }


def print_report(report: dict) -> None:
    line = "=" * 96
    print(line)
    print("PATCH-ANALYSIS ABLATION")
    print(line)
    print(f"Model     : {report['model']['architecture']} on {report['device']}")
    print(f"Images    : {report['images']}   threshold {report['threshold']} (frozen across modes)")
    print()
    header = (
        f"  {'mode':<16}{'acc':>7}{'bal':>7}{'F1':>7}{'recall':>8}{'FPR':>7}{'AUC':>7}"
        f"{'fwd/img':>9}{'s/img':>8}{'cover':>8}{'changed':>9}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for mode, payload in report["modes"].items():
        m = payload["metrics"]
        cover = f"{payload['mean_coverage']:.1%}" if payload["mean_coverage"] else "-"
        print(
            f"  {mode:<16}{m['accuracy']:>7.3f}{m['balanced_accuracy']:>7.3f}{m['f1']:>7.3f}"
            f"{m['recall']:>8.3f}{m['false_positive_rate']:>7.3f}{(m['auc'] or 0):>7.3f}"
            f"{payload['mean_forward_passes']:>9.1f}{payload['mean_seconds_per_image']:>8.2f}"
            f"{cover:>8}{payload['decisions_changed']:>9}"
        )
    print()
    verdict = report["verdict"]
    print(f"VERDICT: {verdict['conclusion']}")
    print(f"  {verdict['reason']}")
    print(line)


if __name__ == "__main__":
    raise SystemExit(main())
