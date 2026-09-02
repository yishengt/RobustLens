#!/usr/bin/env python3
"""Score a folder of images, and grade the run when the labels are known.

Built for volume. A long run is resumable: every image is appended to a JSONL
cache as it finishes, so an interrupted run picks up where it stopped instead of
starting over, and re-grading at a different threshold costs no forward passes
at all.

Labels are optional and are never invented. Put images in class subfolders::

    my_images/
      real/       ← authentic
      ai/         ← AI-generated

and the run is graded. Point at a flat folder instead and you get predictions
with no accuracy claim attached.

    python scripts/batch_test.py --input-dir my_images
    python scripts/batch_test.py --input-dir my_images --fast
    python scripts/batch_test.py --input-dir my_images --resume
    python scripts/batch_test.py --regrade --threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.pipeline.model_loader import ModelSetupError  # noqa: E402
from src.pipeline.pipeline import DetectionPipeline  # noqa: E402
from src.pipeline.validation import list_supported_images  # noqa: E402
from src.utils.config import load_config  # noqa: E402

# Folder names understood as labels. Anything else leaves an image unlabelled
# rather than guessing.
AUTHENTIC_NAMES = {"real", "authentic", "camera", "genuine", "0", "negative"}
AI_NAMES = {
    "ai", "fake", "synthetic", "generated", "ai_generated", "full_synthetic",
    "tampered", "edited", "ai_edited", "1", "positive",
}

# Measured on Apple MPS with the 740M-parameter checkpoint.
SECONDS_PER_IMAGE_FULL = 7.5
SECONDS_PER_IMAGE_FAST = 1.0
STARTUP_SECONDS = 15.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-dir", help="Folder of images (class subfolders are read as labels)")
    parser.add_argument("--checkpoint", default="models/pretrained/pytorch_model.pt")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output-dir", default="outputs/batch_test")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Score the original only, skipping the 14 transformations (~7x faster)",
    )
    parser.add_argument(
        "--patches", action="store_true", help="Also run region analysis (much slower)"
    )
    parser.add_argument("--threshold", type=float, help="Override the decision threshold")
    parser.add_argument("--limit", type=int, help="Stop after N images")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images already present in the cache and continue",
    )
    parser.add_argument(
        "--regrade",
        action="store_true",
        help="Re-grade the cached scores without running the model",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def _resolve(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def effective_threshold(config: Dict[str, Any], override: Optional[float] = None) -> float:
    """The threshold the pipeline will actually decide with.

    DetectionPipeline deep-copies the config and swaps in the calibrated
    operating point, so reading inference.threshold from the caller's dict gives
    the pre-calibration default instead -- grading at 0.5 while the pipeline
    labels at 0.69. This reproduces the pipeline's choice without loading the
    model, so --regrade agrees with a live run.
    """

    if override is not None:
        return float(override)

    calibration = config.get("calibration", {}) or {}
    if calibration.get("enabled", False) and calibration.get("use_selected_threshold", True):
        path = calibration.get("path")
        if path:
            source = _resolve(str(path))
            try:
                from src.evaluation.calibration import ProbabilityCalibrator

                calibrator = ProbabilityCalibrator.load(source)
            except (FileNotFoundError, ValueError, ImportError):
                calibrator = None
            if calibrator is not None and calibrator.selected_thresholds:
                point = str(calibration.get("operating_point", "balanced"))
                selected = calibrator.selected_thresholds.get(
                    point, calibrator.selected_thresholds.get("balanced")
                )
                if selected is not None:
                    return float(selected)
    return float(config.get("inference", {}).get("threshold", 0.5))


def label_for(path: Path, root: Path) -> Optional[int]:
    """Read a label from the folder names above ``path``, or None.

    Checks every folder between the image and the root, so both
    ``real/x.jpg`` and ``batch3/real/x.jpg`` are understood.
    """

    try:
        parts = [part.lower() for part in path.relative_to(root).parts[:-1]]
    except ValueError:
        return None
    for part in reversed(parts):
        if part in AUTHENTIC_NAMES:
            return 0
        if part in AI_NAMES:
            return 1
    return None


def load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    """Read the JSONL cache, tolerating a truncated final line."""

    if not path.is_file():
        return {}
    records: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # A run killed mid-write leaves one partial line. Skip it rather
            # than refusing to resume.
            continue
        records[record["image_path"]] = record
    return records


def grade(records: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    """Metrics when the labels support them, and an honest refusal when not."""

    scored = [r for r in records if r.get("pred") is not None]
    labelled = [r for r in scored if r.get("label") is not None]
    labels = [int(r["label"]) for r in labelled]
    scores = [float(r["pred"]) for r in labelled]

    summary: Dict[str, Any] = {
        "images_seen": len(records),
        "scored": len(scored),
        "failed": len(records) - len(scored),
        "labelled": len(labelled),
        "unlabelled": len(scored) - len(labelled),
        "threshold": round(float(threshold), 6),
    }

    if not labelled:
        summary["metrics"] = None
        summary["note"] = (
            "No labels were found, so no accuracy is reported. Put images in "
            "class subfolders (real/ and ai/) to have the run graded."
        )
        return summary
    if len(set(labels)) < 2:
        only = "authentic" if labels[0] == 0 else "AI-generated"
        summary["metrics"] = None
        summary["note"] = (
            f"Every labelled image is {only}, so precision, recall and AUROC are "
            f"undefined. Only the flagged count below is meaningful."
        )
        summary["flagged"] = sum(1 for s in scores if s >= threshold)
        return summary

    m = compute_metrics(labels, scores, threshold=threshold)
    summary["metrics"] = {
        "accuracy": round(m.accuracy, 6),
        "balanced_accuracy": round(m.balanced_accuracy, 6),
        "precision": round(m.precision, 6),
        "recall": round(m.recall, 6),
        "f1": round(m.f1, 6),
        "auroc": round(m.auc, 6),
        "false_positive_rate": round(m.false_positive_rate, 6),
        "false_negative_rate": round(m.false_negative_rate, 6),
        "confusion_matrix": {
            "true_positives": m.true_positives,
            "false_positives": m.false_positives,
            "true_negatives": m.true_negatives,
            "false_negatives": m.false_negatives,
        },
    }
    wrong = [
        r
        for r in labelled
        if (float(r["pred"]) >= threshold) != bool(int(r["label"]))
    ]
    summary["errors"] = sorted(
        (
            {
                "image_path": r["image_path"],
                "label": "AI-generated" if int(r["label"]) else "authentic",
                "pred": round(float(r["pred"]), 4),
                "kind": "false negative" if int(r["label"]) else "false positive",
            }
            for r in wrong
        ),
        key=lambda item: abs(item["pred"] - threshold),
        reverse=True,
    )
    return summary


def _print_report(summary: Dict[str, Any], elapsed: Optional[float]) -> None:
    print(f"\n{'=' * 60}")
    print(
        f"{summary['scored']} scored · {summary['failed']} failed · "
        f"{summary['labelled']} labelled · {summary['unlabelled']} unlabelled"
    )
    if elapsed:
        rate = summary["scored"] / elapsed if elapsed else 0
        print(f"{elapsed / 60:.1f} min  ({rate * 60:.1f} images/min)")
    print(f"threshold {summary['threshold']}")

    metrics = summary.get("metrics")
    if metrics is None:
        print(f"\n{summary.get('note', '')}")
        if "flagged" in summary:
            print(f"Flagged as AI-generated: {summary['flagged']}/{summary['labelled']}")
        return

    print()
    for key in (
        "accuracy", "balanced_accuracy", "precision", "recall", "f1", "auroc",
        "false_positive_rate", "false_negative_rate",
    ):
        print(f"  {key.replace('_', ' '):<24}{metrics[key]:.4f}")

    cm = metrics["confusion_matrix"]
    print(
        f"\n  {'':>14}{'called AI':>12}{'called real':>13}\n"
        f"  {'is AI':>14}{cm['true_positives']:>12}{cm['false_negatives']:>13}\n"
        f"  {'is real':>14}{cm['false_positives']:>12}{cm['true_negatives']:>13}"
    )

    errors = summary.get("errors") or []
    if errors:
        print(f"\nMost confident mistakes ({len(errors)} total):")
        for item in errors[:10]:
            print(
                f"  {item['pred']:.3f}  {item['kind']:<15} "
                f"{Path(item['image_path']).name[:44]}"
            )


def _collect(root: Path, config: Dict[str, Any], limit: Optional[int]) -> List[Tuple[Path, Optional[int]]]:
    paths = list_supported_images(root, config)
    if limit:
        paths = paths[:limit]
    return [(path, label_for(path, root)) for path in paths]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "scores.jsonl"

    config = load_config(_resolve(args.config))

    # --- regrade only: no model, no forward passes ------------------------
    if args.regrade:
        cached = load_cache(cache_path)
        if not cached:
            print(f"No cached scores at {cache_path}", file=sys.stderr)
            return 1
        threshold = effective_threshold(config, args.threshold)
        summary = grade(list(cached.values()), threshold)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        if not args.quiet:
            _print_report(summary, None)
            print(f"\nRegraded {len(cached)} cached scores. No model was run.")
        return 0

    if not args.input_dir:
        print("--input-dir is required (or use --regrade)", file=sys.stderr)
        return 1

    root = _resolve(args.input_dir)
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    if args.fast:
        config.setdefault("transformations", {})["enabled"] = False
    config.setdefault("patches", {})["enabled"] = bool(args.patches)
    if args.threshold is not None:
        config.setdefault("inference", {})["threshold"] = float(args.threshold)

    try:
        items = _collect(root, config, args.limit)
    except Exception as exc:  # noqa: BLE001 - surface any listing failure plainly
        print(f"Could not list images: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if not items:
        print(f"No supported images found under {root}", file=sys.stderr)
        return 1

    cached = load_cache(cache_path) if args.resume else {}
    todo = [(p, lab) for p, lab in items if str(p) not in cached]

    per_image = SECONDS_PER_IMAGE_FAST if args.fast else SECONDS_PER_IMAGE_FULL
    if args.patches:
        per_image += 8.0
    estimate = STARTUP_SECONDS + len(todo) * per_image
    labelled = sum(1 for _, lab in items if lab is not None)
    if not args.quiet:
        print(f"{len(items)} images · {labelled} labelled · {len(todo)} to score")
        if cached:
            print(f"{len(cached)} already cached, skipping (--resume)")
        print(f"estimated {estimate / 60:.1f} min at ~{per_image:.1f}s/image")
        if not args.fast:
            print("use --fast to skip the 14 transformations (~7x faster)")
        print()

    try:
        pipeline = DetectionPipeline.from_checkpoint(
            _resolve(args.checkpoint), config, device=args.device, explain_images=False
        )
    except ModelSetupError as exc:
        print(f"Model not available: {exc}", file=sys.stderr)
        return 1

    # Take it from the pipeline itself, which has already applied calibration.
    threshold = float(
        pipeline.calibration_status().get("threshold")
        or effective_threshold(config, args.threshold)
    )
    started = time.time()
    # Append as we go: a run killed at image 900 of 1000 keeps its 900 scores.
    with cache_path.open("a", encoding="utf-8") as handle:
        for index, (path, label) in enumerate(todo, start=1):
            outcome = pipeline.safe_analyse_path(path)
            errors = getattr(outcome, "errors", []) or []
            record = {
                "image_path": str(path),
                "label": label,
                "pred": getattr(outcome, "ai_probability", None),
                "predicted_label": getattr(outcome, "label", None),
                "confidence": getattr(getattr(outcome, "confidence", None), "level", None),
                "abstained": bool(
                    getattr(getattr(outcome, "abstention", None), "abstain", False)
                ),
                "errors": errors,
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            cached[str(path)] = record

            if not args.quiet:
                done = index / len(todo)
                remaining = (time.time() - started) / index * (len(todo) - index)
                score = record["pred"]
                shown = f"{score:.3f}" if score is not None else "FAILED"
                print(
                    f"[{index}/{len(todo)}] {done:5.1%}  {shown:>7}  "
                    f"eta {remaining / 60:4.1f}m  {path.name[:40]}"
                )

    elapsed = time.time() - started
    records = [cached[str(p)] for p, _ in items if str(p) in cached]
    summary = grade(records, threshold)
    summary["elapsed_seconds"] = round(elapsed, 1)
    summary["input_dir"] = str(root)
    summary["fast_mode"] = bool(args.fast)
    summary["patches"] = bool(args.patches)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    simple = [
        {"image_path": r["image_path"], "pred": round(float(r["pred"]), 6)}
        for r in records
        if r.get("pred") is not None
    ]
    (output_dir / "predictions.json").write_text(
        json.dumps(simple, indent=2) + "\n", encoding="utf-8"
    )

    try:
        import pandas as pd

        pd.DataFrame(records).to_csv(output_dir / "results.csv", index=False)
    except ImportError:  # pragma: no cover - pandas ships in requirements
        pass

    if not args.quiet:
        _print_report(summary, elapsed)
        print(f"\nWrote {output_dir}/summary.json, predictions.json, results.csv")
        print("Re-grade at another threshold without rerunning the model:")
        print("  python scripts/batch_test.py --regrade --threshold 0.5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
