#!/usr/bin/env python3
"""Ablate the transformation-consistency loss: none vs logit-MSE vs symmetric KL.

Three training runs differing only in the loss. Everything else -- data, split,
seed, epochs, augmentation -- is held fixed, so any difference is the loss term.

The retention rule is fixed before the runs: keep the consistency loss only if
it measurably improves worst-case transformed F1 or local-edit recall without
raising authentic false positives beyond the stated bar.

    python scripts/ablate_consistency_loss.py --config configs/lora_finetune_smoke.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.finetune.train_lora import train  # noqa: E402
from src.utils.config import load_config  # noqa: E402

MIN_GAIN = 0.01
MAX_FPR_INCREASE = 0.05

VARIANTS = {
    "classification_only": {"enabled": False, "weight": 0.0, "method": "mse"},
    "consistency_mse": {"enabled": True, "weight": 0.5, "method": "mse"},
    "consistency_kl": {"enabled": True, "weight": 0.5, "method": "kl"},
}

# Views must actually differ for a consistency term to mean anything, so the
# ablation supplies its own augmentation rather than relying on the config.
ABLATION_TRANSFORMS = ["jpeg_70", "blur_0.5", "resize_0.5x", "color_jitter"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lora_finetune_smoke.yaml")
    parser.add_argument("--output-dir", default="outputs/consistency_ablation")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--quiet", action="store_true")
    return parser


def _resolve(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    unknown = [name for name in args.variants if name not in VARIANTS]
    if unknown:
        print(f"Unknown variant(s): {', '.join(unknown)}", file=sys.stderr)
        return 1

    base = load_config(_resolve(args.config))
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}
    for name in args.variants:
        config = copy.deepcopy(base)
        config.setdefault("training", {})["consistency"] = dict(VARIANTS[name])
        config["training"]["official_transformations"] = list(ABLATION_TRANSFORMS)
        # Every variant sees two transformed views of each image, including the
        # classification-only baseline. Only the loss differs.
        config["training"]["views_per_image"] = 2
        # Each variant gets its own adapter, feature cache and results directory
        # so a cached tensor from one loss can never be reused by another.
        config.setdefault("model", {})["output_adapter"] = f"models/adapters/consistency_{name}"
        config.setdefault("outputs", {})["directory"] = f"outputs/consistency_ablation/{name}"
        config["outputs"]["feature_cache_dir"] = (
            f"outputs/consistency_ablation/{name}/feature_cache"
        )

        if not args.quiet:
            print(f"\n=== {name} ===")
        started = time.time()
        try:
            outcome = train(config, device=args.device)
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  FAILED: {exc}", file=sys.stderr)
            continue
        elapsed = time.time() - started

        test = outcome.get("test_metrics", {})
        validation = outcome.get("best_validation_metrics", {})
        results[name] = {
            "consistency": outcome.get("consistency"),
            "best_epoch": outcome.get("best_epoch"),
            "runtime_seconds": round(elapsed, 2),
            "validation": {
                k: v for k, v in validation.items() if isinstance(v, (int, float))
            },
            "test": {k: v for k, v in test.items() if isinstance(v, (int, float))},
            "test_subgroups": {
                key: value.get("metrics")
                for key, value in (test.get("subgroups") or {}).items()
            },
        }

    # --- pre-registered retention decision ---------------------------------
    baseline = results.get("classification_only", {})
    base_test = baseline.get("test", {}) if isinstance(baseline, dict) else {}
    decision: Dict[str, Any] = {
        "rule": (
            f"Keep the consistency loss only if F1 or recall improves by >= {MIN_GAIN} "
            f"with authentic FPR rising by <= {MAX_FPR_INCREASE}."
        ),
        "baseline": "classification_only",
        "candidates": {},
        "keep": None,
    }
    best_name = None
    best_gain = 0.0
    for name, entry in results.items():
        if name == "classification_only" or "error" in entry:
            continue
        test = entry.get("test", {})
        gains = {
            key: float(test.get(key, 0.0)) - float(base_test.get(key, 0.0))
            for key in ("f1", "recall", "auroc")
        }
        fpr_delta = float(test.get("fpr", 0.0)) - float(base_test.get("fpr", 0.0))
        passed = (
            max(gains["f1"], gains["recall"]) >= MIN_GAIN and fpr_delta <= MAX_FPR_INCREASE
        )
        decision["candidates"][name] = {
            "gains": {k: round(v, 6) for k, v in gains.items()},
            "fpr_delta": round(fpr_delta, 6),
            "passed": passed,
        }
        if passed and max(gains["f1"], gains["recall"]) > best_gain:
            best_gain = max(gains["f1"], gains["recall"])
            best_name = name
    decision["keep"] = best_name
    decision["verdict"] = (
        f"Adopt {best_name}."
        if best_name
        else "Keep the classification-only loss; no consistency variant met the bar."
    )

    report = {
        "config": str(_resolve(args.config)),
        "augmentation": ABLATION_TRANSFORMS,
        "variants": VARIANTS,
        "results": results,
        "decision": decision,
    }
    destination = output_dir / "consistency_ablation.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if not args.quiet:
        header = (
            f"\n{'variant':<24}{'F1':>8}{'recall':>8}{'FPR':>8}"
            f"{'AUROC':>8}{'bal_acc':>9}{'secs':>8}"
        )
        print(header)
        print("-" * len(header))
        for name, entry in results.items():
            if "error" in entry:
                print(f"{name:<24}  FAILED: {entry['error'][:40]}")
                continue
            t = entry["test"]
            print(
                f"{name:<24}{t.get('f1', float('nan')):>8.4f}{t.get('recall', float('nan')):>8.4f}"
                f"{t.get('fpr', float('nan')):>8.4f}{t.get('auroc', float('nan')):>8.4f}"
                f"{t.get('balanced_accuracy', float('nan')):>9.4f}"
                f"{entry['runtime_seconds']:>8.1f}"
            )
        print(f"\nDECISION: {decision['verdict']}")
        print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
