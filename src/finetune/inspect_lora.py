"""Inspect a RobustLens checkpoint without modifying or replacing it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from src.models.bombek_siglip2_dinov2 import bombek_settings
from src.pipeline.model_loader import (
    _extract_state_dict,
    _load_payload,
    build_architecture,
    detect_architecture,
    normalise_architecture,
)

LORA_MARKERS = ("lora_A", "lora_B", "lora_dropout", "lora_embedding_A", "lora_embedding_B")


def _lora_type(keys: Iterable[str]) -> str:
    key_list = list(keys)
    lowered = " ".join(key_list).lower()
    if "lora_magnitude_vector" in lowered or "dora" in lowered:
        return "DoRA"
    if "lora_e" in lowered or "adalora" in lowered:
        return "AdaLoRA"
    has_a = any("lora_a" in key.lower() or "lora_embedding_a" in key.lower() for key in key_list)
    has_b = any("lora_b" in key.lower() or "lora_embedding_b" in key.lower() for key in key_list)
    if has_a and has_b:
        return "standard LoRA"
    return "unknown"


def _module_name_from_key(key: str) -> Optional[str]:
    for marker in (".lora_A", ".lora_B", ".lora_embedding_A", ".lora_embedding_B"):
        if marker in key:
            return key.split(marker, 1)[0]
    return None


def _rank_and_shapes(state_dict: Dict[str, torch.Tensor]) -> Tuple[Optional[int], Dict[str, List[int]]]:
    ranks: List[int] = []
    shapes: Dict[str, List[int]] = {}
    for key, tensor in state_dict.items():
        if "lora_A" in key or "lora_embedding_A" in key:
            if tensor.ndim >= 2:
                ranks.append(int(tensor.shape[0]))
                shapes[key] = list(tensor.shape)
        elif "lora_B" in key or "lora_embedding_B" in key:
            shapes[key] = list(tensor.shape)
    return (ranks[0] if ranks and len(set(ranks)) == 1 else None), shapes


def _classifier_structure(model: Any) -> List[Dict[str, Any]]:
    classifier = getattr(model, "classifier", None)
    if classifier is None:
        return []
    return [
        {
            "name": name or "classifier",
            "type": module.__class__.__name__,
            "parameters": sum(parameter.numel() for parameter in module.parameters(recurse=False)),
        }
        for name, module in classifier.named_modules()
    ]


def inspect_checkpoint(
    checkpoint_path: str | Path,
    config: Optional[Dict[str, Any]] = None,
    build_model_for_modules: bool = True,
) -> Dict[str, Any]:
    """Return a serialisable report of checkpoint format and LoRA details."""

    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = _load_payload(path, torch.device("cpu"))
    state_dict = _extract_state_dict(payload)
    payload_config = dict(payload.get("config", {}) or {}) if isinstance(payload, dict) else {}
    model_config = dict((config or {}).get("model", {}) or {})
    settings = dict(model_config.get("bombek", {}) or {})
    settings.update(payload_config)
    lora_keys = [key for key in state_dict if any(marker in key for marker in LORA_MARKERS)]
    module_names_from_keys = sorted({name for key in lora_keys if (name := _module_name_from_key(key))})
    peft_used = any("base_model" in key or ".default." in key for key in lora_keys)
    separate_adapter = bool(
        isinstance(payload, dict)
        and any(key in payload for key in ("adapter_config", "adapter_model", "peft_config"))
    )
    rank, lora_shapes = _rank_and_shapes(state_dict)

    model = None
    model_error = None
    named_module_names: List[str] = []
    classifier_structure: List[Dict[str, Any]] = []
    parameter_report = {"total": None, "trainable": None, "frozen": None}
    can_finetune_directly = False
    architecture = detect_architecture(state_dict) or model_config.get("name")
    if build_model_for_modules and architecture:
        try:
            canonical = normalise_architecture(str(architecture))
            if canonical == "bombek_siglip2_dinov2":
                model = build_architecture(canonical, num_classes=1, pretrained=False, dual_config=settings)
                named_module_names = [name for name, _ in model.named_modules()]
                classifier_structure = _classifier_structure(model)
                total = sum(parameter.numel() for parameter in model.parameters())
                trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
                parameter_report = {"total": total, "trainable": trainable, "frozen": total - trainable}
                can_finetune_directly = set(state_dict) == set(model.state_dict())
        except Exception as exc:  # inspection should still report tensor evidence
            model_error = f"{type(exc).__name__}: {exc}"

    verified_targets = [name for name in module_names_from_keys if name in named_module_names]
    report = {
        "checkpoint": str(path),
        "checkpoint_format": "separate_adapter" if separate_adapter else "monolithic_state_dict",
        "state_dict_tensor_count": len(state_dict),
        "architecture": architecture,
        "lora_tensors_exist": bool(lora_keys),
        "lora_tensor_count": len(lora_keys),
        "peft_used": peft_used,
        "lora_type": _lora_type(lora_keys),
        "lora_rank": rank,
        "lora_alpha": settings.get("lora_alpha", settings.get("alpha")),
        "lora_dropout": settings.get("lora_dropout"),
        "target_modules": verified_targets or module_names_from_keys,
        "target_modules_verified_with_named_modules": bool(verified_targets) and len(verified_targets) == len(module_names_from_keys),
        "target_modules_from_state_dict": module_names_from_keys,
        "base_model_names": {
            "siglip": settings.get("siglip_model", settings.get("siglip_name", bombek_settings(settings)["siglip_model"])),
            "dinov2": settings.get("dinov2_model", settings.get("dinov2_name", bombek_settings(settings)["dinov2_model"])),
        },
        "trainable_parameter_count": parameter_report["trainable"],
        "frozen_parameter_count": parameter_report["frozen"],
        "total_parameter_count": parameter_report["total"],
        "under_2b_parameter_limit": parameter_report["total"] is not None and parameter_report["total"] < 2_000_000_000,
        "classifier_head_structure": classifier_structure,
        "can_finetune_directly": can_finetune_directly,
        "limitation": (
            "The checkpoint is monolithic rather than a standalone PEFT adapter; "
            "restore it with the recovered architecture, then export a separate adapter."
            if not separate_adapter
            else None
        ),
        "model_module_inspection_error": model_error,
        "lora_tensor_shapes": lora_shapes,
    }
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", default="models/pretrained/pytorch_model.pt")
    parser.add_argument("--no-build-model", action="store_true", help="Only inspect state-dict evidence")
    parser.add_argument("--output", type=Path, help="Write JSON report to this path")
    args = parser.parse_args(argv)
    report = inspect_checkpoint(args.checkpoint, build_model_for_modules=not args.no_build_model)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
