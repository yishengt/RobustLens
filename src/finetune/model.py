"""Safe fine-tuning wrapper around the existing Bombek1 detector.

The current checkpoint is not a PEFT adapter-only export.  It is a monolithic
state dictionary containing full SigLIP2/DINOv2 weights, the existing LoRA
weights, and the classifier.  This module strictly restores that architecture,
then trains either the classifier only or the already-present LoRA tensors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from src.models.bombek_siglip2_dinov2 import (
    BOMBEK_ARCHITECTURE,
    align_checkpoint_keys,
    bombek_settings,
)
from src.pipeline.model_loader import (
    _extract_state_dict,
    _load_payload,
    build_architecture,
    detect_architecture,
    normalise_architecture,
)
from src.utils.device import device_from_config

LORA_TOKENS = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")


def parameter_counts(model: nn.Module) -> Dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def _is_lora_parameter(name: str) -> bool:
    return any(token in name for token in LORA_TOKENS)


def lora_parameter_names(model: nn.Module) -> List[str]:
    return [name for name, parameter in model.named_parameters() if _is_lora_parameter(name) and parameter.requires_grad]


def discovered_lora_modules(model: nn.Module) -> List[str]:
    """Return module paths verified from ``named_modules()``.

    The function intentionally inspects module objects rather than guessing
    from common transformer names.  It supports both PEFT modules and the
    vendored custom DINOv2 ``LoRALinear`` implementation.
    """

    names: List[str] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            names.append(name)
    return names


class FineTuneModel(nn.Module):
    """Loaded detector with explicit head-only and existing-LoRA modes."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        mode: str = "head_only",
        checkpoint_path: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.device = torch.device(device)
        self.mode = ""
        self.checkpoint_path = checkpoint_path
        self.model_config = dict(model_config or {})
        self.to(self.device)
        self.configure_trainable(mode)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str | torch.device] = None,
        mode: str = "head_only",
    ) -> "FineTuneModel":
        """Restore the exact existing model with a strict state-dict load."""

        config = config or {}
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        torch_device = (
            torch.device(device)
            if device is not None and str(device).lower() != "auto"
            else device_from_config(config)
        )
        payload = _load_payload(path, torch_device)
        state_dict = _extract_state_dict(payload)
        checkpoint_config = dict(payload.get("config", {}) or {}) if isinstance(payload, dict) else {}
        model_config = dict(config.get("model", {}) or {})
        settings = dict(model_config.get("bombek", {}) or {})
        settings.update(checkpoint_config)
        architecture = detect_architecture(state_dict) or model_config.get("name", BOMBEK_ARCHITECTURE)
        if normalise_architecture(str(architecture)) != BOMBEK_ARCHITECTURE:
            raise ValueError(
                "LoRA fine-tuning is implemented for the existing "
                f"{BOMBEK_ARCHITECTURE} checkpoint; detected '{architecture}'."
            )
        model = build_architecture(
            BOMBEK_ARCHITECTURE,
            num_classes=1,
            pretrained=False,
            dual_config=settings,
        )
        state_dict, _ = align_checkpoint_keys(state_dict, model)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                "The checkpoint cannot be fine-tuned directly because it does not "
                f"strictly match the recovered Bombek architecture: {exc}"
            ) from exc
        return cls(model, torch_device, mode, str(path), settings)

    def configure_trainable(self, mode: str) -> Dict[str, int]:
        mode = str(mode).strip().lower()
        if mode not in {"head_only", "lora"}:
            raise ValueError("Fine-tuning mode must be 'head_only' or 'lora'")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for name, parameter in self.model.named_parameters():
            if name.startswith("classifier.") or (mode == "lora" and _is_lora_parameter(name)):
                parameter.requires_grad_(True)
        if not any(parameter.requires_grad for parameter in self.model.classifier.parameters()):
            raise ValueError("The loaded model has no classifier head to fine-tune")
        if mode == "lora" and not any(_is_lora_parameter(name) for name, _ in self.model.named_parameters()):
            raise ValueError("The checkpoint contains no existing LoRA tensors; refusing to add a second adapter")
        self.mode = mode
        self.train(True)
        return parameter_counts(self.model)

    def train(self, mode: bool = True):  # type: ignore[override]
        """Train the head/adapters while keeping the original backbones frozen."""

        super().train(mode)
        for branch_name in ("siglip", "dinov2"):
            branch = getattr(self.model, branch_name, None)
            if branch is not None:
                branch.eval()
        classifier = getattr(self.model, "classifier", None)
        if classifier is not None:
            classifier.train(mode)
        # LoRA dropout is part of the trainable adapter path.  Re-enable only
        # those dropout modules after putting the frozen branches in eval mode.
        for name, module in self.model.named_modules():
            if "lora_dropout" in name:
                module.train(mode)
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                dropout = getattr(module, "dropout", None)
                if dropout is not None:
                    dropout.train(mode)
        return self

    def forward(self, siglip_pixels: torch.Tensor, dinov2_pixels: torch.Tensor) -> torch.Tensor:
        return self.model(siglip_pixels, dinov2_pixels)

    def forward_head_only(self, siglip_pixels: torch.Tensor, dinov2_pixels: torch.Tensor) -> torch.Tensor:
        """Use no-grad feature extraction in head-only mode to save memory."""

        if not hasattr(self.model, "forward_with_features"):
            return self.model(siglip_pixels, dinov2_pixels)
        with torch.no_grad():
            _, siglip_features, dinov2_features = self.model.forward_with_features(
                siglip_pixels, dinov2_pixels
            )
        return self.model.classifier(torch.cat([siglip_features.float(), dinov2_features.float()], dim=-1)).reshape(-1)

    def adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return only existing LoRA tensors, with no full-backbone weights."""

        return {
            name: value.detach().cpu().contiguous()
            for name, value in self.model.state_dict().items()
            if _is_lora_parameter(name)
        }

    def classifier_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            name[len("classifier.") :]: value.detach().cpu().contiguous()
            for name, value in self.model.state_dict().items()
            if name.startswith("classifier.")
        }

    def save_adapter(self, output_dir: str | Path, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Save a standalone adapter export and classifier, never the source checkpoint."""

        output = Path(output_dir).expanduser()
        output.mkdir(parents=True, exist_ok=True)
        adapter_path = output / "adapter_model.safetensors"
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise ImportError("Saving adapters requires safetensors; install requirements.txt") from exc
        save_file(self.adapter_state_dict(), str(adapter_path), metadata={"format": "robustlens-lora"})
        classifier_path = output / "classifier_head.pt"
        torch.save(self.classifier_state_dict(), classifier_path)
        settings = bombek_settings(self.model_config)
        module_names = discovered_lora_modules(self.model)
        suffixes = sorted({name.rsplit(".", 1)[-1] for name in module_names})
        adapter_config = {
            "peft_type": "LORA",
            "format": "robustlens_existing_lora",
            "base_model_names": {
                "siglip": settings["siglip_model"],
                "dinov2": settings["dinov2_model"],
            },
            "r": settings["lora_rank"],
            "lora_alpha": settings["lora_alpha"],
            "lora_dropout": settings["lora_dropout"],
            "bias": "none",
            "target_modules": suffixes,
            "target_module_names": module_names,
            "architecture": BOMBEK_ARCHITECTURE,
            "mode": self.mode,
            "parameter_counts": parameter_counts(self.model),
            **(metadata or {}),
        }
        with (output / "adapter_config.json").open("w", encoding="utf-8") as handle:
            json.dump(adapter_config, handle, indent=2, sort_keys=True)
        return {
            "adapter_config": str(output / "adapter_config.json"),
            "adapter_model": str(adapter_path),
            "classifier_head": str(classifier_path),
        }

    def load_saved_adapter(self, adapter_dir: str | Path) -> None:
        """Load this export onto a model restored from the original checkpoint."""

        directory = Path(adapter_dir).expanduser()
        from safetensors.torch import load_file

        adapter_state = load_file(str(directory / "adapter_model.safetensors"), device="cpu")
        model_keys = set(self.model.state_dict())
        expected_adapter_keys = {
            name for name in model_keys if _is_lora_parameter(name)
        }
        exported_keys = set(adapter_state)
        missing_adapter = sorted(expected_adapter_keys - exported_keys)
        unexpected = sorted(exported_keys - model_keys)
        non_adapter = sorted(name for name in exported_keys if not _is_lora_parameter(name))
        if unexpected or missing_adapter or non_adapter:
            raise ValueError(
                "Adapter export does not match this model: "
                f"unexpected={unexpected[:3]}, missing_adapter={missing_adapter[:3]}, "
                f"non_adapter={non_adapter[:3]}"
            )
        self.model.load_state_dict(adapter_state, strict=False)
        classifier_state = torch.load(
            directory / "classifier_head.pt", map_location="cpu", weights_only=True
        )
        self.model.classifier.load_state_dict(classifier_state, strict=True)


def load_saved_adapter_into_model(model: nn.Module, adapter_dir: str | Path) -> None:
    """Apply an adapter export to a detector already restored from the base checkpoint."""

    directory = Path(adapter_dir).expanduser()
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError("Loading adapters requires safetensors; install requirements.txt") from exc
    adapter_state = load_file(str(directory / "adapter_model.safetensors"), device="cpu")
    model_keys = set(model.state_dict())
    expected = {name for name in model_keys if _is_lora_parameter(name)}
    exported = set(adapter_state)
    missing = sorted(expected - exported)
    unexpected = sorted(exported - model_keys)
    non_adapter = sorted(name for name in exported if not _is_lora_parameter(name))
    if unexpected or missing or non_adapter:
        raise ValueError(
            "Adapter export does not match this model: "
            f"unexpected={unexpected[:3]}, missing_adapter={missing[:3]}, "
            f"non_adapter={non_adapter[:3]}"
        )
    model.load_state_dict(adapter_state, strict=False)
    classifier_state = torch.load(
        directory / "classifier_head.pt", map_location="cpu", weights_only=True
    )
    classifier = getattr(model, "classifier", None)
    if classifier is None:
        raise ValueError("Adapter target model has no classifier head")
    classifier.load_state_dict(classifier_state, strict=True)
