"""Stage 5: checkpoint loading.

The pipeline is inference-only. A real trained checkpoint is required; when one
is missing or unusable this module raises :class:`ModelSetupError` with setup
instructions rather than returning an untrained model that would emit
meaningless predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models

from src.utils.device import describe_device, device_from_config

# Architectures small enough for a hackathon and far below the 2B parameter cap.
SUPPORTED_ARCHITECTURES: Tuple[str, ...] = ("efficientnet_b0", "resnet18", "convnext_tiny")
DUAL_BACKBONE_ARCHITECTURE = "dual_backbone"
BOMBEK_ARCHITECTURE = "bombek_siglip2_dinov2"
ALL_ARCHITECTURES: Tuple[str, ...] = SUPPORTED_ARCHITECTURES + (
    DUAL_BACKBONE_ARCHITECTURE,
    BOMBEK_ARCHITECTURE,
)

# Architectures that take two separately preprocessed pixel tensors.
DUAL_INPUT_ARCHITECTURES: Tuple[str, ...] = (DUAL_BACKBONE_ARCHITECTURE, BOMBEK_ARCHITECTURE)
MAX_PARAMETERS = 2_000_000_000

SETUP_HINT = (
    "This project does not train models. Place a trained checkpoint at the "
    "configured path (default: checkpoints/best.pt) or pass --checkpoint. "
    "See models/README.md for the expected checkpoint format."
)


class ModelSetupError(RuntimeError):
    """Raised when no usable model checkpoint could be loaded."""


@dataclass
class ModelBundle:
    """A loaded model together with everything inference needs to know."""

    model: nn.Module
    device: torch.device
    architecture: str
    num_classes: int
    num_parameters: int
    checkpoint_path: str
    ai_class_index: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    input_kind: str = "single"
    processors: Optional[Tuple[Any, Any]] = None

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serialisable description for logs and the demo UI."""

        return {
            "architecture": self.architecture,
            "num_classes": self.num_classes,
            "num_parameters": self.num_parameters,
            "parameters_millions": round(self.num_parameters / 1e6, 2),
            "under_2b_parameter_limit": self.num_parameters < MAX_PARAMETERS,
            "checkpoint_path": self.checkpoint_path,
            "device": describe_device(self.device),
            "input_kind": self.input_kind,
        }


def normalise_architecture(name: str) -> str:
    """Normalise user-facing architecture spellings to the canonical name."""

    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "efficientnet_b0": "efficientnet_b0",
        "efficientnetb0": "efficientnet_b0",
        "resnet18": "resnet18",
        "resnet_18": "resnet18",
        "convnext_tiny": "convnext_tiny",
        "convnexttiny": "convnext_tiny",
        "dual_backbone": DUAL_BACKBONE_ARCHITECTURE,
        "dual": DUAL_BACKBONE_ARCHITECTURE,
        "siglip2_dinov2": DUAL_BACKBONE_ARCHITECTURE,
        # The external Bombek1 LoRA detector is a distinct architecture; it is
        # deliberately NOT an alias of dual_backbone.
        "bombek_siglip2_dinov2": BOMBEK_ARCHITECTURE,
        "bombek": BOMBEK_ARCHITECTURE,
        "ensembleaidetector": BOMBEK_ARCHITECTURE,
    }
    if key not in aliases:
        raise ModelSetupError(
            f"Unsupported model architecture '{name}'. "
            f"Supported architectures: {', '.join(ALL_ARCHITECTURES)}."
        )
    return aliases[key]


def build_architecture(
    name: str = "efficientnet_b0",
    num_classes: int = 1,
    pretrained: bool = False,
    dual_config: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Create a lightweight backbone with a fresh binary classification head.

    ``num_classes=1`` gives a single logit read through a sigmoid;
    ``num_classes=2`` gives a softmax pair. Both checkpoint styles are common,
    so both are supported.
    """

    architecture = normalise_architecture(name)
    num_classes = int(num_classes)
    if architecture == BOMBEK_ARCHITECTURE:
        if num_classes != 1:
            raise ModelSetupError(
                "bombek_siglip2_dinov2 has a single binary output logit; "
                f"got num_classes={num_classes}"
            )
        from src.models.bombek_siglip2_dinov2 import build_bombek_detector

        return build_bombek_detector(dual_config or {}, pretrained_backbones=bool(pretrained))
    if architecture == DUAL_BACKBONE_ARCHITECTURE:
        if num_classes != 1:
            raise ModelSetupError("dual_backbone currently supports one binary output logit only")
        from src.models.dual_backbone import DualBackboneDetector

        settings = dual_config or {}
        return DualBackboneDetector(
            siglip_name=str(settings.get("siglip_name", "google/siglip2-so400m-patch14-384")),
            dinov2_name=str(settings.get("dinov2_name", "facebook/dinov2-large")),
            hidden_dim=int(settings.get("hidden_dim", 3584)),
            dropout=float(settings.get("dropout", 0.2)),
            pretrained=bool(pretrained),
        )
    if num_classes not in (1, 2):
        raise ModelSetupError(
            f"model.num_classes must be 1 (sigmoid) or 2 (softmax), got {num_classes}"
        )

    if architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:  # convnext_tiny
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    return model


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters, used to prove the 2B-parameter limit is met."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad or not trainable_only
    )


def _load_payload(path: Path, device: torch.device) -> Any:
    """Load a checkpoint file, tolerating both weights_only defaults."""

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only argument
        return torch.load(path, map_location=device)


def _extract_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    """Pull the tensor state dict out of the common checkpoint layouts."""

    candidate: Any = payload
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model", "net", "weights"):
            value = payload.get(key)
            if isinstance(value, dict) and value:
                candidate = value
                break

    if isinstance(candidate, nn.Module):  # a whole pickled model
        candidate = candidate.state_dict()

    if not isinstance(candidate, dict) or not candidate:
        raise ModelSetupError(
            "Checkpoint does not contain a state dict. Expected either a raw "
            "state_dict or a dict with a 'model_state_dict'/'state_dict' key."
        )
    if not all(isinstance(value, torch.Tensor) for value in candidate.values()):
        raise ModelSetupError(
            "Checkpoint state dict contains non-tensor values; it does not look "
            "like a model checkpoint."
        )

    # Strip wrapper prefixes left behind by DataParallel / Lightning, but only
    # when every key carries the prefix. Stripping per-key would corrupt
    # architectures with a legitimate top-level "model." submodule.
    cleaned: Dict[str, torch.Tensor] = {str(k): v for k, v in candidate.items()}
    for prefix in ("module.", "model."):
        if cleaned and all(key.startswith(prefix) for key in cleaned):
            cleaned = {key[len(prefix) :]: value for key, value in cleaned.items()}
    return cleaned


def infer_num_classes(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    """Infer the head width from the last 1-D bias or 2-D weight tensor."""

    for key in reversed(list(state_dict)):
        tensor = state_dict[key]
        if key.endswith("bias") and tensor.dim() == 1 and tensor.numel() in (1, 2):
            return int(tensor.numel())
        if key.endswith("weight") and tensor.dim() == 2 and tensor.shape[0] in (1, 2):
            return int(tensor.shape[0])
    return None


def detect_architecture(state_dict: Dict[str, torch.Tensor]) -> Optional[str]:
    """Identify an architecture from its state-dict key signature.

    Returns ``None`` when the keys are not distinctive, leaving the configured
    or recorded architecture in charge. Only signatures that cannot collide are
    reported here -- the point is to make a correct strict load possible, never
    to guess.
    """

    from src.models.bombek_siglip2_dinov2 import looks_like_bombek_state_dict

    if looks_like_bombek_state_dict(state_dict):
        return BOMBEK_ARCHITECTURE
    return None


def load_model(
    checkpoint_path: str | Path,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[str | torch.device] = None,
) -> ModelBundle:
    """Load a checkpoint into a lightweight backbone and return a ready bundle.

    Raises :class:`ModelSetupError` with an actionable message when the file is
    missing, unreadable, or does not match the configured architecture.
    """

    config = config or {}
    model_config = config.get("model", {}) or {}

    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        raise ModelSetupError(f"Model checkpoint not found: {path}\n{SETUP_HINT}")
    if not path.is_file():
        raise ModelSetupError(f"Model checkpoint path is not a file: {path}\n{SETUP_HINT}")
    if path.stat().st_size == 0:
        raise ModelSetupError(f"Model checkpoint is empty (0 bytes): {path}\n{SETUP_HINT}")

    torch_device = torch.device(device) if device is not None else device_from_config(config)

    try:
        payload = _load_payload(path, torch_device)
    except ModelSetupError:
        raise
    except Exception as exc:  # torch raises a wide range of unpickling errors
        raise ModelSetupError(
            f"Could not read checkpoint '{path}': {type(exc).__name__}: {exc}\n{SETUP_HINT}"
        ) from exc

    state_dict = _extract_state_dict(payload)
    metadata = {
        key: value
        for key, value in (payload.items() if isinstance(payload, dict) else [])
        if isinstance(value, (str, int, float, bool))
    }

    # Some checkpoints (including Bombek1's) carry a nested "config" mapping
    # describing the backbones and LoRA hyper-parameters they were built with.
    checkpoint_config: Dict[str, Any] = {}
    if isinstance(payload, dict) and isinstance(payload.get("config"), dict):
        checkpoint_config = dict(payload["config"])
        metadata.update(
            {
                key: value
                for key, value in checkpoint_config.items()
                if isinstance(value, (str, int, float, bool))
            }
        )

    # Resolve the architecture. Tensor evidence beats any recorded label: the
    # state-dict signature is what strict loading will actually be judged
    # against, and it lets a Bombek1 checkpoint be used with the stock config.
    detected = detect_architecture(state_dict)
    declared = (
        metadata.get("model_name")
        or metadata.get("architecture")
        or model_config.get("name", "efficientnet_b0")
    )
    if detected is not None:
        architecture = detected
        metadata["detected_architecture"] = detected
        try:
            declared_canonical = normalise_architecture(str(declared))
        except ModelSetupError:
            declared_canonical = None
        if declared_canonical is not None and declared_canonical != detected:
            metadata["configured_architecture_overridden"] = str(declared)
    else:
        architecture = str(declared)

    num_classes = infer_num_classes(state_dict) or int(model_config.get("num_classes", 1))

    canonical = normalise_architecture(architecture)
    if canonical == BOMBEK_ARCHITECTURE:
        # The checkpoint carries complete backbone weights, so building the
        # towers from the hub first would download ~3 GB only to overwrite it.
        dual_settings = dict(model_config.get("bombek", {}) or {})
        dual_settings.update(checkpoint_config)
        use_pretrained_backbones = False
    else:
        dual_settings = dict(model_config.get("dual", {}) or {})
        if isinstance(payload, dict):
            for key in ("siglip_name", "dinov2_name", "hidden_dim", "dropout"):
                if key in payload:
                    dual_settings[key] = payload[key]
        use_pretrained_backbones = bool(
            dual_settings.get(
                "pretrained",
                model_config.get("pretrained", canonical == DUAL_BACKBONE_ARCHITECTURE),
            )
        )
    try:
        model = build_architecture(
            architecture,
            num_classes=num_classes,
            pretrained=use_pretrained_backbones,
            dual_config=dual_settings,
        )
    except ModelSetupError:
        raise
    except Exception as exc:
        raise ModelSetupError(
            f"Could not construct '{architecture}' from its configured backbones: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if canonical == BOMBEK_ARCHITECTURE:
        # Reconcile the transformers 4.x/5.x SigLIP layout before strict loading.
        from src.models.bombek_siglip2_dinov2 import align_checkpoint_keys

        state_dict, alignment_notes = align_checkpoint_keys(state_dict, model)
        if alignment_notes:
            metadata["checkpoint_key_alignment"] = " ".join(alignment_notes)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        summary = ""
        if canonical == BOMBEK_ARCHITECTURE:
            from src.models.bombek_siglip2_dinov2 import describe_state_dict_mismatch

            summary = f"\nKey comparison: {describe_state_dict_mismatch(state_dict, model)}"
        raise ModelSetupError(
            f"Checkpoint '{path}' does not match architecture "
            f"'{canonical}' with {num_classes} output class(es).\n"
            f"Set model.name in your config to the architecture the checkpoint was "
            f"trained with (one of {', '.join(ALL_ARCHITECTURES)}).{summary}\n"
            f"Details: {exc}"
        ) from exc

    num_parameters = count_parameters(model)
    limit = int(model_config.get("max_parameters", MAX_PARAMETERS))
    if num_parameters >= limit:
        raise ModelSetupError(
            f"Model has {num_parameters:,} parameters, at or above the "
            f"{limit:,} parameter limit for this task."
        )

    model.to(torch_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    processors = None
    input_kind = "single"
    if canonical in DUAL_INPUT_ARCHITECTURES:
        try:
            if canonical == BOMBEK_ARCHITECTURE:
                from src.models.bombek_siglip2_dinov2 import build_bombek_processors

                processors = build_bombek_processors(dual_settings)
            else:
                from src.models.dual_backbone import build_processors

                processors = build_processors(
                    str(dual_settings.get("siglip_name", "google/siglip2-so400m-patch14-384")),
                    str(dual_settings.get("dinov2_name", "facebook/dinov2-large")),
                )
        except Exception as exc:
            raise ModelSetupError(
                f"Could not load the image processors for '{canonical}': "
                f"{type(exc).__name__}: {exc}. Check transformers installation and network access."
            ) from exc
        input_kind = "dual"

    return ModelBundle(
        model=model,
        device=torch_device,
        architecture=canonical,
        num_classes=num_classes,
        num_parameters=num_parameters,
        checkpoint_path=str(path),
        ai_class_index=int(model_config.get("ai_class_index", 1)),
        metadata=metadata,
        input_kind=input_kind,
        processors=processors,
    )
