"""Adapter for the external Bombek1 SigLIP2 + DINOv2 LoRA detector.

Source checkpoint
-----------------
https://huggingface.co/Bombek1/ai-image-detector-siglip-dinov2 (``pytorch_model.pt``,
2.11 GB). The architecture is vendored at the repository root in ``model.py``;
this module reuses ``ClassificationHead`` and ``LoRALinear`` from there so the
head and LoRA definitions stay single-sourced and the state-dict keys match
byte for byte.

Why this is a separate architecture
-----------------------------------
This is **not** compatible with :mod:`src.models.dual_backbone`, and no key
renaming can make it so. Verified against a locally constructed model
(740,371,777 parameters, 954 state-dict keys):

===================  ==========================================  ==================================================
component            native ``dual_backbone``                    ``bombek_siglip2_dinov2``
===================  ==========================================  ==================================================
SigLIP2 keys         ``siglip.encoder.layers.N.…``               ``siglip.base_model.model.encoder.layers.N.…``
DINOv2 source        transformers ``Dinov2Model``                timm ``vit_large_patch14_dinov2.lvd142m``
DINOv2 keys          ``dinov2.encoder.layer.N.…``                ``dinov2.blocks.N.attn.qkv.original.…``
LoRA                 none                                        108 SigLIP + 48 DINOv2 adapter tensors
head                 ``head.{0,2,5}``  2176 -> 3584 -> 1         ``classifier.head.{0,1,4,7}``  2176 -> 512 -> 256 -> 1
input resolution     SigLIP 384 / DINOv2 224                     SigLIP 384 / DINOv2 392
===================  ==========================================  ==================================================

Loading is therefore strict against *this* architecture. If the checkpoint does
not match, the loader reports the mismatch rather than forcing a partial load
that would yield meaningless predictions.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

BOMBEK_ARCHITECTURE = "bombek_siglip2_dinov2"

DEFAULT_SIGLIP = "google/siglip2-so400m-patch14-384"
DEFAULT_DINOV2 = "vit_large_patch14_dinov2.lvd142m"
DEFAULT_IMAGE_SIZE = 392
DEFAULT_LORA_RANK = 32
DEFAULT_LORA_ALPHA = 64
DEFAULT_LORA_DROPOUT = 0.1
DEFAULT_HIDDEN_DIM = 512
DEFAULT_CLASSIFIER_DROPOUT = 0.3

# The vendored reference implementation lives at the repository root.
_VENDORED_MODEL_PATH = Path(__file__).resolve().parents[2] / "model.py"

MISSING_DEPENDENCY_HINT = (
    "The Bombek1 detector needs 'peft' and 'timm'. Install them with:\n"
    "    pip install -r requirements.txt"
)


def _load_vendored_module() -> Any:
    """Import the vendored ``model.py`` without relying on sys.path order.

    Loading it by file path avoids clashing with any other top-level module
    called ``model`` that happens to be importable.
    """

    if "bombek_vendored_model" in sys.modules:
        return sys.modules["bombek_vendored_model"]
    if not _VENDORED_MODEL_PATH.is_file():
        raise ImportError(
            f"Vendored Bombek1 architecture not found at {_VENDORED_MODEL_PATH}. "
            "It defines ClassificationHead and LoRALinear, which this adapter reuses."
        )
    spec = importlib.util.spec_from_file_location("bombek_vendored_model", _VENDORED_MODEL_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Could not build an import spec for {_VENDORED_MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["bombek_vendored_model"] = module
    spec.loader.exec_module(module)
    return module


class BombekSigLIP2DINOv2Detector(nn.Module):
    """SigLIP2 + DINOv2 ensemble with LoRA adapters on both branches.

    Submodules are named ``siglip``, ``dinov2`` and ``classifier`` to match the
    published checkpoint exactly. Unlike the upstream ``EnsembleAIDetector``,
    :meth:`forward` returns **only** the logits tensor; the upstream three-tuple
    would break every downstream stage that expects a tensor. Use
    :meth:`forward_with_features` when the branch features are wanted.
    """

    def __init__(
        self,
        siglip: nn.Module,
        dinov2: nn.Module,
        classifier: nn.Module,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        super().__init__()
        self.siglip = siglip
        self.dinov2 = dinov2
        self.classifier = classifier
        self.image_size = int(image_size)

    def forward_with_features(
        self, siglip_pixels: torch.Tensor, dinov2_pixels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(logits, siglip_features, dinov2_features)``."""

        # The SigLIP tower is stored in bfloat16; feed it a matching tensor and
        # bring the features back to float32 before concatenating.
        siglip_dtype = next(self.siglip.parameters()).dtype
        siglip_features = self.siglip(pixel_values=siglip_pixels.to(siglip_dtype)).pooler_output
        dinov2_features = self.dinov2(dinov2_pixels.to(next(self.dinov2.parameters()).dtype))
        combined = torch.cat([siglip_features.float(), dinov2_features.float()], dim=-1)
        logits = self.classifier(combined)
        return logits, siglip_features, dinov2_features

    def forward(self, siglip_pixels: torch.Tensor, dinov2_pixels: torch.Tensor) -> torch.Tensor:
        """Return one logit per image. Apply sigmoid for P(AI-generated)."""

        logits, _, _ = self.forward_with_features(siglip_pixels, dinov2_pixels)
        return logits.reshape(-1)


def bombek_settings(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolve adapter settings, accepting both our and upstream key spellings."""

    settings = dict(config or {})
    return {
        "siglip_model": str(
            settings.get("siglip_model") or settings.get("siglip_name") or DEFAULT_SIGLIP
        ),
        "dinov2_model": str(
            settings.get("dinov2_model") or settings.get("dinov2_name") or DEFAULT_DINOV2
        ),
        "image_size": int(settings.get("image_size", DEFAULT_IMAGE_SIZE)),
        "lora_rank": int(settings.get("lora_rank", settings.get("rank", DEFAULT_LORA_RANK))),
        "lora_alpha": int(settings.get("lora_alpha", settings.get("alpha", DEFAULT_LORA_ALPHA))),
        "lora_dropout": float(settings.get("lora_dropout", DEFAULT_LORA_DROPOUT)),
        "hidden_dim": int(settings.get("hidden_dim", DEFAULT_HIDDEN_DIM)),
        "classifier_dropout": float(settings.get("classifier_dropout", DEFAULT_CLASSIFIER_DROPOUT)),
    }


def build_bombek_detector(
    config: Optional[Dict[str, Any]] = None, pretrained_backbones: bool = False
) -> BombekSigLIP2DINOv2Detector:
    """Construct the architecture the published checkpoint expects.

    ``pretrained_backbones`` defaults to False because the checkpoint carries
    complete weights for both towers; downloading the upstream backbones first
    would fetch roughly 3 GB only to overwrite it.
    """

    settings = bombek_settings(config)

    try:
        import timm
        from peft import LoraConfig, get_peft_model
        from transformers import AutoConfig, SiglipVisionModel
    except ImportError as exc:
        raise ImportError(f"{exc}\n{MISSING_DEPENDENCY_HINT}") from exc

    vendored = _load_vendored_module()

    # --- SigLIP2 vision tower, held in bfloat16 as upstream does -------------
    siglip_name = settings["siglip_model"]
    if pretrained_backbones:
        siglip = SiglipVisionModel.from_pretrained(siglip_name, torch_dtype=torch.bfloat16)
    else:
        vision_config = AutoConfig.from_pretrained(siglip_name)
        vision_config = getattr(vision_config, "vision_config", vision_config)
        siglip = SiglipVisionModel(vision_config).to(torch.bfloat16)
    siglip_dim = int(siglip.config.hidden_size)

    # --- DINOv2 from timm (NOT transformers Dinov2Model) --------------------
    dinov2 = timm.create_model(
        settings["dinov2_model"],
        pretrained=bool(pretrained_backbones),
        num_classes=0,
        img_size=settings["image_size"],
    )
    dinov2_dim = int(dinov2.num_features)

    classifier = vendored.ClassificationHead(
        siglip_dim + dinov2_dim,
        hidden_dim=settings["hidden_dim"],
        dropout=settings["classifier_dropout"],
    )

    # --- LoRA on both branches, exactly as create_model_with_lora does ------
    siglip = get_peft_model(
        siglip,
        LoraConfig(
            r=settings["lora_rank"],
            lora_alpha=settings["lora_alpha"],
            target_modules=["q_proj", "v_proj"],
            lora_dropout=settings["lora_dropout"],
            bias="none",
        ),
    )
    for _, module in dinov2.named_modules():
        if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear):
            module.qkv = vendored.LoRALinear(
                module.qkv,
                settings["lora_rank"],
                settings["lora_alpha"],
                settings["lora_dropout"],
            )

    return BombekSigLIP2DINOv2Detector(
        siglip=siglip,
        dinov2=dinov2,
        classifier=classifier,
        image_size=settings["image_size"],
    )


def looks_like_bombek_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
    """Detect this architecture from its distinctive key signature.

    Requires all three markers so a native ``dual_backbone`` checkpoint (which
    has ``head.*`` and no LoRA tensors) can never be misidentified.
    """

    keys = list(state_dict)
    has_classifier_head = any(key.startswith("classifier.head.") for key in keys)
    has_lora = any("lora_A" in key or "lora_B" in key for key in keys)
    has_peft_siglip = any(key.startswith("siglip.base_model.") for key in keys)
    return has_classifier_head and has_lora and has_peft_siglip


def describe_state_dict_mismatch(state_dict: Dict[str, torch.Tensor], model: nn.Module) -> str:
    """Summarise how a checkpoint differs from this architecture."""

    expected = set(model.state_dict())
    provided = set(state_dict)
    missing = sorted(expected - provided)
    unexpected = sorted(provided - expected)
    lines: List[str] = [
        f"expected {len(expected)} tensors, checkpoint has {len(provided)}",
    ]
    if missing:
        lines.append(f"missing {len(missing)}, e.g. {missing[:3]}")
    if unexpected:
        lines.append(f"unexpected {len(unexpected)}, e.g. {unexpected[:3]}")
    return "; ".join(lines)


class TorchvisionImageProcessor:
    """Give a torchvision transform the Hugging Face processor call convention.

    ``src.pipeline.prediction._processor_batch`` calls every processor as
    ``processor(images=[...], return_tensors="pt")["pixel_values"]``. The DINOv2
    branch uses a plain torchvision pipeline, so this wrapper lets both branches
    share that one code path.
    """

    def __init__(self, transform: Any, size: int) -> None:
        self.transform = transform
        self.size = int(size)

    def __call__(self, images: Any = None, return_tensors: str = "pt", **_: Any) -> Dict[str, Any]:
        from PIL import Image as PILImage

        if images is None:
            raise ValueError("TorchvisionImageProcessor requires images")
        if isinstance(images, PILImage.Image):
            images = [images]
        batch = torch.stack([self.transform(image.convert("RGB")) for image in list(images)], dim=0)
        if return_tensors != "pt":
            raise ValueError(
                f"TorchvisionImageProcessor only produces torch tensors, got '{return_tensors}'"
            )
        return {"pixel_values": batch}

    def get(self, key: str, default: Any = None) -> Any:  # pragma: no cover - dict-like shim
        return default


def build_bombek_processors(
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, TorchvisionImageProcessor]:
    """Return ``(siglip_processor, dinov2_processor)`` for this detector.

    The two branches deliberately differ: SigLIP2 uses its own published
    processor at its native 384 px with SigLIP normalisation, while DINOv2 uses
    the upstream torchvision pipeline at 392 px with ImageNet normalisation.
    Feeding either branch the other's tensor runs without error and silently
    degrades the features, so they stay separate.
    """

    settings = bombek_settings(config)
    vendored = _load_vendored_module()

    try:
        from transformers import AutoImageProcessor
    except ImportError as exc:  # pragma: no cover - transformers is required
        raise ImportError(f"{exc}\n{MISSING_DEPENDENCY_HINT}") from exc

    try:
        siglip_processor = AutoImageProcessor.from_pretrained(settings["siglip_model"])
    except Exception:
        from transformers import AutoProcessor

        siglip_processor = AutoProcessor.from_pretrained(settings["siglip_model"])

    dinov2_processor = TorchvisionImageProcessor(
        vendored.create_transforms(settings["image_size"]), settings["image_size"]
    )
    return siglip_processor, dinov2_processor


# ---------------------------------------------------------------------------
# transformers 4.x <-> 5.x SigLIP layout reconciliation
# ---------------------------------------------------------------------------
#
# The published checkpoint was trained on transformers 4.x, where
# ``SiglipVisionModel`` held its transformer in a ``vision_model`` submodule.
# transformers 5.x flattened that away, and declares the equivalence itself via
# ``SiglipVisionModel.base_model_prefix = "vision_model"`` -- the same mechanism
# ``from_pretrained`` uses to load older checkpoints.
#
# Verified against the real 2.11 GB checkpoint: both sides have exactly 954
# tensors, and reconciling this single path segment leaves 0 missing, 0
# unexpected and 0 shape mismatches. Nothing else is renamed, and the result is
# still loaded with ``strict=True`` so any drift fails loudly.

SIGLIP_PEFT_PREFIX = "siglip.base_model.model."
SIGLIP_LEGACY_PREFIX = SIGLIP_PEFT_PREFIX + "vision_model."


def align_checkpoint_keys(
    state_dict: Dict[str, torch.Tensor], model: nn.Module
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Reconcile the SigLIP ``vision_model`` segment with the installed layout.

    Compares the checkpoint's key layout against *this* model's actual keys, so
    it adapts in whichever direction the installed transformers requires and
    does nothing when the two already agree. Only the SigLIP branch is touched;
    DINOv2 and classifier keys are passed through untouched.
    """

    model_keys = set(model.state_dict())
    checkpoint_is_legacy = any(key.startswith(SIGLIP_LEGACY_PREFIX) for key in state_dict)
    model_is_legacy = any(key.startswith(SIGLIP_LEGACY_PREFIX) for key in model_keys)
    notes: List[str] = []

    if checkpoint_is_legacy and not model_is_legacy:
        state_dict = {
            (
                SIGLIP_PEFT_PREFIX + key[len(SIGLIP_LEGACY_PREFIX) :]
                if key.startswith(SIGLIP_LEGACY_PREFIX)
                else key
            ): value
            for key, value in state_dict.items()
        }
        notes.append(
            "Removed the legacy SigLIP 'vision_model.' segment to match "
            "transformers >= 5 (SiglipVisionModel.base_model_prefix)."
        )
    elif model_is_legacy and not checkpoint_is_legacy:
        state_dict = {
            (
                SIGLIP_LEGACY_PREFIX + key[len(SIGLIP_PEFT_PREFIX) :]
                if key.startswith(SIGLIP_PEFT_PREFIX)
                else key
            ): value
            for key, value in state_dict.items()
        }
        notes.append("Added the SigLIP 'vision_model.' segment to match transformers < 5.")
    return state_dict, notes
