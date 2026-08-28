"""Dual frozen-backbone detector: SigLIP2 + DINOv2 with a trainable head.

The two backbones are chosen because they disagree usefully. SigLIP2 is
language-supervised, so its features encode semantic and stylistic
properties -- roughly "does this look like something a photographer would
have shot". DINOv2 is self-supervised on images alone, so its features lean
toward texture, structure and local geometry, which is where several
generator artifacts live. Concatenating them gives the head two fairly
different views of the same image, and a weakness in one is often covered by
the other.

Both backbones stay frozen. That is the same reasoning as the CLIP probe: a
backbone free to update will drift toward the training generator's
fingerprint, which scores well in-distribution and collapses on unseen
generators. Only the head learns, so capacity stays proportionate to the
dataset.

Parameter budget (competition limit is 2B):

    SigLIP2 So400m vision tower   428,225,600   frozen
    DINOv2-Large                  304,368,640   frozen
    Classification head           ~7,800,000    trainable
    -------------------------------------------------
    Total                         ~740,000,000

Run scripts/count_params.py for the exact, machine-verified figures.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

DEFAULT_SIGLIP = "google/siglip2-so400m-patch14-384"
DEFAULT_DINOV2 = "facebook/dinov2-large"
DEFAULT_HIDDEN = 3584

# Competition rule: submitted models must stay under 2B parameters.
MAX_PARAMETERS = 2_000_000_000


class DualBackboneDetector(nn.Module):
    """Frozen SigLIP2 + DINOv2 backbones with a small trainable MLP head.

    forward() takes the two preprocessed tensors separately because the
    backbones do not share a preprocessing convention: SigLIP2 expects its own
    normalisation at 384px, DINOv2 expects ImageNet normalisation at 224px.
    Feeding one backbone the other's tensor runs without error and silently
    degrades the features, so the interface keeps them explicit.
    """

    def __init__(
        self,
        siglip_name: str = DEFAULT_SIGLIP,
        dinov2_name: str = DEFAULT_DINOV2,
        hidden_dim: int = DEFAULT_HIDDEN,
        dropout: float = 0.2,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel, Dinov2Model

        # --- SigLIP2: keep only the vision tower; the text tower is unused ---
        if pretrained:
            siglip = AutoModel.from_pretrained(siglip_name)
        else:
            # from_config gives correct shapes with random weights, which is
            # enough to verify the parameter budget without downloading GBs.
            siglip = AutoModel.from_config(AutoConfig.from_pretrained(siglip_name))
        self.siglip = siglip.vision_model
        siglip_dim = self.siglip.config.hidden_size

        # --- DINOv2 ---
        if pretrained:
            self.dinov2 = Dinov2Model.from_pretrained(dinov2_name)
        else:
            self.dinov2 = Dinov2Model(AutoConfig.from_pretrained(dinov2_name))
        dinov2_dim = self.dinov2.config.hidden_size

        self.freeze_backbones()

        # --- Trainable head over the concatenated features ---
        self.head = nn.Sequential(
            nn.LayerNorm(siglip_dim + dinov2_dim),
            nn.Dropout(dropout),
            nn.Linear(siglip_dim + dinov2_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.feature_dim = siglip_dim + dinov2_dim

    def freeze_backbones(self) -> None:
        """Freeze both backbones and hold them in eval mode.

        eval() matters as well as requires_grad: it stops dropout and any
        normalisation layer from updating running statistics when the head is
        trained, which would otherwise drift the "frozen" features.
        """

        for backbone in (self.siglip, self.dinov2):
            backbone.eval()
            for parameter in backbone.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):  # type: ignore[override]
        """Put the head in train mode but never the frozen backbones."""

        super().train(mode)
        self.siglip.eval()
        self.dinov2.eval()
        return self

    @torch.no_grad()
    def encode(self, siglip_pixels: torch.Tensor, dinov2_pixels: torch.Tensor) -> torch.Tensor:
        """Return concatenated frozen features, [N, siglip_dim + dinov2_dim].

        Separated from forward() so features can be extracted once and cached,
        exactly as with the CLIP path: the head then re-trains in seconds
        without re-running either backbone.
        """

        siglip_features = self.siglip(pixel_values=siglip_pixels).pooler_output
        dinov2_features = self.dinov2(pixel_values=dinov2_pixels).pooler_output
        return torch.cat([siglip_features, dinov2_features], dim=-1)

    def forward(self, siglip_pixels: torch.Tensor, dinov2_pixels: torch.Tensor) -> torch.Tensor:
        """Return one logit per image. Apply sigmoid for a probability."""

        return self.head(self.encode(siglip_pixels, dinov2_pixels)).reshape(-1)

    def head_forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run only the head, for training against cached features."""

        return self.head(features).reshape(-1)


def build_processors(
    siglip_name: str = DEFAULT_SIGLIP,
    dinov2_name: str = DEFAULT_DINOV2,
) -> Tuple[Any, Any]:
    """Return the two image processors, one per backbone.

    They differ in resolution and normalisation constants, so each backbone
    must be fed its own tensor.
    """

    from transformers import AutoImageProcessor

    return (
        AutoImageProcessor.from_pretrained(siglip_name),
        AutoImageProcessor.from_pretrained(dinov2_name),
    )


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Return total, trainable and frozen parameter counts.

    ``total`` is the figure the competition's <2B limit applies to.
    """

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def assert_within_budget(model: nn.Module, limit: int = MAX_PARAMETERS) -> Dict[str, int]:
    """Raise if the model exceeds the competition parameter limit."""

    counts = count_parameters(model)
    if counts["total"] > limit:
        raise ValueError(
            f"Model has {counts['total']:,} parameters, exceeding the {limit:,} limit."
        )
    return counts
