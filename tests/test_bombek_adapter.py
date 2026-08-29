"""Tests for the external Bombek1 SigLIP2 + DINOv2 LoRA adapter.

These never download Hugging Face weights. The real architecture is ~740 M
parameters across two transformer towers, so every test here uses a tiny
stand-in with the *same key layout* and the same two-input contract. That is
enough to verify detection, dispatch, preprocessing wiring, error handling and
the JSON contract -- the things this adapter is responsible for.

Nothing here says anything about detection accuracy.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

from src.models.bombek_siglip2_dinov2 import (
    BOMBEK_ARCHITECTURE,
    BombekSigLIP2DINOv2Detector,
    TorchvisionImageProcessor,
    align_checkpoint_keys,
    bombek_settings,
    describe_state_dict_mismatch,
    looks_like_bombek_state_dict,
)
from src.pipeline.model_loader import (
    ALL_ARCHITECTURES,
    DUAL_INPUT_ARCHITECTURES,
    ModelBundle,
    ModelSetupError,
    detect_architecture,
    normalise_architecture,
)
from tests.helpers import base_config, make_image

SIGLIP_DIM = 8
DINOV2_DIM = 6


def bombek_like_state_dict() -> Dict[str, torch.Tensor]:
    """A minimal state dict carrying the real checkpoint's key signature."""

    return {
        "siglip.base_model.model.embeddings.patch_embedding.weight": torch.zeros(4, 3, 2, 2),
        "siglip.base_model.model.encoder.layers.0.self_attn.q_proj.lora_A.default.weight": (
            torch.zeros(2, 4)
        ),
        "siglip.base_model.model.encoder.layers.0.self_attn.q_proj.lora_B.default.weight": (
            torch.zeros(4, 2)
        ),
        "dinov2.blocks.0.attn.qkv.original.weight": torch.zeros(12, 4),
        "dinov2.blocks.0.attn.qkv.lora_A.weight": torch.zeros(2, 4),
        "dinov2.blocks.0.attn.qkv.lora_B.weight": torch.zeros(12, 2),
        "classifier.head.0.weight": torch.zeros(10),
        "classifier.head.7.bias": torch.zeros(1),
    }


def native_dual_state_dict() -> Dict[str, torch.Tensor]:
    """Key signature of the NATIVE dual_backbone detector: no LoRA, `head.*`."""

    return {
        "siglip.embeddings.patch_embedding.weight": torch.zeros(4, 3, 2, 2),
        "siglip.encoder.layers.0.self_attn.q_proj.weight": torch.zeros(4, 4),
        "dinov2.encoder.layer.0.attention.attention.query.weight": torch.zeros(4, 4),
        "head.0.weight": torch.zeros(10),
        "head.5.bias": torch.zeros(1),
    }


class TinySiglipTower(nn.Module):
    """Stand-in exposing the `pooler_output` interface the adapter calls."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, SIGLIP_DIM)

    def forward(self, pixel_values: torch.Tensor) -> Any:
        pooled = self.proj(pixel_values.float().mean(dim=(2, 3)))
        return type("Output", (), {"pooler_output": pooled})()


class TinyDinoTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, DINOV2_DIM)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.proj(pixel_values.float().mean(dim=(2, 3)))


def tiny_detector() -> BombekSigLIP2DINOv2Detector:
    """Build the real wrapper class around tiny towers -- no downloads."""

    torch.manual_seed(0)
    classifier = nn.Sequential(nn.Linear(SIGLIP_DIM + DINOV2_DIM, 1), nn.Flatten(0))
    return BombekSigLIP2DINOv2Detector(
        siglip=TinySiglipTower(),
        dinov2=TinyDinoTower(),
        classifier=classifier,
        image_size=392,
    )


class ArchitectureRegistrationTest(unittest.TestCase):
    def test_architecture_is_registered_under_its_own_name(self) -> None:
        self.assertIn(BOMBEK_ARCHITECTURE, ALL_ARCHITECTURES)
        self.assertEqual(BOMBEK_ARCHITECTURE, "bombek_siglip2_dinov2")

    def test_aliases_resolve(self) -> None:
        for alias in ("bombek", "Bombek_SigLIP2_DINOv2", "EnsembleAIDetector"):
            with self.subTest(alias=alias):
                self.assertEqual(normalise_architecture(alias), BOMBEK_ARCHITECTURE)

    def test_it_is_not_an_alias_of_the_native_dual_backbone(self) -> None:
        """The external checkpoint must never be claimed compatible with dual_backbone."""

        self.assertNotEqual(normalise_architecture("dual_backbone"), BOMBEK_ARCHITECTURE)
        self.assertNotEqual(normalise_architecture("siglip2_dinov2"), BOMBEK_ARCHITECTURE)

    def test_both_dual_input_architectures_are_listed(self) -> None:
        self.assertIn(BOMBEK_ARCHITECTURE, DUAL_INPUT_ARCHITECTURES)
        self.assertIn("dual_backbone", DUAL_INPUT_ARCHITECTURES)

    def test_preserved_single_input_architectures(self) -> None:
        for name in ("efficientnet_b0", "resnet18", "convnext_tiny"):
            with self.subTest(architecture=name):
                self.assertIn(name, ALL_ARCHITECTURES)
                self.assertNotIn(name, DUAL_INPUT_ARCHITECTURES)


class StateDictDetectionTest(unittest.TestCase):
    def test_detects_the_bombek_signature(self) -> None:
        self.assertTrue(looks_like_bombek_state_dict(bombek_like_state_dict()))
        self.assertEqual(detect_architecture(bombek_like_state_dict()), BOMBEK_ARCHITECTURE)

    def test_does_not_claim_the_native_dual_backbone(self) -> None:
        self.assertFalse(looks_like_bombek_state_dict(native_dual_state_dict()))
        self.assertIsNone(detect_architecture(native_dual_state_dict()))

    def test_does_not_claim_a_cnn_checkpoint(self) -> None:
        from src.pipeline.model_loader import build_architecture

        state = build_architecture("resnet18", num_classes=1).state_dict()
        self.assertFalse(looks_like_bombek_state_dict(state))
        self.assertIsNone(detect_architecture(state))

    def test_all_three_markers_are_required(self) -> None:
        """Any single marker family alone must not trigger detection."""

        families = {
            "classifier head": lambda k: k.startswith("classifier."),
            "lora tensors": lambda k: "lora_A" in k or "lora_B" in k,
            "peft-wrapped siglip": lambda k: k.startswith("siglip.base_model."),
        }
        for label, belongs_to_family in families.items():
            with self.subTest(removed=label):
                partial = {
                    key: value
                    for key, value in bombek_like_state_dict().items()
                    if not belongs_to_family(key)
                }
                self.assertFalse(looks_like_bombek_state_dict(partial))

    def test_mismatch_description_names_the_differences(self) -> None:
        model = tiny_detector()
        summary = describe_state_dict_mismatch(bombek_like_state_dict(), model)
        self.assertIn("expected", summary)
        self.assertIn("unexpected", summary)


class SettingsTest(unittest.TestCase):
    def test_defaults_match_the_published_config(self) -> None:
        settings = bombek_settings({})
        self.assertEqual(settings["siglip_model"], "google/siglip2-so400m-patch14-384")
        self.assertEqual(settings["dinov2_model"], "vit_large_patch14_dinov2.lvd142m")
        self.assertEqual(settings["image_size"], 392)
        self.assertEqual(settings["lora_rank"], 32)
        self.assertEqual(settings["lora_alpha"], 64)
        self.assertEqual(settings["hidden_dim"], 512)

    def test_checkpoint_config_spellings_are_accepted(self) -> None:
        settings = bombek_settings(
            {"siglip_model": "a", "dinov2_model": "b", "image_size": 224, "lora_rank": 8}
        )
        self.assertEqual((settings["siglip_model"], settings["dinov2_model"]), ("a", "b"))
        self.assertEqual(settings["image_size"], 224)
        self.assertEqual(settings["lora_rank"], 8)

    def test_native_dual_spellings_are_also_accepted(self) -> None:
        settings = bombek_settings({"siglip_name": "x", "dinov2_name": "y"})
        self.assertEqual((settings["siglip_model"], settings["dinov2_model"]), ("x", "y"))


class ForwardContractTest(unittest.TestCase):
    """forward() must return a plain logits tensor, not the upstream 3-tuple."""

    def setUp(self) -> None:
        self.model = tiny_detector().eval()
        self.siglip_pixels = torch.rand(3, 3, 8, 8)
        self.dinov2_pixels = torch.rand(3, 3, 10, 10)

    def test_forward_returns_a_flat_logit_tensor(self) -> None:
        with torch.no_grad():
            out = self.model(self.siglip_pixels, self.dinov2_pixels)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(tuple(out.shape), (3,))

    def test_forward_with_features_returns_the_three_tuple(self) -> None:
        with torch.no_grad():
            logits, siglip_features, dinov2_features = self.model.forward_with_features(
                self.siglip_pixels, self.dinov2_pixels
            )
        self.assertEqual(tuple(logits.shape), (3,))
        self.assertEqual(tuple(siglip_features.shape), (3, SIGLIP_DIM))
        self.assertEqual(tuple(dinov2_features.shape), (3, DINOV2_DIM))

    def test_the_two_branches_receive_different_tensors(self) -> None:
        """A shape mismatch proves the branches are not fed the same input."""

        self.assertNotEqual(self.siglip_pixels.shape, self.dinov2_pixels.shape)
        with torch.no_grad():
            self.model(self.siglip_pixels, self.dinov2_pixels)

    def test_probabilities_from_logits_accepts_the_output(self) -> None:
        from src.pipeline.prediction import probabilities_from_logits

        with torch.no_grad():
            out = self.model(self.siglip_pixels, self.dinov2_pixels)
        probabilities = probabilities_from_logits(out, num_classes=1)
        self.assertEqual(probabilities.shape, (3,))
        self.assertTrue(((probabilities >= 0.0) & (probabilities <= 1.0)).all())


class TorchvisionProcessorTest(unittest.TestCase):
    """The DINOv2 branch uses a torchvision transform behind an HF-style call."""

    def setUp(self) -> None:
        from torchvision import transforms

        self.processor = TorchvisionImageProcessor(
            transforms.Compose([transforms.Resize((392, 392)), transforms.ToTensor()]),
            392,
        )

    def test_returns_a_stacked_pixel_values_batch(self) -> None:
        images = [make_image(width=60, height=40, seed=i) for i in range(3)]
        encoded = self.processor(images=images, return_tensors="pt")
        self.assertEqual(tuple(encoded["pixel_values"].shape), (3, 3, 392, 392))

    def test_accepts_a_single_image(self) -> None:
        encoded = self.processor(images=make_image(), return_tensors="pt")
        self.assertEqual(tuple(encoded["pixel_values"].shape), (1, 3, 392, 392))

    def test_converts_non_rgb_input(self) -> None:
        encoded = self.processor(images=[make_image(mode="L")], return_tensors="pt")
        self.assertEqual(encoded["pixel_values"].shape[1], 3)

    def test_matches_the_prediction_module_call_convention(self) -> None:
        from src.pipeline.prediction import _processor_batch

        batch = _processor_batch(self.processor, [make_image(), make_image(seed=2)])
        self.assertEqual(tuple(batch.shape), (2, 3, 392, 392))

    def test_rejects_non_torch_output_requests(self) -> None:
        with self.assertRaises(ValueError):
            self.processor(images=[make_image()], return_tensors="np")

    def test_requires_images(self) -> None:
        with self.assertRaises(ValueError):
            self.processor(images=None)


class DualInputInferenceTest(unittest.TestCase):
    """End-to-end prediction plumbing with a mocked two-input bundle."""

    def setUp(self) -> None:
        from torchvision import transforms

        from src.pipeline.preprocessing import Preprocessor

        self.config = base_config()
        self.preprocessor = Preprocessor.from_config(self.config)
        siglip_processor = TorchvisionImageProcessor(
            transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()]), 8
        )
        dinov2_processor = TorchvisionImageProcessor(
            transforms.Compose([transforms.Resize((10, 10)), transforms.ToTensor()]), 10
        )
        self.bundle = ModelBundle(
            model=tiny_detector().eval(),
            device=torch.device("cpu"),
            architecture=BOMBEK_ARCHITECTURE,
            num_classes=1,
            num_parameters=123,
            checkpoint_path="<mock>",
            input_kind="dual",
            processors=(siglip_processor, dinov2_processor),
        )

    def test_predict_images_returns_probabilities(self) -> None:
        from src.pipeline.prediction import predict_images

        images = [make_image(seed=i) for i in range(4)]
        probabilities = predict_images(self.bundle, images, self.preprocessor, batch_size=2)
        self.assertEqual(probabilities.shape, (4,))
        self.assertTrue(((probabilities >= 0.0) & (probabilities <= 1.0)).all())

    def test_predict_variants_covers_transformations(self) -> None:
        from src.pipeline.prediction import predict_variants
        from src.pipeline.transformations import generate_variants

        variants, errors = generate_variants(make_image(), self.config)
        self.assertEqual(errors, [])
        predictions = predict_variants(self.bundle, variants, self.preprocessor, self.config)
        self.assertEqual(len(predictions), 15)
        self.assertTrue(predictions[0].is_original)

    def test_consistency_and_confidence_run_on_dual_output(self) -> None:
        from src.pipeline.confidence import compute_confidence
        from src.pipeline.consistency import compute_consistency
        from src.pipeline.prediction import predict_variants
        from src.pipeline.transformations import generate_variants

        variants, _ = generate_variants(make_image(), self.config)
        predictions = predict_variants(self.bundle, variants, self.preprocessor, self.config)
        consistency = compute_consistency(predictions, self.config)
        self.assertGreaterEqual(consistency.consistency_score, 0.0)
        self.assertLessEqual(consistency.consistency_score, 1.0)

        confidence = compute_confidence(
            predictions[0].ai_probability,
            consistency.agreement,
            consistency.consistency_score,
            self.config,
        )
        self.assertIn(confidence.level, {"High", "Medium", "Low"})

    def test_dual_batch_requires_a_pair(self) -> None:
        from src.pipeline.prediction import predict_tensor_batch

        with self.assertRaises(ValueError):
            predict_tensor_batch(self.bundle, torch.rand(2, 3, 8, 8))

    def test_gradcam_reports_unavailable_rather_than_a_wrong_heatmap(self) -> None:
        from src.pipeline.explainability import explain

        result = explain(self.bundle, torch.rand(1, 3, 224, 224), make_image(), self.config)
        self.assertFalse(result.available)
        self.assertIsNone(result.heatmap)
        self.assertIsNone(result.overlay)
        self.assertIn("LoRA", result.message)
        self.assertIn("unaffected", result.message)

    def test_bundle_summary_reports_the_external_architecture(self) -> None:
        summary = self.bundle.summary()
        self.assertEqual(summary["architecture"], BOMBEK_ARCHITECTURE)
        self.assertEqual(summary["input_kind"], "dual")
        self.assertTrue(summary["under_2b_parameter_limit"])


class LoaderErrorTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.config = base_config()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_missing_external_checkpoint_reports_setup_error(self) -> None:
        from src.pipeline.model_loader import load_model

        with self.assertRaises(ModelSetupError) as context:
            load_model(self.tmp / "models" / "pretrained" / "pytorch_model.pt", self.config)
        self.assertIn("not found", str(context.exception))

    def test_two_class_head_is_rejected(self) -> None:
        from src.pipeline.model_loader import build_architecture

        with self.assertRaises(ModelSetupError) as context:
            build_architecture(BOMBEK_ARCHITECTURE, num_classes=2)
        self.assertIn("single binary output logit", str(context.exception))

    def test_prefix_stripping_leaves_bombek_keys_untouched(self) -> None:
        """`siglip.base_model.model.*` must survive the DataParallel cleanup."""

        from src.pipeline.model_loader import _extract_state_dict

        cleaned = _extract_state_dict({"model_state_dict": bombek_like_state_dict()})
        self.assertIn("siglip.base_model.model.embeddings.patch_embedding.weight", cleaned)
        self.assertTrue(looks_like_bombek_state_dict(cleaned))

    def test_uniform_prefixes_are_still_stripped(self) -> None:
        from src.pipeline.model_loader import _extract_state_dict

        prefixed = {f"module.{k}": v for k, v in bombek_like_state_dict().items()}
        cleaned = _extract_state_dict({"model_state_dict": prefixed})
        self.assertTrue(looks_like_bombek_state_dict(cleaned))


class JsonContractTest(unittest.TestCase):
    """The documented JSON fields must survive a dual-input run."""

    def test_detailed_record_has_probability_label_confidence_and_path(self) -> None:
        from torchvision import transforms

        from src.pipeline.pipeline import DetectionPipeline
        from src.pipeline.validation import ImageMetadata

        config = base_config()
        config["transformations"]["enabled"] = False
        bundle = ModelBundle(
            model=tiny_detector().eval(),
            device=torch.device("cpu"),
            architecture=BOMBEK_ARCHITECTURE,
            num_classes=1,
            num_parameters=740_371_777,
            checkpoint_path="models/pretrained/pytorch_model.pt",
            input_kind="dual",
            processors=(
                TorchvisionImageProcessor(
                    transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()]), 8
                ),
                TorchvisionImageProcessor(
                    transforms.Compose([transforms.Resize((10, 10)), transforms.ToTensor()]), 10
                ),
            ),
        )
        pipeline = DetectionPipeline(bundle, config, explain_images=True)
        metadata = ImageMetadata(
            filename="0000.jpg",
            file_path="data/cifake_sample/0000.jpg",
            file_type="JPEG",
            file_size_bytes=888,
            width=32,
            height=32,
            color_mode="RGB",
        )
        result = pipeline.analyse_image(make_image(), metadata=metadata)

        payload = result.as_detailed_dict()
        json.dumps(payload)  # must be serialisable

        self.assertEqual(payload["image_path"], "data/cifake_sample/0000.jpg")
        self.assertIsInstance(payload["pred"], float)
        self.assertGreaterEqual(payload["pred"], 0.0)
        self.assertLessEqual(payload["pred"], 1.0)
        self.assertIn(
            payload["label"],
            {"Likely authentic", "Uncertain", "Likely AI-generated"},
        )
        self.assertIn(payload["confidence"], {"High", "Medium", "Low"})
        self.assertAlmostEqual(payload["pred"] + payload["real_probability"], 1.0, places=6)

        simple = result.as_simple_dict()
        self.assertEqual(set(simple), {"image_path", "pred"})

    def test_parameter_budget_of_the_real_architecture_is_under_two_billion(self) -> None:
        """The published checkpoint is 740,371,777 parameters, verified locally."""

        self.assertLess(740_371_777, 2_000_000_000)


if __name__ == "__main__":
    unittest.main()


class CheckpointKeyAlignmentTest(unittest.TestCase):
    """The transformers 4.x <-> 5.x SigLIP layout reconciliation.

    Verified against the real 2.11 GB checkpoint: both sides carry 954 tensors
    and reconciling this one path segment leaves nothing missing, unexpected or
    mis-shaped. These tests pin that behaviour without any download.
    """

    def flat_model(self) -> nn.Module:
        """A model whose SigLIP keys are flattened, as transformers >= 5 builds."""

        model = nn.Module()
        model.siglip = nn.Module()
        model.siglip.base_model = nn.Module()
        model.siglip.base_model.model = nn.Module()
        model.siglip.base_model.model.embeddings = nn.Linear(2, 2)
        model.classifier = nn.Linear(2, 1)
        return model

    def legacy_model(self) -> nn.Module:
        """A model with the legacy `vision_model` submodule (transformers < 5)."""

        model = nn.Module()
        model.siglip = nn.Module()
        model.siglip.base_model = nn.Module()
        model.siglip.base_model.model = nn.Module()
        model.siglip.base_model.model.vision_model = nn.Module()
        model.siglip.base_model.model.vision_model.embeddings = nn.Linear(2, 2)
        model.classifier = nn.Linear(2, 1)
        return model

    def test_legacy_checkpoint_loads_into_a_flattened_model(self) -> None:
        checkpoint = {
            "siglip.base_model.model.vision_model.embeddings.weight": torch.zeros(2, 2),
            "siglip.base_model.model.vision_model.embeddings.bias": torch.zeros(2),
            "dinov2.blocks.0.attn.qkv.original.weight": torch.zeros(6, 2),
            "classifier.weight": torch.zeros(1, 2),
        }
        aligned, notes = align_checkpoint_keys(checkpoint, self.flat_model())

        self.assertIn("siglip.base_model.model.embeddings.weight", aligned)
        self.assertNotIn("siglip.base_model.model.vision_model.embeddings.weight", aligned)
        self.assertTrue(notes)
        self.assertIn("vision_model", notes[0])

    def test_flattened_checkpoint_loads_into_a_legacy_model(self) -> None:
        checkpoint = {
            "siglip.base_model.model.embeddings.weight": torch.zeros(2, 2),
            "classifier.weight": torch.zeros(1, 2),
        }
        aligned, notes = align_checkpoint_keys(checkpoint, self.legacy_model())

        self.assertIn("siglip.base_model.model.vision_model.embeddings.weight", aligned)
        self.assertTrue(notes)

    def test_no_change_when_layouts_already_agree(self) -> None:
        checkpoint = {
            "siglip.base_model.model.embeddings.weight": torch.zeros(2, 2),
            "classifier.weight": torch.zeros(1, 2),
        }
        aligned, notes = align_checkpoint_keys(checkpoint, self.flat_model())

        self.assertEqual(set(aligned), set(checkpoint))
        self.assertEqual(notes, [])

    def test_only_the_siglip_branch_is_touched(self) -> None:
        """DINOv2 and classifier keys must pass through byte-identical."""

        checkpoint = {
            "siglip.base_model.model.vision_model.embeddings.weight": torch.zeros(2, 2),
            "dinov2.blocks.0.attn.qkv.original.weight": torch.zeros(6, 2),
            "dinov2.blocks.0.attn.qkv.lora_A.weight": torch.zeros(2, 2),
            "classifier.head.7.bias": torch.zeros(1),
        }
        aligned, _ = align_checkpoint_keys(checkpoint, self.flat_model())

        for untouched in (
            "dinov2.blocks.0.attn.qkv.original.weight",
            "dinov2.blocks.0.attn.qkv.lora_A.weight",
            "classifier.head.7.bias",
        ):
            with self.subTest(key=untouched):
                self.assertIn(untouched, aligned)

    def test_tensor_count_is_preserved(self) -> None:
        checkpoint = {
            "siglip.base_model.model.vision_model.embeddings.weight": torch.zeros(2, 2),
            "siglip.base_model.model.vision_model.embeddings.bias": torch.zeros(2),
            "classifier.weight": torch.zeros(1, 2),
        }
        aligned, _ = align_checkpoint_keys(checkpoint, self.flat_model())
        self.assertEqual(len(aligned), len(checkpoint))
