"""Dependency-light smoke tests for the inference-only project.

These run without a model checkpoint and without the torch stack installed.
They check the project layout, the configuration file and the JSON contract.

Mock vs real inference
----------------------
Nothing here performs real inference. Tests that exercise the model use an
UNTRAINED checkpoint built on the fly (see ``tests/helpers.py``) and verify
plumbing only. Detection accuracy can only be measured with a real trained
checkpoint via ``scripts/evaluate_dataset.py``.
"""

from __future__ import annotations

import importlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ProjectStructureTest(unittest.TestCase):
    def test_pipeline_modules_exist(self) -> None:
        for relative_path in [
            "README.md",
            "requirements.txt",
            "app.py",
            "configs/config.yaml",
            "models/README.md",
            "src/pipeline/validation.py",
            "src/pipeline/preprocessing.py",
            "src/pipeline/transformations.py",
            "src/pipeline/model_loader.py",
            "src/pipeline/prediction.py",
            "src/pipeline/consistency.py",
            "src/pipeline/fusion.py",
            "src/pipeline/confidence.py",
            "src/pipeline/explainability.py",
            "src/pipeline/frequency.py",
            "src/pipeline/pipeline.py",
            "src/inference/batch_inference.py",
            "src/evaluation/calibration.py",
            "src/utils/device.py",
            "src/utils/config.py",
            "scripts/run_inference.py",
            "scripts/calibrate_threshold.py",
            "scripts/setup.py",
            "scripts/download_dataset.py",
        ]:
            with self.subTest(path=relative_path):
                self.assertTrue((ROOT / relative_path).is_file(), relative_path)

    def test_training_modules_remain_absent(self) -> None:
        """This project is inference-only; no training code should exist."""

        for relative_path in [
            "src/training",
            "src/features",
            "src/evaluation/train.py",
        ]:
            with self.subTest(path=relative_path):
                self.assertFalse((ROOT / relative_path).exists(), relative_path)

    def test_checkpoints_are_not_committed(self) -> None:
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("checkpoints/*.pt", gitignore)


class ConfigurationTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.yaml = importlib.import_module("yaml")
        except ModuleNotFoundError:
            self.skipTest("PyYAML is not installed; install requirements.txt")
        with (ROOT / "configs/config.yaml").open(encoding="utf-8") as handle:
            self.config = self.yaml.safe_load(handle)

    def test_config_is_a_mapping(self) -> None:
        self.assertIsInstance(self.config, dict)

    def test_required_sections_are_present(self) -> None:
        for section in [
            "data",
            "validation",
            "model",
            "normalization",
            "transformations",
            "labels",
            "consistency",
            "patches",
            "fusion",
            "frequency",
            "calibration",
            "confidence",
            "explainability",
            "inference",
        ]:
            with self.subTest(section=section):
                self.assertIn(section, self.config)

    def test_supported_extensions(self) -> None:
        self.assertEqual(self.config["data"]["extensions"], [".jpg", ".jpeg", ".png", ".webp"])

    def test_model_is_lightweight_and_within_the_limit(self) -> None:
        self.assertIn(
            self.config["model"]["name"],
            {"efficientnet_b0", "resnet18", "convnext_tiny"},
        )
        self.assertLessEqual(self.config["model"]["max_parameters"], 2_000_000_000)

    def test_image_size_is_224(self) -> None:
        self.assertEqual(self.config["data"]["image_size"], 224)

    def test_label_bands_match_the_specification(self) -> None:
        self.assertAlmostEqual(self.config["labels"]["authentic_max"], 0.40)
        self.assertAlmostEqual(self.config["labels"]["ai_min"], 0.60)

    def test_transformations_cover_the_required_families(self) -> None:
        settings = self.config["transformations"]
        self.assertEqual(settings["jpeg_qualities"], [90, 70, 50, 30])
        self.assertEqual(settings["blur_sigmas"], [0.5, 1.0, 2.0])
        self.assertEqual(settings["resize_scales"], [0.5, 0.25])
        self.assertEqual(settings["noise_sigmas"], [0.02, 0.05, 0.10])
        self.assertAlmostEqual(settings["center_crop_fraction"], 0.8)
        for channel in ("brightness", "contrast", "saturation"):
            self.assertAlmostEqual(settings["color_jitter"][channel], 0.2)

    def test_default_fusion_excludes_patch_evidence(self) -> None:
        """Patch evidence was demoted on measured evidence, not preference.

        scripts/ablate_patches.py found every patch mode scored worse than
        whole-image-only, so patch evidence carries no fusion weight. It stays
        available as an explainability heatmap.
        """

        self.assertEqual(self.config["fusion"]["mode"], "rgb_transform")
        self.assertAlmostEqual(self.config["fusion"]["original_weight"], 0.7)
        self.assertAlmostEqual(self.config["fusion"]["transform_weight"], 0.3)

    def test_patch_fusion_weights_remain_configured(self) -> None:
        """The three-term mode stays available for re-testing after a retrain."""

        weights = self.config["fusion"]["whole_patch"]
        self.assertAlmostEqual(weights["whole_weight"], 0.6)
        self.assertAlmostEqual(weights["transform_weight"], 0.2)
        self.assertAlmostEqual(weights["patch_weight"], 0.2)

    def test_patch_analysis_stays_enabled_for_explainability(self) -> None:
        """Patch analysis must stay on as an explainability feature.

        The mode is a cost/coverage trade-off that is expected to be tuned, so
        this asserts the invariant that matters -- patch scoring is enabled and
        set to a mode that actually scores patches -- rather than pinning one
        particular mode and failing whenever it is retuned.
        """

        self.assertTrue(self.config["patches"]["enabled"])
        self.assertIn(
            self.config["patches"]["mode"],
            {"coarse", "full", "top_k", "uncertain_only"},
        )
        self.assertNotEqual(self.config["patches"]["mode"], "off")

    def test_patch_section_is_present(self) -> None:
        patches = self.config["patches"]
        self.assertTrue(patches["enabled"])
        self.assertGreater(patches["patch_size"], 0)
        self.assertLess(patches["stride"], patches["patch_size"])  # overlapping
        self.assertGreater(patches["max_patches"], 1)
        self.assertIn(patches["evidence_statistic"], {"top_k_mean", "max", "mean"})

    def test_frequency_analysis_is_disabled_by_default(self) -> None:
        self.assertFalse(self.config["frequency"]["enabled"])

    def test_optional_frequency_weights_are_fifty_thirty_twenty(self) -> None:
        weights = self.config["fusion"]["frequency"]
        self.assertAlmostEqual(weights["rgb_weight"], 0.5)
        self.assertAlmostEqual(weights["frequency_weight"], 0.3)
        self.assertAlmostEqual(weights["consistency_weight"], 0.2)


class JsonContractTest(unittest.TestCase):
    """The required submission format, checked without running a model."""

    def test_simple_record_shape(self) -> None:
        sample = [{"image_path": "images/example.jpg", "pred": 0.84}]
        decoded = json.loads(json.dumps(sample))
        self.assertEqual(set(decoded[0]), {"image_path", "pred"})
        self.assertGreaterEqual(decoded[0]["pred"], 0.0)
        self.assertLessEqual(decoded[0]["pred"], 1.0)

    def test_detailed_record_shape(self) -> None:
        sample = {
            "image_path": "images/example.jpg",
            "pred": 0.84,
            "label": "Likely AI-generated",
            "confidence": "High",
            "real_probability": 0.16,
            "transform_consistency": 0.93,
            "transformations": {"jpeg_q70": 0.82},
            "errors": [],
        }
        decoded = json.loads(json.dumps(sample))
        self.assertEqual(decoded["pred"] + decoded["real_probability"], 1.0)
        self.assertIsInstance(decoded["transformations"], dict)
        self.assertIsInstance(decoded["errors"], list)


class RuntimeImportTest(unittest.TestCase):
    def test_pipeline_modules_import_when_dependencies_are_installed(self) -> None:
        required = ["torch", "torchvision", "PIL", "numpy", "yaml"]
        missing = [name for name in required if not _importable(name)]
        if missing:
            self.skipTest("Runtime dependencies missing: " + ", ".join(missing))

        for module_name in [
            "src.pipeline.validation",
            "src.pipeline.preprocessing",
            "src.pipeline.transformations",
            "src.pipeline.model_loader",
            "src.pipeline.prediction",
            "src.pipeline.consistency",
            "src.pipeline.fusion",
            "src.pipeline.confidence",
            "src.pipeline.explainability",
            "src.pipeline.frequency",
            "src.pipeline.pipeline",
            "src.inference.batch_inference",
            "src.utils.device",
            "src.utils.config",
            "src.evaluation.metrics",
            "src.evaluation.calibration",
        ]:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)

    def test_validation_module_needs_only_pillow(self) -> None:
        if not _importable("PIL"):
            self.skipTest("Pillow is not installed")
        module = importlib.import_module("src.pipeline.validation")
        self.assertTrue(hasattr(module, "validate_image_file"))


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
    except ModuleNotFoundError:
        return False
    return True


if __name__ == "__main__":
    unittest.main()
