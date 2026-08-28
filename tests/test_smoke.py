"""Dependency-light checks for the inference-only project layout."""

from __future__ import annotations

import importlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class InferenceSmokeTest(unittest.TestCase):
    def test_expected_structure_exists(self) -> None:
        expected = [
            "README.md",
            "requirements.txt",
            "configs/config.yaml",
            "src/data/dataset.py",
            "src/data/augmentations.py",
            "src/models/classifier.py",
            "src/utils/checkpoint.py",
            "src/utils/config.py",
            "src/inference/predict.py",
            "src/explainability/gradcam.py",
            "scripts/run_inference.py",
            "app.py",
        ]
        for relative_path in expected:
            self.assertTrue((ROOT / relative_path).is_file(), relative_path)

    def test_removed_training_and_evaluation_modules_are_absent(self) -> None:
        for relative_path in [
            "src/training",
            "src/evaluation",
            "src/features",
            "src/utils/seed.py",
        ]:
            self.assertFalse((ROOT / relative_path).exists(), relative_path)

    def test_config_is_valid_yaml_when_dependency_is_available(self) -> None:
        try:
            yaml = importlib.import_module("yaml")
        except ModuleNotFoundError:
            self.skipTest("PyYAML is not installed; install requirements.txt for runtime checks")
        with (ROOT / "configs/config.yaml").open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        self.assertEqual(config["model"]["name"], "efficientnet_b0")
        self.assertEqual(config["inference"]["threshold"], 0.5)
        self.assertEqual(config["data"]["extensions"], [".jpg", ".jpeg", ".png", ".webp"])

    def test_inference_json_contract(self) -> None:
        sample = [{"image_path": "path/to/image.jpg", "pred": 0.91}]
        decoded = json.loads(json.dumps(sample))
        self.assertEqual(set(decoded[0]), {"image_path", "pred"})
        self.assertGreaterEqual(decoded[0]["pred"], 0.0)
        self.assertLessEqual(decoded[0]["pred"], 1.0)

    def test_runtime_imports_when_dependencies_are_available(self) -> None:
        required = [
            "torch",
            "torchvision",
            "albumentations",
            "PIL",
            "cv2",
            "numpy",
            "yaml",
            "streamlit",
        ]
        missing = []
        for module_name in required:
            try:
                importlib.import_module(module_name)
            except ModuleNotFoundError:
                missing.append(module_name)
        if missing:
            self.skipTest("Runtime dependencies are not installed: " + ", ".join(missing))
        for module_name in [
            "src.data.dataset",
            "src.data.augmentations",
            "src.models.classifier",
            "src.utils.checkpoint",
            "src.utils.config",
            "src.inference.predict",
            "src.explainability.gradcam",
        ]:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
