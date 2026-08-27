"""Small checks that run before or after installing the ML dependencies."""

from __future__ import annotations

import importlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FoundationSmokeTest(unittest.TestCase):
    def test_expected_structure_exists(self) -> None:
        expected = [
            "README.md",
            "requirements.txt",
            "configs/config.yaml",
            "src/data/dataset.py",
            "src/data/augmentations.py",
            "src/models/classifier.py",
            "src/training/train.py",
            "src/evaluation/evaluate.py",
            "src/evaluation/robustness_test.py",
            "src/evaluation/error_analysis.py",
            "src/evaluation/materialize_transforms.py",
            "src/evaluation/materialized.py",
            "src/inference/predict.py",
            "src/features/engineering.py",
            "src/explainability/gradcam.py",
            "scripts/run_inference.py",
            "app.py",
        ]
        for relative_path in expected:
            self.assertTrue((ROOT / relative_path).is_file(), relative_path)

    def test_config_is_valid_yaml_when_dependency_is_available(self) -> None:
        try:
            yaml = importlib.import_module("yaml")
        except ModuleNotFoundError:
            self.skipTest("PyYAML is not installed; install requirements.txt for runtime checks")
        with (ROOT / "configs/config.yaml").open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        self.assertEqual(config["model"]["name"], "efficientnet_b0")
        self.assertEqual(config["training"]["threshold"], 0.5)

    def test_inference_json_contract(self) -> None:
        sample = [{"image_path": "path/to/image.jpg", "pred": 0.91}]
        encoded = json.dumps(sample)
        decoded = json.loads(encoded)
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
            "sklearn",
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
            self.skipTest(
                "Runtime dependencies are not installed: "
                + ", ".join(missing)
            )
        for module_name in [
            "src.data.dataset",
            "src.data.augmentations",
            "src.models.classifier",
            "src.training.train",
            "src.evaluation.evaluate",
            "src.evaluation.robustness_test",
            "src.evaluation.error_analysis",
            "src.evaluation.materialize_transforms",
            "src.evaluation.materialized",
            "src.inference.predict",
            "src.features.engineering",
            "src.explainability.gradcam",
        ]:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
