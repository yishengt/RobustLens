"""Stage 5 tests: checkpoint loading, error handling and the parameter limit.

Checkpoints here are UNTRAINED. These tests check loading mechanics and error
messages, never detection quality.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.pipeline.model_loader import (
    MAX_PARAMETERS,
    SUPPORTED_ARCHITECTURES,
    ModelBundle,
    ModelSetupError,
    build_architecture,
    count_parameters,
    infer_num_classes,
    load_model,
    normalise_architecture,
)
from tests.helpers import MOCK_ARCHITECTURE, base_config, write_mock_checkpoint


class ArchitectureTest(unittest.TestCase):
    def test_all_supported_architectures_build(self) -> None:
        for name in SUPPORTED_ARCHITECTURES:
            with self.subTest(architecture=name):
                model = build_architecture(name, num_classes=1)
                output = model.eval()(torch.zeros(1, 3, 224, 224))
                self.assertEqual(tuple(output.shape), (1, 1))

    def test_two_class_head_is_supported(self) -> None:
        model = build_architecture(MOCK_ARCHITECTURE, num_classes=2)
        output = model.eval()(torch.zeros(1, 3, 224, 224))
        self.assertEqual(tuple(output.shape), (1, 2))

    def test_every_architecture_is_below_the_two_billion_limit(self) -> None:
        for name in SUPPORTED_ARCHITECTURES:
            with self.subTest(architecture=name):
                total = count_parameters(build_architecture(name))
                self.assertLess(total, MAX_PARAMETERS)
                self.assertLess(total, 50_000_000)  # all are lightweight

    def test_architecture_aliases_are_normalised(self) -> None:
        self.assertEqual(normalise_architecture("EfficientNet-B0"), "efficientnet_b0")
        self.assertEqual(normalise_architecture("ResNet_18"), "resnet18")
        self.assertEqual(normalise_architecture("ConvNeXt-Tiny"), "convnext_tiny")

    def test_unsupported_architecture_is_rejected(self) -> None:
        with self.assertRaises(ModelSetupError) as context:
            build_architecture("resnet152")
        self.assertIn("Supported architectures", str(context.exception))

    def test_invalid_class_count_is_rejected(self) -> None:
        with self.assertRaises(ModelSetupError):
            build_architecture(MOCK_ARCHITECTURE, num_classes=7)

    def test_num_classes_is_inferred_from_a_state_dict(self) -> None:
        for num_classes in (1, 2):
            with self.subTest(num_classes=num_classes):
                state = build_architecture(MOCK_ARCHITECTURE, num_classes=num_classes).state_dict()
                self.assertEqual(infer_num_classes(state), num_classes)


class CheckpointLoadingFailureTest(unittest.TestCase):
    """Missing or unusable checkpoints must fail loudly, never silently."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.config = base_config()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_missing_checkpoint_raises_with_setup_guidance(self) -> None:
        with self.assertRaises(ModelSetupError) as context:
            load_model(self.tmp / "absent.pt", self.config)
        message = str(context.exception)
        self.assertIn("not found", message)
        self.assertIn("models/README.md", message)

    def test_directory_instead_of_file_raises(self) -> None:
        directory = self.tmp / "checkpoint.pt"
        directory.mkdir()
        with self.assertRaises(ModelSetupError):
            load_model(directory, self.config)

    def test_empty_checkpoint_raises(self) -> None:
        path = self.tmp / "empty.pt"
        path.write_bytes(b"")
        with self.assertRaises(ModelSetupError) as context:
            load_model(path, self.config)
        self.assertIn("empty", str(context.exception).lower())

    def test_garbage_checkpoint_raises(self) -> None:
        path = self.tmp / "garbage.pt"
        path.write_bytes(b"this is definitely not a torch checkpoint" * 20)
        with self.assertRaises(ModelSetupError) as context:
            load_model(path, self.config)
        self.assertIn("Could not read checkpoint", str(context.exception))

    def test_checkpoint_without_a_state_dict_raises(self) -> None:
        path = self.tmp / "wrong.pt"
        torch.save({"epoch": 3, "notes": "no weights here"}, path)
        with self.assertRaises(ModelSetupError) as context:
            load_model(path, self.config)
        self.assertIn("state dict", str(context.exception).lower())

    def test_architecture_mismatch_raises_actionable_error(self) -> None:
        path = write_mock_checkpoint(self.tmp / "resnet.pt", architecture="resnet18")
        config = base_config()
        config["model"]["name"] = "convnext_tiny"
        # The checkpoint records its own architecture, so strip that hint to
        # force a genuine mismatch against the configured architecture.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        torch.save({"model_state_dict": payload["model_state_dict"]}, path)

        with self.assertRaises(ModelSetupError) as context:
            load_model(path, config)
        self.assertIn("does not match architecture", str(context.exception))

    def test_parameter_limit_is_enforced(self) -> None:
        path = write_mock_checkpoint(self.tmp / "model.pt")
        config = base_config()
        config["model"]["max_parameters"] = 1000
        with self.assertRaises(ModelSetupError) as context:
            load_model(path, config)
        self.assertIn("parameter limit", str(context.exception))


class CheckpointLoadingSuccessTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.config = base_config()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_loads_and_reports_a_summary(self) -> None:
        path = write_mock_checkpoint(self.tmp / "model.pt")
        bundle = load_model(path, self.config)

        self.assertIsInstance(bundle, ModelBundle)
        self.assertEqual(bundle.architecture, MOCK_ARCHITECTURE)
        self.assertEqual(bundle.num_classes, 1)
        summary = bundle.summary()
        self.assertTrue(summary["under_2b_parameter_limit"])
        self.assertGreater(summary["num_parameters"], 0)

    def test_model_is_in_eval_mode_with_gradients_disabled(self) -> None:
        bundle = load_model(write_mock_checkpoint(self.tmp / "model.pt"), self.config)
        self.assertFalse(bundle.model.training)
        self.assertTrue(all(not p.requires_grad for p in bundle.model.parameters()))

    def test_accepts_a_bare_state_dict(self) -> None:
        path = write_mock_checkpoint(self.tmp / "bare.pt", wrap_key=None)
        self.assertEqual(load_model(path, self.config).architecture, MOCK_ARCHITECTURE)

    def test_accepts_alternative_wrapper_keys(self) -> None:
        for key in ("model_state_dict", "state_dict", "weights"):
            with self.subTest(key=key):
                path = write_mock_checkpoint(self.tmp / f"{key}.pt", wrap_key=key)
                self.assertEqual(load_model(path, self.config).num_classes, 1)

    def test_strips_dataparallel_prefixes(self) -> None:
        model = build_architecture(MOCK_ARCHITECTURE, num_classes=1)
        prefixed = {f"module.{k}": v for k, v in model.state_dict().items()}
        path = self.tmp / "prefixed.pt"
        torch.save({"model_state_dict": prefixed}, path)
        self.assertEqual(load_model(path, self.config).architecture, MOCK_ARCHITECTURE)

    def test_two_class_checkpoint_is_detected(self) -> None:
        path = write_mock_checkpoint(self.tmp / "two.pt", num_classes=2)
        self.assertEqual(load_model(path, self.config).num_classes, 2)

    def test_cpu_device_is_honoured(self) -> None:
        bundle = load_model(write_mock_checkpoint(self.tmp / "m.pt"), self.config, device="cpu")
        self.assertEqual(bundle.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
