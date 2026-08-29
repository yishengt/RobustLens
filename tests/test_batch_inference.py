"""Batch inference tests: JSON output formats, error handling and the CLI."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.inference.batch_inference import run_batch, write_json
from src.pipeline.model_loader import ModelSetupError
from src.pipeline.validation import ImageValidationError
from tests.helpers import (
    base_config,
    write_corrupted_image,
    write_image,
    write_mock_checkpoint,
)


class BatchInferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.config = base_config()
        cls.checkpoint = write_mock_checkpoint(cls.tmp / "mock.pt")

        cls.images = cls.tmp / "images"
        write_image(cls.images, "one.jpg", seed=1)
        write_image(cls.images, "two.png", seed=2, image_format="PNG")
        write_image(cls.images / "nested", "three.webp", seed=3, image_format="WEBP")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def run_batch(self, **overrides):
        options = {
            "image_dir": self.images,
            "checkpoint_path": self.checkpoint,
            "config": self.config,
            "device": "cpu",
        }
        options.update(overrides)
        return run_batch(**options)

    def test_processes_every_supported_image_recursively(self) -> None:
        report = self.run_batch()
        self.assertEqual(report.total, 3)
        self.assertEqual(report.processed, 3)
        self.assertEqual(report.failed, 0)

    def test_simple_output_matches_the_required_contract(self) -> None:
        output = self.tmp / "out" / "predictions.json"
        self.run_batch(output_path=output)

        payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), 3)
        for record in payload:
            with self.subTest(record=record["image_path"]):
                self.assertEqual(set(record), {"image_path", "pred"})
                self.assertIsInstance(record["pred"], float)
                self.assertGreaterEqual(record["pred"], 0.0)
                self.assertLessEqual(record["pred"], 1.0)

    def test_detailed_output_contains_the_documented_fields(self) -> None:
        detailed_path = self.tmp / "out" / "detailed.json"
        self.run_batch(detailed_output_path=detailed_path)

        payload = json.loads(detailed_path.read_text(encoding="utf-8"))
        self.assertEqual(len(payload), 3)
        for record in payload:
            with self.subTest(record=record["image_path"]):
                for field in [
                    "image_path",
                    "pred",
                    "label",
                    "confidence",
                    "real_probability",
                    "transform_consistency",
                    "transformations",
                    "errors",
                ]:
                    self.assertIn(field, record)
                self.assertEqual(len(record["transformations"]), 14)

    def test_limit_restricts_the_number_of_images(self) -> None:
        report = self.run_batch(limit=2)
        self.assertEqual(report.processed, 2)

    def test_relative_paths_are_written_when_requested(self) -> None:
        report = self.run_batch(relative_to=self.images)
        paths = sorted(record["image_path"] for record in report.simple)
        self.assertEqual(paths, ["nested/three.webp", "one.jpg", "two.png"])

    def test_missing_directory_raises(self) -> None:
        with self.assertRaises(ImageValidationError):
            self.run_batch(image_dir=self.tmp / "does_not_exist")

    def test_directory_without_images_raises(self) -> None:
        empty = self.tmp / "empty"
        empty.mkdir(exist_ok=True)
        with self.assertRaises(ImageValidationError):
            self.run_batch(image_dir=empty)

    def test_missing_checkpoint_raises_model_setup_error(self) -> None:
        with self.assertRaises(ModelSetupError):
            self.run_batch(checkpoint_path=self.tmp / "absent.pt")

    def test_invalid_on_error_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.run_batch(on_error="explode")

    def test_out_of_range_fallback_pred_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.run_batch(on_error="fallback", fallback_pred=1.5)


class BatchErrorHandlingTest(unittest.TestCase):
    """A corrupted file must not abort the run."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.config = base_config()
        cls.checkpoint = write_mock_checkpoint(cls.tmp / "mock.pt")
        cls.images = cls.tmp / "mixed"
        write_image(cls.images, "good.jpg", seed=1)
        write_corrupted_image(cls.images, "bad.png")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_skip_mode_omits_failures_from_the_simple_output(self) -> None:
        report = run_batch(
            image_dir=self.images,
            checkpoint_path=self.checkpoint,
            config=self.config,
            device="cpu",
            on_error="skip",
        )
        self.assertEqual(report.processed, 1)
        self.assertEqual(report.failed, 1)
        self.assertEqual(len(report.simple), 1)
        self.assertEqual(len(report.detailed), 2)
        self.assertTrue(report.errors)

    def test_fallback_mode_records_a_neutral_score(self) -> None:
        report = run_batch(
            image_dir=self.images,
            checkpoint_path=self.checkpoint,
            config=self.config,
            device="cpu",
            on_error="fallback",
            fallback_pred=0.5,
        )
        self.assertEqual(len(report.simple), 2)
        fallback = [r for r in report.simple if r["image_path"].endswith("bad.png")]
        self.assertEqual(len(fallback), 1)
        self.assertEqual(fallback[0]["pred"], 0.5)

    def test_failure_record_carries_the_error_message(self) -> None:
        report = run_batch(
            image_dir=self.images,
            checkpoint_path=self.checkpoint,
            config=self.config,
            device="cpu",
        )
        failed = [r for r in report.detailed if r["image_path"].endswith("bad.png")][0]
        self.assertIsNone(failed["pred"])
        self.assertTrue(failed["errors"])
        self.assertIn("corrupted", failed["errors"][0]["error"].lower())


class WriteJsonTest(unittest.TestCase):
    def test_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "deep" / "nested" / "out.json"
            write_json([{"image_path": "a.jpg", "pred": 0.5}], target)
            self.assertTrue(target.is_file())
            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")),
                [{"image_path": "a.jpg", "pred": 0.5}],
            )


class CommandLineInterfaceTest(unittest.TestCase):
    """The CLI must parse arguments without importing the ML stack."""

    def test_parser_builds_and_requires_input_dir(self) -> None:
        import scripts.run_inference as cli

        parser = cli.build_parser()
        args = parser.parse_args(["--input-dir", "images"])
        self.assertEqual(args.input_dir, "images")
        self.assertEqual(args.output, "outputs/predictions.json")
        self.assertEqual(args.checkpoint, "checkpoints/best.pt")

        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_optional_flags_are_accepted(self) -> None:
        import scripts.run_inference as cli

        args = cli.build_parser().parse_args(
            [
                "--input-dir", "images",
                "--detailed-output", "out/detailed.json",
                "--device", "cpu",
                "--no-transformations",
                "--limit", "5",
                "--on-error", "fallback",
            ]
        )
        self.assertEqual(args.detailed_output, "out/detailed.json")
        self.assertEqual(args.device, "cpu")
        self.assertTrue(args.no_transformations)
        self.assertEqual(args.limit, 5)
        self.assertEqual(args.on_error, "fallback")

    def test_missing_checkpoint_returns_exit_code_three(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_image(root / "images", "a.jpg")
            import scripts.run_inference as cli

            code = cli.main(
                [
                    "--input-dir", str(root / "images"),
                    "--checkpoint", str(root / "absent.pt"),
                    "--output", str(root / "out.json"),
                    "--quiet",
                ]
            )
            self.assertEqual(code, 3)


if __name__ == "__main__":
    unittest.main()
