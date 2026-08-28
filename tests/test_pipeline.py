"""End-to-end pipeline tests using an UNTRAINED checkpoint.

These verify wiring, output contracts and error handling. They say nothing
about detection accuracy, which requires a real trained checkpoint.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.pipeline.pipeline import DetectionPipeline, FailedResult, PipelineResult
from src.pipeline.prediction import LABEL_AI, LABEL_AUTHENTIC, LABEL_UNCERTAIN
from src.pipeline.validation import ImageValidationError
from tests.helpers import (
    base_config,
    make_image,
    write_corrupted_image,
    write_image,
    write_mock_checkpoint,
)


class PipelineTestBase(unittest.TestCase):
    """Loads one mock checkpoint for the whole class; model loading is slow."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.config = base_config()
        cls.checkpoint = write_mock_checkpoint(cls.tmp / "mock.pt")
        cls.pipeline = DetectionPipeline.from_checkpoint(
            cls.checkpoint, cls.config, device="cpu"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()


class PipelineResultTest(PipelineTestBase):
    def setUp(self) -> None:
        self.image_path = write_image(self.tmp / "images", "sample.jpg", width=160, height=120)
        self.result = self.pipeline.analyse_path(self.image_path)

    def test_returns_a_pipeline_result(self) -> None:
        self.assertIsInstance(self.result, PipelineResult)

    def test_probabilities_are_valid_and_complementary(self) -> None:
        self.assertGreaterEqual(self.result.ai_probability, 0.0)
        self.assertLessEqual(self.result.ai_probability, 1.0)
        self.assertAlmostEqual(
            self.result.ai_probability + self.result.real_probability, 1.0, places=6
        )

    def test_label_is_one_of_the_three_bands(self) -> None:
        self.assertIn(self.result.label, {LABEL_AI, LABEL_AUTHENTIC, LABEL_UNCERTAIN})

    def test_every_image_version_was_scored(self) -> None:
        # original + 14 configured transformations
        self.assertEqual(len(self.result.predictions), 15)
        self.assertTrue(self.result.predictions[0].is_original)
        for prediction in self.result.predictions:
            with self.subTest(version=prediction.name):
                self.assertGreaterEqual(prediction.ai_probability, 0.0)
                self.assertLessEqual(prediction.ai_probability, 1.0)

    def test_consistency_statistics_are_populated(self) -> None:
        consistency = self.result.consistency
        self.assertEqual(consistency.num_versions, 15)
        self.assertGreaterEqual(consistency.consistency_score, 0.0)
        self.assertLessEqual(consistency.consistency_score, 1.0)
        self.assertLessEqual(consistency.minimum, consistency.maximum)

    def test_confidence_is_reported(self) -> None:
        self.assertIn(self.result.confidence.level, {"High", "Medium", "Low"})
        self.assertIn("not proof", self.result.confidence.statement.lower())

    def test_metadata_is_attached(self) -> None:
        self.assertIsNotNone(self.result.metadata)
        self.assertEqual(self.result.metadata.filename, "sample.jpg")
        self.assertEqual((self.result.metadata.width, self.result.metadata.height), (160, 120))

    def test_original_image_is_preserved_at_full_resolution(self) -> None:
        self.assertIsNotNone(self.result.original_image)
        self.assertEqual(self.result.original_image.size, (160, 120))

    def test_explainability_runs_or_explains_itself(self) -> None:
        explanation = self.result.explanation
        self.assertIsNotNone(explanation)
        self.assertTrue(explanation.message)
        if explanation.available:
            self.assertIsNotNone(explanation.overlay)
            self.assertEqual(explanation.overlay.shape[:2], (120, 160))
        self.assertIn("prediction_comparison", explanation.charts)
        self.assertIn("confidence_components", explanation.charts)

    def test_no_errors_for_a_clean_image(self) -> None:
        self.assertEqual(self.result.errors, [])


class JsonContractTest(PipelineTestBase):
    def setUp(self) -> None:
        self.image_path = write_image(self.tmp / "images", "contract.jpg")
        self.result = self.pipeline.analyse_path(self.image_path)

    def test_simple_output_matches_the_required_format(self) -> None:
        import json

        payload = self.result.as_simple_dict()
        self.assertEqual(set(payload), {"image_path", "pred"})
        self.assertIsInstance(payload["image_path"], str)
        self.assertIsInstance(payload["pred"], float)
        self.assertGreaterEqual(payload["pred"], 0.0)
        self.assertLessEqual(payload["pred"], 1.0)
        # Must survive a JSON round trip unchanged.
        self.assertEqual(json.loads(json.dumps([payload]))[0], payload)

    def test_detailed_output_contains_every_required_field(self) -> None:
        import json

        payload = self.result.as_detailed_dict()
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
            with self.subTest(field=field):
                self.assertIn(field, payload)

        self.assertEqual(len(payload["transformations"]), 14)
        self.assertIsInstance(payload["errors"], list)
        # Whole payload must be JSON-serialisable.
        json.dumps(payload)

    def test_detailed_transformation_scores_are_probabilities(self) -> None:
        for name, value in self.result.as_detailed_dict()["transformations"].items():
            with self.subTest(transform=name):
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)


class PipelineBehaviourTest(PipelineTestBase):
    def test_analyse_bytes_matches_analyse_path(self) -> None:
        path = write_image(self.tmp / "images", "bytes.png", image_format="PNG")
        from_path = self.pipeline.analyse_path(path)
        from_bytes = self.pipeline.analyse_bytes(path.read_bytes(), "bytes.png")
        self.assertAlmostEqual(
            from_path.ai_probability, from_bytes.ai_probability, places=5
        )

    def test_analysis_is_deterministic(self) -> None:
        path = write_image(self.tmp / "images", "deterministic.jpg", seed=11)
        first = self.pipeline.analyse_path(path)
        second = self.pipeline.analyse_path(path)
        self.assertAlmostEqual(first.ai_probability, second.ai_probability, places=6)

    def test_invalid_image_raises_through_analyse_path(self) -> None:
        corrupted = write_corrupted_image(self.tmp / "images")
        with self.assertRaises(ImageValidationError):
            self.pipeline.analyse_path(corrupted)

    def test_safe_analyse_returns_a_failure_record(self) -> None:
        corrupted = write_corrupted_image(self.tmp / "images", "broken2.png")
        result = self.pipeline.safe_analyse_path(corrupted)
        self.assertIsInstance(result, FailedResult)
        payload = result.as_detailed_dict()
        self.assertIsNone(payload["pred"])
        self.assertTrue(payload["errors"])

    def test_transformations_can_be_disabled(self) -> None:
        config = base_config()
        config["transformations"]["enabled"] = False
        pipeline = DetectionPipeline.from_checkpoint(self.checkpoint, config, device="cpu")
        result = pipeline.analyse_image(make_image())

        self.assertEqual(len(result.predictions), 1)
        self.assertEqual(result.consistency.consistency_score, 1.0)
        # With no transforms the fused score is exactly the original score.
        self.assertAlmostEqual(
            result.ai_probability, result.predictions[0].ai_probability, places=6
        )

    def test_optional_frequency_module_populates_features(self) -> None:
        config = base_config()
        config["frequency"]["enabled"] = True
        pipeline = DetectionPipeline.from_checkpoint(self.checkpoint, config, device="cpu")
        result = pipeline.analyse_image(make_image())

        self.assertIsNotNone(result.frequency_features)
        self.assertIn("fft_radial_profile", result.frequency_features)
        # No frequency model is configured, so it must report that, not invent one.
        self.assertTrue(any(error["stage"] == "frequency" for error in result.errors))
        self.assertEqual(result.fusion.mode, "rgb_transform")

    def test_grayscale_image_flows_through(self) -> None:
        result = self.pipeline.analyse_image(make_image(mode="L"))
        self.assertGreaterEqual(result.ai_probability, 0.0)
        self.assertLessEqual(result.ai_probability, 1.0)

    def test_explainability_can_be_skipped(self) -> None:
        pipeline = DetectionPipeline.from_checkpoint(
            self.checkpoint, self.config, device="cpu", explain_images=False
        )
        self.assertIsNone(pipeline.analyse_image(make_image()).explanation)


if __name__ == "__main__":
    unittest.main()
