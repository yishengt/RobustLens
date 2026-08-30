"""Synthetic tests for calibration and fixed threshold selection."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.evaluation.calibration import (
    ProbabilityCalibrator,
    calibration_summary,
    search_thresholds,
)
from src.evaluation.metrics import compute_metrics
from tests.helpers import base_config, make_image, write_mock_checkpoint


class CalibrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = [0, 0, 0, 0, 1, 1, 1, 1]
        self.raw = [0.10, 0.20, 0.35, 0.45, 0.55, 0.70, 0.80, 0.90]

    def test_calibrated_output_is_a_probability(self) -> None:
        calibrator = ProbabilityCalibrator.fit(self.raw, self.labels)
        calibrated = calibrator.transform(self.raw)
        self.assertTrue(np.all(calibrated >= 0.0))
        self.assertTrue(np.all(calibrated <= 1.0))

    def test_calibration_parameters_round_trip(self) -> None:
        calibrator = ProbabilityCalibrator.fit(self.raw, self.labels)
        with tempfile.TemporaryDirectory() as directory:
            path = calibrator.save(Path(directory) / "calibration.json")
            loaded = ProbabilityCalibrator.load(path)
        np.testing.assert_allclose(loaded.transform(self.raw), calibrator.transform(self.raw))

    def test_threshold_search_covers_expected_grid(self) -> None:
        selection = search_thresholds(self.labels, self.raw)
        self.assertEqual(len(selection.curve), 99)
        self.assertEqual(selection.curve[0]["threshold"], 0.01)
        self.assertEqual(selection.curve[-1]["threshold"], 0.99)
        self.assertTrue(0.0 < selection.balanced < 1.0)
        self.assertTrue(0.0 < selection.low_false_positive < 1.0)
        self.assertTrue(0.0 < selection.high_recall < 1.0)

    def test_fixed_threshold_is_not_retuned_by_transformation(self) -> None:
        selection = search_thresholds(self.labels, self.raw)
        fixed = selection.balanced
        clean = compute_metrics(self.labels, self.raw, fixed).threshold
        transformed = compute_metrics(self.labels, list(reversed(self.raw)), fixed).threshold
        self.assertEqual(clean, fixed)
        self.assertEqual(transformed, fixed)

    def test_calibration_summary_has_ten_confidence_bins(self) -> None:
        summary = calibration_summary(self.labels, self.raw)
        self.assertEqual(len(summary["accuracy_by_confidence_bin"]), 10)
        self.assertGreaterEqual(summary["expected_calibration_error"], 0.0)
        self.assertGreaterEqual(summary["brier_score"], 0.0)

    def test_fit_is_explicitly_clean_validation_only(self) -> None:
        calibrator = ProbabilityCalibrator.fit(self.raw, self.labels)
        self.assertEqual(calibrator.fitted_on, "clean_validation")
        self.assertNotIn("test", calibrator.fitted_on.lower())
        self.assertNotIn("demo", calibrator.fitted_on.lower())

    def test_serialised_calibration_is_json(self) -> None:
        calibrator = ProbabilityCalibrator.fit(self.raw, self.labels)
        payload = json.loads(json.dumps(calibrator.as_dict()))
        self.assertEqual(payload["method"], "platt")


if __name__ == "__main__":
    unittest.main()


class TemperatureScalingTest(unittest.TestCase):
    """Temperature scaling: one parameter, no decision-boundary shift."""

    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.labels = np.concatenate([np.zeros(200), np.ones(200)]).astype(int)
        raw = np.clip(np.concatenate([rng.beta(2, 5, 200), rng.beta(5, 2, 200)]), 1e-4, 1 - 1e-4)
        # Sharpen into over-confidence, the regime temperature scaling targets.
        self.raw = 1.0 / (1.0 + np.exp(-3.0 * np.log(raw / (1.0 - raw))))

    def test_temperature_fit_pins_the_intercept_at_zero(self) -> None:
        calibrator = ProbabilityCalibrator.fit(self.raw, self.labels, method="temperature")
        self.assertEqual(calibrator.method, "temperature")
        self.assertEqual(calibrator.bias, 0.0)
        self.assertGreater(calibrator.scale, 0.0)

    def test_temperature_property_is_the_inverse_slope(self) -> None:
        calibrator = ProbabilityCalibrator.fit(self.raw, self.labels, method="temperature")
        self.assertAlmostEqual(calibrator.temperature, 1.0 / calibrator.scale, places=9)

    def test_platt_has_no_temperature(self) -> None:
        self.assertIsNone(ProbabilityCalibrator.fit(self.raw, self.labels).temperature)

    def test_temperature_preserves_ranking(self) -> None:
        """A positive slope with zero intercept cannot reorder scores."""

        calibrator = ProbabilityCalibrator.fit(self.raw, self.labels, method="temperature")
        calibrated = calibrator.transform(self.raw)
        self.assertTrue(
            np.array_equal(np.argsort(np.argsort(self.raw)), np.argsort(np.argsort(calibrated)))
        )

    def test_both_methods_improve_badly_calibrated_scores(self) -> None:
        """The real case: scores that are genuinely miscalibrated."""

        # Push the scores hard toward 0/1 so they are clearly over-confident.
        skewed = 1.0 / (1.0 + np.exp(-6.0 * np.log(self.raw / (1.0 - self.raw))))
        before = calibration_summary(self.labels, skewed)
        for method in ("platt", "temperature"):
            with self.subTest(method=method):
                calibrator = ProbabilityCalibrator.fit(skewed, self.labels, method=method)
                after = calibration_summary(self.labels, calibrator.transform(skewed))
                self.assertLess(
                    after["expected_calibration_error"], before["expected_calibration_error"]
                )

    def test_calibrating_already_good_scores_costs_little(self) -> None:
        """Platt target smoothing shifts predictions slightly toward the centre.

        On input that is already well calibrated that costs a small amount of
        ECE. It is accepted deliberately: without smoothing, separable
        validation data collapses the fit into a 0/1 step function, which makes
        every decision threshold equivalent.
        """

        before = calibration_summary(self.labels, self.raw)
        for method in ("platt", "temperature"):
            with self.subTest(method=method):
                calibrator = ProbabilityCalibrator.fit(self.raw, self.labels, method=method)
                after = calibration_summary(self.labels, calibrator.transform(self.raw))
                self.assertLess(
                    after["expected_calibration_error"],
                    before["expected_calibration_error"] + 0.01,
                )

    def test_unknown_method_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ProbabilityCalibrator.fit(self.raw, self.labels, method="isotonic")
        with self.assertRaises(ValueError):
            ProbabilityCalibrator(method="isotonic")

    def test_temperature_survives_a_save_load_round_trip(self) -> None:
        calibrator = ProbabilityCalibrator.fit(self.raw, self.labels, method="temperature")
        with tempfile.TemporaryDirectory() as tmp:
            path = calibrator.save(Path(tmp) / "calibration.json")
            restored = ProbabilityCalibrator.load(path)
        self.assertEqual(restored.method, "temperature")
        self.assertAlmostEqual(restored.scale, calibrator.scale, places=9)
        self.assertAlmostEqual(restored.temperature, calibrator.temperature, places=9)


class OperatingPointTest(unittest.TestCase):
    """All four operating points, selected from clean validation scores."""

    def setUp(self) -> None:
        rng = np.random.default_rng(1)
        self.labels = np.concatenate([np.zeros(300), np.ones(150)]).astype(int)
        self.scores = np.clip(np.concatenate([rng.beta(2, 6, 300), rng.beta(6, 2, 150)]), 0.0, 1.0)
        self.selection = search_thresholds(self.labels, self.scores, 0.01)

    def test_all_four_points_are_selected(self) -> None:
        for name in ("balanced", "f1_optimal", "low_false_positive", "high_recall"):
            with self.subTest(point=name):
                self.assertIn(name, self.selection.metrics)

    def test_f1_optimal_really_maximises_f1(self) -> None:
        best = max(row["f1"] for row in self.selection.curve)
        self.assertAlmostEqual(self.selection.metrics["f1_optimal"]["f1"], best, places=9)

    def test_low_fpr_point_meets_its_target(self) -> None:
        self.assertTrue(self.selection.target_met)
        self.assertLessEqual(
            self.selection.metrics["low_false_positive"]["false_positive_rate"], 0.01
        )

    def test_high_recall_point_has_the_highest_recall(self) -> None:
        best = max(row["recall"] for row in self.selection.curve)
        self.assertAlmostEqual(self.selection.metrics["high_recall"]["recall"], best, places=9)

    def test_low_fpr_threshold_is_stricter_than_high_recall(self) -> None:
        self.assertGreater(self.selection.low_false_positive, self.selection.high_recall)

    def test_threshold_grid_spans_one_to_ninety_nine(self) -> None:
        thresholds = [row["threshold"] for row in self.selection.curve]
        self.assertEqual(len(thresholds), 99)
        self.assertAlmostEqual(min(thresholds), 0.01, places=6)
        self.assertAlmostEqual(max(thresholds), 0.99, places=6)

    def test_every_required_metric_is_reported_per_threshold(self) -> None:
        for field in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "balanced_accuracy",
            "false_positive_rate",
            "false_negative_rate",
            "youden_j",
        ):
            with self.subTest(metric=field):
                self.assertIn(field, self.selection.curve[0])

    def test_f1_optimal_is_serialised(self) -> None:
        self.assertIn("f1_optimal_threshold", self.selection.as_dict())


class CalibrationStatusTest(unittest.TestCase):
    """The demo must never blur a raw score with a calibrated probability."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.checkpoint = write_mock_checkpoint(cls.tmp / "mock.pt")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def build(self, config):
        from src.pipeline.pipeline import DetectionPipeline

        return DetectionPipeline.from_checkpoint(
            self.checkpoint, config, device="cpu", explain_images=False
        )

    def test_uncalibrated_state_is_declared(self) -> None:
        config = base_config()
        config["transformations"]["enabled"] = False
        config["patches"] = {"enabled": False}
        status = self.build(config).calibration_status()

        self.assertFalse(status["calibrated"])
        self.assertEqual(status["probability_kind"], "uncalibrated model score")
        self.assertEqual(status["threshold_source"], "interface default")
        self.assertEqual(status["label_bands_source"], "interface default")
        self.assertIsNone(status["method"])

        detailed = self.build(config).analyse_image(make_image()).as_detailed_dict()
        self.assertIsNone(detailed["calibrated_probability"])
        self.assertEqual(detailed["probability_kind"], "fused uncalibrated model score")

    def test_calibrated_state_reports_data_derived_threshold(self) -> None:
        rng = np.random.default_rng(3)
        labels = np.concatenate([np.zeros(80), np.ones(80)]).astype(int)
        scores = np.clip(np.concatenate([rng.beta(2, 6, 80), rng.beta(6, 2, 80)]), 0, 1)
        calibrator = ProbabilityCalibrator.fit(scores, labels)
        selection = search_thresholds(labels, calibrator.transform(scores), 0.01)
        stored = ProbabilityCalibrator(
            method=calibrator.method,
            input_type=calibrator.input_type,
            scale=calibrator.scale,
            bias=calibrator.bias,
            fitted_on="clean_validation",
            selected_thresholds={
                "balanced": selection.balanced,
                "f1_optimal": selection.f1_optimal,
                "low_false_positive": selection.low_false_positive,
                "high_recall": selection.high_recall,
            },
        )
        path = stored.save(self.tmp / "calibration.json")

        config = base_config()
        config["transformations"]["enabled"] = False
        config["patches"] = {"enabled": False}
        config["calibration"] = {
            "enabled": True,
            "path": str(path),
            "use_selected_threshold": True,
            "operating_point": "f1_optimal",
            "uncertainty_margin": 0.1,
        }
        status = self.build(config).calibration_status()

        self.assertTrue(status["calibrated"])
        self.assertEqual(status["probability_kind"], "calibrated probability")
        self.assertIn("data-derived", status["threshold_source"])
        self.assertIn("f1_optimal", status["threshold_source"])
        self.assertAlmostEqual(status["threshold"], selection.f1_optimal, places=6)
        self.assertIn("data-derived", status["label_bands_source"])

        detailed = self.build(config).analyse_image(make_image()).as_detailed_dict()
        self.assertIsNotNone(detailed["calibrated_probability"])
        self.assertEqual(
            detailed["probability_kind"],
            "fused score from calibrated per-view probabilities",
        )

    def test_unknown_operating_point_falls_back_to_balanced(self) -> None:
        rng = np.random.default_rng(4)
        labels = np.concatenate([np.zeros(60), np.ones(60)]).astype(int)
        scores = np.clip(np.concatenate([rng.beta(2, 6, 60), rng.beta(6, 2, 60)]), 0, 1)
        selection = search_thresholds(labels, scores, 0.01)
        stored = ProbabilityCalibrator(
            fitted_on="clean_validation",
            selected_thresholds={"balanced": selection.balanced},
        )
        path = stored.save(self.tmp / "calibration_fallback.json")

        config = base_config()
        config["transformations"]["enabled"] = False
        config["patches"] = {"enabled": False}
        config["calibration"] = {
            "enabled": True,
            "path": str(path),
            "use_selected_threshold": True,
            "operating_point": "does_not_exist",
            "uncertainty_margin": 0.1,
        }
        status = self.build(config).calibration_status()
        self.assertAlmostEqual(status["threshold"], selection.balanced, places=6)
