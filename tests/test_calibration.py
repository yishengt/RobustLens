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
