"""Tests for calibration measured on held-out and transformed images.

The module under test exists because the shipped confidence report measures ECE
on the same images the calibrator was fitted on. These tests pin the two
properties that make the new numbers trustworthy: the held-out split is really
held out, and a condition whose calibration collapses is actually flagged.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.evaluation.calibration import ProbabilityCalibrator
from src.evaluation.calibration_robustness import (
    ECE_DEGRADATION_LIMIT,
    calibration_robustness,
    condition_calibration,
    condition_names,
)
from src.evaluation.protocol import ScoredImage


def _record(img_id: str, label: int, scores: dict) -> ScoredImage:
    return ScoredImage(
        img_id=img_id,
        source_label=label,
        class_name="ai" if label else "real",
        binary_label=label,
        version_scores=scores,
    )


def _separable(count: int = 40, shift: float = 0.0) -> list:
    """Records whose clean scores separate the classes, plus one transform."""

    rng = np.random.default_rng(7)
    records = []
    for index in range(count):
        label = index % 2
        clean = float(np.clip(rng.normal(0.8 if label else 0.2, 0.05), 0.01, 0.99))
        moved = float(np.clip(clean + shift, 0.001, 0.999))
        records.append(
            _record(f"img{index:03d}", label, {"clean": clean, "jpeg_q50": moved})
        )
    return records


class ConditionNamesTest(unittest.TestCase):
    def test_clean_is_reported_first(self) -> None:
        records = [_record("a", 1, {"jpeg_q50": 0.4, "clean": 0.9, "blur_s1": 0.5})]
        self.assertEqual(condition_names(records)[0], "clean")
        self.assertEqual(set(condition_names(records)), {"clean", "jpeg_q50", "blur_s1"})

    def test_empty_input_yields_no_conditions(self) -> None:
        self.assertEqual(condition_names([]), [])


class ConditionCalibrationTest(unittest.TestCase):
    def test_missing_condition_is_reported_not_silently_skipped(self) -> None:
        records = [
            _record("a", 1, {"clean": 0.9, "jpeg_q50": 0.8}),
            _record("b", 0, {"clean": 0.2}),
        ]
        with self.assertRaises(KeyError) as caught:
            condition_calibration(records, "jpeg_q50", None, 0.5)
        self.assertIn("b", str(caught.exception))

    def test_empty_records_rejected(self) -> None:
        with self.assertRaises(ValueError):
            condition_calibration([], "clean", None, 0.5)

    def test_score_shift_is_measured_against_clean(self) -> None:
        records = _separable(20, shift=-0.10)
        result = condition_calibration(records, "jpeg_q50", None, 0.5)
        self.assertAlmostEqual(result.mean_score_shift, -0.10, places=6)
        clean = condition_calibration(records, "clean", None, 0.5)
        self.assertAlmostEqual(clean.mean_score_shift, 0.0, places=9)

    def test_uncalibrated_column_is_the_raw_score(self) -> None:
        """Passing no calibrator must leave both ECE columns identical."""

        records = _separable(20)
        result = condition_calibration(records, "clean", None, 0.5)
        self.assertAlmostEqual(
            result.expected_calibration_error,
            result.expected_calibration_error_uncalibrated,
            places=9,
        )


class CalibrationRobustnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = _separable(40)
        labels = [r.binary_label for r in self.records]
        clean = [r.version_scores["clean"] for r in self.records]
        self.calibrator = ProbabilityCalibrator.fit(clean, labels, method="platt")

    def test_a_stable_transformation_is_not_flagged(self) -> None:
        report = calibration_robustness(self.records, self.calibrator, 0.5)
        self.assertTrue(report["calibration_holds_under_transformation"])
        self.assertLessEqual(report["ece_degradation"], ECE_DEGRADATION_LIMIT)

    def test_a_transformation_that_breaks_calibration_is_flagged(self) -> None:
        """A transform that pushes every score to the wrong side must fail."""

        broken = [
            _record(
                r.img_id,
                r.binary_label,
                {
                    "clean": r.version_scores["clean"],
                    # Invert: AI images now score low, authentic ones high.
                    "jpeg_q50": 1.0 - r.version_scores["clean"],
                },
            )
            for r in self.records
        ]
        report = calibration_robustness(broken, self.calibrator, 0.5)
        self.assertFalse(report["calibration_holds_under_transformation"])
        self.assertGreater(report["ece_degradation"], ECE_DEGRADATION_LIMIT)
        self.assertEqual(report["worst_transformed_condition"], "jpeg_q50")
        self.assertIn("degrades", report["statement"])

    def test_in_sample_optimism_is_reported_when_the_fitting_split_is_given(self) -> None:
        """The whole point: show the in-sample number beside the held-out one."""

        fitted_on = self.records[:20]
        held_out = self.records[20:]
        report = calibration_robustness(
            held_out, self.calibrator, 0.5, fitted_on=fitted_on
        )
        self.assertIsNotNone(report["clean_in_sample_ece"])
        self.assertIsNotNone(report["in_sample_optimism"])
        # Every field is rounded to 6 dp before it reaches the payload, so the
        # difference of two rounded values can differ from the rounded
        # difference by one unit in the last place.
        self.assertAlmostEqual(
            report["in_sample_optimism"],
            report["clean_held_out_ece"] - report["clean_in_sample_ece"],
            places=5,
        )

    def test_optimism_is_absent_when_no_fitting_split_is_supplied(self) -> None:
        report = calibration_robustness(self.records, self.calibrator, 0.5)
        self.assertIsNone(report["clean_in_sample_ece"])
        self.assertIsNone(report["in_sample_optimism"])

    def test_uncalibrated_run_says_so_rather_than_implying_probabilities(self) -> None:
        report = calibration_robustness(self.records, None, 0.5)
        self.assertFalse(report["calibrated"])
        self.assertIn("uncalibrated", report["statement"].lower())

    def test_every_condition_appears_exactly_once(self) -> None:
        report = calibration_robustness(self.records, self.calibrator, 0.5)
        names = [item["condition"] for item in report["conditions"]]
        self.assertEqual(names, ["clean", "jpeg_q50"])
        self.assertEqual(len(names), len(set(names)))

    def test_report_is_json_serialisable(self) -> None:
        import json

        report = calibration_robustness(self.records, self.calibrator, 0.5)
        self.assertIsInstance(json.dumps(report), str)


if __name__ == "__main__":
    unittest.main()
