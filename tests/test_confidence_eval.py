"""Tests for the confidence ablation and the decisions it drove.

No model, no downloads: the cached ``ScoredImage`` records are synthesised.
These pin the properties that make the ablation trustworthy -- that confidence
never touches the predicted probability, that the pre-registered rule is
applied as written, and that patch evidence carries zero fusion weight.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.evaluation.confidence_eval import (
    MIN_AUROC_GAIN,
    VARIANT_GATED,
    VARIANT_WITH,
    VARIANT_WITHOUT,
    decide,
    evaluate_variant,
    patch_agreement_for,
)
from src.evaluation.protocol import CLEAN_KEY, ScoredImage
from src.pipeline.confidence import compute_confidence
from src.pipeline.fusion import fuse_predictions

CONFIG = {
    "fusion": {"mode": "rgb_transform", "original_weight": 0.7, "transform_weight": 0.3},
    "confidence": {
        "decisiveness_weight": 0.4,
        "agreement_weight": 0.3,
        "consistency_weight": 0.3,
        "patch_agreement_weight": 0.0,
        "high_min": 0.70,
        "medium_min": 0.45,
    },
    "labels": {"authentic_max": 0.40, "ai_min": 0.60},
    "inference": {"threshold": 0.5},
    "consistency": {
        "std_reference": 0.20,
        "range_reference": 0.60,
        "std_weight": 0.5,
        "range_weight": 0.5,
    },
}

TRANSFORMS = ["jpeg_q90", "jpeg_q50", "blur_s1", "center_crop_80"]


def record(index: int, label: int, clean: float, patch_evidence=None, agreement=1.0):
    scores = {CLEAN_KEY: clean}
    scores.update(dict.fromkeys(TRANSFORMS, clean))
    return ScoredImage(
        img_id=f"i{index:03d}",
        source_label=label,
        class_name="real" if label == 0 else "full_synthetic",
        binary_label=label,
        version_scores=scores,
        patch_evidence=patch_evidence,
        patch_agreement=None if patch_evidence is None else agreement,
        patch_available=patch_evidence is not None,
        num_patches=0 if patch_evidence is None else 12,
    )


def population(n: int = 24):
    out = []
    for i in range(n // 2):
        out.append(record(i, 0, 0.05 + 0.01 * i, patch_evidence=0.30))
    for i in range(n // 2):
        out.append(record(100 + i, 1, 0.95 - 0.01 * i, patch_evidence=0.90))
    return out


class PatchAgreementSelectionTest(unittest.TestCase):
    def test_variant_a_never_uses_patch_agreement(self) -> None:
        self.assertIsNone(patch_agreement_for(record(1, 1, 0.9, 0.9), VARIANT_WITHOUT))

    def test_variant_b_always_uses_it_when_available(self) -> None:
        self.assertAlmostEqual(
            patch_agreement_for(record(1, 1, 0.9, 0.9, agreement=0.8), VARIANT_WITH), 0.8
        )

    def test_variant_b_is_none_when_patches_unavailable(self) -> None:
        self.assertIsNone(patch_agreement_for(record(1, 1, 0.9, None), VARIANT_WITH))

    def test_variant_c_distrusts_patch_evidence_that_contradicts_the_whole_image(self) -> None:
        """The measured failure mode: patch evidence inflated on authentic images."""

        conflicting = record(1, 0, clean=0.05, patch_evidence=0.60)  # gap 0.55
        self.assertIsNone(patch_agreement_for(conflicting, VARIANT_GATED, margin=0.25))

    def test_variant_c_accepts_agreeing_patch_evidence(self) -> None:
        agreeing = record(1, 0, clean=0.10, patch_evidence=0.20)  # gap 0.10
        self.assertIsNotNone(patch_agreement_for(agreeing, VARIANT_GATED, margin=0.25))


class ConfidenceDoesNotAffectProbabilityTest(unittest.TestCase):
    """The whole point of the ablation: confidence must not leak into scoring."""

    def test_probability_metrics_identical_across_variants(self) -> None:
        records = population()
        results = {
            variant: evaluate_variant(records, variant, CONFIG, 0.5)
            for variant in (VARIANT_WITHOUT, VARIANT_WITH, VARIANT_GATED)
        }
        baseline = results[VARIANT_WITHOUT]["probability_metrics"]
        for variant in (VARIANT_WITH, VARIANT_GATED):
            with self.subTest(variant=variant):
                self.assertEqual(results[variant]["probability_metrics"], baseline)

    def test_reliability_fields_present(self) -> None:
        payload = evaluate_variant(population(), VARIANT_WITHOUT, CONFIG, 0.5)
        reliability = payload["confidence_reliability"]
        for field in (
            "auroc_confidence_vs_correctness",
            "confidence_gap",
            "point_biserial_correlation",
            "expected_calibration_error",
            "accuracy_by_confidence_level",
        ):
            with self.subTest(field=field):
                self.assertIn(field, reliability)


class DecisionRuleTest(unittest.TestCase):
    """The pre-registered rule must be applied exactly as written."""

    @staticmethod
    def make(auroc: float, gap: float):
        return {
            "confidence_reliability": {
                "auroc_confidence_vs_correctness": auroc,
                "confidence_gap": gap,
            }
        }

    def test_removes_when_no_gain(self) -> None:
        results = {
            VARIANT_WITHOUT: self.make(0.80, 0.10),
            VARIANT_WITH: self.make(0.80, 0.10),
            VARIANT_GATED: self.make(0.79, 0.09),
        }
        self.assertEqual(decide(results)["conclusion"], "remove_patch_agreement")

    def test_removes_when_gain_is_below_the_margin(self) -> None:
        results = {
            VARIANT_WITHOUT: self.make(0.80, 0.10),
            VARIANT_WITH: self.make(0.80 + MIN_AUROC_GAIN / 2, 0.11),
        }
        self.assertEqual(decide(results)["conclusion"], "remove_patch_agreement")

    def test_removes_when_gap_shrinks_despite_auroc_gain(self) -> None:
        results = {
            VARIANT_WITHOUT: self.make(0.80, 0.10),
            VARIANT_WITH: self.make(0.90, 0.05),
        }
        self.assertEqual(decide(results)["conclusion"], "remove_patch_agreement")

    def test_retains_when_the_rule_is_genuinely_met(self) -> None:
        results = {
            VARIANT_WITHOUT: self.make(0.80, 0.10),
            VARIANT_WITH: self.make(0.85, 0.12),
        }
        decision = decide(results)
        self.assertEqual(decision["conclusion"], "retain_patch_agreement")
        self.assertEqual(decision["variants_that_passed"][0]["variant"], VARIANT_WITH)


class ShippedDecisionTest(unittest.TestCase):
    """The measured outcome is encoded in the shipped configuration."""

    def setUp(self) -> None:
        from pathlib import Path

        import yaml

        root = Path(__file__).resolve().parents[1]
        self.config = yaml.safe_load((root / "configs/config.yaml").read_text(encoding="utf-8"))

    def test_patch_agreement_weight_is_zero(self) -> None:
        self.assertEqual(self.config["confidence"]["patch_agreement_weight"], 0.0)

    def test_confidence_weights_sum_to_one_without_patches(self) -> None:
        c = self.config["confidence"]
        total = c["decisiveness_weight"] + c["agreement_weight"] + c["consistency_weight"]
        self.assertAlmostEqual(total, 1.0)

    def test_zero_weight_removes_the_term_entirely(self) -> None:
        report = compute_confidence(
            0.9, 1.0, 1.0, {"confidence": self.config["confidence"]}, patch_agreement=0.5
        )
        self.assertNotIn("patch_agreement", report.weights)

    def test_patch_input_cannot_change_confidence(self) -> None:
        config = {"confidence": self.config["confidence"]}
        without = compute_confidence(0.9, 1.0, 1.0, config, patch_agreement=None)
        with_patch = compute_confidence(0.9, 1.0, 1.0, config, patch_agreement=0.0)
        self.assertAlmostEqual(without.score, with_patch.score, places=12)

    def test_patch_evidence_has_zero_probability_fusion_weight(self) -> None:
        """Patch evidence must never reach the predicted probability."""

        self.assertEqual(self.config["fusion"]["mode"], "rgb_transform")
        settings = {"fusion": self.config["fusion"]}
        without = fuse_predictions(0.9, [0.5], settings)
        with_patch = fuse_predictions(0.9, [0.5], settings, patch_evidence=0.99)
        self.assertAlmostEqual(without.final_probability, with_patch.final_probability, places=12)
        self.assertNotIn("patch", with_patch.weights)


class ThresholdIsFixedAcrossTransformationsTest(unittest.TestCase):
    """One threshold must apply to every condition, never retuned per transform."""

    def test_protocol_uses_one_threshold_for_every_condition(self) -> None:
        from src.evaluation.protocol import per_transformation_metrics

        records = population()
        metrics = per_transformation_metrics(records, threshold=0.37)
        thresholds = {payload["threshold"] for payload in metrics.values()}
        self.assertEqual(thresholds, {0.37})
        self.assertGreater(len(metrics), 1)

    def test_threshold_selection_reads_clean_scores_only(self) -> None:
        from src.evaluation.protocol import select_fixed_threshold

        clean = population()
        baseline = select_fixed_threshold(clean, 0.05).f1_optimal

        corrupted = []
        for item in clean:
            copy = record(0, item.binary_label, item.clean_score, item.patch_evidence)
            copy.img_id = item.img_id
            copy.version_scores = {CLEAN_KEY: item.clean_score}
            copy.version_scores.update(dict.fromkeys(TRANSFORMS, 0.5))
            corrupted.append(copy)
        self.assertEqual(select_fixed_threshold(corrupted, 0.05).f1_optimal, baseline)


class CalibrationFallbackTest(unittest.TestCase):
    """A missing calibration file must not stop inference."""

    def test_pipeline_survives_a_missing_calibration_file(self) -> None:
        import tempfile
        from pathlib import Path

        from src.pipeline.pipeline import DetectionPipeline
        from tests.helpers import base_config, write_mock_checkpoint

        config = base_config()
        config["transformations"]["enabled"] = False
        config["patches"] = {"enabled": False}
        config["calibration"] = {"enabled": True, "path": "outputs/absent_calibration.json"}

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = write_mock_checkpoint(Path(tmp) / "m.pt")
            pipeline = DetectionPipeline.from_checkpoint(
                checkpoint, config, device="cpu", explain_images=False
            )
            status = pipeline.calibration_status()

        self.assertIsNone(pipeline.calibrator)
        self.assertFalse(status["calibrated"])
        self.assertIsNotNone(status["calibration_error"])
        self.assertIn("UNCALIBRATED", status["probability_note"])


class PlattStabilityTest(unittest.TestCase):
    """Platt fitting must not diverge on near-separable validation data."""

    def test_fit_stays_finite_on_separable_scores(self) -> None:
        from src.evaluation.calibration import ProbabilityCalibrator

        labels = [0] * 20 + [1] * 20
        scores = [0.01] * 20 + [0.99] * 20  # perfectly separable
        calibrator = ProbabilityCalibrator.fit(scores, labels)

        self.assertTrue(np.isfinite(calibrator.scale))
        self.assertLess(abs(calibrator.scale), 1e4)
        calibrated = calibrator.transform(scores)
        self.assertGreater(float(np.std(calibrated)), 0.0)

    def test_calibration_does_not_collapse_to_a_step_function(self) -> None:
        from src.evaluation.calibration import ProbabilityCalibrator

        rng = np.random.default_rng(0)
        labels = np.concatenate([np.zeros(40), np.ones(40)]).astype(int)
        scores = np.clip(np.concatenate([rng.beta(1, 12, 40), rng.beta(12, 1, 40)]), 1e-4, 1 - 1e-4)
        calibrated = ProbabilityCalibrator.fit(scores, labels).transform(scores)

        # A diverged fit produces only 0.0 and 1.0, making every threshold equal.
        self.assertGreater(len(set(np.round(calibrated, 3).tolist())), 5)

    def test_calibration_preserves_ranking(self) -> None:
        from src.evaluation.calibration import ProbabilityCalibrator

        rng = np.random.default_rng(1)
        labels = np.concatenate([np.zeros(30), np.ones(30)]).astype(int)
        scores = np.clip(np.concatenate([rng.beta(2, 5, 30), rng.beta(5, 2, 30)]), 0, 1)
        calibrated = ProbabilityCalibrator.fit(scores, labels).transform(scores)

        self.assertTrue(
            np.array_equal(np.argsort(np.argsort(scores)), np.argsort(np.argsort(calibrated)))
        )


if __name__ == "__main__":
    unittest.main()
