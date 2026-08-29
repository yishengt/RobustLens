"""Stages 6-10 tests: labels, prediction ranges, consistency, fusion, confidence."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.pipeline.confidence import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    compute_confidence,
    decisiveness,
)
from src.pipeline.consistency import compute_consistency, consistency_score
from src.pipeline.fusion import MODE_FREQUENCY, MODE_RGB_TRANSFORM, fuse_predictions
from src.pipeline.model_loader import ModelBundle
from src.pipeline.prediction import (
    LABEL_AI,
    LABEL_AUTHENTIC,
    LABEL_UNCERTAIN,
    Prediction,
    label_for_probability,
    predict_tensor_batch,
    probabilities_from_logits,
)
from tests.helpers import base_config


def make_predictions(values, threshold_names=None):
    """Build a prediction list where the first entry is the original."""

    predictions = []
    for index, value in enumerate(values):
        predictions.append(
            Prediction(
                name="original" if index == 0 else f"t{index}",
                ai_probability=float(value),
                real_probability=float(1.0 - value),
                label=label_for_probability(value),
                is_original=(index == 0),
            )
        )
    return predictions


class LabelLogicTest(unittest.TestCase):
    def test_documented_label_bands(self) -> None:
        cases = [
            (0.00, LABEL_AUTHENTIC),
            (0.20, LABEL_AUTHENTIC),
            (0.39, LABEL_AUTHENTIC),
            (0.40, LABEL_UNCERTAIN),
            (0.50, LABEL_UNCERTAIN),
            (0.59, LABEL_UNCERTAIN),
            (0.60, LABEL_AI),
            (0.84, LABEL_AI),
            (1.00, LABEL_AI),
        ]
        for probability, expected in cases:
            with self.subTest(probability=probability):
                self.assertEqual(label_for_probability(probability, base_config()), expected)

    def test_thresholds_are_configurable(self) -> None:
        config = base_config()
        config["labels"] = {"authentic_max": 0.2, "ai_min": 0.8}
        self.assertEqual(label_for_probability(0.3, config), LABEL_UNCERTAIN)
        self.assertEqual(label_for_probability(0.85, config), LABEL_AI)
        self.assertEqual(label_for_probability(0.1, config), LABEL_AUTHENTIC)

    def test_invalid_thresholds_are_rejected(self) -> None:
        config = base_config()
        config["labels"] = {"authentic_max": 0.9, "ai_min": 0.2}
        with self.assertRaises(ValueError):
            label_for_probability(0.5, config)


class ProbabilityConversionTest(unittest.TestCase):
    def test_single_logit_uses_sigmoid(self) -> None:
        logits = torch.tensor([[0.0], [10.0], [-10.0]])
        probabilities = probabilities_from_logits(logits, num_classes=1)
        self.assertAlmostEqual(float(probabilities[0]), 0.5, places=5)
        self.assertGreater(probabilities[1], 0.99)
        self.assertLess(probabilities[2], 0.01)

    def test_two_logits_use_softmax(self) -> None:
        logits = torch.tensor([[0.0, 0.0], [0.0, 5.0]])
        probabilities = probabilities_from_logits(logits, num_classes=2, ai_class_index=1)
        self.assertAlmostEqual(float(probabilities[0]), 0.5, places=5)
        self.assertGreater(probabilities[1], 0.99)

    def test_ai_class_index_selects_the_column(self) -> None:
        logits = torch.tensor([[0.0, 5.0]])
        self.assertLess(probabilities_from_logits(logits, 2, ai_class_index=0)[0], 0.01)

    def test_probabilities_always_lie_in_the_unit_interval(self) -> None:
        rng = np.random.default_rng(0)
        logits = torch.tensor(rng.normal(0, 50, (200, 1)), dtype=torch.float32)
        probabilities = probabilities_from_logits(logits, num_classes=1)
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertTrue(np.all(probabilities <= 1.0))

    def test_out_of_range_class_index_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            probabilities_from_logits(torch.zeros(1, 2), num_classes=2, ai_class_index=5)


class DualBackboneInputTest(unittest.TestCase):
    def test_dual_backbone_receives_two_processor_specific_batches(self) -> None:
        class StubDualModel(torch.nn.Module):
            def forward(self, siglip_pixels, dinov2_pixels):
                self.seen_shapes = (tuple(siglip_pixels.shape), tuple(dinov2_pixels.shape))
                return siglip_pixels.mean(dim=(1, 2, 3)) + dinov2_pixels.mean(dim=(1, 2, 3))

        model = StubDualModel()
        bundle = ModelBundle(
            model=model,
            device=torch.device("cpu"),
            architecture="dual_backbone",
            num_classes=1,
            num_parameters=2,
            checkpoint_path="stub.pt",
            input_kind="dual",
            processors=(object(), object()),
        )
        result = predict_tensor_batch(
            bundle,
            (torch.ones(2, 3, 384, 384), torch.zeros(2, 3, 224, 224)),
        )
        self.assertEqual(model.seen_shapes, ((2, 3, 384, 384), (2, 3, 224, 224)))
        self.assertEqual(result.shape, (2,))
        self.assertTrue(np.all(result > 0.5))


class ConsistencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = base_config()

    def test_identical_scores_are_perfectly_consistent(self) -> None:
        self.assertEqual(consistency_score([0.8] * 5, self.config), 1.0)

    def test_a_single_score_is_trivially_consistent(self) -> None:
        self.assertEqual(consistency_score([0.42], self.config), 1.0)

    def test_wide_spread_scores_poorly(self) -> None:
        tight = consistency_score([0.80, 0.81, 0.79], self.config)
        wide = consistency_score([0.05, 0.95, 0.50], self.config)
        self.assertGreater(tight, wide)
        self.assertLess(wide, 0.3)

    def test_score_is_always_within_the_unit_interval(self) -> None:
        rng = np.random.default_rng(1)
        for _ in range(50):
            values = rng.uniform(0, 1, size=int(rng.integers(2, 16)))
            score = consistency_score(values, self.config)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_report_statistics_are_correct(self) -> None:
        values = [0.9, 0.7, 0.8, 0.6]
        report = compute_consistency(make_predictions(values), self.config)

        self.assertAlmostEqual(report.mean, float(np.mean(values)), places=6)
        self.assertAlmostEqual(report.minimum, 0.6, places=6)
        self.assertAlmostEqual(report.maximum, 0.9, places=6)
        self.assertAlmostEqual(report.std, float(np.std(values)), places=6)
        self.assertAlmostEqual(report.score_range, 0.3, places=6)
        self.assertEqual(report.num_versions, 4)

    def test_agreement_counts_versions_on_the_original_side(self) -> None:
        # Original 0.9 is AI; two versions fall below the 0.5 threshold.
        report = compute_consistency(make_predictions([0.9, 0.8, 0.3, 0.2]), base_config())
        self.assertAlmostEqual(report.agreement, 0.5, places=6)

    def test_full_agreement(self) -> None:
        report = compute_consistency(make_predictions([0.9, 0.8, 0.7]), base_config())
        self.assertEqual(report.agreement, 1.0)

    def test_empty_predictions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compute_consistency([], self.config)


class FusionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = base_config()

    def test_default_seventy_thirty_split(self) -> None:
        result = fuse_predictions(0.9, [0.5, 0.5, 0.5], self.config)
        self.assertAlmostEqual(result.final_probability, 0.7 * 0.9 + 0.3 * 0.5, places=6)
        self.assertEqual(result.mode, MODE_RGB_TRANSFORM)
        self.assertAlmostEqual(result.weights["original"], 0.7, places=6)

    def test_weights_are_configurable(self) -> None:
        config = base_config()
        config["fusion"]["original_weight"] = 0.5
        config["fusion"]["transform_weight"] = 0.5
        result = fuse_predictions(1.0, [0.0], config)
        self.assertAlmostEqual(result.final_probability, 0.5, places=6)

    def test_weights_are_normalised_when_they_do_not_sum_to_one(self) -> None:
        config = base_config()
        config["fusion"]["original_weight"] = 7.0
        config["fusion"]["transform_weight"] = 3.0
        result = fuse_predictions(0.9, [0.5], config)
        self.assertAlmostEqual(result.final_probability, 0.7 * 0.9 + 0.3 * 0.5, places=6)

    def test_no_transformed_versions_uses_the_original_alone(self) -> None:
        result = fuse_predictions(0.83, [], self.config)
        self.assertAlmostEqual(result.final_probability, 0.83, places=6)

    def test_result_is_always_a_valid_probability(self) -> None:
        rng = np.random.default_rng(3)
        for _ in range(50):
            original = float(rng.uniform(0, 1))
            transformed = rng.uniform(0, 1, size=int(rng.integers(0, 10))).tolist()
            value = fuse_predictions(original, transformed, self.config).final_probability
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_frequency_mode_uses_the_fifty_thirty_twenty_split(self) -> None:
        config = base_config()
        config["fusion"]["mode"] = MODE_FREQUENCY
        result = fuse_predictions(
            0.8, [0.8], config, consistency=1.0, frequency_probability=0.4
        )
        rgb = 0.7 * 0.8 + 0.3 * 0.8
        self.assertEqual(result.mode, MODE_FREQUENCY)
        self.assertAlmostEqual(
            result.final_probability, 0.5 * rgb + 0.3 * 0.4 + 0.2 * 1.0, places=6
        )

    def test_frequency_mode_falls_back_without_a_frequency_prediction(self) -> None:
        config = base_config()
        config["fusion"]["mode"] = MODE_FREQUENCY
        result = fuse_predictions(0.9, [0.5], config, consistency=1.0, frequency_probability=None)

        self.assertEqual(result.mode, MODE_RGB_TRANSFORM)
        self.assertIsNotNone(result.fallback_reason)
        self.assertAlmostEqual(result.final_probability, 0.7 * 0.9 + 0.3 * 0.5, places=6)

    def test_unknown_mode_is_rejected(self) -> None:
        config = base_config()
        config["fusion"]["mode"] = "telepathy"
        with self.assertRaises(ValueError):
            fuse_predictions(0.5, [0.5], config)

    def test_zero_weights_are_rejected(self) -> None:
        config = base_config()
        config["fusion"]["original_weight"] = 0.0
        config["fusion"]["transform_weight"] = 0.0
        with self.assertRaises(ValueError):
            fuse_predictions(0.5, [0.5], config)


class ConfidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = base_config()

    def test_decisiveness_is_distance_from_the_midpoint(self) -> None:
        self.assertAlmostEqual(decisiveness(0.5), 0.0, places=6)
        self.assertAlmostEqual(decisiveness(1.0), 1.0, places=6)
        self.assertAlmostEqual(decisiveness(0.0), 1.0, places=6)
        self.assertAlmostEqual(decisiveness(0.75), 0.5, places=6)

    def test_decisive_and_stable_gives_high_confidence(self) -> None:
        report = compute_confidence(0.97, agreement=1.0, consistency=1.0, config=self.config)
        self.assertEqual(report.level, CONFIDENCE_HIGH)
        self.assertGreater(report.score, 0.7)

    def test_borderline_and_unstable_gives_low_confidence(self) -> None:
        report = compute_confidence(0.51, agreement=0.3, consistency=0.1, config=self.config)
        self.assertEqual(report.level, CONFIDENCE_LOW)

    def test_middling_inputs_give_medium_confidence(self) -> None:
        report = compute_confidence(0.75, agreement=0.6, consistency=0.6, config=self.config)
        self.assertEqual(report.level, CONFIDENCE_MEDIUM)

    def test_uncertain_label_is_never_high_confidence(self) -> None:
        report = compute_confidence(
            0.5, agreement=1.0, consistency=1.0, config=self.config, label=LABEL_UNCERTAIN
        )
        self.assertNotEqual(report.level, CONFIDENCE_HIGH)

    def test_score_is_always_within_the_unit_interval(self) -> None:
        rng = np.random.default_rng(5)
        for _ in range(50):
            report = compute_confidence(
                float(rng.uniform(0, 1)),
                float(rng.uniform(0, 1)),
                float(rng.uniform(0, 1)),
                self.config,
            )
            self.assertGreaterEqual(report.score, 0.0)
            self.assertLessEqual(report.score, 1.0)
            self.assertIn(report.level, {CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW})

    def test_bands_are_configurable(self) -> None:
        config = base_config()
        config["confidence"]["high_min"] = 0.99
        report = compute_confidence(0.95, agreement=1.0, consistency=1.0, config=config)
        self.assertNotEqual(report.level, CONFIDENCE_HIGH)

    def test_wording_is_probabilistic_not_absolute(self) -> None:
        report = compute_confidence(
            0.95, agreement=1.0, consistency=1.0, config=self.config, label=LABEL_AI
        )
        statement = report.statement.lower()
        self.assertIn("likely ai-generated", statement)
        self.assertNotIn("proof of", statement.replace("not proof of", ""))
        self.assertIn("not proof", statement)

    def test_authentic_wording(self) -> None:
        report = compute_confidence(
            0.05, agreement=1.0, consistency=1.0, config=self.config, label=LABEL_AUTHENTIC
        )
        self.assertIn("likely authentic", report.statement.lower())

    def test_zero_weights_are_rejected(self) -> None:
        config = base_config()
        config["confidence"].update(
            {"decisiveness_weight": 0, "agreement_weight": 0, "consistency_weight": 0}
        )
        with self.assertRaises(ValueError):
            compute_confidence(0.9, 1.0, 1.0, config)


if __name__ == "__main__":
    unittest.main()
