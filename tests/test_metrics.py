"""Tests for the NumPy-only evaluation metrics."""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.evaluation.metrics import average_precision, compute_metrics, roc_auc
from src.evaluation.sid_set import to_binary_label


class RocAucTest(unittest.TestCase):
    def test_perfect_separation_scores_one(self) -> None:
        self.assertAlmostEqual(roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)

    def test_inverted_separation_scores_zero(self) -> None:
        self.assertAlmostEqual(roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]), 0.0)

    def test_constant_scores_give_one_half(self) -> None:
        self.assertAlmostEqual(roc_auc([0, 1, 0, 1], [0.5] * 4), 0.5)

    def test_known_value(self) -> None:
        # Positives score 0.4 and 0.8; negatives 0.5 and 0.35. Of the four
        # positive/negative pairs only (0.4, 0.5) is mis-ordered, so AUC = 3/4.
        self.assertAlmostEqual(roc_auc([0, 1, 0, 1], [0.5, 0.4, 0.35, 0.8]), 0.75)

    def test_all_pairs_ordered_scores_one(self) -> None:
        self.assertAlmostEqual(roc_auc([0, 1, 0, 1], [0.1, 0.4, 0.35, 0.8]), 1.0)

    def test_single_class_is_undefined(self) -> None:
        self.assertTrue(math.isnan(roc_auc([1, 1, 1], [0.2, 0.5, 0.9])))

    def test_ties_use_average_ranks(self) -> None:
        self.assertAlmostEqual(roc_auc([0, 1], [0.5, 0.5]), 0.5)

    def test_matches_brute_force_on_random_data(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(20):
            labels = rng.integers(0, 2, 40)
            if labels.min() == labels.max():
                continue
            scores = rng.uniform(0, 1, 40).round(2)  # rounding forces ties
            positives = scores[labels == 1]
            negatives = scores[labels == 0]
            # Brute-force definition: P(score_pos > score_neg) + 0.5 P(equal).
            comparisons = positives[:, None] - negatives[None, :]
            expected = float(
                ((comparisons > 0).sum() + 0.5 * (comparisons == 0).sum())
                / comparisons.size
            )
            self.assertAlmostEqual(roc_auc(labels, scores), expected, places=9)


class AveragePrecisionTest(unittest.TestCase):
    def test_perfect_ranking_scores_one(self) -> None:
        self.assertAlmostEqual(average_precision([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)

    def test_no_positives_is_undefined(self) -> None:
        self.assertTrue(math.isnan(average_precision([0, 0], [0.3, 0.7])))


class ComputeMetricsTest(unittest.TestCase):
    def test_confusion_matrix_and_derived_scores(self) -> None:
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.3, 0.8, 0.1]
        metrics = compute_metrics(labels, scores, threshold=0.5)

        self.assertEqual(metrics.true_positives, 1)
        self.assertEqual(metrics.false_negatives, 1)
        self.assertEqual(metrics.false_positives, 1)
        self.assertEqual(metrics.true_negatives, 1)
        self.assertAlmostEqual(metrics.accuracy, 0.5)
        self.assertAlmostEqual(metrics.precision, 0.5)
        self.assertAlmostEqual(metrics.recall, 0.5)
        self.assertAlmostEqual(metrics.f1, 0.5)

    def test_threshold_shifts_the_decision(self) -> None:
        labels = [1, 1, 0, 0]
        scores = [0.6, 0.55, 0.45, 0.2]
        self.assertAlmostEqual(compute_metrics(labels, scores, 0.5).accuracy, 1.0)
        self.assertAlmostEqual(compute_metrics(labels, scores, 0.9).accuracy, 0.5)

    def test_serialised_output_is_json_safe(self) -> None:
        import json

        payload = compute_metrics([0, 1], [0.2, 0.8]).as_dict()
        json.dumps(payload)
        self.assertIn("confusion_matrix", payload)
        self.assertEqual(payload["count"], 2)

    def test_undefined_auc_serialises_as_null(self) -> None:
        payload = compute_metrics([1, 1], [0.2, 0.8]).as_dict()
        self.assertIsNone(payload["auc"])

    def test_mismatched_lengths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compute_metrics([0, 1], [0.5])

    def test_empty_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compute_metrics([], [])


class LabelMappingTest(unittest.TestCase):
    def test_sid_set_labels_map_to_binary(self) -> None:
        self.assertEqual(to_binary_label(0), 0)  # real
        self.assertEqual(to_binary_label(1), 1)  # fully synthetic
        self.assertEqual(to_binary_label(2), 1)  # tampered

    def test_unknown_label_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            to_binary_label(7)


if __name__ == "__main__":
    unittest.main()
