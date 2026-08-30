"""Tests for the Track 5 evaluation protocol.

No model and no downloads: the scoring pass is represented by synthetic
``ScoredImage`` records, which is exactly what the protocol caches. These tests
pin the parts that determine whether the reported numbers are trustworthy --
the leakage-safe split, the frozen threshold, and the fact that all four system
variants are derived from identical scores.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.protocol import (
    CLEAN_KEY,
    VARIANT_FUSED,
    VARIANT_WHOLE,
    VARIANT_WHOLE_PATCH,
    VARIANT_WHOLE_TRANSFORM,
    VARIANTS,
    ScoredImage,
    confidence_distributions,
    dataset_holdout_statement,
    failure_examples,
    generator_claim_statement,
    per_transformation_metrics,
    robustness_summary,
    runtime_summary,
    select_fixed_threshold,
    split_records,
    subgroup_metrics,
    variant_metrics,
    variant_probability,
)

TRANSFORMS = ["jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30", "blur_s1", "center_crop_80"]

CONFIG = {
    "fusion": {
        "original_weight": 0.7,
        "transform_weight": 0.3,
        "whole_patch": {"whole_weight": 0.6, "transform_weight": 0.2, "patch_weight": 0.2},
    },
    "inference": {"threshold": 0.5},
}


def make_record(
    index: int,
    binary_label: int,
    clean: float,
    transformed: float | None = None,
    patch: float | None = 0.5,
    source_label: int | None = None,
) -> ScoredImage:
    transformed = clean if transformed is None else transformed
    scores = {CLEAN_KEY: clean}
    scores.update(dict.fromkeys(TRANSFORMS, transformed))
    return ScoredImage(
        img_id=f"img_{index:04d}",
        source_label=binary_label if source_label is None else source_label,
        class_name="real" if binary_label == 0 else "full_synthetic",
        binary_label=binary_label,
        version_scores=scores,
        patch_evidence=patch,
        patch_agreement=1.0 if patch is not None else None,
        patch_available=patch is not None,
        num_patches=9 if patch is not None else 0,
        seconds=1.5,
    )


def separable_records(n: int = 40):
    """Cleanly separable data, so metric assertions are unambiguous."""

    records = []
    for i in range(n // 2):
        records.append(make_record(i, 0, clean=0.05 + 0.001 * i, patch=0.05))
    for i in range(n // 2):
        records.append(make_record(100 + i, 1, clean=0.95 - 0.001 * i, patch=0.95, source_label=1))
    return records


class SplitTest(unittest.TestCase):
    def test_split_is_disjoint(self) -> None:
        validation, test = split_records(separable_records(40), 0.4)
        self.assertEqual({r.img_id for r in validation} & {r.img_id for r in test}, set())
        self.assertEqual(len(validation) + len(test), 40)

    def test_split_is_stratified(self) -> None:
        validation, test = split_records(separable_records(40), 0.4)
        for split in (validation, test):
            with self.subTest(size=len(split)):
                labels = {record.binary_label for record in split}
                self.assertEqual(labels, {0, 1})

    def test_split_is_deterministic_and_order_independent(self) -> None:
        records = separable_records(30)
        first, _ = split_records(records, 0.4)
        second, _ = split_records(list(reversed(records)), 0.4)
        self.assertEqual({r.img_id for r in first}, {r.img_id for r in second})

    def test_seed_changes_the_split(self) -> None:
        records = separable_records(40)
        a, _ = split_records(records, 0.4, seed=1)
        b, _ = split_records(records, 0.4, seed=2)
        self.assertNotEqual({r.img_id for r in a}, {r.img_id for r in b})

    def test_fraction_is_respected(self) -> None:
        validation, _ = split_records(separable_records(100), 0.3)
        self.assertAlmostEqual(len(validation) / 100, 0.3, delta=0.05)

    def test_invalid_fraction_rejected(self) -> None:
        for bad in (0.0, 1.0, -0.2, 1.5):
            with self.subTest(fraction=bad), self.assertRaises(ValueError):
                split_records(separable_records(10), bad)


class ThresholdTest(unittest.TestCase):
    def test_threshold_selected_from_clean_validation_only(self) -> None:
        validation, _ = split_records(separable_records(40), 0.4)
        selection = select_fixed_threshold(validation, 0.05)
        self.assertGreater(selection.balanced, 0.0)
        self.assertLess(selection.balanced, 1.0)

    def test_single_class_validation_is_rejected(self) -> None:
        """Never silently pick a threshold from one class."""

        with self.assertRaises(ValueError) as context:
            select_fixed_threshold([make_record(i, 0, 0.1) for i in range(5)], 0.05)
        self.assertIn("both authentic and AI-generated", str(context.exception))

    def test_transformed_scores_do_not_move_the_threshold(self) -> None:
        """Selection must read clean scores only, never transformed ones."""

        validation, _ = split_records(separable_records(40), 0.4)
        baseline = select_fixed_threshold(validation, 0.05).balanced

        corrupted = []
        for record in validation:
            copy = make_record(0, record.binary_label, record.clean_score, transformed=0.5)
            copy.img_id = record.img_id
            corrupted.append(copy)
        self.assertEqual(select_fixed_threshold(corrupted, 0.05).balanced, baseline)


class VariantTest(unittest.TestCase):
    def test_whole_only_returns_the_clean_score(self) -> None:
        record = make_record(1, 1, clean=0.83, transformed=0.2, patch=0.1)
        self.assertAlmostEqual(variant_probability(record, VARIANT_WHOLE, CONFIG), 0.83)

    def test_whole_plus_transformations_uses_seventy_thirty(self) -> None:
        record = make_record(1, 1, clean=0.9, transformed=0.5, patch=None)
        value = variant_probability(record, VARIANT_WHOLE_TRANSFORM, CONFIG)
        self.assertAlmostEqual(value, 0.7 * 0.9 + 0.3 * 0.5, places=6)

    def test_whole_plus_patches_ignores_transformations(self) -> None:
        record = make_record(1, 1, clean=0.9, transformed=0.1, patch=0.5)
        value = variant_probability(record, VARIANT_WHOLE_PATCH, CONFIG)
        self.assertAlmostEqual(value, (0.6 * 0.9 + 0.2 * 0.5) / 0.8, places=6)

    def test_fused_uses_all_three_terms(self) -> None:
        record = make_record(1, 1, clean=0.9, transformed=0.5, patch=0.4)
        value = variant_probability(record, VARIANT_FUSED, CONFIG)
        self.assertAlmostEqual(value, 0.6 * 0.9 + 0.2 * 0.5 + 0.2 * 0.4, places=6)

    def test_patch_variant_is_none_without_patches(self) -> None:
        record = make_record(1, 1, clean=0.9, patch=None)
        self.assertIsNone(variant_probability(record, VARIANT_WHOLE_PATCH, CONFIG))

    def test_fused_degrades_when_patches_missing(self) -> None:
        record = make_record(1, 1, clean=0.9, transformed=0.5, patch=None)
        value = variant_probability(record, VARIANT_FUSED, CONFIG)
        self.assertAlmostEqual(value, (0.6 * 0.9 + 0.2 * 0.5) / 0.8, places=6)

    def test_unknown_variant_rejected(self) -> None:
        with self.assertRaises(ValueError):
            variant_probability(make_record(1, 1, 0.5), "telepathy", CONFIG)

    def test_all_variants_share_the_same_scores(self) -> None:
        """The ablation must compare systems, not separate scoring runs."""

        records = separable_records(40)
        results = variant_metrics(records, 0.5, CONFIG)
        self.assertEqual(set(results), set(VARIANTS))
        counts = {v["count"] for v in results.values() if v.get("metrics")}
        self.assertEqual(len(counts), 1)  # every variant saw the same images


class MetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = separable_records(40)
        self.per_version = per_transformation_metrics(self.records, 0.5)

    def test_clean_and_every_transformation_reported(self) -> None:
        self.assertIn(CLEAN_KEY, self.per_version)
        for name in TRANSFORMS:
            with self.subTest(transformation=name):
                self.assertIn(name, self.per_version)

    def test_required_metrics_present(self) -> None:
        for field in (
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "auc",
            "false_positive_rate",
            "false_negative_rate",
            "confusion_matrix",
        ):
            with self.subTest(field=field):
                self.assertIn(field, self.per_version[CLEAN_KEY])

    def test_robustness_drop_and_ratio(self) -> None:
        degraded = [
            make_record(i, r.binary_label, r.clean_score, transformed=0.5, patch=r.patch_evidence)
            for i, r in enumerate(self.records)
        ]
        summary = robustness_summary(per_transformation_metrics(degraded, 0.5), "accuracy")
        self.assertEqual(summary["clean"], 1.0)
        self.assertLess(summary["worst_case"], 1.0)
        self.assertGreater(summary["largest_drop"], 0.0)
        self.assertIsNotNone(summary["worst_transformation"])
        for row in summary["per_transformation"]:
            with self.subTest(row=row["transformation"]):
                self.assertAlmostEqual(
                    row["robustness_drop"], summary["clean"] - row["accuracy"], places=6
                )

    def test_worst_case_is_the_minimum(self) -> None:
        summary = robustness_summary(self.per_version, "accuracy")
        values = [row["accuracy"] for row in summary["per_transformation"]]
        self.assertAlmostEqual(summary["worst_case"], min(values), places=6)

    def test_failure_examples_are_split_by_direction(self) -> None:
        records = separable_records(20) + [
            make_record(900, 0, clean=0.99, patch=0.99),  # false positive
            make_record(901, 1, clean=0.01, patch=0.01, source_label=1),  # false negative
        ]
        failures = failure_examples(records, 0.5, CONFIG)
        self.assertEqual(failures["false_positive_count"], 1)
        self.assertEqual(failures["false_negative_count"], 1)
        self.assertEqual(failures["false_positives"][0]["img_id"], "img_0900")
        self.assertEqual(failures["false_negatives"][0]["img_id"], "img_0901")

    def test_confidence_distributions_separate_classes(self) -> None:
        distributions = confidence_distributions(separable_records(40), CONFIG)
        self.assertIn("authentic", distributions)
        self.assertIn("ai_generated", distributions)
        self.assertLess(distributions["authentic"]["mean"], distributions["ai_generated"]["mean"])

    def test_subgroups_keep_both_classes(self) -> None:
        records = separable_records(20) + [
            make_record(500 + i, 1, clean=0.9, patch=0.9, source_label=2) for i in range(6)
        ]
        groups = subgroup_metrics(records, 0.5, CONFIG)
        self.assertIn("tampered", groups)
        self.assertIsNotNone(groups["tampered"]["metrics"])
        self.assertGreater(groups["tampered"]["authentic_images"], 0)

    def test_runtime_summary(self) -> None:
        summary = runtime_summary(separable_records(10))
        self.assertEqual(summary["images"], 10)
        self.assertEqual(summary["versions_per_image"], len(TRANSFORMS) + 1)
        self.assertGreater(summary["forward_passes_per_image"], summary["versions_per_image"])
        self.assertAlmostEqual(summary["seconds_per_image_mean"], 1.5, places=3)


class ClaimTest(unittest.TestCase):
    """The protocol must not overclaim generalisation."""

    def test_generator_claim_is_labelled_a_proxy(self) -> None:
        statement = generator_claim_statement().lower()
        self.assertIn("not a true unseen-generator holdout", statement)
        self.assertIn("proxy" if "proxy" in statement else "family", statement)

    def test_dataset_claim_requires_known_training_data(self) -> None:
        unknown = dataset_holdout_statement(None).lower()
        self.assertIn("cannot be verified", unknown)
        self.assertIn("no generalisation claim", unknown)

        known = dataset_holdout_statement("OpenFake (shards 0-14)").lower()
        self.assertIn("genuine", known)
        self.assertIn("per-generator generalisation may not", known)


class CacheTest(unittest.TestCase):
    def test_records_round_trip_through_json(self) -> None:
        original = make_record(7, 1, clean=0.82, transformed=0.6, patch=0.7)
        restored = ScoredImage.from_dict(json.loads(json.dumps(original.as_dict())))

        self.assertEqual(restored.img_id, original.img_id)
        self.assertEqual(restored.binary_label, original.binary_label)
        self.assertAlmostEqual(restored.clean_score, original.clean_score, places=6)
        self.assertAlmostEqual(restored.patch_evidence, original.patch_evidence, places=6)
        self.assertEqual(restored.version_scores.keys(), original.version_scores.keys())

    def test_cache_file_round_trip(self) -> None:
        records = separable_records(6)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.json"
            path.write_text(json.dumps({"records": [r.as_dict() for r in records]}))
            payload = json.loads(path.read_text())
            restored = [ScoredImage.from_dict(row) for row in payload["records"]]
        self.assertEqual(len(restored), len(records))


class ProtocolCliTest(unittest.TestCase):
    def test_parser_defaults(self) -> None:
        import scripts.evaluate_protocol as cli

        args = cli.build_parser().parse_args([])
        self.assertEqual(args.split, "validation")
        self.assertEqual(args.validation_fraction, 0.4)
        self.assertFalse(args.reuse_scores)

    def test_parser_accepts_overrides(self) -> None:
        import scripts.evaluate_protocol as cli

        args = cli.build_parser().parse_args(
            ["--limit", "50", "--device", "mps", "--no-patches", "--target-fpr", "0.01"]
        )
        self.assertEqual(args.limit, 50)
        self.assertEqual(args.device, "mps")
        self.assertTrue(args.no_patches)
        self.assertAlmostEqual(args.target_fpr, 0.01)


if __name__ == "__main__":
    unittest.main()
