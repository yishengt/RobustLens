"""Tests for patch-level detection: tiling, heatmaps, fusion and degradation.

Uses an UNTRAINED checkpoint, so these verify plumbing and geometry only --
patch coordinates, coverage, score ranges, safe fallbacks and the JSON schema.
They say nothing about localisation accuracy.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.pipeline.model_loader import load_model
from src.pipeline.patches import (
    MODE_COARSE,
    PatchReport,
    PatchScorer,
    _boxes_for_mode,
    analyse_patches,
    build_heatmap,
    generate_grid_boxes,
    generate_patch_boxes,
    overlay_patch_heatmap,
    patch_settings,
    refine_boxes,
)
from src.pipeline.pipeline import DetectionPipeline
from src.pipeline.preprocessing import Preprocessor
from tests.helpers import base_config, make_image, write_mock_checkpoint


def patch_config(**overrides):
    """Base config for patch tests; overrides land in the patches block."""

    config = base_config()
    config["transformations"]["enabled"] = False  # keep the mock runs fast
    config["patches"] = {
        "enabled": True,
        "patch_size": 96,
        "stride": 64,
        "min_patch_size": 32,
        "max_patches": 6,
        "top_k": 2,
        "heatmap_threshold": 0.5,
        "evidence_statistic": "top_k_mean",
    }
    config["patches"].update(overrides)
    config["fusion"] = {
        "mode": "whole_patch_transform",
        "whole_patch": {"whole_weight": 0.6, "transform_weight": 0.2, "patch_weight": 0.2},
    }
    return config


class PatchSettingsTest(unittest.TestCase):
    def test_defaults(self) -> None:
        settings = patch_settings({})
        self.assertTrue(settings["enabled"])
        self.assertEqual(settings["patch_size"], 256)
        self.assertEqual(settings["evidence_statistic"], "top_k_mean")

    def test_invalid_values_rejected(self) -> None:
        for bad in (
            {"patch_size": 0},
            {"stride": -1},
            {"min_patch_size": 0},
            {"top_k": 0},
            {"max_patches": 0},
            {"heatmap_threshold": 1.5},
            {"evidence_statistic": "median"},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                patch_settings({"patches": bad})


class PatchGeometryTest(unittest.TestCase):
    def test_boxes_overlap_when_stride_is_smaller(self) -> None:
        boxes = generate_patch_boxes(400, 400, patch_size=200, stride=100, max_patches=99)
        xs = sorted({box[0] for box in boxes})
        self.assertGreater(len(xs), 2)
        self.assertLess(xs[1] - xs[0], 200)  # consecutive windows overlap

    def test_boxes_stay_inside_the_image(self) -> None:
        width, height = 333, 257
        for x, y, w, h in generate_patch_boxes(width, height, 128, 64, max_patches=99):
            with self.subTest(box=(x, y)):
                self.assertGreaterEqual(x, 0)
                self.assertGreaterEqual(y, 0)
                self.assertLessEqual(x + w, width)
                self.assertLessEqual(y + h, height)

    def test_right_and_bottom_edges_are_covered(self) -> None:
        width, height = 300, 300
        boxes = generate_patch_boxes(width, height, 128, 100, max_patches=99)
        self.assertTrue(any(x + w == width for x, _, w, _ in boxes))
        self.assertTrue(any(y + h == height for _, y, _, h in boxes))

    def test_max_patches_is_respected(self) -> None:
        boxes = generate_patch_boxes(2000, 2000, 128, 64, max_patches=7)
        self.assertLessEqual(len(boxes), 7)

    def test_four_nine_and_sixteen_patch_budgets_are_supported(self) -> None:
        """The required budgets produce exactly that many spatial samples."""

        for budget in (4, 9, 16):
            with self.subTest(budget=budget):
                boxes = generate_patch_boxes(
                    2000, 2000, patch_size=256, stride=128, max_patches=budget
                )
                self.assertEqual(len(boxes), budget)
                self.assertEqual(len(set(boxes)), budget)

    def test_image_below_minimum_yields_no_boxes(self) -> None:
        self.assertEqual(generate_patch_boxes(24, 24, 96, 64, min_patch_size=32), [])

    def test_small_image_still_gets_a_grid(self) -> None:
        boxes = generate_patch_boxes(120, 120, patch_size=256, stride=192, min_patch_size=32)
        self.assertTrue(boxes)
        self.assertLessEqual(boxes[0][2], 120)


class HeatmapTest(unittest.TestCase):
    def test_single_patch_fills_its_region(self) -> None:
        heatmap, coverage = build_heatmap([(0, 0, 10, 10)], [0.8], 20, 20)
        self.assertAlmostEqual(float(heatmap[5, 5]), 0.8, places=5)
        self.assertTrue(coverage[5, 5])

    def test_uncovered_pixels_are_flagged_not_scored_zero(self) -> None:
        """A 0.0 in an unmeasured region must be distinguishable from a real 0.0."""

        heatmap, coverage = build_heatmap([(0, 0, 10, 10)], [0.8], 20, 20)
        self.assertFalse(coverage[15, 15])
        self.assertEqual(float(heatmap[15, 15]), 0.0)
        self.assertLess(float(coverage.mean()), 1.0)

    def test_overlapping_patches_are_averaged(self) -> None:
        heatmap, _ = build_heatmap([(0, 0, 10, 10), (5, 0, 10, 10)], [0.0, 1.0], 15, 10)
        self.assertAlmostEqual(float(heatmap[5, 7]), 0.5, places=5)

    def test_values_stay_in_unit_interval(self) -> None:
        heatmap, _ = build_heatmap([(0, 0, 5, 5)], [1.0], 5, 5)
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)


class PatchPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.checkpoint = write_mock_checkpoint(cls.tmp / "mock.pt")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def build(self, config=None) -> DetectionPipeline:
        return DetectionPipeline.from_checkpoint(
            self.checkpoint, config or patch_config(), device="cpu", explain_images=False
        )

    def test_patch_report_is_populated(self) -> None:
        result = self.build().analyse_image(make_image(width=300, height=220, seed=3))
        report = result.patches

        self.assertIsInstance(report, PatchReport)
        self.assertTrue(report.available)
        self.assertGreater(len(report.patches), 1)
        self.assertLessEqual(len(report.top_patches), 2)
        self.assertEqual(report.heatmap.shape, (220, 300))

    def test_patch_scores_are_probabilities(self) -> None:
        report = self.build().analyse_image(make_image(width=260, height=260)).patches
        for patch in report.patches:
            with self.subTest(index=patch.index):
                self.assertGreaterEqual(patch.ai_probability, 0.0)
                self.assertLessEqual(patch.ai_probability, 1.0)
        self.assertGreaterEqual(report.evidence, 0.0)
        self.assertLessEqual(report.evidence, 1.0)

    def test_top_patches_are_the_highest_scoring(self) -> None:
        report = self.build().analyse_image(make_image(width=300, height=300)).patches
        best = max(patch.ai_probability for patch in report.patches)
        self.assertAlmostEqual(report.top_patches[0].ai_probability, best, places=6)
        self.assertEqual(
            report.highest_risk_region["score"], report.top_patches[0].as_dict()["score"]
        )

    def test_highest_risk_region_has_pixel_coordinates(self) -> None:
        image = make_image(width=300, height=220)
        region = self.build().analyse_image(image).patches.highest_risk_region
        for key in ("x", "y", "width", "height", "score"):
            self.assertIn(key, region)
        self.assertLessEqual(region["x"] + region["width"], 300)
        self.assertLessEqual(region["y"] + region["height"], 220)

    def test_patch_evidence_reaches_fusion(self) -> None:
        result = self.build().analyse_image(make_image(width=300, height=300))
        self.assertEqual(result.fusion.mode, "whole_patch_transform")
        self.assertIn("patch", result.fusion.weights)
        self.assertIn("patch_evidence", result.fusion.components)

    def test_patch_agreement_is_excluded_from_confidence(self) -> None:
        """Removed on evidence: it made confidence worse at spotting errors.

        scripts/evaluate_confidence.py measured AUROC(confidence vs correctness)
        falling from 0.767 to 0.737 when patch agreement was included. The patch
        report still computes agreement for display; it just carries no weight.
        """

        config = patch_config()
        config["confidence"]["patch_agreement_weight"] = 0.0
        result = DetectionPipeline.from_checkpoint(
            self.checkpoint, config, device="cpu", explain_images=False
        ).analyse_image(make_image(width=300, height=300))

        self.assertNotIn("patch_agreement", result.confidence.weights)
        self.assertIsNotNone(result.patches.agreement)  # still reported for the UI

    def test_disabled_patches_fall_back_safely(self) -> None:
        result = self.build(patch_config(enabled=False)).analyse_image(
            make_image(width=300, height=300)
        )

        self.assertFalse(result.patches.available)
        self.assertNotIn("patch", result.fusion.weights)
        self.assertIsNone(result.confidence.patch_agreement)
        # The whole-image result must still be a valid probability.
        self.assertGreaterEqual(result.ai_probability, 0.0)
        self.assertLessEqual(result.ai_probability, 1.0)

    def test_tiny_image_skips_patches_without_failing(self) -> None:
        """An image below min_patch_size cannot be tiled at all."""

        result = self.build().analyse_image(make_image(width=24, height=24))

        self.assertFalse(result.patches.available)
        self.assertIn("smaller than", result.patches.message)
        self.assertNotIn("patch", result.fusion.weights)
        self.assertGreaterEqual(result.ai_probability, 0.0)

    def test_single_patch_image_is_skipped_to_avoid_double_counting(self) -> None:
        """One whole-image patch would re-measure the whole-image score."""

        result = self.build().analyse_image(make_image(width=32, height=32))

        self.assertFalse(result.patches.available)
        self.assertIn("single patch", result.patches.message)
        self.assertNotIn("patch", result.fusion.weights)
        self.assertIsNone(result.confidence.patch_agreement)

    def test_overlay_renders_and_respects_coverage(self) -> None:
        image = make_image(width=300, height=220, seed=8)
        report = self.build().analyse_image(image).patches
        overlay = overlay_patch_heatmap(image, report, draw_top_boxes=False)

        self.assertEqual(overlay.shape, (220, 300, 3))
        self.assertEqual(overlay.dtype, np.uint8)
        uncovered = ~report.coverage
        if uncovered.any():
            original = np.asarray(image.convert("RGB"))
            self.assertTrue((overlay[uncovered] == original[uncovered]).all())

    def test_overlay_returns_none_when_unavailable(self) -> None:
        self.assertIsNone(
            overlay_patch_heatmap(make_image(), PatchReport(available=False, message="off"))
        )

    def test_report_schema_contains_patch_fields(self) -> None:
        result = self.build().analyse_image(make_image(width=300, height=300))
        report = result.as_report_dict()

        for field in (
            "image_path",
            "raw_probability",
            "final_probability",
            "real_probability",
            "label",
            "confidence",
            "transformation_consistency",
            "estimated_manipulation_severity",
            "highest_risk_region",
            "per_transformation_predictions",
        ):
            with self.subTest(field=field):
                self.assertIn(field, report)
        self.assertIn(report["confidence"], {"high", "medium", "low"})
        self.assertIn(report["estimated_manipulation_severity"], {"low", "medium", "high"})
        self.assertIn("clean", report["per_transformation_predictions"])
        json.dumps(report)

    def test_detailed_dict_includes_patch_analysis(self) -> None:
        detailed = self.build().analyse_image(make_image(width=300, height=300)).as_detailed_dict()

        self.assertIn("patch_analysis", detailed)
        self.assertTrue(detailed["patch_analysis"]["available"])
        self.assertGreater(detailed["patch_analysis"]["num_patches"], 0)
        self.assertIn("heatmap_coverage", detailed["patch_analysis"])
        json.dumps(detailed)  # must stay serialisable with the arrays omitted

    def test_simple_output_contract_unchanged(self) -> None:
        simple = self.build().analyse_image(make_image(width=300, height=300)).as_simple_dict()
        self.assertEqual(set(simple), {"image_path", "pred"})
        self.assertGreaterEqual(simple["pred"], 0.0)
        self.assertLessEqual(simple["pred"], 1.0)


if __name__ == "__main__":
    unittest.main()


class PatchModeTest(unittest.TestCase):
    """Modes, cost instrumentation and the early-stop gate."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.checkpoint = write_mock_checkpoint(cls.tmp / "mock.pt")
        cls.bundle = load_model(cls.checkpoint, base_config(), device="cpu")
        cls.preprocessor = Preprocessor.from_config(base_config())
        cls.image = make_image(width=300, height=220, seed=5)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def run_mode(self, mode: str, whole: float = 0.5, **overrides):
        options = {"max_patches": 12, "coarse_max_patches": 4, "refine_factor": 2}
        options.update(overrides)
        config = patch_config(mode=mode, **options)
        return analyse_patches(
            self.bundle, self.image, self.preprocessor, config, whole_image_probability=whole
        )

    def test_off_mode_spends_nothing(self) -> None:
        report = self.run_mode("off")
        self.assertFalse(report.available)
        self.assertEqual(report.forward_passes, 0)

    def test_coarse_is_cheaper_than_full(self) -> None:
        coarse = self.run_mode("coarse")
        full = self.run_mode("full")
        self.assertLess(coarse.forward_passes, full.forward_passes)
        self.assertLessEqual(len(coarse.patches), len(full.patches))

    def test_full_covers_more_than_coarse(self) -> None:
        coarse = self.run_mode("coarse")
        full = self.run_mode("full")
        self.assertGreater(float(full.coverage.mean()), float(coarse.coverage.mean()))

    def test_top_k_refines_into_smaller_patches(self) -> None:
        """The second pass buys detail inside the chosen regions."""

        report = self.run_mode("top_k")
        self.assertTrue(report.available)
        sizes = {patch.width for patch in report.patches}
        self.assertGreater(len(sizes), 1)  # coarse boxes plus refined children

    def test_uncertain_only_skips_confident_images(self) -> None:
        for whole in (0.02, 0.99):
            with self.subTest(whole_score=whole):
                report = self.run_mode("uncertain_only", whole=whole, base_mode="coarse")
                self.assertFalse(report.available)
                self.assertEqual(report.forward_passes, 0)
                self.assertIn("early stop", report.message)

    def test_uncertain_only_runs_for_undecided_images(self) -> None:
        report = self.run_mode("uncertain_only", whole=0.5, base_mode="coarse")
        self.assertTrue(report.available)
        self.assertGreater(report.forward_passes, 0)

    def test_every_mode_records_cost(self) -> None:
        for mode in ("coarse", "full", "top_k"):
            with self.subTest(mode=mode):
                report = self.run_mode(mode)
                self.assertEqual(report.mode, mode)
                self.assertGreater(report.forward_passes, 0)
                self.assertGreater(report.seconds, 0.0)
                self.assertIn("forward_passes", report.as_dict())
                self.assertIn("peak_memory_mb", report.as_dict())

    def test_invalid_mode_rejected(self) -> None:
        report = self.run_mode("telepathy")
        self.assertFalse(report.available)
        self.assertIn("Invalid patch configuration", report.message)

    def test_message_avoids_claiming_proof(self) -> None:
        """The heatmap must never be described as evidence of editing."""

        message = self.run_mode("full").message.lower()
        self.assertIn("not proof", message)
        self.assertIn("suspicious", message)
        self.assertNotIn("proves", message)


class PatchScorerTest(unittest.TestCase):
    """Caching and whole-image reuse must avoid duplicate forward passes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.bundle = load_model(
            write_mock_checkpoint(cls.tmp / "mock.pt"), base_config(), device="cpu"
        )
        cls.preprocessor = Preprocessor.from_config(base_config())

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_whole_image_box_is_reused_not_recomputed(self) -> None:
        image = make_image(width=128, height=128)
        scorer = PatchScorer(
            self.bundle, self.preprocessor, image, batch_size=4, whole_image_probability=0.77
        )
        scores = scorer.score([(0, 0, 128, 128)])

        self.assertAlmostEqual(scores[0], 0.77, places=6)
        self.assertEqual(scorer.forward_passes, 0)
        self.assertEqual(scorer.reused, 1)

    def test_repeated_boxes_are_cached(self) -> None:
        image = make_image(width=200, height=200)
        scorer = PatchScorer(self.bundle, self.preprocessor, image, batch_size=4)
        first = scorer.score([(0, 0, 64, 64), (64, 0, 64, 64)])
        after_first = scorer.forward_passes
        second = scorer.score([(0, 0, 64, 64), (64, 0, 64, 64)])

        self.assertEqual(first, second)
        self.assertEqual(scorer.forward_passes, after_first)  # no extra work

    def test_scores_are_probabilities(self) -> None:
        image = make_image(width=200, height=200)
        scorer = PatchScorer(self.bundle, self.preprocessor, image, batch_size=2)
        for score in scorer.score([(0, 0, 64, 64), (64, 64, 64, 64), (0, 64, 64, 64)]):
            with self.subTest(score=score):
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)


class RefineBoxesTest(unittest.TestCase):
    def test_subdivides_into_factor_squared_children(self) -> None:
        children = refine_boxes([(0, 0, 256, 256)], 2, 64, 1024, 1024)
        self.assertEqual(len(children), 4)
        self.assertTrue(all(w == 128 and h == 128 for _, _, w, h in children))

    def test_children_stay_inside_the_image(self) -> None:
        for x, y, w, h in refine_boxes([(768, 768, 256, 256)], 2, 64, 1024, 1024):
            with self.subTest(box=(x, y)):
                self.assertLessEqual(x + w, 1024)
                self.assertLessEqual(y + h, 1024)

    def test_refusal_below_minimum_size(self) -> None:
        self.assertEqual(refine_boxes([(0, 0, 96, 96)], 2, 64, 512, 512), [])

    def test_factor_one_is_a_no_op(self) -> None:
        self.assertEqual(refine_boxes([(0, 0, 256, 256)], 1, 64, 1024, 1024), [])


class AblationCliTest(unittest.TestCase):
    def test_parser_defaults_and_overrides(self) -> None:
        import scripts.ablate_patches as cli

        args = cli.build_parser().parse_args([])
        self.assertEqual(args.threshold, 0.42)
        self.assertIn("off", args.modes)

        args = cli.build_parser().parse_args(["--modes", "off", "full", "--limit", "10"])
        self.assertEqual(args.modes, ["off", "full"])
        self.assertEqual(args.limit, 10)

    def test_verdict_demotes_when_nothing_improves(self) -> None:
        import scripts.ablate_patches as cli

        summary = {
            "off": {"metrics": {"f1": 0.8, "recall": 0.7, "false_positive_rate": 0.04}},
            "full": {
                "metrics": {"f1": 0.8, "recall": 0.7, "false_positive_rate": 0.04},
                "delta_vs_whole_image_only": {"f1": 0.0, "recall": 0.0, "false_positive_rate": 0.0},
            },
        }
        verdict = cli.build_verdict(summary)
        self.assertEqual(verdict["conclusion"], "keep_as_explainability_only")

    def test_verdict_keeps_scoring_when_a_mode_helps(self) -> None:
        import scripts.ablate_patches as cli

        summary = {
            "off": {"metrics": {"f1": 0.80, "recall": 0.70, "false_positive_rate": 0.04}},
            "full": {
                "metrics": {"f1": 0.85, "recall": 0.78, "false_positive_rate": 0.03},
                "delta_vs_whole_image_only": {
                    "f1": 0.05,
                    "recall": 0.08,
                    "false_positive_rate": -0.01,
                },
            },
        }
        verdict = cli.build_verdict(summary)
        self.assertEqual(verdict["conclusion"], "keep_as_scoring_component")
        self.assertIn("full", verdict["modes_that_improved"])


class GridBoxesTest(unittest.TestCase):
    """Fixed NxN tiling, so the region count does not follow the aspect ratio."""

    def test_grid_yields_exactly_n_squared_tiles_on_any_aspect(self) -> None:
        for width, height in [(1024, 768), (1024, 1024), (1920, 1080), (768, 1024)]:
            with self.subTest(size=(width, height)):
                self.assertEqual(len(generate_grid_boxes(width, height, 4)), 16)

    def test_tiles_cover_the_whole_image_with_no_gaps_or_overlap(self) -> None:
        """Remainder pixels must go somewhere, or an edge strip goes unscored."""

        width, height = 1024, 681  # 681 is not divisible by 4
        boxes = generate_grid_boxes(width, height, 4)
        covered = np.zeros((height, width), dtype=np.int32)
        for x, y, box_width, box_height in boxes:
            covered[y : y + box_height, x : x + box_width] += 1
        self.assertEqual(int(covered.min()), 1, "a pixel was left unscored")
        self.assertEqual(int(covered.max()), 1, "a pixel was scored twice")

    def test_boxes_stay_inside_the_image(self) -> None:
        width, height = 1000, 700
        for x, y, box_width, box_height in generate_grid_boxes(width, height, 3):
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + box_width, width)
            self.assertLessEqual(y + box_height, height)

    def test_image_too_small_for_the_grid_yields_nothing(self) -> None:
        """Better no regions than sixteen meaningless 12px slivers."""

        self.assertEqual(generate_grid_boxes(100, 100, 4, min_patch_size=64), [])

    def test_grid_below_one_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generate_grid_boxes(512, 512, 0)


class GridSettingsTest(unittest.TestCase):
    def test_grid_is_optional_and_defaults_to_the_sliding_window(self) -> None:
        self.assertIsNone(patch_settings({"patches": {}})["grid"])

    def test_grid_exceeding_the_cap_is_rejected_with_a_usable_message(self) -> None:
        with self.assertRaises(ValueError) as caught:
            patch_settings({"patches": {"grid": 5, "max_patches": 16}})
        message = str(caught.exception)
        self.assertIn("25", message)
        self.assertIn("max_patches", message)

    def test_grid_within_the_cap_is_accepted(self) -> None:
        self.assertEqual(patch_settings({"patches": {"grid": 4, "max_patches": 24}})["grid"], 4)

    def test_grid_overrides_the_sliding_window_and_ignores_the_mode_cap(self) -> None:
        """The caller asked for a fixed grid; stride and coarse cap must not shrink it."""

        settings = patch_settings(
            {"patches": {"grid": 4, "max_patches": 24, "coarse_max_patches": 4}}
        )
        boxes = _boxes_for_mode(MODE_COARSE, settings, 1024, 768)
        self.assertEqual(len(boxes), 16)
