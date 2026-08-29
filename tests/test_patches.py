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

from src.pipeline.patches import (
    PatchReport,
    build_heatmap,
    generate_patch_boxes,
    overlay_patch_heatmap,
    patch_settings,
)
from src.pipeline.pipeline import DetectionPipeline
from tests.helpers import base_config, make_image, write_mock_checkpoint


def patch_config(**overrides):
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

    def test_patch_agreement_reaches_confidence(self) -> None:
        result = self.build().analyse_image(make_image(width=300, height=300))
        self.assertIsNotNone(result.confidence.patch_agreement)
        self.assertIn("patch_agreement", result.confidence.weights)

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
