"""Static UI wording and backwards-compatible output-contract checks."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class PatchExplainabilityWordingTest(unittest.TestCase):
    def test_unavailable_gradcam_section_is_not_rendered(self) -> None:
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("Suspicious-region heatmap", source)
        self.assertNotIn("render_explainability", source)

    # The exact sentence the demo is required to show. Kept verbatim here so a
    # reword cannot quietly soften the disclaimer.
    REQUIRED_CAVEAT = (
        "a highlighted region is a suspicious region that influenced the model's "
        "score. it is not proof of ai editing, a segmentation mask, or a "
        "reconstruction of editing history."
    )

    def test_demo_disclaims_segmentation_and_proof(self) -> None:
        """Source-level check on fragments.

        The caveat is written across several adjacent string literals, so the
        full sentence never appears contiguously in the source. The exact
        wording is asserted against the RENDERED captions instead, in
        test_rendered_patch_caption_contains_the_caveat.
        """

        source = (ROOT / "app.py").read_text(encoding="utf-8").lower()
        for fragment in (
            "a highlighted region is a suspicious region that influenced the model",
            "it is not proof of ai editing, a segmentation mask, or a",
            "reconstruction of editing history.",
        ):
            self.assertIn(fragment, source)
        self.assertIn("nothing was measured there, not because they look authentic", source)

    def test_demo_states_patch_evidence_carries_no_scoring_weight(self) -> None:
        collapsed = " ".join((ROOT / "app.py").read_text(encoding="utf-8").lower().split())
        self.assertIn("zero weight", collapsed)
        self.assertIn("confidence score", collapsed)

    def test_rendered_patch_caption_contains_the_caveat(self) -> None:
        import app

        captions = []
        # MagicMock supports the context-manager protocol, so st.expander(...)
        # works inside a `with` block without extra configuration.
        fake_st = SimpleNamespace(
            subheader=MagicMock(),
            info=MagicMock(),
            warning=MagicMock(),
            image=MagicMock(),
            expander=MagicMock(),
            columns=lambda count: [MagicMock() for _ in range(count)],
            caption=lambda value: captions.append(value),
            markdown=MagicMock(),
            dataframe=MagicMock(),
        )
        patch_result = SimpleNamespace(
            x=0, y=0, width=32, height=32, ai_probability=0.8
        )
        report = SimpleNamespace(
            available=True,
            message="",
            patches=[patch_result],
            top_patches=[patch_result],
            mean_probability=0.6,
            max_probability=0.8,
            agreement=1.0,
            coverage=np.zeros((4, 4), dtype=np.float32),
        )
        result = SimpleNamespace(patches=report, original_image=None)
        with patch.object(app, "st", fake_st), patch.object(
            app, "overlay_patch_heatmap", return_value=None
        ):
            app.render_patches(result)

        rendered = " ".join(" ".join(captions).lower().split())
        self.assertIn("nothing was measured there", rendered)
        for fragment in self.REQUIRED_CAVEAT.split(". "):
            self.assertIn(fragment.rstrip("."), rendered)
        self.assertIn("zero weight", rendered)


class RobustnessChartTest(unittest.TestCase):
    """The chart is read at a glance, so orientation and units are contractual."""

    @staticmethod
    def _result(count: int = 15):
        names = [
            "original", "jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30",
            "blur_s0.5", "blur_s1", "blur_s2", "resize_0.5x", "resize_0.25x",
            "noise_s0.02", "noise_s0.05", "noise_s0.1", "color_jitter",
            "center_crop_80",
        ][:count]
        return SimpleNamespace(
            predictions=[
                SimpleNamespace(
                    name=name,
                    ai_probability=0.5 + index * 0.01,
                    is_original=(name == "original"),
                )
                for index, name in enumerate(names)
            ],
            threshold_used=0.69,
        )

    def test_labels_are_human_readable_not_internal_ids(self) -> None:
        import app

        versions = set(app.robustness_frame(self._result())["Version"])
        self.assertIn("JPEG quality 90", versions)
        self.assertIn("Resized to 25%", versions)
        self.assertIn("Cropped to 80%", versions)
        self.assertIn("Original", versions)
        for raw in ("jpeg_q90", "resize_0.25x", "center_crop_80", "original"):
            self.assertNotIn(raw, versions)

    def test_values_are_percentages_not_fractions(self) -> None:
        import app

        values = app.robustness_frame(self._result())["Likelihood"]
        self.assertTrue(all(v > 1.0 for v in values), f"looks like fractions: {list(values)}")
        self.assertTrue(all(0.0 <= v <= 100.0 for v in values))

    def test_chart_is_horizontal(self) -> None:
        """Categories on y, magnitude on x -- rotated word labels are unreadable."""

        import app

        frame = app.robustness_frame(self._result())
        spec = app.robustness_chart(frame, 69.0).to_dict()
        encoding = spec["layer"][0]["encoding"]
        self.assertEqual(encoding["y"]["field"], "Version")
        self.assertEqual(encoding["y"]["type"], "nominal")
        self.assertEqual(encoding["x"]["field"], "Likelihood")
        self.assertEqual(encoding["x"]["type"], "quantitative")

    def test_axis_is_a_full_zero_to_hundred_percent_scale(self) -> None:
        """A truncated axis would exaggerate small differences between versions."""

        import app

        frame = app.robustness_frame(self._result())
        spec = app.robustness_chart(frame, 69.0).to_dict()
        self.assertEqual(spec["layer"][0]["encoding"]["x"]["scale"]["domain"], [0, 100])
        self.assertIn("%", spec["layer"][0]["encoding"]["x"]["title"])

    def test_generation_order_is_preserved_not_sorted_by_score(self) -> None:
        """Sorting by value would scatter the JPEG/blur/resize families."""

        import app

        frame = app.robustness_frame(self._result())
        spec = app.robustness_chart(frame, 69.0).to_dict()
        self.assertEqual(spec["layer"][0]["encoding"]["y"]["sort"], frame["Version"].tolist())

    def test_decision_threshold_is_drawn(self) -> None:
        import app

        frame = app.robustness_frame(self._result())
        spec = app.robustness_chart(frame, 69.0).to_dict()
        self.assertEqual(len(spec["layer"]), 3)
        self.assertEqual(spec["layer"][2]["encoding"]["x"]["field"], "Threshold")

    def test_chart_grows_with_the_number_of_versions(self) -> None:
        """Fifteen rows squeezed into a fixed height would be unreadable."""

        import app

        short = app.robustness_chart(app.robustness_frame(self._result(3)), 69.0).to_dict()
        tall = app.robustness_chart(app.robustness_frame(self._result(15)), 69.0).to_dict()
        self.assertGreater(tall["height"], short["height"])


class RegionTableTest(unittest.TestCase):
    """The table must account for every region the metric above it claims."""

    @staticmethod
    def _report(scored: int = 12, top_k: int = 3):
        patches = [
            SimpleNamespace(
                x=(index % 4) * 256,
                y=(index // 4) * 256,
                width=256,
                height=256,
                ai_probability=1.0 - index * 0.05,
            )
            for index in range(scored)
        ]
        ordered = sorted(patches, key=lambda p: p.ai_probability, reverse=True)
        return SimpleNamespace(
            available=True,
            message="",
            patches=patches,
            top_patches=ordered[:top_k],
            mean_probability=0.5,
            max_probability=1.0,
            agreement=1.0,
            coverage=np.ones((4, 4), dtype=np.float32),
        )

    def _render(self, report):
        import app

        frames = []
        captions = []
        fake_st = SimpleNamespace(
            subheader=MagicMock(),
            info=MagicMock(),
            warning=MagicMock(),
            image=MagicMock(),
            expander=MagicMock(),
            columns=lambda count: [MagicMock() for _ in range(count)],
            caption=lambda value: captions.append(value),
            markdown=MagicMock(),
            dataframe=lambda frame, **kwargs: frames.append(frame),
        )
        result = SimpleNamespace(patches=report, original_image=None)
        with patch.object(app, "st", fake_st), patch.object(
            app, "overlay_patch_heatmap", return_value=None
        ):
            app.render_patches(result)
        return frames, captions

    def test_every_scored_region_appears_not_just_the_outlined_ones(self) -> None:
        """Regression: the table used to show only top_k rows while the metric
        beside it reported the full count."""

        frames, _ = self._render(self._report(scored=12, top_k=3))
        self.assertEqual(len(frames), 1)
        self.assertEqual(len(frames[0]), 12)

    def test_rows_are_ranked_by_score(self) -> None:
        frames, _ = self._render(self._report(scored=9, top_k=3))
        likelihoods = [
            float(value.rstrip("%")) for value in frames[0]["Likelihood AI-generated"]
        ]
        self.assertEqual(likelihoods, sorted(likelihoods, reverse=True))

    def test_outlined_regions_are_marked_so_the_image_and_table_agree(self) -> None:
        frames, _ = self._render(self._report(scored=12, top_k=3))
        self.assertEqual(sum(1 for v in frames[0]["Outlined"] if v == "yes"), 3)

    def test_scores_are_percentages(self) -> None:
        frames, _ = self._render(self._report(scored=5, top_k=2))
        self.assertTrue(all("%" in v for v in frames[0]["Likelihood AI-generated"]))


class PredictionJsonContractTest(unittest.TestCase):
    def test_simple_contract_source_contains_only_image_path_and_pred(self) -> None:
        # The behavioural batch-inference test exercises real serialization;
        # this guard names the public contract directly at its owning method.
        import inspect

        from src.pipeline.pipeline import PipelineResult

        source = inspect.getsource(PipelineResult.as_simple_dict)
        self.assertIn('"image_path"', source)
        self.assertIn('"pred"', source)
        self.assertNotIn('"timestamp"', source)


if __name__ == "__main__":
    unittest.main()
