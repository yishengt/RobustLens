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
        fake_st = SimpleNamespace(
            subheader=MagicMock(),
            info=MagicMock(),
            warning=MagicMock(),
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
