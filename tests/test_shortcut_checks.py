"""Repository versions of the leakage and file-format scratchpad probes."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.evaluation.shortcut_checks import evaluate_format_reencoding


class FileFormatShortcutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.paths = []
        for index, extension in enumerate(("jpg", "png", "jpg", "png")):
            pixels = np.full((24, 24, 3), 30 + index * 60, dtype=np.uint8)
            path = self.root / f"image_{index}.{extension}"
            Image.fromarray(pixels).save(path)
            self.paths.append(path)
        self.labels = [0, 0, 1, 1]

    @staticmethod
    def pixel_scorer(path: Path) -> float:
        with Image.open(path) as image:
            return float(np.asarray(image.convert("RGB"), dtype=np.float64).mean() / 255.0)

    def test_pixel_based_scores_survive_container_normalisation(self) -> None:
        report = evaluate_format_reencoding(
            self.paths, self.labels, self.pixel_scorer, 0.5, output_format="PNG"
        )
        self.assertAlmostEqual(report["mean_absolute_score_delta"], 0.0, places=12)
        self.assertEqual(report["decision_preservation_rate"], 1.0)
        self.assertEqual(report["original_metrics"]["auc"], 1.0)
        self.assertEqual(report["reencoded_metrics"]["auc"], 1.0)

    def test_extension_shortcut_is_exposed_by_normalisation(self) -> None:
        def scorer(path: Path) -> float:
            return 0.9 if path.suffix.lower() == ".png" else 0.1

        report = evaluate_format_reencoding(
            self.paths, self.labels, scorer, 0.5, output_format="PNG"
        )
        self.assertLess(report["decision_preservation_rate"], 1.0)
        self.assertGreater(report["mean_absolute_score_delta"], 0.0)

    def test_probe_never_fits_a_threshold(self) -> None:
        report = evaluate_format_reencoding(
            self.paths, self.labels, self.pixel_scorer, 0.37, output_format="PNG"
        )
        self.assertEqual(report["threshold"], 0.37)
        self.assertIn("never fitted", report["threshold_note"])


if __name__ == "__main__":
    unittest.main()
