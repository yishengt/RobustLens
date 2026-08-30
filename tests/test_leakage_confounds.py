"""Leakage and shortcut audits, moved out of scratchpad probes into the repo.

These check the DATA, not the model. A clean result here says a shortcut is not
available to be learned; ruling out that the detector uses a confound needs the
re-scoring probe in ``evaluate_format_reencoding``.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.evaluation.shortcut_checks import (
    CONFOUND_SEPARATION_LIMIT,
    dataset_confounds,
    filename_leakage,
    image_attributes,
    transformed_copy_leakage,
)


def _write(path: Path, seed: int, size=(64, 64), fmt="JPEG", quality=95) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    if fmt == "JPEG":
        image.save(path, format=fmt, quality=quality)
    else:
        image.save(path, format=fmt)
    return path


class ImageAttributeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_attributes_are_read_from_the_container(self) -> None:
        path = _write(self.root / "a.png", 1, size=(120, 60), fmt="PNG")
        found = image_attributes(path)
        self.assertEqual(found["extension"], "png")
        self.assertEqual(found["format"], "PNG")
        self.assertEqual(found["width"], 120)
        self.assertEqual(found["height"], 60)
        self.assertEqual(found["resolution"], "120x60")

    def test_unreadable_file_returns_none(self) -> None:
        broken = self.root / "broken.png"
        broken.write_bytes(b"not an image")
        self.assertIsNone(image_attributes(broken))


class ConfoundSeparationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_format_confound_is_flagged_when_it_splits_the_classes(self) -> None:
        """The real defect found in SID_Set: one class is entirely PNG."""

        paths, labels = [], []
        for index in range(6):
            paths.append(_write(self.root / f"real_{index}.jpg", index, fmt="JPEG"))
            labels.append(0)
            paths.append(_write(self.root / f"ai_{index}.png", 100 + index, fmt="PNG"))
            labels.append(1)
        report = dataset_confounds(paths, labels)
        self.assertEqual(report["attributes"]["extension"]["separation"], 1.0)
        self.assertIn("extension", report["flagged"])
        self.assertIn("format", report["flagged"])

    def test_balanced_attributes_are_not_flagged(self) -> None:
        paths, labels = [], []
        for index in range(6):
            fmt, ext = ("JPEG", "jpg") if index % 2 else ("PNG", "png")
            paths.append(_write(self.root / f"real_{index}.{ext}", index, fmt=fmt))
            labels.append(0)
            paths.append(_write(self.root / f"ai_{index}.{ext}", 100 + index, fmt=fmt))
            labels.append(1)
        report = dataset_confounds(paths, labels)
        self.assertNotIn("extension", report["flagged"])
        self.assertLess(report["attributes"]["extension"]["separation"], CONFOUND_SEPARATION_LIMIT)

    def test_resolution_confound_is_flagged(self) -> None:
        paths, labels = [], []
        for index in range(5):
            paths.append(_write(self.root / f"r{index}.jpg", index, size=(64, 64)))
            labels.append(0)
            paths.append(_write(self.root / f"a{index}.jpg", 100 + index, size=(256, 256)))
            labels.append(1)
        report = dataset_confounds(paths, labels)
        self.assertEqual(report["attributes"]["resolution"]["separation"], 1.0)
        self.assertIn("resolution", report["flagged"])

    def test_aspect_ratio_confound_is_flagged(self) -> None:
        paths, labels = [], []
        for index in range(5):
            paths.append(_write(self.root / f"r{index}.jpg", index, size=(200, 100)))
            labels.append(0)
            paths.append(_write(self.root / f"a{index}.jpg", 100 + index, size=(100, 200)))
            labels.append(1)
        report = dataset_confounds(paths, labels)
        self.assertIn("aspect_ratio", report["flagged"])

    def test_compression_history_confound_is_flagged(self) -> None:
        """Bytes-per-pixel separates a q10 class from a lossless one."""

        paths, labels = [], []
        for index in range(5):
            paths.append(
                _write(self.root / f"low_{index}.jpg", index, size=(128, 128), quality=10)
            )
            labels.append(0)
            paths.append(
                _write(self.root / f"high_{index}.png", 100 + index, size=(128, 128), fmt="PNG")
            )
            labels.append(1)
        report = dataset_confounds(paths, labels)
        self.assertIn("bytes_per_pixel", report["flagged"])

    def test_unreadable_images_are_reported_not_silently_dropped(self) -> None:
        good = _write(self.root / "good.jpg", 1)
        broken = self.root / "broken.jpg"
        broken.write_bytes(b"nope")
        report = dataset_confounds([good, broken], [0, 1])
        self.assertEqual(report["images"], 1)
        self.assertEqual(len(report["unreadable"]), 1)

    def test_mismatched_lengths_are_rejected(self) -> None:
        good = _write(self.root / "good.jpg", 1)
        with self.assertRaises(ValueError):
            dataset_confounds([good, good], [0])


class FilenameLeakageTest(unittest.TestCase):
    def test_class_named_files_are_detected(self) -> None:
        report = filename_leakage(
            ["real_0001.jpg", "real_0002.jpg", "fake_0001.png", "fake_0002.png"], [0, 0, 1, 1]
        )
        self.assertTrue(report["filename_encodes_label"])
        self.assertIn("real", report["authentic_only_tokens"])
        self.assertIn("fake", report["ai_only_tokens"])

    def test_neutral_filenames_are_clean(self) -> None:
        report = filename_leakage(
            ["img_0001.jpg", "img_0002.jpg", "img_0003.png", "img_0004.png"], [0, 0, 1, 1]
        )
        self.assertFalse(report["filename_encodes_label"])

    def test_report_states_filenames_never_reach_the_model(self) -> None:
        report = filename_leakage(["real_1.jpg"], [0])
        self.assertIn("never reach the detector", report["note"])


class TransformedCopyLeakageTest(unittest.TestCase):
    def test_group_spanning_two_splits_is_caught(self) -> None:
        report = transformed_copy_leakage(
            {"train": ["g1", "g2"], "validation": ["g3"], "test": ["g2"]}
        )
        self.assertFalse(report["clean"])
        self.assertEqual(report["leaked_groups"], 1)
        self.assertEqual(report["examples"][0]["group_id"], "g2")

    def test_disjoint_groups_are_clean(self) -> None:
        report = transformed_copy_leakage(
            {"train": ["g1", "g2"], "validation": ["g3"], "test": ["g4"]}
        )
        self.assertTrue(report["clean"])
        self.assertEqual(report["leaked_groups"], 0)
        self.assertEqual(report["groups"], 4)

    def test_repeated_group_within_one_split_is_not_leakage(self) -> None:
        """Many versions of one source in one split is the intended layout."""

        report = transformed_copy_leakage({"train": ["g1", "g1", "g1"], "test": ["g2"]})
        self.assertTrue(report["clean"])


class RealDatasetConfoundTest(unittest.TestCase):
    """Runs against the actual evaluation set when it is present."""

    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1] / "data/extracted/sid_set"
        if not (self.root / "labels.json").is_file():
            self.skipTest("SID_Set is not extracted; run scripts/extract_dataset.py")

    def test_known_format_confound_is_still_present_and_documented(self) -> None:
        """SID_Set's full_synthetic class is 100% PNG; real is 0% PNG.

        The confound is real. It was ruled out as the *cause* of the AUC by
        re-scoring both classes re-encoded to a common format, which left AUC
        and class separation unchanged. This test pins the data fact so the
        conclusion cannot drift away from what the data actually looks like.
        """

        import json

        payload = json.loads((self.root / "labels.json").read_text(encoding="utf-8"))
        items = payload.get("images", payload)
        synthetic = [i for i in items if i["class_name"] == "full_synthetic"]
        real = [i for i in items if i["class_name"] == "real"]
        if not synthetic or not real:
            self.skipTest("Expected classes are absent from labels.json")
        synthetic_png = sum(1 for i in synthetic if str(i.get("format")).upper() == "PNG")
        real_png = sum(1 for i in real if str(i.get("format")).upper() == "PNG")
        self.assertEqual(synthetic_png, len(synthetic))
        self.assertEqual(real_png, 0)


if __name__ == "__main__":
    unittest.main()
