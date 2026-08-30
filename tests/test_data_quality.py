"""Tests for the local-edit data-quality audit and the quarantine it produces.

The behaviour that matters is negative: images carrying contradictory labels
must not reach training or validation unless someone explicitly asks for them.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.finetune.data_quality import (
    REASON_IDENTICAL_BYTES,
    REASON_IDENTICAL_PIXELS,
    audit_splits,
    filter_records,
    load_quarantine,
    quick_conflict_audit,
)
from src.finetune.dataset import discover_split


def _write(path: Path, seed: int, size=(64, 64), fmt="JPEG") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(0, 255, (*size, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path, format=fmt, quality=95)


class DataQualityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        # Distinct seeds per split, so the fixture itself contains no
        # duplicates and each test creates exactly the defect it is about.
        for offset, split in enumerate(("train", "validation", "test")):
            for index in range(3):
                base = offset * 1000 + index * 10
                _write(self.root / split / "authentic" / f"g{split}{index}" / "source.jpg", base)
                _write(
                    self.root / split / "ai_edited" / f"g{split}{index}" / "edit_0.jpg",
                    base + 5,
                )

    def _summaries(self):
        return [discover_split(self.root / s, s) for s in ("train", "validation", "test")]

    def _make_conflict(self, split: str = "train", group: str = "gtrain0") -> Path:
        """Copy a source over its edit so the two are byte-identical."""

        source = self.root / split / "authentic" / group / "source.jpg"
        edit = self.root / split / "ai_edited" / group / "edit_0.jpg"
        shutil.copyfile(source, edit)
        return edit

    def test_clean_dataset_reports_no_conflicts(self) -> None:
        report = audit_splits(self._summaries(), root=self.root)
        self.assertEqual(report.conflicts, [])
        self.assertEqual(report.quarantined_paths, set())
        self.assertEqual(report.total_files, 18)

    def test_byte_identical_conflict_is_found(self) -> None:
        self._make_conflict()
        report = audit_splits(self._summaries(), root=self.root)
        self.assertEqual(len(report.conflicts), 1)
        self.assertEqual(report.conflicts[0].reason, REASON_IDENTICAL_BYTES)
        self.assertEqual(report.conflicts[0].labels, (0, 1))

    def test_both_sides_of_a_conflict_are_quarantined(self) -> None:
        """Never guess which label is right -- exclude the pair."""

        edit = self._make_conflict()
        report = audit_splits(self._summaries(), root=self.root)
        source = self.root / "train" / "authentic" / "gtrain0" / "source.jpg"
        self.assertEqual(report.quarantined_paths, {edit, source})

    def test_reencoded_conflict_is_found_by_pixel_hash(self) -> None:
        """A re-encoded copy has a different file hash but the same pixels."""

        source = self.root / "train" / "authentic" / "gtrain1" / "source.jpg"
        edit = self.root / "train" / "ai_edited" / "gtrain1" / "edit_0.jpg"
        with Image.open(source) as image:
            image.convert("RGB").save(edit, format="JPEG", quality=95)
        with Image.open(source) as a, Image.open(edit) as b:
            if np.array_equal(np.asarray(a.convert("RGB")), np.asarray(b.convert("RGB"))):
                report = audit_splits(self._summaries(), root=self.root)
                reasons = {c.reason for c in report.conflicts}
                self.assertTrue(
                    reasons & {REASON_IDENTICAL_BYTES, REASON_IDENTICAL_PIXELS},
                    f"expected a conflict, got {reasons}",
                )
            else:  # pragma: no cover - re-encode was not idempotent here
                self.skipTest("JPEG re-encode was not pixel-identical on this platform")

    def test_quick_audit_finds_the_same_byte_conflicts(self) -> None:
        self._make_conflict()
        full = audit_splits(self._summaries(), root=self.root)
        quick = quick_conflict_audit(self._summaries())
        self.assertEqual(quick.quarantined_paths, full.quarantined_paths)

    def test_quick_audit_does_not_report_every_file_as_undecodable(self) -> None:
        """Pixel hashing is off, which is not the same as failing to decode."""

        self.assertEqual(quick_conflict_audit(self._summaries()).undecodable, [])

    def test_audit_is_deterministic(self) -> None:
        self._make_conflict()
        first = audit_splits(self._summaries(), root=self.root).as_dict()
        second = audit_splits(self._summaries(), root=self.root).as_dict()
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))

    def test_report_records_file_and_image_hashes_and_group_ids(self) -> None:
        self._make_conflict()
        entry = audit_splits(self._summaries(), root=self.root).conflicts[0].as_dict(self.root)
        self.assertIn("shared_file_hash", entry)
        self.assertIn("shared_image_hash", entry)
        self.assertTrue(entry["same_group"])
        for member in entry["members"]:
            self.assertIn("group_id", member)
            self.assertIn("file_hash", member)
            self.assertIn("label_name", member)

    def test_format_counts_are_reported_per_class(self) -> None:
        counts = audit_splits(self._summaries(), root=self.root).format_counts
        self.assertEqual(set(counts), {"authentic", "ai_edited"})
        self.assertEqual(counts["authentic"]["jpg"], 9)

    def test_cross_split_duplicate_is_reported(self) -> None:
        source = self.root / "train" / "authentic" / "gtrain0" / "source.jpg"
        target = self.root / "test" / "authentic" / "gtest0" / "source.jpg"
        shutil.copyfile(source, target)
        report = audit_splits(self._summaries(), root=self.root)
        self.assertEqual(len(report.cross_split_duplicates), 1)


class FilterRecordsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        for index in range(3):
            _write(self.root / "train" / "authentic" / f"g{index}" / "source.jpg", index)
            _write(self.root / "train" / "ai_edited" / f"g{index}" / "edit_0.jpg", index + 100)
        self.summary = discover_split(self.root / "train", "train")

    def test_conflicts_are_excluded_by_default(self) -> None:
        quarantine = {self.summary.records[0].image_path}
        kept, dropped = filter_records(self.summary.records, quarantine)
        self.assertEqual(dropped, 1)
        self.assertNotIn(self.summary.records[0].image_path, [r.image_path for r in kept])

    def test_explicit_opt_in_keeps_them(self) -> None:
        quarantine = {self.summary.records[0].image_path}
        kept, dropped = filter_records(self.summary.records, quarantine, include_conflicts=True)
        self.assertEqual(dropped, 0)
        self.assertEqual(len(kept), len(self.summary.records))

    def test_empty_quarantine_changes_nothing(self) -> None:
        kept, dropped = filter_records(self.summary.records, set())
        self.assertEqual(dropped, 0)
        self.assertEqual(len(kept), len(self.summary.records))


class QuarantineManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        _write(self.root / "train" / "authentic" / "g0" / "source.jpg", 1)
        shutil.copyfile(
            self.root / "train" / "authentic" / "g0" / "source.jpg",
            self._ensure(self.root / "train" / "ai_edited" / "g0" / "edit_0.jpg"),
        )

    @staticmethod
    def _ensure(path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def test_manifest_round_trips_to_absolute_paths(self) -> None:
        summaries = [discover_split(self.root / "train", "train")]
        report = audit_splits(summaries, root=self.root)
        manifest = self.root / "quarantine.json"
        manifest.write_text(json.dumps(report.quarantine_manifest()), encoding="utf-8")
        self.assertEqual(load_quarantine(manifest), report.quarantined_paths)

    def test_missing_manifest_yields_an_empty_set(self) -> None:
        self.assertEqual(load_quarantine(self.root / "nope.json"), set())

    def test_manifest_states_the_no_guessing_policy(self) -> None:
        summaries = [discover_split(self.root / "train", "train")]
        manifest = audit_splits(summaries, root=self.root).quarantine_manifest()
        self.assertEqual(manifest["quarantined_file_count"], 2)
        self.assertIn("does not guess", manifest["policy"])


if __name__ == "__main__":
    unittest.main()
