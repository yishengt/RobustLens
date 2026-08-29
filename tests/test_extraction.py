"""Tests for SID_Set extraction: raw-byte fidelity and the on-disk layout.

A small parquet file is built in memory, so nothing is downloaded. The point of
these tests is that extraction is lossless -- re-encoding would resample the
compression artefacts AI-image detection depends on.
"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import List

from PIL import Image

from src.evaluation.sid_set import (
    CLASS_NAMES,
    RawRecord,
    find_shards,
    iter_raw_records,
    to_binary_label,
)
from tests.helpers import make_image


def encoded(image: Image.Image, image_format: str) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def write_parquet(path: Path, rows: List[dict]) -> Path:
    """Write a parquet shard shaped like a real SID_Set file."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "img_id": pa.array([r["img_id"] for r in rows], pa.string()),
            "image": pa.array(
                [{"bytes": r["bytes"], "path": None} for r in rows],
                pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
            ),
            "label": pa.array([r["label"] for r in rows], pa.int64()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


class RawRecordTest(unittest.TestCase):
    def test_extension_per_format(self) -> None:
        for image_format, expected in [("JPEG", ".jpg"), ("PNG", ".png"), ("WEBP", ".webp")]:
            with self.subTest(format=image_format):
                record = RawRecord(b"", image_format, 0, 0, "id", "shard")
                self.assertEqual(record.extension, expected)

    def test_unknown_format_falls_back(self) -> None:
        self.assertEqual(RawRecord(b"", "TIFF", 0, 0, "i", "s").extension, ".bin")

    def test_class_name_mapping(self) -> None:
        for label, name in CLASS_NAMES.items():
            with self.subTest(label=label):
                self.assertEqual(RawRecord(b"", "JPEG", label, 0, "i", "s").class_name, name)


class RawExtractionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.payloads = {
            "real_aaa": (encoded(make_image(seed=1), "JPEG"), "JPEG", 0),
            "full_synthetic_001": (encoded(make_image(seed=2), "PNG"), "PNG", 1),
            "tampered_001": (encoded(make_image(seed=3), "JPEG"), "JPEG", 2),
        }
        write_parquet(
            self.tmp / "data" / "validation-00000-of-00001.parquet",
            [
                {"img_id": key, "bytes": value[0], "label": value[2]}
                for key, value in self.payloads.items()
            ],
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def shards(self) -> List[Path]:
        return find_shards(self.tmp, "validation")

    def test_finds_the_shard(self) -> None:
        self.assertEqual(len(self.shards()), 1)

    def test_bytes_are_returned_verbatim(self) -> None:
        """The whole point: extraction must not re-encode."""

        for record in iter_raw_records(self.shards()):
            with self.subTest(img_id=record.img_id):
                original = self.payloads[record.img_id][0]
                self.assertEqual(
                    hashlib.sha256(record.data).hexdigest(),
                    hashlib.sha256(original).hexdigest(),
                )

    def test_original_format_is_detected(self) -> None:
        for record in iter_raw_records(self.shards()):
            with self.subTest(img_id=record.img_id):
                self.assertEqual(record.image_format, self.payloads[record.img_id][1])

    def test_binary_labels_collapse_the_three_classes(self) -> None:
        mapping = {r.img_id: r.binary_label for r in iter_raw_records(self.shards())}
        self.assertEqual(mapping["real_aaa"], 0)
        self.assertEqual(mapping["full_synthetic_001"], 1)
        self.assertEqual(mapping["tampered_001"], 1)

    def test_limit_is_respected(self) -> None:
        self.assertEqual(len(list(iter_raw_records(self.shards(), limit=2))), 2)

    def test_per_class_limit_balances_the_sample(self) -> None:
        records = list(iter_raw_records(self.shards(), per_class_limit=1))
        self.assertEqual(len(records), 3)  # one of each source class

    def test_extracted_files_reopen_as_images(self) -> None:
        for record in iter_raw_records(self.shards()):
            with self.subTest(img_id=record.img_id):
                with Image.open(io.BytesIO(record.data)) as image:
                    image.load()
                    self.assertEqual(image.format, record.image_format)

    def test_rows_with_unreadable_bytes_are_skipped(self) -> None:
        write_parquet(
            self.tmp / "data" / "validation-00001-of-00002.parquet",
            [{"img_id": "broken", "bytes": b"not an image at all", "label": 0}],
        )
        ids = {r.img_id for r in iter_raw_records(find_shards(self.tmp, "validation"))}
        self.assertNotIn("broken", ids)
        self.assertIn("real_aaa", ids)


class ExtractionScriptTest(unittest.TestCase):
    """The CLI's layout contract: class folders plus a labels.json manifest."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        write_parquet(
            self.tmp / "shards" / "data" / "validation-00000-of-00001.parquet",
            [
                {"img_id": "aaa111", "bytes": encoded(make_image(seed=1), "JPEG"), "label": 0},
                {
                    "img_id": "full_synthetic_007",
                    "bytes": encoded(make_image(seed=2), "PNG"),
                    "label": 1,
                },
                {
                    "img_id": "tampered_009",
                    "bytes": encoded(make_image(seed=3), "JPEG"),
                    "label": 2,
                },
            ],
        )
        self.output = self.tmp / "out"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_script(self, *extra: str) -> int:
        import scripts.extract_dataset as cli

        return cli.main(
            [
                "--data-dir",
                str(self.tmp / "shards"),
                "--output",
                str(self.output),
                "--quiet",
                *extra,
            ]
        )

    def test_creates_binary_class_folders(self) -> None:
        self.assertEqual(self.run_script(), 0)
        self.assertTrue((self.output / "real").is_dir())
        self.assertTrue((self.output / "ai_generated").is_dir())
        self.assertEqual(len(list((self.output / "real").glob("*"))), 1)
        self.assertEqual(len(list((self.output / "ai_generated").glob("*"))), 2)

    def test_class_prefix_is_not_duplicated(self) -> None:
        self.run_script()
        names = sorted(p.name for p in (self.output / "ai_generated").glob("*"))
        self.assertEqual(names, ["full_synthetic_007.png", "tampered_009.jpg"])

    def test_bare_ids_receive_a_class_prefix(self) -> None:
        self.run_script()
        self.assertEqual([p.name for p in (self.output / "real").glob("*")], ["real_aaa111.jpg"])

    def test_manifest_structure(self) -> None:
        self.run_script()
        manifest = json.loads((self.output / "labels.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["dataset"], "SID_Set")
        self.assertEqual(manifest["count"], 3)
        self.assertEqual(manifest["class_distribution"]["real"], 1)
        for entry in manifest["images"]:
            with self.subTest(entry=entry["img_id"]):
                for field in ("image_path", "img_id", "label", "source_label", "class_name"):
                    self.assertIn(field, entry)
                self.assertIn(entry["label"], (0, 1))
                self.assertEqual(entry["label"], to_binary_label(entry["source_label"]))
                self.assertTrue((self.output / entry["image_path"]).is_file())

    def test_manifest_paths_are_relative(self) -> None:
        self.run_script()
        manifest = json.loads((self.output / "labels.json").read_text(encoding="utf-8"))
        for entry in manifest["images"]:
            with self.subTest(path=entry["image_path"]):
                self.assertFalse(Path(entry["image_path"]).is_absolute())

    def test_extracted_bytes_match_the_shard(self) -> None:
        self.run_script()
        by_id = {r.img_id: r.data for r in iter_raw_records(find_shards(self.tmp / "shards"))}
        manifest = json.loads((self.output / "labels.json").read_text(encoding="utf-8"))
        for entry in manifest["images"]:
            with self.subTest(img_id=entry["img_id"]):
                on_disk = (self.output / entry["image_path"]).read_bytes()
                self.assertEqual(on_disk, by_id[entry["img_id"]])

    def test_rerun_is_idempotent(self) -> None:
        self.assertEqual(self.run_script(), 0)
        first = {p.name for p in (self.output / "real").glob("*")}
        self.assertEqual(self.run_script(), 0)
        self.assertEqual({p.name for p in (self.output / "real").glob("*")}, first)

    def test_missing_data_directory_returns_error_code(self) -> None:
        import scripts.extract_dataset as cli

        code = cli.main(
            ["--data-dir", str(self.tmp / "absent"), "--output", str(self.output), "--quiet"]
        )
        self.assertEqual(code, 4)


if __name__ == "__main__":
    unittest.main()
