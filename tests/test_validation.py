"""Stage 2 tests: input validation, metadata capture and RGB conversion."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.pipeline.validation import (
    ImageMetadata,
    ImageValidationError,
    list_supported_images,
    load_validated_image,
    validate_image_bytes,
    validate_image_file,
)
from tests.helpers import (
    base_config,
    make_image,
    write_corrupted_image,
    write_image,
    write_truncated_jpeg,
)


class ValidImageTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.config = base_config()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_valid_jpeg_returns_complete_metadata(self) -> None:
        path = write_image(self.tmp, "photo.jpg", width=120, height=80)
        metadata = validate_image_file(path, self.config)

        self.assertIsInstance(metadata, ImageMetadata)
        self.assertEqual(metadata.filename, "photo.jpg")
        self.assertEqual(metadata.file_path, str(path))
        self.assertEqual(metadata.file_type, "JPEG")
        self.assertEqual((metadata.width, metadata.height), (120, 80))
        self.assertEqual(metadata.color_mode, "RGB")
        self.assertGreater(metadata.file_size_bytes, 0)
        self.assertIn("KB", metadata.file_size_human)

    def test_every_supported_format_validates(self) -> None:
        for name, image_format in [
            ("a.jpg", "JPEG"),
            ("b.jpeg", "JPEG"),
            ("c.png", "PNG"),
            ("d.webp", "WEBP"),
        ]:
            with self.subTest(name=name):
                path = write_image(self.tmp, name, image_format=image_format)
                metadata = validate_image_file(path, self.config)
                self.assertEqual(metadata.file_type, image_format)

    def test_load_returns_rgb_image_at_full_resolution(self) -> None:
        path = write_image(self.tmp, "photo.png", width=100, height=70, image_format="PNG")
        image, metadata = load_validated_image(path, self.config)

        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (100, 70))
        self.assertEqual(metadata.width, 100)

    def test_greyscale_image_is_converted_to_rgb(self) -> None:
        path = self.tmp / "grey.png"
        make_image(mode="L").save(path, format="PNG")

        metadata = validate_image_file(path, self.config)
        self.assertEqual(metadata.color_mode, "L")  # original mode is recorded

        image, _ = load_validated_image(path, self.config)
        self.assertEqual(image.mode, "RGB")  # but the pipeline receives RGB

    def test_rgba_png_is_converted_to_rgb(self) -> None:
        path = self.tmp / "alpha.png"
        make_image(mode="RGBA").save(path, format="PNG")

        image, metadata = load_validated_image(path, self.config)
        self.assertEqual(metadata.color_mode, "RGBA")
        self.assertEqual(image.mode, "RGB")


class InvalidImageTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.config = base_config()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_missing_file_reports_the_path(self) -> None:
        missing = self.tmp / "nope.jpg"
        with self.assertRaises(ImageValidationError) as context:
            validate_image_file(missing, self.config)
        self.assertIn("does not exist", str(context.exception))

    def test_unsupported_extension_is_rejected(self) -> None:
        path = self.tmp / "notes.txt"
        path.write_text("not an image", encoding="utf-8")
        with self.assertRaises(ImageValidationError) as context:
            validate_image_file(path, self.config)
        self.assertIn("unsupported file type", str(context.exception).lower())

    def test_empty_file_is_rejected(self) -> None:
        path = self.tmp / "empty.jpg"
        path.write_bytes(b"")
        with self.assertRaises(ImageValidationError) as context:
            validate_image_file(path, self.config)
        self.assertIn("empty", str(context.exception).lower())

    def test_corrupted_image_is_rejected(self) -> None:
        path = write_corrupted_image(self.tmp)
        with self.assertRaises(ImageValidationError) as context:
            validate_image_file(path, self.config)
        self.assertIn("corrupted", str(context.exception).lower())

    def test_truncated_jpeg_is_rejected(self) -> None:
        path = write_truncated_jpeg(self.tmp)
        with self.assertRaises(ImageValidationError):
            validate_image_file(path, self.config)

    def test_directory_path_is_rejected(self) -> None:
        directory = self.tmp / "folder.png"
        directory.mkdir()
        with self.assertRaises(ImageValidationError):
            validate_image_file(directory, self.config)

    def test_image_below_minimum_size_is_rejected(self) -> None:
        path = write_image(self.tmp, "tiny.png", width=8, height=8, image_format="PNG")
        with self.assertRaises(ImageValidationError) as context:
            validate_image_file(path, self.config)
        self.assertIn("too small", str(context.exception))

    def test_image_above_maximum_size_is_rejected(self) -> None:
        config = base_config()
        config["validation"]["max_side"] = 50
        path = write_image(self.tmp, "big.png", width=120, height=60, image_format="PNG")
        with self.assertRaises(ImageValidationError) as context:
            validate_image_file(path, config)
        self.assertIn("too large", str(context.exception))

    def test_pixel_budget_is_enforced(self) -> None:
        config = base_config()
        config["validation"]["max_pixels"] = 100
        path = write_image(self.tmp, "wide.png", width=120, height=60, image_format="PNG")
        with self.assertRaises(ImageValidationError) as context:
            validate_image_file(path, config)
        self.assertIn("pixel safety limit", str(context.exception))

    def test_file_size_limit_is_enforced(self) -> None:
        config = base_config()
        config["validation"]["max_file_size_mb"] = 0.000_001
        path = write_image(self.tmp, "heavy.png", image_format="PNG")
        with self.assertRaises(ImageValidationError) as context:
            validate_image_file(path, config)
        self.assertIn("above the", str(context.exception))

    def test_error_message_always_names_the_file(self) -> None:
        with self.assertRaises(ImageValidationError) as context:
            validate_image_file(self.tmp / "ghost.jpg", self.config)
        self.assertIn("ghost.jpg", str(context.exception))


class UploadValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = base_config()

    def test_valid_upload_bytes_are_accepted(self) -> None:
        import io

        buffer = io.BytesIO()
        make_image(width=64, height=48).save(buffer, format="PNG")
        image, metadata = validate_image_bytes(buffer.getvalue(), "upload.png", self.config)

        self.assertEqual(image.mode, "RGB")
        self.assertEqual((metadata.width, metadata.height), (64, 48))
        self.assertEqual(metadata.file_type, "PNG")

    def test_empty_upload_is_rejected(self) -> None:
        with self.assertRaises(ImageValidationError):
            validate_image_bytes(b"", "upload.png", self.config)

    def test_corrupted_upload_is_rejected(self) -> None:
        with self.assertRaises(ImageValidationError):
            validate_image_bytes(b"\x89PNG\r\n\x1a\n garbage", "upload.png", self.config)


class DirectoryListingTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.config = base_config()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_lists_only_supported_files_recursively(self) -> None:
        write_image(self.tmp, "a.jpg")
        write_image(self.tmp / "nested", "b.png", image_format="PNG")
        write_image(self.tmp, "c.webp", image_format="WEBP")
        (self.tmp / "d.txt").write_text("ignore me", encoding="utf-8")
        (self.tmp / ".hidden.jpg").write_bytes(b"skip")

        found = list_supported_images(self.tmp, self.config)
        names = sorted(path.name for path in found)
        self.assertEqual(names, ["a.jpg", "b.png", "c.webp"])

    def test_missing_directory_raises(self) -> None:
        with self.assertRaises(ImageValidationError):
            list_supported_images(self.tmp / "absent", self.config)


if __name__ == "__main__":
    unittest.main()
