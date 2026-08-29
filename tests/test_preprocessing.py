"""Stage 3 tests: RGB conversion, resizing, normalization and tensor output."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.pipeline.preprocessing import Preprocessor, preserve_original
from tests.helpers import base_config, make_image


class PreprocessorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = base_config()
        self.preprocessor = Preprocessor.from_config(self.config)

    def test_built_from_config(self) -> None:
        self.assertEqual(self.preprocessor.image_size, 224)
        self.assertEqual(self.preprocessor.mean, (0.485, 0.456, 0.406))
        self.assertEqual(self.preprocessor.std, (0.229, 0.224, 0.225))

    def test_rgb_conversion(self) -> None:
        for mode in ("L", "RGBA", "CMYK"):
            with self.subTest(mode=mode):
                converted = Preprocessor.to_rgb(make_image(mode=mode))
                self.assertEqual(converted.mode, "RGB")

    def test_resize_to_224_square(self) -> None:
        for width, height in [(96, 64), (500, 500), (17, 233), (224, 224)]:
            with self.subTest(size=(width, height)):
                resized = self.preprocessor.resize(make_image(width=width, height=height))
                self.assertEqual(resized.size, (224, 224))

    def test_resize_honours_configured_image_size(self) -> None:
        preprocessor = Preprocessor.from_config(base_config(image_size=128))
        self.assertEqual(preprocessor.resize(make_image()).size, (128, 128))

    def test_tensor_shape_and_dtype(self) -> None:
        tensor = self.preprocessor(make_image(width=300, height=200))
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_normalization_is_applied(self) -> None:
        # A mid-grey image normalizes to a predictable per-channel value.
        from PIL import Image

        grey = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8), mode="RGB")
        tensor = self.preprocessor(grey)
        expected = [
            (128 / 255.0 - mean) / std
            for mean, std in zip(self.preprocessor.mean, self.preprocessor.std)
        ]
        for channel, value in enumerate(expected):
            self.assertAlmostEqual(float(tensor[channel].mean()), value, places=4)

    def test_tensor_values_leave_the_raw_0_1_range(self) -> None:
        tensor = self.preprocessor(make_image())
        # Normalization shifts values outside [0, 1]; a tensor still inside it
        # would mean normalization silently did nothing.
        self.assertTrue(float(tensor.min()) < 0.0)

    def test_batch_stacks_images(self) -> None:
        images = [make_image(seed=index) for index in range(4)]
        batch = self.preprocessor.batch(images)
        self.assertEqual(tuple(batch.shape), (4, 3, 224, 224))

    def test_batch_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            self.preprocessor.batch([])

    def test_none_image_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.preprocessor(None)

    def test_different_images_give_different_tensors(self) -> None:
        first = self.preprocessor(make_image(seed=1))
        second = self.preprocessor(make_image(seed=2))
        self.assertGreater(float((first - second).abs().sum()), 0.0)

    def test_preprocessing_is_deterministic(self) -> None:
        image = make_image(seed=7)
        self.assertTrue(torch.equal(self.preprocessor(image), self.preprocessor(image)))

    def test_denormalize_round_trip(self) -> None:
        image = make_image(width=224, height=224, seed=5)
        restored = self.preprocessor.denormalize(self.preprocessor(image))
        self.assertEqual(restored.shape, (224, 224, 3))
        self.assertEqual(restored.dtype, np.uint8)
        # Round-tripping through float normalization is lossy by at most a step.
        original = np.asarray(image, dtype=np.int16)
        self.assertLess(float(np.abs(original - restored.astype(np.int16)).mean()), 2.0)

    def test_preserve_original_returns_independent_copy(self) -> None:
        image = make_image()
        preserved = preserve_original(image)
        self.assertEqual(preserved.mode, "RGB")
        self.assertEqual(preserved.size, image.size)
        self.assertIsNot(preserved, image)

    def test_invalid_normalization_is_rejected(self) -> None:
        config = base_config()
        config["normalization"]["std"] = [0.0, 0.1, 0.1]
        with self.assertRaises(ValueError):
            Preprocessor.from_config(config)

    def test_invalid_image_size_is_rejected(self) -> None:
        config = base_config()
        config["data"]["image_size"] = 0
        with self.assertRaises(ValueError):
            Preprocessor.from_config(config)


if __name__ == "__main__":
    unittest.main()
