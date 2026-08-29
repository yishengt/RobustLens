"""Stage 4 tests: one test per transformation family, plus spec construction."""

from __future__ import annotations

import unittest

import numpy as np

from src.pipeline.transformations import (
    ORIGINAL_KEY,
    build_transform_specs,
    center_crop,
    color_jitter,
    downscale_upscale,
    gaussian_blur,
    gaussian_noise,
    generate_variants,
    jpeg_compress,
)
from tests.helpers import base_config, make_image


def mean_abs_difference(first, second) -> float:
    """Mean absolute pixel difference between two same-sized images."""

    a = np.asarray(first.convert("RGB"), dtype=np.float32)
    b = np.asarray(second.convert("RGB"), dtype=np.float32)
    return float(np.abs(a - b).mean())


class IndividualTransformTest(unittest.TestCase):
    def setUp(self) -> None:
        self.image = make_image(width=128, height=96, seed=1)

    def test_jpeg_compression_changes_pixels_and_keeps_size(self) -> None:
        previous = 0.0
        for quality in (90, 70, 50, 30):
            with self.subTest(quality=quality):
                result = jpeg_compress(self.image, quality)
                self.assertEqual(result.size, self.image.size)
                self.assertEqual(result.mode, "RGB")
                difference = mean_abs_difference(self.image, result)
                self.assertGreater(difference, 0.0)
                previous = difference
        # Quality 30 should distort more than quality 90.
        self.assertGreater(
            mean_abs_difference(self.image, jpeg_compress(self.image, 30)),
            mean_abs_difference(self.image, jpeg_compress(self.image, 90)),
        )
        self.assertGreater(previous, 0.0)

    def test_jpeg_rejects_invalid_quality(self) -> None:
        for quality in (0, 101, -5):
            with self.subTest(quality=quality), self.assertRaises(ValueError):
                jpeg_compress(self.image, quality)

    def test_gaussian_blur_increases_with_sigma(self) -> None:
        differences = []
        for sigma in (0.5, 1.0, 2.0):
            result = gaussian_blur(self.image, sigma)
            self.assertEqual(result.size, self.image.size)
            differences.append(mean_abs_difference(self.image, result))
        self.assertTrue(all(value > 0 for value in differences))
        self.assertEqual(differences, sorted(differences))

    def test_blur_rejects_negative_sigma(self) -> None:
        with self.assertRaises(ValueError):
            gaussian_blur(self.image, -1.0)

    def test_downscale_upscale_restores_size_and_loses_detail(self) -> None:
        for scale in (0.5, 0.25):
            with self.subTest(scale=scale):
                result = downscale_upscale(self.image, scale)
                self.assertEqual(result.size, self.image.size)
                self.assertGreater(mean_abs_difference(self.image, result), 0.0)
        self.assertGreater(
            mean_abs_difference(self.image, downscale_upscale(self.image, 0.25)),
            mean_abs_difference(self.image, downscale_upscale(self.image, 0.5)),
        )

    def test_resize_rejects_invalid_scale(self) -> None:
        for scale in (0.0, -0.5, 1.5):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                downscale_upscale(self.image, scale)

    def test_gaussian_noise_scales_with_sigma(self) -> None:
        rng = np.random.default_rng(0)
        differences = []
        for sigma in (0.02, 0.05, 0.10):
            result = gaussian_noise(self.image, sigma, np.random.default_rng(0))
            self.assertEqual(result.size, self.image.size)
            differences.append(mean_abs_difference(self.image, result))
        self.assertEqual(differences, sorted(differences))
        self.assertGreater(differences[0], 0.0)
        self.assertIsNotNone(rng)

    def test_noise_is_reproducible_for_a_fixed_generator(self) -> None:
        first = gaussian_noise(self.image, 0.05, np.random.default_rng(42))
        second = gaussian_noise(self.image, 0.05, np.random.default_rng(42))
        self.assertEqual(mean_abs_difference(first, second), 0.0)

    def test_colour_jitter_changes_the_image_within_bounds(self) -> None:
        result = color_jitter(self.image, 0.2, 0.2, 0.2, np.random.default_rng(0))
        self.assertEqual(result.size, self.image.size)
        self.assertGreater(mean_abs_difference(self.image, result), 0.0)

    def test_colour_jitter_rejects_out_of_range_limits(self) -> None:
        with self.assertRaises(ValueError):
            color_jitter(self.image, brightness=1.5)

    def test_centre_crop_keeps_the_configured_fraction(self) -> None:
        result = center_crop(self.image, 0.8)
        self.assertEqual(result.size, (round(128 * 0.8), round(96 * 0.8)))

    def test_centre_crop_is_centred(self) -> None:
        cropped = center_crop(self.image, 0.5)
        expected = self.image.crop((32, 24, 32 + 64, 24 + 48))
        self.assertEqual(mean_abs_difference(cropped, expected), 0.0)

    def test_centre_crop_rejects_invalid_fraction(self) -> None:
        for fraction in (0.0, -0.2, 1.5):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                center_crop(self.image, fraction)


class TransformSpecTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = base_config()

    def test_default_config_builds_all_fourteen_transforms(self) -> None:
        specs = build_transform_specs(self.config)
        names = [spec.name for spec in specs]
        self.assertEqual(
            names,
            [
                "jpeg_q90",
                "jpeg_q70",
                "jpeg_q50",
                "jpeg_q30",
                "blur_s0.5",
                "blur_s1",
                "blur_s2",
                "resize_0.5x",
                "resize_0.25x",
                "noise_s0.02",
                "noise_s0.05",
                "noise_s0.1",
                "color_jitter",
                "center_crop_80",
            ],
        )

    def test_transforms_are_configurable(self) -> None:
        config = base_config()
        config["transformations"]["jpeg_qualities"] = [80]
        config["transformations"]["blur_sigmas"] = []
        config["transformations"]["resize_scales"] = []
        config["transformations"]["noise_sigmas"] = []
        config["transformations"]["color_jitter"]["variants"] = 0
        config["transformations"]["center_crop_fraction"] = None

        names = [spec.name for spec in build_transform_specs(config)]
        self.assertEqual(names, ["jpeg_q80"])

    def test_disabling_transformations_yields_no_specs(self) -> None:
        config = base_config()
        config["transformations"]["enabled"] = False
        self.assertEqual(build_transform_specs(config), [])


class VariantGenerationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = base_config()
        self.image = make_image(width=128, height=96, seed=2)

    def test_variants_include_the_original_first(self) -> None:
        variants, errors = generate_variants(self.image, self.config)
        self.assertEqual(errors, [])
        self.assertEqual(next(iter(variants)), ORIGINAL_KEY)
        self.assertEqual(len(variants), 15)  # original + 14 transformations

    def test_every_variant_differs_from_the_original(self) -> None:
        variants, _ = generate_variants(self.image, self.config)
        original = variants[ORIGINAL_KEY]
        for name, image in variants.items():
            if name == ORIGINAL_KEY:
                continue
            with self.subTest(transform=name):
                self.assertEqual(image.mode, "RGB")
                if image.size == original.size:
                    self.assertGreater(mean_abs_difference(original, image), 0.0)

    def test_generation_is_reproducible_with_a_fixed_seed(self) -> None:
        first, _ = generate_variants(self.image, self.config)
        second, _ = generate_variants(self.image, self.config)
        for name in first:
            with self.subTest(transform=name):
                self.assertEqual(mean_abs_difference(first[name], second[name]), 0.0)

    def test_different_seeds_change_the_stochastic_transforms(self) -> None:
        first, _ = generate_variants(self.image, self.config, seed=1)
        second, _ = generate_variants(self.image, self.config, seed=999)
        self.assertGreater(mean_abs_difference(first["noise_s0.05"], second["noise_s0.05"]), 0.0)
        # Deterministic transforms must be unaffected by the seed.
        self.assertEqual(mean_abs_difference(first["jpeg_q70"], second["jpeg_q70"]), 0.0)

    def test_disabled_transformations_return_only_the_original(self) -> None:
        config = base_config()
        config["transformations"]["enabled"] = False
        variants, errors = generate_variants(self.image, config)
        self.assertEqual(list(variants), [ORIGINAL_KEY])
        self.assertEqual(errors, [])

    def test_none_image_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generate_variants(None, self.config)


if __name__ == "__main__":
    unittest.main()
