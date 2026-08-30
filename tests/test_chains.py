"""Determinism and score-drift tests for compound transformation chains."""

from __future__ import annotations

import unittest

import numpy as np

from src.evaluation.chains import (
    CLEAN_CHAIN,
    ChainRecord,
    ChainSpec,
    apply_chain,
    build_generation_chains,
    chain_drift,
    generate_chain_variants,
)
from tests.helpers import make_image


class ChainDeterminismTest(unittest.TestCase):
    def test_seeded_generation_chain_definitions_are_reproducible(self) -> None:
        self.assertEqual(build_generation_chains(1234), build_generation_chains(1234))
        self.assertNotEqual(build_generation_chains(1234), build_generation_chains(4321))

    def test_stochastic_chain_pixels_are_reproducible(self) -> None:
        image = make_image(seed=7)
        spec = ChainSpec(
            "stochastic", ("noise_s0.02", "color_jitter", "jpeg_q70")
        )
        first = np.asarray(apply_chain(image, spec, seed=55))
        second = np.asarray(apply_chain(image, spec, seed=55))
        np.testing.assert_array_equal(first, second)

    def test_variants_always_include_untouched_clean_image(self) -> None:
        image = make_image(seed=9)
        variants, errors = generate_chain_variants(
            image, [ChainSpec("one", ("jpeg_q70",))], seed=3
        )
        self.assertEqual(errors, [])
        self.assertEqual(list(variants)[0], CLEAN_CHAIN)
        np.testing.assert_array_equal(np.asarray(variants[CLEAN_CHAIN]), np.asarray(image))

    def test_unknown_operation_is_reported_without_aborting(self) -> None:
        variants, errors = generate_chain_variants(
            make_image(), [ChainSpec("bad", ("not_an_operation",))]
        )
        self.assertEqual(list(variants), [CLEAN_CHAIN])
        self.assertEqual(errors[0]["chain"], "bad")


class ChainDriftTest(unittest.TestCase):
    def test_negative_drift_means_scores_moved_toward_authentic(self) -> None:
        records = [
            ChainRecord("a", 1, chain_scores={CLEAN_CHAIN: 0.9, "chain": 0.5}),
            ChainRecord("b", 1, chain_scores={CLEAN_CHAIN: 0.8, "chain": 0.6}),
        ]
        self.assertAlmostEqual(chain_drift(records, "chain"), -0.3)

    def test_label_filter_never_mixes_authentic_and_ai_drift(self) -> None:
        records = [
            ChainRecord("real", 0, chain_scores={CLEAN_CHAIN: 0.2, "chain": 0.4}),
            ChainRecord("ai", 1, chain_scores={CLEAN_CHAIN: 0.9, "chain": 0.3}),
        ]
        self.assertAlmostEqual(chain_drift(records, "chain", label=0), 0.2)
        self.assertAlmostEqual(chain_drift(records, "chain", label=1), -0.6)


if __name__ == "__main__":
    unittest.main()
