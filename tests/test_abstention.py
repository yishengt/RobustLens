"""Tests for evidence-driven abstention.

The rule these pin down is the one the audit found broken: an image whose score
collapsed because it was degraded must not be reported as confidently authentic.
"""

from __future__ import annotations

import unittest

from src.pipeline.abstention import (
    ALL_RULES,
    RULE_BORDERLINE,
    RULE_BOUNDARY_CROSSING,
    RULE_EVIDENCE_UNAVAILABLE,
    RULE_LOW_AGREEMENT,
    RULE_LOW_CONSISTENCY,
    RULE_LOW_PATCH_COVERAGE,
    RULE_TRANSFORMED_DRIFT,
    abstention_settings,
    evaluate_abstention,
    would_report_confident_authentic_after_degradation,
)
from src.pipeline.prediction import LABEL_AI, LABEL_AUTHENTIC, LABEL_UNCERTAIN

ON = {"abstention": {"enabled": True}}
THRESHOLD = 0.69


def _decide(**kwargs):
    params = {
        "label": LABEL_AI,
        "final_probability": 0.95,
        "threshold": THRESHOLD,
        "config": ON,
    }
    params.update(kwargs)
    return evaluate_abstention(**params)


class SettingsTest(unittest.TestCase):
    def test_disabled_by_default(self) -> None:
        self.assertFalse(abstention_settings({})["enabled"])

    def test_out_of_range_threshold_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            abstention_settings({"abstention": {"min_consistency": 1.5}})

    def test_unknown_rule_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as caught:
            abstention_settings({"abstention": {"rules": ["not_a_rule"]}})
        self.assertIn("not_a_rule", str(caught.exception))

    def test_empty_rule_list_means_no_rules(self) -> None:
        """Needed so a baseline 'never abstain' run is expressible."""

        self.assertEqual(abstention_settings({"abstention": {"rules": []}})["rules"], [])

    def test_missing_rule_key_means_every_rule(self) -> None:
        self.assertEqual(set(abstention_settings({})["rules"]), set(ALL_RULES))


class RuleTest(unittest.TestCase):
    def test_stable_confident_prediction_does_not_abstain(self) -> None:
        decision = _decide(
            clean_probability=0.95,
            transformed_probabilities=[0.94, 0.95, 0.93],
            consistency_score=0.98,
            agreement=1.0,
        )
        self.assertFalse(decision.abstain)
        self.assertEqual(decision.label, LABEL_AI)

    def test_stable_authentic_prediction_does_not_abstain(self) -> None:
        decision = _decide(
            label=LABEL_AUTHENTIC,
            final_probability=0.05,
            clean_probability=0.05,
            transformed_probabilities=[0.04, 0.06, 0.05],
            consistency_score=0.99,
            agreement=1.0,
        )
        self.assertFalse(decision.abstain)
        self.assertEqual(decision.label, LABEL_AUTHENTIC)

    def test_severe_degradation_forces_abstention(self) -> None:
        """The core defect: AI-like when clean, collapsing once transformed."""

        decision = _decide(
            final_probability=0.45,
            clean_probability=0.92,
            transformed_probabilities=[0.30, 0.25, 0.20],
            consistency_score=0.9,
            agreement=1.0,
        )
        self.assertTrue(decision.abstain)
        self.assertEqual(decision.label, LABEL_UNCERTAIN)
        self.assertIn(RULE_TRANSFORMED_DRIFT, decision.triggered_rules)

    def test_borderline_probability_abstains(self) -> None:
        decision = _decide(
            final_probability=THRESHOLD + 0.01,
            clean_probability=0.70,
            transformed_probabilities=[0.70, 0.70],
            consistency_score=1.0,
            agreement=1.0,
        )
        self.assertTrue(decision.abstain)
        self.assertIn(RULE_BORDERLINE, decision.triggered_rules)

    def test_low_consistency_abstains(self) -> None:
        decision = _decide(
            clean_probability=0.95,
            transformed_probabilities=[0.95, 0.95],
            consistency_score=0.05,
            agreement=1.0,
        )
        self.assertTrue(decision.abstain)
        self.assertIn(RULE_LOW_CONSISTENCY, decision.triggered_rules)

    def test_versions_crossing_both_boundaries_abstain(self) -> None:
        decision = _decide(
            final_probability=0.80,
            clean_probability=0.95,
            transformed_probabilities=[0.95, 0.20, 0.15, 0.90],
            consistency_score=0.9,
            agreement=1.0,
        )
        self.assertTrue(decision.abstain)
        self.assertIn(RULE_BOUNDARY_CROSSING, decision.triggered_rules)

    def test_low_agreement_abstains(self) -> None:
        decision = _decide(
            clean_probability=0.95,
            transformed_probabilities=[0.95, 0.95],
            consistency_score=1.0,
            agreement=0.1,
        )
        self.assertTrue(decision.abstain)
        self.assertIn(RULE_LOW_AGREEMENT, decision.triggered_rules)

    def test_low_patch_coverage_abstains_only_when_patches_ran(self) -> None:
        stable = {
            "clean_probability": 0.95,
            "transformed_probabilities": [0.95, 0.95],
            "consistency_score": 1.0,
            "agreement": 1.0,
        }
        ran = _decide(patch_available=True, patch_coverage=0.01, **stable)
        self.assertTrue(ran.abstain)
        self.assertIn(RULE_LOW_PATCH_COVERAGE, ran.triggered_rules)

        # Patch analysis being unavailable is missing evidence, not bad evidence.
        skipped = _decide(patch_available=False, patch_coverage=None, **stable)
        self.assertFalse(skipped.abstain)

    def test_missing_evidence_alone_never_abstains(self) -> None:
        decision = _decide(final_probability=0.95)
        self.assertFalse(decision.abstain)

    def test_evidence_errors_abstain_only_when_configured(self) -> None:
        errors = [{"stage": "transformation", "error": "boom"}]
        off = _decide(
            clean_probability=0.95,
            transformed_probabilities=[0.95],
            consistency_score=1.0,
            agreement=1.0,
            errors=errors,
        )
        self.assertFalse(off.abstain)

        on = evaluate_abstention(
            label=LABEL_AI,
            final_probability=0.95,
            threshold=THRESHOLD,
            clean_probability=0.95,
            transformed_probabilities=[0.95],
            consistency_score=1.0,
            agreement=1.0,
            errors=errors,
            config={"abstention": {"enabled": True, "abstain_on_evidence_errors": True}},
        )
        self.assertTrue(on.abstain)
        self.assertIn(RULE_EVIDENCE_UNAVAILABLE, on.triggered_rules)

    def test_patch_failures_do_not_count_as_evidence_errors(self) -> None:
        """Patch analysis is explainability; its failure must not force Uncertain."""

        decision = evaluate_abstention(
            label=LABEL_AI,
            final_probability=0.95,
            threshold=THRESHOLD,
            clean_probability=0.95,
            transformed_probabilities=[0.95],
            consistency_score=1.0,
            agreement=1.0,
            errors=[{"stage": "patches", "error": "too small"}],
            config={"abstention": {"enabled": True, "abstain_on_evidence_errors": True}},
        )
        self.assertFalse(decision.abstain)


class SafetyTest(unittest.TestCase):
    def test_disabled_abstention_leaves_the_label_alone(self) -> None:
        decision = evaluate_abstention(
            label=LABEL_AI,
            final_probability=0.95,
            threshold=THRESHOLD,
            clean_probability=0.95,
            transformed_probabilities=[0.1, 0.1],
            consistency_score=0.0,
            agreement=0.0,
            config={"abstention": {"enabled": False}},
        )
        self.assertFalse(decision.abstain)
        self.assertEqual(decision.label, LABEL_AI)

    def test_abstention_never_flips_to_the_opposite_class(self) -> None:
        for label in (LABEL_AI, LABEL_AUTHENTIC):
            decision = _decide(
                label=label,
                final_probability=0.5,
                clean_probability=0.95,
                transformed_probabilities=[0.1, 0.1],
                consistency_score=0.0,
                agreement=0.0,
            )
            self.assertEqual(decision.label, LABEL_UNCERTAIN)
            self.assertEqual(decision.original_label, label)

    def test_decision_is_json_serialisable_and_explains_itself(self) -> None:
        import json

        decision = _decide(
            final_probability=0.45,
            clean_probability=0.92,
            transformed_probabilities=[0.2, 0.2],
            consistency_score=0.9,
            agreement=1.0,
        )
        payload = decision.as_dict()
        self.assertIsInstance(json.dumps(payload), str)
        self.assertTrue(payload["abstained"])
        self.assertIn("Uncertain", payload["statement"])


class DegradationHelperTest(unittest.TestCase):
    def test_detects_degradation_driven_flip(self) -> None:
        self.assertTrue(
            would_report_confident_authentic_after_degradation(0.92, [0.2, 0.25], THRESHOLD)
        )

    def test_ignores_images_that_were_never_ai_like(self) -> None:
        self.assertFalse(
            would_report_confident_authentic_after_degradation(0.10, [0.05, 0.04], THRESHOLD)
        )

    def test_ignores_stable_ai_predictions(self) -> None:
        self.assertFalse(
            would_report_confident_authentic_after_degradation(0.95, [0.93, 0.94], THRESHOLD)
        )

    def test_no_transformed_versions_is_not_a_flip(self) -> None:
        self.assertFalse(
            would_report_confident_authentic_after_degradation(0.95, [], THRESHOLD)
        )


if __name__ == "__main__":
    unittest.main()
