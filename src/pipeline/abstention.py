"""Evidence-driven abstention.

A probability band around the threshold is not enough. The measured failure it
misses is specific: accumulating transformations push scores *downward* for
every origin, so a heavily degraded synthetic image does not drift into the
uncertain band and stop there -- it passes straight through and out the bottom,
and the system reports "likely authentic" with rising confidence as the evidence
is destroyed. Confidence is highest exactly where the evidence is weakest.

This module adds the rules a band cannot express. Each looks at a different way
the evidence can fail, and any one of them firing turns the verdict into
``Uncertain``:

* the score sits too close to the decision threshold;
* the clean image looked AI-like but transformed copies fall away sharply;
* the score is unstable across transformations;
* transformed versions disagree about which side of the threshold they are on;
* patch coverage is too thin to support a spatial claim;
* evidence a rule depended on was unavailable or failed.

Abstention never flips an image to the opposite class. It only withdraws a
claim, which is the one action that is always safe when evidence is thin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.pipeline.prediction import LABEL_AI, LABEL_UNCERTAIN

RULE_BORDERLINE = "borderline_probability"
RULE_TRANSFORMED_DRIFT = "clean_to_transformed_drift"
RULE_LOW_CONSISTENCY = "low_transformation_consistency"
RULE_BOUNDARY_CROSSING = "transformed_versions_cross_the_threshold"
RULE_LOW_AGREEMENT = "low_agreement_between_versions"
RULE_LOW_PATCH_COVERAGE = "insufficient_patch_coverage"
RULE_EVIDENCE_UNAVAILABLE = "evidence_unavailable"

ALL_RULES = (
    RULE_BORDERLINE,
    RULE_TRANSFORMED_DRIFT,
    RULE_LOW_CONSISTENCY,
    RULE_BOUNDARY_CROSSING,
    RULE_LOW_AGREEMENT,
    RULE_LOW_PATCH_COVERAGE,
    RULE_EVIDENCE_UNAVAILABLE,
)

# Defaults are deliberately permissive so enabling abstention cannot silently
# turn a working detector into one that abstains on everything. Fit them on
# validation data with scripts/select_abstention_thresholds.py.
DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "borderline_margin": 0.05,
    "max_transformed_drift": 0.25,
    "min_consistency": 0.30,
    "min_agreement": 0.60,
    "min_patch_coverage": 0.25,
    # Share of versions that must sit on the minority side of the threshold
    # before the split counts as real disagreement. Values above 1.0 disable
    # the rule, which is how a baseline "never abstain" run is expressed.
    "boundary_crossing_fraction": 0.2,
    "abstain_on_evidence_errors": False,
    "rules": list(ALL_RULES),
}


@dataclass(frozen=True)
class AbstentionRule:
    """One rule and whether the evidence tripped it."""

    name: str
    triggered: bool
    detail: str
    observed: Optional[float] = None
    limit: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.name,
            "triggered": self.triggered,
            "detail": self.detail,
            "observed": None if self.observed is None else round(float(self.observed), 6),
            "limit": None if self.limit is None else round(float(self.limit), 6),
        }


@dataclass
class AbstentionDecision:
    """The verdict after the abstention rules have been applied."""

    abstain: bool
    label: str
    original_label: str
    enabled: bool
    rules: List[AbstentionRule] = field(default_factory=list)

    @property
    def triggered_rules(self) -> List[str]:
        return [rule.name for rule in self.rules if rule.triggered]

    @property
    def statement(self) -> str:
        if not self.enabled:
            return "Evidence-driven abstention is disabled; the label comes from the probability bands alone."
        if not self.abstain:
            return "No abstention rule fired: the evidence was stable enough to report a verdict."
        reasons = "; ".join(rule.detail for rule in self.rules if rule.triggered)
        return f"Reported as Uncertain because {reasons}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "abstained": self.abstain,
            "label": self.label,
            "label_before_abstention": self.original_label,
            "triggered_rules": self.triggered_rules,
            "statement": self.statement,
            "rules": [rule.as_dict() for rule in self.rules],
        }


def abstention_settings(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolve and validate the abstention configuration."""

    section = (config or {}).get("abstention", {}) or {}
    settings = dict(DEFAULTS)
    settings.update({key: value for key, value in section.items() if key in DEFAULTS})

    for key in (
        "borderline_margin",
        "max_transformed_drift",
        "min_consistency",
        "min_agreement",
        "min_patch_coverage",
    ):
        value = float(settings[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"abstention.{key} must be within [0, 1], got {value}")
        settings[key] = value

    crossing = float(settings["boundary_crossing_fraction"])
    if crossing < 0.0:
        raise ValueError(
            f"abstention.boundary_crossing_fraction must be non-negative, got {crossing}"
        )
    settings["boundary_crossing_fraction"] = crossing

    # An explicitly empty list means "no rules", which is how a baseline run
    # turns abstention off. Only a missing key falls back to every rule.
    rules = settings.get("rules")
    if rules is None:
        rules = list(ALL_RULES)
    unknown = [name for name in rules if name not in ALL_RULES]
    if unknown:
        raise ValueError(
            f"Unknown abstention rule(s): {', '.join(unknown)}. "
            f"Valid rules: {', '.join(ALL_RULES)}."
        )
    settings["rules"] = list(rules)
    settings["enabled"] = bool(settings["enabled"])
    settings["abstain_on_evidence_errors"] = bool(settings["abstain_on_evidence_errors"])
    return settings


def evaluate_abstention(
    label: str,
    final_probability: float,
    threshold: float,
    clean_probability: Optional[float] = None,
    transformed_probabilities: Optional[Sequence[float]] = None,
    consistency_score: Optional[float] = None,
    agreement: Optional[float] = None,
    patch_coverage: Optional[float] = None,
    patch_available: bool = False,
    errors: Optional[Sequence[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> AbstentionDecision:
    """Decide whether the evidence supports the label, or only Uncertain.

    Every argument that can be missing is optional, and a rule whose evidence is
    absent simply does not fire. Missing evidence must never manufacture an
    abstention on its own -- that is what ``abstain_on_evidence_errors`` is for,
    and it is off by default.
    """

    settings = abstention_settings(config)
    active = set(settings["rules"])
    rules: List[AbstentionRule] = []

    def consider(
        name: str, triggered: bool, detail: str, observed=None, limit=None
    ) -> None:
        if name not in active:
            return
        rules.append(
            AbstentionRule(
                name=name, triggered=bool(triggered), detail=detail, observed=observed, limit=limit
            )
        )

    probability = float(final_probability)
    threshold = float(threshold)

    # 1. Too close to the fence to call.
    margin = settings["borderline_margin"]
    distance = abs(probability - threshold)
    consider(
        RULE_BORDERLINE,
        distance < margin,
        f"the score {probability:.3f} sits {distance:.3f} from the {threshold:.2f} "
        f"threshold, {'inside' if distance < margin else 'outside'} the {margin:.2f} "
        f"borderline margin",
        observed=distance,
        limit=margin,
    )

    # 2. The measured failure mode: AI-like when clean, collapsing once
    #    transformed. Degradation destroying evidence is not evidence of
    #    authenticity, so this withdraws the claim rather than accepting it.
    transformed = [float(v) for v in (transformed_probabilities or [])]
    if clean_probability is not None and transformed:
        clean = float(clean_probability)
        drop = clean - float(np.mean(transformed))
        looked_ai = clean >= threshold
        consider(
            RULE_TRANSFORMED_DRIFT,
            looked_ai and drop > settings["max_transformed_drift"],
            f"the clean score {clean:.3f} was AI-like but transformed versions "
            f"averaged {float(np.mean(transformed)):.3f}, a drop of {drop:.3f}",
            observed=drop,
            limit=settings["max_transformed_drift"],
        )

    # 3. Unstable under transformation.
    if consistency_score is not None:
        consider(
            RULE_LOW_CONSISTENCY,
            float(consistency_score) < settings["min_consistency"],
            f"transformation consistency {float(consistency_score):.3f} is "
            f"{'below' if float(consistency_score) < settings['min_consistency'] else 'at or above'} "
            f"the {settings['min_consistency']:.2f} minimum",
            observed=float(consistency_score),
            limit=settings["min_consistency"],
        )

    # 4. Versions genuinely disagree about which side of the fence they are on.
    if transformed and clean_probability is not None:
        every = [float(clean_probability), *transformed]
        above = sum(1 for value in every if value >= threshold)
        crosses = 0 < above < len(every)
        minority = min(above, len(every) - above) / len(every)
        limit = settings["boundary_crossing_fraction"]
        consider(
            RULE_BOUNDARY_CROSSING,
            crosses and minority >= limit,
            f"{above} of {len(every)} versions scored AI-generated and "
            f"{len(every) - above} did not"
            + (", straddling the decision threshold" if crosses and minority >= limit
               else "; not a material split"),
            observed=minority,
            limit=limit,
        )

    # 5. Low agreement with the original.
    if agreement is not None:
        consider(
            RULE_LOW_AGREEMENT,
            float(agreement) < settings["min_agreement"],
            f"{float(agreement):.2f} of versions agreed with the original, "
            f"{'below' if float(agreement) < settings['min_agreement'] else 'at or above'} "
            f"the {settings['min_agreement']:.2f} minimum",
            observed=float(agreement),
            limit=settings["min_agreement"],
        )

    # 6. Not enough of the image was actually measured to support a claim that
    #    depends on patches. Only applies when patch analysis ran at all.
    if patch_available and patch_coverage is not None:
        consider(
            RULE_LOW_PATCH_COVERAGE,
            float(patch_coverage) < settings["min_patch_coverage"],
            f"patches covered {float(patch_coverage):.2f} of the image, "
            f"{'below' if float(patch_coverage) < settings['min_patch_coverage'] else 'at or above'} "
            f"the {settings['min_patch_coverage']:.2f} minimum",
            observed=float(patch_coverage),
            limit=settings["min_patch_coverage"],
        )

    # 7. Something the analysis needed failed outright.
    failures = [item for item in (errors or []) if item.get("stage") != "patches"]
    consider(
        RULE_EVIDENCE_UNAVAILABLE,
        settings["abstain_on_evidence_errors"] and bool(failures),
        f"{len(failures)} analysis stage(s) reported an error",
        observed=float(len(failures)),
        limit=0.0,
    )

    abstain = settings["enabled"] and any(rule.triggered for rule in rules)
    return AbstentionDecision(
        abstain=abstain,
        label=LABEL_UNCERTAIN if abstain else label,
        original_label=label,
        enabled=settings["enabled"],
        rules=rules,
    )


def would_report_confident_authentic_after_degradation(
    clean_probability: float,
    transformed_probabilities: Sequence[float],
    threshold: float,
    max_drift: float = DEFAULTS["max_transformed_drift"],
) -> bool:
    """True when degradation alone moved an AI-like image to the authentic side.

    Exposed separately so the evaluation code can count how often the pathology
    occurs without having to run the whole rule set.
    """

    transformed = [float(v) for v in transformed_probabilities]
    if not transformed:
        return False
    clean = float(clean_probability)
    mean_transformed = float(np.mean(transformed))
    return (
        clean >= float(threshold)
        and mean_transformed < float(threshold)
        and (clean - mean_transformed) > float(max_drift)
    )


def label_after_abstention(decision: AbstentionDecision) -> str:
    """The label to report, which is ``Uncertain`` whenever a rule fired."""

    return LABEL_UNCERTAIN if decision.abstain else decision.label


__all__ = [
    "ALL_RULES",
    "DEFAULTS",
    "LABEL_AI",
    "AbstentionDecision",
    "AbstentionRule",
    "abstention_settings",
    "evaluate_abstention",
    "label_after_abstention",
    "would_report_confident_authentic_after_degradation",
]
