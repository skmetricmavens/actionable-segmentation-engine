"""
Module: trait_explainer

Purpose: Generate human-readable explanations for trait value scores.

This module provides explanations for WHY a trait is valuable for segmentation,
retention, or personalization - not just THAT it is valuable.

Key components:
- TraitExplanation: Dataclass holding per-dimension explanations
- TraitExplainer: Generates explanations from TraitValueScore data
"""

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.analysis.trait_discovery import TraitValueScore


# =============================================================================
# ACTIONABILITY PATTERNS
# =============================================================================

# Traits matching these patterns are likely NOT actionable
# (they are results of behavior, not drivers of it)
LIKELY_NON_ACTIONABLE_PATTERNS = [
    r"payment",  # Payment method is chosen after purchase decision
    r"shipping",  # Shipping is a result of checkout
    r"order_id",  # Order attributes are outcomes
    r"purchase_id",  # Purchase attributes are outcomes
    r"transaction",  # Transaction details are outcomes
    r"checkout",  # Checkout details are outcomes
    r"invoice",  # Invoice is a result
    r"receipt",  # Receipt is a result
    r"confirmation",  # Confirmation is a result
]

# Traits matching these patterns are likely actionable
# (they represent preferences or behaviors that can be targeted)
LIKELY_ACTIONABLE_PATTERNS = [
    r"category",  # Product category preferences
    r"brand",  # Brand preferences
    r"color",  # Style preferences
    r"size",  # Size preferences
    r"device",  # Device/channel preferences
    r"channel",  # Channel preferences
    r"location",  # Geographic targeting
    r"language",  # Language preferences
    r"country",  # Geographic targeting
    r"state",  # Geographic targeting
    r"region",  # Geographic targeting
    r"style",  # Style preferences
    r"material",  # Product preferences
    r"price_range",  # Price sensitivity
    r"margin",  # Margin-based targeting
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class TraitExplanation:
    """Human-readable explanations for a trait's value."""

    trait_name: str

    # Per-dimension explanations
    revenue_explanation: str | None = None
    retention_explanation: str | None = None
    personalization_explanation: str | None = None

    # Overall assessment
    overall_summary: str = ""
    actionability_assessment: str = ""
    caveats: list[str] = field(default_factory=list)

    # Suggested actions
    suggested_actions: list[str] = field(default_factory=list)

    # Confidence assessment
    confidence_level: str = "medium"  # low, medium, high
    confidence_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "trait_name": self.trait_name,
            "revenue_explanation": self.revenue_explanation,
            "retention_explanation": self.retention_explanation,
            "personalization_explanation": self.personalization_explanation,
            "overall_summary": self.overall_summary,
            "actionability_assessment": self.actionability_assessment,
            "caveats": self.caveats,
            "suggested_actions": self.suggested_actions,
            "confidence_level": self.confidence_level,
            "confidence_reasons": self.confidence_reasons,
        }


# =============================================================================
# TRAIT EXPLAINER
# =============================================================================


class TraitExplainer:
    """
    Generates human-readable explanations for trait scores.

    Example:
        >>> explainer = TraitExplainer()
        >>> explanation = explainer.explain(trait_score)
        >>> print(explanation.overall_summary)
        "Strong segmentation candidate: 3.8x revenue difference between values"
    """

    # Compiled patterns for performance
    _non_actionable_patterns: list[re.Pattern] = [
        re.compile(p, re.IGNORECASE) for p in LIKELY_NON_ACTIONABLE_PATTERNS
    ]
    _actionable_patterns: list[re.Pattern] = [
        re.compile(p, re.IGNORECASE) for p in LIKELY_ACTIONABLE_PATTERNS
    ]

    def __init__(
        self,
        *,
        high_revenue_threshold: float = 0.5,
        high_retention_threshold: float = 0.5,
        high_personalization_threshold: float = 0.6,
        significance_threshold: float = 0.05,
        high_coverage_threshold: float = 0.7,
        low_coverage_threshold: float = 0.3,
    ):
        """
        Initialize TraitExplainer.

        Args:
            high_revenue_threshold: Revenue impact score considered "high"
            high_retention_threshold: Retention impact score considered "high"
            high_personalization_threshold: Personalization value considered "high"
            significance_threshold: p-value threshold for statistical significance
            high_coverage_threshold: Customer coverage considered "high"
            low_coverage_threshold: Customer coverage considered "low"
        """
        self.high_revenue_threshold = high_revenue_threshold
        self.high_retention_threshold = high_retention_threshold
        self.high_personalization_threshold = high_personalization_threshold
        self.significance_threshold = significance_threshold
        self.high_coverage_threshold = high_coverage_threshold
        self.low_coverage_threshold = low_coverage_threshold

    def explain(self, trait: "TraitValueScore") -> TraitExplanation:
        """
        Generate complete explanation for a trait.

        Args:
            trait: TraitValueScore to explain

        Returns:
            TraitExplanation with all dimensions explained
        """
        explanation = TraitExplanation(trait_name=trait.trait_name)

        # Generate per-dimension explanations
        explanation.revenue_explanation = self._explain_revenue(trait)
        explanation.retention_explanation = self._explain_retention(trait)
        explanation.personalization_explanation = self._explain_personalization(trait)

        # Assess actionability
        explanation.actionability_assessment = self._assess_actionability(trait)

        # Identify caveats
        explanation.caveats = self._identify_caveats(trait)

        # Generate overall summary
        explanation.overall_summary = self._generate_summary(trait, explanation)

        # Suggest actions
        explanation.suggested_actions = self._suggest_actions(trait, explanation)

        # Assess confidence
        explanation.confidence_level, explanation.confidence_reasons = (
            self._assess_confidence(trait)
        )

        return explanation

    def _explain_revenue(self, trait: "TraitValueScore") -> str | None:
        """Generate revenue impact explanation."""
        if not trait.top_revenue_values or len(trait.top_revenue_values) < 2:
            return None

        best_value, best_revenue = trait.top_revenue_values[0]
        worst_value, worst_revenue = trait.top_revenue_values[-1]

        # Calculate multiplier
        if worst_revenue > 0:
            multiplier = best_revenue / worst_revenue
        else:
            multiplier = float("inf")

        # High impact with significance
        if trait.revenue_impact >= self.high_revenue_threshold:
            if trait.revenue_p_value < self.significance_threshold:
                if multiplier < float("inf"):
                    return (
                        f"Customers with '{best_value}' generate ${best_revenue:,.0f} avg revenue "
                        f"vs ${worst_revenue:,.0f} for '{worst_value}' - "
                        f"a {multiplier:.1f}x difference (p<{trait.revenue_p_value:.3f})"
                    )
                else:
                    return (
                        f"Customers with '{best_value}' generate ${best_revenue:,.0f} avg revenue "
                        f"vs ${worst_revenue:,.0f} for '{worst_value}' (p<{trait.revenue_p_value:.3f})"
                    )
            else:
                return (
                    f"Revenue varies by {trait.trait_name} (${best_revenue:,.0f} vs ${worst_revenue:,.0f}) "
                    f"but not statistically significant (p={trait.revenue_p_value:.2f})"
                )

        # Low impact
        return (
            f"No significant revenue difference across {trait.trait_name} values "
            f"(F={trait.revenue_f_statistic:.1f}, p={trait.revenue_p_value:.2f})"
        )

    def _explain_retention(self, trait: "TraitValueScore") -> str | None:
        """Generate retention impact explanation."""
        if not trait.top_retention_values or len(trait.top_retention_values) < 2:
            return None

        best_value, best_retention = trait.top_retention_values[0]
        worst_value, worst_retention = trait.top_retention_values[-1]

        # High impact with significance
        if trait.retention_impact >= self.high_retention_threshold:
            if trait.retention_p_value < self.significance_threshold:
                return (
                    f"'{best_value}' customers show {best_retention:.0%} retention "
                    f"vs {worst_retention:.0%} for '{worst_value}' (p<{trait.retention_p_value:.3f})"
                )
            else:
                return (
                    f"Retention varies by {trait.trait_name} ({best_retention:.0%} vs {worst_retention:.0%}) "
                    f"but not statistically significant (p={trait.retention_p_value:.2f})"
                )

        # Low impact
        if trait.retention_chi2_statistic > 0:
            return (
                f"No significant retention difference across {trait.trait_name} values "
                f"(chi2={trait.retention_chi2_statistic:.1f}, p={trait.retention_p_value:.2f})"
            )
        return None

    def _explain_personalization(self, trait: "TraitValueScore") -> str | None:
        """Generate personalization value explanation."""
        # High personalization value
        if trait.personalization_value >= self.high_personalization_threshold:
            if trait.concentration < 0.5:
                return (
                    f"{trait.distinct_values} distinct values with {trait.entropy:.1f} bits of entropy - "
                    f"enough variety for meaningful personalization without overwhelming complexity"
                )
            else:
                return (
                    f"{trait.distinct_values} distinct values with {trait.entropy:.1f} bits of entropy, "
                    f"though one value dominates ({trait.concentration:.0%} concentration)"
                )

        # Medium personalization
        if trait.personalization_value >= 0.3:
            if trait.distinct_values < 3:
                return (
                    f"Only {trait.distinct_values} values - limited personalization options"
                )
            if trait.concentration > 0.7:
                return (
                    f"{trait.distinct_values} values but {trait.concentration:.0%} concentrated "
                    f"in one value - limited personalization benefit"
                )
            return (
                f"Moderate personalization potential: {trait.distinct_values} values "
                f"with {trait.entropy:.1f} bits of entropy"
            )

        # Low personalization
        return (
            f"Low personalization value: "
            f"{trait.distinct_values} values with {trait.concentration:.0%} concentration"
        )

    def _assess_actionability(self, trait: "TraitValueScore") -> str:
        """Assess whether the trait is actionable."""
        trait_lower = trait.trait_name.lower()
        path_lower = trait.trait_path.lower()

        # Check for non-actionable patterns
        for pattern in self._non_actionable_patterns:
            if pattern.search(trait_lower) or pattern.search(path_lower):
                return (
                    f"LIKELY NOT ACTIONABLE: '{trait.trait_name}' appears to be a result of "
                    f"purchase behavior, not a driver. Cannot target customers based on this "
                    f"before they make a purchase."
                )

        # Check for actionable patterns
        for pattern in self._actionable_patterns:
            if pattern.search(trait_lower) or pattern.search(path_lower):
                return (
                    f"ACTIONABLE: '{trait.trait_name}' represents customer preferences "
                    f"that can be used for targeting and personalization."
                )

        # Unknown - default to cautiously actionable
        return (
            f"POTENTIALLY ACTIONABLE: '{trait.trait_name}' may be usable for targeting, "
            f"but verify it represents a customer preference rather than a purchase outcome."
        )

    def _identify_caveats(self, trait: "TraitValueScore") -> list[str]:
        """Identify caveats and warnings for the trait."""
        caveats = []

        # Statistical significance caveats
        if trait.revenue_p_value >= self.significance_threshold:
            if trait.revenue_impact > 0.3:
                caveats.append(
                    f"Revenue correlation is not statistically significant (p={trait.revenue_p_value:.2f})"
                )

        if trait.retention_p_value >= self.significance_threshold:
            if trait.retention_impact > 0.3:
                caveats.append(
                    f"Retention correlation is not statistically significant (p={trait.retention_p_value:.2f})"
                )

        # Coverage caveats
        if trait.customer_coverage < self.low_coverage_threshold:
            caveats.append(
                f"Low coverage: only {trait.customer_coverage:.0%} of customers have this trait"
            )

        # Concentration caveats
        if trait.concentration > 0.9:
            caveats.append(
                f"High concentration: {trait.concentration:.0%} of customers have the same value"
            )

        # Actionability caveat
        trait_lower = trait.trait_name.lower()
        for pattern in self._non_actionable_patterns:
            if pattern.search(trait_lower):
                caveats.append(
                    "This trait may be a result of behavior, not a driver - correlation may not be actionable"
                )
                break

        return caveats

    def _generate_summary(
        self, trait: "TraitValueScore", explanation: TraitExplanation
    ) -> str:
        """Generate overall summary."""
        strengths = []

        # Check each dimension
        if (
            trait.revenue_impact >= self.high_revenue_threshold
            and trait.revenue_p_value < self.significance_threshold
        ):
            if trait.top_revenue_values:
                best_rev = trait.top_revenue_values[0][1]
                worst_rev = trait.top_revenue_values[-1][1]
                if worst_rev > 0:
                    multiplier = best_rev / worst_rev
                    strengths.append(f"{multiplier:.1f}x revenue difference between values")

        if (
            trait.retention_impact >= self.high_retention_threshold
            and trait.retention_p_value < self.significance_threshold
        ):
            strengths.append("significant retention impact")

        if trait.personalization_value >= self.high_personalization_threshold:
            strengths.append(f"{trait.distinct_values} values for personalization")

        # Build summary
        if strengths:
            if "NOT ACTIONABLE" in explanation.actionability_assessment:
                return (
                    f"Strong statistical signal ({', '.join(strengths)}) but likely not actionable - "
                    f"this trait appears to be a result of behavior, not a driver"
                )
            return f"Strong candidate for {'segmentation' if len(strengths) > 1 else strengths[0]}: {', '.join(strengths)}"

        # No strong dimensions
        return f"Limited value: no strong impact on revenue, retention, or personalization"

    def _suggest_actions(
        self, trait: "TraitValueScore", explanation: TraitExplanation
    ) -> list[str]:
        """Suggest concrete actions based on trait analysis."""
        actions = []

        # Skip if not actionable
        if "NOT ACTIONABLE" in explanation.actionability_assessment:
            actions.append("Use for descriptive analytics only, not for targeting")
            return actions

        # Revenue-based actions
        if (
            trait.revenue_impact >= self.high_revenue_threshold
            and trait.revenue_p_value < self.significance_threshold
            and trait.top_revenue_values
        ):
            best_value = trait.top_revenue_values[0][0]
            actions.append(
                f"Create high-value segment for customers preferring '{best_value}'"
            )
            if len(trait.top_revenue_values) > 1:
                worst_value = trait.top_revenue_values[-1][0]
                actions.append(
                    f"Investigate why '{worst_value}' customers have lower value - potential improvement opportunity"
                )

        # Retention-based actions
        if (
            trait.retention_impact >= self.high_retention_threshold
            and trait.retention_p_value < self.significance_threshold
            and trait.top_retention_values
        ):
            worst_value, worst_retention = trait.top_retention_values[-1]
            actions.append(
                f"Target '{worst_value}' customers with retention campaigns ({worst_retention:.0%} at risk)"
            )

        # Personalization actions
        if trait.personalization_value >= self.high_personalization_threshold:
            actions.append(
                f"Use {trait.trait_name} for content personalization ({trait.distinct_values} variants)"
            )

        if not actions:
            actions.append("Monitor this trait for future opportunities")

        return actions

    def _assess_confidence(self, trait: "TraitValueScore") -> tuple[str, list[str]]:
        """Assess confidence level in the analysis."""
        reasons = []
        score = 0

        # Coverage check
        if trait.customer_coverage >= self.high_coverage_threshold:
            score += 2
            reasons.append(f"High coverage ({trait.customer_coverage:.0%} of customers)")
        elif trait.customer_coverage >= self.low_coverage_threshold:
            score += 1
            reasons.append(f"Moderate coverage ({trait.customer_coverage:.0%} of customers)")
        else:
            reasons.append(f"Low coverage ({trait.customer_coverage:.0%} of customers)")

        # Statistical significance check
        if trait.revenue_p_value < 0.01 or trait.retention_p_value < 0.01:
            score += 2
            reasons.append("Highly significant (p<0.01)")
        elif trait.revenue_p_value < 0.05 or trait.retention_p_value < 0.05:
            score += 1
            reasons.append("Significant (p<0.05)")
        else:
            reasons.append("Not statistically significant")

        # Determine confidence level
        if score >= 3:
            return "high", reasons
        elif score >= 1:
            return "medium", reasons
        else:
            return "low", reasons


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def explain_trait(trait: "TraitValueScore") -> TraitExplanation:
    """
    Convenience function to generate explanation for a trait.

    Args:
        trait: TraitValueScore to explain

    Returns:
        TraitExplanation
    """
    explainer = TraitExplainer()
    return explainer.explain(trait)


def format_explanation(explanation: TraitExplanation) -> str:
    """
    Format explanation as human-readable text.

    Args:
        explanation: TraitExplanation to format

    Returns:
        Formatted string
    """
    lines = []

    lines.append(f"TRAIT: {explanation.trait_name}")
    lines.append("-" * 50)

    if explanation.overall_summary:
        lines.append(f"SUMMARY: {explanation.overall_summary}")
        lines.append("")

    if explanation.revenue_explanation:
        lines.append(f"REVENUE: {explanation.revenue_explanation}")

    if explanation.retention_explanation:
        lines.append(f"RETENTION: {explanation.retention_explanation}")

    if explanation.personalization_explanation:
        lines.append(f"PERSONALIZATION: {explanation.personalization_explanation}")

    lines.append("")
    lines.append(f"ACTIONABILITY: {explanation.actionability_assessment}")

    if explanation.caveats:
        lines.append("")
        lines.append("CAVEATS:")
        for caveat in explanation.caveats:
            lines.append(f"  - {caveat}")

    if explanation.suggested_actions:
        lines.append("")
        lines.append("SUGGESTED ACTIONS:")
        for action in explanation.suggested_actions:
            lines.append(f"  - {action}")

    lines.append("")
    lines.append(
        f"CONFIDENCE: {explanation.confidence_level.upper()} ({', '.join(explanation.confidence_reasons)})"
    )

    return "\n".join(lines)
