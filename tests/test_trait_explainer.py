"""Tests for trait explainer module."""

import pytest

from src.analysis.trait_discovery import TraitValueScore
from src.analysis.trait_explainer import (
    TraitExplainer,
    TraitExplanation,
    explain_trait,
    format_explanation,
    LIKELY_NON_ACTIONABLE_PATTERNS,
    LIKELY_ACTIONABLE_PATTERNS,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def high_revenue_trait() -> TraitValueScore:
    """Create a trait with high revenue impact."""
    return TraitValueScore(
        trait_name="category_level_1",
        trait_path="extra.category_level_1",
        trait_type="categorical",
        revenue_impact=0.8,
        retention_impact=0.2,
        personalization_value=0.5,
        revenue_f_statistic=25.0,
        revenue_p_value=0.001,
        retention_chi2_statistic=5.0,
        retention_p_value=0.3,
        entropy=2.5,
        distinct_values=8,
        customer_coverage=0.85,
        concentration=0.3,
        top_revenue_values=[
            ("Electronics", 450.0),
            ("Home", 320.0),
            ("Clothing", 280.0),
            ("Accessories", 120.0),
        ],
        top_retention_values=[
            ("Electronics", 0.85),
            ("Home", 0.78),
            ("Clothing", 0.72),
            ("Accessories", 0.65),
        ],
    )


@pytest.fixture
def payment_method_trait() -> TraitValueScore:
    """Create a payment method trait (likely not actionable)."""
    return TraitValueScore(
        trait_name="payment_method",
        trait_path="custom.payment_method",
        trait_type="categorical",
        revenue_impact=1.0,
        retention_impact=0.1,
        personalization_value=0.3,
        revenue_f_statistic=116.0,
        revenue_p_value=0.0001,
        retention_chi2_statistic=2.0,
        retention_p_value=0.5,
        entropy=1.9,
        distinct_values=12,
        customer_coverage=1.0,
        concentration=0.54,
        top_revenue_values=[
            ("buckaroo_ideal", 3872.0),
            ("creditcard", 248.0),
            ("giftcard", 198.0),
        ],
        top_retention_values=[
            ("buckaroo_ideal", 0.82),
            ("creditcard", 0.75),
            ("giftcard", 0.70),
        ],
    )


@pytest.fixture
def low_impact_trait() -> TraitValueScore:
    """Create a trait with low impact scores."""
    return TraitValueScore(
        trait_name="some_field",
        trait_path="extra.some_field",
        trait_type="categorical",
        revenue_impact=0.1,
        retention_impact=0.05,
        personalization_value=0.2,
        revenue_f_statistic=1.5,
        revenue_p_value=0.25,
        retention_chi2_statistic=1.0,
        retention_p_value=0.6,
        entropy=0.8,
        distinct_values=3,
        customer_coverage=0.4,
        concentration=0.6,
        top_revenue_values=[
            ("A", 200.0),
            ("B", 195.0),
            ("C", 190.0),
        ],
        top_retention_values=[],
    )


@pytest.fixture
def high_personalization_trait() -> TraitValueScore:
    """Create a trait good for personalization."""
    return TraitValueScore(
        trait_name="device_type",
        trait_path="device_type",
        trait_type="categorical",
        revenue_impact=0.2,
        retention_impact=0.15,
        personalization_value=0.75,
        revenue_f_statistic=4.0,
        revenue_p_value=0.02,
        retention_chi2_statistic=3.0,
        retention_p_value=0.2,
        entropy=1.8,
        distinct_values=4,
        customer_coverage=1.0,
        concentration=0.45,
        top_revenue_values=[
            ("Web", 327.0),
            ("iOS", 245.0),
            ("Android", 176.0),
            ("Other", 150.0),
        ],
        top_retention_values=[
            ("Web", 0.80),
            ("iOS", 0.75),
            ("Android", 0.70),
            ("Other", 0.65),
        ],
    )


@pytest.fixture
def explainer() -> TraitExplainer:
    """Create a TraitExplainer instance."""
    return TraitExplainer()


# =============================================================================
# TESTS: TraitExplanation
# =============================================================================


class TestTraitExplanation:
    """Tests for TraitExplanation dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        explanation = TraitExplanation(
            trait_name="test_trait",
            revenue_explanation="Revenue goes up",
            overall_summary="Good trait",
            actionability_assessment="Actionable",
            caveats=["Caveat 1"],
            suggested_actions=["Action 1"],
            confidence_level="high",
            confidence_reasons=["High coverage"],
        )

        result = explanation.to_dict()

        assert result["trait_name"] == "test_trait"
        assert result["revenue_explanation"] == "Revenue goes up"
        assert result["overall_summary"] == "Good trait"
        assert result["caveats"] == ["Caveat 1"]
        assert result["confidence_level"] == "high"

    def test_default_values(self):
        """Test default values are set correctly."""
        explanation = TraitExplanation(trait_name="test")

        assert explanation.revenue_explanation is None
        assert explanation.retention_explanation is None
        assert explanation.personalization_explanation is None
        assert explanation.overall_summary == ""
        assert explanation.caveats == []
        assert explanation.suggested_actions == []
        assert explanation.confidence_level == "medium"


# =============================================================================
# TESTS: TraitExplainer
# =============================================================================


class TestTraitExplainer:
    """Tests for TraitExplainer class."""

    def test_explain_high_revenue_trait(self, explainer, high_revenue_trait):
        """Test explanation generation for high revenue impact trait."""
        explanation = explainer.explain(high_revenue_trait)

        assert explanation.trait_name == "category_level_1"
        assert explanation.revenue_explanation is not None
        assert "450" in explanation.revenue_explanation  # Best revenue
        assert "120" in explanation.revenue_explanation  # Worst revenue
        assert "Electronics" in explanation.revenue_explanation
        assert explanation.overall_summary != ""
        assert "ACTIONABLE" in explanation.actionability_assessment

    def test_explain_payment_method_trait(self, explainer, payment_method_trait):
        """Test explanation identifies payment_method as not actionable."""
        explanation = explainer.explain(payment_method_trait)

        assert explanation.trait_name == "payment_method"
        assert "NOT ACTIONABLE" in explanation.actionability_assessment
        assert len(explanation.caveats) > 0
        # Should have caveat about being a result not driver
        assert any("result" in c.lower() or "driver" in c.lower() for c in explanation.caveats)

    def test_explain_low_impact_trait(self, explainer, low_impact_trait):
        """Test explanation for low impact trait."""
        explanation = explainer.explain(low_impact_trait)

        assert explanation.trait_name == "some_field"
        # Should note no significant difference
        assert "No significant" in explanation.revenue_explanation or "not statistically" in explanation.revenue_explanation.lower()
        # Limited value summary
        assert "Limited" in explanation.overall_summary or "no strong" in explanation.overall_summary.lower()

    def test_explain_high_personalization_trait(self, explainer, high_personalization_trait):
        """Test explanation for personalization-focused trait."""
        explanation = explainer.explain(high_personalization_trait)

        assert explanation.trait_name == "device_type"
        assert explanation.personalization_explanation is not None
        assert "4" in explanation.personalization_explanation  # 4 distinct values
        assert "entropy" in explanation.personalization_explanation.lower() or "variety" in explanation.personalization_explanation.lower()
        # Should suggest personalization use
        assert any("personalization" in action.lower() for action in explanation.suggested_actions)

    def test_confidence_assessment_high_coverage(self, explainer, high_revenue_trait):
        """Test confidence is high for high coverage traits."""
        # High coverage + significant p-value should give high confidence
        explanation = explainer.explain(high_revenue_trait)

        assert explanation.confidence_level in ("high", "medium")
        assert any("coverage" in r.lower() for r in explanation.confidence_reasons)

    def test_confidence_assessment_low_coverage(self, explainer, low_impact_trait):
        """Test confidence reflects low coverage."""
        explanation = explainer.explain(low_impact_trait)

        # Low coverage should be noted
        assert any("coverage" in r.lower() for r in explanation.confidence_reasons)

    def test_suggested_actions_for_high_revenue(self, explainer, high_revenue_trait):
        """Test suggested actions for high revenue trait."""
        explanation = explainer.explain(high_revenue_trait)

        assert len(explanation.suggested_actions) > 0
        # Should suggest segmentation
        assert any("segment" in action.lower() for action in explanation.suggested_actions)

    def test_caveats_for_low_coverage(self, explainer):
        """Test caveats are added for low coverage traits."""
        # Create trait with very low coverage (below 0.3 threshold)
        low_coverage_trait = TraitValueScore(
            trait_name="rare_trait",
            trait_path="extra.rare_trait",
            trait_type="categorical",
            revenue_impact=0.5,
            revenue_p_value=0.01,
            customer_coverage=0.15,  # Below 30% threshold
            distinct_values=5,
            top_revenue_values=[("A", 300.0), ("B", 200.0)],
        )

        explanation = explainer.explain(low_coverage_trait)

        # Should have low coverage caveat
        assert any("coverage" in c.lower() for c in explanation.caveats)


class TestActionabilityPatterns:
    """Tests for actionability pattern matching."""

    def test_non_actionable_patterns_exist(self):
        """Test that non-actionable patterns are defined."""
        assert len(LIKELY_NON_ACTIONABLE_PATTERNS) > 0
        assert any("payment" in p for p in LIKELY_NON_ACTIONABLE_PATTERNS)
        assert any("shipping" in p for p in LIKELY_NON_ACTIONABLE_PATTERNS)

    def test_actionable_patterns_exist(self):
        """Test that actionable patterns are defined."""
        assert len(LIKELY_ACTIONABLE_PATTERNS) > 0
        assert any("category" in p for p in LIKELY_ACTIONABLE_PATTERNS)
        assert any("brand" in p for p in LIKELY_ACTIONABLE_PATTERNS)
        assert any("device" in p for p in LIKELY_ACTIONABLE_PATTERNS)

    def test_shipping_trait_not_actionable(self, explainer):
        """Test shipping-related traits are marked not actionable."""
        trait = TraitValueScore(
            trait_name="shipping_method",
            trait_path="custom.shipping_method",
            trait_type="categorical",
            revenue_impact=0.5,
            customer_coverage=1.0,
            distinct_values=5,
            top_revenue_values=[("express", 500.0), ("standard", 200.0)],
        )

        explanation = explainer.explain(trait)
        assert "NOT ACTIONABLE" in explanation.actionability_assessment

    def test_category_trait_actionable(self, explainer):
        """Test category traits are marked actionable."""
        trait = TraitValueScore(
            trait_name="product_category",
            trait_path="extra.product_category",
            trait_type="categorical",
            revenue_impact=0.5,
            customer_coverage=0.9,
            distinct_values=10,
            top_revenue_values=[("Electronics", 400.0), ("Clothing", 200.0)],
        )

        explanation = explainer.explain(trait)
        assert "ACTIONABLE" in explanation.actionability_assessment
        assert "NOT ACTIONABLE" not in explanation.actionability_assessment


# =============================================================================
# TESTS: Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_explain_trait_function(self, high_revenue_trait):
        """Test explain_trait convenience function."""
        explanation = explain_trait(high_revenue_trait)

        assert isinstance(explanation, TraitExplanation)
        assert explanation.trait_name == high_revenue_trait.trait_name

    def test_format_explanation_function(self, high_revenue_trait):
        """Test format_explanation function."""
        explanation = explain_trait(high_revenue_trait)
        formatted = format_explanation(explanation)

        assert isinstance(formatted, str)
        assert "category_level_1" in formatted
        assert "SUMMARY" in formatted or "Summary" in formatted.lower()
        assert "ACTIONABILITY" in formatted
        assert "CONFIDENCE" in formatted


# =============================================================================
# TESTS: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_top_values(self, explainer):
        """Test handling of empty top_revenue_values."""
        trait = TraitValueScore(
            trait_name="empty_trait",
            trait_path="extra.empty_trait",
            trait_type="categorical",
            revenue_impact=0.3,
            customer_coverage=0.5,
            distinct_values=5,
            top_revenue_values=[],
            top_retention_values=[],
        )

        explanation = explainer.explain(trait)
        # Should not crash, should handle gracefully
        assert explanation is not None
        assert explanation.revenue_explanation is None

    def test_single_value_trait(self, explainer):
        """Test handling of trait with single top value."""
        trait = TraitValueScore(
            trait_name="single_value",
            trait_path="extra.single_value",
            trait_type="categorical",
            revenue_impact=0.3,
            customer_coverage=0.5,
            distinct_values=2,
            top_revenue_values=[("only_value", 300.0)],
        )

        explanation = explainer.explain(trait)
        # Should handle gracefully
        assert explanation is not None

    def test_zero_revenue_values(self, explainer):
        """Test handling of zero revenue values (division by zero)."""
        trait = TraitValueScore(
            trait_name="zero_revenue",
            trait_path="extra.zero_revenue",
            trait_type="categorical",
            revenue_impact=0.5,
            revenue_p_value=0.01,
            customer_coverage=0.8,
            distinct_values=3,
            top_revenue_values=[
                ("high", 500.0),
                ("zero", 0.0),
            ],
        )

        explanation = explainer.explain(trait)
        # Should not crash on division by zero
        assert explanation is not None
        assert explanation.revenue_explanation is not None

    def test_unknown_trait_actionability(self, explainer):
        """Test actionability for unknown trait pattern."""
        trait = TraitValueScore(
            trait_name="xyz_unknown_field",
            trait_path="extra.xyz_unknown_field",
            trait_type="categorical",
            revenue_impact=0.5,
            customer_coverage=0.7,
            distinct_values=5,
            top_revenue_values=[("a", 300.0), ("b", 200.0)],
        )

        explanation = explainer.explain(trait)
        # Should be cautiously actionable
        assert "POTENTIALLY ACTIONABLE" in explanation.actionability_assessment
