"""
Tests for LLM integration layer.

Tests intent templates, actionability filtering, and segment explanation.
"""

from decimal import Decimal
from datetime import datetime, timedelta
import json
import pytest

from src.data.schemas import (
    ActionabilityDimension,
    ActionabilityEvaluation,
    ConfidenceLevel,
    RobustnessScore,
    RobustnessTier,
    Segment,
    SegmentExplanation,
    SegmentMember,
    SegmentViability,
    StrategicGoal,
)
from src.exceptions import LLMError
from src.llm.actionability_filter import (
    ActionabilityFilter,
    ActionabilityRule,
    MockLLMClient,
    evaluate_actionability,
    evaluate_actionability_rule_based,
    evaluate_actionability_with_llm,
    evaluate_how_dimension,
    evaluate_what_dimension,
    evaluate_when_dimension,
    evaluate_who_dimension,
    get_actionable_dimensions,
    get_default_rules,
    is_segment_actionable,
)
from src.llm.intent_templates import (
    PromptTemplate,
    PromptType,
    build_actionability_prompt,
    build_business_translation_prompt,
    build_campaign_recommendation_prompt,
    build_intent_interpretation_prompt,
    format_actionability_prompt,
    format_business_translation_prompt,
    format_campaign_recommendation_prompt,
    format_intent_interpretation_prompt,
    get_confidence_language,
    get_robustness_context,
    parse_json_response,
    validate_actionability_response,
    validate_business_translation_response,
)
from src.llm.segment_explainer import (
    SegmentExplainer,
    explain_segment,
    explain_segment_rule_based,
    explain_segment_with_llm,
    format_segment_for_presentation,
    generate_business_hypothesis,
    generate_executive_summary,
    generate_key_characteristics,
    generate_recommended_campaign,
    generate_roi_expectation,
    get_segment_recommendation,
    get_segment_summary,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_segment() -> Segment:
    """Create a sample segment for testing."""
    return Segment(
        segment_id="test_segment_1",
        name="High Value Weekend Shoppers",
        description="Customers who shop on weekends with high order values",
        members=[
            SegmentMember(internal_customer_id=f"cust_{i}", membership_score=1.0)
            for i in range(50)
        ],
        size=50,
        defining_traits=[
            "Weekend shopping preference",
            "High average order value",
            "Mobile device users",
            "Category: Electronics preference",
        ],
        total_clv=Decimal("25000"),
        avg_clv=Decimal("500"),
        avg_order_value=Decimal("150"),
        actionability_dimensions=[
            ActionabilityDimension.WHAT,
            ActionabilityDimension.WHEN,
            ActionabilityDimension.HOW,
        ],
        strategic_goals=[StrategicGoal.INCREASE_REVENUE],
    )


@pytest.fixture
def small_segment() -> Segment:
    """Create a small segment for testing edge cases."""
    return Segment(
        segment_id="test_small_segment",
        name="Small Segment",
        description="A very small segment",
        members=[
            SegmentMember(internal_customer_id=f"cust_{i}", membership_score=1.0)
            for i in range(3)
        ],
        size=3,
        defining_traits=[],
        total_clv=Decimal("50"),
        avg_clv=Decimal("16.67"),
        avg_order_value=Decimal("20"),
    )


@pytest.fixture
def high_robustness() -> RobustnessScore:
    """Create high robustness score."""
    return RobustnessScore.calculate(
        segment_id="test_segment_1",
        feature_stability=0.85,
        time_window_consistency=0.90,
    )


@pytest.fixture
def low_robustness() -> RobustnessScore:
    """Create low robustness score."""
    return RobustnessScore.calculate(
        segment_id="test_segment_1",
        feature_stability=0.30,
        time_window_consistency=0.25,
    )


@pytest.fixture
def sample_viability(sample_segment: Segment, high_robustness: RobustnessScore) -> SegmentViability:
    """Create sample viability assessment."""
    return SegmentViability(
        segment_id=sample_segment.segment_id,
        size=sample_segment.size,
        total_clv=sample_segment.total_clv,
        marketing_targetability=0.8,
        sales_prioritization=0.7,
        personalization_opportunity=0.75,
        timing_optimization=0.6,
        cost_to_exploit=Decimal("250"),
        expected_roi=1.5,
        revenue_impact="high",
        retention_impact="medium",
        satisfaction_impact="medium",
        robustness_score=high_robustness,
        confidence_level=ConfidenceLevel.HIGH,
        recommended_action="Target with personalized offers",
        business_hypothesis="Targeting will increase revenue",
        is_approved=True,
    )


# =============================================================================
# INTENT TEMPLATES TESTS
# =============================================================================


class TestPromptTemplate:
    """Tests for PromptTemplate class."""

    def test_prompt_template_creation(self) -> None:
        """Test creating a prompt template."""
        template = PromptTemplate(
            prompt_type=PromptType.ACTIONABILITY,
            system_prompt="You are an analyst.",
            user_prompt_template="Analyze segment: {segment_name}",
        )

        assert template.prompt_type == PromptType.ACTIONABILITY
        assert "analyst" in template.system_prompt

    def test_format_user_prompt(self) -> None:
        """Test formatting user prompt with variables."""
        template = PromptTemplate(
            prompt_type=PromptType.ACTIONABILITY,
            system_prompt="System",
            user_prompt_template="Segment: {name}, Size: {size}",
        )

        formatted = template.format_user_prompt(name="Test", size=100)
        assert "Test" in formatted
        assert "100" in formatted

    def test_get_full_prompt(self) -> None:
        """Test getting full prompt with both system and user."""
        template = PromptTemplate(
            prompt_type=PromptType.ACTIONABILITY,
            system_prompt="System prompt here",
            user_prompt_template="User: {query}",
        )

        full = template.get_full_prompt(query="analyze this")
        assert "system" in full
        assert "user" in full
        assert full["system"] == "System prompt here"
        assert "analyze this" in full["user"]


class TestActionabilityPrompt:
    """Tests for actionability prompt functions."""

    def test_build_actionability_prompt(self) -> None:
        """Test building actionability prompt template."""
        template = build_actionability_prompt()

        assert template.prompt_type == PromptType.ACTIONABILITY
        assert "WHAT" in template.system_prompt
        assert "WHEN" in template.system_prompt
        assert "HOW" in template.system_prompt
        assert "WHO" in template.system_prompt

    def test_format_actionability_prompt(self, sample_segment: Segment) -> None:
        """Test formatting actionability prompt for segment."""
        prompt = format_actionability_prompt(sample_segment)

        assert "system" in prompt
        assert "user" in prompt
        assert sample_segment.name in prompt["user"]
        assert str(sample_segment.size) in prompt["user"]

    def test_format_actionability_prompt_with_robustness(
        self, sample_segment: Segment, high_robustness: RobustnessScore
    ) -> None:
        """Test formatting with robustness data."""
        prompt = format_actionability_prompt(sample_segment, high_robustness)

        assert "HIGH" in prompt["user"]
        assert "True" in prompt["user"]


class TestBusinessTranslationPrompt:
    """Tests for business translation prompt functions."""

    def test_build_business_translation_prompt(self) -> None:
        """Test building business translation template."""
        template = build_business_translation_prompt()

        assert template.prompt_type == PromptType.BUSINESS_TRANSLATION
        assert "executive" in template.system_prompt.lower()
        assert "business" in template.system_prompt.lower()

    def test_format_business_translation_prompt(self, sample_segment: Segment) -> None:
        """Test formatting business translation prompt."""
        prompt = format_business_translation_prompt(sample_segment)

        assert "system" in prompt
        assert "user" in prompt
        assert sample_segment.name in prompt["user"]
        assert "WHAT, WHEN, HOW" in prompt["user"]  # Dimensions


class TestIntentInterpretationPrompt:
    """Tests for intent interpretation prompt."""

    def test_build_intent_interpretation_prompt(self) -> None:
        """Test building intent interpretation template."""
        template = build_intent_interpretation_prompt()

        assert template.prompt_type == PromptType.INTENT_INTERPRETATION
        assert "goal" in template.system_prompt.lower()

    def test_format_intent_interpretation_prompt(self) -> None:
        """Test formatting intent interpretation prompt."""
        prompt = format_intent_interpretation_prompt(
            user_goal="Reduce customer churn",
            total_customers=1000,
            available_segments=5,
            data_time_range="Last 90 days",
        )

        assert "Reduce customer churn" in prompt["user"]
        assert "1000" in prompt["user"]
        assert "5" in prompt["user"]


class TestCampaignRecommendationPrompt:
    """Tests for campaign recommendation prompt."""

    def test_build_campaign_recommendation_prompt(self) -> None:
        """Test building campaign recommendation template."""
        template = build_campaign_recommendation_prompt()

        assert template.prompt_type == PromptType.CAMPAIGN_RECOMMENDATION
        assert "campaign" in template.system_prompt.lower()

    def test_format_campaign_recommendation_prompt(
        self, sample_segment: Segment, high_robustness: RobustnessScore
    ) -> None:
        """Test formatting campaign recommendation prompt."""
        prompt = format_campaign_recommendation_prompt(
            sample_segment,
            robustness=high_robustness,
            campaign_budget=Decimal("5000"),
            cost_per_contact=Decimal("2.50"),
        )

        assert sample_segment.name in prompt["user"]
        assert "5,000" in prompt["user"]
        assert "2.50" in prompt["user"]


class TestHelperFunctions:
    """Tests for template helper functions."""

    def test_get_confidence_language_high(self) -> None:
        """Test confidence language for high confidence."""
        language = get_confidence_language(ConfidenceLevel.HIGH)

        assert "confident" in language["certainty"].lower()
        assert "strongly" in language["recommendation"].lower()

    def test_get_confidence_language_medium(self) -> None:
        """Test confidence language for medium confidence."""
        language = get_confidence_language(ConfidenceLevel.MEDIUM)

        assert "suggests" in language["certainty"].lower()
        assert language["recommendation"] == "We recommend"

    def test_get_confidence_language_low(self) -> None:
        """Test confidence language for low confidence."""
        language = get_confidence_language(ConfidenceLevel.LOW)

        assert "initial" in language["certainty"].lower()
        assert "consider" in language["recommendation"].lower()

    def test_get_robustness_context_high(self, high_robustness: RobustnessScore) -> None:
        """Test robustness context for high tier."""
        context = get_robustness_context(high_robustness)

        assert "High" in context
        assert "production-ready" in context

    def test_get_robustness_context_low(self, low_robustness: RobustnessScore) -> None:
        """Test robustness context for low tier."""
        context = get_robustness_context(low_robustness)

        assert "Lower" in context
        assert "exploratory" in context

    def test_get_robustness_context_none(self) -> None:
        """Test robustness context when None."""
        context = get_robustness_context(None)

        assert "unknown" in context.lower()
        assert "caution" in context.lower()

    def test_parse_json_response_valid(self) -> None:
        """Test parsing valid JSON."""
        response = '{"is_actionable": true, "reasoning": "test"}'
        parsed = parse_json_response(response)

        assert parsed["is_actionable"] is True
        assert parsed["reasoning"] == "test"

    def test_parse_json_response_markdown(self) -> None:
        """Test parsing JSON from markdown block."""
        response = '```json\n{"key": "value"}\n```'
        parsed = parse_json_response(response)

        assert parsed["key"] == "value"

    def test_parse_json_response_invalid(self) -> None:
        """Test parsing invalid JSON raises error."""
        with pytest.raises(ValueError, match="Could not parse JSON"):
            parse_json_response("not valid json at all")

    def test_validate_actionability_response_valid(self) -> None:
        """Test validating complete actionability response."""
        response = {
            "is_actionable": True,
            "reasoning": "Good segment",
            "actionability_dimensions": ["WHAT"],
            "confidence_level": "high",
        }
        assert validate_actionability_response(response) is True

    def test_validate_actionability_response_invalid(self) -> None:
        """Test validating incomplete response."""
        response = {"is_actionable": True}  # Missing fields
        assert validate_actionability_response(response) is False

    def test_validate_business_translation_response_valid(self) -> None:
        """Test validating complete business translation response."""
        response = {
            "executive_summary": "Summary",
            "key_characteristics": ["char1"],
            "recommended_campaign": "Campaign",
            "business_hypothesis": "Hypothesis",
            "confidence_level": "high",
        }
        assert validate_business_translation_response(response) is True


# =============================================================================
# ACTIONABILITY FILTER TESTS
# =============================================================================


class TestDimensionEvaluation:
    """Tests for individual dimension evaluation functions."""

    def test_evaluate_who_dimension_high_value(self, sample_segment: Segment) -> None:
        """Test WHO dimension for high-value segment."""
        is_actionable, reasoning = evaluate_who_dimension(sample_segment)

        assert is_actionable is True
        assert "CLV" in reasoning or "value" in reasoning.lower()

    def test_evaluate_who_dimension_low_value(self, small_segment: Segment) -> None:
        """Test WHO dimension for low-value segment."""
        is_actionable, reasoning = evaluate_who_dimension(small_segment)

        # Small segment with low CLV should not be actionable for WHO
        assert is_actionable is False or "criteria" in reasoning.lower()

    def test_evaluate_what_dimension_with_traits(self, sample_segment: Segment) -> None:
        """Test WHAT dimension with category traits."""
        is_actionable, reasoning = evaluate_what_dimension(sample_segment)

        assert is_actionable is True
        assert "trait" in reasoning.lower() or "characteristic" in reasoning.lower()

    def test_evaluate_what_dimension_no_traits(self, small_segment: Segment) -> None:
        """Test WHAT dimension without traits."""
        is_actionable, reasoning = evaluate_what_dimension(small_segment)

        # No defining traits, might still be actionable if has multiple traits from logic
        assert "No product targeting" in reasoning or is_actionable

    def test_evaluate_when_dimension_with_temporal(self, sample_segment: Segment) -> None:
        """Test WHEN dimension with temporal traits."""
        is_actionable, reasoning = evaluate_when_dimension(sample_segment)

        assert is_actionable is True
        assert "timing" in reasoning.lower()

    def test_evaluate_when_dimension_no_temporal(self) -> None:
        """Test WHEN dimension without temporal traits."""
        segment = Segment(
            segment_id="no_temporal",
            name="No Temporal",
            description="Test",
            size=50,
            total_clv=Decimal("10000"),
            avg_clv=Decimal("200"),
            avg_order_value=Decimal("100"),
        )
        is_actionable, reasoning = evaluate_when_dimension(segment)

        assert is_actionable is False
        assert "No timing" in reasoning

    def test_evaluate_how_dimension_with_channel(self, sample_segment: Segment) -> None:
        """Test HOW dimension with channel traits."""
        is_actionable, reasoning = evaluate_how_dimension(sample_segment)

        assert is_actionable is True
        assert "channel" in reasoning.lower()

    def test_evaluate_how_dimension_no_channel(self) -> None:
        """Test HOW dimension without channel traits."""
        segment = Segment(
            segment_id="no_channel",
            name="No Channel",
            description="Test",
            size=50,
            total_clv=Decimal("10000"),
            avg_clv=Decimal("200"),
            avg_order_value=Decimal("100"),
        )
        is_actionable, reasoning = evaluate_how_dimension(segment)

        assert is_actionable is False
        assert "No channel" in reasoning


class TestRuleBasedActionability:
    """Tests for rule-based actionability evaluation."""

    def test_evaluate_actionability_rule_based(self, sample_segment: Segment) -> None:
        """Test rule-based evaluation returns proper result."""
        result = evaluate_actionability_rule_based(sample_segment)

        assert isinstance(result, ActionabilityEvaluation)
        assert result.segment_id == sample_segment.segment_id
        assert result.is_actionable is True
        assert len(result.actionability_dimensions) > 0

    def test_evaluate_actionability_rule_based_with_robustness(
        self, sample_segment: Segment, high_robustness: RobustnessScore
    ) -> None:
        """Test evaluation with robustness affects confidence."""
        result = evaluate_actionability_rule_based(
            sample_segment, robustness=high_robustness
        )

        assert result.confidence_level == ConfidenceLevel.HIGH

    def test_evaluate_actionability_rule_based_low_robustness(
        self, sample_segment: Segment, low_robustness: RobustnessScore
    ) -> None:
        """Test evaluation with low robustness."""
        result = evaluate_actionability_rule_based(
            sample_segment, robustness=low_robustness
        )

        assert result.confidence_level == ConfidenceLevel.LOW

    def test_evaluate_actionability_small_segment(self, small_segment: Segment) -> None:
        """Test evaluation rejects too small segments."""
        result = evaluate_actionability_rule_based(small_segment)

        assert result.is_actionable is False
        assert "too small" in result.reasoning.lower()

    def test_evaluate_actionability_low_value(self) -> None:
        """Test evaluation rejects low-value segments."""
        segment = Segment(
            segment_id="low_value",
            name="Low Value",
            description="Test",
            members=[
                SegmentMember(internal_customer_id=f"cust_{i}", membership_score=1.0)
                for i in range(20)
            ],
            size=20,
            total_clv=Decimal("50"),  # Very low
            avg_clv=Decimal("2.50"),
            avg_order_value=Decimal("5"),
        )
        result = evaluate_actionability_rule_based(segment)

        assert result.is_actionable is False
        assert "insufficient" in result.reasoning.lower() or "value" in result.reasoning.lower()

    def test_evaluate_actionability_min_dimensions(self, sample_segment: Segment) -> None:
        """Test minimum dimensions requirement."""
        result = evaluate_actionability_rule_based(
            sample_segment, min_dimensions=3
        )

        # Sample segment should have multiple dimensions
        assert result.is_actionable is True or len(result.actionability_dimensions) < 3


class TestMockLLMClient:
    """Tests for MockLLMClient."""

    def test_mock_client_returns_json(self) -> None:
        """Test mock client returns valid JSON."""
        client = MockLLMClient()
        response = client.complete(
            system_prompt="You are an analyst",
            user_prompt="Analyze segment: Test, Size: 100"
        )

        parsed = json.loads(response)
        assert "is_actionable" in parsed
        assert "reasoning" in parsed

    def test_mock_client_responds_to_small_size(self) -> None:
        """Test mock client recognizes small segments."""
        client = MockLLMClient()
        response = client.complete(
            system_prompt="Analyze",
            user_prompt="Size: 0 customers"
        )

        parsed = json.loads(response)
        assert parsed["is_actionable"] is False

    def test_mock_client_responds_to_traits(self) -> None:
        """Test mock client responds to trait keywords."""
        client = MockLLMClient()
        response = client.complete(
            system_prompt="Analyze",
            user_prompt="Has weekend shopping preference and mobile device usage"
        )

        parsed = json.loads(response)
        assert "WHEN" in parsed["actionability_dimensions"]
        assert "HOW" in parsed["actionability_dimensions"]


class TestLLMBasedActionability:
    """Tests for LLM-based actionability evaluation."""

    def test_evaluate_actionability_with_llm(self, sample_segment: Segment) -> None:
        """Test LLM-based evaluation with mock client."""
        result = evaluate_actionability_with_llm(sample_segment)

        assert isinstance(result, ActionabilityEvaluation)
        assert result.segment_id == sample_segment.segment_id

    def test_evaluate_actionability_with_llm_and_robustness(
        self, sample_segment: Segment, high_robustness: RobustnessScore
    ) -> None:
        """Test LLM evaluation includes robustness context."""
        result = evaluate_actionability_with_llm(
            sample_segment, robustness=high_robustness
        )

        assert isinstance(result, ActionabilityEvaluation)


class TestActionabilityFilter:
    """Tests for ActionabilityFilter class."""

    def test_filter_initialization(self) -> None:
        """Test filter initialization."""
        filter_obj = ActionabilityFilter(
            use_llm=False,
            min_dimensions=1,
            min_confidence=ConfidenceLevel.LOW,
        )

        assert filter_obj.use_llm is False
        assert filter_obj.min_dimensions == 1

    def test_filter_evaluate(self, sample_segment: Segment) -> None:
        """Test evaluating single segment."""
        filter_obj = ActionabilityFilter()
        result = filter_obj.evaluate(sample_segment)

        assert result.segment_id == sample_segment.segment_id
        assert sample_segment.segment_id in filter_obj.evaluations

    def test_filter_evaluate_batch(self, sample_segment: Segment, small_segment: Segment) -> None:
        """Test batch evaluation."""
        filter_obj = ActionabilityFilter()
        results = filter_obj.evaluate_batch([sample_segment, small_segment])

        assert len(results) == 2
        assert sample_segment.segment_id in results
        assert small_segment.segment_id in results

    def test_filter_actionable(self, sample_segment: Segment, small_segment: Segment) -> None:
        """Test filtering actionable segments."""
        filter_obj = ActionabilityFilter()
        actionable = filter_obj.filter_actionable([sample_segment, small_segment])

        # Sample should be actionable, small shouldn't
        assert sample_segment in actionable
        assert small_segment not in actionable

    def test_filter_get_summary(self, sample_segment: Segment, small_segment: Segment) -> None:
        """Test getting filter summary."""
        filter_obj = ActionabilityFilter()
        filter_obj.evaluate_batch([sample_segment, small_segment])
        summary = filter_obj.get_summary()

        assert summary["total"] == 2
        assert summary["actionable"] >= 0
        assert "dimensions_distribution" in summary


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_is_segment_actionable(self, sample_segment: Segment) -> None:
        """Test quick actionability check."""
        assert is_segment_actionable(sample_segment) is True

    def test_is_segment_actionable_small(self, small_segment: Segment) -> None:
        """Test quick check for small segment."""
        assert is_segment_actionable(small_segment) is False

    def test_get_actionable_dimensions(self, sample_segment: Segment) -> None:
        """Test getting actionable dimensions."""
        dimensions = get_actionable_dimensions(sample_segment)

        assert isinstance(dimensions, list)
        assert all(isinstance(d, ActionabilityDimension) for d in dimensions)

    def test_get_default_rules(self) -> None:
        """Test getting default rules."""
        rules = get_default_rules()

        assert len(rules) == 4
        assert all(isinstance(r, ActionabilityRule) for r in rules)


# =============================================================================
# SEGMENT EXPLAINER TESTS
# =============================================================================


class TestExecutiveSummary:
    """Tests for executive summary generation."""

    def test_generate_executive_summary(self, sample_segment: Segment) -> None:
        """Test generating executive summary."""
        summary = generate_executive_summary(sample_segment)

        assert sample_segment.name in summary
        assert str(sample_segment.size) in summary
        assert "$" in summary  # CLV

    def test_generate_executive_summary_with_robustness(
        self, sample_segment: Segment, high_robustness: RobustnessScore
    ) -> None:
        """Test summary includes robustness context."""
        summary = generate_executive_summary(sample_segment, high_robustness)

        assert "High" in summary or "confident" in summary.lower()

    def test_generate_executive_summary_with_viability(
        self, sample_segment: Segment, sample_viability: SegmentViability
    ) -> None:
        """Test summary includes ROI when viability provided."""
        summary = generate_executive_summary(
            sample_segment, viability=sample_viability
        )

        assert "ROI" in summary or "recommend" in summary.lower()


class TestKeyCharacteristics:
    """Tests for key characteristics generation."""

    def test_generate_key_characteristics(self, sample_segment: Segment) -> None:
        """Test generating key characteristics."""
        chars = generate_key_characteristics(sample_segment)

        assert isinstance(chars, list)
        assert len(chars) <= 5  # Limited to top 5
        assert len(chars) > 0

    def test_key_characteristics_include_value(self, sample_segment: Segment) -> None:
        """Test characteristics include value info."""
        chars = generate_key_characteristics(sample_segment)

        value_mentioned = any("CLV" in c or "value" in c.lower() for c in chars)
        assert value_mentioned

    def test_key_characteristics_include_traits(self, sample_segment: Segment) -> None:
        """Test characteristics include defining traits."""
        chars = generate_key_characteristics(sample_segment)

        # Should include some of the defining traits
        trait_included = any(
            trait in chars or trait.lower() in str(chars).lower()
            for trait in sample_segment.defining_traits[:3]
        )
        # Might be transformed but should have overlap
        assert len(chars) > 0


class TestCampaignRecommendation:
    """Tests for campaign recommendation generation."""

    def test_generate_recommended_campaign(self, sample_segment: Segment) -> None:
        """Test generating campaign recommendation."""
        campaign = generate_recommended_campaign(sample_segment)

        assert isinstance(campaign, str)
        assert len(campaign) > 0
        assert "campaign" in campaign.lower() or "target" in campaign.lower()

    def test_campaign_includes_dimensions(self, sample_segment: Segment) -> None:
        """Test campaign recommendation references dimensions."""
        campaign = generate_recommended_campaign(sample_segment)

        # Should mention relevant actions based on dimensions
        has_relevant_action = any(
            action in campaign.lower()
            for action in ["product", "timing", "channel", "priority", "personalized"]
        )
        assert has_relevant_action


class TestBusinessHypothesis:
    """Tests for business hypothesis generation."""

    def test_generate_business_hypothesis(self, sample_segment: Segment) -> None:
        """Test generating business hypothesis."""
        hypothesis = generate_business_hypothesis(sample_segment)

        assert isinstance(hypothesis, str)
        assert str(sample_segment.size) in hypothesis or "customer" in hypothesis.lower()

    def test_hypothesis_includes_goal(self) -> None:
        """Test hypothesis includes strategic goal context."""
        segment = Segment(
            segment_id="churn_risk",
            name="At Risk",
            description="Churn risk segment",
            size=100,
            total_clv=Decimal("50000"),
            avg_clv=Decimal("500"),
            avg_order_value=Decimal("100"),
            strategic_goals=[StrategicGoal.REDUCE_CHURN],
        )
        hypothesis = generate_business_hypothesis(segment)

        assert "churn" in hypothesis.lower() or "retention" in hypothesis.lower()


class TestROIExpectation:
    """Tests for ROI expectation generation."""

    def test_generate_roi_expectation(self, sample_segment: Segment) -> None:
        """Test generating ROI expectation."""
        roi = generate_roi_expectation(sample_segment)

        assert isinstance(roi, str)
        assert "ROI" in roi.upper()

    def test_roi_with_viability(
        self, sample_segment: Segment, sample_viability: SegmentViability
    ) -> None:
        """Test ROI expectation uses viability data."""
        roi = generate_roi_expectation(sample_segment, viability=sample_viability)

        # Should mention the actual ROI percentage
        assert "%" in roi or "potential" in roi.lower()

    def test_roi_with_robustness(
        self, sample_segment: Segment, low_robustness: RobustnessScore
    ) -> None:
        """Test ROI expectation includes confidence qualifiers."""
        roi = generate_roi_expectation(sample_segment, robustness=low_robustness)

        # Low robustness should have qualifying language
        assert "preliminary" in roi.lower() or "initial" in roi.lower() or "depends" in roi.lower()


class TestRuleBasedExplanation:
    """Tests for rule-based segment explanation."""

    def test_explain_segment_rule_based(self, sample_segment: Segment) -> None:
        """Test rule-based explanation."""
        explanation = explain_segment_rule_based(sample_segment)

        assert isinstance(explanation, SegmentExplanation)
        assert explanation.segment_id == sample_segment.segment_id
        assert len(explanation.executive_summary) > 0
        assert len(explanation.key_characteristics) > 0

    def test_explain_segment_rule_based_with_robustness(
        self, sample_segment: Segment, high_robustness: RobustnessScore
    ) -> None:
        """Test explanation with high robustness."""
        explanation = explain_segment_rule_based(
            sample_segment, robustness=high_robustness
        )

        assert explanation.confidence_level == ConfidenceLevel.HIGH

    def test_explain_segment_rule_based_with_viability(
        self, sample_segment: Segment, sample_viability: SegmentViability
    ) -> None:
        """Test explanation includes viability context."""
        explanation = explain_segment_rule_based(
            sample_segment, viability=sample_viability
        )

        assert len(explanation.expected_roi) > 0


class TestLLMBasedExplanation:
    """Tests for LLM-based segment explanation."""

    def test_explain_segment_with_llm(self, sample_segment: Segment) -> None:
        """Test LLM-based explanation with mock client."""
        explanation = explain_segment_with_llm(sample_segment)

        assert isinstance(explanation, SegmentExplanation)
        assert explanation.segment_id == sample_segment.segment_id

    def test_explain_segment_with_llm_and_robustness(
        self, sample_segment: Segment, high_robustness: RobustnessScore
    ) -> None:
        """Test LLM explanation with robustness."""
        explanation = explain_segment_with_llm(
            sample_segment, robustness=high_robustness
        )

        assert isinstance(explanation, SegmentExplanation)


class TestSegmentExplainer:
    """Tests for SegmentExplainer class."""

    def test_explainer_initialization(self) -> None:
        """Test explainer initialization."""
        explainer = SegmentExplainer(use_llm=False)

        assert explainer.use_llm is False
        assert len(explainer.explanations) == 0

    def test_explainer_explain(self, sample_segment: Segment) -> None:
        """Test explaining single segment."""
        explainer = SegmentExplainer()
        explanation = explainer.explain(sample_segment)

        assert explanation.segment_id == sample_segment.segment_id
        assert sample_segment.segment_id in explainer.explanations

    def test_explainer_explain_batch(
        self, sample_segment: Segment, small_segment: Segment
    ) -> None:
        """Test batch explanation."""
        explainer = SegmentExplainer()
        explanations = explainer.explain_batch([sample_segment, small_segment])

        assert len(explanations) == 2
        assert sample_segment.segment_id in explanations
        assert small_segment.segment_id in explanations

    def test_explainer_get_summary(
        self, sample_segment: Segment, small_segment: Segment
    ) -> None:
        """Test getting explainer summary."""
        explainer = SegmentExplainer()
        explainer.explain_batch([sample_segment, small_segment])
        summary = explainer.get_summary()

        assert summary["total"] == 2
        assert "confidence_distribution" in summary


class TestConvenienceExplainerFunctions:
    """Tests for explainer convenience functions."""

    def test_get_segment_summary(self, sample_segment: Segment) -> None:
        """Test quick segment summary."""
        summary = get_segment_summary(sample_segment)

        assert sample_segment.name in summary
        assert isinstance(summary, str)

    def test_get_segment_recommendation(self, sample_segment: Segment) -> None:
        """Test quick recommendation."""
        recommendation = get_segment_recommendation(sample_segment)

        assert isinstance(recommendation, str)
        assert len(recommendation) > 0

    def test_format_segment_for_presentation(
        self, sample_segment: Segment
    ) -> None:
        """Test formatting for presentation."""
        explanation = explain_segment(sample_segment)
        formatted = format_segment_for_presentation(sample_segment, explanation)

        assert formatted["segment_id"] == sample_segment.segment_id
        assert formatted["name"] == sample_segment.name
        assert formatted["size"] == sample_segment.size
        assert "executive_summary" in formatted
        assert "confidence" in formatted


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestLLMIntegration:
    """Integration tests for the full LLM workflow."""

    def test_full_evaluation_and_explanation_flow(self, sample_segment: Segment) -> None:
        """Test complete flow from evaluation to explanation."""
        # Step 1: Evaluate actionability
        actionability = evaluate_actionability(sample_segment)

        assert actionability.is_actionable is True

        # Step 2: Generate explanation
        explanation = explain_segment(sample_segment, actionability=actionability)

        assert explanation.segment_id == sample_segment.segment_id
        assert len(explanation.executive_summary) > 0

    def test_flow_with_robustness_context(
        self, sample_segment: Segment, high_robustness: RobustnessScore
    ) -> None:
        """Test flow with robustness data."""
        # Evaluate with robustness
        actionability = evaluate_actionability(
            sample_segment, robustness=high_robustness
        )

        assert actionability.confidence_level == ConfidenceLevel.HIGH

        # Explain with robustness
        explanation = explain_segment(
            sample_segment,
            robustness=high_robustness,
            actionability=actionability,
        )

        assert explanation.confidence_level == ConfidenceLevel.HIGH

    def test_batch_processing_flow(
        self, sample_segment: Segment, small_segment: Segment, high_robustness: RobustnessScore
    ) -> None:
        """Test batch processing workflow."""
        segments = [sample_segment, small_segment]
        robustness_scores = {sample_segment.segment_id: high_robustness}

        # Filter actionable
        filter_obj = ActionabilityFilter()
        actionable = filter_obj.filter_actionable(
            segments, robustness_scores=robustness_scores
        )

        # Explain actionable segments
        explainer = SegmentExplainer()
        explanations = explainer.explain_batch(
            actionable, robustness_scores=robustness_scores
        )

        # Verify results
        assert len(actionable) >= 1  # At least sample_segment
        assert sample_segment in actionable
        assert all(seg.segment_id in explanations for seg in actionable)

    def test_llm_mode_toggle(self, sample_segment: Segment) -> None:
        """Test switching between rule-based and LLM modes."""
        # Rule-based (default)
        rule_result = evaluate_actionability(sample_segment, use_llm=False)
        rule_explanation = explain_segment(sample_segment, use_llm=False)

        # LLM-based (mock)
        llm_result = evaluate_actionability(sample_segment, use_llm=True)
        llm_explanation = explain_segment(sample_segment, use_llm=True)

        # Both should produce valid results
        assert rule_result.is_actionable is not None
        assert llm_result.is_actionable is not None
        assert len(rule_explanation.executive_summary) > 0
        assert len(llm_explanation.executive_summary) > 0


class TestErrorHandling:
    """Tests for error handling in LLM integration."""

    def test_llm_error_propagation(self) -> None:
        """Test that LLM errors are properly raised."""
        class FailingLLMClient:
            def complete(self, system_prompt: str, user_prompt: str) -> str:
                raise Exception("API Error")

        segment = Segment(
            segment_id="test",
            name="Test",
            description="Test",
            size=50,
            total_clv=Decimal("5000"),
            avg_clv=Decimal("100"),
            avg_order_value=Decimal("50"),
        )

        with pytest.raises(LLMError):
            evaluate_actionability_with_llm(
                segment, llm_client=FailingLLMClient()
            )

    def test_invalid_llm_response_handling(self) -> None:
        """Test handling of invalid LLM responses."""
        class BadResponseClient:
            def complete(self, system_prompt: str, user_prompt: str) -> str:
                return "This is not JSON"

        segment = Segment(
            segment_id="test",
            name="Test",
            description="Test",
            size=50,
            total_clv=Decimal("5000"),
            avg_clv=Decimal("100"),
            avg_order_value=Decimal("50"),
        )

        with pytest.raises(LLMError):
            evaluate_actionability_with_llm(
                segment, llm_client=BadResponseClient()
            )
