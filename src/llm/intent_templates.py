"""
Module: intent_templates

Purpose: LLM prompt templates for segment evaluation and translation.

Key Classes:
- ActionabilityPrompt: Template for actionability evaluation
- BusinessTranslationPrompt: Template for business language translation
- IntentInterpreterPrompt: Template for goal interpretation

Architecture Notes:
- Templates use f-string formatting
- Include robustness context in prompts
- Support JSON output parsing
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any

from src.data.schemas import (
    ActionabilityDimension,
    ConfidenceLevel,
    RobustnessScore,
    RobustnessTier,
    Segment,
    StrategicGoal,
)


class PromptType(str, Enum):
    """Types of LLM prompts."""

    ACTIONABILITY = "actionability"
    BUSINESS_TRANSLATION = "business_translation"
    INTENT_INTERPRETATION = "intent_interpretation"
    CAMPAIGN_RECOMMENDATION = "campaign_recommendation"


@dataclass
class PromptTemplate:
    """Base class for prompt templates."""

    prompt_type: PromptType
    system_prompt: str
    user_prompt_template: str

    def format_user_prompt(self, **kwargs: Any) -> str:
        """Format the user prompt with provided variables."""
        return self.user_prompt_template.format(**kwargs)

    def get_full_prompt(self, **kwargs: Any) -> dict[str, str]:
        """Get full prompt with system and user messages."""
        return {
            "system": self.system_prompt,
            "user": self.format_user_prompt(**kwargs),
        }


# =============================================================================
# ACTIONABILITY PROMPT
# =============================================================================

ACTIONABILITY_SYSTEM_PROMPT = """You are a marketing analytics expert evaluating customer segments for commercial actionability.

Your role is to determine if a segment is actionable across four dimensions:
- WHAT: Can we determine what products/offers to target?
- WHEN: Can we optimize timing of communications?
- HOW: Can we personalize the channel/message?
- WHO: Can we prioritize within this segment?

A segment is actionable if it provides clear guidance for at least one dimension.
Reject segments that are too generic, too small, or lack clear business application.

Always provide reasoning and confidence levels based on the data quality indicators provided."""

ACTIONABILITY_USER_TEMPLATE = """Evaluate the following customer segment for commercial actionability:

SEGMENT DETAILS:
- Name: {segment_name}
- Size: {segment_size} customers
- Description: {segment_description}

VALUE METRICS:
- Total CLV: ${total_clv:,.2f}
- Average CLV: ${avg_clv:,.2f}
- Average Order Value: ${avg_order_value:,.2f}

DEFINING CHARACTERISTICS:
{defining_traits}

ROBUSTNESS INDICATORS:
- Overall Robustness: {overall_robustness:.1%}
- Robustness Tier: {robustness_tier}
- Feature Stability: {feature_stability:.1%}
- Time Consistency: {time_consistency:.1%}
- Production Ready: {is_production_ready}

Provide your evaluation in the following JSON format:
{{
    "is_actionable": true/false,
    "reasoning": "Explanation of your evaluation",
    "actionability_dimensions": ["WHAT", "WHEN", "HOW", "WHO"],
    "recommended_action": "Specific action recommendation",
    "confidence_level": "high/medium/low"
}}"""


def build_actionability_prompt() -> PromptTemplate:
    """Build the actionability evaluation prompt template."""
    return PromptTemplate(
        prompt_type=PromptType.ACTIONABILITY,
        system_prompt=ACTIONABILITY_SYSTEM_PROMPT,
        user_prompt_template=ACTIONABILITY_USER_TEMPLATE,
    )


def format_actionability_prompt(
    segment: Segment,
    robustness: RobustnessScore | None = None,
) -> dict[str, str]:
    """
    Format actionability prompt for a segment.

    Args:
        segment: Segment to evaluate
        robustness: Optional robustness score

    Returns:
        Dictionary with system and user prompts
    """
    # Format defining traits
    traits_text = "\n".join(
        f"- {trait}" for trait in segment.defining_traits
    ) if segment.defining_traits else "- No specific traits identified"

    # Handle robustness data
    if robustness:
        overall_robustness = robustness.overall_robustness
        robustness_tier = robustness.robustness_tier.value
        feature_stability = robustness.feature_stability
        time_consistency = robustness.time_window_consistency
        is_production_ready = str(robustness.is_production_ready)
    else:
        overall_robustness = 0.5
        robustness_tier = "UNKNOWN"
        feature_stability = 0.5
        time_consistency = 0.5
        is_production_ready = "Unknown"

    template = build_actionability_prompt()
    return template.get_full_prompt(
        segment_name=segment.name,
        segment_size=segment.size,
        segment_description=segment.description,
        total_clv=float(segment.total_clv),
        avg_clv=float(segment.avg_clv),
        avg_order_value=float(segment.avg_order_value),
        defining_traits=traits_text,
        overall_robustness=overall_robustness,
        robustness_tier=robustness_tier,
        feature_stability=feature_stability,
        time_consistency=time_consistency,
        is_production_ready=is_production_ready,
    )


# =============================================================================
# BUSINESS TRANSLATION PROMPT
# =============================================================================

BUSINESS_TRANSLATION_SYSTEM_PROMPT = """You are a business strategist translating data-driven customer segments into actionable business insights.

Your role is to:
1. Create executive-friendly summaries without technical jargon
2. Identify key characteristics that matter for business decisions
3. Recommend specific marketing campaigns
4. Articulate the business hypothesis behind targeting this segment

Use confident language for high-robustness segments and qualifying language for lower-robustness segments.
Always tie recommendations to expected business outcomes."""

BUSINESS_TRANSLATION_USER_TEMPLATE = """Translate the following customer segment into business insights:

SEGMENT OVERVIEW:
- Name: {segment_name}
- Size: {segment_size} customers
- Description: {segment_description}

BUSINESS VALUE:
- Total Customer Lifetime Value: ${total_clv:,.2f}
- Average CLV per Customer: ${avg_clv:,.2f}
- Average Order Value: ${avg_order_value:,.2f}

SEGMENT CHARACTERISTICS:
{defining_traits}

ACTIONABILITY ASSESSMENT:
- Actionable Dimensions: {actionability_dimensions}
- Strategic Goals: {strategic_goals}

DATA QUALITY (affects confidence):
- Overall Robustness: {overall_robustness:.1%}
- Data Tier: {robustness_tier}
- Feature Stability: {feature_stability:.1%}
- Time Consistency: {time_consistency:.1%}

Provide your business translation in the following JSON format:
{{
    "executive_summary": "2-3 sentence summary for executives",
    "key_characteristics": ["characteristic 1", "characteristic 2", "characteristic 3"],
    "recommended_campaign": "Specific campaign recommendation",
    "business_hypothesis": "The business hypothesis for targeting this segment",
    "expected_roi": "Expected ROI description",
    "confidence_level": "high/medium/low",
    "confidence_justification": "Why this confidence level"
}}"""


def build_business_translation_prompt() -> PromptTemplate:
    """Build the business translation prompt template."""
    return PromptTemplate(
        prompt_type=PromptType.BUSINESS_TRANSLATION,
        system_prompt=BUSINESS_TRANSLATION_SYSTEM_PROMPT,
        user_prompt_template=BUSINESS_TRANSLATION_USER_TEMPLATE,
    )


def format_business_translation_prompt(
    segment: Segment,
    robustness: RobustnessScore | None = None,
) -> dict[str, str]:
    """
    Format business translation prompt for a segment.

    Args:
        segment: Segment to translate
        robustness: Optional robustness score

    Returns:
        Dictionary with system and user prompts
    """
    # Format defining traits
    traits_text = "\n".join(
        f"- {trait}" for trait in segment.defining_traits
    ) if segment.defining_traits else "- General customer segment"

    # Format actionability dimensions
    dimensions_text = ", ".join(
        dim.value for dim in segment.actionability_dimensions
    ) if segment.actionability_dimensions else "None identified"

    # Format strategic goals
    goals_text = ", ".join(
        goal.value.replace("_", " ").title()
        for goal in segment.strategic_goals
    ) if segment.strategic_goals else "General business improvement"

    # Handle robustness data
    if robustness:
        overall_robustness = robustness.overall_robustness
        robustness_tier = robustness.robustness_tier.value
        feature_stability = robustness.feature_stability
        time_consistency = robustness.time_window_consistency
    else:
        overall_robustness = 0.5
        robustness_tier = "UNKNOWN"
        feature_stability = 0.5
        time_consistency = 0.5

    template = build_business_translation_prompt()
    return template.get_full_prompt(
        segment_name=segment.name,
        segment_size=segment.size,
        segment_description=segment.description,
        total_clv=float(segment.total_clv),
        avg_clv=float(segment.avg_clv),
        avg_order_value=float(segment.avg_order_value),
        defining_traits=traits_text,
        actionability_dimensions=dimensions_text,
        strategic_goals=goals_text,
        overall_robustness=overall_robustness,
        robustness_tier=robustness_tier,
        feature_stability=feature_stability,
        time_consistency=time_consistency,
    )


# =============================================================================
# INTENT INTERPRETATION PROMPT
# =============================================================================

INTENT_INTERPRETATION_SYSTEM_PROMPT = """You are a business analyst interpreting user goals and mapping them to customer segmentation strategies.

Your role is to:
1. Understand the user's business objective
2. Map it to relevant strategic goals (increase_revenue, reduce_churn, increase_satisfaction)
3. Identify which actionability dimensions are most relevant
4. Suggest segment characteristics to look for

Be specific and actionable in your recommendations."""

INTENT_INTERPRETATION_USER_TEMPLATE = """Interpret the following business goal and map it to segmentation strategy:

USER GOAL: {user_goal}

AVAILABLE DATA CONTEXT:
- Total Customers: {total_customers}
- Available Segments: {available_segments}
- Data Time Range: {data_time_range}

Provide your interpretation in the following JSON format:
{{
    "interpreted_goal": "Clear statement of the business objective",
    "strategic_goals": ["increase_revenue", "reduce_churn", "increase_satisfaction"],
    "relevant_dimensions": ["WHAT", "WHEN", "HOW", "WHO"],
    "recommended_segment_criteria": "What to look for in segments",
    "success_metrics": ["metric 1", "metric 2"]
}}"""


def build_intent_interpretation_prompt() -> PromptTemplate:
    """Build the intent interpretation prompt template."""
    return PromptTemplate(
        prompt_type=PromptType.INTENT_INTERPRETATION,
        system_prompt=INTENT_INTERPRETATION_SYSTEM_PROMPT,
        user_prompt_template=INTENT_INTERPRETATION_USER_TEMPLATE,
    )


def format_intent_interpretation_prompt(
    user_goal: str,
    total_customers: int,
    available_segments: int,
    data_time_range: str,
) -> dict[str, str]:
    """
    Format intent interpretation prompt.

    Args:
        user_goal: User's stated business goal
        total_customers: Total customer count
        available_segments: Number of available segments
        data_time_range: Description of data time range

    Returns:
        Dictionary with system and user prompts
    """
    template = build_intent_interpretation_prompt()
    return template.get_full_prompt(
        user_goal=user_goal,
        total_customers=total_customers,
        available_segments=available_segments,
        data_time_range=data_time_range,
    )


# =============================================================================
# CAMPAIGN RECOMMENDATION PROMPT
# =============================================================================

CAMPAIGN_RECOMMENDATION_SYSTEM_PROMPT = """You are a marketing strategist creating specific campaign recommendations for customer segments.

Your role is to:
1. Create detailed, actionable campaign recommendations
2. Specify channel, message, timing, and offer details
3. Set clear KPIs and success metrics
4. Account for segment characteristics and robustness

Be specific and practical in your recommendations."""

CAMPAIGN_RECOMMENDATION_USER_TEMPLATE = """Create a campaign recommendation for the following segment:

SEGMENT PROFILE:
- Name: {segment_name}
- Size: {segment_size} customers
- Total CLV: ${total_clv:,.2f}
- Avg Order Value: ${avg_order_value:,.2f}

KEY CHARACTERISTICS:
{defining_traits}

ACTIONABILITY:
- Dimensions: {actionability_dimensions}
- Goals: {strategic_goals}

DATA CONFIDENCE:
- Robustness: {robustness_tier}
- Production Ready: {is_production_ready}

BUDGET CONTEXT:
- Campaign Budget: ${campaign_budget:,.2f}
- Cost per Contact: ${cost_per_contact:.2f}

Provide your campaign recommendation in the following JSON format:
{{
    "campaign_name": "Campaign name",
    "campaign_type": "email/sms/push/display/direct_mail",
    "target_audience": "Description of who to target",
    "key_message": "Primary message/value proposition",
    "offer_details": "Specific offer if applicable",
    "timing": "When to execute",
    "channel_mix": ["primary channel", "secondary channel"],
    "success_kpis": ["KPI 1", "KPI 2"],
    "expected_results": "Expected outcomes",
    "risk_factors": ["Risk 1", "Risk 2"],
    "confidence_note": "Note on data confidence"
}}"""


def build_campaign_recommendation_prompt() -> PromptTemplate:
    """Build the campaign recommendation prompt template."""
    return PromptTemplate(
        prompt_type=PromptType.CAMPAIGN_RECOMMENDATION,
        system_prompt=CAMPAIGN_RECOMMENDATION_SYSTEM_PROMPT,
        user_prompt_template=CAMPAIGN_RECOMMENDATION_USER_TEMPLATE,
    )


def format_campaign_recommendation_prompt(
    segment: Segment,
    robustness: RobustnessScore | None = None,
    campaign_budget: Decimal = Decimal("10000"),
    cost_per_contact: Decimal = Decimal("5.00"),
) -> dict[str, str]:
    """
    Format campaign recommendation prompt for a segment.

    Args:
        segment: Segment for campaign
        robustness: Optional robustness score
        campaign_budget: Total campaign budget
        cost_per_contact: Cost per customer contact

    Returns:
        Dictionary with system and user prompts
    """
    # Format defining traits
    traits_text = "\n".join(
        f"- {trait}" for trait in segment.defining_traits
    ) if segment.defining_traits else "- General customer segment"

    # Format actionability dimensions
    dimensions_text = ", ".join(
        dim.value for dim in segment.actionability_dimensions
    ) if segment.actionability_dimensions else "General targeting"

    # Format strategic goals
    goals_text = ", ".join(
        goal.value.replace("_", " ").title()
        for goal in segment.strategic_goals
    ) if segment.strategic_goals else "Business growth"

    # Handle robustness data
    if robustness:
        robustness_tier = robustness.robustness_tier.value
        is_production_ready = str(robustness.is_production_ready)
    else:
        robustness_tier = "UNKNOWN"
        is_production_ready = "Unknown"

    template = build_campaign_recommendation_prompt()
    return template.get_full_prompt(
        segment_name=segment.name,
        segment_size=segment.size,
        total_clv=float(segment.total_clv),
        avg_order_value=float(segment.avg_order_value),
        defining_traits=traits_text,
        actionability_dimensions=dimensions_text,
        strategic_goals=goals_text,
        robustness_tier=robustness_tier,
        is_production_ready=is_production_ready,
        campaign_budget=float(campaign_budget),
        cost_per_contact=float(cost_per_contact),
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_confidence_language(confidence: ConfidenceLevel) -> dict[str, str]:
    """
    Get appropriate language qualifiers based on confidence level.

    Args:
        confidence: Confidence level

    Returns:
        Dictionary with language qualifiers
    """
    if confidence == ConfidenceLevel.HIGH:
        return {
            "certainty": "We are confident that",
            "recommendation": "We strongly recommend",
            "expectation": "will likely",
            "qualifier": "",
        }
    elif confidence == ConfidenceLevel.MEDIUM:
        return {
            "certainty": "Our analysis suggests that",
            "recommendation": "We recommend",
            "expectation": "should",
            "qualifier": "Based on current data, ",
        }
    else:
        return {
            "certainty": "Initial analysis indicates that",
            "recommendation": "Consider",
            "expectation": "may",
            "qualifier": "Preliminary findings suggest ",
        }


def get_robustness_context(robustness: RobustnessScore | None) -> str:
    """
    Generate robustness context for prompts.

    Args:
        robustness: Robustness score

    Returns:
        Context string about data quality
    """
    if robustness is None:
        return "Data quality unknown - interpret results with caution."

    if robustness.robustness_tier == RobustnessTier.HIGH:
        return (
            f"High data quality ({robustness.overall_robustness:.0%} robustness). "
            "Segment is stable and production-ready."
        )
    elif robustness.robustness_tier == RobustnessTier.MEDIUM:
        return (
            f"Moderate data quality ({robustness.overall_robustness:.0%} robustness). "
            "Segment requires monitoring; some instability observed."
        )
    else:
        return (
            f"Lower data quality ({robustness.overall_robustness:.0%} robustness). "
            "Segment may be unstable; use for exploratory purposes."
        )


def parse_json_response(response: str) -> dict[str, Any]:
    """
    Parse JSON from LLM response.

    Handles common formatting issues in LLM outputs.

    Args:
        response: Raw LLM response string

    Returns:
        Parsed JSON dictionary

    Raises:
        ValueError: If JSON cannot be parsed
    """
    import json
    import re

    # Try direct parse first
    try:
        result: dict[str, Any] = json.loads(response)
        return result
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    json_patterns = [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
        r"\{[\s\S]*\}",
    ]

    for pattern in json_patterns:
        match = re.search(pattern, response)
        if match:
            try:
                json_str = match.group(1) if "```" in pattern else match.group(0)
                parsed: dict[str, Any] = json.loads(json_str)
                return parsed
            except (json.JSONDecodeError, IndexError):
                continue

    raise ValueError(f"Could not parse JSON from response: {response[:200]}...")


def validate_actionability_response(response: dict[str, Any]) -> bool:
    """
    Validate actionability response structure.

    Args:
        response: Parsed response dictionary

    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        "is_actionable",
        "reasoning",
        "actionability_dimensions",
        "confidence_level",
    ]
    return all(field in response for field in required_fields)


def validate_business_translation_response(response: dict[str, Any]) -> bool:
    """
    Validate business translation response structure.

    Args:
        response: Parsed response dictionary

    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        "executive_summary",
        "key_characteristics",
        "recommended_campaign",
        "business_hypothesis",
        "confidence_level",
    ]
    return all(field in response for field in required_fields)
