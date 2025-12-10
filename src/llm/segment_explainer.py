"""
Module: segment_explainer

Purpose: Translate data-driven segments into business insights.

Key Functions:
- explain_segment: Generate business-language explanation
- generate_executive_summary: Create executive summary
- SegmentExplanation: Container for segment explanation

Architecture Notes:
- Uses mocked LLM responses for MVP
- Includes confidence levels based on robustness
- Produces human-readable recommendations
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

from src.data.schemas import (
    ActionabilityDimension,
    ActionabilityEvaluation,
    ConfidenceLevel,
    RobustnessScore,
    RobustnessTier,
    Segment,
    SegmentExplanation,
    SegmentViability,
    StrategicGoal,
)
from src.exceptions import LLMError
from src.llm.intent_templates import (
    format_business_translation_prompt,
    format_campaign_recommendation_prompt,
    get_confidence_language,
    get_robustness_context,
    parse_json_response,
    validate_business_translation_response,
)


# =============================================================================
# PROTOCOLS
# =============================================================================


class LLMClient(Protocol):
    """Protocol for LLM client implementations."""

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Send prompt to LLM and get response."""
        ...


# =============================================================================
# MOCK LLM CLIENT
# =============================================================================


class MockLLMClient:
    """
    Mock LLM client for testing and MVP.

    Generates deterministic business explanations based on segment characteristics.
    """

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Return mock LLM response based on prompt content."""
        return self._generate_mock_response(user_prompt)

    def _generate_mock_response(self, user_prompt: str) -> str:
        """Generate mock business translation response."""
        import json
        import re

        # Extract segment info from prompt
        prompt_lower = user_prompt.lower()

        # Extract segment name
        name_match = re.search(r"name:\s*([^\n]+)", user_prompt)
        segment_name = name_match.group(1).strip() if name_match else "Customer Segment"

        # Extract size
        size_match = re.search(r"size:\s*(\d+)", prompt_lower)
        segment_size = int(size_match.group(1)) if size_match else 100

        # Extract CLV
        clv_match = re.search(r"total.*clv[^$]*\$([0-9,.]+)", prompt_lower)
        total_clv = clv_match.group(1).replace(",", "") if clv_match else "10000"

        # Determine confidence based on robustness
        confidence = "medium"
        confidence_justification = "Moderate data quality supports these recommendations."

        if "robustness tier: high" in prompt_lower or "robustness: 7" in prompt_lower or "robustness: 8" in prompt_lower or "robustness: 9" in prompt_lower:
            confidence = "high"
            confidence_justification = "High data quality and segment stability provide strong confidence in these insights."
        elif "robustness tier: low" in prompt_lower or "robustness: 0" in prompt_lower or "robustness: 1" in prompt_lower or "robustness: 2" in prompt_lower:
            confidence = "low"
            confidence_justification = "Lower data stability suggests these recommendations should be validated with testing."

        # Generate characteristics based on traits
        characteristics = []
        if "high value" in prompt_lower or "clv" in prompt_lower:
            characteristics.append("High-value customers with significant lifetime value")
        if "weekend" in prompt_lower:
            characteristics.append("Weekend shopping preference")
        if "mobile" in prompt_lower:
            characteristics.append("Mobile-first engagement behavior")
        if "category" in prompt_lower or "product" in prompt_lower:
            characteristics.append("Clear product category preferences")
        if "frequent" in prompt_lower:
            characteristics.append("Frequent purchase behavior")

        if not characteristics:
            characteristics = [
                "Defined customer behavior patterns",
                "Identifiable value characteristics",
                "Targetable segment attributes",
            ]

        # Generate campaign recommendation
        campaign = "Targeted email campaign with personalized product recommendations"
        if "mobile" in prompt_lower:
            campaign = "Mobile-first push notification campaign with exclusive offers"
        elif "weekend" in prompt_lower:
            campaign = "Weekend-optimized promotional campaign"
        elif "high value" in prompt_lower:
            campaign = "VIP experience campaign with early access and exclusive benefits"

        # Generate business hypothesis
        hypothesis = f"Targeting this segment of {segment_size} customers with personalized engagement will increase conversion rates and customer lifetime value"

        # Generate ROI expectation
        roi = "Positive ROI expected based on segment value and targeting precision"
        if confidence == "high":
            roi = "Strong positive ROI expected with estimated 15-25% uplift in segment engagement"
        elif confidence == "low":
            roi = "ROI potential exists but requires validation through controlled testing"

        # Generate executive summary
        summary = (
            f"This segment represents {segment_size} customers with ${total_clv} in total lifetime value. "
            f"Analysis indicates clear opportunities for targeted engagement. "
            f"{'Strong data quality supports confident action.' if confidence == 'high' else 'Recommend pilot testing before full rollout.'}"
        )

        response = {
            "executive_summary": summary,
            "key_characteristics": characteristics[:3],
            "recommended_campaign": campaign,
            "business_hypothesis": hypothesis,
            "expected_roi": roi,
            "confidence_level": confidence,
            "confidence_justification": confidence_justification,
        }

        return json.dumps(response)


# =============================================================================
# RULE-BASED EXPLANATION GENERATION
# =============================================================================


def generate_executive_summary(
    segment: Segment,
    robustness: RobustnessScore | None = None,
    viability: SegmentViability | None = None,
) -> str:
    """
    Generate executive summary for a segment.

    Args:
        segment: Segment to summarize
        robustness: Optional robustness score
        viability: Optional viability assessment

    Returns:
        Executive summary string
    """
    # Get confidence language
    confidence = ConfidenceLevel.MEDIUM
    if robustness:
        if robustness.robustness_tier == RobustnessTier.HIGH:
            confidence = ConfidenceLevel.HIGH
        elif robustness.robustness_tier == RobustnessTier.LOW:
            confidence = ConfidenceLevel.LOW

    language = get_confidence_language(confidence)

    # Build summary parts
    parts: list[str] = []

    # Size and value
    total_clv = float(segment.total_clv)
    avg_clv = float(segment.avg_clv)

    parts.append(
        f"{segment.name} contains {segment.size} customers "
        f"representing ${total_clv:,.2f} in total lifetime value."
    )

    # Value characterization
    if avg_clv > 500:
        parts.append("These are premium customers with high individual value.")
    elif avg_clv > 200:
        parts.append("This segment shows above-average customer value.")
    elif avg_clv > 100:
        parts.append("Customer value is moderate with growth potential.")

    # Actionability
    if segment.actionability_dimensions:
        dims = [d.value for d in segment.actionability_dimensions]
        parts.append(
            f"{language['certainty']} this segment is actionable "
            f"across {', '.join(dims)} dimensions."
        )

    # Robustness context
    if robustness:
        context = get_robustness_context(robustness)
        parts.append(context)

    # ROI potential
    if viability:
        if viability.expected_roi > 1.0:
            parts.append(
                f"{language['recommendation']} prioritizing this segment "
                f"with expected ROI of {viability.expected_roi:.0%}."
            )

    return " ".join(parts)


def generate_key_characteristics(segment: Segment) -> list[str]:
    """
    Generate list of key characteristics for a segment.

    Args:
        segment: Segment to analyze

    Returns:
        List of characteristic descriptions
    """
    characteristics: list[str] = []

    # Value characteristics
    avg_clv = float(segment.avg_clv)
    avg_aov = float(segment.avg_order_value)

    if avg_clv > 500:
        characteristics.append(f"Premium customer value (${avg_clv:.0f} avg CLV)")
    elif avg_clv > 200:
        characteristics.append(f"High customer value (${avg_clv:.0f} avg CLV)")
    elif avg_clv > 100:
        characteristics.append(f"Moderate customer value (${avg_clv:.0f} avg CLV)")

    if avg_aov > 150:
        characteristics.append(f"High average order value (${avg_aov:.0f})")
    elif avg_aov > 75:
        characteristics.append(f"Moderate order value (${avg_aov:.0f})")

    # Size characteristics
    if segment.size > 500:
        characteristics.append(f"Large segment ({segment.size} customers)")
    elif segment.size > 100:
        characteristics.append(f"Medium-sized segment ({segment.size} customers)")
    else:
        characteristics.append(f"Focused segment ({segment.size} customers)")

    # Defining traits
    for trait in segment.defining_traits[:3]:
        characteristics.append(trait)

    # Strategic goals
    for goal in segment.strategic_goals:
        if goal == StrategicGoal.REDUCE_CHURN:
            characteristics.append("Churn risk mitigation opportunity")
        elif goal == StrategicGoal.INCREASE_REVENUE:
            characteristics.append("Revenue growth potential")
        elif goal == StrategicGoal.INCREASE_SATISFACTION:
            characteristics.append("Customer satisfaction focus")

    return characteristics[:5]  # Limit to top 5


def generate_recommended_campaign(
    segment: Segment,
    actionability: ActionabilityEvaluation | None = None,
) -> str:
    """
    Generate campaign recommendation for a segment.

    Args:
        segment: Segment for campaign
        actionability: Optional actionability evaluation

    Returns:
        Campaign recommendation string
    """
    campaign_parts: list[str] = []

    # Determine campaign type based on dimensions
    dimensions = segment.actionability_dimensions
    if actionability:
        dimensions = actionability.actionability_dimensions or dimensions

    # Base campaign type
    if ActionabilityDimension.WHAT in dimensions:
        campaign_parts.append("personalized product recommendations")

    if ActionabilityDimension.WHEN in dimensions:
        campaign_parts.append("timing-optimized delivery")

    if ActionabilityDimension.HOW in dimensions:
        campaign_parts.append("channel-specific messaging")

    if ActionabilityDimension.WHO in dimensions:
        campaign_parts.append("priority customer outreach")

    # Strategic goal influence
    if StrategicGoal.REDUCE_CHURN in segment.strategic_goals:
        campaign_parts.append("retention-focused incentives")
    elif StrategicGoal.INCREASE_REVENUE in segment.strategic_goals:
        campaign_parts.append("upsell and cross-sell opportunities")

    if campaign_parts:
        return f"Launch a targeted campaign with {', '.join(campaign_parts)}."

    return "Develop a targeted engagement campaign based on segment characteristics."


def generate_business_hypothesis(
    segment: Segment,
    viability: SegmentViability | None = None,
) -> str:
    """
    Generate business hypothesis for targeting a segment.

    Args:
        segment: Segment to analyze
        viability: Optional viability assessment

    Returns:
        Business hypothesis string
    """
    hypotheses: list[str] = []

    # Value-based hypothesis
    total_clv = float(segment.total_clv)
    if total_clv > 50000:
        hypotheses.append(
            f"Targeted engagement with this ${total_clv:,.0f} value segment "
            "can drive significant revenue impact."
        )
    elif total_clv > 10000:
        hypotheses.append(
            f"This segment's ${total_clv:,.0f} total value justifies "
            "dedicated marketing investment."
        )

    # Goal-based hypothesis
    if StrategicGoal.REDUCE_CHURN in segment.strategic_goals:
        hypotheses.append(
            f"Proactive retention efforts for {segment.size} at-risk customers "
            "can prevent revenue loss and improve LTV."
        )
    elif StrategicGoal.INCREASE_REVENUE in segment.strategic_goals:
        hypotheses.append(
            f"Personalized offers to {segment.size} customers "
            "should increase purchase frequency and order value."
        )
    elif StrategicGoal.INCREASE_SATISFACTION in segment.strategic_goals:
        hypotheses.append(
            f"Improved experience for {segment.size} customers "
            "will boost satisfaction scores and advocacy."
        )

    # ROI-based hypothesis
    if viability and viability.expected_roi > 0:
        hypotheses.append(
            f"Campaign investment is expected to yield {viability.expected_roi:.0%} ROI "
            "based on segment economics."
        )

    if hypotheses:
        return " ".join(hypotheses[:2])

    return (
        f"Targeting {segment.size} customers with tailored engagement "
        "should improve key business metrics."
    )


def generate_roi_expectation(
    segment: Segment,
    viability: SegmentViability | None = None,
    robustness: RobustnessScore | None = None,
) -> str:
    """
    Generate ROI expectation description.

    Args:
        segment: Segment to analyze
        viability: Optional viability assessment
        robustness: Optional robustness score

    Returns:
        ROI expectation string
    """
    # Get confidence
    confidence = ConfidenceLevel.MEDIUM
    if robustness:
        if robustness.robustness_tier == RobustnessTier.HIGH:
            confidence = ConfidenceLevel.HIGH
        elif robustness.robustness_tier == RobustnessTier.LOW:
            confidence = ConfidenceLevel.LOW

    language = get_confidence_language(confidence)

    if viability:
        roi = viability.expected_roi
        if roi > 2.0:
            return (
                f"{language['qualifier']}Excellent ROI potential of {roi:.0%}. "
                f"This segment {language['expectation']} deliver strong returns."
            )
        elif roi > 1.0:
            return (
                f"{language['qualifier']}Good ROI potential of {roi:.0%}. "
                f"Campaign investment {language['expectation']} be profitable."
            )
        elif roi > 0.5:
            return (
                f"{language['qualifier']}Moderate ROI potential of {roi:.0%}. "
                "Returns are expected but margins may be thin."
            )
        else:
            return (
                f"{language['qualifier']}Limited ROI potential of {roi:.0%}. "
                "Consider optimizing targeting or reducing costs."
            )

    # Without viability data
    total_clv = float(segment.total_clv)
    if total_clv > 50000:
        return f"{language['qualifier']}High value segment suggests strong ROI potential."
    elif total_clv > 10000:
        return f"{language['qualifier']}Moderate value suggests positive ROI is achievable."
    else:
        return f"{language['qualifier']}ROI depends on campaign cost efficiency."


def explain_segment_rule_based(
    segment: Segment,
    *,
    robustness: RobustnessScore | None = None,
    viability: SegmentViability | None = None,
    actionability: ActionabilityEvaluation | None = None,
) -> SegmentExplanation:
    """
    Generate segment explanation using rule-based logic.

    This is the primary explanation method for MVP, not requiring LLM.

    Args:
        segment: Segment to explain
        robustness: Optional robustness score
        viability: Optional viability assessment
        actionability: Optional actionability evaluation

    Returns:
        SegmentExplanation
    """
    # Determine confidence level
    confidence = ConfidenceLevel.MEDIUM
    if robustness:
        if robustness.robustness_tier == RobustnessTier.HIGH:
            confidence = ConfidenceLevel.HIGH
        elif robustness.robustness_tier == RobustnessTier.LOW:
            confidence = ConfidenceLevel.LOW

    # Generate components
    executive_summary = generate_executive_summary(segment, robustness, viability)
    key_characteristics = generate_key_characteristics(segment)
    recommended_campaign = generate_recommended_campaign(segment, actionability)
    business_hypothesis = generate_business_hypothesis(segment, viability)
    expected_roi = generate_roi_expectation(segment, viability, robustness)
    confidence_justification = get_robustness_context(robustness)

    return SegmentExplanation(
        segment_id=segment.segment_id,
        executive_summary=executive_summary,
        key_characteristics=key_characteristics,
        recommended_campaign=recommended_campaign,
        business_hypothesis=business_hypothesis,
        expected_roi=expected_roi,
        confidence_level=confidence,
        confidence_justification=confidence_justification,
    )


# =============================================================================
# LLM-BASED EXPLANATION GENERATION
# =============================================================================


def explain_segment_with_llm(
    segment: Segment,
    *,
    robustness: RobustnessScore | None = None,
    llm_client: LLMClient | None = None,
) -> SegmentExplanation:
    """
    Generate segment explanation using LLM.

    Args:
        segment: Segment to explain
        robustness: Optional robustness score
        llm_client: LLM client (uses MockLLMClient if None)

    Returns:
        SegmentExplanation

    Raises:
        LLMError: If LLM explanation fails
    """
    if llm_client is None:
        llm_client = MockLLMClient()

    # Format prompt
    prompt = format_business_translation_prompt(segment, robustness)

    try:
        # Get LLM response
        response_text = llm_client.complete(
            system_prompt=prompt["system"],
            user_prompt=prompt["user"],
        )

        # Parse response
        response = parse_json_response(response_text)

        # Validate response structure
        if not validate_business_translation_response(response):
            raise ValueError("Invalid response structure from LLM")

        # Parse confidence
        confidence_str = response.get("confidence_level", "medium").lower()
        if confidence_str == "high":
            confidence = ConfidenceLevel.HIGH
        elif confidence_str == "low":
            confidence = ConfidenceLevel.LOW
        else:
            confidence = ConfidenceLevel.MEDIUM

        return SegmentExplanation(
            segment_id=segment.segment_id,
            executive_summary=response.get("executive_summary", ""),
            key_characteristics=response.get("key_characteristics", []),
            recommended_campaign=response.get("recommended_campaign", ""),
            business_hypothesis=response.get("business_hypothesis", ""),
            expected_roi=response.get("expected_roi", ""),
            confidence_level=confidence,
            confidence_justification=response.get("confidence_justification", ""),
        )

    except Exception as e:
        raise LLMError(
            f"Segment explanation failed: {e}",
            prompt_type="business_translation",
            segment_id=segment.segment_id,
        ) from e


# =============================================================================
# MAIN EXPLANATION FUNCTION
# =============================================================================


def explain_segment(
    segment: Segment,
    *,
    robustness: RobustnessScore | None = None,
    viability: SegmentViability | None = None,
    actionability: ActionabilityEvaluation | None = None,
    use_llm: bool = False,
    llm_client: LLMClient | None = None,
) -> SegmentExplanation:
    """
    Generate business explanation for a segment.

    Primary entry point for segment explanation. Uses rule-based
    explanation by default, with optional LLM enhancement.

    Args:
        segment: Segment to explain
        robustness: Optional robustness score
        viability: Optional viability assessment
        actionability: Optional actionability evaluation
        use_llm: Whether to use LLM for explanation
        llm_client: LLM client (uses mock if None)

    Returns:
        SegmentExplanation
    """
    if use_llm:
        return explain_segment_with_llm(
            segment,
            robustness=robustness,
            llm_client=llm_client,
        )
    else:
        return explain_segment_rule_based(
            segment,
            robustness=robustness,
            viability=viability,
            actionability=actionability,
        )


# =============================================================================
# SEGMENT EXPLAINER CLASS
# =============================================================================


class SegmentExplainer:
    """
    Generate business explanations for segments.

    Translates data-driven segments into actionable business insights.
    """

    def __init__(
        self,
        *,
        use_llm: bool = False,
        llm_client: LLMClient | None = None,
    ) -> None:
        """
        Initialize SegmentExplainer.

        Args:
            use_llm: Whether to use LLM for explanations
            llm_client: LLM client instance
        """
        self.use_llm = use_llm
        self.llm_client = llm_client

        self._explanations: dict[str, SegmentExplanation] = {}

    def explain(
        self,
        segment: Segment,
        *,
        robustness: RobustnessScore | None = None,
        viability: SegmentViability | None = None,
        actionability: ActionabilityEvaluation | None = None,
    ) -> SegmentExplanation:
        """
        Generate explanation for a single segment.

        Args:
            segment: Segment to explain
            robustness: Optional robustness score
            viability: Optional viability assessment
            actionability: Optional actionability evaluation

        Returns:
            SegmentExplanation
        """
        explanation = explain_segment(
            segment,
            robustness=robustness,
            viability=viability,
            actionability=actionability,
            use_llm=self.use_llm,
            llm_client=self.llm_client,
        )

        self._explanations[segment.segment_id] = explanation
        return explanation

    def explain_batch(
        self,
        segments: list[Segment],
        *,
        robustness_scores: dict[str, RobustnessScore] | None = None,
        viabilities: dict[str, SegmentViability] | None = None,
        actionabilities: dict[str, ActionabilityEvaluation] | None = None,
    ) -> dict[str, SegmentExplanation]:
        """
        Generate explanations for multiple segments.

        Args:
            segments: Segments to explain
            robustness_scores: Optional mapping of segment_id to RobustnessScore
            viabilities: Optional mapping of segment_id to SegmentViability
            actionabilities: Optional mapping of segment_id to ActionabilityEvaluation

        Returns:
            Dictionary mapping segment_id to SegmentExplanation
        """
        robustness_scores = robustness_scores or {}
        viabilities = viabilities or {}
        actionabilities = actionabilities or {}

        results: dict[str, SegmentExplanation] = {}

        for segment in segments:
            results[segment.segment_id] = self.explain(
                segment,
                robustness=robustness_scores.get(segment.segment_id),
                viability=viabilities.get(segment.segment_id),
                actionability=actionabilities.get(segment.segment_id),
            )

        return results

    def get_summary(self) -> dict[str, Any]:
        """
        Get summary of generated explanations.

        Returns:
            Summary statistics
        """
        if not self._explanations:
            return {
                "total": 0,
                "confidence_distribution": {},
            }

        total = len(self._explanations)

        # Count confidence levels
        confidence_counts: dict[str, int] = {}
        for explanation in self._explanations.values():
            level = explanation.confidence_level.value
            confidence_counts[level] = confidence_counts.get(level, 0) + 1

        return {
            "total": total,
            "confidence_distribution": confidence_counts,
        }

    @property
    def explanations(self) -> dict[str, SegmentExplanation]:
        """Get all explanations."""
        return self._explanations.copy()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_segment_summary(
    segment: Segment,
    robustness: RobustnessScore | None = None,
) -> str:
    """
    Get quick summary of a segment.

    Args:
        segment: Segment to summarize
        robustness: Optional robustness score

    Returns:
        Summary string
    """
    return generate_executive_summary(segment, robustness)


def get_segment_recommendation(
    segment: Segment,
    actionability: ActionabilityEvaluation | None = None,
) -> str:
    """
    Get campaign recommendation for a segment.

    Args:
        segment: Segment for recommendation
        actionability: Optional actionability evaluation

    Returns:
        Recommendation string
    """
    return generate_recommended_campaign(segment, actionability)


def format_segment_for_presentation(
    segment: Segment,
    explanation: SegmentExplanation,
) -> dict[str, Any]:
    """
    Format segment and explanation for presentation.

    Args:
        segment: Segment data
        explanation: Generated explanation

    Returns:
        Formatted dictionary for presentation
    """
    return {
        "segment_id": segment.segment_id,
        "name": segment.name,
        "size": segment.size,
        "total_clv": float(segment.total_clv),
        "avg_clv": float(segment.avg_clv),
        "executive_summary": explanation.executive_summary,
        "key_characteristics": explanation.key_characteristics,
        "recommended_campaign": explanation.recommended_campaign,
        "business_hypothesis": explanation.business_hypothesis,
        "expected_roi": explanation.expected_roi,
        "confidence": explanation.confidence_level.value,
        "confidence_note": explanation.confidence_justification,
    }
