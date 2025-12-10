"""
Module: actionability_filter

Purpose: LLM-powered actionability evaluation for segments.

Key Functions:
- evaluate_actionability: Assess if segment is commercially actionable
- ActionabilityResult: Container for evaluation result

Architecture Notes:
- Uses mocked LLM responses for MVP
- Evaluates WHAT/WHEN/HOW/WHO dimensions
- Rejects segments with no clear business play
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
    StrategicGoal,
)
from src.exceptions import LLMError
from src.llm.intent_templates import (
    format_actionability_prompt,
    get_confidence_language,
    get_robustness_context,
    parse_json_response,
    validate_actionability_response,
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
        """
        Send prompt to LLM and get response.

        Args:
            system_prompt: System/context prompt
            user_prompt: User/query prompt

        Returns:
            LLM response string
        """
        ...


# =============================================================================
# MOCK LLM CLIENT
# =============================================================================


class MockLLMClient:
    """
    Mock LLM client for testing and MVP.

    Returns deterministic responses based on segment characteristics.
    """

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Return mock LLM response based on prompt content."""
        # Parse key information from user prompt to generate response
        response = self._generate_mock_response(user_prompt)
        return response

    def _generate_mock_response(self, user_prompt: str) -> str:
        """Generate mock response based on prompt analysis."""
        import json

        # Extract segment info from prompt
        is_actionable = True
        dimensions: list[str] = []
        confidence = "medium"
        reasoning = "Based on segment characteristics analysis."
        recommended_action = "Target with personalized campaigns."

        # Simple heuristics based on prompt content
        prompt_lower = user_prompt.lower()

        # Check size
        if "size: 0 customers" in prompt_lower or "size: 1 customers" in prompt_lower:
            is_actionable = False
            reasoning = "Segment too small for meaningful targeting."
            confidence = "high"

        # Check CLV
        if "total clv: $0" in prompt_lower:
            is_actionable = False
            reasoning = "Segment has no economic value."
            confidence = "high"

        # Determine dimensions based on traits and characteristics
        if "high value" in prompt_lower or "clv" in prompt_lower:
            dimensions.append("WHO")
        if "weekend" in prompt_lower or "morning" in prompt_lower or "evening" in prompt_lower:
            dimensions.append("WHEN")
        if "mobile" in prompt_lower or "device" in prompt_lower or "channel" in prompt_lower:
            dimensions.append("HOW")
        if "category" in prompt_lower or "product" in prompt_lower or "preference" in prompt_lower:
            dimensions.append("WHAT")

        # Default dimensions if none found
        if not dimensions and is_actionable:
            dimensions = ["WHO"]

        # Check robustness
        if "robustness tier: high" in prompt_lower:
            confidence = "high"
        elif "robustness tier: low" in prompt_lower:
            confidence = "low"

        # Generate appropriate action
        if dimensions:
            action_parts = []
            if "WHAT" in dimensions:
                action_parts.append("product recommendations")
            if "WHEN" in dimensions:
                action_parts.append("optimized send times")
            if "HOW" in dimensions:
                action_parts.append("channel-specific messaging")
            if "WHO" in dimensions:
                action_parts.append("priority outreach")

            if action_parts:
                recommended_action = f"Target with {', '.join(action_parts)}."

        response = {
            "is_actionable": is_actionable,
            "reasoning": reasoning,
            "actionability_dimensions": dimensions,
            "recommended_action": recommended_action if is_actionable else None,
            "confidence_level": confidence,
        }

        return json.dumps(response)


# =============================================================================
# RULE-BASED ACTIONABILITY EVALUATION
# =============================================================================


@dataclass
class ActionabilityRule:
    """Rule for evaluating actionability."""

    name: str
    dimension: ActionabilityDimension
    condition_description: str


def get_default_rules() -> list[ActionabilityRule]:
    """Get default actionability evaluation rules."""
    return [
        ActionabilityRule(
            name="high_value_customer",
            dimension=ActionabilityDimension.WHO,
            condition_description="High CLV or high order value customers",
        ),
        ActionabilityRule(
            name="category_preference",
            dimension=ActionabilityDimension.WHAT,
            condition_description="Clear product/category preferences",
        ),
        ActionabilityRule(
            name="temporal_pattern",
            dimension=ActionabilityDimension.WHEN,
            condition_description="Identifiable timing patterns",
        ),
        ActionabilityRule(
            name="channel_preference",
            dimension=ActionabilityDimension.HOW,
            condition_description="Clear channel or device preferences",
        ),
    ]


def evaluate_who_dimension(segment: Segment) -> tuple[bool, str]:
    """
    Evaluate WHO actionability dimension.

    WHO: Can we prioritize customers within this segment?

    Args:
        segment: Segment to evaluate

    Returns:
        Tuple of (is_actionable, reasoning)
    """
    reasons: list[str] = []

    # Check for high-value indicators
    avg_clv = float(segment.avg_clv)
    avg_aov = float(segment.avg_order_value)

    if avg_clv > 200:
        reasons.append(f"High average CLV (${avg_clv:.2f})")
    elif avg_clv > 100:
        reasons.append(f"Moderate CLV (${avg_clv:.2f})")

    if avg_aov > 100:
        reasons.append(f"High average order value (${avg_aov:.2f})")

    # Check for size suitability
    if 10 <= segment.size <= 500:
        reasons.append(f"Suitable segment size ({segment.size})")

    # Check for value concentration
    total_clv = float(segment.total_clv)
    if total_clv > 10000:
        reasons.append(f"Significant total value (${total_clv:,.2f})")

    is_actionable = len(reasons) >= 2
    reasoning = "; ".join(reasons) if reasons else "No prioritization criteria met"

    return is_actionable, reasoning


def evaluate_what_dimension(segment: Segment) -> tuple[bool, str]:
    """
    Evaluate WHAT actionability dimension.

    WHAT: Can we determine what products/offers to target?

    Args:
        segment: Segment to evaluate

    Returns:
        Tuple of (is_actionable, reasoning)
    """
    reasons: list[str] = []

    # Check for product/category traits
    category_keywords = ["category", "product", "preference", "affinity", "interest"]
    category_traits = [
        trait for trait in segment.defining_traits
        if any(kw in trait.lower() for kw in category_keywords)
    ]

    if category_traits:
        reasons.append(f"Has {len(category_traits)} product-related traits")

    # Check trait summary for category info
    if segment.trait_summary:
        if "top_category" in segment.trait_summary:
            reasons.append(f"Clear top category: {segment.trait_summary['top_category']}")
        if "category_preferences" in segment.trait_summary:
            reasons.append("Has category preference data")

    # Check defining traits for specificity
    if len(segment.defining_traits) >= 2:
        reasons.append(f"{len(segment.defining_traits)} defining characteristics")

    is_actionable = len(reasons) >= 1
    reasoning = "; ".join(reasons) if reasons else "No product targeting criteria identified"

    return is_actionable, reasoning


def evaluate_when_dimension(segment: Segment) -> tuple[bool, str]:
    """
    Evaluate WHEN actionability dimension.

    WHEN: Can we optimize timing of communications?

    Args:
        segment: Segment to evaluate

    Returns:
        Tuple of (is_actionable, reasoning)
    """
    reasons: list[str] = []

    # Check for temporal traits
    temporal_keywords = [
        "weekend", "weekday", "morning", "evening", "night",
        "seasonal", "early", "late", "time", "hour", "day"
    ]
    temporal_traits = [
        trait for trait in segment.defining_traits
        if any(kw in trait.lower() for kw in temporal_keywords)
    ]

    if temporal_traits:
        reasons.append(f"Has {len(temporal_traits)} timing-related traits")

    # Check trait summary for temporal info
    if segment.trait_summary:
        if "preferred_day" in segment.trait_summary:
            reasons.append(f"Preferred day identified: {segment.trait_summary['preferred_day']}")
        if "preferred_hour" in segment.trait_summary:
            reasons.append(f"Preferred hour identified: {segment.trait_summary['preferred_hour']}")
        if "purchase_timing" in segment.trait_summary:
            reasons.append("Has purchase timing data")

    is_actionable = len(reasons) >= 1
    reasoning = "; ".join(reasons) if reasons else "No timing optimization criteria identified"

    return is_actionable, reasoning


def evaluate_how_dimension(segment: Segment) -> tuple[bool, str]:
    """
    Evaluate HOW actionability dimension.

    HOW: Can we personalize the channel/message?

    Args:
        segment: Segment to evaluate

    Returns:
        Tuple of (is_actionable, reasoning)
    """
    reasons: list[str] = []

    # Check for channel/device traits
    channel_keywords = [
        "mobile", "desktop", "device", "channel", "email",
        "sms", "push", "app", "web", "browser"
    ]
    channel_traits = [
        trait for trait in segment.defining_traits
        if any(kw in trait.lower() for kw in channel_keywords)
    ]

    if channel_traits:
        reasons.append(f"Has {len(channel_traits)} channel-related traits")

    # Check trait summary for channel info
    if segment.trait_summary:
        if "primary_device" in segment.trait_summary:
            reasons.append(f"Primary device: {segment.trait_summary['primary_device']}")
        if "mobile_preference" in segment.trait_summary:
            reasons.append("Has mobile preference data")
        if "preferred_channel" in segment.trait_summary:
            reasons.append(f"Preferred channel: {segment.trait_summary['preferred_channel']}")

    is_actionable = len(reasons) >= 1
    reasoning = "; ".join(reasons) if reasons else "No channel personalization criteria identified"

    return is_actionable, reasoning


def evaluate_actionability_rule_based(
    segment: Segment,
    *,
    robustness: RobustnessScore | None = None,
    min_dimensions: int = 1,
) -> ActionabilityEvaluation:
    """
    Evaluate segment actionability using rule-based logic.

    This is the primary evaluation method for MVP, not requiring LLM.

    Args:
        segment: Segment to evaluate
        robustness: Optional robustness score
        min_dimensions: Minimum actionable dimensions required

    Returns:
        ActionabilityEvaluation
    """
    # Evaluate each dimension
    who_actionable, who_reasoning = evaluate_who_dimension(segment)
    what_actionable, what_reasoning = evaluate_what_dimension(segment)
    when_actionable, when_reasoning = evaluate_when_dimension(segment)
    how_actionable, how_reasoning = evaluate_how_dimension(segment)

    # Collect actionable dimensions
    actionable_dimensions: list[ActionabilityDimension] = []
    reasoning_parts: list[str] = []

    if who_actionable:
        actionable_dimensions.append(ActionabilityDimension.WHO)
        reasoning_parts.append(f"WHO: {who_reasoning}")

    if what_actionable:
        actionable_dimensions.append(ActionabilityDimension.WHAT)
        reasoning_parts.append(f"WHAT: {what_reasoning}")

    if when_actionable:
        actionable_dimensions.append(ActionabilityDimension.WHEN)
        reasoning_parts.append(f"WHEN: {when_reasoning}")

    if how_actionable:
        actionable_dimensions.append(ActionabilityDimension.HOW)
        reasoning_parts.append(f"HOW: {how_reasoning}")

    # Determine overall actionability
    is_actionable = len(actionable_dimensions) >= min_dimensions

    # Apply size constraints
    if segment.size < 5:
        is_actionable = False
        reasoning_parts.insert(0, "Segment too small for meaningful targeting")

    if float(segment.total_clv) < 100:
        is_actionable = False
        reasoning_parts.insert(0, "Segment has insufficient economic value")

    # Determine confidence based on robustness
    if robustness:
        if robustness.robustness_tier == RobustnessTier.HIGH:
            confidence = ConfidenceLevel.HIGH
        elif robustness.robustness_tier == RobustnessTier.MEDIUM:
            confidence = ConfidenceLevel.MEDIUM
        else:
            confidence = ConfidenceLevel.LOW
    else:
        # Base confidence on segment characteristics
        if segment.size >= 50 and len(actionable_dimensions) >= 2:
            confidence = ConfidenceLevel.MEDIUM
        else:
            confidence = ConfidenceLevel.LOW

    # Generate recommended action
    recommended_action = None
    if is_actionable:
        recommended_action = _generate_recommended_action(actionable_dimensions)

    return ActionabilityEvaluation(
        segment_id=segment.segment_id,
        is_actionable=is_actionable,
        reasoning=" | ".join(reasoning_parts) if reasoning_parts else "No actionability criteria met",
        recommended_action=recommended_action,
        confidence_level=confidence,
        actionability_dimensions=actionable_dimensions,
    )


def _generate_recommended_action(dimensions: list[ActionabilityDimension]) -> str:
    """Generate recommended action based on actionable dimensions."""
    actions = []

    if ActionabilityDimension.WHAT in dimensions:
        actions.append("personalized product recommendations")
    if ActionabilityDimension.WHEN in dimensions:
        actions.append("timing-optimized campaigns")
    if ActionabilityDimension.HOW in dimensions:
        actions.append("channel-specific messaging")
    if ActionabilityDimension.WHO in dimensions:
        actions.append("priority outreach")

    if actions:
        return f"Target with {', '.join(actions)}"
    return "Review for targeting opportunities"


# =============================================================================
# LLM-BASED ACTIONABILITY EVALUATION
# =============================================================================


def evaluate_actionability_with_llm(
    segment: Segment,
    *,
    robustness: RobustnessScore | None = None,
    llm_client: LLMClient | None = None,
) -> ActionabilityEvaluation:
    """
    Evaluate segment actionability using LLM.

    Args:
        segment: Segment to evaluate
        robustness: Optional robustness score
        llm_client: LLM client (uses MockLLMClient if None)

    Returns:
        ActionabilityEvaluation

    Raises:
        LLMError: If LLM evaluation fails
    """
    if llm_client is None:
        llm_client = MockLLMClient()

    # Format prompt
    prompt = format_actionability_prompt(segment, robustness)

    try:
        # Get LLM response
        response_text = llm_client.complete(
            system_prompt=prompt["system"],
            user_prompt=prompt["user"],
        )

        # Parse response
        response = parse_json_response(response_text)

        # Validate response structure
        if not validate_actionability_response(response):
            raise ValueError("Invalid response structure from LLM")

        # Parse dimensions
        dimensions = [
            ActionabilityDimension(dim)
            for dim in response.get("actionability_dimensions", [])
            if dim in [d.value for d in ActionabilityDimension]
        ]

        # Parse confidence
        confidence_str = response.get("confidence_level", "medium").lower()
        if confidence_str == "high":
            confidence = ConfidenceLevel.HIGH
        elif confidence_str == "low":
            confidence = ConfidenceLevel.LOW
        else:
            confidence = ConfidenceLevel.MEDIUM

        return ActionabilityEvaluation(
            segment_id=segment.segment_id,
            is_actionable=response.get("is_actionable", False),
            reasoning=response.get("reasoning", ""),
            recommended_action=response.get("recommended_action"),
            confidence_level=confidence,
            actionability_dimensions=dimensions,
        )

    except Exception as e:
        raise LLMError(
            f"Actionability evaluation failed: {e}",
            prompt_type="actionability",
            segment_id=segment.segment_id,
        ) from e


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================


def evaluate_actionability(
    segment: Segment,
    *,
    robustness: RobustnessScore | None = None,
    use_llm: bool = False,
    llm_client: LLMClient | None = None,
    min_dimensions: int = 1,
) -> ActionabilityEvaluation:
    """
    Evaluate segment actionability.

    Primary entry point for actionability evaluation. Uses rule-based
    evaluation by default, with optional LLM enhancement.

    Args:
        segment: Segment to evaluate
        robustness: Optional robustness score
        use_llm: Whether to use LLM for evaluation
        llm_client: LLM client (uses mock if None)
        min_dimensions: Minimum actionable dimensions required

    Returns:
        ActionabilityEvaluation
    """
    if use_llm:
        return evaluate_actionability_with_llm(
            segment,
            robustness=robustness,
            llm_client=llm_client,
        )
    else:
        return evaluate_actionability_rule_based(
            segment,
            robustness=robustness,
            min_dimensions=min_dimensions,
        )


# =============================================================================
# ACTIONABILITY FILTER CLASS
# =============================================================================


class ActionabilityFilter:
    """
    Filter segments based on actionability criteria.

    Evaluates segments and filters out those that are not commercially actionable.
    """

    def __init__(
        self,
        *,
        use_llm: bool = False,
        llm_client: LLMClient | None = None,
        min_dimensions: int = 1,
        min_confidence: ConfidenceLevel = ConfidenceLevel.LOW,
    ) -> None:
        """
        Initialize ActionabilityFilter.

        Args:
            use_llm: Whether to use LLM for evaluation
            llm_client: LLM client instance
            min_dimensions: Minimum actionable dimensions required
            min_confidence: Minimum confidence level required
        """
        self.use_llm = use_llm
        self.llm_client = llm_client
        self.min_dimensions = min_dimensions
        self.min_confidence = min_confidence

        self._evaluations: dict[str, ActionabilityEvaluation] = {}

    def evaluate(
        self,
        segment: Segment,
        *,
        robustness: RobustnessScore | None = None,
    ) -> ActionabilityEvaluation:
        """
        Evaluate a single segment.

        Args:
            segment: Segment to evaluate
            robustness: Optional robustness score

        Returns:
            ActionabilityEvaluation
        """
        evaluation = evaluate_actionability(
            segment,
            robustness=robustness,
            use_llm=self.use_llm,
            llm_client=self.llm_client,
            min_dimensions=self.min_dimensions,
        )

        self._evaluations[segment.segment_id] = evaluation
        return evaluation

    def evaluate_batch(
        self,
        segments: list[Segment],
        *,
        robustness_scores: dict[str, RobustnessScore] | None = None,
    ) -> dict[str, ActionabilityEvaluation]:
        """
        Evaluate multiple segments.

        Args:
            segments: Segments to evaluate
            robustness_scores: Optional mapping of segment_id to RobustnessScore

        Returns:
            Dictionary mapping segment_id to ActionabilityEvaluation
        """
        robustness_scores = robustness_scores or {}
        results: dict[str, ActionabilityEvaluation] = {}

        for segment in segments:
            robustness = robustness_scores.get(segment.segment_id)
            results[segment.segment_id] = self.evaluate(segment, robustness=robustness)

        return results

    def filter_actionable(
        self,
        segments: list[Segment],
        *,
        robustness_scores: dict[str, RobustnessScore] | None = None,
    ) -> list[Segment]:
        """
        Filter to only actionable segments.

        Args:
            segments: Segments to filter
            robustness_scores: Optional robustness scores

        Returns:
            List of actionable segments
        """
        evaluations = self.evaluate_batch(segments, robustness_scores=robustness_scores)

        actionable = []
        for segment in segments:
            evaluation = evaluations.get(segment.segment_id)
            if evaluation and self._passes_filter(evaluation):
                actionable.append(segment)

        return actionable

    def _passes_filter(self, evaluation: ActionabilityEvaluation) -> bool:
        """Check if evaluation passes filter criteria."""
        if not evaluation.is_actionable:
            return False

        # Check minimum dimensions
        if len(evaluation.actionability_dimensions) < self.min_dimensions:
            return False

        # Check minimum confidence
        confidence_order = {
            ConfidenceLevel.LOW: 0,
            ConfidenceLevel.MEDIUM: 1,
            ConfidenceLevel.HIGH: 2,
        }

        if confidence_order[evaluation.confidence_level] < confidence_order[self.min_confidence]:
            return False

        return True

    def get_summary(self) -> dict[str, Any]:
        """
        Get summary of evaluations.

        Returns:
            Summary statistics
        """
        if not self._evaluations:
            return {
                "total": 0,
                "actionable": 0,
                "not_actionable": 0,
                "actionability_rate": 0.0,
                "dimensions_distribution": {},
            }

        total = len(self._evaluations)
        actionable = sum(1 for e in self._evaluations.values() if e.is_actionable)

        # Count dimension occurrences
        dimension_counts: dict[str, int] = {}
        for evaluation in self._evaluations.values():
            for dim in evaluation.actionability_dimensions:
                dimension_counts[dim.value] = dimension_counts.get(dim.value, 0) + 1

        return {
            "total": total,
            "actionable": actionable,
            "not_actionable": total - actionable,
            "actionability_rate": actionable / total if total > 0 else 0.0,
            "dimensions_distribution": dimension_counts,
        }

    @property
    def evaluations(self) -> dict[str, ActionabilityEvaluation]:
        """Get all evaluations."""
        return self._evaluations.copy()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def is_segment_actionable(
    segment: Segment,
    *,
    robustness: RobustnessScore | None = None,
    min_dimensions: int = 1,
) -> bool:
    """
    Quick check if segment is actionable.

    Args:
        segment: Segment to check
        robustness: Optional robustness score
        min_dimensions: Minimum dimensions required

    Returns:
        True if actionable, False otherwise
    """
    evaluation = evaluate_actionability(
        segment,
        robustness=robustness,
        min_dimensions=min_dimensions,
    )
    return evaluation.is_actionable


def get_actionable_dimensions(segment: Segment) -> list[ActionabilityDimension]:
    """
    Get list of actionable dimensions for a segment.

    Args:
        segment: Segment to analyze

    Returns:
        List of actionable dimensions
    """
    evaluation = evaluate_actionability(segment)
    return evaluation.actionability_dimensions
