"""
Module: segment_validator

Purpose: Economic viability and robustness validation for segments.

Key Functions:
- validate_segment: Check segment against viability criteria
- calculate_roi_estimate: Estimate ROI for segment targeting
- assess_confidence_level: Determine confidence based on robustness

Architecture Notes:
- Applies rejection criteria from discovery document
- Integrates robustness scores from sensitivity analysis
- Raises SegmentRejectedError for invalid segments
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

from src.data.schemas import (
    ActionabilityDimension,
    ConfidenceLevel,
    CustomerProfile,
    RobustnessScore,
    RobustnessTier,
    Segment,
    SegmentViability,
    StrategicGoal,
)
from src.exceptions import SegmentRejectedError


# =============================================================================
# VALIDATION CRITERIA
# =============================================================================


@dataclass
class ValidationCriteria:
    """Configurable criteria for segment validation."""

    # Size thresholds
    min_segment_size: int = 10
    max_segment_size_pct: float = 0.5  # Max 50% of customer base

    # Value thresholds
    min_total_clv: Decimal = Decimal("1000")
    min_avg_clv: Decimal = Decimal("50")

    # Robustness thresholds
    min_feature_stability: float = 0.3
    min_time_consistency: float = 0.3
    min_overall_robustness: float = 0.4

    # Economic thresholds
    min_expected_roi: float = 0.5  # 50% minimum ROI
    max_cost_to_exploit_ratio: float = 0.3  # Max 30% of CLV

    # Actionability requirements
    require_actionable_dimension: bool = True
    min_actionability_score: float = 0.3


@dataclass
class ValidationResult:
    """Result of segment validation."""

    is_valid: bool
    rejection_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


# =============================================================================
# ROI CALCULATION
# =============================================================================


def calculate_roi_estimate(
    segment: Segment,
    *,
    campaign_cost_per_customer: Decimal = Decimal("5.00"),
    expected_conversion_lift: float = 0.1,
    margin_rate: float = 0.3,
) -> dict[str, Any]:
    """
    Calculate estimated ROI for targeting a segment.

    ROI = (Expected Additional Revenue - Campaign Cost) / Campaign Cost

    Args:
        segment: Segment to analyze
        campaign_cost_per_customer: Cost per customer in campaign
        expected_conversion_lift: Expected % improvement in conversion
        margin_rate: Profit margin rate

    Returns:
        Dictionary with ROI metrics
    """
    if segment.size == 0:
        return {
            "expected_roi": 0.0,
            "total_campaign_cost": Decimal("0"),
            "expected_additional_revenue": Decimal("0"),
            "expected_profit": Decimal("0"),
            "cost_to_clv_ratio": 0.0,
        }

    total_campaign_cost = campaign_cost_per_customer * segment.size

    # Expected additional revenue from targeting
    # Based on CLV uplift from conversion improvement
    expected_additional_revenue = segment.total_clv * Decimal(str(expected_conversion_lift))

    # Calculate profit considering margin
    expected_profit = expected_additional_revenue * Decimal(str(margin_rate)) - total_campaign_cost

    # ROI calculation
    expected_roi = 0.0
    if total_campaign_cost > 0:
        expected_roi = float(expected_profit / total_campaign_cost)

    # Cost to CLV ratio
    cost_to_clv_ratio = 0.0
    if segment.total_clv > 0:
        cost_to_clv_ratio = float(total_campaign_cost / segment.total_clv)

    return {
        "expected_roi": expected_roi,
        "total_campaign_cost": total_campaign_cost,
        "expected_additional_revenue": expected_additional_revenue,
        "expected_profit": expected_profit,
        "cost_to_clv_ratio": cost_to_clv_ratio,
    }


def estimate_cost_to_exploit(
    segment: Segment,
    *,
    base_cost_per_customer: Decimal = Decimal("5.00"),
    complexity_factor: float = 1.0,
) -> Decimal:
    """
    Estimate total cost to exploit/target a segment.

    Args:
        segment: Segment to analyze
        base_cost_per_customer: Base cost per customer
        complexity_factor: Multiplier for segment complexity

    Returns:
        Estimated total cost
    """
    base_cost = base_cost_per_customer * segment.size
    return base_cost * Decimal(str(complexity_factor))


# =============================================================================
# ACTIONABILITY SCORING
# =============================================================================


def calculate_marketing_targetability(
    segment: Segment,
    profiles: list[CustomerProfile] | None = None,
) -> float:
    """
    Calculate how targetable a segment is for marketing.

    Based on:
    - Segment homogeneity (based on centroid distance)
    - Size appropriateness
    - Presence of defining traits

    Args:
        segment: Segment to score
        profiles: Optional list of profiles for deeper analysis

    Returns:
        Score from 0 to 1
    """
    score = 0.5  # Base score

    # Size factor: prefer medium-sized segments
    if 20 <= segment.size <= 500:
        score += 0.2
    elif segment.size < 20 or segment.size > 1000:
        score -= 0.1

    # Has defining traits
    if segment.defining_traits:
        score += 0.2

    # Has actionability dimensions
    if segment.actionability_dimensions:
        score += 0.1

    return max(0.0, min(1.0, score))


def calculate_sales_prioritization(
    segment: Segment,
) -> float:
    """
    Calculate how suitable a segment is for sales prioritization.

    Based on:
    - Average CLV
    - Segment value

    Args:
        segment: Segment to score

    Returns:
        Score from 0 to 1
    """
    if segment.size == 0:
        return 0.0

    score = 0.0

    # High CLV is good for sales prioritization
    avg_clv = float(segment.avg_clv)
    if avg_clv > 500:
        score += 0.4
    elif avg_clv > 200:
        score += 0.3
    elif avg_clv > 100:
        score += 0.2
    else:
        score += 0.1

    # Total value matters for ROI
    total_clv = float(segment.total_clv)
    if total_clv > 50000:
        score += 0.3
    elif total_clv > 10000:
        score += 0.2
    else:
        score += 0.1

    # Manageable size for sales
    if 10 <= segment.size <= 100:
        score += 0.2
    elif segment.size <= 200:
        score += 0.1

    return max(0.0, min(1.0, score))


def calculate_personalization_opportunity(
    segment: Segment,
) -> float:
    """
    Calculate personalization opportunity score.

    Based on:
    - Presence of WHAT and HOW dimensions
    - Defining traits for personalization

    Args:
        segment: Segment to score

    Returns:
        Score from 0 to 1
    """
    score = 0.3  # Base score

    # Check for relevant dimensions
    has_what = ActionabilityDimension.WHAT in segment.actionability_dimensions
    has_how = ActionabilityDimension.HOW in segment.actionability_dimensions

    if has_what:
        score += 0.3
    if has_how:
        score += 0.2

    # Defining traits enable personalization
    if len(segment.defining_traits) >= 2:
        score += 0.2

    return max(0.0, min(1.0, score))


def calculate_timing_optimization(
    segment: Segment,
) -> float:
    """
    Calculate timing optimization opportunity score.

    Based on:
    - Presence of WHEN dimension
    - Temporal traits

    Args:
        segment: Segment to score

    Returns:
        Score from 0 to 1
    """
    score = 0.2  # Base score

    # Check for WHEN dimension
    if ActionabilityDimension.WHEN in segment.actionability_dimensions:
        score += 0.5

    # Look for temporal traits in defining_traits
    temporal_keywords = ["weekend", "morning", "evening", "night", "seasonal", "early", "late"]
    temporal_traits = [
        t for t in segment.defining_traits
        if any(kw in t.lower() for kw in temporal_keywords)
    ]

    if temporal_traits:
        score += 0.3

    return max(0.0, min(1.0, score))


# =============================================================================
# CONFIDENCE ASSESSMENT
# =============================================================================


def assess_confidence_level(
    robustness: RobustnessScore | None,
    segment: Segment,
) -> ConfidenceLevel:
    """
    Assess confidence level based on robustness and segment characteristics.

    Args:
        robustness: Optional robustness score from sensitivity analysis
        segment: Segment being assessed

    Returns:
        ConfidenceLevel (high, medium, low)
    """
    if robustness is None:
        # Without robustness data, base on segment size
        if segment.size >= 50:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    # Use robustness tier as primary indicator
    if robustness.robustness_tier == RobustnessTier.HIGH:
        return ConfidenceLevel.HIGH
    elif robustness.robustness_tier == RobustnessTier.MEDIUM:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW


def determine_strategic_impact(
    segment: Segment,
    metric: str,
) -> Literal["high", "medium", "low"]:
    """
    Determine strategic impact level for a specific metric.

    Args:
        segment: Segment to assess
        metric: One of "revenue", "retention", "satisfaction"

    Returns:
        Impact level
    """
    if metric == "revenue":
        total_clv = float(segment.total_clv)
        if total_clv > 100000:
            return "high"
        elif total_clv > 25000:
            return "medium"
        return "low"

    elif metric == "retention":
        # Check for churn-related goals
        has_churn_goal = StrategicGoal.REDUCE_CHURN in segment.strategic_goals
        if has_churn_goal and segment.size > 20:
            return "high"
        elif has_churn_goal:
            return "medium"
        return "low"

    elif metric == "satisfaction":
        # Check for satisfaction goal
        has_satisfaction_goal = StrategicGoal.INCREASE_SATISFACTION in segment.strategic_goals
        if has_satisfaction_goal:
            return "medium"  # Hard to achieve high certainty
        return "low"

    return "low"


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_segment_size(
    segment: Segment,
    total_customers: int,
    criteria: ValidationCriteria,
) -> ValidationResult:
    """
    Validate segment size against criteria.

    Args:
        segment: Segment to validate
        total_customers: Total customer count for percentage calculation
        criteria: Validation criteria

    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True)

    # Check minimum size
    if segment.size < criteria.min_segment_size:
        result.is_valid = False
        result.rejection_reasons.append(
            f"Segment too small: {segment.size} < {criteria.min_segment_size}"
        )

    # Check maximum size percentage
    if total_customers > 0:
        pct = segment.size / total_customers
        if pct > criteria.max_segment_size_pct:
            result.is_valid = False
            result.rejection_reasons.append(
                f"Segment too large: {pct:.1%} > {criteria.max_segment_size_pct:.1%}"
            )

    result.scores["size_valid"] = 1.0 if result.is_valid else 0.0
    return result


def validate_segment_value(
    segment: Segment,
    criteria: ValidationCriteria,
) -> ValidationResult:
    """
    Validate segment economic value.

    Args:
        segment: Segment to validate
        criteria: Validation criteria

    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True)

    # Check minimum total CLV
    if segment.total_clv < criteria.min_total_clv:
        result.is_valid = False
        result.rejection_reasons.append(
            f"Total CLV too low: ${segment.total_clv:.2f} < ${criteria.min_total_clv:.2f}"
        )

    # Check minimum average CLV
    if segment.avg_clv < criteria.min_avg_clv:
        result.warnings.append(
            f"Low average CLV: ${segment.avg_clv:.2f} < ${criteria.min_avg_clv:.2f}"
        )

    result.scores["value_valid"] = 1.0 if result.is_valid else 0.0
    return result


def validate_segment_robustness(
    robustness: RobustnessScore | None,
    criteria: ValidationCriteria,
) -> ValidationResult:
    """
    Validate segment robustness.

    Args:
        robustness: Robustness score from sensitivity analysis
        criteria: Validation criteria

    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True)

    if robustness is None:
        result.warnings.append("No robustness data available")
        result.scores["robustness_valid"] = 0.5  # Uncertain
        return result

    # Check feature stability
    if robustness.feature_stability < criteria.min_feature_stability:
        result.warnings.append(
            f"Low feature stability: {robustness.feature_stability:.2f} < {criteria.min_feature_stability}"
        )

    # Check time consistency
    if robustness.time_window_consistency < criteria.min_time_consistency:
        result.warnings.append(
            f"Low time consistency: {robustness.time_window_consistency:.2f} < {criteria.min_time_consistency}"
        )

    # Check overall robustness
    if robustness.overall_robustness < criteria.min_overall_robustness:
        result.is_valid = False
        result.rejection_reasons.append(
            f"Overall robustness too low: {robustness.overall_robustness:.2f} < {criteria.min_overall_robustness}"
        )

    result.scores["robustness_valid"] = robustness.overall_robustness
    return result


def validate_segment_actionability(
    segment: Segment,
    criteria: ValidationCriteria,
) -> ValidationResult:
    """
    Validate segment actionability.

    Args:
        segment: Segment to validate
        criteria: Validation criteria

    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True)

    # Check for actionability dimensions
    if criteria.require_actionable_dimension and not segment.actionability_dimensions:
        result.is_valid = False
        result.rejection_reasons.append(
            "No actionability dimensions defined"
        )

    # Check for defining traits
    if not segment.defining_traits:
        result.warnings.append("No defining traits specified")

    # Calculate actionability scores
    marketing = calculate_marketing_targetability(segment)
    sales = calculate_sales_prioritization(segment)
    personalization = calculate_personalization_opportunity(segment)
    timing = calculate_timing_optimization(segment)

    avg_actionability = (marketing + sales + personalization + timing) / 4

    if avg_actionability < criteria.min_actionability_score:
        result.warnings.append(
            f"Low actionability score: {avg_actionability:.2f} < {criteria.min_actionability_score}"
        )

    result.scores["marketing_targetability"] = marketing
    result.scores["sales_prioritization"] = sales
    result.scores["personalization_opportunity"] = personalization
    result.scores["timing_optimization"] = timing
    result.scores["avg_actionability"] = avg_actionability

    return result


def validate_segment_economics(
    segment: Segment,
    criteria: ValidationCriteria,
    *,
    campaign_cost_per_customer: Decimal = Decimal("5.00"),
) -> ValidationResult:
    """
    Validate segment economic viability.

    Args:
        segment: Segment to validate
        criteria: Validation criteria
        campaign_cost_per_customer: Cost per customer for ROI calculation

    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True)

    roi_metrics = calculate_roi_estimate(
        segment,
        campaign_cost_per_customer=campaign_cost_per_customer,
    )

    # Check ROI
    if roi_metrics["expected_roi"] < criteria.min_expected_roi:
        result.is_valid = False
        result.rejection_reasons.append(
            f"Expected ROI too low: {roi_metrics['expected_roi']:.1%} < {criteria.min_expected_roi:.1%}"
        )

    # Check cost to CLV ratio
    if roi_metrics["cost_to_clv_ratio"] > criteria.max_cost_to_exploit_ratio:
        result.warnings.append(
            f"High cost ratio: {roi_metrics['cost_to_clv_ratio']:.1%} > {criteria.max_cost_to_exploit_ratio:.1%}"
        )

    result.scores["expected_roi"] = roi_metrics["expected_roi"]
    result.scores["cost_to_clv_ratio"] = roi_metrics["cost_to_clv_ratio"]

    return result


def validate_segment(
    segment: Segment,
    *,
    total_customers: int,
    robustness: RobustnessScore | None = None,
    criteria: ValidationCriteria | None = None,
    raise_on_invalid: bool = False,
) -> ValidationResult:
    """
    Run complete validation on a segment.

    Args:
        segment: Segment to validate
        total_customers: Total customer count
        robustness: Optional robustness score
        criteria: Validation criteria (uses defaults if None)
        raise_on_invalid: If True, raise SegmentRejectedError on failure

    Returns:
        ValidationResult with all validation outcomes

    Raises:
        SegmentRejectedError: If raise_on_invalid=True and segment fails
    """
    if criteria is None:
        criteria = ValidationCriteria()

    # Run all validations
    size_result = validate_segment_size(segment, total_customers, criteria)
    value_result = validate_segment_value(segment, criteria)
    robustness_result = validate_segment_robustness(robustness, criteria)
    actionability_result = validate_segment_actionability(segment, criteria)
    economics_result = validate_segment_economics(segment, criteria)

    # Aggregate results
    all_reasons: list[str] = []
    all_warnings: list[str] = []
    all_scores: dict[str, float] = {}

    for result in [size_result, value_result, robustness_result, actionability_result, economics_result]:
        all_reasons.extend(result.rejection_reasons)
        all_warnings.extend(result.warnings)
        all_scores.update(result.scores)

    is_valid = (
        size_result.is_valid
        and value_result.is_valid
        and robustness_result.is_valid
        and actionability_result.is_valid
        and economics_result.is_valid
    )

    final_result = ValidationResult(
        is_valid=is_valid,
        rejection_reasons=all_reasons,
        warnings=all_warnings,
        scores=all_scores,
    )

    if raise_on_invalid and not is_valid:
        raise SegmentRejectedError(
            f"Segment {segment.segment_id} failed validation",
            segment_id=segment.segment_id,
            rejection_reasons=all_reasons,
        )

    return final_result


# =============================================================================
# VIABILITY BUILDER
# =============================================================================


def build_segment_viability(
    segment: Segment,
    *,
    robustness: RobustnessScore | None = None,
    validation_result: ValidationResult | None = None,
    recommended_action: str = "",
    business_hypothesis: str = "",
) -> SegmentViability:
    """
    Build a complete SegmentViability assessment.

    Args:
        segment: Segment to assess
        robustness: Robustness score
        validation_result: Pre-computed validation result
        recommended_action: Recommended action for segment
        business_hypothesis: Business hypothesis for segment

    Returns:
        SegmentViability with all assessments
    """
    # Run validation if not provided
    if validation_result is None:
        validation_result = validate_segment(
            segment,
            total_customers=segment.size * 10,  # Estimate
            robustness=robustness,
        )

    # Calculate scores
    marketing = validation_result.scores.get(
        "marketing_targetability",
        calculate_marketing_targetability(segment),
    )
    sales = validation_result.scores.get(
        "sales_prioritization",
        calculate_sales_prioritization(segment),
    )
    personalization = validation_result.scores.get(
        "personalization_opportunity",
        calculate_personalization_opportunity(segment),
    )
    timing = validation_result.scores.get(
        "timing_optimization",
        calculate_timing_optimization(segment),
    )

    # Calculate ROI
    roi_metrics = calculate_roi_estimate(segment)
    expected_roi = roi_metrics["expected_roi"]

    # Estimate cost
    cost_to_exploit = estimate_cost_to_exploit(segment)

    # Determine impacts
    revenue_impact = determine_strategic_impact(segment, "revenue")
    retention_impact = determine_strategic_impact(segment, "retention")
    satisfaction_impact = determine_strategic_impact(segment, "satisfaction")

    # Get or create robustness score
    if robustness is None:
        robustness = RobustnessScore.calculate(
            segment_id=segment.segment_id,
            feature_stability=0.5,
            time_window_consistency=0.5,
        )

    # Assess confidence
    confidence = assess_confidence_level(robustness, segment)

    # Generate default action and hypothesis if not provided
    if not recommended_action:
        recommended_action = _generate_recommended_action(segment)

    if not business_hypothesis:
        business_hypothesis = _generate_business_hypothesis(segment)

    return SegmentViability(
        segment_id=segment.segment_id,
        size=segment.size,
        total_clv=segment.total_clv,
        marketing_targetability=marketing,
        sales_prioritization=sales,
        personalization_opportunity=personalization,
        timing_optimization=timing,
        cost_to_exploit=cost_to_exploit,
        expected_roi=expected_roi,
        revenue_impact=revenue_impact,
        retention_impact=retention_impact,
        satisfaction_impact=satisfaction_impact,
        robustness_score=robustness,
        confidence_level=confidence,
        recommended_action=recommended_action,
        business_hypothesis=business_hypothesis,
        is_approved=validation_result.is_valid,
        rejection_reasons=validation_result.rejection_reasons,
    )


def _generate_recommended_action(segment: Segment) -> str:
    """Generate a recommended action based on segment characteristics."""
    actions = []

    if ActionabilityDimension.WHAT in segment.actionability_dimensions:
        actions.append("personalized product recommendations")
    if ActionabilityDimension.WHEN in segment.actionability_dimensions:
        actions.append("optimized send-time campaigns")
    if ActionabilityDimension.HOW in segment.actionability_dimensions:
        actions.append("channel-specific messaging")
    if ActionabilityDimension.WHO in segment.actionability_dimensions:
        actions.append("priority outreach")

    if actions:
        return f"Target with {', '.join(actions)}"
    return "Review segment for targeting opportunities"


def _generate_business_hypothesis(segment: Segment) -> str:
    """Generate a business hypothesis based on segment characteristics."""
    if StrategicGoal.REDUCE_CHURN in segment.strategic_goals:
        return f"Targeted intervention for {segment.size} customers could reduce churn and retain CLV"
    if StrategicGoal.INCREASE_REVENUE in segment.strategic_goals:
        return f"Personalized campaigns for {segment.size} customers could increase revenue per customer"
    if StrategicGoal.INCREASE_SATISFACTION in segment.strategic_goals:
        return f"Improved experience for {segment.size} customers could boost satisfaction scores"

    return f"Targeting {segment.size} customers with relevant offers could improve business metrics"


# =============================================================================
# SEGMENT VALIDATOR CLASS
# =============================================================================


class SegmentValidator:
    """
    High-level validator for segment assessment.

    Coordinates validation, scoring, and viability assessment.
    """

    def __init__(
        self,
        *,
        criteria: ValidationCriteria | None = None,
        total_customers: int = 0,
    ) -> None:
        """
        Initialize SegmentValidator.

        Args:
            criteria: Validation criteria (uses defaults if None)
            total_customers: Total customer count for size validation
        """
        self.criteria = criteria or ValidationCriteria()
        self.total_customers = total_customers
        self._validated_segments: dict[str, ValidationResult] = {}

    def validate(
        self,
        segment: Segment,
        *,
        robustness: RobustnessScore | None = None,
        raise_on_invalid: bool = False,
    ) -> ValidationResult:
        """
        Validate a single segment.

        Args:
            segment: Segment to validate
            robustness: Optional robustness score
            raise_on_invalid: Raise exception on failure

        Returns:
            ValidationResult
        """
        result = validate_segment(
            segment,
            total_customers=self.total_customers,
            robustness=robustness,
            criteria=self.criteria,
            raise_on_invalid=raise_on_invalid,
        )

        self._validated_segments[segment.segment_id] = result
        return result

    def validate_batch(
        self,
        segments: list[Segment],
        *,
        robustness_scores: dict[str, RobustnessScore] | None = None,
    ) -> dict[str, ValidationResult]:
        """
        Validate multiple segments.

        Args:
            segments: List of segments to validate
            robustness_scores: Optional mapping of segment_id to RobustnessScore

        Returns:
            Dictionary mapping segment_id to ValidationResult
        """
        robustness_scores = robustness_scores or {}
        results: dict[str, ValidationResult] = {}

        for segment in segments:
            robustness = robustness_scores.get(segment.segment_id)
            results[segment.segment_id] = self.validate(segment, robustness=robustness)

        return results

    def build_viability(
        self,
        segment: Segment,
        *,
        robustness: RobustnessScore | None = None,
    ) -> SegmentViability:
        """
        Build viability assessment for a segment.

        Args:
            segment: Segment to assess
            robustness: Optional robustness score

        Returns:
            SegmentViability
        """
        # Get or compute validation result
        validation_result = self._validated_segments.get(segment.segment_id)
        if validation_result is None:
            validation_result = self.validate(segment, robustness=robustness)

        return build_segment_viability(
            segment,
            robustness=robustness,
            validation_result=validation_result,
        )

    def filter_valid_segments(
        self,
        segments: list[Segment],
        *,
        robustness_scores: dict[str, RobustnessScore] | None = None,
    ) -> list[Segment]:
        """
        Filter to only valid segments.

        Args:
            segments: Segments to filter
            robustness_scores: Optional robustness scores

        Returns:
            List of valid segments
        """
        results = self.validate_batch(segments, robustness_scores=robustness_scores)
        return [seg for seg in segments if results[seg.segment_id].is_valid]

    def get_validation_summary(self) -> dict[str, Any]:
        """
        Get summary of all validated segments.

        Returns:
            Summary statistics
        """
        if not self._validated_segments:
            return {"total": 0, "valid": 0, "invalid": 0, "validation_rate": 0.0}

        total = len(self._validated_segments)
        valid = sum(1 for r in self._validated_segments.values() if r.is_valid)

        return {
            "total": total,
            "valid": valid,
            "invalid": total - valid,
            "validation_rate": valid / total if total > 0 else 0.0,
        }

    @property
    def validated_segments(self) -> dict[str, ValidationResult]:
        """Get all validated segments."""
        return self._validated_segments.copy()
