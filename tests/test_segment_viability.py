"""
Tests for segment validator and viability assessment.
"""

from decimal import Decimal

import pytest

from src.data.schemas import (
    ActionabilityDimension,
    ConfidenceLevel,
    RobustnessScore,
    RobustnessTier,
    Segment,
    SegmentMember,
    SegmentViability,
    StrategicGoal,
)
from src.exceptions import SegmentRejectedError
from src.segmentation.segment_validator import (
    SegmentValidator,
    ValidationCriteria,
    ValidationResult,
    assess_confidence_level,
    build_segment_viability,
    calculate_marketing_targetability,
    calculate_personalization_opportunity,
    calculate_roi_estimate,
    calculate_sales_prioritization,
    calculate_timing_optimization,
    determine_strategic_impact,
    estimate_cost_to_exploit,
    validate_segment,
    validate_segment_actionability,
    validate_segment_economics,
    validate_segment_robustness,
    validate_segment_size,
    validate_segment_value,
)


# =============================================================================
# FIXTURES
# =============================================================================


def make_segment(
    segment_id: str = "test_segment",
    *,
    size: int = 50,
    total_clv: Decimal = Decimal("10000"),
    avg_clv: Decimal | None = None,
    actionability_dimensions: list[ActionabilityDimension] | None = None,
    defining_traits: list[str] | None = None,
    strategic_goals: list[StrategicGoal] | None = None,
) -> Segment:
    """Create a test segment."""
    if avg_clv is None:
        avg_clv = total_clv / size if size > 0 else Decimal("0")

    return Segment(
        segment_id=segment_id,
        name=f"Test Segment {segment_id}",
        description="Test segment for validation",
        members=[SegmentMember(internal_customer_id=f"cust_{i}") for i in range(size)],
        size=size,
        total_clv=total_clv,
        avg_clv=avg_clv,
        avg_order_value=Decimal("50"),
        actionability_dimensions=actionability_dimensions or [],
        defining_traits=defining_traits or [],
        strategic_goals=strategic_goals or [],
    )


def make_robustness_score(
    segment_id: str = "test_segment",
    *,
    feature_stability: float = 0.7,
    time_consistency: float = 0.7,
) -> RobustnessScore:
    """Create a test robustness score."""
    return RobustnessScore.calculate(
        segment_id=segment_id,
        feature_stability=feature_stability,
        time_window_consistency=time_consistency,
    )


# =============================================================================
# ROI CALCULATION TESTS
# =============================================================================


class TestROICalculation:
    """Tests for ROI calculation functions."""

    def test_calculate_roi_estimate_basic(self) -> None:
        """Test basic ROI calculation."""
        segment = make_segment(size=100, total_clv=Decimal("50000"))

        roi = calculate_roi_estimate(segment)

        assert "expected_roi" in roi
        assert "total_campaign_cost" in roi
        assert "expected_additional_revenue" in roi
        assert "expected_profit" in roi
        assert "cost_to_clv_ratio" in roi

    def test_calculate_roi_estimate_empty_segment(self) -> None:
        """Test ROI calculation for empty segment."""
        segment = make_segment(size=0, total_clv=Decimal("0"))

        roi = calculate_roi_estimate(segment)

        assert roi["expected_roi"] == 0.0
        assert roi["total_campaign_cost"] == Decimal("0")

    def test_calculate_roi_estimate_high_value_segment(self) -> None:
        """Test ROI for high-value segment."""
        segment = make_segment(size=50, total_clv=Decimal("100000"))

        roi = calculate_roi_estimate(
            segment,
            campaign_cost_per_customer=Decimal("5.00"),
            expected_conversion_lift=0.1,
            margin_rate=0.3,
        )

        # Expected: 100000 * 0.1 = 10000 additional revenue
        # Profit: 10000 * 0.3 - 50 * 5 = 3000 - 250 = 2750
        # ROI: 2750 / 250 = 11.0
        assert roi["expected_roi"] > 1.0  # Positive ROI

    def test_estimate_cost_to_exploit(self) -> None:
        """Test cost estimation."""
        segment = make_segment(size=100)

        cost = estimate_cost_to_exploit(
            segment,
            base_cost_per_customer=Decimal("10.00"),
            complexity_factor=1.5,
        )

        assert cost == Decimal("1500.00")


# =============================================================================
# ACTIONABILITY SCORING TESTS
# =============================================================================


class TestActionabilityScoring:
    """Tests for actionability scoring functions."""

    def test_marketing_targetability_medium_size(self) -> None:
        """Test marketing targetability for medium-sized segment."""
        segment = make_segment(
            size=100,
            defining_traits=["High Value", "Engaged"],
            actionability_dimensions=[ActionabilityDimension.WHO],
        )

        score = calculate_marketing_targetability(segment)

        assert 0.5 <= score <= 1.0  # Should be above base

    def test_marketing_targetability_small_segment(self) -> None:
        """Test marketing targetability for small segment."""
        segment = make_segment(size=5)

        score = calculate_marketing_targetability(segment)

        assert score < 0.6  # Penalized for small size

    def test_sales_prioritization_high_value(self) -> None:
        """Test sales prioritization for high-value segment."""
        segment = make_segment(
            size=50,
            total_clv=Decimal("60000"),
            avg_clv=Decimal("1200"),
        )

        score = calculate_sales_prioritization(segment)

        assert score >= 0.7  # High CLV should score well

    def test_sales_prioritization_empty_segment(self) -> None:
        """Test sales prioritization for empty segment."""
        segment = make_segment(size=0, total_clv=Decimal("0"))

        score = calculate_sales_prioritization(segment)

        assert score == 0.0

    def test_personalization_with_dimensions(self) -> None:
        """Test personalization score with relevant dimensions."""
        segment = make_segment(
            actionability_dimensions=[ActionabilityDimension.WHAT, ActionabilityDimension.HOW],
            defining_traits=["Category Specialist", "Mobile First"],
        )

        score = calculate_personalization_opportunity(segment)

        assert score >= 0.8  # Should score high

    def test_timing_optimization_with_when(self) -> None:
        """Test timing optimization with WHEN dimension."""
        segment = make_segment(
            actionability_dimensions=[ActionabilityDimension.WHEN],
            defining_traits=["Weekend Shopper"],
        )

        score = calculate_timing_optimization(segment)

        assert score >= 0.7  # WHEN dimension should boost score


# =============================================================================
# CONFIDENCE ASSESSMENT TESTS
# =============================================================================


class TestConfidenceAssessment:
    """Tests for confidence level assessment."""

    def test_assess_confidence_high_robustness(self) -> None:
        """Test confidence with high robustness."""
        segment = make_segment()
        robustness = make_robustness_score(feature_stability=0.9, time_consistency=0.8)

        confidence = assess_confidence_level(robustness, segment)

        assert confidence == ConfidenceLevel.HIGH

    def test_assess_confidence_medium_robustness(self) -> None:
        """Test confidence with medium robustness."""
        segment = make_segment()
        robustness = make_robustness_score(feature_stability=0.6, time_consistency=0.6)

        confidence = assess_confidence_level(robustness, segment)

        assert confidence == ConfidenceLevel.MEDIUM

    def test_assess_confidence_low_robustness(self) -> None:
        """Test confidence with low robustness."""
        segment = make_segment()
        robustness = make_robustness_score(feature_stability=0.2, time_consistency=0.2)

        confidence = assess_confidence_level(robustness, segment)

        assert confidence == ConfidenceLevel.LOW

    def test_assess_confidence_no_robustness_large_segment(self) -> None:
        """Test confidence without robustness data for large segment."""
        segment = make_segment(size=100)

        confidence = assess_confidence_level(None, segment)

        assert confidence == ConfidenceLevel.MEDIUM

    def test_assess_confidence_no_robustness_small_segment(self) -> None:
        """Test confidence without robustness data for small segment."""
        segment = make_segment(size=10)

        confidence = assess_confidence_level(None, segment)

        assert confidence == ConfidenceLevel.LOW


# =============================================================================
# STRATEGIC IMPACT TESTS
# =============================================================================


class TestStrategicImpact:
    """Tests for strategic impact determination."""

    def test_revenue_impact_high(self) -> None:
        """Test high revenue impact."""
        segment = make_segment(total_clv=Decimal("150000"))

        impact = determine_strategic_impact(segment, "revenue")

        assert impact == "high"

    def test_revenue_impact_medium(self) -> None:
        """Test medium revenue impact."""
        segment = make_segment(total_clv=Decimal("50000"))

        impact = determine_strategic_impact(segment, "revenue")

        assert impact == "medium"

    def test_revenue_impact_low(self) -> None:
        """Test low revenue impact."""
        segment = make_segment(total_clv=Decimal("5000"))

        impact = determine_strategic_impact(segment, "revenue")

        assert impact == "low"

    def test_retention_impact_with_churn_goal(self) -> None:
        """Test retention impact with churn reduction goal."""
        segment = make_segment(
            size=50,
            strategic_goals=[StrategicGoal.REDUCE_CHURN],
        )

        impact = determine_strategic_impact(segment, "retention")

        assert impact == "high"

    def test_satisfaction_impact_with_goal(self) -> None:
        """Test satisfaction impact with goal."""
        segment = make_segment(
            strategic_goals=[StrategicGoal.INCREASE_SATISFACTION],
        )

        impact = determine_strategic_impact(segment, "satisfaction")

        assert impact == "medium"


# =============================================================================
# VALIDATION FUNCTION TESTS
# =============================================================================


class TestValidateSizeFunction:
    """Tests for validate_segment_size function."""

    def test_valid_size(self) -> None:
        """Test segment with valid size."""
        segment = make_segment(size=50)
        criteria = ValidationCriteria(min_segment_size=10, max_segment_size_pct=0.5)

        result = validate_segment_size(segment, total_customers=200, criteria=criteria)

        assert result.is_valid
        assert not result.rejection_reasons

    def test_too_small(self) -> None:
        """Test segment that is too small."""
        segment = make_segment(size=5)
        criteria = ValidationCriteria(min_segment_size=10)

        result = validate_segment_size(segment, total_customers=100, criteria=criteria)

        assert not result.is_valid
        assert any("too small" in r for r in result.rejection_reasons)

    def test_too_large_percentage(self) -> None:
        """Test segment that is too large a percentage."""
        segment = make_segment(size=80)
        criteria = ValidationCriteria(max_segment_size_pct=0.5)

        result = validate_segment_size(segment, total_customers=100, criteria=criteria)

        assert not result.is_valid
        assert any("too large" in r for r in result.rejection_reasons)


class TestValidateValueFunction:
    """Tests for validate_segment_value function."""

    def test_valid_value(self) -> None:
        """Test segment with valid value."""
        segment = make_segment(total_clv=Decimal("5000"), avg_clv=Decimal("100"))
        criteria = ValidationCriteria(min_total_clv=Decimal("1000"), min_avg_clv=Decimal("50"))

        result = validate_segment_value(segment, criteria)

        assert result.is_valid
        assert not result.rejection_reasons

    def test_low_total_clv(self) -> None:
        """Test segment with low total CLV."""
        segment = make_segment(total_clv=Decimal("500"))
        criteria = ValidationCriteria(min_total_clv=Decimal("1000"))

        result = validate_segment_value(segment, criteria)

        assert not result.is_valid
        assert any("CLV too low" in r for r in result.rejection_reasons)


class TestValidateRobustnessFunction:
    """Tests for validate_segment_robustness function."""

    def test_valid_robustness(self) -> None:
        """Test segment with valid robustness."""
        robustness = make_robustness_score(feature_stability=0.7, time_consistency=0.7)
        criteria = ValidationCriteria(min_overall_robustness=0.5)

        result = validate_segment_robustness(robustness, criteria)

        assert result.is_valid

    def test_low_robustness(self) -> None:
        """Test segment with low robustness."""
        robustness = make_robustness_score(feature_stability=0.2, time_consistency=0.2)
        criteria = ValidationCriteria(min_overall_robustness=0.5)

        result = validate_segment_robustness(robustness, criteria)

        assert not result.is_valid
        assert any("robustness too low" in r for r in result.rejection_reasons)

    def test_no_robustness_data(self) -> None:
        """Test validation without robustness data."""
        criteria = ValidationCriteria()

        result = validate_segment_robustness(None, criteria)

        assert result.is_valid  # Valid but with warning
        assert any("No robustness data" in w for w in result.warnings)


class TestValidateActionabilityFunction:
    """Tests for validate_segment_actionability function."""

    def test_with_actionability(self) -> None:
        """Test segment with actionability dimensions."""
        segment = make_segment(
            actionability_dimensions=[ActionabilityDimension.WHAT],
            defining_traits=["Category Specialist"],
        )
        criteria = ValidationCriteria(require_actionable_dimension=True)

        result = validate_segment_actionability(segment, criteria)

        assert result.is_valid

    def test_missing_actionability(self) -> None:
        """Test segment missing actionability dimensions."""
        segment = make_segment(actionability_dimensions=[])
        criteria = ValidationCriteria(require_actionable_dimension=True)

        result = validate_segment_actionability(segment, criteria)

        assert not result.is_valid
        assert any("No actionability" in r for r in result.rejection_reasons)


class TestValidateEconomicsFunction:
    """Tests for validate_segment_economics function."""

    def test_valid_economics(self) -> None:
        """Test segment with valid economics."""
        segment = make_segment(size=50, total_clv=Decimal("50000"))
        criteria = ValidationCriteria(min_expected_roi=0.5)

        result = validate_segment_economics(segment, criteria)

        assert result.is_valid

    def test_low_roi(self) -> None:
        """Test segment with low expected ROI."""
        segment = make_segment(size=100, total_clv=Decimal("500"))
        criteria = ValidationCriteria(min_expected_roi=0.5)

        result = validate_segment_economics(segment, criteria)

        assert not result.is_valid
        assert any("ROI too low" in r for r in result.rejection_reasons)


# =============================================================================
# FULL VALIDATION TESTS
# =============================================================================


class TestValidateSegment:
    """Tests for complete segment validation."""

    def test_valid_segment(self) -> None:
        """Test completely valid segment."""
        segment = make_segment(
            size=50,
            total_clv=Decimal("50000"),
            actionability_dimensions=[ActionabilityDimension.WHAT],
        )
        robustness = make_robustness_score(feature_stability=0.8, time_consistency=0.8)

        result = validate_segment(
            segment,
            total_customers=500,
            robustness=robustness,
        )

        assert result.is_valid
        assert not result.rejection_reasons

    def test_invalid_segment_multiple_reasons(self) -> None:
        """Test segment that fails multiple criteria."""
        segment = make_segment(
            size=5,  # Too small
            total_clv=Decimal("100"),  # Too low CLV
            actionability_dimensions=[],  # No actionability
        )

        result = validate_segment(
            segment,
            total_customers=100,
        )

        assert not result.is_valid
        assert len(result.rejection_reasons) >= 2

    def test_raise_on_invalid(self) -> None:
        """Test exception raising for invalid segment."""
        segment = make_segment(size=3)  # Too small

        with pytest.raises(SegmentRejectedError) as exc_info:
            validate_segment(
                segment,
                total_customers=100,
                raise_on_invalid=True,
            )

        assert segment.segment_id == exc_info.value.segment_id


# =============================================================================
# VIABILITY BUILDER TESTS
# =============================================================================


class TestBuildSegmentViability:
    """Tests for viability building."""

    def test_build_viability_basic(self) -> None:
        """Test basic viability building."""
        segment = make_segment(
            size=50,
            total_clv=Decimal("50000"),
            actionability_dimensions=[ActionabilityDimension.WHAT],
        )

        viability = build_segment_viability(segment)

        assert isinstance(viability, SegmentViability)
        assert viability.segment_id == segment.segment_id
        assert viability.size == segment.size
        assert viability.total_clv == segment.total_clv

    def test_build_viability_with_robustness(self) -> None:
        """Test viability building with robustness."""
        segment = make_segment()
        robustness = make_robustness_score(feature_stability=0.9, time_consistency=0.9)

        viability = build_segment_viability(segment, robustness=robustness)

        assert viability.confidence_level == ConfidenceLevel.HIGH
        assert viability.robustness_score.segment_id == segment.segment_id

    def test_build_viability_scores(self) -> None:
        """Test viability includes all scores."""
        segment = make_segment(
            actionability_dimensions=[ActionabilityDimension.WHAT, ActionabilityDimension.WHEN],
        )

        viability = build_segment_viability(segment)

        assert 0 <= viability.marketing_targetability <= 1
        assert 0 <= viability.sales_prioritization <= 1
        assert 0 <= viability.personalization_opportunity <= 1
        assert 0 <= viability.timing_optimization <= 1


# =============================================================================
# SEGMENT VALIDATOR CLASS TESTS
# =============================================================================


class TestSegmentValidatorClass:
    """Tests for SegmentValidator class."""

    def test_validator_initialization(self) -> None:
        """Test validator initialization."""
        criteria = ValidationCriteria(min_segment_size=20)
        validator = SegmentValidator(criteria=criteria, total_customers=1000)

        assert validator.criteria.min_segment_size == 20
        assert validator.total_customers == 1000

    def test_validate_single_segment(self) -> None:
        """Test validating a single segment."""
        segment = make_segment(
            size=50,
            total_clv=Decimal("50000"),
            actionability_dimensions=[ActionabilityDimension.WHO],
        )
        validator = SegmentValidator(total_customers=500)

        result = validator.validate(segment)

        assert result.is_valid
        assert segment.segment_id in validator.validated_segments

    def test_validate_batch(self) -> None:
        """Test batch validation."""
        segments = [
            make_segment(
                segment_id=f"seg_{i}",
                size=50,
                total_clv=Decimal("50000"),
                actionability_dimensions=[ActionabilityDimension.WHAT],
            )
            for i in range(3)
        ]
        validator = SegmentValidator(total_customers=500)

        results = validator.validate_batch(segments)

        assert len(results) == 3
        assert all(r.is_valid for r in results.values())

    def test_filter_valid_segments(self) -> None:
        """Test filtering to valid segments only."""
        segments = [
            make_segment(
                segment_id="valid",
                size=50,
                total_clv=Decimal("50000"),
                actionability_dimensions=[ActionabilityDimension.WHAT],
            ),
            make_segment(
                segment_id="invalid",
                size=3,  # Too small
            ),
        ]
        validator = SegmentValidator(total_customers=500)

        valid = validator.filter_valid_segments(segments)

        assert len(valid) == 1
        assert valid[0].segment_id == "valid"

    def test_build_viability(self) -> None:
        """Test viability building through validator."""
        segment = make_segment(
            size=50,
            total_clv=Decimal("50000"),
            actionability_dimensions=[ActionabilityDimension.WHAT],
        )
        robustness = make_robustness_score()
        validator = SegmentValidator(total_customers=500)

        viability = validator.build_viability(segment, robustness=robustness)

        assert isinstance(viability, SegmentViability)
        assert viability.is_approved

    def test_validation_summary(self) -> None:
        """Test validation summary statistics."""
        segments = [
            make_segment(
                segment_id="valid1",
                size=50,
                total_clv=Decimal("50000"),
                actionability_dimensions=[ActionabilityDimension.WHAT],
            ),
            make_segment(
                segment_id="valid2",
                size=30,
                total_clv=Decimal("30000"),
                actionability_dimensions=[ActionabilityDimension.WHO],
            ),
            make_segment(
                segment_id="invalid",
                size=3,
            ),
        ]
        validator = SegmentValidator(total_customers=500)
        validator.validate_batch(segments)

        summary = validator.get_validation_summary()

        assert summary["total"] == 3
        assert summary["valid"] == 2
        assert summary["invalid"] == 1
        assert summary["validation_rate"] == pytest.approx(2 / 3)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestValidatorIntegration:
    """Integration tests for validator with other modules."""

    def test_integration_with_clustering_segments(self) -> None:
        """Test validation of segments from clustering."""
        from datetime import datetime, timezone

        from src.data.joiner import resolve_customer_merges
        from src.data.synthetic_generator import SyntheticDataGenerator
        from src.features.profile_builder import build_profiles_batch
        from src.segmentation.clusterer import CustomerClusterer
        from src.segmentation.sensitivity import SensitivityAnalyzer

        # Generate data
        generator = SyntheticDataGenerator(seed=42)
        date_range = (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 31, tzinfo=timezone.utc),
        )
        dataset = generator.generate_dataset(n_customers=50, date_range=date_range)

        # Build profiles
        merge_map = resolve_customer_merges(dataset.id_history)
        ref_date = datetime(2024, 4, 1, tzinfo=timezone.utc)
        profiles = build_profiles_batch(
            dataset.events,
            merge_map=merge_map,
            reference_date=ref_date,
        )

        # Cluster
        clusterer = CustomerClusterer(n_clusters=4)
        segments = clusterer.create_segments(profiles)

        # Add actionability to segments (simulating enrichment)
        for segment in segments:
            segment.actionability_dimensions = [ActionabilityDimension.WHO]

        # Run sensitivity analysis
        analyzer = SensitivityAnalyzer(n_clusters=4, n_bootstrap=3)
        robustness_scores = analyzer.analyze_segments(profiles, segments)

        # Validate
        validator = SegmentValidator(total_customers=len(profiles))
        results = validator.validate_batch(segments, robustness_scores=robustness_scores)

        # Verify
        assert len(results) == len(segments)

        # Get summary
        summary = validator.get_validation_summary()
        assert summary["total"] == len(segments)

        # Build viabilities for valid segments
        valid_segments = validator.filter_valid_segments(
            segments,
            robustness_scores=robustness_scores,
        )

        for seg in valid_segments:
            viability = validator.build_viability(
                seg,
                robustness=robustness_scores.get(seg.segment_id),
            )
            assert viability.is_approved
