"""
Tests for data schemas and exceptions.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.data.schemas import (
    ActionabilityDimension,
    ActionabilityEvaluation,
    ActionableTrait,
    CategoryAffinity,
    ConfidenceLevel,
    CustomerIdHistory,
    CustomerProfile,
    CustomerProperties,
    CustomerTraits,
    EventProperties,
    EventRecord,
    EventType,
    FeatureSensitivityResult,
    RobustnessScore,
    RobustnessTier,
    Segment,
    SegmentExplanation,
    SegmentMember,
    SegmentViability,
    SegmentationReport,
    StrategicGoal,
    SyntheticDataset,
    TimeWindowSensitivityResult,
)
from src.exceptions import (
    CircularMergeError,
    ClusteringError,
    CustomerMergeError,
    DataValidationError,
    FeatureSensitivityError,
    InsufficientDataError,
    LLMIntegrationError,
    LLMResponseParseError,
    MergeChainTooDeepError,
    ProfileBuildError,
    ReportGenerationError,
    SegmentationEngineError,
    SegmentRejectedError,
    SensitivityTestError,
    TimeWindowSensitivityError,
    TraitExtractionError,
)


# =============================================================================
# EXCEPTION TESTS
# =============================================================================


class TestExceptions:
    """Tests for exception hierarchy."""

    def test_base_exception(self) -> None:
        """Test base exception with context."""
        exc = SegmentationEngineError("Test error", context={"key": "value"})
        assert exc.message == "Test error"
        assert exc.context == {"key": "value"}
        assert "Test error" in str(exc)

    def test_base_exception_repr(self) -> None:
        """Test exception repr."""
        exc = SegmentationEngineError("Test", context={"a": 1})
        repr_str = repr(exc)
        assert "SegmentationEngineError" in repr_str
        assert "Test" in repr_str

    def test_data_validation_error(self) -> None:
        """Test data validation error with field info."""
        exc = DataValidationError(
            "Invalid field",
            field="email",
            value="not-an-email",
        )
        assert exc.field == "email"
        assert exc.value == "not-an-email"
        assert exc.context["field"] == "email"

    def test_customer_merge_error(self) -> None:
        """Test customer merge error."""
        exc = CustomerMergeError(
            "Merge failed",
            customer_ids=["cust1", "cust2"],
        )
        assert exc.customer_ids == ["cust1", "cust2"]
        assert "cust1" in exc.context["customer_ids"]

    def test_circular_merge_error(self) -> None:
        """Test circular merge error."""
        exc = CircularMergeError(
            "Circular detected",
            cycle_path=["a", "b", "c", "a"],
        )
        assert exc.cycle_path == ["a", "b", "c", "a"]

    def test_merge_chain_too_deep_error(self) -> None:
        """Test merge chain depth error."""
        exc = MergeChainTooDeepError(
            "Chain too deep",
            max_depth=10,
            actual_depth=15,
            chain_path=["a", "b", "c"],
        )
        assert exc.max_depth == 10
        assert exc.actual_depth == 15
        assert exc.chain_path == ["a", "b", "c"]

    def test_segment_rejected_error(self) -> None:
        """Test segment rejection error."""
        exc = SegmentRejectedError(
            "Segment not viable",
            segment_id="seg-123",
            rejection_reasons=["too small", "low ROI"],
        )
        assert exc.segment_id == "seg-123"
        assert len(exc.rejection_reasons) == 2

    def test_sensitivity_test_error(self) -> None:
        """Test sensitivity test error."""
        exc = SensitivityTestError("Test failed", test_type="feature")
        assert exc.test_type == "feature"

    def test_feature_sensitivity_error(self) -> None:
        """Test feature sensitivity error."""
        exc = FeatureSensitivityError(
            "Feature test failed",
            dropped_features=["feature1", "feature2"],
        )
        assert exc.test_type == "feature_sensitivity"
        assert exc.dropped_features == ["feature1", "feature2"]

    def test_time_window_sensitivity_error(self) -> None:
        """Test time window sensitivity error."""
        exc = TimeWindowSensitivityError(
            "Time window test failed",
            windows_tested=[30, 60, 90],
        )
        assert exc.test_type == "time_window_sensitivity"
        assert exc.windows_tested == [30, 60, 90]

    def test_llm_integration_error(self) -> None:
        """Test LLM integration error."""
        exc = LLMIntegrationError(
            "API error",
            llm_model="gpt-4",
            prompt_type="actionability",
        )
        assert exc.llm_model == "gpt-4"
        assert exc.prompt_type == "actionability"

    def test_llm_response_parse_error(self) -> None:
        """Test LLM response parse error with truncation."""
        long_response = "x" * 1000
        exc = LLMResponseParseError(
            "Parse failed",
            raw_response=long_response,
            expected_format="JSON",
        )
        assert exc.expected_format == "JSON"
        # Should truncate to 500 chars in context
        assert len(exc.context["raw_response"]) == 500

    def test_clustering_error(self) -> None:
        """Test clustering error."""
        exc = ClusteringError(
            "Clustering failed",
            n_clusters=5,
            n_samples=100,
        )
        assert exc.n_clusters == 5
        assert exc.n_samples == 100

    def test_insufficient_data_error(self) -> None:
        """Test insufficient data error."""
        exc = InsufficientDataError(
            "Not enough data",
            required=100,
            actual=50,
            data_type="customers",
        )
        assert exc.required == 100
        assert exc.actual == 50
        assert exc.data_type == "customers"

    def test_profile_build_error(self) -> None:
        """Test profile build error."""
        exc = ProfileBuildError(
            "Profile build failed",
            customer_id="cust-123",
        )
        assert exc.customer_id == "cust-123"

    def test_trait_extraction_error(self) -> None:
        """Test trait extraction error."""
        exc = TraitExtractionError(
            "Trait extraction failed",
            trait_name="high_value_dormant",
        )
        assert exc.trait_name == "high_value_dormant"

    def test_report_generation_error(self) -> None:
        """Test report generation error."""
        exc = ReportGenerationError(
            "Report failed",
            report_type="segment_report",
        )
        assert exc.report_type == "segment_report"


# =============================================================================
# EVENT SCHEMA TESTS
# =============================================================================


class TestEventSchemas:
    """Tests for event-related schemas."""

    def test_event_properties_defaults(self) -> None:
        """Test EventProperties with defaults."""
        props = EventProperties()
        assert props.product_id is None
        assert props.extra == {}

    def test_event_properties_with_values(self) -> None:
        """Test EventProperties with values."""
        props = EventProperties(
            product_id="prod-123",
            product_name="Test Product",
            product_price=Decimal("99.99"),
            quantity=2,
        )
        assert props.product_id == "prod-123"
        assert props.product_price == Decimal("99.99")

    def test_event_record_creation(self) -> None:
        """Test EventRecord creation."""
        event = EventRecord(
            event_id="evt-123",
            internal_customer_id="cust-456",
            event_type=EventType.VIEW_ITEM,
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        )
        assert event.event_id == "evt-123"
        assert event.event_type == EventType.VIEW_ITEM

    def test_event_record_repr(self) -> None:
        """Test EventRecord repr."""
        event = EventRecord(
            event_id="evt-123",
            internal_customer_id="cust-456",
            event_type=EventType.PURCHASE,
            timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
        )
        repr_str = repr(event)
        assert "evt-123" in repr_str
        assert "purchase" in repr_str

    def test_event_record_frozen(self) -> None:
        """Test EventRecord is immutable."""
        event = EventRecord(
            event_id="evt-123",
            internal_customer_id="cust-456",
            event_type=EventType.VIEW_ITEM,
            timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
        )
        with pytest.raises(ValidationError):
            event.event_id = "new-id"  # type: ignore[misc]


# =============================================================================
# CUSTOMER SCHEMA TESTS
# =============================================================================


class TestCustomerSchemas:
    """Tests for customer-related schemas."""

    def test_customer_properties(self) -> None:
        """Test CustomerProperties creation."""
        props = CustomerProperties(
            internal_customer_id="cust-123",
            email="test@example.com",
            first_name="John",
            loyalty_tier="gold",
        )
        assert props.internal_customer_id == "cust-123"
        assert props.email == "test@example.com"

    def test_customer_id_history(self) -> None:
        """Test CustomerIdHistory creation."""
        history = CustomerIdHistory(
            internal_customer_id="cust-canonical",
            past_id="cust-old",
            merge_timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
        )
        assert history.internal_customer_id == "cust-canonical"
        assert history.past_id == "cust-old"

    def test_customer_id_history_repr(self) -> None:
        """Test CustomerIdHistory repr."""
        history = CustomerIdHistory(
            internal_customer_id="canonical",
            past_id="old",
            merge_timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
        )
        repr_str = repr(history)
        assert "old" in repr_str
        assert "canonical" in repr_str


# =============================================================================
# CUSTOMER PROFILE TESTS
# =============================================================================


class TestCustomerProfile:
    """Tests for CustomerProfile schema."""

    def test_profile_creation(self) -> None:
        """Test CustomerProfile creation with defaults."""
        profile = CustomerProfile(
            internal_customer_id="cust-123",
            first_seen=datetime(2024, 1, 1, tzinfo=timezone.utc),
            last_seen=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        assert profile.total_purchases == 0
        assert profile.total_revenue == Decimal("0")

    def test_profile_with_metrics(self) -> None:
        """Test CustomerProfile with metrics."""
        profile = CustomerProfile(
            internal_customer_id="cust-123",
            first_seen=datetime(2024, 1, 1, tzinfo=timezone.utc),
            last_seen=datetime(2024, 6, 1, tzinfo=timezone.utc),
            total_purchases=10,
            total_revenue=Decimal("1500.00"),
            avg_order_value=Decimal("150.00"),
            cart_abandonment_rate=0.25,
        )
        assert profile.total_purchases == 10
        assert profile.total_revenue == Decimal("1500.00")
        assert profile.cart_abandonment_rate == 0.25

    def test_profile_validation_negative_values(self) -> None:
        """Test profile rejects negative values."""
        with pytest.raises(ValidationError):
            CustomerProfile(
                internal_customer_id="cust-123",
                first_seen=datetime(2024, 1, 1, tzinfo=timezone.utc),
                last_seen=datetime(2024, 6, 1, tzinfo=timezone.utc),
                total_purchases=-1,
            )

    def test_profile_validation_rate_bounds(self) -> None:
        """Test profile validates rate bounds."""
        with pytest.raises(ValidationError):
            CustomerProfile(
                internal_customer_id="cust-123",
                first_seen=datetime(2024, 1, 1, tzinfo=timezone.utc),
                last_seen=datetime(2024, 6, 1, tzinfo=timezone.utc),
                cart_abandonment_rate=1.5,  # > 1.0
            )

    def test_profile_repr(self) -> None:
        """Test CustomerProfile repr."""
        profile = CustomerProfile(
            internal_customer_id="cust-123",
            first_seen=datetime(2024, 1, 1, tzinfo=timezone.utc),
            last_seen=datetime(2024, 6, 1, tzinfo=timezone.utc),
            total_purchases=5,
            total_revenue=Decimal("500.00"),
        )
        repr_str = repr(profile)
        assert "cust-123" in repr_str
        assert "purchases=5" in repr_str


# =============================================================================
# TRAIT SCHEMA TESTS
# =============================================================================


class TestTraitSchemas:
    """Tests for trait-related schemas."""

    def test_actionable_trait_creation(self) -> None:
        """Test ActionableTrait creation."""
        trait = ActionableTrait(
            name="high_value_dormant",
            description="High historical value, no recent purchases",
            value=True,
            actionability_dimension=ActionabilityDimension.WHO,
            business_relevance="Reactivation campaign priority",
        )
        assert trait.name == "high_value_dormant"
        assert trait.actionability_dimension == ActionabilityDimension.WHO

    def test_trait_rejects_technical_jargon(self) -> None:
        """Test trait rejects PCA and similar technical terms."""
        with pytest.raises(ValidationError) as exc_info:
            ActionableTrait(
                name="pca_component_1",
                description="First PCA component",
                value=0.5,
                actionability_dimension=ActionabilityDimension.WHO,
                business_relevance="Technical feature",
            )
        assert "technical jargon" in str(exc_info.value).lower()

    def test_trait_rejects_embedding_terms(self) -> None:
        """Test trait rejects embedding terms."""
        with pytest.raises(ValidationError):
            ActionableTrait(
                name="embedding_cluster_3",
                description="Embedding cluster",
                value=3,
                actionability_dimension=ActionabilityDimension.WHO,
                business_relevance="Technical",
            )

    def test_customer_traits_get_trait(self) -> None:
        """Test CustomerTraits.get_trait method."""
        trait = ActionableTrait(
            name="weekend_shopper",
            description="Shops on weekends",
            value=True,
            actionability_dimension=ActionabilityDimension.WHEN,
            business_relevance="Timing optimization",
        )
        traits = CustomerTraits(
            internal_customer_id="cust-123",
            traits=[trait],
            extraction_timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
        )
        found = traits.get_trait("weekend_shopper")
        assert found is not None
        assert found.name == "weekend_shopper"

        not_found = traits.get_trait("nonexistent")
        assert not_found is None

    def test_customer_traits_has_trait(self) -> None:
        """Test CustomerTraits.has_trait method."""
        trait = ActionableTrait(
            name="discount_sensitive",
            description="Responds to discounts",
            value=True,
            actionability_dimension=ActionabilityDimension.WHAT,
            business_relevance="Promotion targeting",
        )
        traits = CustomerTraits(
            internal_customer_id="cust-123",
            traits=[trait],
            extraction_timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
        )
        assert traits.has_trait("discount_sensitive") is True
        assert traits.has_trait("nonexistent") is False


# =============================================================================
# SEGMENT SCHEMA TESTS
# =============================================================================


class TestSegmentSchemas:
    """Tests for segment-related schemas."""

    def test_segment_member(self) -> None:
        """Test SegmentMember creation."""
        member = SegmentMember(
            internal_customer_id="cust-123",
            membership_score=0.85,
        )
        assert member.internal_customer_id == "cust-123"
        assert member.membership_score == 0.85

    def test_segment_creation(self) -> None:
        """Test Segment creation."""
        segment = Segment(
            segment_id="seg-001",
            name="High-Value Dormant",
            description="Customers with high CLV but no recent activity",
            size=150,
            total_clv=Decimal("75000"),
        )
        assert segment.segment_id == "seg-001"
        assert segment.size == 150

    def test_segment_repr(self) -> None:
        """Test Segment repr."""
        segment = Segment(
            segment_id="seg-001",
            name="Test Segment",
            description="Test",
            size=100,
        )
        repr_str = repr(segment)
        assert "seg-001" in repr_str
        assert "size=100" in repr_str


# =============================================================================
# ROBUSTNESS SCHEMA TESTS
# =============================================================================


class TestRobustnessSchemas:
    """Tests for robustness and sensitivity schemas."""

    def test_feature_sensitivity_result(self) -> None:
        """Test FeatureSensitivityResult creation."""
        result = FeatureSensitivityResult(
            segment_id="seg-001",
            feature_stability=0.78,
            critical_features=["total_revenue", "purchase_frequency"],
            iterations_passed=8,
            total_iterations=10,
        )
        assert result.feature_stability == 0.78
        assert len(result.critical_features) == 2

    def test_time_window_sensitivity_result(self) -> None:
        """Test TimeWindowSensitivityResult creation."""
        result = TimeWindowSensitivityResult(
            segment_id="seg-001",
            time_consistency=0.82,
            windows_tested=[30, 60, 90],
            stable_across_windows=True,
        )
        assert result.time_consistency == 0.82
        assert result.stable_across_windows is True

    def test_robustness_score_calculate_high(self) -> None:
        """Test RobustnessScore.calculate with high scores."""
        score = RobustnessScore.calculate(
            segment_id="seg-001",
            feature_stability=0.85,
            time_window_consistency=0.80,
        )
        assert score.robustness_tier == RobustnessTier.HIGH
        assert score.is_production_ready is True
        assert score.requires_monitoring is False
        # 0.6 * 0.85 + 0.4 * 0.80 = 0.51 + 0.32 = 0.83
        assert 0.82 < score.overall_robustness < 0.84

    def test_robustness_score_calculate_medium(self) -> None:
        """Test RobustnessScore.calculate with medium scores."""
        score = RobustnessScore.calculate(
            segment_id="seg-001",
            feature_stability=0.60,
            time_window_consistency=0.55,
        )
        assert score.robustness_tier == RobustnessTier.MEDIUM
        assert score.is_production_ready is True
        assert score.requires_monitoring is True

    def test_robustness_score_calculate_low(self) -> None:
        """Test RobustnessScore.calculate with low scores."""
        score = RobustnessScore.calculate(
            segment_id="seg-001",
            feature_stability=0.30,
            time_window_consistency=0.40,
        )
        assert score.robustness_tier == RobustnessTier.LOW
        assert score.is_production_ready is False

    def test_robustness_score_with_optional_scores(self) -> None:
        """Test RobustnessScore with all four scores."""
        score = RobustnessScore.calculate(
            segment_id="seg-001",
            feature_stability=0.80,
            time_window_consistency=0.75,
            sampling_stability=0.70,
            threshold_robustness=0.65,
        )
        # Full scoring: 0.4*0.8 + 0.3*0.75 + 0.2*0.70 + 0.1*0.65 = 0.755
        assert 0.74 < score.overall_robustness < 0.77


# =============================================================================
# VIABILITY AND LLM SCHEMA TESTS
# =============================================================================


class TestViabilityAndLLMSchemas:
    """Tests for viability and LLM-related schemas."""

    def test_segment_viability(self) -> None:
        """Test SegmentViability creation."""
        robustness = RobustnessScore.calculate(
            segment_id="seg-001",
            feature_stability=0.80,
            time_window_consistency=0.75,
        )
        viability = SegmentViability(
            segment_id="seg-001",
            size=200,
            total_clv=Decimal("100000"),
            marketing_targetability=0.9,
            sales_prioritization=0.7,
            personalization_opportunity=0.8,
            timing_optimization=0.6,
            cost_to_exploit=Decimal("5000"),
            expected_roi=8.5,
            revenue_impact="high",
            retention_impact="medium",
            satisfaction_impact="low",
            robustness_score=robustness,
            confidence_level=ConfidenceLevel.HIGH,
            recommended_action="Launch reactivation campaign",
            business_hypothesis="If we target with 15% discount, expect 25% reactivation",
            is_approved=True,
        )
        assert viability.is_approved is True
        assert viability.expected_roi == 8.5

    def test_actionability_evaluation(self) -> None:
        """Test ActionabilityEvaluation creation."""
        evaluation = ActionabilityEvaluation(
            segment_id="seg-001",
            is_actionable=True,
            reasoning="Clear targeting opportunity on weekends",
            recommended_action="Weekend push notification campaign",
            confidence_level=ConfidenceLevel.HIGH,
            actionability_dimensions=[
                ActionabilityDimension.WHEN,
                ActionabilityDimension.HOW,
            ],
        )
        assert evaluation.is_actionable is True
        assert len(evaluation.actionability_dimensions) == 2

    def test_segment_explanation(self) -> None:
        """Test SegmentExplanation creation."""
        explanation = SegmentExplanation(
            segment_id="seg-001",
            executive_summary="347 high-value customers inactive for 120+ days",
            key_characteristics=["High historical CLV", "No purchases in 4 months"],
            recommended_campaign="Personalized reactivation email with 15% discount",
            business_hypothesis="If we re-engage these customers, expect 25% reactivation",
            expected_roi="8:1 based on similar campaigns",
            confidence_level=ConfidenceLevel.HIGH,
            confidence_justification="Segment passed all sensitivity tests",
        )
        assert len(explanation.key_characteristics) == 2


# =============================================================================
# REPORT SCHEMA TESTS
# =============================================================================


class TestReportSchemas:
    """Tests for report-related schemas."""

    def test_segmentation_report(self) -> None:
        """Test SegmentationReport creation."""
        report = SegmentationReport(
            report_id="rpt-001",
            generated_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
            data_source="synthetic",
            n_customers_analyzed=10000,
            n_segments_generated=10,
            n_segments_approved=5,
            total_clv_covered=Decimal("500000"),
            coverage_percentage=0.65,
        )
        assert report.n_segments_approved == 5
        assert report.coverage_percentage == 0.65

    def test_segmentation_report_repr(self) -> None:
        """Test SegmentationReport repr."""
        report = SegmentationReport(
            report_id="rpt-001",
            generated_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
            data_source="synthetic",
            n_customers_analyzed=10000,
            n_segments_generated=10,
            n_segments_approved=5,
            total_clv_covered=Decimal("500000"),
            coverage_percentage=0.65,
        )
        repr_str = repr(report)
        assert "rpt-001" in repr_str
        assert "approved_segments=5" in repr_str


# =============================================================================
# SYNTHETIC DATASET TESTS
# =============================================================================


class TestSyntheticDataset:
    """Tests for SyntheticDataset schema."""

    def test_synthetic_dataset_creation(self) -> None:
        """Test SyntheticDataset creation."""
        dataset = SyntheticDataset(
            seed=42,
            n_customers=1000,
            date_range_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            date_range_end=datetime(2024, 6, 30, tzinfo=timezone.utc),
        )
        assert dataset.seed == 42
        assert dataset.n_customers == 1000
        assert dataset.events == []

    def test_synthetic_dataset_repr(self) -> None:
        """Test SyntheticDataset repr."""
        dataset = SyntheticDataset(
            seed=42,
            n_customers=1000,
            date_range_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            date_range_end=datetime(2024, 6, 30, tzinfo=timezone.utc),
            n_events=5000,
        )
        repr_str = repr(dataset)
        assert "seed=42" in repr_str
        assert "events=5000" in repr_str


# =============================================================================
# ENUM TESTS
# =============================================================================


class TestEnums:
    """Tests for enum definitions."""

    def test_event_type_values(self) -> None:
        """Test EventType enum values."""
        assert EventType.SESSION_START.value == "session_start"
        assert EventType.PURCHASE.value == "purchase"
        assert len(EventType) == 10

    def test_actionability_dimension_values(self) -> None:
        """Test ActionabilityDimension enum values."""
        assert ActionabilityDimension.WHAT.value == "WHAT"
        assert ActionabilityDimension.WHEN.value == "WHEN"
        assert len(ActionabilityDimension) == 4

    def test_strategic_goal_values(self) -> None:
        """Test StrategicGoal enum values."""
        assert StrategicGoal.INCREASE_REVENUE.value == "increase_revenue"
        assert len(StrategicGoal) == 3

    def test_robustness_tier_values(self) -> None:
        """Test RobustnessTier enum values."""
        assert RobustnessTier.HIGH.value == "HIGH"
        assert RobustnessTier.MEDIUM.value == "MEDIUM"
        assert RobustnessTier.LOW.value == "LOW"

    def test_confidence_level_values(self) -> None:
        """Test ConfidenceLevel enum values."""
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.LOW.value == "low"
