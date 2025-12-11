"""
Module: schemas

Purpose: Pydantic models for all data structures in the segmentation engine.

All models use Pydantic v2 for validation with strict type hints.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# =============================================================================
# ENUMS
# =============================================================================


class EventType(str, Enum):
    """Bloomreach EBQ event types."""

    SESSION_START = "session_start"
    VIEW_CATEGORY = "view_category"
    VIEW_ITEM = "view_item"
    ADD_TO_CART = "add_to_cart"
    CHECKOUT = "checkout"
    PURCHASE = "purchase"
    PURCHASE_ITEM = "purchase_item"
    REFUND = "refund"
    REFUND_ITEM = "refund_item"
    DELIVERY_EVENTS = "delivery_events"


class ActionabilityDimension(str, Enum):
    """Dimensions that make a segment actionable."""

    WHAT = "WHAT"  # Product/offer targeting
    WHEN = "WHEN"  # Timing optimization
    HOW = "HOW"  # Channel/message personalization
    WHO = "WHO"  # Prioritization


class StrategicGoal(str, Enum):
    """Strategic business goals a segment can influence."""

    INCREASE_REVENUE = "increase_revenue"
    REDUCE_CHURN = "reduce_churn"
    INCREASE_SATISFACTION = "increase_satisfaction"


class RobustnessTier(str, Enum):
    """Robustness classification tiers."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ConfidenceLevel(str, Enum):
    """Confidence level for segment insights."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# BASE MODELS
# =============================================================================


class BaseSchema(BaseModel):
    """Base model with common configuration."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )


class MutableBaseSchema(BaseModel):
    """Base model for mutable schemas."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


# =============================================================================
# EVENT SCHEMAS (Bloomreach EBQ)
# =============================================================================


class EventProperties(BaseSchema):
    """Nested properties within an event record."""

    # Common properties
    product_id: str | None = None
    product_name: str | None = None
    product_category: str | None = None
    product_price: Decimal | None = None
    quantity: int | None = None

    # Purchase-specific
    order_id: str | None = None
    total_amount: Decimal | None = None
    order_total: Decimal | None = None  # Alternative name for total_amount
    discount_amount: Decimal | None = None
    currency: str | None = None

    # Page/view-specific
    page_url: str | None = None
    page_title: str | None = None

    # Search-specific
    search_query: str | None = None

    # Session-specific
    session_id: str | None = None
    device_type: str | None = None
    browser: str | None = None
    referrer: str | None = None

    # Cart-specific
    cart_id: str | None = None

    # Custom properties (extensible)
    extra: dict[str, Any] = Field(default_factory=dict)
    custom_properties: dict[str, Any] | None = None  # For BQ adapter overflow


class EventRecord(BaseSchema):
    """Single event from Bloomreach EBQ events tables."""

    event_id: str
    internal_customer_id: str
    event_type: EventType
    timestamp: datetime
    properties: EventProperties = Field(default_factory=EventProperties)

    def __repr__(self) -> str:
        return (
            f"EventRecord(event_id={self.event_id!r}, "
            f"customer_id={self.internal_customer_id!r}, "
            f"type={self.event_type.value!r}, "
            f"timestamp={self.timestamp.isoformat()})"
        )


# =============================================================================
# CUSTOMER SCHEMAS (Bloomreach EBQ)
# =============================================================================


class CustomerProperties(BaseSchema):
    """Customer properties from customers_properties table."""

    internal_customer_id: str
    email: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    registration_date: datetime | None = None
    country: str | None = None
    city: str | None = None
    loyalty_tier: str | None = None

    # Custom properties (extensible)
    extra: dict[str, Any] = Field(default_factory=dict)

    def __repr__(self) -> str:
        return f"CustomerProperties(customer_id={self.internal_customer_id!r})"


class CustomerIdHistory(BaseSchema):
    """Customer ID merge history from customers_id_history table."""

    internal_customer_id: str  # The canonical ID (merged-into)
    past_id: str  # The historical ID (merged-from)
    merge_timestamp: datetime

    def __repr__(self) -> str:
        return (
            f"CustomerIdHistory(past_id={self.past_id!r} -> "
            f"canonical_id={self.internal_customer_id!r})"
        )


class CustomerExternalId(BaseSchema):
    """External customer IDs from customers_external_ids table."""

    internal_customer_id: str
    external_id_type: str  # e.g., "crm_id", "email", "loyalty_id"
    external_id_value: str

    def __repr__(self) -> str:
        return (
            f"CustomerExternalId(customer_id={self.internal_customer_id!r}, "
            f"type={self.external_id_type!r})"
        )


# =============================================================================
# CUSTOMER PROFILE SCHEMA
# =============================================================================


class CategoryAffinity(BaseSchema):
    """Category affinity score for a customer."""

    category: str
    engagement_score: Annotated[float, Field(ge=0.0, le=1.0)]
    view_count: int = 0
    purchase_count: int = 0


class CustomerProfile(MutableBaseSchema):
    """Canonical customer profile with aggregated metrics and traits."""

    # Identity
    internal_customer_id: str
    merged_from_ids: list[str] = Field(default_factory=list)

    # Temporal
    first_seen: datetime
    last_seen: datetime

    # Transactional metrics
    total_purchases: Annotated[int, Field(ge=0)] = 0
    total_revenue: Annotated[Decimal, Field(ge=0)] = Decimal("0")
    avg_order_value: Annotated[Decimal, Field(ge=0)] = Decimal("0")
    days_since_last_purchase: int | None = None
    purchase_frequency_per_month: Annotated[float, Field(ge=0.0)] = 0.0

    # Engagement metrics
    total_sessions: Annotated[int, Field(ge=0)] = 0
    total_page_views: Annotated[int, Field(ge=0)] = 0
    total_items_viewed: Annotated[int, Field(ge=0)] = 0
    total_cart_additions: Annotated[int, Field(ge=0)] = 0
    cart_abandonment_rate: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0

    # Category behavior
    category_affinities: list[CategoryAffinity] = Field(default_factory=list)
    top_category: str | None = None

    # Temporal patterns
    preferred_day_of_week: int | None = None  # 0=Monday, 6=Sunday
    preferred_hour_of_day: int | None = None  # 0-23

    # Device/Channel
    primary_device_type: str | None = None
    mobile_session_ratio: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0

    # Value metrics
    clv_estimate: Annotated[Decimal, Field(ge=0)] = Decimal("0")
    churn_risk_score: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0

    # Refund metrics
    total_refunds: Annotated[int, Field(ge=0)] = 0
    refund_rate: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0

    def __repr__(self) -> str:
        return (
            f"CustomerProfile(id={self.internal_customer_id!r}, "
            f"purchases={self.total_purchases}, "
            f"revenue=${self.total_revenue:.2f})"
        )


# =============================================================================
# TRAIT SCHEMAS
# =============================================================================


class ActionableTrait(BaseSchema):
    """A single actionable trait extracted from customer profile."""

    name: str
    description: str
    value: bool | float | str
    actionability_dimension: ActionabilityDimension
    business_relevance: str  # Why this trait matters for business

    @field_validator("name")
    @classmethod
    def validate_no_technical_jargon(cls, v: str) -> str:
        """Ensure trait names are business-friendly, not technical."""
        forbidden_terms = ["pca", "embedding", "latent", "component", "cluster_"]
        lower_v = v.lower()
        for term in forbidden_terms:
            if term in lower_v:
                msg = f"Trait name '{v}' contains technical jargon '{term}'. Use business-friendly names."
                raise ValueError(msg)
        return v


class CustomerTraits(MutableBaseSchema):
    """Collection of actionable traits for a customer."""

    internal_customer_id: str
    traits: list[ActionableTrait] = Field(default_factory=list)
    extraction_timestamp: datetime

    def get_trait(self, name: str) -> ActionableTrait | None:
        """Get trait by name."""
        for trait in self.traits:
            if trait.name == name:
                return trait
        return None

    def has_trait(self, name: str) -> bool:
        """Check if customer has a specific trait."""
        return self.get_trait(name) is not None


# =============================================================================
# SEGMENT SCHEMAS
# =============================================================================


class SegmentMember(BaseSchema):
    """A customer belonging to a segment."""

    internal_customer_id: str
    membership_score: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0


class Segment(MutableBaseSchema):
    """Customer segment with business attributes."""

    segment_id: str
    name: str
    description: str

    # Membership
    members: list[SegmentMember] = Field(default_factory=list)
    size: Annotated[int, Field(ge=0)] = 0

    # Defining characteristics
    defining_traits: list[str] = Field(default_factory=list)
    trait_summary: dict[str, Any] = Field(default_factory=dict)

    # Value metrics
    total_clv: Annotated[Decimal, Field(ge=0)] = Decimal("0")
    avg_clv: Annotated[Decimal, Field(ge=0)] = Decimal("0")
    avg_order_value: Annotated[Decimal, Field(ge=0)] = Decimal("0")

    # Actionability
    actionability_dimensions: list[ActionabilityDimension] = Field(default_factory=list)
    strategic_goals: list[StrategicGoal] = Field(default_factory=list)

    # Clustering metadata
    cluster_label: int | None = None
    centroid: list[float] | None = None

    def __repr__(self) -> str:
        return f"Segment(id={self.segment_id!r}, name={self.name!r}, size={self.size})"


# =============================================================================
# ROBUSTNESS & SENSITIVITY SCHEMAS
# =============================================================================


class FeatureSensitivityResult(BaseSchema):
    """Result of feature sensitivity test for a single segment."""

    segment_id: str
    feature_stability: Annotated[float, Field(ge=0.0, le=1.0)]
    critical_features: list[str] = Field(default_factory=list)
    iterations_passed: int
    total_iterations: int


class TimeWindowSensitivityResult(BaseSchema):
    """Result of time window sensitivity test for a single segment."""

    segment_id: str
    time_consistency: Annotated[float, Field(ge=0.0, le=1.0)]
    windows_tested: list[int] = Field(default_factory=list)
    stable_across_windows: bool


class RobustnessScore(BaseSchema):
    """Composite robustness score for a segment."""

    segment_id: str

    # Individual scores (0-1, higher = more robust)
    feature_stability: Annotated[float, Field(ge=0.0, le=1.0)]
    time_window_consistency: Annotated[float, Field(ge=0.0, le=1.0)]

    # Optional scores (for roadmap features)
    sampling_stability: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    threshold_robustness: Annotated[float, Field(ge=0.0, le=1.0)] | None = None

    # Composite
    overall_robustness: Annotated[float, Field(ge=0.0, le=1.0)]
    robustness_tier: RobustnessTier

    # Recommendations
    is_production_ready: bool
    requires_monitoring: bool
    stability_notes: str | None = None

    @classmethod
    def calculate(
        cls,
        *,
        segment_id: str,
        feature_stability: float,
        time_window_consistency: float,
        sampling_stability: float | None = None,
        threshold_robustness: float | None = None,
    ) -> "RobustnessScore":
        """Calculate robustness score from individual components."""
        # MVP scoring: 60% feature, 40% time window
        overall = 0.6 * feature_stability + 0.4 * time_window_consistency

        # Adjust if optional scores available
        if sampling_stability is not None and threshold_robustness is not None:
            overall = (
                0.4 * feature_stability
                + 0.3 * time_window_consistency
                + 0.2 * sampling_stability
                + 0.1 * threshold_robustness
            )

        # Determine tier
        if overall >= 0.75:
            tier = RobustnessTier.HIGH
        elif overall >= 0.5:
            tier = RobustnessTier.MEDIUM
        else:
            tier = RobustnessTier.LOW

        return cls(
            segment_id=segment_id,
            feature_stability=feature_stability,
            time_window_consistency=time_window_consistency,
            sampling_stability=sampling_stability,
            threshold_robustness=threshold_robustness,
            overall_robustness=overall,
            robustness_tier=tier,
            is_production_ready=overall >= 0.5,
            requires_monitoring=tier == RobustnessTier.MEDIUM,
        )


# =============================================================================
# SEGMENT VIABILITY SCHEMAS
# =============================================================================


class SegmentViability(BaseSchema):
    """Economic viability assessment for a segment."""

    segment_id: str
    size: int
    total_clv: Decimal

    # Actionability scores (0-1)
    marketing_targetability: Annotated[float, Field(ge=0.0, le=1.0)]
    sales_prioritization: Annotated[float, Field(ge=0.0, le=1.0)]
    personalization_opportunity: Annotated[float, Field(ge=0.0, le=1.0)]
    timing_optimization: Annotated[float, Field(ge=0.0, le=1.0)]

    # Economic metrics
    cost_to_exploit: Decimal
    expected_roi: float

    # Strategic impact
    revenue_impact: Literal["high", "medium", "low"]
    retention_impact: Literal["high", "medium", "low"]
    satisfaction_impact: Literal["high", "medium", "low"]

    # Robustness
    robustness_score: RobustnessScore
    confidence_level: ConfidenceLevel

    # Business play
    recommended_action: str
    business_hypothesis: str

    # Decision
    is_approved: bool
    rejection_reasons: list[str] = Field(default_factory=list)


# =============================================================================
# LLM SCHEMAS
# =============================================================================


class ActionabilityEvaluation(BaseSchema):
    """LLM evaluation of segment actionability."""

    segment_id: str
    is_actionable: bool
    reasoning: str
    recommended_action: str | None = None
    confidence_level: ConfidenceLevel
    actionability_dimensions: list[ActionabilityDimension] = Field(default_factory=list)


class SegmentExplanation(BaseSchema):
    """Business-language explanation of a segment."""

    segment_id: str
    executive_summary: str
    key_characteristics: list[str]
    recommended_campaign: str
    business_hypothesis: str
    expected_roi: str
    confidence_level: ConfidenceLevel
    confidence_justification: str


# =============================================================================
# REPORT SCHEMAS
# =============================================================================


class SegmentInsight(MutableBaseSchema):
    """Complete insight for a single segment."""

    segment: Segment
    viability: SegmentViability
    explanation: SegmentExplanation
    robustness: RobustnessScore


class SegmentationReport(MutableBaseSchema):
    """Complete segmentation analysis report."""

    report_id: str
    generated_at: datetime
    data_source: str  # "synthetic" or BigQuery project ID

    # Configuration
    n_customers_analyzed: int
    n_segments_generated: int
    n_segments_approved: int

    # Results
    segments: list[SegmentInsight] = Field(default_factory=list)
    rejected_segments: list[str] = Field(default_factory=list)  # segment_ids

    # Summary
    total_clv_covered: Decimal
    coverage_percentage: float  # % of customers in approved segments

    def __repr__(self) -> str:
        return (
            f"SegmentationReport(id={self.report_id!r}, "
            f"approved_segments={self.n_segments_approved})"
        )


# =============================================================================
# SYNTHETIC DATA SCHEMAS
# =============================================================================


class SyntheticDataset(MutableBaseSchema):
    """Container for synthetic dataset."""

    seed: int
    n_customers: int
    date_range_start: datetime
    date_range_end: datetime

    events: list[EventRecord] = Field(default_factory=list)
    customer_properties: list[CustomerProperties] = Field(default_factory=list)
    id_history: list[CustomerIdHistory] = Field(default_factory=list)

    # Statistics
    n_events: int = 0
    n_merges: int = 0
    event_type_distribution: dict[str, int] = Field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"SyntheticDataset(seed={self.seed}, "
            f"customers={self.n_customers}, "
            f"events={self.n_events})"
        )
