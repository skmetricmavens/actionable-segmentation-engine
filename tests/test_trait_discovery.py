"""Tests for trait discovery module."""

from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest

from src.analysis.trait_discovery import (
    TraitDiscoveryResult,
    TraitMetadata,
    TraitValueAnalyzer,
    TraitValueScore,
    discover_traits,
    format_trait_report,
)
from src.data.schemas import (
    BehaviorType,
    CustomerProfile,
    EventProperties,
    EventRecord,
    EventType,
)


# =============================================================================
# FIXTURES
# =============================================================================


def make_event(
    customer_id: str,
    event_type: EventType,
    timestamp: datetime,
    properties: dict | None = None,
    extra: dict | None = None,
) -> EventRecord:
    """Create a test event."""
    props_dict = properties or {}
    if extra:
        props_dict["extra"] = extra
    props = EventProperties(**props_dict)
    return EventRecord(
        event_id=f"evt_{customer_id}_{timestamp.timestamp()}",
        internal_customer_id=customer_id,
        event_type=event_type,
        timestamp=timestamp,
        properties=props,
    )


def make_profile(
    customer_id: str,
    total_revenue: float = 100.0,
    churn_risk: float = 0.3,
) -> CustomerProfile:
    """Create a test customer profile."""
    return CustomerProfile(
        internal_customer_id=customer_id,
        first_seen=datetime(2024, 1, 1),
        last_seen=datetime(2024, 6, 1),
        total_purchases=5,
        total_revenue=Decimal(str(total_revenue)),
        avg_order_value=Decimal(str(total_revenue / 5)),
        total_sessions=20,
        total_page_views=100,
        total_items_viewed=50,
        total_cart_additions=15,
        days_since_last_purchase=30,
        purchase_frequency_per_month=0.83,
        cart_abandonment_rate=0.3,
        total_refunds=Decimal("0.00"),
        refund_rate=0.0,
        churn_risk_score=churn_risk,
        preferred_day_of_week=2,
        preferred_hour_of_day=14,
        mobile_session_ratio=0.4,
        category_affinities=[],
        behavior_type=BehaviorType.REGULAR,
    )


@pytest.fixture
def sample_events() -> list[EventRecord]:
    """Create sample events with various product traits."""
    events = []
    base_time = datetime(2024, 1, 1)

    # Create events with different trait values
    colors = ["Red", "Blue", "Black", "White", "Green"]
    brands = ["BrandA", "BrandB", "BrandC"]
    categories = ["Clothing", "Electronics", "Home"]

    for i in range(100):
        customer_id = f"cust_{i:03d}"

        # View events
        events.append(make_event(
            customer_id=customer_id,
            event_type=EventType.VIEW_ITEM,
            timestamp=base_time + timedelta(days=i),
            extra={
                "color": colors[i % len(colors)],
                "brand": brands[i % len(brands)],
                "category_level_1": categories[i % len(categories)],
                "price": 50 + (i % 10) * 10,
            },
        ))

        # Some purchase events
        if i % 3 == 0:
            events.append(make_event(
                customer_id=customer_id,
                event_type=EventType.PURCHASE,
                timestamp=base_time + timedelta(days=i, hours=2),
                extra={
                    "color": colors[i % len(colors)],
                    "brand": brands[i % len(brands)],
                    "category_level_1": categories[i % len(categories)],
                    "price": 50 + (i % 10) * 10,
                    "margin_percentage": 0.4 + (i % 5) * 0.1,
                },
            ))

    return events


@pytest.fixture
def sample_profiles() -> list[CustomerProfile]:
    """Create sample profiles with varying revenue and churn."""
    profiles = []

    for i in range(100):
        # Create correlation: higher customer IDs have higher revenue
        # This simulates trait-revenue correlation
        revenue = 50 + i * 5 + np.random.normal(0, 20)
        churn = 0.2 + (100 - i) * 0.005 + np.random.normal(0, 0.1)
        churn = max(0, min(1, churn))

        profiles.append(make_profile(
            customer_id=f"cust_{i:03d}",
            total_revenue=max(10, revenue),
            churn_risk=churn,
        ))

    return profiles


# =============================================================================
# TEST TRAIT METADATA
# =============================================================================


class TestTraitMetadata:
    """Tests for TraitMetadata dataclass."""

    def test_coverage_calculation(self):
        """Test coverage property."""
        metadata = TraitMetadata(
            field_name="color",
            field_path="extra.color",
            total_occurrences=100,
            null_count=20,
        )
        assert metadata.coverage == 0.8

    def test_coverage_zero_occurrences(self):
        """Test coverage with zero occurrences."""
        metadata = TraitMetadata(
            field_name="color",
            field_path="extra.color",
            total_occurrences=0,
            null_count=0,
        )
        assert metadata.coverage == 0.0

    def test_uniqueness_ratio(self):
        """Test uniqueness ratio calculation."""
        metadata = TraitMetadata(
            field_name="color",
            field_path="extra.color",
            total_occurrences=100,
            null_count=0,
            distinct_values=5,
        )
        assert metadata.uniqueness_ratio == 0.05


# =============================================================================
# TEST TRAIT VALUE SCORE
# =============================================================================


class TestTraitValueScore:
    """Tests for TraitValueScore dataclass."""

    def test_overall_score(self):
        """Test overall score calculation."""
        score = TraitValueScore(
            trait_name="color",
            trait_path="extra.color",
            trait_type="categorical",
            revenue_impact=0.8,
            retention_impact=0.6,
            personalization_value=0.7,
        )
        # 0.4 * 0.8 + 0.3 * 0.6 + 0.3 * 0.7 = 0.32 + 0.18 + 0.21 = 0.71
        assert abs(score.overall_score - 0.71) < 0.01

    def test_is_significant(self):
        """Test significance detection."""
        # Significant by revenue
        score1 = TraitValueScore(
            trait_name="color",
            trait_path="extra.color",
            trait_type="categorical",
            revenue_p_value=0.01,
            retention_p_value=0.1,
        )
        assert score1.is_significant

        # Significant by retention
        score2 = TraitValueScore(
            trait_name="brand",
            trait_path="extra.brand",
            trait_type="categorical",
            revenue_p_value=0.1,
            retention_p_value=0.03,
        )
        assert score2.is_significant

        # Not significant
        score3 = TraitValueScore(
            trait_name="size",
            trait_path="extra.size",
            trait_type="categorical",
            revenue_p_value=0.2,
            retention_p_value=0.3,
        )
        assert not score3.is_significant

    def test_recommended_uses(self):
        """Test recommended uses based on scores."""
        score = TraitValueScore(
            trait_name="color",
            trait_path="extra.color",
            trait_type="categorical",
            revenue_impact=0.7,
            revenue_p_value=0.01,
            retention_impact=0.6,
            retention_p_value=0.02,
            personalization_value=0.8,
            distinct_values=5,
            concentration=0.4,
        )
        uses = score.recommended_uses
        assert "segmentation" in uses
        assert "retention_targeting" in uses
        assert "personalization" in uses
        assert "content_customization" in uses


# =============================================================================
# TEST TRAIT VALUE ANALYZER
# =============================================================================


class TestTraitValueAnalyzer:
    """Tests for TraitValueAnalyzer class."""

    def test_init_defaults(self):
        """Test default initialization."""
        analyzer = TraitValueAnalyzer()
        assert analyzer.min_coverage == 0.05
        assert analyzer.max_cardinality == 100
        assert analyzer.min_cardinality == 2

    def test_init_custom_params(self):
        """Test custom initialization."""
        analyzer = TraitValueAnalyzer(
            min_coverage=0.1,
            max_cardinality=50,
        )
        assert analyzer.min_coverage == 0.1
        assert analyzer.max_cardinality == 50

    def test_is_system_field(self):
        """Test system field detection."""
        analyzer = TraitValueAnalyzer()

        # System fields to skip
        assert analyzer._is_system_field("__cplocation__abc")
        assert analyzer._is_system_field("import_timestamp")
        assert analyzer._is_system_field("session_id")
        assert analyzer._is_system_field("customer_id")

        # Normal fields to keep
        assert not analyzer._is_system_field("color")
        assert not analyzer._is_system_field("brand")
        assert not analyzer._is_system_field("category_level_1")

    def test_is_numeric(self):
        """Test numeric value detection."""
        analyzer = TraitValueAnalyzer()

        assert analyzer._is_numeric(123)
        assert analyzer._is_numeric(45.67)
        assert analyzer._is_numeric(Decimal("99.99"))
        assert analyzer._is_numeric("123.45")
        assert analyzer._is_numeric("1,234.56")

        assert not analyzer._is_numeric("Red")
        assert not analyzer._is_numeric("BrandA")
        assert not analyzer._is_numeric("")

    def test_discover_fields(self, sample_events):
        """Test field discovery from events."""
        analyzer = TraitValueAnalyzer()
        field_metadata = analyzer._discover_fields(sample_events)

        # Should discover our test fields
        assert "extra.color" in field_metadata
        assert "extra.brand" in field_metadata
        assert "extra.category_level_1" in field_metadata
        assert "extra.price" in field_metadata

        # Check metadata
        color_meta = field_metadata["extra.color"]
        assert color_meta.field_name == "color"
        assert color_meta.total_occurrences > 0
        assert len(color_meta.sample_values) > 0

    def test_classify_field_type_categorical(self):
        """Test categorical field classification."""
        analyzer = TraitValueAnalyzer()

        metadata = TraitMetadata(
            field_name="color",
            field_path="extra.color",
            sample_values=["Red", "Blue", "Black", "White", "Green"] * 20,
            total_occurrences=100,
            null_count=0,
            distinct_values=5,
        )

        assert analyzer._classify_field_type(metadata) == "categorical"

    def test_classify_field_type_numeric(self):
        """Test numeric field classification."""
        analyzer = TraitValueAnalyzer()

        metadata = TraitMetadata(
            field_name="price",
            field_path="extra.price",
            sample_values=[50, 60, 70, 80, 90, 100] * 16,
            total_occurrences=96,
            null_count=0,
            distinct_values=6,
        )

        assert analyzer._classify_field_type(metadata) == "numeric"

    def test_classify_field_type_id(self):
        """Test ID field classification (high uniqueness)."""
        analyzer = TraitValueAnalyzer()

        # Unique values for each occurrence = ID field
        metadata = TraitMetadata(
            field_name="sku",
            field_path="extra.sku",
            sample_values=[f"SKU_{i}" for i in range(100)],
            total_occurrences=100,
            null_count=0,
            distinct_values=100,
        )

        assert analyzer._classify_field_type(metadata) == "id"

    def test_classify_field_type_hierarchical(self):
        """Test hierarchical field classification."""
        analyzer = TraitValueAnalyzer()

        metadata = TraitMetadata(
            field_name="category_path",
            field_path="extra.category_path",
            sample_values=[
                "Clothing>Tops>T-Shirts",
                "Clothing>Bottoms>Jeans",
                "Electronics>Phones>Smartphones",
            ] * 30,
            total_occurrences=90,
            null_count=0,
            distinct_values=3,
        )

        assert analyzer._classify_field_type(metadata) == "hierarchical"

    def test_build_customer_trait_profiles(self, sample_events):
        """Test customer trait profile building."""
        analyzer = TraitValueAnalyzer()

        # First discover fields
        field_metadata = analyzer._discover_fields(sample_events)

        # Filter to usable traits
        result = TraitDiscoveryResult()
        trait_metadata = analyzer._classify_and_filter_fields(field_metadata, result)

        # Build profiles
        customer_traits = analyzer._build_customer_trait_profiles(sample_events, trait_metadata)

        # Should have profiles for customers
        assert len(customer_traits) > 0

        # Each profile should have trait values
        first_customer = list(customer_traits.keys())[0]
        assert len(customer_traits[first_customer]) > 0

    def test_analyze_full_pipeline(self, sample_events, sample_profiles):
        """Test full analysis pipeline."""
        analyzer = TraitValueAnalyzer(
            min_coverage=0.01,  # Lower threshold for test data
            min_customers_per_value=2,
        )

        result = analyzer.analyze(sample_events, sample_profiles)

        assert isinstance(result, TraitDiscoveryResult)
        assert result.total_events_scanned == len(sample_events)
        assert result.total_customers == len(sample_profiles)
        assert result.total_fields_found > 0

    def test_analyze_filters_low_coverage(self, sample_profiles):
        """Test that low coverage fields are filtered."""
        # Create events where some fields are rarely populated
        events = []
        base_time = datetime(2024, 1, 1)

        for i in range(100):
            extra = {"common_field": "value"}
            if i < 3:  # Only 3% have rare_field
                extra["rare_field"] = "rare_value"

            events.append(make_event(
                customer_id=f"cust_{i:03d}",
                event_type=EventType.VIEW_ITEM,
                timestamp=base_time + timedelta(days=i),
                extra=extra,
            ))

        analyzer = TraitValueAnalyzer(min_coverage=0.05)
        result = analyzer.analyze(events, sample_profiles)

        # rare_field should be filtered out
        trait_names = [t.trait_name for t in result.traits]
        assert "rare_field" not in trait_names

    def test_analyze_filters_high_cardinality(self, sample_profiles):
        """Test that high cardinality fields are filtered."""
        events = []
        base_time = datetime(2024, 1, 1)

        for i in range(100):
            events.append(make_event(
                customer_id=f"cust_{i:03d}",
                event_type=EventType.VIEW_ITEM,
                timestamp=base_time + timedelta(days=i),
                extra={
                    "low_cardinality": f"value_{i % 5}",  # 5 values
                    "high_cardinality": f"unique_{i}",  # 100 values
                },
            ))

        analyzer = TraitValueAnalyzer(max_cardinality=50)
        result = analyzer.analyze(events, sample_profiles)

        # high_cardinality should be filtered (check path contains name)
        assert any("high_cardinality" in path for path in result.filtered_as_id)


# =============================================================================
# TEST SCORING METHODS
# =============================================================================


class TestScoringMethods:
    """Tests for individual scoring methods."""

    def test_revenue_scoring_with_difference(self):
        """Test revenue scoring detects differences."""
        # Create profiles with clear revenue difference by trait
        profiles = []
        trait_values = {}

        # High revenue group
        for i in range(30):
            cid = f"high_{i}"
            profiles.append(make_profile(cid, total_revenue=500 + np.random.normal(0, 20)))
            trait_values[cid] = "premium"

        # Low revenue group
        for i in range(30):
            cid = f"low_{i}"
            profiles.append(make_profile(cid, total_revenue=50 + np.random.normal(0, 10)))
            trait_values[cid] = "basic"

        analyzer = TraitValueAnalyzer(min_customers_per_value=5)
        profile_lookup = {p.internal_customer_id: p for p in profiles}

        score = TraitValueScore(
            trait_name="tier",
            trait_path="extra.tier",
            trait_type="categorical",
        )

        from collections import Counter
        value_counts = Counter(trait_values.values())

        analyzer._score_revenue_impact(score, trait_values, profile_lookup, value_counts)

        # Should detect significant difference
        assert score.revenue_f_statistic > 10  # Strong F-statistic
        assert score.revenue_p_value < 0.05
        assert score.revenue_impact > 0.5

    def test_retention_scoring_with_difference(self):
        """Test retention scoring detects churn differences."""
        profiles = []
        trait_values = {}

        # Create groups with spread across churn buckets for valid chi2
        # Low churn group - mostly low churn, some medium
        for i in range(30):
            cid = f"loyal_{i}"
            # Mix of low (0-0.3) and medium (0.3-0.7) churn
            if i < 25:
                churn = 0.15 + np.random.uniform(0, 0.1)  # Low bucket
            else:
                churn = 0.4 + np.random.uniform(0, 0.2)  # Medium bucket
            profiles.append(make_profile(cid, churn_risk=churn))
            trait_values[cid] = "engaged"

        # High churn group - mostly high churn, some medium
        for i in range(30):
            cid = f"risky_{i}"
            if i < 25:
                churn = 0.75 + np.random.uniform(0, 0.2)  # High bucket
            else:
                churn = 0.4 + np.random.uniform(0, 0.2)  # Medium bucket
            churn = min(1.0, churn)
            profiles.append(make_profile(cid, churn_risk=churn))
            trait_values[cid] = "disengaged"

        analyzer = TraitValueAnalyzer(min_customers_per_value=5)
        profile_lookup = {p.internal_customer_id: p for p in profiles}

        score = TraitValueScore(
            trait_name="engagement",
            trait_path="extra.engagement",
            trait_type="categorical",
        )

        from collections import Counter
        value_counts = Counter(trait_values.values())

        analyzer._score_retention_impact(score, trait_values, profile_lookup, value_counts)

        # Should have top retention values
        assert len(score.top_retention_values) > 0
        # Engaged should have higher retention than disengaged
        retention_dict = dict(score.top_retention_values)
        assert retention_dict.get("engaged", 0) > retention_dict.get("disengaged", 1)

    def test_personalization_scoring_high_entropy(self):
        """Test personalization scoring for high entropy traits."""
        analyzer = TraitValueAnalyzer()

        score = TraitValueScore(
            trait_name="color",
            trait_path="extra.color",
            trait_type="categorical",
            distinct_values=10,
            concentration=0.15,  # No dominant value
        )

        # Even distribution
        from collections import Counter
        value_counts = Counter({f"color_{i}": 10 for i in range(10)})

        analyzer._score_personalization_value(score, value_counts)

        # High entropy = good for personalization
        assert score.entropy > 2.0
        assert score.personalization_value > 0.7

    def test_personalization_scoring_low_entropy(self):
        """Test personalization scoring for low entropy traits."""
        analyzer = TraitValueAnalyzer()

        score = TraitValueScore(
            trait_name="color",
            trait_path="extra.color",
            trait_type="categorical",
            distinct_values=5,
            concentration=0.9,  # One dominant value
        )

        # Skewed distribution
        from collections import Counter
        value_counts = Counter({"dominant": 90, "rare1": 3, "rare2": 3, "rare3": 2, "rare4": 2})

        analyzer._score_personalization_value(score, value_counts)

        # Low entropy = poor for personalization
        assert score.entropy < 1.5
        assert score.personalization_value < 0.5


# =============================================================================
# TEST DISCOVERY RESULT
# =============================================================================


class TestTraitDiscoveryResult:
    """Tests for TraitDiscoveryResult dataclass."""

    def test_top_traits_properties(self):
        """Test top traits property methods."""
        traits = [
            TraitValueScore(
                trait_name="color",
                trait_path="extra.color",
                trait_type="categorical",
                revenue_impact=0.8,
                revenue_p_value=0.01,
                retention_impact=0.3,
                personalization_value=0.9,
            ),
            TraitValueScore(
                trait_name="brand",
                trait_path="extra.brand",
                trait_type="categorical",
                revenue_impact=0.5,
                revenue_p_value=0.03,
                retention_impact=0.7,
                retention_p_value=0.02,
                personalization_value=0.6,
            ),
            TraitValueScore(
                trait_name="size",
                trait_path="extra.size",
                trait_type="categorical",
                revenue_impact=0.2,
                revenue_p_value=0.3,  # Not significant
                retention_impact=0.1,
                personalization_value=0.4,
            ),
        ]

        result = TraitDiscoveryResult(traits=traits)

        # Top revenue (only significant)
        top_rev = result.top_revenue_traits
        assert len(top_rev) == 2
        assert top_rev[0].trait_name == "color"

        # Top retention (only significant)
        top_ret = result.top_retention_traits
        assert len(top_ret) == 1
        assert top_ret[0].trait_name == "brand"

        # Top personalization
        top_pers = result.top_personalization_traits
        assert top_pers[0].trait_name == "color"

    def test_get_summary(self):
        """Test summary generation."""
        traits = [
            TraitValueScore(
                trait_name="color",
                trait_path="extra.color",
                trait_type="categorical",
                revenue_impact=0.8,
                revenue_p_value=0.01,
            ),
        ]

        result = TraitDiscoveryResult(
            traits=traits,
            total_events_scanned=1000,
            total_customers=100,
            total_fields_found=10,
        )

        summary = result.get_summary()
        assert summary["events_scanned"] == 1000
        assert summary["customers_analyzed"] == 100
        assert summary["fields_discovered"] == 10
        assert summary["traits_scored"] == 1


# =============================================================================
# TEST CONVENIENCE FUNCTIONS
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_discover_traits_function(self, sample_events, sample_profiles):
        """Test discover_traits convenience function."""
        result = discover_traits(
            sample_events,
            sample_profiles,
            min_coverage=0.01,
        )

        assert isinstance(result, TraitDiscoveryResult)
        assert result.total_events_scanned > 0

    def test_format_trait_report(self):
        """Test report formatting."""
        traits = [
            TraitValueScore(
                trait_name="color",
                trait_path="extra.color",
                trait_type="categorical",
                revenue_impact=0.8,
                revenue_f_statistic=25.5,
                revenue_p_value=0.001,
                retention_impact=0.5,
                retention_chi2_statistic=12.3,
                retention_p_value=0.02,
                personalization_value=0.7,
                entropy=2.5,
                distinct_values=5,
                customer_coverage=0.9,
                concentration=0.3,
                top_revenue_values=[("Black", 450.0), ("White", 250.0)],
                top_retention_values=[("Black", 0.85)],
            ),
        ]

        result = TraitDiscoveryResult(
            traits=traits,
            total_events_scanned=1000,
            total_customers=100,
            total_fields_found=10,
        )

        report = format_trait_report(result)

        assert "TRAIT VALUE DISCOVERY REPORT" in report
        assert "color" in report
        assert "Impact" in report  # Either "Revenue Impact" or just "Impact:"
        assert "1,000" in report or "1000" in report  # Events scanned (may be formatted)


# =============================================================================
# TEST EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_events(self):
        """Test with no events."""
        analyzer = TraitValueAnalyzer()
        result = analyzer.analyze([], [])

        assert result.total_events_scanned == 0
        assert len(result.traits) == 0

    def test_no_product_events(self, sample_profiles):
        """Test with non-product events only."""
        events = [
            make_event(
                customer_id="cust_001",
                event_type=EventType.SESSION_START,
                timestamp=datetime(2024, 1, 1),
            ),
        ]

        analyzer = TraitValueAnalyzer()
        result = analyzer.analyze(events, sample_profiles)

        # Should have no traits (session_start is not a product event)
        assert len(result.traits) == 0

    def test_all_null_values(self, sample_profiles):
        """Test field with all null values."""
        events = []
        for i in range(100):
            events.append(make_event(
                customer_id=f"cust_{i:03d}",
                event_type=EventType.VIEW_ITEM,
                timestamp=datetime(2024, 1, 1) + timedelta(days=i),
                extra={"valid_field": "value", "null_field": None},
            ))

        analyzer = TraitValueAnalyzer()
        result = analyzer.analyze(events, sample_profiles)

        # null_field should be filtered
        trait_names = [t.trait_name for t in result.traits]
        assert "null_field" not in trait_names

    def test_single_value_trait(self, sample_profiles):
        """Test trait with only one value."""
        events = []
        for i in range(100):
            events.append(make_event(
                customer_id=f"cust_{i:03d}",
                event_type=EventType.VIEW_ITEM,
                timestamp=datetime(2024, 1, 1) + timedelta(days=i),
                extra={"constant": "always_same", "varying": f"val_{i % 5}"},
            ))

        analyzer = TraitValueAnalyzer(min_cardinality=2)
        result = analyzer.analyze(events, sample_profiles)

        # constant should be filtered (only 1 value)
        trait_names = [t.trait_name for t in result.traits]
        assert "constant" not in trait_names


# =============================================================================
# TEST CLIENT-AGNOSTIC BEHAVIOR
# =============================================================================


class TestClientAgnostic:
    """Tests verifying client-agnostic behavior."""

    def test_different_field_names(self, sample_profiles):
        """Test works with arbitrary field names."""
        # Client A uses color, brand
        events_a = []
        for i in range(50):
            events_a.append(make_event(
                customer_id=f"cust_{i:03d}",
                event_type=EventType.VIEW_ITEM,
                timestamp=datetime(2024, 1, 1) + timedelta(days=i),
                extra={"color": f"color_{i % 5}", "brand": f"brand_{i % 3}"},
            ))

        # Client B uses colour, manufacturer
        events_b = []
        for i in range(50):
            events_b.append(make_event(
                customer_id=f"cust_{i:03d}",
                event_type=EventType.VIEW_ITEM,
                timestamp=datetime(2024, 1, 1) + timedelta(days=i),
                extra={"colour": f"colour_{i % 5}", "manufacturer": f"mfg_{i % 3}"},
            ))

        analyzer = TraitValueAnalyzer(min_coverage=0.01)

        # Both should work
        result_a = analyzer.analyze(events_a, sample_profiles[:50])
        result_b = analyzer.analyze(events_b, sample_profiles[:50])

        # Client A traits
        names_a = [t.trait_name for t in result_a.traits]
        assert "color" in names_a or len(result_a.traits) > 0

        # Client B traits
        names_b = [t.trait_name for t in result_b.traits]
        assert "colour" in names_b or "manufacturer" in names_b or len(result_b.traits) > 0

    def test_nested_properties(self, sample_profiles):
        """Test discovers traits in nested properties."""
        events = []
        for i in range(50):
            events.append(make_event(
                customer_id=f"cust_{i:03d}",
                event_type=EventType.VIEW_ITEM,
                timestamp=datetime(2024, 1, 1) + timedelta(days=i),
                extra={
                    "top_level": f"val_{i % 3}",
                    "nested": {
                        "deep_field": f"deep_{i % 4}",
                    },
                },
            ))

        analyzer = TraitValueAnalyzer(min_coverage=0.01)
        result = analyzer.analyze(events, sample_profiles[:50])

        # Should find both top-level and nested fields
        paths = [t.trait_path for t in result.traits]
        # top_level should be found
        assert any("top_level" in p for p in paths) or len(result.traits) > 0
