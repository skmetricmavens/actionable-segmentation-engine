"""
Tests for behavioral and transactional aggregation functions.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.data.schemas import (
    EventProperties,
    EventRecord,
    EventType,
)
from src.features.aggregators import (
    DeviceMetrics,
    PurchaseIntervalMetrics,
    PurchaseMetrics,
    SessionMetrics,
    TemporalMetrics,
    aggregate_categories,
    aggregate_device,
    aggregate_purchase_intervals,
    aggregate_purchases,
    aggregate_sessions,
    aggregate_temporal,
    calculate_churn_risk,
    calculate_clv_estimate,
)


# =============================================================================
# FIXTURES
# =============================================================================


def make_event(
    event_id: str,
    customer_id: str,
    event_type: EventType,
    timestamp: datetime | None = None,
    total_amount: Decimal | None = None,
    product_category: str | None = None,
    device_type: str | None = None,
) -> EventRecord:
    """Helper to create EventRecord."""
    if timestamp is None:
        timestamp = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)

    properties = EventProperties(
        total_amount=total_amount,
        product_category=product_category,
        device_type=device_type,
    )

    return EventRecord(
        event_id=event_id,
        internal_customer_id=customer_id,
        event_type=event_type,
        timestamp=timestamp,
        properties=properties,
    )


# =============================================================================
# PURCHASE AGGREGATION TESTS
# =============================================================================


class TestAggregatePurchases:
    """Tests for aggregate_purchases function."""

    def test_empty_events(self) -> None:
        """Test with empty events list."""
        result = aggregate_purchases([])
        assert result["total_purchases"] == 0
        assert result["total_revenue"] == Decimal("0")
        assert result["avg_order_value"] == Decimal("0")

    def test_single_purchase(self) -> None:
        """Test with single purchase event."""
        purchase_date = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        ref_date = datetime(2024, 1, 20, 10, 0, tzinfo=timezone.utc)
        events = [
            make_event("e1", "cust_1", EventType.PURCHASE, timestamp=purchase_date, total_amount=Decimal("100.00"))
        ]

        result = aggregate_purchases(events, reference_date=ref_date)

        assert result["total_purchases"] == 1
        assert result["total_revenue"] == Decimal("100.00")
        assert result["avg_order_value"] == Decimal("100.00")
        assert result["days_since_last_purchase"] == 5

    def test_multiple_purchases(self) -> None:
        """Test with multiple purchase events."""
        events = [
            make_event(
                "e1", "cust_1", EventType.PURCHASE,
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                total_amount=Decimal("50.00"),
            ),
            make_event(
                "e2", "cust_1", EventType.PURCHASE,
                timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
                total_amount=Decimal("100.00"),
            ),
            make_event(
                "e3", "cust_1", EventType.PURCHASE,
                timestamp=datetime(2024, 1, 31, tzinfo=timezone.utc),
                total_amount=Decimal("150.00"),
            ),
        ]
        ref_date = datetime(2024, 2, 15, tzinfo=timezone.utc)

        result = aggregate_purchases(events, reference_date=ref_date)

        assert result["total_purchases"] == 3
        assert result["total_revenue"] == Decimal("300.00")
        assert result["avg_order_value"] == Decimal("100.00")

    def test_purchase_frequency(self) -> None:
        """Test purchase frequency calculation."""
        events = [
            make_event(
                "e1", "cust_1", EventType.PURCHASE,
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                total_amount=Decimal("50.00"),
            ),
            make_event(
                "e2", "cust_1", EventType.PURCHASE,
                timestamp=datetime(2024, 2, 1, tzinfo=timezone.utc),
                total_amount=Decimal("50.00"),
            ),
        ]

        result = aggregate_purchases(events)

        # 2 purchases over ~31 days = ~2/month
        assert result["purchase_frequency_per_month"] > 1.5

    def test_refund_rate(self) -> None:
        """Test refund rate calculation."""
        events = [
            make_event("e1", "cust_1", EventType.PURCHASE, total_amount=Decimal("100.00")),
            make_event("e2", "cust_1", EventType.PURCHASE, total_amount=Decimal("100.00")),
            make_event("e3", "cust_1", EventType.REFUND),
        ]

        result = aggregate_purchases(events)

        assert result["total_refunds"] == 1
        assert result["refund_rate"] == 0.5  # 1 refund / 2 purchases

    def test_no_purchases_with_other_events(self) -> None:
        """Test with only non-purchase events."""
        events = [
            make_event("e1", "cust_1", EventType.VIEW_ITEM),
            make_event("e2", "cust_1", EventType.ADD_TO_CART),
        ]

        result = aggregate_purchases(events)

        assert result["total_purchases"] == 0
        assert result["days_since_last_purchase"] is None


# =============================================================================
# SESSION AGGREGATION TESTS
# =============================================================================


class TestAggregateSessions:
    """Tests for aggregate_sessions function."""

    def test_empty_events(self) -> None:
        """Test with empty events list."""
        result = aggregate_sessions([])
        assert result["total_sessions"] == 0
        assert result["total_page_views"] == 0

    def test_single_session(self) -> None:
        """Test single session with various events."""
        events = [
            make_event("e1", "cust_1", EventType.SESSION_START),
            make_event("e2", "cust_1", EventType.VIEW_CATEGORY),
            make_event("e3", "cust_1", EventType.VIEW_ITEM),
            make_event("e4", "cust_1", EventType.VIEW_ITEM),
        ]

        result = aggregate_sessions(events)

        assert result["total_sessions"] == 1
        assert result["total_page_views"] == 3  # 1 category + 2 items
        assert result["total_items_viewed"] == 2

    def test_cart_abandonment(self) -> None:
        """Test cart abandonment rate calculation."""
        events = [
            make_event("e1", "cust_1", EventType.ADD_TO_CART),
            make_event("e2", "cust_1", EventType.ADD_TO_CART),
            make_event("e3", "cust_1", EventType.PURCHASE),
        ]

        result = aggregate_sessions(events)

        assert result["total_cart_additions"] == 2
        # 1 purchase / 2 cart additions = 50% conversion = 50% abandonment
        assert result["cart_abandonment_rate"] == 0.5

    def test_no_cart_abandonment(self) -> None:
        """Test when all carts convert."""
        events = [
            make_event("e1", "cust_1", EventType.ADD_TO_CART),
            make_event("e2", "cust_1", EventType.PURCHASE),
        ]

        result = aggregate_sessions(events)

        assert result["cart_abandonment_rate"] == 0.0


# =============================================================================
# CATEGORY AGGREGATION TESTS
# =============================================================================


class TestAggregateCategories:
    """Tests for aggregate_categories function."""

    def test_empty_events(self) -> None:
        """Test with empty events list."""
        result = aggregate_categories([])
        assert result == []

    def test_single_category(self) -> None:
        """Test with single category."""
        events = [
            make_event("e1", "cust_1", EventType.VIEW_ITEM, product_category="Electronics"),
            make_event("e2", "cust_1", EventType.VIEW_ITEM, product_category="Electronics"),
        ]

        result = aggregate_categories(events)

        assert len(result) == 1
        assert result[0].category == "Electronics"
        assert result[0].engagement_score == 1.0  # Only category = max score
        assert result[0].view_count == 2

    def test_multiple_categories(self) -> None:
        """Test with multiple categories."""
        events = [
            make_event("e1", "cust_1", EventType.VIEW_ITEM, product_category="Electronics"),
            make_event("e2", "cust_1", EventType.VIEW_ITEM, product_category="Electronics"),
            make_event("e3", "cust_1", EventType.VIEW_ITEM, product_category="Books"),
        ]

        result = aggregate_categories(events)

        assert len(result) == 2
        # Electronics should be first (more views)
        assert result[0].category == "Electronics"
        assert result[0].engagement_score == 1.0
        assert result[1].category == "Books"
        assert result[1].engagement_score == 0.5  # 1/2 of max

    def test_purchase_weight(self) -> None:
        """Test that purchases add more weight."""
        events = [
            make_event("e1", "cust_1", EventType.VIEW_ITEM, product_category="Electronics"),
            make_event("e2", "cust_1", EventType.PURCHASE, product_category="Books"),
        ]

        result = aggregate_categories(events)

        # Books should be first due to purchase weight (5 points vs 1)
        assert result[0].category == "Books"
        assert result[0].purchase_count == 1
        assert result[1].category == "Electronics"

    def test_events_without_category(self) -> None:
        """Test that events without category are ignored."""
        events = [
            make_event("e1", "cust_1", EventType.VIEW_ITEM),  # No category
            make_event("e2", "cust_1", EventType.VIEW_ITEM, product_category="Books"),
        ]

        result = aggregate_categories(events)

        assert len(result) == 1
        assert result[0].category == "Books"


# =============================================================================
# TEMPORAL AGGREGATION TESTS
# =============================================================================


class TestAggregateTemporal:
    """Tests for aggregate_temporal function."""

    def test_empty_events(self) -> None:
        """Test with empty events list."""
        result = aggregate_temporal([])
        assert result["first_seen"] is None
        assert result["last_seen"] is None
        assert result["preferred_day_of_week"] is None

    def test_single_event(self) -> None:
        """Test with single event."""
        timestamp = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)  # Monday at 14:30
        events = [make_event("e1", "cust_1", EventType.VIEW_ITEM, timestamp=timestamp)]

        result = aggregate_temporal(events)

        assert result["first_seen"] == timestamp
        assert result["last_seen"] == timestamp
        assert result["preferred_day_of_week"] == 0  # Monday
        assert result["preferred_hour_of_day"] == 14

    def test_multiple_events_different_times(self) -> None:
        """Test with multiple events at different times."""
        events = [
            make_event(
                "e1", "cust_1", EventType.VIEW_ITEM,
                timestamp=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),  # Monday
            ),
            make_event(
                "e2", "cust_1", EventType.VIEW_ITEM,
                timestamp=datetime(2024, 1, 16, 10, 0, tzinfo=timezone.utc),  # Tuesday
            ),
            make_event(
                "e3", "cust_1", EventType.VIEW_ITEM,
                timestamp=datetime(2024, 1, 22, 10, 0, tzinfo=timezone.utc),  # Monday
            ),
            make_event(
                "e4", "cust_1", EventType.VIEW_ITEM,
                timestamp=datetime(2024, 1, 23, 10, 0, tzinfo=timezone.utc),  # Tuesday
            ),
            make_event(
                "e5", "cust_1", EventType.VIEW_ITEM,
                timestamp=datetime(2024, 1, 29, 10, 0, tzinfo=timezone.utc),  # Monday
            ),
        ]

        result = aggregate_temporal(events)

        # Monday (3 events) should be preferred over Tuesday (2 events)
        assert result["preferred_day_of_week"] == 0  # Monday

    def test_first_and_last_seen(self) -> None:
        """Test first_seen and last_seen calculation."""
        events = [
            make_event(
                "e1", "cust_1", EventType.VIEW_ITEM,
                timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            ),
            make_event(
                "e2", "cust_1", EventType.VIEW_ITEM,
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
            make_event(
                "e3", "cust_1", EventType.VIEW_ITEM,
                timestamp=datetime(2024, 1, 31, tzinfo=timezone.utc),
            ),
        ]

        result = aggregate_temporal(events)

        assert result["first_seen"] == datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert result["last_seen"] == datetime(2024, 1, 31, tzinfo=timezone.utc)


# =============================================================================
# DEVICE AGGREGATION TESTS
# =============================================================================


class TestAggregateDevice:
    """Tests for aggregate_device function."""

    def test_empty_events(self) -> None:
        """Test with empty events list."""
        result = aggregate_device([])
        assert result["primary_device_type"] is None
        assert result["mobile_session_ratio"] == 0.0

    def test_single_device(self) -> None:
        """Test with single device type."""
        events = [
            make_event("e1", "cust_1", EventType.VIEW_ITEM, device_type="desktop"),
            make_event("e2", "cust_1", EventType.VIEW_ITEM, device_type="desktop"),
        ]

        result = aggregate_device(events)

        assert result["primary_device_type"] == "desktop"
        assert result["mobile_session_ratio"] == 0.0

    def test_mobile_detection(self) -> None:
        """Test mobile device detection."""
        events = [
            make_event("e1", "cust_1", EventType.VIEW_ITEM, device_type="mobile"),
            make_event("e2", "cust_1", EventType.VIEW_ITEM, device_type="desktop"),
        ]

        result = aggregate_device(events)

        assert result["mobile_session_ratio"] == 0.5

    def test_multiple_device_types(self) -> None:
        """Test primary device determination with multiple types."""
        events = [
            make_event("e1", "cust_1", EventType.VIEW_ITEM, device_type="mobile"),
            make_event("e2", "cust_1", EventType.VIEW_ITEM, device_type="mobile"),
            make_event("e3", "cust_1", EventType.VIEW_ITEM, device_type="desktop"),
        ]

        result = aggregate_device(events)

        assert result["primary_device_type"] == "mobile"
        # 2 mobile / 3 total = 0.667
        assert round(result["mobile_session_ratio"], 2) == 0.67


# =============================================================================
# CLV CALCULATION TESTS
# =============================================================================


class TestCalculateClv:
    """Tests for calculate_clv_estimate function."""

    def test_no_purchases(self) -> None:
        """Test CLV with no purchases."""
        metrics = PurchaseMetrics(
            total_purchases=0,
            total_revenue=Decimal("0"),
            avg_order_value=Decimal("0"),
            total_refunds=0,
            refund_rate=0.0,
            days_since_last_purchase=None,
            purchase_frequency_per_month=0.0,
        )

        result = calculate_clv_estimate(metrics, customer_tenure_days=100)

        assert result == Decimal("0")

    def test_with_purchases(self) -> None:
        """Test CLV calculation with purchase history."""
        metrics = PurchaseMetrics(
            total_purchases=10,
            total_revenue=Decimal("1000"),
            avg_order_value=Decimal("100"),
            total_refunds=0,
            refund_rate=0.0,
            days_since_last_purchase=30,
            purchase_frequency_per_month=1.0,  # 1 purchase/month
        )

        result = calculate_clv_estimate(metrics, customer_tenure_days=365)

        # 1 purchase/month * $100 AOV * 36 months (3 years) with discounting
        assert result > Decimal("0")
        assert result < Decimal("4000")  # Should be less than undiscounted

    def test_zero_tenure(self) -> None:
        """Test CLV with zero tenure returns zero."""
        metrics = PurchaseMetrics(
            total_purchases=1,
            total_revenue=Decimal("100"),
            avg_order_value=Decimal("100"),
            total_refunds=0,
            refund_rate=0.0,
            days_since_last_purchase=0,
            purchase_frequency_per_month=1.0,
        )

        result = calculate_clv_estimate(metrics, customer_tenure_days=0)

        assert result == Decimal("0")


# =============================================================================
# CHURN RISK TESTS
# =============================================================================


class TestCalculateChurnRisk:
    """Tests for calculate_churn_risk function."""

    def test_active_customer_low_risk(self) -> None:
        """Test active customer has low churn risk."""
        purchase_metrics = PurchaseMetrics(
            total_purchases=5,
            total_revenue=Decimal("500"),
            avg_order_value=Decimal("100"),
            total_refunds=0,
            refund_rate=0.0,
            days_since_last_purchase=7,  # Recent purchase
            purchase_frequency_per_month=2.0,  # Frequent
        )
        session_metrics = SessionMetrics(
            total_sessions=10,
            total_page_views=50,
            total_items_viewed=30,
            total_cart_additions=5,
            cart_abandonment_rate=0.2,  # Low abandonment
        )

        risk = calculate_churn_risk(purchase_metrics, session_metrics, customer_tenure_days=180)

        assert risk < 0.3  # Low risk

    def test_dormant_customer_high_risk(self) -> None:
        """Test dormant customer has high churn risk."""
        purchase_metrics = PurchaseMetrics(
            total_purchases=1,
            total_revenue=Decimal("50"),
            avg_order_value=Decimal("50"),
            total_refunds=0,
            refund_rate=0.0,
            days_since_last_purchase=200,  # Long time ago
            purchase_frequency_per_month=0.1,  # Infrequent
        )
        session_metrics = SessionMetrics(
            total_sessions=0,
            total_page_views=0,
            total_items_viewed=0,
            total_cart_additions=0,
            cart_abandonment_rate=0.0,
        )

        risk = calculate_churn_risk(purchase_metrics, session_metrics, customer_tenure_days=365)

        assert risk > 0.7  # High risk

    def test_never_purchased_customer(self) -> None:
        """Test customer who never purchased has elevated risk."""
        purchase_metrics = PurchaseMetrics(
            total_purchases=0,
            total_revenue=Decimal("0"),
            avg_order_value=Decimal("0"),
            total_refunds=0,
            refund_rate=0.0,
            days_since_last_purchase=None,
            purchase_frequency_per_month=0.0,
        )
        session_metrics = SessionMetrics(
            total_sessions=5,
            total_page_views=20,
            total_items_viewed=10,
            total_cart_additions=2,
            cart_abandonment_rate=1.0,  # 100% abandonment
        )

        risk = calculate_churn_risk(purchase_metrics, session_metrics, customer_tenure_days=90)

        assert 0.4 < risk < 0.9  # Moderate to high risk

    def test_risk_bounded_0_to_1(self) -> None:
        """Test that risk is always between 0 and 1."""
        # Extreme high-risk profile
        purchase_metrics = PurchaseMetrics(
            total_purchases=0,
            total_revenue=Decimal("0"),
            avg_order_value=Decimal("0"),
            total_refunds=0,
            refund_rate=0.0,
            days_since_last_purchase=None,
            purchase_frequency_per_month=0.0,
        )
        session_metrics = SessionMetrics(
            total_sessions=0,
            total_page_views=0,
            total_items_viewed=0,
            total_cart_additions=10,
            cart_abandonment_rate=1.0,
        )

        risk = calculate_churn_risk(purchase_metrics, session_metrics, customer_tenure_days=365)

        assert 0.0 <= risk <= 1.0


# =============================================================================
# PURCHASE INTERVAL TESTS
# =============================================================================


class TestAggregatePurchaseIntervals:
    """Tests for aggregate_purchase_intervals function."""

    def test_empty_events(self) -> None:
        """Test with empty events list."""
        result = aggregate_purchase_intervals([])

        assert result["intervals"] == []
        assert result["interval_mean"] is None
        assert result["interval_std"] is None
        assert result["regularity_index"] is None

    def test_single_purchase(self) -> None:
        """Test with single purchase - can't calculate intervals."""
        events = [
            make_event("e1", "cust_1", EventType.PURCHASE,
                       timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc))
        ]

        result = aggregate_purchase_intervals(events)

        assert result["intervals"] == []
        assert result["interval_mean"] is None
        assert result["interval_cv"] is None

    def test_two_purchases(self) -> None:
        """Test with two purchases - single interval."""
        events = [
            make_event("e1", "cust_1", EventType.PURCHASE,
                       timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc)),
            make_event("e2", "cust_1", EventType.PURCHASE,
                       timestamp=datetime(2024, 1, 31, tzinfo=timezone.utc)),
        ]

        result = aggregate_purchase_intervals(events)

        assert len(result["intervals"]) == 1
        assert result["interval_mean"] == 30.0  # 30 days
        assert result["interval_min"] == 30.0
        assert result["interval_max"] == 30.0
        # With only 1 interval, can't calculate std/cv
        assert result["interval_std"] is None
        assert result["interval_cv"] is None

    def test_regular_purchases(self) -> None:
        """Test customer with regular purchase intervals."""
        # Purchase every 30 days consistently
        events = [
            make_event("e1", "cust_1", EventType.PURCHASE,
                       timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc)),
            make_event("e2", "cust_1", EventType.PURCHASE,
                       timestamp=datetime(2024, 1, 31, tzinfo=timezone.utc)),
            make_event("e3", "cust_1", EventType.PURCHASE,
                       timestamp=datetime(2024, 3, 1, tzinfo=timezone.utc)),
            make_event("e4", "cust_1", EventType.PURCHASE,
                       timestamp=datetime(2024, 3, 31, tzinfo=timezone.utc)),
        ]

        result = aggregate_purchase_intervals(events)

        assert len(result["intervals"]) == 3
        assert result["interval_mean"] is not None
        assert result["interval_std"] is not None
        assert result["interval_cv"] is not None
        # Regular customer should have high regularity index
        assert result["regularity_index"] is not None
        assert result["regularity_index"] > 0.5

    def test_irregular_purchases(self) -> None:
        """Test customer with irregular purchase intervals."""
        # Purchases at very different intervals: 7 days, 60 days, 3 days
        events = [
            make_event("e1", "cust_1", EventType.PURCHASE,
                       timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc)),
            make_event("e2", "cust_1", EventType.PURCHASE,
                       timestamp=datetime(2024, 1, 8, tzinfo=timezone.utc)),  # +7 days
            make_event("e3", "cust_1", EventType.PURCHASE,
                       timestamp=datetime(2024, 3, 8, tzinfo=timezone.utc)),  # +60 days
            make_event("e4", "cust_1", EventType.PURCHASE,
                       timestamp=datetime(2024, 3, 11, tzinfo=timezone.utc)),  # +3 days
        ]

        result = aggregate_purchase_intervals(events)

        assert len(result["intervals"]) == 3
        assert result["interval_cv"] is not None
        # Irregular customer should have high CV (std/mean)
        assert result["interval_cv"] > 0.5
        # And lower regularity index
        assert result["regularity_index"] is not None
        assert result["regularity_index"] < 0.7

    def test_ignores_non_purchase_events(self) -> None:
        """Test that non-purchase events are ignored."""
        events = [
            make_event("e1", "cust_1", EventType.VIEW_ITEM,
                       timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc)),
            make_event("e2", "cust_1", EventType.PURCHASE,
                       timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc)),
            make_event("e3", "cust_1", EventType.ADD_TO_CART,
                       timestamp=datetime(2024, 1, 20, tzinfo=timezone.utc)),
            make_event("e4", "cust_1", EventType.PURCHASE,
                       timestamp=datetime(2024, 2, 15, tzinfo=timezone.utc)),
        ]

        result = aggregate_purchase_intervals(events)

        # Should only consider 2 purchase events
        assert len(result["intervals"]) == 1
        assert result["interval_mean"] == 31.0  # Jan 15 to Feb 15

    def test_unsorted_events(self) -> None:
        """Test that events are sorted by timestamp."""
        # Events out of order
        events = [
            make_event("e3", "cust_1", EventType.PURCHASE,
                       timestamp=datetime(2024, 3, 1, tzinfo=timezone.utc)),
            make_event("e1", "cust_1", EventType.PURCHASE,
                       timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc)),
            make_event("e2", "cust_1", EventType.PURCHASE,
                       timestamp=datetime(2024, 2, 1, tzinfo=timezone.utc)),
        ]

        result = aggregate_purchase_intervals(events)

        # Should sort and calculate correctly: Jan→Feb, Feb→Mar
        assert len(result["intervals"]) == 2
        assert result["interval_min"] is not None
        assert result["interval_min"] >= 28  # Feb has fewer days

    def test_same_day_purchases(self) -> None:
        """Test purchases on the same day."""
        events = [
            make_event("e1", "cust_1", EventType.PURCHASE,
                       timestamp=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)),
            make_event("e2", "cust_1", EventType.PURCHASE,
                       timestamp=datetime(2024, 1, 15, 14, 0, tzinfo=timezone.utc)),
            make_event("e3", "cust_1", EventType.PURCHASE,
                       timestamp=datetime(2024, 1, 16, 10, 0, tzinfo=timezone.utc)),
        ]

        result = aggregate_purchase_intervals(events)

        # Intervals should be in days (including fractions)
        assert len(result["intervals"]) == 2
        # First interval is 4 hours = 4/24 = 0.167 days
        assert result["interval_min"] is not None
        assert result["interval_min"] < 1.0

    def test_regularity_index_bounds(self) -> None:
        """Test that regularity index is between 0 and 1."""
        # Very irregular: intervals of 1 day and 100 days
        events = [
            make_event("e1", "cust_1", EventType.PURCHASE,
                       timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc)),
            make_event("e2", "cust_1", EventType.PURCHASE,
                       timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc)),  # +1 day
            make_event("e3", "cust_1", EventType.PURCHASE,
                       timestamp=datetime(2024, 4, 11, tzinfo=timezone.utc)),  # +99 days
        ]

        result = aggregate_purchase_intervals(events)

        assert result["regularity_index"] is not None
        assert 0.0 <= result["regularity_index"] <= 1.0
