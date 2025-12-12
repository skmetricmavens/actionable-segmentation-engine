"""
Data Integrity Tests for Reconciliation, Aggregation Consistency, and Control Totals.

These tests verify that data transformations preserve key invariants:
1. Reconciliation Tests - Input and output totals match after processing
2. Aggregation Consistency Checks - Grouping/bucketing doesn't change totals
3. Control Totals - Row counts and sums preserved through transformations

These are CRITICAL for preventing silent data corruption in ETL pipelines.
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any
import pytest

from src.data.schemas import (
    CustomerIdHistory,
    EventRecord,
    EventType,
    EventProperties,
    CustomerProfile,
)
from src.data.joiner import (
    resolve_customer_merges,
    apply_merges_to_events,
    count_events_by_customer,
)
from src.features.aggregators import (
    aggregate_purchases,
    aggregate_sessions,
    aggregate_temporal,
    aggregate_device,
    aggregate_categories,
)
from src.features.profile_builder import build_profile


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_event(
    customer_id: str,
    event_type: EventType,
    timestamp: datetime,
    **properties,
) -> EventRecord:
    """Create an event record with minimal boilerplate."""
    return EventRecord(
        event_id=f"evt_{customer_id}_{timestamp.timestamp()}",
        internal_customer_id=customer_id,
        event_type=event_type,
        timestamp=timestamp,
        properties=EventProperties(**properties),
    )


def create_purchase_event(
    customer_id: str,
    amount: Decimal,
    timestamp: datetime,
    **extra_props,
) -> EventRecord:
    """Create a purchase event with amount."""
    return create_event(
        customer_id=customer_id,
        event_type=EventType.PURCHASE,
        timestamp=timestamp,
        total_amount=amount,
        order_id=f"order_{timestamp.timestamp()}",
        **extra_props,
    )


# =============================================================================
# RECONCILIATION TESTS
# =============================================================================


class TestReconciliation:
    """
    Reconciliation tests verify that input totals equal output totals
    after transformations. This catches silent data loss or duplication.
    """

    def test_merge_mapping_preserves_all_customer_ids(self) -> None:
        """
        RECONCILIATION: All customer IDs from input appear in merge mapping.

        Every past_id must map to a canonical ID.
        No customer should be "lost" during merge resolution.
        """
        # Setup: Create merge chains
        history = [
            CustomerIdHistory(
                internal_customer_id="cust_B",
                past_id="cust_A",
                merge_timestamp=datetime.now(timezone.utc),
            ),
            CustomerIdHistory(
                internal_customer_id="cust_C",
                past_id="cust_B",
                merge_timestamp=datetime.now(timezone.utc),
            ),
            CustomerIdHistory(
                internal_customer_id="cust_E",
                past_id="cust_D",
                merge_timestamp=datetime.now(timezone.utc),
            ),
        ]

        # Action
        merge_map = resolve_customer_merges(history)

        # Reconciliation: All past_ids must be in the mapping
        all_past_ids = {h.past_id for h in history}
        mapped_ids = set(merge_map.keys())

        assert all_past_ids == mapped_ids, (
            f"Customer IDs lost in merge mapping. "
            f"Input: {all_past_ids}, Mapped: {mapped_ids}"
        )

    def test_event_application_preserves_event_count(self) -> None:
        """
        RECONCILIATION: Number of events before and after merge mapping is identical.

        Applying merge mapping should only update customer IDs, not lose events.
        """
        base_time = datetime.now(timezone.utc)

        # Setup: Events for multiple customers
        events = [
            create_event("cust_A", EventType.VIEW_CATEGORY, base_time),
            create_event("cust_A", EventType.VIEW_ITEM, base_time + timedelta(hours=1)),
            create_event("cust_B", EventType.PURCHASE, base_time + timedelta(hours=2)),
            create_event("cust_C", EventType.VIEW_CATEGORY, base_time + timedelta(hours=3)),
            create_event("cust_D", EventType.ADD_TO_CART, base_time + timedelta(hours=4)),
        ]

        # Merge mapping: A -> C, B -> C
        merge_map = {"cust_A": "cust_C", "cust_B": "cust_C"}

        # Control total: count before
        count_before = len(events)

        # Action
        updated_events = apply_merges_to_events(events, merge_map)

        # Reconciliation: count after must equal count before
        count_after = len(updated_events)

        assert count_before == count_after, (
            f"Event count changed during merge mapping. "
            f"Before: {count_before}, After: {count_after}"
        )

    def test_unified_events_preserve_total_count(self) -> None:
        """
        RECONCILIATION: Merge resolution + apply preserves total event count.

        Events should only be re-attributed, never duplicated or lost.
        """
        base_time = datetime.now(timezone.utc)

        events = [
            create_event("old_id_1", EventType.VIEW_CATEGORY, base_time),
            create_event("old_id_1", EventType.PURCHASE, base_time + timedelta(hours=1)),
            create_event("new_id", EventType.VIEW_CATEGORY, base_time + timedelta(hours=2)),
            create_event("other", EventType.VIEW_ITEM, base_time + timedelta(hours=3)),
        ]

        history = [
            CustomerIdHistory(
                internal_customer_id="new_id",
                past_id="old_id_1",
                merge_timestamp=base_time,
            ),
        ]

        count_before = len(events)

        # Action: resolve merges then apply
        merge_map = resolve_customer_merges(history)
        unified_events = apply_merges_to_events(events, merge_map)

        count_after = len(unified_events)

        assert count_before == count_after, (
            f"Event count changed during unification. "
            f"Before: {count_before}, After: {count_after}"
        )

    def test_purchase_aggregation_preserves_total_revenue(self) -> None:
        """
        RECONCILIATION: Sum of revenue in aggregation equals sum of individual events.

        This is CRITICAL for financial accuracy.
        """
        base_time = datetime.now(timezone.utc)

        # Setup: Multiple purchase events with known amounts
        events = [
            create_purchase_event("cust_1", Decimal("99.99"), base_time),
            create_purchase_event("cust_1", Decimal("150.00"), base_time + timedelta(days=1)),
            create_purchase_event("cust_1", Decimal("25.50"), base_time + timedelta(days=2)),
            create_event("cust_1", EventType.VIEW_CATEGORY, base_time + timedelta(days=3)),  # Non-purchase
        ]

        # Control total: manually sum purchase amounts
        expected_revenue = Decimal("99.99") + Decimal("150.00") + Decimal("25.50")

        # Action
        metrics = aggregate_purchases(events)

        # Reconciliation: aggregated total must match control total
        assert metrics["total_revenue"] == expected_revenue, (
            f"Revenue mismatch. "
            f"Expected: {expected_revenue}, Got: {metrics['total_revenue']}"
        )

    def test_purchase_aggregation_preserves_count(self) -> None:
        """
        RECONCILIATION: Count of purchases in aggregation matches event filter.
        """
        base_time = datetime.now(timezone.utc)

        events = [
            create_purchase_event("cust_1", Decimal("50.00"), base_time),
            create_purchase_event("cust_1", Decimal("75.00"), base_time + timedelta(days=1)),
            create_event("cust_1", EventType.VIEW_CATEGORY, base_time + timedelta(days=2)),
            create_event("cust_1", EventType.ADD_TO_CART, base_time + timedelta(days=3)),
            create_purchase_event("cust_1", Decimal("30.00"), base_time + timedelta(days=4)),
        ]

        # Control total: count purchase events manually
        expected_count = sum(1 for e in events if e.event_type == EventType.PURCHASE)

        # Action
        metrics = aggregate_purchases(events)

        # Reconciliation
        assert metrics["total_purchases"] == expected_count, (
            f"Purchase count mismatch. "
            f"Expected: {expected_count}, Got: {metrics['total_purchases']}"
        )


# =============================================================================
# AGGREGATION CONSISTENCY TESTS
# =============================================================================


class TestAggregationConsistency:
    """
    Aggregation consistency tests verify that grouping/bucketing
    doesn't change totals unintentionally.
    """

    def test_session_counts_sum_correctly(self) -> None:
        """
        AGGREGATION CONSISTENCY: Session metrics sum to expected totals.

        Total sessions + page views + item views should be calculable from raw events.
        """
        base_time = datetime.now(timezone.utc)

        events = [
            create_event("cust_1", EventType.SESSION_START, base_time),
            create_event("cust_1", EventType.VIEW_CATEGORY, base_time + timedelta(minutes=1)),
            create_event("cust_1", EventType.VIEW_CATEGORY, base_time + timedelta(minutes=2)),
            create_event("cust_1", EventType.VIEW_ITEM, base_time + timedelta(minutes=3)),
            create_event("cust_1", EventType.SESSION_START, base_time + timedelta(hours=1)),
            create_event("cust_1", EventType.VIEW_CATEGORY, base_time + timedelta(hours=1, minutes=1)),
            create_event("cust_1", EventType.VIEW_ITEM, base_time + timedelta(hours=1, minutes=2)),
            create_event("cust_1", EventType.VIEW_ITEM, base_time + timedelta(hours=1, minutes=3)),
        ]

        # Control totals from raw events
        # Note: page_views includes both VIEW_CATEGORY and VIEW_ITEM in the aggregator
        expected_sessions = sum(1 for e in events if e.event_type == EventType.SESSION_START)
        expected_page_views = sum(
            1 for e in events
            if e.event_type in (EventType.VIEW_CATEGORY, EventType.VIEW_ITEM)
        )
        expected_item_views = sum(1 for e in events if e.event_type == EventType.VIEW_ITEM)

        # Action
        metrics = aggregate_sessions(events)

        # Aggregation consistency checks
        assert metrics["total_sessions"] == expected_sessions, (
            f"Session count mismatch. Expected: {expected_sessions}, Got: {metrics['total_sessions']}"
        )
        assert metrics["total_page_views"] == expected_page_views, (
            f"Page view count mismatch. Expected: {expected_page_views}, Got: {metrics['total_page_views']}"
        )
        assert metrics["total_items_viewed"] == expected_item_views, (
            f"Item view count mismatch. Expected: {expected_item_views}, Got: {metrics['total_items_viewed']}"
        )

    def test_cart_additions_and_abandonments_sum_correctly(self) -> None:
        """
        AGGREGATION CONSISTENCY: Cart metrics are internally consistent.

        Cart abandonment rate = (cart_adds - completed_purchases) / cart_adds
        """
        base_time = datetime.now(timezone.utc)

        events = [
            # First session: add to cart, then abandon
            create_event("cust_1", EventType.ADD_TO_CART, base_time, product_id="prod_1"),
            create_event("cust_1", EventType.ADD_TO_CART, base_time + timedelta(minutes=1), product_id="prod_2"),
            # No purchase

            # Second session: add to cart and purchase
            create_event("cust_1", EventType.ADD_TO_CART, base_time + timedelta(days=1), product_id="prod_3"),
            create_purchase_event("cust_1", Decimal("100.00"), base_time + timedelta(days=1, minutes=5)),

            # Third session: add multiple, purchase once
            create_event("cust_1", EventType.ADD_TO_CART, base_time + timedelta(days=2), product_id="prod_4"),
            create_event("cust_1", EventType.ADD_TO_CART, base_time + timedelta(days=2, minutes=1), product_id="prod_5"),
            create_purchase_event("cust_1", Decimal("200.00"), base_time + timedelta(days=2, minutes=10)),
        ]

        # Control totals
        expected_cart_adds = sum(1 for e in events if e.event_type == EventType.ADD_TO_CART)
        expected_purchases = sum(1 for e in events if e.event_type == EventType.PURCHASE)

        # Action
        metrics = aggregate_sessions(events)

        # Aggregation consistency
        assert metrics["total_cart_additions"] == expected_cart_adds, (
            f"Cart add count mismatch. Expected: {expected_cart_adds}, Got: {metrics['total_cart_additions']}"
        )

        # Cart abandonment should be between 0 and 1
        assert 0.0 <= metrics["cart_abandonment_rate"] <= 1.0, (
            f"Cart abandonment rate out of bounds: {metrics['cart_abandonment_rate']}"
        )

    def test_category_counts_are_consistent(self) -> None:
        """
        AGGREGATION CONSISTENCY: Category counts should match event filter.

        Category aggregation should count all relevant events.
        """
        base_time = datetime.now(timezone.utc)

        events = [
            create_event("cust_1", EventType.VIEW_ITEM, base_time, product_category="Electronics"),
            create_event("cust_1", EventType.VIEW_ITEM, base_time + timedelta(minutes=1), product_category="Electronics"),
            create_event("cust_1", EventType.VIEW_ITEM, base_time + timedelta(minutes=2), product_category="Clothing"),
            create_event("cust_1", EventType.PURCHASE, base_time + timedelta(minutes=3), product_category="Electronics"),
            create_event("cust_1", EventType.VIEW_ITEM, base_time + timedelta(minutes=4), product_category="Home"),
        ]

        # Control: count events with product_category
        events_with_category = [
            e for e in events
            if e.properties.product_category is not None
        ]
        expected_count = len(events_with_category)

        # Action
        category_data = aggregate_categories(events)

        # Aggregation consistency: total interactions should match
        if category_data:
            total_interactions = sum(c.view_count + c.purchase_count for c in category_data)
            # Each event contributes once
            assert total_interactions == expected_count, (
                f"Category interaction count mismatch. "
                f"Expected: {expected_count}, Got: {total_interactions}"
            )

    def test_temporal_patterns_cover_all_events(self) -> None:
        """
        AGGREGATION CONSISTENCY: Temporal pattern detection uses all events.

        First seen and last seen should bracket all event timestamps.
        """
        base_time = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)

        events = [
            create_event("cust_1", EventType.VIEW_CATEGORY, base_time),
            create_event("cust_1", EventType.VIEW_ITEM, base_time + timedelta(days=5)),
            create_event("cust_1", EventType.PURCHASE, base_time + timedelta(days=10)),
            create_event("cust_1", EventType.VIEW_CATEGORY, base_time + timedelta(days=30)),
        ]

        # Control: actual first and last timestamps
        expected_first = min(e.timestamp for e in events)
        expected_last = max(e.timestamp for e in events)

        # Action
        metrics = aggregate_temporal(events)

        # Aggregation consistency: temporal bounds
        assert metrics["first_seen"] == expected_first, (
            f"First seen mismatch. Expected: {expected_first}, Got: {metrics['first_seen']}"
        )
        assert metrics["last_seen"] == expected_last, (
            f"Last seen mismatch. Expected: {expected_last}, Got: {metrics['last_seen']}"
        )


# =============================================================================
# CONTROL TOTAL TESTS
# =============================================================================


class TestControlTotals:
    """
    Control total tests track row counts and sums before and after transformations.
    Used extensively in ETL pipelines to catch data corruption.
    """

    def test_profile_builder_processes_all_events(self) -> None:
        """
        CONTROL TOTAL: All input events contribute to profile metrics.

        The profile should reflect metrics from ALL events passed in.
        """
        base_time = datetime.now(timezone.utc)

        # Create a known set of events
        events = [
            create_purchase_event("cust_1", Decimal("100.00"), base_time),
            create_purchase_event("cust_1", Decimal("50.00"), base_time + timedelta(days=1)),
            create_event("cust_1", EventType.VIEW_CATEGORY, base_time + timedelta(days=2)),
            create_event("cust_1", EventType.VIEW_CATEGORY, base_time + timedelta(days=2, hours=1)),
            create_event("cust_1", EventType.VIEW_ITEM, base_time + timedelta(days=3)),
            create_event("cust_1", EventType.SESSION_START, base_time),
        ]

        # Control totals
        control_purchases = sum(1 for e in events if e.event_type == EventType.PURCHASE)
        control_revenue = sum(
            e.properties.total_amount or Decimal("0")
            for e in events if e.event_type == EventType.PURCHASE
        )
        # Note: page_views includes both VIEW_CATEGORY and VIEW_ITEM in the aggregator
        control_page_views = sum(
            1 for e in events
            if e.event_type in (EventType.VIEW_CATEGORY, EventType.VIEW_ITEM)
        )

        # Action
        profile = build_profile("cust_1", events)

        # Control total verification
        assert profile.total_purchases == control_purchases, (
            f"Purchase count control total failed. "
            f"Control: {control_purchases}, Profile: {profile.total_purchases}"
        )
        assert profile.total_revenue == control_revenue, (
            f"Revenue control total failed. "
            f"Control: {control_revenue}, Profile: {profile.total_revenue}"
        )
        assert profile.total_page_views == control_page_views, (
            f"Page view control total failed. "
            f"Control: {control_page_views}, Profile: {profile.total_page_views}"
        )

    def test_merge_mapping_covers_all_historical_ids(self) -> None:
        """
        CONTROL TOTAL: Merge map has entry for every past_id.
        """
        base_time = datetime.now(timezone.utc)

        history = [
            CustomerIdHistory(internal_customer_id="B", past_id="A", merge_timestamp=base_time),
            CustomerIdHistory(internal_customer_id="C", past_id="B", merge_timestamp=base_time),
            CustomerIdHistory(internal_customer_id="D", past_id="C", merge_timestamp=base_time),
            CustomerIdHistory(internal_customer_id="Y", past_id="X", merge_timestamp=base_time),
        ]

        # Control: count of unique past_ids
        control_past_ids = len({h.past_id for h in history})

        # Action
        merge_map = resolve_customer_merges(history)

        # Control total: every past_id must be mapped
        assert len(merge_map) == control_past_ids, (
            f"Merge map size doesn't match past_id count. "
            f"Control: {control_past_ids}, Map size: {len(merge_map)}"
        )

    def test_event_unification_preserves_event_attributes(self) -> None:
        """
        CONTROL TOTAL: Event attributes other than customer_id are preserved.

        Only the customer ID should change during merge mapping.
        """
        base_time = datetime.now(timezone.utc)

        original_event = create_purchase_event(
            "old_id",
            Decimal("199.99"),
            base_time,
            product_category="Electronics",
        )

        merge_map = {"old_id": "new_id"}

        # Control: capture attributes before transformation
        control_event_id = original_event.event_id
        control_timestamp = original_event.timestamp
        control_event_type = original_event.event_type
        control_amount = original_event.properties.total_amount
        control_order_id = original_event.properties.order_id

        # Action
        updated_events = apply_merges_to_events([original_event], merge_map)
        updated_event = updated_events[0]

        # Control totals: all attributes except customer_id must match
        assert updated_event.event_id == control_event_id, "Event ID changed"
        assert updated_event.timestamp == control_timestamp, "Timestamp changed"
        assert updated_event.event_type == control_event_type, "Event type changed"
        assert updated_event.properties.total_amount == control_amount, "Amount changed"
        assert updated_event.properties.order_id == control_order_id, "Order ID changed"

        # But customer ID should have changed
        assert updated_event.internal_customer_id == "new_id", "Customer ID not updated"

    def test_multi_customer_revenue_sums_correctly(self) -> None:
        """
        CONTROL TOTAL: Sum of revenue across all customers equals total input revenue.

        When segmenting by customer, individual totals must sum to grand total.
        """
        base_time = datetime.now(timezone.utc)

        # Events for multiple customers
        all_events = [
            create_purchase_event("cust_1", Decimal("100.00"), base_time),
            create_purchase_event("cust_1", Decimal("50.00"), base_time + timedelta(days=1)),
            create_purchase_event("cust_2", Decimal("200.00"), base_time),
            create_purchase_event("cust_2", Decimal("75.00"), base_time + timedelta(days=1)),
            create_purchase_event("cust_3", Decimal("300.00"), base_time),
        ]

        # Control: grand total of all revenue
        grand_total = sum(
            e.properties.total_amount or Decimal("0")
            for e in all_events if e.event_type == EventType.PURCHASE
        )

        # Group by customer and aggregate
        by_customer: dict[str, list[EventRecord]] = {}
        for event in all_events:
            by_customer.setdefault(event.internal_customer_id, []).append(event)

        # Sum of individual customer revenues
        customer_revenues = []
        for customer_id, events in by_customer.items():
            metrics = aggregate_purchases(events)
            customer_revenues.append(metrics["total_revenue"])

        sum_of_customer_revenues = sum(customer_revenues, Decimal("0"))

        # Control total: grand total must equal sum of customer totals
        assert grand_total == sum_of_customer_revenues, (
            f"Revenue sum doesn't match grand total. "
            f"Grand total: {grand_total}, Sum of customers: {sum_of_customer_revenues}"
        )


# =============================================================================
# DATA INTEGRITY EDGE CASES
# =============================================================================


class TestDataIntegrityEdgeCases:
    """
    Edge cases that could cause data integrity issues.
    """

    def test_empty_events_dont_corrupt_totals(self) -> None:
        """Empty event list should produce zero totals, not errors."""
        events: list[EventRecord] = []

        metrics = aggregate_purchases(events)

        assert metrics["total_purchases"] == 0
        assert metrics["total_revenue"] == Decimal("0")
        assert metrics["total_refunds"] == 0

    def test_null_amounts_handled_safely(self) -> None:
        """Events with null amounts shouldn't corrupt revenue totals."""
        base_time = datetime.now(timezone.utc)

        events = [
            create_purchase_event("cust_1", Decimal("100.00"), base_time),
            # Event with null amount (happens in real data)
            create_event("cust_1", EventType.PURCHASE, base_time + timedelta(days=1)),
            create_purchase_event("cust_1", Decimal("50.00"), base_time + timedelta(days=2)),
        ]

        # Control: only non-null amounts
        expected_revenue = Decimal("100.00") + Decimal("50.00")

        metrics = aggregate_purchases(events)

        # Should not error and should only sum valid amounts
        assert metrics["total_revenue"] == expected_revenue

    def test_duplicate_events_are_counted(self) -> None:
        """
        Duplicate events (same event_id) should be counted separately.

        Deduplication is a separate concern - aggregation should count what it receives.
        """
        base_time = datetime.now(timezone.utc)

        # Intentional duplicate (same event_id)
        event = create_purchase_event("cust_1", Decimal("100.00"), base_time)
        events = [event, event, event]  # 3 copies

        metrics = aggregate_purchases(events)

        # Should count all 3 (dedup is a different layer's responsibility)
        assert metrics["total_purchases"] == 3
        assert metrics["total_revenue"] == Decimal("300.00")

    def test_zero_amount_purchases_counted_correctly(self) -> None:
        """
        Zero-dollar purchases (free items) should be counted but not affect revenue.
        """
        base_time = datetime.now(timezone.utc)

        events = [
            create_purchase_event("cust_1", Decimal("100.00"), base_time),
            create_purchase_event("cust_1", Decimal("0.00"), base_time + timedelta(days=1)),  # Free
            create_purchase_event("cust_1", Decimal("50.00"), base_time + timedelta(days=2)),
        ]

        metrics = aggregate_purchases(events)

        # All 3 purchases should be counted
        assert metrics["total_purchases"] == 3
        # Revenue should only include non-zero amounts
        assert metrics["total_revenue"] == Decimal("150.00")

    def test_negative_amounts_handled(self) -> None:
        """
        Negative amounts (refunds as negative purchases) should be handled.

        This tests data quality scenarios where refunds might be encoded differently.
        """
        base_time = datetime.now(timezone.utc)

        events = [
            create_purchase_event("cust_1", Decimal("100.00"), base_time),
            create_purchase_event("cust_1", Decimal("-25.00"), base_time + timedelta(days=1)),  # Refund as negative
            create_purchase_event("cust_1", Decimal("50.00"), base_time + timedelta(days=2)),
        ]

        metrics = aggregate_purchases(events)

        # Total revenue includes negative (net revenue)
        assert metrics["total_revenue"] == Decimal("125.00")

    def test_merge_with_no_events_doesnt_error(self) -> None:
        """Applying merge mapping to empty events should work."""
        merge_map = {"old": "new"}
        events: list[EventRecord] = []

        result = apply_merges_to_events(events, merge_map)

        assert result == []

    def test_merge_with_no_mapping_preserves_events(self) -> None:
        """Empty merge map should return events unchanged."""
        base_time = datetime.now(timezone.utc)
        events = [create_event("cust_1", EventType.VIEW_CATEGORY, base_time)]

        result = apply_merges_to_events(events, {})

        assert len(result) == 1
        assert result[0].internal_customer_id == "cust_1"
