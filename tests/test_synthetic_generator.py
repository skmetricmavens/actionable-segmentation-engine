"""
Tests for synthetic data generator.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.data.schemas import EventType, SyntheticDataset
from src.data.synthetic_generator import (
    BEHAVIOR_WEIGHTS,
    PRODUCT_CATEGORIES,
    CustomerBehaviorType,
    SyntheticDataGenerator,
    dataset_statistics,
    generate_small_dataset,
    preview_dataset,
)


# =============================================================================
# GENERATOR INITIALIZATION TESTS
# =============================================================================


class TestGeneratorInitialization:
    """Tests for generator initialization and reproducibility."""

    def test_generator_initialization(self) -> None:
        """Test generator initializes with seed."""
        generator = SyntheticDataGenerator(seed=42)
        assert generator.seed == 42
        assert generator._event_counter == 0
        assert generator._customer_counter == 0

    def test_reproducibility_same_seed(self) -> None:
        """Test same seed produces same results."""
        gen1 = SyntheticDataGenerator(seed=42)
        gen2 = SyntheticDataGenerator(seed=42)

        date_range = (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 31, tzinfo=timezone.utc),
        )

        ds1 = gen1.generate_dataset(n_customers=50, date_range=date_range)
        ds2 = gen2.generate_dataset(n_customers=50, date_range=date_range)

        assert ds1.n_events == ds2.n_events
        assert ds1.n_merges == ds2.n_merges
        assert len(ds1.events) == len(ds2.events)

        # Check first few events are identical
        for e1, e2 in zip(ds1.events[:10], ds2.events[:10], strict=True):
            assert e1.event_id == e2.event_id
            assert e1.internal_customer_id == e2.internal_customer_id
            assert e1.event_type == e2.event_type

    def test_different_seeds_different_results(self) -> None:
        """Test different seeds produce different results."""
        gen1 = SyntheticDataGenerator(seed=42)
        gen2 = SyntheticDataGenerator(seed=123)

        date_range = (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 31, tzinfo=timezone.utc),
        )

        ds1 = gen1.generate_dataset(n_customers=50, date_range=date_range)
        ds2 = gen2.generate_dataset(n_customers=50, date_range=date_range)

        # Results should differ
        assert ds1.events[0].event_id != ds2.events[0].event_id


# =============================================================================
# ID GENERATION TESTS
# =============================================================================


class TestIdGeneration:
    """Tests for ID generation."""

    def test_customer_id_format(self) -> None:
        """Test customer ID has correct format."""
        generator = SyntheticDataGenerator(seed=42)
        cust_id = generator._generate_customer_id()
        assert cust_id.startswith("cust_")
        assert len(cust_id) > 10

    def test_event_id_format(self) -> None:
        """Test event ID has correct format."""
        generator = SyntheticDataGenerator(seed=42)
        event_id = generator._generate_event_id()
        assert event_id.startswith("evt_")
        assert len(event_id) > 10

    def test_customer_ids_unique(self) -> None:
        """Test generated customer IDs are unique."""
        generator = SyntheticDataGenerator(seed=42)
        ids = [generator._generate_customer_id() for _ in range(100)]
        assert len(ids) == len(set(ids))

    def test_event_ids_unique(self) -> None:
        """Test generated event IDs are unique."""
        generator = SyntheticDataGenerator(seed=42)
        ids = [generator._generate_event_id() for _ in range(100)]
        assert len(ids) == len(set(ids))


# =============================================================================
# DATASET GENERATION TESTS
# =============================================================================


class TestDatasetGeneration:
    """Tests for complete dataset generation."""

    def test_generate_dataset_structure(self) -> None:
        """Test generated dataset has correct structure."""
        generator = SyntheticDataGenerator(seed=42)
        date_range = (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 31, tzinfo=timezone.utc),
        )

        dataset = generator.generate_dataset(n_customers=100, date_range=date_range)

        assert isinstance(dataset, SyntheticDataset)
        assert dataset.seed == 42
        assert dataset.n_customers == 100
        assert dataset.date_range_start == date_range[0]
        assert dataset.date_range_end == date_range[1]
        assert len(dataset.events) > 0
        assert len(dataset.customer_properties) == 100
        assert dataset.n_events == len(dataset.events)

    def test_generate_dataset_has_events(self) -> None:
        """Test generated dataset has events."""
        generator = SyntheticDataGenerator(seed=42)
        date_range = (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 31, tzinfo=timezone.utc),
        )

        dataset = generator.generate_dataset(n_customers=100, date_range=date_range)

        assert len(dataset.events) > 0
        # Expect roughly 10-30 events per customer on average
        assert dataset.n_events > 100  # At least some events

    def test_generate_dataset_has_merges(self) -> None:
        """Test generated dataset has merge history."""
        generator = SyntheticDataGenerator(seed=42)
        date_range = (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 31, tzinfo=timezone.utc),
        )

        dataset = generator.generate_dataset(
            n_customers=100,
            date_range=date_range,
            merge_probability=0.2,
        )

        # With 20% merge probability, expect some merges
        assert len(dataset.id_history) > 0
        assert dataset.n_merges == len(dataset.id_history)

    def test_generate_dataset_customer_properties(self) -> None:
        """Test customer properties are valid."""
        generator = SyntheticDataGenerator(seed=42)
        date_range = (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 31, tzinfo=timezone.utc),
        )

        dataset = generator.generate_dataset(n_customers=50, date_range=date_range)

        for customer in dataset.customer_properties:
            assert customer.internal_customer_id
            assert customer.email
            assert "@example.com" in customer.email
            assert customer.first_name
            assert customer.last_name
            assert customer.country

    def test_event_timestamps_in_range(self) -> None:
        """Test event timestamps are within specified range."""
        generator = SyntheticDataGenerator(seed=42)
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 3, 31, tzinfo=timezone.utc)

        dataset = generator.generate_dataset(n_customers=50, date_range=(start, end))

        for event in dataset.events:
            # Allow some slack for pre-merge events
            assert event.timestamp >= start - __import__("datetime").timedelta(days=1)
            assert event.timestamp <= end + __import__("datetime").timedelta(days=30)

    def test_events_sorted_by_timestamp(self) -> None:
        """Test events are sorted by timestamp."""
        generator = SyntheticDataGenerator(seed=42)
        date_range = (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 31, tzinfo=timezone.utc),
        )

        dataset = generator.generate_dataset(n_customers=50, date_range=date_range)

        for i in range(len(dataset.events) - 1):
            assert dataset.events[i].timestamp <= dataset.events[i + 1].timestamp


# =============================================================================
# EVENT TYPE TESTS
# =============================================================================


class TestEventTypes:
    """Tests for event type distribution."""

    def test_event_type_distribution(self) -> None:
        """Test event types are distributed correctly."""
        generator = SyntheticDataGenerator(seed=42)
        date_range = (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 31, tzinfo=timezone.utc),
        )

        dataset = generator.generate_dataset(n_customers=100, date_range=date_range)

        # Should have multiple event types
        assert len(dataset.event_type_distribution) > 1

        # Should have session starts
        assert "session_start" in dataset.event_type_distribution

        # Should have view_item events
        assert "view_item" in dataset.event_type_distribution

    def test_has_purchase_events(self) -> None:
        """Test dataset has purchase events."""
        generator = SyntheticDataGenerator(seed=42)
        date_range = (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 31, tzinfo=timezone.utc),
        )

        dataset = generator.generate_dataset(n_customers=100, date_range=date_range)

        purchase_events = [e for e in dataset.events if e.event_type == EventType.PURCHASE]
        assert len(purchase_events) > 0

    def test_has_add_to_cart_events(self) -> None:
        """Test dataset has add_to_cart events."""
        generator = SyntheticDataGenerator(seed=42)
        date_range = (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 31, tzinfo=timezone.utc),
        )

        dataset = generator.generate_dataset(n_customers=100, date_range=date_range)

        cart_events = [e for e in dataset.events if e.event_type == EventType.ADD_TO_CART]
        assert len(cart_events) > 0

    def test_purchase_event_properties(self) -> None:
        """Test purchase events have correct properties."""
        generator = SyntheticDataGenerator(seed=42)
        date_range = (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 31, tzinfo=timezone.utc),
        )

        dataset = generator.generate_dataset(n_customers=100, date_range=date_range)

        purchase_events = [e for e in dataset.events if e.event_type == EventType.PURCHASE]

        for event in purchase_events[:10]:
            assert event.properties.order_id is not None
            assert event.properties.total_amount is not None
            assert event.properties.total_amount > 0
            assert event.properties.currency == "USD"


# =============================================================================
# CUSTOMER BEHAVIOR TESTS
# =============================================================================


class TestCustomerBehaviors:
    """Tests for customer behavior types."""

    def test_behavior_weights_sum_to_one(self) -> None:
        """Test behavior weights sum to 1.0."""
        total = sum(BEHAVIOR_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_all_behavior_types_defined(self) -> None:
        """Test all behavior types have weights."""
        expected_types = [
            CustomerBehaviorType.HIGH_VALUE_ACTIVE,
            CustomerBehaviorType.HIGH_VALUE_DORMANT,
            CustomerBehaviorType.FREQUENT_LOW_VALUE,
            CustomerBehaviorType.HIGH_INTENT_BROWSER,
            CustomerBehaviorType.CART_ABANDONER,
            CustomerBehaviorType.ONE_TIME_BUYER,
            CustomerBehaviorType.DISCOUNT_HUNTER,
            CustomerBehaviorType.WEEKEND_SHOPPER,
            CustomerBehaviorType.SINGLE_EVENT,
            CustomerBehaviorType.REFUND_HEAVY,
        ]

        for behavior in expected_types:
            assert behavior in BEHAVIOR_WEIGHTS


# =============================================================================
# MERGE HISTORY TESTS
# =============================================================================


class TestMergeHistory:
    """Tests for customer ID merge history."""

    def test_merge_history_structure(self) -> None:
        """Test merge history has correct structure."""
        generator = SyntheticDataGenerator(seed=42)
        date_range = (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 31, tzinfo=timezone.utc),
        )

        dataset = generator.generate_dataset(
            n_customers=100,
            date_range=date_range,
            merge_probability=0.2,
        )

        for history in dataset.id_history:
            assert history.internal_customer_id  # canonical ID
            assert history.past_id  # merged-from ID
            assert history.merge_timestamp
            assert history.past_id != history.internal_customer_id

    def test_merge_probability_affects_count(self) -> None:
        """Test merge probability affects number of merges."""
        generator = SyntheticDataGenerator(seed=42)
        date_range = (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 31, tzinfo=timezone.utc),
        )

        low_merge = generator.generate_dataset(
            n_customers=100,
            date_range=date_range,
            merge_probability=0.05,
        )

        # Reset for fair comparison
        generator = SyntheticDataGenerator(seed=42)
        high_merge = generator.generate_dataset(
            n_customers=100,
            date_range=date_range,
            merge_probability=0.30,
        )

        # Higher probability should result in more merges
        assert high_merge.n_merges > low_merge.n_merges

    def test_canonical_ids_in_customer_properties(self) -> None:
        """Test canonical IDs from merges are in customer properties."""
        generator = SyntheticDataGenerator(seed=42)
        date_range = (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 31, tzinfo=timezone.utc),
        )

        dataset = generator.generate_dataset(
            n_customers=100,
            date_range=date_range,
            merge_probability=0.2,
        )

        customer_ids = {c.internal_customer_id for c in dataset.customer_properties}

        for history in dataset.id_history:
            assert history.internal_customer_id in customer_ids


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in data generation."""

    def test_single_customer(self) -> None:
        """Test generating data for single customer."""
        generator = SyntheticDataGenerator(seed=42)
        date_range = (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 31, tzinfo=timezone.utc),
        )

        dataset = generator.generate_dataset(n_customers=1, date_range=date_range)

        assert dataset.n_customers == 1
        assert len(dataset.customer_properties) == 1
        assert len(dataset.events) > 0

    def test_no_merges(self) -> None:
        """Test generating data with no merges."""
        generator = SyntheticDataGenerator(seed=42)
        date_range = (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 31, tzinfo=timezone.utc),
        )

        dataset = generator.generate_dataset(
            n_customers=50,
            date_range=date_range,
            merge_probability=0.0,
        )

        assert dataset.n_merges == 0
        assert len(dataset.id_history) == 0

    def test_short_date_range(self) -> None:
        """Test generating data with short date range."""
        generator = SyntheticDataGenerator(seed=42)
        date_range = (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 7, tzinfo=timezone.utc),  # Only 7 days
        )

        dataset = generator.generate_dataset(n_customers=20, date_range=date_range)

        assert len(dataset.events) > 0
        # Events should be within range
        for event in dataset.events:
            assert event.timestamp >= date_range[0]


# =============================================================================
# PRODUCT GENERATION TESTS
# =============================================================================


class TestProductGeneration:
    """Tests for product generation."""

    def test_product_categories_valid(self) -> None:
        """Test product categories are from defined list."""
        generator = SyntheticDataGenerator(seed=42)
        date_range = (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 31, tzinfo=timezone.utc),
        )

        dataset = generator.generate_dataset(n_customers=50, date_range=date_range)

        view_events = [e for e in dataset.events if e.event_type == EventType.VIEW_ITEM]

        for event in view_events[:20]:
            if event.properties.product_category:
                assert event.properties.product_category in PRODUCT_CATEGORIES

    def test_product_prices_reasonable(self) -> None:
        """Test product prices are within reasonable range."""
        generator = SyntheticDataGenerator(seed=42)
        date_range = (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 31, tzinfo=timezone.utc),
        )

        dataset = generator.generate_dataset(n_customers=50, date_range=date_range)

        view_events = [e for e in dataset.events if e.event_type == EventType.VIEW_ITEM]

        for event in view_events[:20]:
            if event.properties.product_price:
                assert event.properties.product_price > 0
                assert event.properties.product_price < Decimal("5000")


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_generate_small_dataset(self) -> None:
        """Test generate_small_dataset helper."""
        dataset = generate_small_dataset(seed=42)

        assert dataset.n_customers == 1000
        assert len(dataset.events) > 0
        assert len(dataset.customer_properties) == 1000

    def test_preview_dataset(self) -> None:
        """Test preview_dataset helper."""
        dataset = generate_small_dataset(seed=42)
        preview = preview_dataset(dataset, n_samples=3)

        assert "events" in preview
        assert "customers" in preview
        assert "merges" in preview
        assert len(preview["events"]) <= 3
        assert len(preview["customers"]) <= 3

    def test_dataset_statistics(self) -> None:
        """Test dataset_statistics helper."""
        dataset = generate_small_dataset(seed=42)
        stats = dataset_statistics(dataset)

        assert "n_customers" in stats
        assert "n_events" in stats
        assert "n_merges" in stats
        assert "n_purchases" in stats
        assert "total_revenue" in stats
        assert "avg_events_per_customer" in stats
        assert "event_distribution" in stats

        assert stats["n_customers"] == 1000
        assert stats["n_events"] == dataset.n_events
        assert stats["total_revenue"] > 0


# =============================================================================
# VALIDATION TESTS
# =============================================================================


class TestDataValidation:
    """Tests for data validation."""

    def test_all_events_have_customer_id(self) -> None:
        """Test all events have customer ID."""
        dataset = generate_small_dataset(seed=42)

        for event in dataset.events:
            assert event.internal_customer_id

    def test_all_events_have_timestamp(self) -> None:
        """Test all events have timestamp."""
        dataset = generate_small_dataset(seed=42)

        for event in dataset.events:
            assert event.timestamp is not None

    def test_all_events_have_event_id(self) -> None:
        """Test all events have unique event ID."""
        dataset = generate_small_dataset(seed=42)

        event_ids = [e.event_id for e in dataset.events]
        assert len(event_ids) == len(set(event_ids))

    def test_event_schema_valid(self) -> None:
        """Test all events have valid schema."""
        dataset = generate_small_dataset(seed=42)

        # Just accessing properties should work if schema is valid
        for event in dataset.events[:100]:
            _ = event.event_type.value
            _ = event.properties.model_dump()


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================


class TestPerformance:
    """Tests for generation performance."""

    @pytest.mark.slow
    def test_generate_medium_dataset_time(self) -> None:
        """Test medium dataset generates in reasonable time."""
        import time

        generator = SyntheticDataGenerator(seed=42)
        date_range = (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 6, 30, tzinfo=timezone.utc),
        )

        start = time.time()
        dataset = generator.generate_dataset(n_customers=1000, date_range=date_range)
        elapsed = time.time() - start

        # Should complete in less than 30 seconds for 1000 customers
        assert elapsed < 30
        assert dataset.n_customers == 1000
