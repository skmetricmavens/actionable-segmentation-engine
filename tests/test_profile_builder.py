"""
Tests for customer profile building.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.data.joiner import MergeMap, resolve_customer_merges
from src.data.schemas import (
    CustomerIdHistory,
    CustomerProfile,
    EventProperties,
    EventRecord,
    EventType,
)
from src.exceptions import InsufficientDataError, ProfileBuildError
from src.features.profile_builder import (
    ProfileBuilder,
    build_profile,
    build_profiles_batch,
    build_profiles_iter,
    group_events_by_customer,
    profile_summary_stats,
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


def make_customer_journey(
    customer_id: str,
    start_date: datetime,
) -> list[EventRecord]:
    """Create a realistic customer journey."""
    return [
        make_event(
            f"{customer_id}_e1", customer_id, EventType.SESSION_START,
            timestamp=start_date,
            device_type="desktop",
        ),
        make_event(
            f"{customer_id}_e2", customer_id, EventType.VIEW_CATEGORY,
            timestamp=start_date,
            product_category="Electronics",
        ),
        make_event(
            f"{customer_id}_e3", customer_id, EventType.VIEW_ITEM,
            timestamp=start_date,
            product_category="Electronics",
        ),
        make_event(
            f"{customer_id}_e4", customer_id, EventType.ADD_TO_CART,
            timestamp=start_date,
            product_category="Electronics",
        ),
        make_event(
            f"{customer_id}_e5", customer_id, EventType.PURCHASE,
            timestamp=start_date,
            total_amount=Decimal("150.00"),
            product_category="Electronics",
        ),
    ]


# =============================================================================
# GROUP EVENTS TESTS
# =============================================================================


class TestGroupEventsByCustomer:
    """Tests for group_events_by_customer function."""

    def test_empty_events(self) -> None:
        """Test with empty events list."""
        result = group_events_by_customer([])
        assert result == {}

    def test_single_customer(self) -> None:
        """Test events from single customer."""
        events = [
            make_event("e1", "cust_1", EventType.VIEW_ITEM),
            make_event("e2", "cust_1", EventType.PURCHASE),
        ]

        result = group_events_by_customer(events)

        assert "cust_1" in result
        assert len(result["cust_1"]) == 2

    def test_multiple_customers(self) -> None:
        """Test events from multiple customers."""
        events = [
            make_event("e1", "cust_1", EventType.VIEW_ITEM),
            make_event("e2", "cust_2", EventType.VIEW_ITEM),
            make_event("e3", "cust_1", EventType.PURCHASE),
        ]

        result = group_events_by_customer(events)

        assert len(result) == 2
        assert len(result["cust_1"]) == 2
        assert len(result["cust_2"]) == 1

    def test_with_merge_map(self) -> None:
        """Test grouping with merge map application."""
        events = [
            make_event("e1", "old_id", EventType.VIEW_ITEM),
            make_event("e2", "new_id", EventType.PURCHASE),
        ]
        merge_map: MergeMap = {"old_id": "new_id"}

        result = group_events_by_customer(events, merge_map)

        # Both events should be grouped under canonical ID
        assert len(result) == 1
        assert "new_id" in result
        assert len(result["new_id"]) == 2


# =============================================================================
# BUILD PROFILE TESTS
# =============================================================================


class TestBuildProfile:
    """Tests for build_profile function."""

    def test_empty_events_raises(self) -> None:
        """Test that empty events raises ProfileBuildError."""
        with pytest.raises(ProfileBuildError) as exc_info:
            build_profile("cust_1", [])

        assert "no events" in str(exc_info.value)
        assert exc_info.value.customer_id == "cust_1"

    def test_simple_profile(self) -> None:
        """Test building profile from simple events."""
        events = make_customer_journey(
            "cust_1",
            datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
        )
        ref_date = datetime(2024, 1, 20, tzinfo=timezone.utc)

        profile = build_profile("cust_1", events, reference_date=ref_date)

        assert profile.internal_customer_id == "cust_1"
        assert profile.total_purchases == 1
        assert profile.total_revenue == Decimal("150.00")
        assert profile.total_sessions == 1
        assert profile.top_category == "Electronics"

    def test_profile_with_merged_ids(self) -> None:
        """Test profile building with merge map."""
        events = make_customer_journey(
            "canonical_id",
            datetime(2024, 1, 15, tzinfo=timezone.utc),
        )
        merge_map: MergeMap = {"old_id_1": "canonical_id", "old_id_2": "canonical_id"}

        profile = build_profile("canonical_id", events, merge_map=merge_map)

        assert "old_id_1" in profile.merged_from_ids
        assert "old_id_2" in profile.merged_from_ids
        assert len(profile.merged_from_ids) == 2

    def test_profile_temporal_bounds(self) -> None:
        """Test profile first_seen and last_seen."""
        events = [
            make_event(
                "e1", "cust_1", EventType.VIEW_ITEM,
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
            make_event(
                "e2", "cust_1", EventType.VIEW_ITEM,
                timestamp=datetime(2024, 1, 31, tzinfo=timezone.utc),
            ),
        ]

        profile = build_profile("cust_1", events)

        assert profile.first_seen == datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert profile.last_seen == datetime(2024, 1, 31, tzinfo=timezone.utc)

    def test_profile_clv_calculated(self) -> None:
        """Test that CLV is calculated."""
        events = [
            make_event(
                "e1", "cust_1", EventType.SESSION_START,
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
            make_event(
                "e2", "cust_1", EventType.PURCHASE,
                timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
                total_amount=Decimal("100.00"),
            ),
            make_event(
                "e3", "cust_1", EventType.PURCHASE,
                timestamp=datetime(2024, 2, 15, tzinfo=timezone.utc),
                total_amount=Decimal("100.00"),
            ),
        ]
        ref_date = datetime(2024, 3, 1, tzinfo=timezone.utc)

        profile = build_profile("cust_1", events, reference_date=ref_date)

        assert profile.clv_estimate > Decimal("0")

    def test_profile_churn_risk_calculated(self) -> None:
        """Test that churn risk is calculated."""
        events = make_customer_journey(
            "cust_1",
            datetime(2024, 1, 15, tzinfo=timezone.utc),
        )
        ref_date = datetime(2024, 4, 15, tzinfo=timezone.utc)  # 3 months later

        profile = build_profile("cust_1", events, reference_date=ref_date)

        assert 0.0 <= profile.churn_risk_score <= 1.0


# =============================================================================
# BUILD PROFILES BATCH TESTS
# =============================================================================


class TestBuildProfilesBatch:
    """Tests for build_profiles_batch function."""

    def test_empty_events_raises(self) -> None:
        """Test that empty events raises InsufficientDataError."""
        with pytest.raises(InsufficientDataError):
            build_profiles_batch([])

    def test_single_customer(self) -> None:
        """Test building profiles for single customer."""
        events = make_customer_journey(
            "cust_1",
            datetime(2024, 1, 15, tzinfo=timezone.utc),
        )

        profiles = build_profiles_batch(events)

        assert len(profiles) == 1
        assert profiles[0].internal_customer_id == "cust_1"

    def test_multiple_customers(self) -> None:
        """Test building profiles for multiple customers."""
        events = (
            make_customer_journey("cust_1", datetime(2024, 1, 15, tzinfo=timezone.utc))
            + make_customer_journey("cust_2", datetime(2024, 1, 20, tzinfo=timezone.utc))
        )

        profiles = build_profiles_batch(events)

        assert len(profiles) == 2
        customer_ids = {p.internal_customer_id for p in profiles}
        assert customer_ids == {"cust_1", "cust_2"}

    def test_min_events_filter(self) -> None:
        """Test min_events parameter filters customers."""
        events = (
            make_customer_journey("cust_1", datetime(2024, 1, 15, tzinfo=timezone.utc))
            + [make_event("e1", "cust_2", EventType.VIEW_ITEM)]
        )

        profiles = build_profiles_batch(events, min_events=3)

        assert len(profiles) == 1
        assert profiles[0].internal_customer_id == "cust_1"

    def test_with_merge_map(self) -> None:
        """Test batch profile building with merge map."""
        events = [
            make_event(
                "e1", "old_id", EventType.SESSION_START,
                timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            ),
            make_event(
                "e2", "old_id", EventType.VIEW_ITEM,
                timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            ),
            make_event(
                "e3", "new_id", EventType.PURCHASE,
                timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
                total_amount=Decimal("100.00"),
            ),
        ]
        merge_map: MergeMap = {"old_id": "new_id"}

        profiles = build_profiles_batch(events, merge_map=merge_map)

        assert len(profiles) == 1
        assert profiles[0].internal_customer_id == "new_id"
        assert profiles[0].total_purchases == 1
        assert profiles[0].total_sessions == 1


# =============================================================================
# BUILD PROFILES ITERATOR TESTS
# =============================================================================


class TestBuildProfilesIter:
    """Tests for build_profiles_iter function."""

    def test_produces_same_results_as_batch(self) -> None:
        """Test iterator produces same results as batch."""
        events = (
            make_customer_journey("cust_1", datetime(2024, 1, 15, tzinfo=timezone.utc))
            + make_customer_journey("cust_2", datetime(2024, 1, 20, tzinfo=timezone.utc))
        )
        ref_date = datetime(2024, 2, 1, tzinfo=timezone.utc)

        batch_profiles = build_profiles_batch(events, reference_date=ref_date)
        iter_profiles = list(build_profiles_iter(iter(events), reference_date=ref_date))

        assert len(batch_profiles) == len(iter_profiles)

        batch_ids = {p.internal_customer_id for p in batch_profiles}
        iter_ids = {p.internal_customer_id for p in iter_profiles}
        assert batch_ids == iter_ids


# =============================================================================
# PROFILE BUILDER CLASS TESTS
# =============================================================================


class TestProfileBuilder:
    """Tests for ProfileBuilder class."""

    def test_build_single(self) -> None:
        """Test building single profile."""
        builder = ProfileBuilder()
        events = make_customer_journey(
            "cust_1",
            datetime(2024, 1, 15, tzinfo=timezone.utc),
        )

        profile = builder.build("cust_1", events)

        assert profile.internal_customer_id == "cust_1"

    def test_build_all(self) -> None:
        """Test building all profiles."""
        builder = ProfileBuilder(min_events=1)
        events = (
            make_customer_journey("cust_1", datetime(2024, 1, 15, tzinfo=timezone.utc))
            + make_customer_journey("cust_2", datetime(2024, 1, 20, tzinfo=timezone.utc))
        )

        profiles = builder.build_all(events)

        assert len(profiles) == 2

    def test_cache_functionality(self) -> None:
        """Test profile caching."""
        builder = ProfileBuilder()
        events = make_customer_journey(
            "cust_1",
            datetime(2024, 1, 15, tzinfo=timezone.utc),
        )

        # First build - should cache
        profile1 = builder.build_with_cache("cust_1", events)

        # Second build - should return cached
        profile2 = builder.build_with_cache("cust_1", events)

        assert profile1 is profile2

    def test_get_cached_profile(self) -> None:
        """Test getting cached profile."""
        builder = ProfileBuilder()
        events = make_customer_journey(
            "cust_1",
            datetime(2024, 1, 15, tzinfo=timezone.utc),
        )

        # Not cached yet
        assert builder.get_cached_profile("cust_1") is None

        # Build with cache
        builder.build_with_cache("cust_1", events)

        # Now cached
        assert builder.get_cached_profile("cust_1") is not None

    def test_clear_cache(self) -> None:
        """Test clearing cache."""
        builder = ProfileBuilder()
        events = make_customer_journey(
            "cust_1",
            datetime(2024, 1, 15, tzinfo=timezone.utc),
        )

        builder.build_with_cache("cust_1", events)
        assert builder.get_cached_profile("cust_1") is not None

        builder.clear_cache()
        assert builder.get_cached_profile("cust_1") is None

    def test_with_merge_map(self) -> None:
        """Test builder with merge map."""
        merge_map: MergeMap = {"old_id": "new_id"}
        builder = ProfileBuilder(merge_map=merge_map)

        events = [
            make_event(
                "e1", "new_id", EventType.SESSION_START,
                timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            ),
            make_event(
                "e2", "new_id", EventType.VIEW_ITEM,
                timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            ),
        ]

        profile = builder.build("new_id", events)

        assert "old_id" in profile.merged_from_ids


# =============================================================================
# PROFILE SUMMARY STATS TESTS
# =============================================================================


class TestProfileSummaryStats:
    """Tests for profile_summary_stats function."""

    def test_empty_profiles(self) -> None:
        """Test with empty profiles list."""
        result = profile_summary_stats([])
        assert result == {}

    def test_single_profile_stats(self) -> None:
        """Test stats with single profile."""
        events = make_customer_journey(
            "cust_1",
            datetime(2024, 1, 15, tzinfo=timezone.utc),
        )
        profile = build_profile(
            "cust_1",
            events,
            reference_date=datetime(2024, 1, 20, tzinfo=timezone.utc),
        )

        stats = profile_summary_stats([profile])

        assert stats["n_customers"] == 1
        assert stats["total_revenue"] == 150.0
        assert stats["avg_revenue_per_customer"] == 150.0
        assert stats["total_purchases"] == 1
        assert stats["pct_with_purchases"] == 1.0

    def test_multiple_profiles_stats(self) -> None:
        """Test stats with multiple profiles."""
        ref_date = datetime(2024, 2, 1, tzinfo=timezone.utc)

        # Create two profiles
        events1 = [
            make_event(
                "e1", "cust_1", EventType.SESSION_START,
                timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            ),
            make_event(
                "e2", "cust_1", EventType.PURCHASE,
                timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
                total_amount=Decimal("100.00"),
            ),
        ]
        events2 = [
            make_event(
                "e3", "cust_2", EventType.SESSION_START,
                timestamp=datetime(2024, 1, 20, tzinfo=timezone.utc),
            ),
            make_event(
                "e4", "cust_2", EventType.VIEW_ITEM,
                timestamp=datetime(2024, 1, 20, tzinfo=timezone.utc),
            ),
        ]

        profile1 = build_profile("cust_1", events1, reference_date=ref_date)
        profile2 = build_profile("cust_2", events2, reference_date=ref_date)

        stats = profile_summary_stats([profile1, profile2])

        assert stats["n_customers"] == 2
        assert stats["total_revenue"] == 100.0  # Only cust_1 has revenue
        assert stats["avg_revenue_per_customer"] == 50.0  # 100/2
        assert stats["pct_with_purchases"] == 0.5  # 1/2 has purchases


# =============================================================================
# INTEGRATION WITH SYNTHETIC DATA
# =============================================================================


class TestWithSyntheticData:
    """Integration tests using synthetic data generator."""

    def test_build_profiles_from_synthetic(self) -> None:
        """Test building profiles from synthetic data."""
        from src.data.synthetic_generator import SyntheticDataGenerator

        generator = SyntheticDataGenerator(seed=42)
        date_range = (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 31, tzinfo=timezone.utc),
        )

        dataset = generator.generate_dataset(
            n_customers=50,
            date_range=date_range,
            merge_probability=0.1,
        )

        # Resolve merges
        from src.data.joiner import resolve_customer_merges

        merge_map = resolve_customer_merges(dataset.id_history)

        # Build profiles
        ref_date = datetime(2024, 4, 1, tzinfo=timezone.utc)
        profiles = build_profiles_batch(
            dataset.events,
            merge_map=merge_map,
            reference_date=ref_date,
        )

        assert len(profiles) > 0
        assert all(isinstance(p, CustomerProfile) for p in profiles)

        # Check stats are reasonable
        stats = profile_summary_stats(profiles)
        assert stats["n_customers"] > 0
        assert 0.0 <= stats["pct_with_purchases"] <= 1.0

    def test_profile_builder_class_with_synthetic(self) -> None:
        """Test ProfileBuilder class with synthetic data."""
        from src.data.synthetic_generator import SyntheticDataGenerator
        from src.data.joiner import resolve_customer_merges

        generator = SyntheticDataGenerator(seed=123)
        date_range = (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 2, 28, tzinfo=timezone.utc),
        )

        dataset = generator.generate_dataset(
            n_customers=30,
            date_range=date_range,
        )

        merge_map = resolve_customer_merges(dataset.id_history)
        ref_date = datetime(2024, 3, 1, tzinfo=timezone.utc)

        builder = ProfileBuilder(
            merge_map=merge_map,
            reference_date=ref_date,
            min_events=3,
        )

        profiles = builder.build_all(dataset.events)

        assert len(profiles) > 0
        # All profiles should have at least 3 events' worth of data
        assert all(
            p.total_sessions + p.total_page_views + p.total_purchases >= 1
            for p in profiles
        )


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_customer_with_only_view_events(self) -> None:
        """Test profile for customer who only views, never purchases."""
        events = [
            make_event(
                "e1", "viewer", EventType.SESSION_START,
                timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            ),
            make_event(
                "e2", "viewer", EventType.VIEW_ITEM,
                timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
                product_category="Books",
            ),
            make_event(
                "e3", "viewer", EventType.VIEW_ITEM,
                timestamp=datetime(2024, 1, 16, tzinfo=timezone.utc),
                product_category="Books",
            ),
        ]
        ref_date = datetime(2024, 2, 1, tzinfo=timezone.utc)

        profile = build_profile("viewer", events, reference_date=ref_date)

        assert profile.total_purchases == 0
        assert profile.total_revenue == Decimal("0")
        assert profile.total_items_viewed == 2
        assert profile.top_category == "Books"
        assert profile.churn_risk_score > 0  # Should have some risk

    def test_customer_with_refunds(self) -> None:
        """Test profile for customer with refunds."""
        events = [
            make_event(
                "e1", "refunder", EventType.SESSION_START,
                timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            ),
            make_event(
                "e2", "refunder", EventType.PURCHASE,
                timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
                total_amount=Decimal("100.00"),
            ),
            make_event(
                "e3", "refunder", EventType.REFUND,
                timestamp=datetime(2024, 1, 20, tzinfo=timezone.utc),
            ),
        ]
        ref_date = datetime(2024, 2, 1, tzinfo=timezone.utc)

        profile = build_profile("refunder", events, reference_date=ref_date)

        assert profile.total_purchases == 1
        assert profile.total_refunds == 1
        assert profile.refund_rate == 1.0

    def test_high_frequency_purchaser(self) -> None:
        """Test profile for high-frequency purchaser."""
        base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        events = [
            make_event(
                "e0", "vip", EventType.SESSION_START,
                timestamp=base_date,
            ),
        ]

        # Add 10 purchases over 30 days
        for i in range(10):
            from datetime import timedelta

            events.append(
                make_event(
                    f"p{i}", "vip", EventType.PURCHASE,
                    timestamp=base_date + timedelta(days=i * 3),
                    total_amount=Decimal("50.00"),
                )
            )

        ref_date = datetime(2024, 2, 1, tzinfo=timezone.utc)
        profile = build_profile("vip", events, reference_date=ref_date)

        assert profile.total_purchases == 10
        assert profile.total_revenue == Decimal("500.00")
        assert profile.purchase_frequency_per_month > 5  # High frequency
        assert profile.churn_risk_score < 0.5  # Low risk
