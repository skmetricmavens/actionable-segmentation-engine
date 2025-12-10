"""
Module: profile_builder

Purpose: Build canonical customer profiles from raw events.

This module aggregates raw event data into structured customer profiles that
can be used for segmentation. It requires merged customer IDs from the joiner
module to ensure unified customer identities.
"""

from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from typing import Iterable, Iterator

from src.data.joiner import MergeMap, apply_merge_to_customer_id, get_all_ids_for_customer
from src.data.schemas import (
    CustomerProfile,
    EventRecord,
)
from src.exceptions import InsufficientDataError, ProfileBuildError
from src.features.aggregators import (
    aggregate_categories,
    aggregate_device,
    aggregate_purchases,
    aggregate_sessions,
    aggregate_temporal,
    calculate_churn_risk,
    calculate_clv_estimate,
)


def group_events_by_customer(
    events: Iterable[EventRecord],
    merge_map: MergeMap | None = None,
) -> dict[str, list[EventRecord]]:
    """
    Group events by customer ID, optionally applying merge mapping.

    Args:
        events: Iterable of event records
        merge_map: Optional mapping from old IDs to canonical IDs

    Returns:
        Dictionary mapping customer_id to list of events
    """
    grouped: dict[str, list[EventRecord]] = defaultdict(list)
    merge_map = merge_map or {}

    for event in events:
        canonical_id = apply_merge_to_customer_id(event.internal_customer_id, merge_map)
        grouped[canonical_id].append(event)

    return dict(grouped)


def build_profile(
    customer_id: str,
    events: list[EventRecord],
    *,
    merge_map: MergeMap | None = None,
    reference_date: datetime | None = None,
) -> CustomerProfile:
    """
    Build a single customer profile from events.

    Args:
        customer_id: The canonical customer ID
        events: List of events for this customer
        merge_map: Optional merge map to get merged_from_ids
        reference_date: Reference date for calculations (defaults to now)

    Returns:
        CustomerProfile with aggregated metrics

    Raises:
        ProfileBuildError: If profile cannot be built
    """
    if not events:
        raise ProfileBuildError(
            f"Cannot build profile for customer {customer_id}: no events",
            customer_id=customer_id,
        )

    # Get merged IDs if merge_map provided
    merged_from_ids: list[str] = []
    if merge_map:
        all_ids = get_all_ids_for_customer(customer_id, merge_map)
        merged_from_ids = [id_ for id_ in all_ids if id_ != customer_id]

    # Aggregate metrics
    temporal = aggregate_temporal(events)
    purchase_metrics = aggregate_purchases(events, reference_date=reference_date)
    session_metrics = aggregate_sessions(events)
    category_affinities = aggregate_categories(events)
    device_metrics = aggregate_device(events)

    # Calculate derived metrics
    first_seen = temporal["first_seen"]
    last_seen = temporal["last_seen"]

    if first_seen is None or last_seen is None:
        raise ProfileBuildError(
            f"Cannot determine temporal bounds for customer {customer_id}",
            customer_id=customer_id,
        )

    # Customer tenure for CLV and churn calculations
    ref_date = reference_date or datetime.now(tz=first_seen.tzinfo)
    customer_tenure_days = (ref_date - first_seen).days

    # CLV estimate
    clv_estimate = calculate_clv_estimate(
        purchase_metrics,
        customer_tenure_days,
    )

    # Churn risk
    churn_risk = calculate_churn_risk(
        purchase_metrics,
        session_metrics,
        customer_tenure_days,
    )

    # Top category
    top_category = category_affinities[0].category if category_affinities else None

    return CustomerProfile(
        # Identity
        internal_customer_id=customer_id,
        merged_from_ids=merged_from_ids,
        # Temporal
        first_seen=first_seen,
        last_seen=last_seen,
        # Transactional
        total_purchases=purchase_metrics["total_purchases"],
        total_revenue=purchase_metrics["total_revenue"],
        avg_order_value=purchase_metrics["avg_order_value"],
        days_since_last_purchase=purchase_metrics["days_since_last_purchase"],
        purchase_frequency_per_month=purchase_metrics["purchase_frequency_per_month"],
        # Engagement
        total_sessions=session_metrics["total_sessions"],
        total_page_views=session_metrics["total_page_views"],
        total_items_viewed=session_metrics["total_items_viewed"],
        total_cart_additions=session_metrics["total_cart_additions"],
        cart_abandonment_rate=session_metrics["cart_abandonment_rate"],
        # Categories
        category_affinities=category_affinities,
        top_category=top_category,
        # Temporal patterns
        preferred_day_of_week=temporal["preferred_day_of_week"],
        preferred_hour_of_day=temporal["preferred_hour_of_day"],
        # Device
        primary_device_type=device_metrics["primary_device_type"],
        mobile_session_ratio=device_metrics["mobile_session_ratio"],
        # Value metrics
        clv_estimate=clv_estimate,
        churn_risk_score=churn_risk,
        # Refunds
        total_refunds=purchase_metrics["total_refunds"],
        refund_rate=purchase_metrics["refund_rate"],
    )


def build_profiles_batch(
    events: list[EventRecord],
    *,
    merge_map: MergeMap | None = None,
    reference_date: datetime | None = None,
    min_events: int = 1,
) -> list[CustomerProfile]:
    """
    Build profiles for all customers in event list.

    Args:
        events: List of all events
        merge_map: Optional merge map for ID unification
        reference_date: Reference date for calculations
        min_events: Minimum events required per customer (default 1)

    Returns:
        List of CustomerProfile objects

    Raises:
        InsufficientDataError: If no events provided
    """
    if not events:
        raise InsufficientDataError(
            "No events provided for profile building",
            required=1,
            actual=0,
            data_type="events",
        )

    # Group events by customer
    grouped = group_events_by_customer(events, merge_map)

    profiles: list[CustomerProfile] = []
    for customer_id, customer_events in grouped.items():
        if len(customer_events) < min_events:
            continue

        try:
            profile = build_profile(
                customer_id,
                customer_events,
                merge_map=merge_map,
                reference_date=reference_date,
            )
            profiles.append(profile)
        except ProfileBuildError:
            # Skip customers where profile building fails
            continue

    return profiles


def build_profiles_iter(
    events: Iterable[EventRecord],
    *,
    merge_map: MergeMap | None = None,
    reference_date: datetime | None = None,
    min_events: int = 1,
) -> Iterator[CustomerProfile]:
    """
    Build profiles lazily using an iterator.

    Memory-efficient version for large datasets. Note that this buffers
    all events in memory to group by customer, so for very large datasets
    consider using streaming approaches.

    Args:
        events: Iterable of events
        merge_map: Optional merge map for ID unification
        reference_date: Reference date for calculations
        min_events: Minimum events required per customer

    Yields:
        CustomerProfile objects
    """
    # Group events by customer (requires buffering)
    grouped = group_events_by_customer(events, merge_map)

    for customer_id, customer_events in grouped.items():
        if len(customer_events) < min_events:
            continue

        try:
            yield build_profile(
                customer_id,
                customer_events,
                merge_map=merge_map,
                reference_date=reference_date,
            )
        except ProfileBuildError:
            continue


class ProfileBuilder:
    """
    Configurable profile builder with caching support.

    Allows customization of profile building parameters and provides
    methods for building single profiles or batches.
    """

    def __init__(
        self,
        *,
        merge_map: MergeMap | None = None,
        reference_date: datetime | None = None,
        min_events: int = 1,
        clv_projection_years: int = 3,
        clv_discount_rate: float = 0.1,
    ) -> None:
        """
        Initialize ProfileBuilder.

        Args:
            merge_map: Merge mapping for ID unification
            reference_date: Reference date for calculations
            min_events: Minimum events required per customer
            clv_projection_years: Years to project for CLV
            clv_discount_rate: Annual discount rate for CLV
        """
        self.merge_map = merge_map or {}
        self.reference_date = reference_date
        self.min_events = min_events
        self.clv_projection_years = clv_projection_years
        self.clv_discount_rate = clv_discount_rate
        self._profile_cache: dict[str, CustomerProfile] = {}

    def build(self, customer_id: str, events: list[EventRecord]) -> CustomerProfile:
        """
        Build profile for a single customer.

        Args:
            customer_id: Customer ID
            events: Events for this customer

        Returns:
            CustomerProfile
        """
        return build_profile(
            customer_id,
            events,
            merge_map=self.merge_map,
            reference_date=self.reference_date,
        )

    def build_all(self, events: list[EventRecord]) -> list[CustomerProfile]:
        """
        Build profiles for all customers.

        Args:
            events: All events

        Returns:
            List of CustomerProfiles
        """
        return build_profiles_batch(
            events,
            merge_map=self.merge_map,
            reference_date=self.reference_date,
            min_events=self.min_events,
        )

    def build_with_cache(
        self,
        customer_id: str,
        events: list[EventRecord],
    ) -> CustomerProfile:
        """
        Build profile with caching.

        Args:
            customer_id: Customer ID
            events: Events for this customer

        Returns:
            CustomerProfile (from cache if available)
        """
        if customer_id in self._profile_cache:
            return self._profile_cache[customer_id]

        profile = self.build(customer_id, events)
        self._profile_cache[customer_id] = profile
        return profile

    def clear_cache(self) -> None:
        """Clear the profile cache."""
        self._profile_cache.clear()

    def get_cached_profile(self, customer_id: str) -> CustomerProfile | None:
        """
        Get profile from cache if available.

        Args:
            customer_id: Customer ID

        Returns:
            Cached CustomerProfile or None
        """
        return self._profile_cache.get(customer_id)


def profile_summary_stats(profiles: list[CustomerProfile]) -> dict[str, float]:
    """
    Calculate summary statistics across all profiles.

    Args:
        profiles: List of customer profiles

    Returns:
        Dictionary of summary statistics
    """
    if not profiles:
        return {}

    total_revenue = sum(float(p.total_revenue) for p in profiles)
    total_purchases = sum(p.total_purchases for p in profiles)
    total_sessions = sum(p.total_sessions for p in profiles)

    return {
        "n_customers": len(profiles),
        "total_revenue": total_revenue,
        "avg_revenue_per_customer": total_revenue / len(profiles),
        "total_purchases": total_purchases,
        "avg_purchases_per_customer": total_purchases / len(profiles),
        "total_sessions": total_sessions,
        "avg_sessions_per_customer": total_sessions / len(profiles),
        "avg_clv": sum(float(p.clv_estimate) for p in profiles) / len(profiles),
        "avg_churn_risk": sum(p.churn_risk_score for p in profiles) / len(profiles),
        "pct_with_purchases": sum(1 for p in profiles if p.total_purchases > 0) / len(profiles),
        "avg_cart_abandonment_rate": sum(p.cart_abandonment_rate for p in profiles) / len(profiles),
    }
