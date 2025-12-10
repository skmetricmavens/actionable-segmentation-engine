"""
Module: aggregators

Purpose: Behavioral and transactional aggregation functions.

Pure functions for computing metrics from event sequences. These are used by
the profile builder to aggregate raw events into actionable customer metrics.
"""

from collections import Counter
from datetime import datetime, timedelta
from decimal import Decimal
from typing import TypedDict

from src.data.schemas import (
    CategoryAffinity,
    EventRecord,
    EventType,
)


class PurchaseMetrics(TypedDict):
    """Aggregated purchase metrics."""

    total_purchases: int
    total_revenue: Decimal
    avg_order_value: Decimal
    total_refunds: int
    refund_rate: float
    days_since_last_purchase: int | None
    purchase_frequency_per_month: float


class SessionMetrics(TypedDict):
    """Aggregated session/engagement metrics."""

    total_sessions: int
    total_page_views: int
    total_items_viewed: int
    total_cart_additions: int
    cart_abandonment_rate: float


class TemporalMetrics(TypedDict):
    """Aggregated temporal patterns."""

    preferred_day_of_week: int | None  # 0=Monday, 6=Sunday
    preferred_hour_of_day: int | None  # 0-23
    first_seen: datetime | None
    last_seen: datetime | None


class DeviceMetrics(TypedDict):
    """Aggregated device/channel metrics."""

    primary_device_type: str | None
    mobile_session_ratio: float


def aggregate_purchases(
    events: list[EventRecord],
    *,
    reference_date: datetime | None = None,
) -> PurchaseMetrics:
    """
    Calculate purchase-related metrics from events.

    Args:
        events: List of event records for a customer
        reference_date: Date to calculate "days since" from (defaults to now)

    Returns:
        PurchaseMetrics with aggregated purchase data
    """
    if reference_date is None:
        reference_date = datetime.now(tz=events[0].timestamp.tzinfo) if events else datetime.now()

    purchase_events = [e for e in events if e.event_type == EventType.PURCHASE]
    refund_events = [e for e in events if e.event_type == EventType.REFUND]

    total_purchases = len(purchase_events)
    total_refunds = len(refund_events)

    # Calculate total revenue from purchase events
    total_revenue = Decimal("0")
    for event in purchase_events:
        if event.properties.total_amount is not None:
            total_revenue += event.properties.total_amount

    # Calculate average order value
    avg_order_value = Decimal("0")
    if total_purchases > 0:
        avg_order_value = total_revenue / total_purchases

    # Calculate refund rate
    refund_rate = 0.0
    if total_purchases > 0:
        refund_rate = total_refunds / total_purchases

    # Days since last purchase
    days_since_last_purchase: int | None = None
    if purchase_events:
        last_purchase = max(e.timestamp for e in purchase_events)
        days_since_last_purchase = (reference_date - last_purchase).days

    # Purchase frequency per month
    purchase_frequency_per_month = 0.0
    if purchase_events and len(purchase_events) >= 1:
        first_purchase = min(e.timestamp for e in purchase_events)
        last_purchase = max(e.timestamp for e in purchase_events)
        days_span = (last_purchase - first_purchase).days
        if days_span > 0:
            months = days_span / 30.0
            purchase_frequency_per_month = total_purchases / months
        elif total_purchases > 0:
            # All purchases on same day - use 1 month window
            purchase_frequency_per_month = float(total_purchases)

    return PurchaseMetrics(
        total_purchases=total_purchases,
        total_revenue=total_revenue,
        avg_order_value=avg_order_value,
        total_refunds=total_refunds,
        refund_rate=refund_rate,
        days_since_last_purchase=days_since_last_purchase,
        purchase_frequency_per_month=purchase_frequency_per_month,
    )


def aggregate_sessions(
    events: list[EventRecord],
) -> SessionMetrics:
    """
    Calculate session/engagement metrics from events.

    Args:
        events: List of event records for a customer

    Returns:
        SessionMetrics with aggregated engagement data
    """
    session_starts = [e for e in events if e.event_type == EventType.SESSION_START]
    page_views = [
        e for e in events if e.event_type in (EventType.VIEW_CATEGORY, EventType.VIEW_ITEM)
    ]
    item_views = [e for e in events if e.event_type == EventType.VIEW_ITEM]
    cart_additions = [e for e in events if e.event_type == EventType.ADD_TO_CART]
    purchases = [e for e in events if e.event_type == EventType.PURCHASE]

    total_sessions = len(session_starts)
    total_page_views = len(page_views)
    total_items_viewed = len(item_views)
    total_cart_additions = len(cart_additions)

    # Cart abandonment rate: carts without purchases / total carts
    # Simplified: if we have cart additions but fewer purchases
    cart_abandonment_rate = 0.0
    if total_cart_additions > 0:
        # Simple approximation: purchases / cart additions
        conversion_rate = len(purchases) / total_cart_additions
        cart_abandonment_rate = max(0.0, 1.0 - conversion_rate)

    return SessionMetrics(
        total_sessions=total_sessions,
        total_page_views=total_page_views,
        total_items_viewed=total_items_viewed,
        total_cart_additions=total_cart_additions,
        cart_abandonment_rate=cart_abandonment_rate,
    )


def aggregate_categories(
    events: list[EventRecord],
) -> list[CategoryAffinity]:
    """
    Calculate category affinity scores from events.

    Engagement score is based on weighted interactions:
    - View: 1 point
    - Add to cart: 2 points
    - Purchase: 5 points

    Args:
        events: List of event records for a customer

    Returns:
        List of CategoryAffinity objects sorted by engagement score descending
    """
    category_views: Counter[str] = Counter()
    category_purchases: Counter[str] = Counter()
    category_engagement: Counter[str] = Counter()

    for event in events:
        category = event.properties.product_category
        if category is None:
            continue

        if event.event_type == EventType.VIEW_ITEM:
            category_views[category] += 1
            category_engagement[category] += 1
        elif event.event_type == EventType.VIEW_CATEGORY:
            category_views[category] += 1
            category_engagement[category] += 1
        elif event.event_type == EventType.ADD_TO_CART:
            category_engagement[category] += 2
        elif event.event_type in (EventType.PURCHASE, EventType.PURCHASE_ITEM):
            category_purchases[category] += 1
            category_engagement[category] += 5

    if not category_engagement:
        return []

    # Normalize engagement scores to 0-1 range
    max_engagement = max(category_engagement.values())

    affinities: list[CategoryAffinity] = []
    for category, engagement in category_engagement.items():
        normalized_score = engagement / max_engagement if max_engagement > 0 else 0.0
        affinities.append(
            CategoryAffinity(
                category=category,
                engagement_score=normalized_score,
                view_count=category_views.get(category, 0),
                purchase_count=category_purchases.get(category, 0),
            )
        )

    # Sort by engagement score descending
    return sorted(affinities, key=lambda x: x.engagement_score, reverse=True)


def aggregate_temporal(
    events: list[EventRecord],
) -> TemporalMetrics:
    """
    Calculate temporal patterns from events.

    Identifies preferred day of week and hour of day based on event frequency.

    Args:
        events: List of event records for a customer

    Returns:
        TemporalMetrics with temporal pattern data
    """
    if not events:
        return TemporalMetrics(
            preferred_day_of_week=None,
            preferred_hour_of_day=None,
            first_seen=None,
            last_seen=None,
        )

    # Count events by day of week and hour
    day_counts: Counter[int] = Counter()
    hour_counts: Counter[int] = Counter()

    for event in events:
        day_counts[event.timestamp.weekday()] += 1
        hour_counts[event.timestamp.hour] += 1

    # Find most common day and hour
    preferred_day = day_counts.most_common(1)[0][0] if day_counts else None
    preferred_hour = hour_counts.most_common(1)[0][0] if hour_counts else None

    # First and last seen
    timestamps = [e.timestamp for e in events]
    first_seen = min(timestamps)
    last_seen = max(timestamps)

    return TemporalMetrics(
        preferred_day_of_week=preferred_day,
        preferred_hour_of_day=preferred_hour,
        first_seen=first_seen,
        last_seen=last_seen,
    )


def aggregate_device(
    events: list[EventRecord],
) -> DeviceMetrics:
    """
    Calculate device/channel metrics from events.

    Args:
        events: List of event records for a customer

    Returns:
        DeviceMetrics with device usage data
    """
    device_counts: Counter[str] = Counter()
    mobile_count = 0
    total_with_device = 0

    for event in events:
        device_type = event.properties.device_type
        if device_type:
            device_counts[device_type] += 1
            total_with_device += 1
            if device_type.lower() in ("mobile", "tablet", "ios", "android"):
                mobile_count += 1

    primary_device_type = None
    if device_counts:
        primary_device_type = device_counts.most_common(1)[0][0]

    mobile_session_ratio = 0.0
    if total_with_device > 0:
        mobile_session_ratio = mobile_count / total_with_device

    return DeviceMetrics(
        primary_device_type=primary_device_type,
        mobile_session_ratio=mobile_session_ratio,
    )


def calculate_clv_estimate(
    purchase_metrics: PurchaseMetrics,
    customer_tenure_days: int,
    *,
    projection_years: int = 3,
    discount_rate: float = 0.1,
) -> Decimal:
    """
    Calculate customer lifetime value estimate.

    Uses a simple CLV model based on purchase frequency and AOV.

    Args:
        purchase_metrics: Aggregated purchase metrics
        customer_tenure_days: Days since customer first seen
        projection_years: Years to project (default 3)
        discount_rate: Annual discount rate (default 10%)

    Returns:
        Estimated CLV as Decimal
    """
    if purchase_metrics["total_purchases"] == 0 or customer_tenure_days <= 0:
        return Decimal("0")

    monthly_frequency = purchase_metrics["purchase_frequency_per_month"]
    aov = purchase_metrics["avg_order_value"]

    # Project monthly revenue
    monthly_revenue = Decimal(str(monthly_frequency)) * aov

    # Sum discounted future value
    clv = Decimal("0")
    monthly_discount = (1 + discount_rate) ** (1 / 12) - 1

    for month in range(projection_years * 12):
        discount_factor = Decimal(str(1 / ((1 + monthly_discount) ** month)))
        clv += monthly_revenue * discount_factor

    return clv.quantize(Decimal("0.01"))


def calculate_churn_risk(
    purchase_metrics: PurchaseMetrics,
    session_metrics: SessionMetrics,
    customer_tenure_days: int,
) -> float:
    """
    Calculate churn risk score (0-1, higher = more risk).

    Factors:
    - Days since last purchase (more days = higher risk)
    - Purchase frequency (lower frequency = higher risk)
    - Session recency (longer gap = higher risk)
    - Cart abandonment (higher abandonment = higher risk)

    Args:
        purchase_metrics: Aggregated purchase metrics
        session_metrics: Aggregated session metrics
        customer_tenure_days: Days since customer first seen

    Returns:
        Churn risk score 0-1
    """
    risk_score = 0.0

    # Days since last purchase (max 180 days = 100% risk contribution)
    days_since = purchase_metrics["days_since_last_purchase"]
    if days_since is not None:
        purchase_recency_risk = min(1.0, days_since / 180.0)
    else:
        # Never purchased - high risk
        purchase_recency_risk = 1.0 if customer_tenure_days > 30 else 0.5

    # Purchase frequency risk (less than 1/month = risky)
    freq = purchase_metrics["purchase_frequency_per_month"]
    frequency_risk = max(0.0, 1.0 - freq) if freq <= 1.0 else 0.0

    # Cart abandonment risk
    abandonment_risk = session_metrics["cart_abandonment_rate"]

    # Engagement risk (no sessions = high risk)
    engagement_risk = 1.0 if session_metrics["total_sessions"] == 0 else 0.0

    # Weighted combination
    risk_score = (
        0.4 * purchase_recency_risk
        + 0.25 * frequency_risk
        + 0.2 * abandonment_risk
        + 0.15 * engagement_risk
    )

    return min(1.0, max(0.0, risk_score))
