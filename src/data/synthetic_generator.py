"""
Module: synthetic_generator

Purpose: Generate Bloomreach EBQ-compatible synthetic data for testing.

Generates realistic synthetic data with:
- Deterministic generation with seed for reproducibility
- Realistic event sequences (view -> cart -> purchase)
- Edge cases (single-event customers, multiple merges)
- Multiple dataset sizes (1k, 10k, 100k)
"""

import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Iterator

import numpy as np

from src.data.schemas import (
    CustomerIdHistory,
    CustomerProperties,
    EventProperties,
    EventRecord,
    EventType,
    SyntheticDataset,
)


# =============================================================================
# CONSTANTS
# =============================================================================

PRODUCT_CATEGORIES = [
    "electronics",
    "clothing",
    "home_garden",
    "sports",
    "books",
    "beauty",
    "toys",
    "food",
    "automotive",
    "jewelry",
]

PRODUCT_NAMES = {
    "electronics": ["Laptop Pro", "Wireless Earbuds", "Smart Watch", "4K TV", "Gaming Console"],
    "clothing": ["Winter Jacket", "Running Shoes", "Denim Jeans", "Silk Dress", "Wool Sweater"],
    "home_garden": ["Coffee Maker", "Garden Tools Set", "Bed Sheets", "Air Purifier", "Plant Pot"],
    "sports": ["Yoga Mat", "Dumbbells Set", "Tennis Racket", "Running Belt", "Bicycle Helmet"],
    "books": ["Bestseller Novel", "Cookbook", "Self-Help Guide", "History Book", "Science Text"],
    "beauty": ["Face Cream", "Perfume", "Makeup Kit", "Hair Dryer", "Skincare Set"],
    "toys": ["Building Blocks", "Board Game", "Plush Toy", "Puzzle Set", "Remote Car"],
    "food": ["Gourmet Coffee", "Chocolate Box", "Spice Set", "Organic Snacks", "Wine Selection"],
    "automotive": ["Car Charger", "Floor Mats", "Phone Mount", "Air Freshener", "Seat Covers"],
    "jewelry": ["Silver Necklace", "Watch Band", "Earrings Set", "Ring", "Bracelet"],
}

DEVICE_TYPES = ["desktop", "mobile", "tablet"]
BROWSERS = ["Chrome", "Safari", "Firefox", "Edge"]
COUNTRIES = ["US", "UK", "DE", "FR", "CA", "AU", "NL", "ES", "IT", "JP"]
CITIES = ["New York", "London", "Berlin", "Paris", "Toronto", "Sydney", "Amsterdam", "Madrid"]
LOYALTY_TIERS = ["bronze", "silver", "gold", "platinum", None]


# =============================================================================
# CUSTOMER BEHAVIOR PROFILES
# =============================================================================


class CustomerBehaviorType:
    """Define different customer behavior archetypes."""

    # High-value active customer
    HIGH_VALUE_ACTIVE = "high_value_active"
    # High-value but dormant
    HIGH_VALUE_DORMANT = "high_value_dormant"
    # Frequent low-value purchaser
    FREQUENT_LOW_VALUE = "frequent_low_value"
    # Browser but rarely buys
    HIGH_INTENT_BROWSER = "high_intent_browser"
    # Cart abandoner
    CART_ABANDONER = "cart_abandoner"
    # One-time buyer
    ONE_TIME_BUYER = "one_time_buyer"
    # Discount hunter
    DISCOUNT_HUNTER = "discount_hunter"
    # Weekend shopper
    WEEKEND_SHOPPER = "weekend_shopper"
    # Single event (edge case)
    SINGLE_EVENT = "single_event"
    # Refund-heavy (edge case)
    REFUND_HEAVY = "refund_heavy"


BEHAVIOR_WEIGHTS = {
    CustomerBehaviorType.HIGH_VALUE_ACTIVE: 0.10,
    CustomerBehaviorType.HIGH_VALUE_DORMANT: 0.08,
    CustomerBehaviorType.FREQUENT_LOW_VALUE: 0.12,
    CustomerBehaviorType.HIGH_INTENT_BROWSER: 0.15,
    CustomerBehaviorType.CART_ABANDONER: 0.15,
    CustomerBehaviorType.ONE_TIME_BUYER: 0.15,
    CustomerBehaviorType.DISCOUNT_HUNTER: 0.08,
    CustomerBehaviorType.WEEKEND_SHOPPER: 0.10,
    CustomerBehaviorType.SINGLE_EVENT: 0.05,
    CustomerBehaviorType.REFUND_HEAVY: 0.02,
}


# =============================================================================
# SYNTHETIC DATA GENERATOR
# =============================================================================


class SyntheticDataGenerator:
    """
    Generate Bloomreach EBQ-compatible synthetic data for testing.

    Uses numpy random generator with seed for reproducibility.
    """

    def __init__(self, *, seed: int = 42) -> None:
        """
        Initialize generator with seed.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._event_counter = 0
        self._customer_counter = 0

    def _generate_id(self, *, prefix: str) -> str:
        """Generate a unique ID with prefix."""
        return f"{prefix}_{uuid.UUID(int=self.rng.integers(0, 2**128)).hex[:12]}"

    def _generate_customer_id(self) -> str:
        """Generate a unique customer ID."""
        self._customer_counter += 1
        return f"cust_{self._customer_counter:08d}_{self.rng.integers(1000, 9999)}"

    def _generate_event_id(self) -> str:
        """Generate a unique event ID."""
        self._event_counter += 1
        return f"evt_{self._event_counter:010d}"

    def _random_timestamp(
        self,
        *,
        start: datetime,
        end: datetime,
    ) -> datetime:
        """Generate random timestamp between start and end."""
        delta = (end - start).total_seconds()
        random_seconds = self.rng.random() * delta
        return start + timedelta(seconds=random_seconds)

    def _random_timestamp_with_preference(
        self,
        *,
        start: datetime,
        end: datetime,
        prefer_weekends: bool = False,
        prefer_evenings: bool = False,
    ) -> datetime:
        """Generate timestamp with behavioral preferences."""
        base_ts = self._random_timestamp(start=start, end=end)

        if prefer_weekends:
            # Shift toward weekends (5=Saturday, 6=Sunday)
            days_to_weekend = (5 - base_ts.weekday()) % 7
            if self.rng.random() > 0.3:  # 70% weekend preference
                base_ts = base_ts + timedelta(days=days_to_weekend)

        if prefer_evenings:
            # Shift toward evening hours (18-22)
            if self.rng.random() > 0.4:  # 60% evening preference
                evening_hour = int(self.rng.integers(18, 23))
                base_ts = base_ts.replace(hour=evening_hour)

        # Ensure within bounds
        if base_ts > end:
            base_ts = end - timedelta(hours=int(self.rng.integers(1, 48)))
        if base_ts < start:
            base_ts = start + timedelta(hours=int(self.rng.integers(1, 48)))

        return base_ts

    def _select_behavior_type(self) -> str:
        """Select a customer behavior type based on weights."""
        types = list(BEHAVIOR_WEIGHTS.keys())
        weights = list(BEHAVIOR_WEIGHTS.values())
        return str(self.rng.choice(types, p=weights))

    def _generate_product(self) -> tuple[str, str, str, Decimal]:
        """Generate random product (id, name, category, price)."""
        category = str(self.rng.choice(PRODUCT_CATEGORIES))
        name = str(self.rng.choice(PRODUCT_NAMES[category]))

        # Price based on category
        base_prices = {
            "electronics": (100, 2000),
            "clothing": (20, 300),
            "home_garden": (15, 500),
            "sports": (20, 400),
            "books": (10, 50),
            "beauty": (15, 200),
            "toys": (10, 100),
            "food": (5, 100),
            "automotive": (10, 200),
            "jewelry": (50, 1000),
        }
        min_price, max_price = base_prices[category]
        price = Decimal(str(round(self.rng.uniform(min_price, max_price), 2)))

        product_id = f"prod_{category[:3]}_{self.rng.integers(1000, 9999)}"
        return product_id, name, category, price

    def generate_customer_properties(
        self,
        *,
        customer_id: str,
        registration_date: datetime | None = None,
    ) -> CustomerProperties:
        """Generate customer properties for a single customer."""
        first_names = ["John", "Jane", "Mike", "Sarah", "David", "Emma", "Chris", "Lisa"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller"]

        return CustomerProperties(
            internal_customer_id=customer_id,
            email=f"{customer_id}@example.com",
            first_name=str(self.rng.choice(first_names)),
            last_name=str(self.rng.choice(last_names)),
            registration_date=registration_date,
            country=str(self.rng.choice(COUNTRIES)),
            city=str(self.rng.choice(CITIES)),
            loyalty_tier=self.rng.choice(LOYALTY_TIERS),  # type: ignore[arg-type]
        )

    def _generate_session_start(
        self,
        *,
        customer_id: str,
        timestamp: datetime,
    ) -> EventRecord:
        """Generate a session_start event."""
        return EventRecord(
            event_id=self._generate_event_id(),
            internal_customer_id=customer_id,
            event_type=EventType.SESSION_START,
            timestamp=timestamp,
            properties=EventProperties(
                device_type=str(self.rng.choice(DEVICE_TYPES)),
                browser=str(self.rng.choice(BROWSERS)),
            ),
        )

    def _generate_view_item(
        self,
        *,
        customer_id: str,
        timestamp: datetime,
        product_id: str,
        product_name: str,
        category: str,
        price: Decimal,
    ) -> EventRecord:
        """Generate a view_item event."""
        return EventRecord(
            event_id=self._generate_event_id(),
            internal_customer_id=customer_id,
            event_type=EventType.VIEW_ITEM,
            timestamp=timestamp,
            properties=EventProperties(
                product_id=product_id,
                product_name=product_name,
                product_category=category,
                product_price=price,
            ),
        )

    def _generate_add_to_cart(
        self,
        *,
        customer_id: str,
        timestamp: datetime,
        product_id: str,
        product_name: str,
        category: str,
        price: Decimal,
        quantity: int = 1,
    ) -> EventRecord:
        """Generate an add_to_cart event."""
        return EventRecord(
            event_id=self._generate_event_id(),
            internal_customer_id=customer_id,
            event_type=EventType.ADD_TO_CART,
            timestamp=timestamp,
            properties=EventProperties(
                product_id=product_id,
                product_name=product_name,
                product_category=category,
                product_price=price,
                quantity=quantity,
                cart_id=f"cart_{customer_id}_{self.rng.integers(1000, 9999)}",
            ),
        )

    def _generate_purchase(
        self,
        *,
        customer_id: str,
        timestamp: datetime,
        total_amount: Decimal,
        discount_amount: Decimal | None = None,
        order_id: str | None = None,
    ) -> EventRecord:
        """Generate a purchase event."""
        return EventRecord(
            event_id=self._generate_event_id(),
            internal_customer_id=customer_id,
            event_type=EventType.PURCHASE,
            timestamp=timestamp,
            properties=EventProperties(
                order_id=order_id or f"order_{self.rng.integers(100000, 999999)}",
                total_amount=total_amount,
                discount_amount=discount_amount,
                currency="USD",
            ),
        )

    def _generate_purchase_item(
        self,
        *,
        customer_id: str,
        timestamp: datetime,
        order_id: str,
        product_id: str,
        product_name: str,
        category: str,
        price: Decimal,
        quantity: int = 1,
    ) -> EventRecord:
        """Generate a purchase_item event for category-level purchase tracking."""
        return EventRecord(
            event_id=self._generate_event_id(),
            internal_customer_id=customer_id,
            event_type=EventType.PURCHASE_ITEM,
            timestamp=timestamp,
            properties=EventProperties(
                order_id=order_id,
                product_id=product_id,
                product_name=product_name,
                product_category=category,
                product_price=price,
                quantity=quantity,
            ),
        )

    def _generate_refund(
        self,
        *,
        customer_id: str,
        timestamp: datetime,
        order_id: str,
        refund_amount: Decimal,
    ) -> EventRecord:
        """Generate a refund event."""
        return EventRecord(
            event_id=self._generate_event_id(),
            internal_customer_id=customer_id,
            event_type=EventType.REFUND,
            timestamp=timestamp,
            properties=EventProperties(
                order_id=order_id,
                total_amount=refund_amount,
            ),
        )

    def _generate_customer_journey(
        self,
        *,
        customer_id: str,
        behavior_type: str,
        date_range: tuple[datetime, datetime],
    ) -> list[EventRecord]:
        """Generate events for a single customer based on behavior type."""
        events: list[EventRecord] = []
        start_date, end_date = date_range

        prefer_weekends = behavior_type == CustomerBehaviorType.WEEKEND_SHOPPER
        prefer_evenings = behavior_type in [
            CustomerBehaviorType.WEEKEND_SHOPPER,
            CustomerBehaviorType.HIGH_VALUE_ACTIVE,
        ]

        # Determine number of sessions based on behavior
        session_counts = {
            CustomerBehaviorType.HIGH_VALUE_ACTIVE: (15, 30),
            CustomerBehaviorType.HIGH_VALUE_DORMANT: (5, 10),
            CustomerBehaviorType.FREQUENT_LOW_VALUE: (20, 40),
            CustomerBehaviorType.HIGH_INTENT_BROWSER: (15, 35),
            CustomerBehaviorType.CART_ABANDONER: (10, 25),
            CustomerBehaviorType.ONE_TIME_BUYER: (2, 5),
            CustomerBehaviorType.DISCOUNT_HUNTER: (8, 20),
            CustomerBehaviorType.WEEKEND_SHOPPER: (10, 20),
            CustomerBehaviorType.SINGLE_EVENT: (1, 1),
            CustomerBehaviorType.REFUND_HEAVY: (8, 15),
        }

        min_sessions, max_sessions = session_counts.get(behavior_type, (5, 15))
        n_sessions = int(self.rng.integers(min_sessions, max_sessions + 1))

        # For dormant customers, concentrate sessions in first half of period
        if behavior_type == CustomerBehaviorType.HIGH_VALUE_DORMANT:
            mid_point = start_date + (end_date - start_date) / 3
            session_end = mid_point
        else:
            session_end = end_date

        # Purchase probability based on behavior
        purchase_probs = {
            CustomerBehaviorType.HIGH_VALUE_ACTIVE: 0.6,
            CustomerBehaviorType.HIGH_VALUE_DORMANT: 0.5,
            CustomerBehaviorType.FREQUENT_LOW_VALUE: 0.7,
            CustomerBehaviorType.HIGH_INTENT_BROWSER: 0.1,
            CustomerBehaviorType.CART_ABANDONER: 0.15,
            CustomerBehaviorType.ONE_TIME_BUYER: 0.8,
            CustomerBehaviorType.DISCOUNT_HUNTER: 0.4,
            CustomerBehaviorType.WEEKEND_SHOPPER: 0.5,
            CustomerBehaviorType.SINGLE_EVENT: 0.0,
            CustomerBehaviorType.REFUND_HEAVY: 0.6,
        }
        purchase_prob = purchase_probs.get(behavior_type, 0.3)

        order_ids: list[str] = []  # Track for refunds
        order_amounts: list[Decimal] = []

        for _ in range(n_sessions):
            session_ts = self._random_timestamp_with_preference(
                start=start_date,
                end=session_end,
                prefer_weekends=prefer_weekends,
                prefer_evenings=prefer_evenings,
            )

            # Session start
            events.append(
                self._generate_session_start(
                    customer_id=customer_id,
                    timestamp=session_ts,
                )
            )

            # Product views (2-8 per session)
            n_views = int(self.rng.integers(2, 9))
            cart_items: list[tuple[str, str, str, Decimal]] = []

            for i in range(n_views):
                view_ts = session_ts + timedelta(minutes=int(self.rng.integers(1, 15) * (i + 1)))
                product_id, product_name, category, price = self._generate_product()

                events.append(
                    self._generate_view_item(
                        customer_id=customer_id,
                        timestamp=view_ts,
                        product_id=product_id,
                        product_name=product_name,
                        category=category,
                        price=price,
                    )
                )

                # Add to cart probability
                cart_prob = 0.3 if behavior_type != CustomerBehaviorType.CART_ABANDONER else 0.6
                if self.rng.random() < cart_prob:
                    cart_ts = view_ts + timedelta(minutes=int(self.rng.integers(1, 5)))
                    events.append(
                        self._generate_add_to_cart(
                            customer_id=customer_id,
                            timestamp=cart_ts,
                            product_id=product_id,
                            product_name=product_name,
                            category=category,
                            price=price,
                            quantity=int(self.rng.integers(1, 3)),
                        )
                    )
                    cart_items.append((product_id, product_name, category, price))

            # Purchase decision
            if cart_items and self.rng.random() < purchase_prob:
                purchase_ts = session_ts + timedelta(minutes=int(self.rng.integers(30, 90)))

                # Calculate total
                total = sum(item[3] for item in cart_items)

                # Apply discount for discount hunters
                discount = None
                if behavior_type == CustomerBehaviorType.DISCOUNT_HUNTER:
                    discount_pct = Decimal(str(self.rng.uniform(0.1, 0.3)))
                    discount = total * discount_pct
                    total = total - discount

                # Adjust order value based on behavior
                if behavior_type == CustomerBehaviorType.HIGH_VALUE_ACTIVE:
                    total = total * Decimal("1.5")  # Higher value orders
                elif behavior_type == CustomerBehaviorType.HIGH_VALUE_DORMANT:
                    total = total * Decimal("2.0")  # Even higher historical value
                elif behavior_type == CustomerBehaviorType.FREQUENT_LOW_VALUE:
                    total = total * Decimal("0.5")  # Lower value orders

                order_id = f"order_{self.rng.integers(100000, 999999)}"

                purchase_event = self._generate_purchase(
                    customer_id=customer_id,
                    timestamp=purchase_ts,
                    total_amount=total.quantize(Decimal("0.01")),
                    discount_amount=discount.quantize(Decimal("0.01")) if discount else None,
                    order_id=order_id,
                )
                events.append(purchase_event)

                # Generate purchase_item events for each cart item (for category tracking)
                for item_id, item_name, item_category, item_price in cart_items:
                    events.append(
                        self._generate_purchase_item(
                            customer_id=customer_id,
                            timestamp=purchase_ts + timedelta(seconds=1),
                            order_id=order_id,
                            product_id=item_id,
                            product_name=item_name,
                            category=item_category,
                            price=item_price,
                        )
                    )

                order_ids.append(order_id)
                order_amounts.append(total)

        # Add refunds for refund-heavy customers
        if behavior_type == CustomerBehaviorType.REFUND_HEAVY and order_ids:
            n_refunds = min(len(order_ids), int(self.rng.integers(2, 5)))
            for i in range(n_refunds):
                if i < len(order_ids):
                    refund_ts = events[-1].timestamp + timedelta(days=int(self.rng.integers(3, 14)))
                    if refund_ts < end_date:
                        events.append(
                            self._generate_refund(
                                customer_id=customer_id,
                                timestamp=refund_ts,
                                order_id=order_ids[i],
                                refund_amount=order_amounts[i],
                            )
                        )

        # Sort events by timestamp
        events.sort(key=lambda e: e.timestamp)
        return events

    def generate_merge_history(
        self,
        *,
        customer_ids: list[str],
        merge_probability: float = 0.15,
        date_range: tuple[datetime, datetime],
    ) -> tuple[list[CustomerIdHistory], dict[str, str]]:
        """
        Generate customer ID merge history.

        Args:
            customer_ids: List of canonical customer IDs
            merge_probability: Probability that a customer has merge history
            date_range: Date range for merge timestamps

        Returns:
            Tuple of (list of CustomerIdHistory, mapping of past_id -> canonical_id)
        """
        history: list[CustomerIdHistory] = []
        merge_map: dict[str, str] = {}  # past_id -> canonical_id

        for canonical_id in customer_ids:
            if self.rng.random() < merge_probability:
                # Generate 1-3 past IDs that merged into this customer
                n_past_ids = int(self.rng.integers(1, 4))

                for _ in range(n_past_ids):
                    past_id = self._generate_customer_id()
                    merge_ts = self._random_timestamp(
                        start=date_range[0],
                        end=date_range[1],
                    )

                    history.append(
                        CustomerIdHistory(
                            internal_customer_id=canonical_id,
                            past_id=past_id,
                            merge_timestamp=merge_ts,
                        )
                    )
                    merge_map[past_id] = canonical_id

        return history, merge_map

    def generate_events(
        self,
        *,
        customer_ids: list[str],
        date_range: tuple[datetime, datetime],
    ) -> Iterator[EventRecord]:
        """
        Generate events for all customers.

        Args:
            customer_ids: List of customer IDs
            date_range: Start and end dates for events

        Yields:
            EventRecord objects
        """
        for customer_id in customer_ids:
            behavior_type = self._select_behavior_type()
            events = self._generate_customer_journey(
                customer_id=customer_id,
                behavior_type=behavior_type,
                date_range=date_range,
            )
            yield from events

    def generate_dataset(
        self,
        *,
        n_customers: int,
        date_range: tuple[datetime, datetime],
        merge_probability: float = 0.15,
    ) -> SyntheticDataset:
        """
        Generate complete synthetic dataset.

        Args:
            n_customers: Number of customers to generate
            date_range: Start and end dates for the dataset
            merge_probability: Probability of customer having merge history

        Returns:
            SyntheticDataset with events, customer_properties, and id_history
        """
        # Reset counters for reproducibility
        self._event_counter = 0
        self._customer_counter = 0

        # Generate customer IDs
        customer_ids = [self._generate_customer_id() for _ in range(n_customers)]

        # Generate customer properties
        customer_properties = [
            self.generate_customer_properties(
                customer_id=cid,
                registration_date=self._random_timestamp(
                    start=date_range[0] - timedelta(days=365),
                    end=date_range[0],
                ),
            )
            for cid in customer_ids
        ]

        # Generate merge history
        id_history, merge_map = self.generate_merge_history(
            customer_ids=customer_ids,
            merge_probability=merge_probability,
            date_range=date_range,
        )

        # Generate events
        events = list(
            self.generate_events(
                customer_ids=customer_ids,
                date_range=date_range,
            )
        )

        # Also generate some events for "past" IDs (pre-merge)
        past_events: list[EventRecord] = []
        for past_id, canonical_id in merge_map.items():
            # Find merge timestamp
            merge_ts = None
            for hist in id_history:
                if hist.past_id == past_id:
                    merge_ts = hist.merge_timestamp
                    break

            if merge_ts:
                # Generate a few events before merge
                behavior_type = self._select_behavior_type()
                pre_merge_events = self._generate_customer_journey(
                    customer_id=past_id,
                    behavior_type=behavior_type,
                    date_range=(date_range[0], merge_ts - timedelta(hours=1)),
                )
                # Limit pre-merge events
                past_events.extend(pre_merge_events[:5])

        events.extend(past_events)

        # Sort all events by timestamp
        events.sort(key=lambda e: e.timestamp)

        # Calculate statistics
        event_type_counts: Counter[str] = Counter()
        for event in events:
            event_type_counts[event.event_type.value] += 1

        return SyntheticDataset(
            seed=self.seed,
            n_customers=n_customers,
            date_range_start=date_range[0],
            date_range_end=date_range[1],
            events=events,
            customer_properties=customer_properties,
            id_history=id_history,
            n_events=len(events),
            n_merges=len(id_history),
            event_type_distribution=dict(event_type_counts),
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def generate_small_dataset(*, seed: int = 42) -> SyntheticDataset:
    """Generate small dataset (1,000 customers) for unit tests."""
    generator = SyntheticDataGenerator(seed=seed)
    return generator.generate_dataset(
        n_customers=1000,
        date_range=(
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 6, 30, tzinfo=timezone.utc),
        ),
        merge_probability=0.15,
    )


def generate_medium_dataset(*, seed: int = 42) -> SyntheticDataset:
    """Generate medium dataset (10,000 customers) for integration tests."""
    generator = SyntheticDataGenerator(seed=seed)
    return generator.generate_dataset(
        n_customers=10000,
        date_range=(
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 6, 30, tzinfo=timezone.utc),
        ),
        merge_probability=0.15,
    )


def generate_large_dataset(*, seed: int = 42) -> SyntheticDataset:
    """Generate large dataset (100,000 customers) for performance tests."""
    generator = SyntheticDataGenerator(seed=seed)
    return generator.generate_dataset(
        n_customers=100000,
        date_range=(
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 6, 30, tzinfo=timezone.utc),
        ),
        merge_probability=0.10,
    )


def preview_dataset(dataset: SyntheticDataset, *, n_samples: int = 5) -> dict[str, list[str]]:
    """Preview a sample of the dataset for debugging."""
    return {
        "events": [repr(e) for e in dataset.events[:n_samples]],
        "customers": [repr(c) for c in dataset.customer_properties[:n_samples]],
        "merges": [repr(h) for h in dataset.id_history[:n_samples]],
    }


def dataset_statistics(dataset: SyntheticDataset) -> dict[str, int | float | dict[str, int]]:
    """Get summary statistics for a dataset."""
    purchase_events = [e for e in dataset.events if e.event_type == EventType.PURCHASE]
    total_revenue = sum(
        e.properties.total_amount for e in purchase_events if e.properties.total_amount
    )

    return {
        "n_customers": dataset.n_customers,
        "n_events": dataset.n_events,
        "n_merges": dataset.n_merges,
        "n_purchases": len(purchase_events),
        "total_revenue": float(total_revenue),
        "avg_events_per_customer": dataset.n_events / dataset.n_customers if dataset.n_customers else 0,
        "event_distribution": dataset.event_type_distribution,
    }
