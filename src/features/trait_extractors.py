"""
Module: trait_extractors

Purpose: Extract actionable traits from customer profiles.

Traits are business-meaningful characteristics that map to actionable dimensions:
- WHAT: Product/offer targeting
- WHEN: Timing optimization
- HOW: Channel/message personalization
- WHO: Prioritization

Each trait supports concrete business hypotheses and marketing actions.
No technical jargon (PCA, embeddings, clusters) - only business terms.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Any

from src.data.schemas import (
    ActionabilityDimension,
    ActionableTrait,
    BehaviorType,
    CustomerProfile,
    CustomerTraits,
)


class TraitExtractor(ABC):
    """Base class for trait extraction."""

    @property
    @abstractmethod
    def trait_name(self) -> str:
        """Name of the trait this extractor produces."""
        ...

    @property
    @abstractmethod
    def actionability_dimension(self) -> ActionabilityDimension:
        """Which actionability dimension this trait supports."""
        ...

    @property
    @abstractmethod
    def business_relevance(self) -> str:
        """Why this trait matters for business."""
        ...

    @abstractmethod
    def extract(self, profile: CustomerProfile) -> ActionableTrait | None:
        """
        Extract trait from customer profile.

        Args:
            profile: CustomerProfile to analyze

        Returns:
            ActionableTrait if trait is present, None otherwise
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(trait={self.trait_name!r})"


# =============================================================================
# VALUE-BASED EXTRACTORS (WHO - Prioritization)
# =============================================================================


class HighValueCustomerExtractor(TraitExtractor):
    """Identify high-value customers based on CLV and revenue."""

    def __init__(
        self,
        *,
        clv_threshold: Decimal = Decimal("500"),
        revenue_threshold: Decimal = Decimal("200"),
    ) -> None:
        self.clv_threshold = clv_threshold
        self.revenue_threshold = revenue_threshold

    @property
    def trait_name(self) -> str:
        return "high_value_customer"

    @property
    def actionability_dimension(self) -> ActionabilityDimension:
        return ActionabilityDimension.WHO

    @property
    def business_relevance(self) -> str:
        return "Prioritize VIP treatment and retention efforts for highest value customers"

    def extract(self, profile: CustomerProfile) -> ActionableTrait | None:
        is_high_value = (
            profile.clv_estimate >= self.clv_threshold
            or profile.total_revenue >= self.revenue_threshold
        )

        if not is_high_value:
            return None

        return ActionableTrait(
            name=self.trait_name,
            description="Customer with high lifetime value or significant revenue history",
            value=True,
            actionability_dimension=self.actionability_dimension,
            business_relevance=self.business_relevance,
        )


class ChurnRiskExtractor(TraitExtractor):
    """Identify customers at risk of churning."""

    def __init__(self, *, risk_threshold: float = 0.6) -> None:
        self.risk_threshold = risk_threshold

    @property
    def trait_name(self) -> str:
        return "churn_risk"

    @property
    def actionability_dimension(self) -> ActionabilityDimension:
        return ActionabilityDimension.WHO

    @property
    def business_relevance(self) -> str:
        return "Prioritize retention campaigns before customer churns"

    def extract(self, profile: CustomerProfile) -> ActionableTrait | None:
        if profile.churn_risk_score < self.risk_threshold:
            return None

        risk_level = "high" if profile.churn_risk_score >= 0.8 else "moderate"

        return ActionableTrait(
            name=self.trait_name,
            description=f"Customer showing {risk_level} risk of churning",
            value=risk_level,
            actionability_dimension=self.actionability_dimension,
            business_relevance=self.business_relevance,
        )


class HighValueDormantExtractor(TraitExtractor):
    """Identify high-value customers who have become dormant."""

    def __init__(
        self,
        *,
        dormant_days: int = 60,
        min_revenue: Decimal = Decimal("200"),
    ) -> None:
        self.dormant_days = dormant_days
        self.min_revenue = min_revenue

    @property
    def trait_name(self) -> str:
        return "high_value_dormant"

    @property
    def actionability_dimension(self) -> ActionabilityDimension:
        return ActionabilityDimension.WHO

    @property
    def business_relevance(self) -> str:
        return "Re-engage valuable customers before they churn permanently"

    def extract(self, profile: CustomerProfile) -> ActionableTrait | None:
        is_dormant = (
            profile.days_since_last_purchase is not None
            and profile.days_since_last_purchase >= self.dormant_days
        )
        is_high_value = profile.total_revenue >= self.min_revenue

        if not (is_dormant and is_high_value):
            return None

        return ActionableTrait(
            name=self.trait_name,
            description=f"Previously valuable customer inactive for {profile.days_since_last_purchase} days",
            value=True,
            actionability_dimension=self.actionability_dimension,
            business_relevance=self.business_relevance,
        )


class PurchaseBehaviorTypeExtractor(TraitExtractor):
    """Classify customer purchase behavior pattern for CLV prediction.

    Uses purchase interval statistics to classify customers as:
    - REGULAR: Predictable purchase timing (CV < 0.5)
    - IRREGULAR: Variable purchase timing (CV 0.5-1.0)
    - LONG_CYCLE: Infrequent/seasonal purchaser (CV > 1.0 or interval > 90 days)
    - ONE_TIME: Single purchase only
    - NEW: Insufficient data (< 2 purchases)
    """

    def __init__(
        self,
        *,
        regular_cv_threshold: float = 0.5,
        irregular_cv_threshold: float = 1.0,
        long_cycle_interval_days: float = 90.0,
    ) -> None:
        self.regular_cv_threshold = regular_cv_threshold
        self.irregular_cv_threshold = irregular_cv_threshold
        self.long_cycle_interval_days = long_cycle_interval_days

    @property
    def trait_name(self) -> str:
        return "purchase_behavior_type"

    @property
    def actionability_dimension(self) -> ActionabilityDimension:
        return ActionabilityDimension.WHO

    @property
    def business_relevance(self) -> str:
        return "Tailor engagement strategy based on purchase predictability"

    def classify_behavior(self, profile: CustomerProfile) -> BehaviorType:
        """Determine behavior type from profile data."""
        # Not enough purchases to classify
        if profile.total_purchases == 0:
            return BehaviorType.NEW

        if profile.total_purchases == 1:
            return BehaviorType.ONE_TIME

        # Need interval data for classification
        intervals = profile.purchase_intervals
        if intervals is None:
            return BehaviorType.NEW

        # Check for long-cycle first (mean interval > threshold)
        if intervals.interval_mean is not None:
            if intervals.interval_mean > self.long_cycle_interval_days:
                return BehaviorType.LONG_CYCLE

        # Classify by coefficient of variation
        if intervals.interval_cv is not None:
            if intervals.interval_cv < self.regular_cv_threshold:
                return BehaviorType.REGULAR
            elif intervals.interval_cv < self.irregular_cv_threshold:
                return BehaviorType.IRREGULAR
            else:
                return BehaviorType.LONG_CYCLE

        # Fallback: If we have 2+ purchases but no CV, treat as new
        return BehaviorType.NEW

    def extract(self, profile: CustomerProfile) -> ActionableTrait | None:
        behavior_type = self.classify_behavior(profile)

        # Always return a trait - behavior type is informative even for NEW
        descriptions = {
            BehaviorType.REGULAR: "Predictable purchase pattern - good for subscription offers",
            BehaviorType.IRREGULAR: "Variable purchase timing - respond to triggers/promotions",
            BehaviorType.LONG_CYCLE: "Infrequent purchaser - seasonal or considered purchases",
            BehaviorType.ONE_TIME: "Single purchase only - needs reactivation",
            BehaviorType.NEW: "New or insufficient history - needs nurturing",
        }

        return ActionableTrait(
            name=self.trait_name,
            description=descriptions.get(behavior_type, "Unknown pattern"),
            value=behavior_type.value,
            actionability_dimension=self.actionability_dimension,
            business_relevance=self.business_relevance,
        )


# =============================================================================
# BEHAVIOR-BASED EXTRACTORS (WHAT - Product/Offer Targeting)
# =============================================================================


class CartAbandonerExtractor(TraitExtractor):
    """Identify customers with cart abandonment patterns."""

    def __init__(
        self,
        *,
        abandonment_threshold: float = 0.5,
        min_cart_additions: int = 2,
    ) -> None:
        self.abandonment_threshold = abandonment_threshold
        self.min_cart_additions = min_cart_additions

    @property
    def trait_name(self) -> str:
        return "cart_abandoner"

    @property
    def actionability_dimension(self) -> ActionabilityDimension:
        return ActionabilityDimension.WHAT

    @property
    def business_relevance(self) -> str:
        return "Target with cart recovery emails and incentives to complete purchase"

    def extract(self, profile: CustomerProfile) -> ActionableTrait | None:
        if profile.total_cart_additions < self.min_cart_additions:
            return None

        if profile.cart_abandonment_rate < self.abandonment_threshold:
            return None

        return ActionableTrait(
            name=self.trait_name,
            description=f"Abandons {profile.cart_abandonment_rate * 100:.0f}% of carts",
            value=profile.cart_abandonment_rate,
            actionability_dimension=self.actionability_dimension,
            business_relevance=self.business_relevance,
        )


class CategorySpecialistExtractor(TraitExtractor):
    """Identify customers with strong category preference."""

    def __init__(
        self,
        *,
        affinity_threshold: float = 0.7,
        min_purchase_count: int = 2,
    ) -> None:
        self.affinity_threshold = affinity_threshold
        self.min_purchase_count = min_purchase_count

    @property
    def trait_name(self) -> str:
        return "category_specialist"

    @property
    def actionability_dimension(self) -> ActionabilityDimension:
        return ActionabilityDimension.WHAT

    @property
    def business_relevance(self) -> str:
        return "Target with category-specific offers and new arrivals in preferred category"

    def extract(self, profile: CustomerProfile) -> ActionableTrait | None:
        if not profile.category_affinities:
            return None

        top_affinity = profile.category_affinities[0]

        if top_affinity.engagement_score < self.affinity_threshold:
            return None

        if top_affinity.purchase_count < self.min_purchase_count:
            return None

        return ActionableTrait(
            name=self.trait_name,
            description=f"Strong preference for {top_affinity.category} category",
            value=top_affinity.category,
            actionability_dimension=self.actionability_dimension,
            business_relevance=self.business_relevance,
        )


class DiscountSensitiveExtractor(TraitExtractor):
    """Identify customers who respond to discounts."""

    def __init__(
        self,
        *,
        low_aov_threshold: Decimal = Decimal("50"),
        high_frequency_threshold: float = 0.5,
    ) -> None:
        self.low_aov_threshold = low_aov_threshold
        self.high_frequency_threshold = high_frequency_threshold

    @property
    def trait_name(self) -> str:
        return "discount_sensitive"

    @property
    def actionability_dimension(self) -> ActionabilityDimension:
        return ActionabilityDimension.WHAT

    @property
    def business_relevance(self) -> str:
        return "Target with promotional offers and sale notifications"

    def extract(self, profile: CustomerProfile) -> ActionableTrait | None:
        # Proxy for discount sensitivity: low AOV but high purchase frequency
        if profile.total_purchases < 2:
            return None

        is_low_aov = profile.avg_order_value <= self.low_aov_threshold
        is_frequent = profile.purchase_frequency_per_month >= self.high_frequency_threshold

        if not (is_low_aov and is_frequent):
            return None

        return ActionableTrait(
            name=self.trait_name,
            description="Frequent small purchases suggest discount-driven behavior",
            value=True,
            actionability_dimension=self.actionability_dimension,
            business_relevance=self.business_relevance,
        )


class BrowserNotBuyerExtractor(TraitExtractor):
    """Identify customers who browse heavily but rarely purchase."""

    def __init__(
        self,
        *,
        min_views: int = 10,
        max_purchases: int = 1,
    ) -> None:
        self.min_views = min_views
        self.max_purchases = max_purchases

    @property
    def trait_name(self) -> str:
        return "browser_not_buyer"

    @property
    def actionability_dimension(self) -> ActionabilityDimension:
        return ActionabilityDimension.WHAT

    @property
    def business_relevance(self) -> str:
        return "Target with first-purchase incentives and personalized recommendations"

    def extract(self, profile: CustomerProfile) -> ActionableTrait | None:
        is_heavy_browser = profile.total_items_viewed >= self.min_views
        is_low_purchaser = profile.total_purchases <= self.max_purchases

        if not (is_heavy_browser and is_low_purchaser):
            return None

        return ActionableTrait(
            name=self.trait_name,
            description=f"Viewed {profile.total_items_viewed} items but only {profile.total_purchases} purchase(s)",
            value=True,
            actionability_dimension=self.actionability_dimension,
            business_relevance=self.business_relevance,
        )


# =============================================================================
# TEMPORAL EXTRACTORS (WHEN - Timing Optimization)
# =============================================================================


class WeekendShopperExtractor(TraitExtractor):
    """Identify customers who prefer weekend shopping."""

    @property
    def trait_name(self) -> str:
        return "weekend_shopper"

    @property
    def actionability_dimension(self) -> ActionabilityDimension:
        return ActionabilityDimension.WHEN

    @property
    def business_relevance(self) -> str:
        return "Schedule marketing communications for Friday/Saturday for best engagement"

    def extract(self, profile: CustomerProfile) -> ActionableTrait | None:
        if profile.preferred_day_of_week is None:
            return None

        # Weekend is Saturday (5) or Sunday (6)
        is_weekend_shopper = profile.preferred_day_of_week in (5, 6)

        if not is_weekend_shopper:
            return None

        day_name = "Saturday" if profile.preferred_day_of_week == 5 else "Sunday"

        return ActionableTrait(
            name=self.trait_name,
            description=f"Prefers shopping on {day_name}",
            value=day_name,
            actionability_dimension=self.actionability_dimension,
            business_relevance=self.business_relevance,
        )


class EarlyBirdShopperExtractor(TraitExtractor):
    """Identify customers who shop early in the morning."""

    def __init__(self, *, early_hour_threshold: int = 9) -> None:
        self.early_hour_threshold = early_hour_threshold

    @property
    def trait_name(self) -> str:
        return "early_bird_shopper"

    @property
    def actionability_dimension(self) -> ActionabilityDimension:
        return ActionabilityDimension.WHEN

    @property
    def business_relevance(self) -> str:
        return "Send marketing emails early morning for best open rates"

    def extract(self, profile: CustomerProfile) -> ActionableTrait | None:
        if profile.preferred_hour_of_day is None:
            return None

        if profile.preferred_hour_of_day >= self.early_hour_threshold:
            return None

        return ActionableTrait(
            name=self.trait_name,
            description=f"Most active around {profile.preferred_hour_of_day}:00",
            value=profile.preferred_hour_of_day,
            actionability_dimension=self.actionability_dimension,
            business_relevance=self.business_relevance,
        )


class NightOwlShopperExtractor(TraitExtractor):
    """Identify customers who shop late at night."""

    def __init__(self, *, night_hour_threshold: int = 21) -> None:
        self.night_hour_threshold = night_hour_threshold

    @property
    def trait_name(self) -> str:
        return "night_owl_shopper"

    @property
    def actionability_dimension(self) -> ActionabilityDimension:
        return ActionabilityDimension.WHEN

    @property
    def business_relevance(self) -> str:
        return "Send marketing emails in evening for best open rates"

    def extract(self, profile: CustomerProfile) -> ActionableTrait | None:
        if profile.preferred_hour_of_day is None:
            return None

        if profile.preferred_hour_of_day < self.night_hour_threshold:
            return None

        return ActionableTrait(
            name=self.trait_name,
            description=f"Most active around {profile.preferred_hour_of_day}:00",
            value=profile.preferred_hour_of_day,
            actionability_dimension=self.actionability_dimension,
            business_relevance=self.business_relevance,
        )


class SeasonalShopperExtractor(TraitExtractor):
    """Identify customers with infrequent but regular purchase patterns."""

    def __init__(
        self,
        *,
        min_purchases: int = 2,
        max_frequency: float = 0.3,
        min_tenure_days: int = 90,
    ) -> None:
        self.min_purchases = min_purchases
        self.max_frequency = max_frequency
        self.min_tenure_days = min_tenure_days

    @property
    def trait_name(self) -> str:
        return "seasonal_shopper"

    @property
    def actionability_dimension(self) -> ActionabilityDimension:
        return ActionabilityDimension.WHEN

    @property
    def business_relevance(self) -> str:
        return "Time marketing around historical purchase windows"

    def extract(self, profile: CustomerProfile) -> ActionableTrait | None:
        # Calculate tenure
        tenure_days = (profile.last_seen - profile.first_seen).days

        if tenure_days < self.min_tenure_days:
            return None

        if profile.total_purchases < self.min_purchases:
            return None

        if profile.purchase_frequency_per_month > self.max_frequency:
            return None

        return ActionableTrait(
            name=self.trait_name,
            description="Infrequent but regular purchase pattern suggests seasonal buying",
            value=True,
            actionability_dimension=self.actionability_dimension,
            business_relevance=self.business_relevance,
        )


# =============================================================================
# CHANNEL EXTRACTORS (HOW - Channel/Message Personalization)
# =============================================================================


class MobileFirstExtractor(TraitExtractor):
    """Identify customers who primarily shop on mobile."""

    def __init__(self, *, mobile_ratio_threshold: float = 0.7) -> None:
        self.mobile_ratio_threshold = mobile_ratio_threshold

    @property
    def trait_name(self) -> str:
        return "mobile_first"

    @property
    def actionability_dimension(self) -> ActionabilityDimension:
        return ActionabilityDimension.HOW

    @property
    def business_relevance(self) -> str:
        return "Prioritize mobile-optimized content and app-based campaigns"

    def extract(self, profile: CustomerProfile) -> ActionableTrait | None:
        if profile.mobile_session_ratio < self.mobile_ratio_threshold:
            return None

        return ActionableTrait(
            name=self.trait_name,
            description=f"{profile.mobile_session_ratio * 100:.0f}% of sessions on mobile",
            value=profile.mobile_session_ratio,
            actionability_dimension=self.actionability_dimension,
            business_relevance=self.business_relevance,
        )


class DesktopLoyalistExtractor(TraitExtractor):
    """Identify customers who primarily shop on desktop."""

    def __init__(
        self,
        *,
        desktop_ratio_threshold: float = 0.8,
    ) -> None:
        self.desktop_ratio_threshold = desktop_ratio_threshold

    @property
    def trait_name(self) -> str:
        return "desktop_loyalist"

    @property
    def actionability_dimension(self) -> ActionabilityDimension:
        return ActionabilityDimension.HOW

    @property
    def business_relevance(self) -> str:
        return "Target with desktop-optimized email campaigns and website experiences"

    def extract(self, profile: CustomerProfile) -> ActionableTrait | None:
        desktop_ratio = 1.0 - profile.mobile_session_ratio

        if desktop_ratio < self.desktop_ratio_threshold:
            return None

        return ActionableTrait(
            name=self.trait_name,
            description=f"{desktop_ratio * 100:.0f}% of sessions on desktop",
            value=desktop_ratio,
            actionability_dimension=self.actionability_dimension,
            business_relevance=self.business_relevance,
        )


class HighEngagementExtractor(TraitExtractor):
    """Identify highly engaged customers based on session activity."""

    def __init__(
        self,
        *,
        min_sessions: int = 10,
        min_page_views: int = 30,
    ) -> None:
        self.min_sessions = min_sessions
        self.min_page_views = min_page_views

    @property
    def trait_name(self) -> str:
        return "high_engagement"

    @property
    def actionability_dimension(self) -> ActionabilityDimension:
        return ActionabilityDimension.HOW

    @property
    def business_relevance(self) -> str:
        return "Leverage for product feedback, reviews, and advocacy programs"

    def extract(self, profile: CustomerProfile) -> ActionableTrait | None:
        is_highly_engaged = (
            profile.total_sessions >= self.min_sessions
            and profile.total_page_views >= self.min_page_views
        )

        if not is_highly_engaged:
            return None

        return ActionableTrait(
            name=self.trait_name,
            description=f"{profile.total_sessions} sessions with {profile.total_page_views} page views",
            value=True,
            actionability_dimension=self.actionability_dimension,
            business_relevance=self.business_relevance,
        )


# =============================================================================
# TRAIT EXTRACTION ENGINE
# =============================================================================


class TraitExtractionEngine:
    """Engine for extracting multiple traits from customer profiles."""

    DEFAULT_EXTRACTORS: list[type[TraitExtractor]] = [
        # WHO - Prioritization
        HighValueCustomerExtractor,
        ChurnRiskExtractor,
        HighValueDormantExtractor,
        # WHAT - Product/Offer Targeting
        CartAbandonerExtractor,
        CategorySpecialistExtractor,
        DiscountSensitiveExtractor,
        BrowserNotBuyerExtractor,
        # WHEN - Timing Optimization
        WeekendShopperExtractor,
        EarlyBirdShopperExtractor,
        NightOwlShopperExtractor,
        SeasonalShopperExtractor,
        # HOW - Channel Personalization
        MobileFirstExtractor,
        DesktopLoyalistExtractor,
        HighEngagementExtractor,
    ]

    def __init__(
        self,
        extractors: list[TraitExtractor] | None = None,
    ) -> None:
        """
        Initialize trait extraction engine.

        Args:
            extractors: List of trait extractors. If None, uses defaults.
        """
        if extractors is not None:
            self.extractors = extractors
        else:
            self.extractors = [cls() for cls in self.DEFAULT_EXTRACTORS]

    def extract_traits(
        self,
        profile: CustomerProfile,
        *,
        extraction_timestamp: datetime | None = None,
    ) -> CustomerTraits:
        """
        Extract all traits from a customer profile.

        Args:
            profile: Customer profile to analyze
            extraction_timestamp: Timestamp for extraction (defaults to now)

        Returns:
            CustomerTraits with all extracted traits
        """
        if extraction_timestamp is None:
            extraction_timestamp = datetime.now(tz=profile.first_seen.tzinfo)

        traits: list[ActionableTrait] = []

        for extractor in self.extractors:
            trait = extractor.extract(profile)
            if trait is not None:
                traits.append(trait)

        return CustomerTraits(
            internal_customer_id=profile.internal_customer_id,
            traits=traits,
            extraction_timestamp=extraction_timestamp,
        )

    def extract_traits_batch(
        self,
        profiles: list[CustomerProfile],
        *,
        extraction_timestamp: datetime | None = None,
    ) -> list[CustomerTraits]:
        """
        Extract traits from multiple profiles.

        Args:
            profiles: List of customer profiles
            extraction_timestamp: Timestamp for extraction

        Returns:
            List of CustomerTraits
        """
        return [
            self.extract_traits(profile, extraction_timestamp=extraction_timestamp)
            for profile in profiles
        ]

    def get_trait_summary(
        self,
        traits_list: list[CustomerTraits],
    ) -> dict[str, Any]:
        """
        Get summary statistics across all extracted traits.

        Args:
            traits_list: List of CustomerTraits

        Returns:
            Dictionary with trait statistics
        """
        if not traits_list:
            return {}

        trait_counts: dict[str, int] = {}
        dimension_counts: dict[str, int] = {}

        for customer_traits in traits_list:
            for trait in customer_traits.traits:
                trait_counts[trait.name] = trait_counts.get(trait.name, 0) + 1
                dim = trait.actionability_dimension.value
                dimension_counts[dim] = dimension_counts.get(dim, 0) + 1

        total_customers = len(traits_list)

        return {
            "total_customers": total_customers,
            "trait_counts": trait_counts,
            "trait_percentages": {
                name: count / total_customers
                for name, count in trait_counts.items()
            },
            "dimension_counts": dimension_counts,
            "avg_traits_per_customer": sum(
                len(ct.traits) for ct in traits_list
            ) / total_customers,
        }


def extract_traits_for_profile(
    profile: CustomerProfile,
    *,
    extraction_timestamp: datetime | None = None,
) -> CustomerTraits:
    """
    Convenience function to extract traits using default extractors.

    Args:
        profile: Customer profile to analyze
        extraction_timestamp: Timestamp for extraction

    Returns:
        CustomerTraits with extracted traits
    """
    engine = TraitExtractionEngine()
    return engine.extract_traits(profile, extraction_timestamp=extraction_timestamp)


def extract_traits_batch(
    profiles: list[CustomerProfile],
    *,
    extraction_timestamp: datetime | None = None,
) -> list[CustomerTraits]:
    """
    Convenience function to extract traits from multiple profiles.

    Args:
        profiles: List of customer profiles
        extraction_timestamp: Timestamp for extraction

    Returns:
        List of CustomerTraits
    """
    engine = TraitExtractionEngine()
    return engine.extract_traits_batch(profiles, extraction_timestamp=extraction_timestamp)
