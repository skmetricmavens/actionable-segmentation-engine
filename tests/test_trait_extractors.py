"""
Tests for actionable trait extractors.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.data.schemas import (
    ActionabilityDimension,
    CategoryAffinity,
    CustomerProfile,
)
from src.features.trait_extractors import (
    BrowserNotBuyerExtractor,
    CartAbandonerExtractor,
    CategorySpecialistExtractor,
    ChurnRiskExtractor,
    DesktopLoyalistExtractor,
    DiscountSensitiveExtractor,
    EarlyBirdShopperExtractor,
    HighEngagementExtractor,
    HighValueCustomerExtractor,
    HighValueDormantExtractor,
    MobileFirstExtractor,
    NightOwlShopperExtractor,
    SeasonalShopperExtractor,
    TraitExtractionEngine,
    WeekendShopperExtractor,
    extract_traits_batch,
    extract_traits_for_profile,
)


# =============================================================================
# FIXTURES
# =============================================================================


def make_profile(
    customer_id: str = "test_customer",
    *,
    total_revenue: Decimal = Decimal("100"),
    clv_estimate: Decimal = Decimal("300"),
    churn_risk_score: float = 0.3,
    days_since_last_purchase: int | None = 30,
    total_purchases: int = 3,
    avg_order_value: Decimal = Decimal("50"),
    purchase_frequency_per_month: float = 0.5,
    total_cart_additions: int = 5,
    cart_abandonment_rate: float = 0.4,
    total_items_viewed: int = 20,
    total_sessions: int = 5,
    total_page_views: int = 15,
    preferred_day_of_week: int | None = 1,  # Tuesday
    preferred_hour_of_day: int | None = 14,  # 2 PM
    mobile_session_ratio: float = 0.3,
    category_affinities: list[CategoryAffinity] | None = None,
    first_seen: datetime | None = None,
    last_seen: datetime | None = None,
) -> CustomerProfile:
    """Helper to create CustomerProfile for testing."""
    if first_seen is None:
        first_seen = datetime(2024, 1, 1, tzinfo=timezone.utc)
    if last_seen is None:
        last_seen = datetime(2024, 3, 31, tzinfo=timezone.utc)
    if category_affinities is None:
        category_affinities = []

    return CustomerProfile(
        internal_customer_id=customer_id,
        first_seen=first_seen,
        last_seen=last_seen,
        total_purchases=total_purchases,
        total_revenue=total_revenue,
        avg_order_value=avg_order_value,
        days_since_last_purchase=days_since_last_purchase,
        purchase_frequency_per_month=purchase_frequency_per_month,
        total_sessions=total_sessions,
        total_page_views=total_page_views,
        total_items_viewed=total_items_viewed,
        total_cart_additions=total_cart_additions,
        cart_abandonment_rate=cart_abandonment_rate,
        category_affinities=category_affinities,
        preferred_day_of_week=preferred_day_of_week,
        preferred_hour_of_day=preferred_hour_of_day,
        mobile_session_ratio=mobile_session_ratio,
        clv_estimate=clv_estimate,
        churn_risk_score=churn_risk_score,
    )


# =============================================================================
# WHO - VALUE-BASED EXTRACTOR TESTS
# =============================================================================


class TestHighValueCustomerExtractor:
    """Tests for HighValueCustomerExtractor."""

    def test_high_clv_customer(self) -> None:
        """Test customer with high CLV gets trait."""
        profile = make_profile(clv_estimate=Decimal("600"))
        extractor = HighValueCustomerExtractor()

        trait = extractor.extract(profile)

        assert trait is not None
        assert trait.name == "high_value_customer"
        assert trait.actionability_dimension == ActionabilityDimension.WHO

    def test_high_revenue_customer(self) -> None:
        """Test customer with high revenue gets trait."""
        profile = make_profile(total_revenue=Decimal("300"), clv_estimate=Decimal("100"))
        extractor = HighValueCustomerExtractor()

        trait = extractor.extract(profile)

        assert trait is not None

    def test_low_value_customer(self) -> None:
        """Test low value customer does not get trait."""
        profile = make_profile(clv_estimate=Decimal("100"), total_revenue=Decimal("50"))
        extractor = HighValueCustomerExtractor()

        trait = extractor.extract(profile)

        assert trait is None

    def test_custom_thresholds(self) -> None:
        """Test with custom thresholds."""
        profile = make_profile(clv_estimate=Decimal("200"), total_revenue=Decimal("100"))
        extractor = HighValueCustomerExtractor(
            clv_threshold=Decimal("150"),
            revenue_threshold=Decimal("80"),
        )

        trait = extractor.extract(profile)

        assert trait is not None


class TestChurnRiskExtractor:
    """Tests for ChurnRiskExtractor."""

    def test_high_churn_risk(self) -> None:
        """Test customer with high churn risk gets trait."""
        profile = make_profile(churn_risk_score=0.85)
        extractor = ChurnRiskExtractor()

        trait = extractor.extract(profile)

        assert trait is not None
        assert trait.name == "churn_risk"
        assert trait.value == "high"

    def test_moderate_churn_risk(self) -> None:
        """Test customer with moderate churn risk."""
        profile = make_profile(churn_risk_score=0.65)
        extractor = ChurnRiskExtractor()

        trait = extractor.extract(profile)

        assert trait is not None
        assert trait.value == "moderate"

    def test_low_churn_risk(self) -> None:
        """Test customer with low churn risk does not get trait."""
        profile = make_profile(churn_risk_score=0.3)
        extractor = ChurnRiskExtractor()

        trait = extractor.extract(profile)

        assert trait is None


class TestHighValueDormantExtractor:
    """Tests for HighValueDormantExtractor."""

    def test_high_value_dormant_customer(self) -> None:
        """Test high-value dormant customer gets trait."""
        profile = make_profile(
            total_revenue=Decimal("300"),
            days_since_last_purchase=90,
        )
        extractor = HighValueDormantExtractor()

        trait = extractor.extract(profile)

        assert trait is not None
        assert trait.name == "high_value_dormant"

    def test_active_high_value_customer(self) -> None:
        """Test recently active high-value customer does not get trait."""
        profile = make_profile(
            total_revenue=Decimal("300"),
            days_since_last_purchase=30,
        )
        extractor = HighValueDormantExtractor()

        trait = extractor.extract(profile)

        assert trait is None

    def test_dormant_low_value_customer(self) -> None:
        """Test dormant low-value customer does not get trait."""
        profile = make_profile(
            total_revenue=Decimal("50"),
            days_since_last_purchase=90,
        )
        extractor = HighValueDormantExtractor()

        trait = extractor.extract(profile)

        assert trait is None


# =============================================================================
# WHAT - BEHAVIOR-BASED EXTRACTOR TESTS
# =============================================================================


class TestCartAbandonerExtractor:
    """Tests for CartAbandonerExtractor."""

    def test_high_abandonment_rate(self) -> None:
        """Test customer with high cart abandonment gets trait."""
        profile = make_profile(
            total_cart_additions=10,
            cart_abandonment_rate=0.7,
        )
        extractor = CartAbandonerExtractor()

        trait = extractor.extract(profile)

        assert trait is not None
        assert trait.name == "cart_abandoner"
        assert trait.actionability_dimension == ActionabilityDimension.WHAT

    def test_low_abandonment_rate(self) -> None:
        """Test customer with low abandonment rate does not get trait."""
        profile = make_profile(
            total_cart_additions=10,
            cart_abandonment_rate=0.2,
        )
        extractor = CartAbandonerExtractor()

        trait = extractor.extract(profile)

        assert trait is None

    def test_insufficient_cart_data(self) -> None:
        """Test customer with insufficient cart data does not get trait."""
        profile = make_profile(
            total_cart_additions=1,
            cart_abandonment_rate=0.8,
        )
        extractor = CartAbandonerExtractor()

        trait = extractor.extract(profile)

        assert trait is None


class TestCategorySpecialistExtractor:
    """Tests for CategorySpecialistExtractor."""

    def test_strong_category_preference(self) -> None:
        """Test customer with strong category preference gets trait."""
        affinities = [
            CategoryAffinity(
                category="Electronics",
                engagement_score=0.9,
                view_count=20,
                purchase_count=5,
            ),
        ]
        profile = make_profile(category_affinities=affinities)
        extractor = CategorySpecialistExtractor()

        trait = extractor.extract(profile)

        assert trait is not None
        assert trait.name == "category_specialist"
        assert trait.value == "Electronics"

    def test_weak_category_preference(self) -> None:
        """Test customer with weak category preference does not get trait."""
        affinities = [
            CategoryAffinity(
                category="Electronics",
                engagement_score=0.5,
                view_count=10,
                purchase_count=5,
            ),
        ]
        profile = make_profile(category_affinities=affinities)
        extractor = CategorySpecialistExtractor()

        trait = extractor.extract(profile)

        assert trait is None

    def test_no_category_affinities(self) -> None:
        """Test customer with no category affinities."""
        profile = make_profile(category_affinities=[])
        extractor = CategorySpecialistExtractor()

        trait = extractor.extract(profile)

        assert trait is None


class TestDiscountSensitiveExtractor:
    """Tests for DiscountSensitiveExtractor."""

    def test_discount_sensitive_customer(self) -> None:
        """Test customer with discount-sensitive pattern gets trait."""
        profile = make_profile(
            avg_order_value=Decimal("30"),
            purchase_frequency_per_month=1.0,
            total_purchases=5,
        )
        extractor = DiscountSensitiveExtractor()

        trait = extractor.extract(profile)

        assert trait is not None
        assert trait.name == "discount_sensitive"

    def test_high_aov_customer(self) -> None:
        """Test customer with high AOV does not get trait."""
        profile = make_profile(
            avg_order_value=Decimal("100"),
            purchase_frequency_per_month=1.0,
            total_purchases=5,
        )
        extractor = DiscountSensitiveExtractor()

        trait = extractor.extract(profile)

        assert trait is None


class TestBrowserNotBuyerExtractor:
    """Tests for BrowserNotBuyerExtractor."""

    def test_heavy_browser_low_buyer(self) -> None:
        """Test customer who browses but doesn't buy gets trait."""
        profile = make_profile(
            total_items_viewed=25,
            total_purchases=1,
        )
        extractor = BrowserNotBuyerExtractor()

        trait = extractor.extract(profile)

        assert trait is not None
        assert trait.name == "browser_not_buyer"

    def test_active_buyer(self) -> None:
        """Test customer who buys does not get trait."""
        profile = make_profile(
            total_items_viewed=25,
            total_purchases=10,
        )
        extractor = BrowserNotBuyerExtractor()

        trait = extractor.extract(profile)

        assert trait is None


# =============================================================================
# WHEN - TEMPORAL EXTRACTOR TESTS
# =============================================================================


class TestWeekendShopperExtractor:
    """Tests for WeekendShopperExtractor."""

    def test_saturday_shopper(self) -> None:
        """Test customer who shops on Saturday gets trait."""
        profile = make_profile(preferred_day_of_week=5)  # Saturday
        extractor = WeekendShopperExtractor()

        trait = extractor.extract(profile)

        assert trait is not None
        assert trait.name == "weekend_shopper"
        assert trait.value == "Saturday"
        assert trait.actionability_dimension == ActionabilityDimension.WHEN

    def test_sunday_shopper(self) -> None:
        """Test customer who shops on Sunday gets trait."""
        profile = make_profile(preferred_day_of_week=6)  # Sunday
        extractor = WeekendShopperExtractor()

        trait = extractor.extract(profile)

        assert trait is not None
        assert trait.value == "Sunday"

    def test_weekday_shopper(self) -> None:
        """Test customer who shops on weekday does not get trait."""
        profile = make_profile(preferred_day_of_week=2)  # Wednesday
        extractor = WeekendShopperExtractor()

        trait = extractor.extract(profile)

        assert trait is None


class TestEarlyBirdShopperExtractor:
    """Tests for EarlyBirdShopperExtractor."""

    def test_early_morning_shopper(self) -> None:
        """Test customer who shops early morning gets trait."""
        profile = make_profile(preferred_hour_of_day=6)
        extractor = EarlyBirdShopperExtractor()

        trait = extractor.extract(profile)

        assert trait is not None
        assert trait.name == "early_bird_shopper"

    def test_afternoon_shopper(self) -> None:
        """Test customer who shops in afternoon does not get trait."""
        profile = make_profile(preferred_hour_of_day=14)
        extractor = EarlyBirdShopperExtractor()

        trait = extractor.extract(profile)

        assert trait is None


class TestNightOwlShopperExtractor:
    """Tests for NightOwlShopperExtractor."""

    def test_night_shopper(self) -> None:
        """Test customer who shops at night gets trait."""
        profile = make_profile(preferred_hour_of_day=22)
        extractor = NightOwlShopperExtractor()

        trait = extractor.extract(profile)

        assert trait is not None
        assert trait.name == "night_owl_shopper"

    def test_daytime_shopper(self) -> None:
        """Test customer who shops during day does not get trait."""
        profile = make_profile(preferred_hour_of_day=14)
        extractor = NightOwlShopperExtractor()

        trait = extractor.extract(profile)

        assert trait is None


class TestSeasonalShopperExtractor:
    """Tests for SeasonalShopperExtractor."""

    def test_seasonal_shopper(self) -> None:
        """Test customer with seasonal pattern gets trait."""
        profile = make_profile(
            total_purchases=3,
            purchase_frequency_per_month=0.2,
            first_seen=datetime(2024, 1, 1, tzinfo=timezone.utc),
            last_seen=datetime(2024, 6, 1, tzinfo=timezone.utc),  # 5 months tenure
        )
        extractor = SeasonalShopperExtractor()

        trait = extractor.extract(profile)

        assert trait is not None
        assert trait.name == "seasonal_shopper"

    def test_frequent_shopper(self) -> None:
        """Test frequent shopper does not get trait."""
        profile = make_profile(
            total_purchases=10,
            purchase_frequency_per_month=2.0,
            first_seen=datetime(2024, 1, 1, tzinfo=timezone.utc),
            last_seen=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        extractor = SeasonalShopperExtractor()

        trait = extractor.extract(profile)

        assert trait is None


# =============================================================================
# HOW - CHANNEL EXTRACTOR TESTS
# =============================================================================


class TestMobileFirstExtractor:
    """Tests for MobileFirstExtractor."""

    def test_mobile_primary_customer(self) -> None:
        """Test customer who primarily uses mobile gets trait."""
        profile = make_profile(mobile_session_ratio=0.8)
        extractor = MobileFirstExtractor()

        trait = extractor.extract(profile)

        assert trait is not None
        assert trait.name == "mobile_first"
        assert trait.actionability_dimension == ActionabilityDimension.HOW

    def test_desktop_primary_customer(self) -> None:
        """Test customer who primarily uses desktop does not get trait."""
        profile = make_profile(mobile_session_ratio=0.3)
        extractor = MobileFirstExtractor()

        trait = extractor.extract(profile)

        assert trait is None


class TestDesktopLoyalistExtractor:
    """Tests for DesktopLoyalistExtractor."""

    def test_desktop_primary_customer(self) -> None:
        """Test customer who primarily uses desktop gets trait."""
        profile = make_profile(mobile_session_ratio=0.1)  # 90% desktop
        extractor = DesktopLoyalistExtractor()

        trait = extractor.extract(profile)

        assert trait is not None
        assert trait.name == "desktop_loyalist"

    def test_mobile_primary_customer(self) -> None:
        """Test customer who primarily uses mobile does not get trait."""
        profile = make_profile(mobile_session_ratio=0.6)  # 40% desktop
        extractor = DesktopLoyalistExtractor()

        trait = extractor.extract(profile)

        assert trait is None


class TestHighEngagementExtractor:
    """Tests for HighEngagementExtractor."""

    def test_highly_engaged_customer(self) -> None:
        """Test highly engaged customer gets trait."""
        profile = make_profile(
            total_sessions=15,
            total_page_views=50,
        )
        extractor = HighEngagementExtractor()

        trait = extractor.extract(profile)

        assert trait is not None
        assert trait.name == "high_engagement"

    def test_low_engagement_customer(self) -> None:
        """Test low engagement customer does not get trait."""
        profile = make_profile(
            total_sessions=3,
            total_page_views=10,
        )
        extractor = HighEngagementExtractor()

        trait = extractor.extract(profile)

        assert trait is None


# =============================================================================
# EXTRACTION ENGINE TESTS
# =============================================================================


class TestTraitExtractionEngine:
    """Tests for TraitExtractionEngine."""

    def test_default_extractors(self) -> None:
        """Test engine initializes with default extractors."""
        engine = TraitExtractionEngine()
        assert len(engine.extractors) == 14  # All default extractors

    def test_custom_extractors(self) -> None:
        """Test engine with custom extractors."""
        extractors = [HighValueCustomerExtractor(), ChurnRiskExtractor()]
        engine = TraitExtractionEngine(extractors=extractors)

        assert len(engine.extractors) == 2

    def test_extract_traits(self) -> None:
        """Test extracting traits from profile."""
        profile = make_profile(
            clv_estimate=Decimal("600"),  # Triggers high_value_customer
            churn_risk_score=0.8,  # Triggers churn_risk
        )
        engine = TraitExtractionEngine()

        traits = engine.extract_traits(profile)

        assert traits.internal_customer_id == profile.internal_customer_id
        assert len(traits.traits) >= 2  # At least high_value and churn_risk

    def test_extract_traits_batch(self) -> None:
        """Test extracting traits from multiple profiles."""
        profiles = [
            make_profile(customer_id="cust_1", clv_estimate=Decimal("600")),
            make_profile(customer_id="cust_2", churn_risk_score=0.8),
        ]
        engine = TraitExtractionEngine()

        traits_list = engine.extract_traits_batch(profiles)

        assert len(traits_list) == 2
        assert traits_list[0].internal_customer_id == "cust_1"
        assert traits_list[1].internal_customer_id == "cust_2"

    def test_get_trait_summary(self) -> None:
        """Test trait summary statistics."""
        profiles = [
            make_profile(customer_id="cust_1", clv_estimate=Decimal("600")),
            make_profile(customer_id="cust_2", clv_estimate=Decimal("600")),
            make_profile(customer_id="cust_3", churn_risk_score=0.8),
        ]
        engine = TraitExtractionEngine()
        traits_list = engine.extract_traits_batch(profiles)

        summary = engine.get_trait_summary(traits_list)

        assert summary["total_customers"] == 3
        assert "trait_counts" in summary
        assert "trait_percentages" in summary
        assert "avg_traits_per_customer" in summary

    def test_get_trait_summary_empty(self) -> None:
        """Test trait summary with empty list."""
        engine = TraitExtractionEngine()

        summary = engine.get_trait_summary([])

        assert summary == {}


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_extract_traits_for_profile(self) -> None:
        """Test convenience function for single profile."""
        profile = make_profile(clv_estimate=Decimal("600"))

        traits = extract_traits_for_profile(profile)

        assert traits.internal_customer_id == profile.internal_customer_id
        assert len(traits.traits) >= 1

    def test_extract_traits_batch(self) -> None:
        """Test convenience function for batch extraction."""
        profiles = [
            make_profile(customer_id="cust_1"),
            make_profile(customer_id="cust_2"),
        ]

        traits_list = extract_traits_batch(profiles)

        assert len(traits_list) == 2


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegrationWithSyntheticData:
    """Integration tests with synthetic data."""

    def test_extract_traits_from_synthetic_profiles(self) -> None:
        """Test extracting traits from synthetic data profiles."""
        from src.data.joiner import resolve_customer_merges
        from src.data.synthetic_generator import SyntheticDataGenerator
        from src.features.profile_builder import build_profiles_batch

        generator = SyntheticDataGenerator(seed=42)
        date_range = (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 31, tzinfo=timezone.utc),
        )

        dataset = generator.generate_dataset(
            n_customers=50,
            date_range=date_range,
        )

        merge_map = resolve_customer_merges(dataset.id_history)
        ref_date = datetime(2024, 4, 1, tzinfo=timezone.utc)
        profiles = build_profiles_batch(
            dataset.events,
            merge_map=merge_map,
            reference_date=ref_date,
        )

        # Extract traits
        traits_list = extract_traits_batch(profiles)

        assert len(traits_list) == len(profiles)

        # Should have some traits extracted across customers
        total_traits = sum(len(ct.traits) for ct in traits_list)
        assert total_traits > 0

        # Get summary
        engine = TraitExtractionEngine()
        summary = engine.get_trait_summary(traits_list)

        assert summary["total_customers"] == len(profiles)
        assert summary["avg_traits_per_customer"] >= 0


class TestTraitActionabilityMapping:
    """Tests to verify traits map to correct actionability dimensions."""

    def test_who_traits(self) -> None:
        """Test WHO dimension traits."""
        who_extractors = [
            HighValueCustomerExtractor(),
            ChurnRiskExtractor(),
            HighValueDormantExtractor(),
        ]

        for extractor in who_extractors:
            assert extractor.actionability_dimension == ActionabilityDimension.WHO

    def test_what_traits(self) -> None:
        """Test WHAT dimension traits."""
        what_extractors = [
            CartAbandonerExtractor(),
            CategorySpecialistExtractor(),
            DiscountSensitiveExtractor(),
            BrowserNotBuyerExtractor(),
        ]

        for extractor in what_extractors:
            assert extractor.actionability_dimension == ActionabilityDimension.WHAT

    def test_when_traits(self) -> None:
        """Test WHEN dimension traits."""
        when_extractors = [
            WeekendShopperExtractor(),
            EarlyBirdShopperExtractor(),
            NightOwlShopperExtractor(),
            SeasonalShopperExtractor(),
        ]

        for extractor in when_extractors:
            assert extractor.actionability_dimension == ActionabilityDimension.WHEN

    def test_how_traits(self) -> None:
        """Test HOW dimension traits."""
        how_extractors = [
            MobileFirstExtractor(),
            DesktopLoyalistExtractor(),
            HighEngagementExtractor(),
        ]

        for extractor in how_extractors:
            assert extractor.actionability_dimension == ActionabilityDimension.HOW
